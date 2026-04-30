import os
import sys
import yaml
import logging
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC

from src.utils.seed import set_seed
from src.data.featurizer import MolFeaturizer
from src.models.hybrid_model import GINMambaHybrid
from src.ordering.atomic_number import get_order as atomic_number_order


class SmilesFeaturizer(torch.utils.data.Dataset):
    def __init__(self, zinc_dataset, featurizer):
        self.zinc_dataset = zinc_dataset
        self.featurizer = featurizer

    def __len__(self):
        return len(self.zinc_dataset)

    def __getitem__(self, idx):
        data = self.zinc_dataset[idx]
        smiles = data.smiles
        if isinstance(smiles, (list, tuple)):
            smiles = smiles[0]
        mol_data = self.featurizer(smiles)
        return mol_data


class PretrainingModel(nn.Module):
    def __init__(self, node_features, d_model, gin_hidden=64, gin_layers=3,
                 mamba_state=16, mamba_conv=4, mamba_expand=2, mamba_layers=1,
                 bidirectional=True, dropout=0.0):
        super().__init__()
        self.hybrid = GINMambaHybrid(
            node_features=node_features,
            d_model=d_model,
            gin_hidden=gin_hidden,
            gin_layers=gin_layers,
            mamba_state=mamba_state,
            mamba_conv=mamba_conv,
            mamba_expand=mamba_expand,
            mamba_layers=mamba_layers,
            bidirectional=bidirectional,
            num_tasks=1,
            dropout=dropout,
        )
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, node_features),
        )
        self.esf_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3),
        )
        self.mask_token = nn.Parameter(torch.randn(1, node_features) * 0.02)

    def forward(self, data, mask_ratio=0.15):
        x_orig = data.x.clone()

        mask = torch.rand(x_orig.size(0), device=x_orig.device) < mask_ratio
        x_masked = x_orig.clone()
        x_masked[mask] = self.mask_token.repeat(mask.sum(), 1)

        data.x = x_masked

        h_fused = self.hybrid.encode_atoms(data, atomic_number_order)

        recon = self.recon_head(h_fused)
        mam_loss = F.mse_loss(recon[mask], x_orig[mask])

        electro_orig = x_orig[:, -3:]
        esf_pred = self.esf_head(h_fused)
        esf_loss = F.mse_loss(esf_pred, electro_orig)

        return mam_loss, esf_loss, mask.sum()


def setup_logger(log_dir="outputs/logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"pretrain_{timestamp}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def pretrain_epoch(model, dataloader, optimizer, device, esf_weight=0.1, mask_ratio=0.15):
    model.train()
    total_mam = 0.0
    total_esf = 0.0
    total_masked = 0.0
    num_batches = len(dataloader)
    use_amp = device.type == "cuda"

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            mam_loss, esf_loss, n_masked = model(batch, mask_ratio=mask_ratio)
            loss = mam_loss + esf_weight * esf_loss

        loss.backward()
        optimizer.step()

        total_mam += mam_loss.item()
        total_esf += esf_loss.item()
        total_masked += n_masked.item()

    avg_mam = total_mam / max(num_batches, 1)
    avg_esf = total_esf / max(num_batches, 1)
    avg_masked = total_masked / max(num_batches, 1)
    return avg_mam, avg_esf, avg_masked


def main():
    parser = argparse.ArgumentParser(description="Pretrain GIN-Mamba Hybrid on ZINC")
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to config file")
    parser.add_argument("--zinc_root", type=str, default="/tmp/ZINC", help="ZINC dataset root path")
    parser.add_argument("--subset", action="store_true", default=True, help="Use ZINC-12K subset")
    parser.add_argument("--max_molecules", type=int, default=None, help="Limit molecules (None = use all)")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config batch_size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    logger = setup_logger()
    logger.info(f"Arguments: {args}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    rwse_walk_length = config.get("model", {}).get("rwse_walk_length", 16)
    featurizer = MolFeaturizer(rwse_walk_length=rwse_walk_length)

    logger.info("Loading ZINC dataset...")
    zinc = ZINC(root=args.zinc_root, subset=args.subset, split="train")
    logger.info(f"ZINC dataset loaded: {len(zinc)} molecules")

    dataset = SmilesFeaturizer(zinc, featurizer)
    sample_data = dataset[0]
    node_features = sample_data.x.size(1)
    logger.info(f"Node feature dimension: {node_features}")

    if args.max_molecules is not None:
        dataset.zinc_dataset = torch.utils.data.Subset(
            zinc, range(min(args.max_molecules, len(zinc)))
        )
        logger.info(f"Limited to {len(dataset.zinc_dataset)} molecules")

    batch_size = args.batch_size or config["data"]["batch_size"]
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0
    )

    m_cfg = config["model"]
    model = PretrainingModel(
        node_features=node_features,
        d_model=m_cfg["d_model"],
        gin_hidden=m_cfg.get("gin_hidden", 64),
        gin_layers=m_cfg["gin_layers"],
        mamba_state=m_cfg["mamba_state"],
        mamba_conv=m_cfg["mamba_conv"],
        mamba_expand=m_cfg.get("mamba_expand", 2),
        mamba_layers=m_cfg["mamba_layers"],
        bidirectional=m_cfg.get("bidirectional", True),
        dropout=m_cfg.get("dropout", 0.0),
    ).to(device)

    lr = args.lr or float(config["training"]["lr"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
        fused=device.type == "cuda",
    )

    epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]
    esf_weight = config["training"].get("esf_weight", 0.1)
    mask_ratio = config["training"].get("mask_ratio", 0.15)

    logger.info(f"Starting pretraining for {epochs} epochs...")
    best_loss = float("inf")
    checkpoint_path = "outputs/checkpoints/pretrained_best.pt"

    for epoch in range(1, epochs + 1):
        mam_loss, esf_loss, avg_masked = pretrain_epoch(
            model, dataloader, optimizer, device,
            esf_weight=esf_weight, mask_ratio=mask_ratio,
        )
        total_loss = mam_loss + esf_weight * esf_loss
        logger.info(
            f"Epoch {epoch:03d} | MAM: {mam_loss:.6f} | ESF: {esf_loss:.6f} "
            f"| Total: {total_loss:.6f} | AvgMasked: {avg_masked:.1f}"
        )

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"-> Saved best pretrained model (loss={best_loss:.6f})")

    final_path = f"outputs/checkpoints/pretrained_final_epoch{epochs}.pt"
    torch.save(model.state_dict(), final_path)
    logger.info(f"Saved final model to {final_path}")
    logger.info("Pretraining completed.")


if __name__ == "__main__":
    main()
