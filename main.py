import os
import sys
import yaml
import json
import logging
import argparse
from datetime import datetime
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.utils.seed import set_seed
from src.data.tox21_dataset import Tox21Dataset
from src.data.featurizer import MolFeaturizer
from src.data.splits import scaffold_split
from src.models.hybrid_model import GINMambaHybrid
from src.training.train import train_epoch
from src.training.eval import evaluate

from src.ordering.random import get_order as random_order
from src.ordering.atomic_number import get_order as atomic_number_order
from src.ordering.electronegativity import get_order as electronegativity_order
from src.ordering.degree import get_order as degree_order
from src.ordering.canonical import get_order as canonical_order
from src.ordering.learned import LearnedOrdering


class ModelWrapper(nn.Module):
    def __init__(self, model, ordering_func):
        super().__init__()
        self.model = model
        self.ordering_func = ordering_func

    def forward(self, data):
        return self.model(data, self.ordering_func)


def setup_logger(ordering):
    os.makedirs("outputs/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"outputs/logs/run_{ordering}_{timestamp}.log"

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


def load_pretrained_weights(model, pretrained_path, logger):
    logger.info(f"Loading pretrained weights from {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)

    pretrained_prefix = "hybrid."
    matched_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key.startswith(pretrained_prefix):
            new_key = key[len(pretrained_prefix):]
            matched_state[new_key] = value
        elif key in ("recon_head", "esf_head", "mask_token") or key.startswith("recon_head.") or key.startswith("esf_head.") or key.startswith("mask_token"):
            skipped.append(key)
        elif key in model.state_dict():
            matched_state[key] = value
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(matched_state, strict=False)
    if missing:
        logger.info(f"Missing keys (will be randomly initialized): {missing}")
    if unexpected:
        logger.info(f"Unexpected keys (ignored): {unexpected}")

    logger.info(f"Loaded {len(matched_state)} parameters; skipped {len(skipped)} pretraining-only keys")
    return model


def main():
    parser = argparse.ArgumentParser(description="Run Tox21 Mamba Modeling")
    parser.add_argument(
        "--ordering",
        type=str,
        default="atomic_number",
        choices=["random", "atomic_number", "electronegativity", "degree", "canonical", "learned"],
        help="Node ordering strategy",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="hybrid",
        choices=["hybrid", "gin"],
        help="Type of model to use (hybrid or standalone gin ablation)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (overrides config)")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint",
    )

    args = parser.parse_args()

    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    logger = setup_logger(args.ordering)
    logger.info(f"Arguments: {args}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Config: {config}")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    rwse_walk_length = config.get("model", {}).get("rwse_walk_length", 16)

    logger.info("Loading dataset...")
    dataset = Tox21Dataset(root=config["data"]["root"])

    train_subset, val_subset, test_subset = scaffold_split(dataset)

    batch_size = config["data"]["batch_size"]
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, pin_memory=True)

    logger.info(
        f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}, Test samples: {len(test_subset)}"
    )

    # Get node ordering function
    if args.ordering == "random":
        ordering_func = random_order
    elif args.ordering == "atomic_number":
        ordering_func = atomic_number_order
    elif args.ordering == "electronegativity":
        ordering_func = electronegativity_order
    elif args.ordering == "degree":
        ordering_func = degree_order
    elif args.ordering == "canonical":
        ordering_func = canonical_order
    elif args.ordering == "learned":
        ordering_func = LearnedOrdering(dataset.num_node_features).to(device)
    else:
        raise ValueError(f"Unknown ordering: {args.ordering}")

    # Set mamba layers to 0 if gin model
    mamba_layers = config["model"]["mamba_layers"]
    if args.model_type == "gin":
        mamba_layers = 0
        logger.info("Using standalone GIN baseline (mamba_layers=0)")

    base_model = GINMambaHybrid(
        node_features=dataset.num_node_features,
        d_model=config["model"]["d_model"],
        gin_layers=config["model"]["gin_layers"],
        mamba_state=config["model"]["mamba_state"],
        mamba_conv=config["model"]["mamba_conv"],
        mamba_layers=mamba_layers,
        bidirectional=config["model"].get("bidirectional", False),
        dropout=config["model"]["dropout"],
        num_tasks=dataset.num_tasks,
    )

    pretrained_path = args.pretrained or config.get("model", {}).get("pretrained_path")
    if pretrained_path and os.path.exists(pretrained_path):
        base_model = load_pretrained_weights(base_model, pretrained_path, logger)

    model = ModelWrapper(base_model, ordering_func).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
        fused=torch.cuda.is_available(),
    )

    logger.info("Computing pos_weight for BCE loss...")
    y_all = torch.cat([data.y for data in train_subset], dim=0)
    valid_mask = ~torch.isnan(y_all)
    pos_counts = ((y_all == 1) & valid_mask).sum(dim=0)
    neg_counts = ((y_all == 0) & valid_mask).sum(dim=0)
    pos_weight = neg_counts.float() / pos_counts.clamp(min=1).float()
    pos_weight = pos_weight.to(device)
    logger.info(f"pos_weight: {pos_weight.tolist()}")

    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    best_val_roc_auc = -1.0
    best_val_f1_score = -1.0
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 15
    best_model_path = (
        f"outputs/checkpoints/best_model_{args.model_type}_{args.ordering}.pt"
    )

    epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]

    logger.info("Starting training...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        val_roc_auc = val_metrics.get("roc_auc", 0.0)
        val_f1_score = val_metrics.get("f1_score", 0.0)
        val_f1_optimal = val_metrics.get("f1_score_optimal", val_f1_score)

        logger.info(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val ROC-AUC: {val_roc_auc:.4f} | Val F1-Score: {val_f1_score:.4f} | Val F1-Optimal: {val_f1_optimal:.4f}"
        )
        
        # Early stopping and model checkpointing based on multiple metrics
        improvement = False
        
        # Check ROC-AUC improvement
        if val_roc_auc > best_val_roc_auc + 0.001:  # 0.1% improvement threshold
            best_val_roc_auc = val_roc_auc
            improvement = True
        
        # Check F1-score improvement
        if val_f1_optimal > best_val_f1_score + 0.005:  # 0.5% improvement threshold
            best_val_f1_score = val_f1_optimal
            improvement = True
        
        # Check loss improvement
        if val_loss < best_val_loss - 0.001:  # 0.1% improvement threshold
            best_val_loss = val_loss
            improvement = True

        if improvement:
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"-> Saved new best model with Val ROC-AUC: {best_val_roc_auc:.4f}, F1-Optimal: {best_val_f1_score:.4f}"
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch} after {patience} epochs without improvement")
                break

    logger.info("Evaluating best model on test set...")
    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)
    )
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    test_roc_auc = test_metrics.get("roc_auc", 0.0)
    test_prc_auc = test_metrics.get("prc_auc", 0.0)
    test_f1_score = test_metrics.get("f1_score", 0.0)

    logger.info(
        f"Test Loss: {test_loss:.4f} | Test ROC-AUC: {test_roc_auc:.4f} | Test PRC-AUC: {test_prc_auc:.4f} | Test F1-Score: {test_f1_score:.4f}"
    )

    results = {
        "model_type": args.model_type,
        "ordering": args.ordering,
        "seed": args.seed,
        "test_roc_auc": test_roc_auc,
        "test_prc_auc": test_prc_auc,
        "test_f1_score": test_f1_score,
        "best_val_roc_auc": best_val_roc_auc,
    }

    results_path = f"outputs/results/results_{args.model_type}_{args.ordering}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
