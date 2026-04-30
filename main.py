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
from src.data.splits import scaffold_split
from src.models.hybrid_model import GINMambaHybrid
from src.training.train import train_epoch
from src.training.eval import evaluate

from src.ordering.random import get_order as random_order
from src.ordering.atomic_number import get_order as atomic_number_order
from src.ordering.electronegativity import get_order as electronegativity_order
from src.ordering.degree import get_order as degree_order
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

    # File handler
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def main():
    parser = argparse.ArgumentParser(description="Run Tox21 Mamba Modeling")
    parser.add_argument(
        "--ordering",
        type=str,
        default="atomic_number",
        choices=["random", "atomic_number", "electronegativity", "degree", "learned"],
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

    args = parser.parse_args()

    # Setup directories
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    logger = setup_logger(args.ordering)
    logger.info(f"Arguments: {args}")

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Config: {config}")

    # Set seed
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info("Loading dataset...")
    dataset = Tox21Dataset(root=config["data"]["root"])

    # Create splits
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
    elif args.ordering == "learned":
        ordering_func = LearnedOrdering(dataset.num_node_features).to(device)
    else:
        raise ValueError(f"Unknown ordering: {args.ordering}")

    # Set mamba layers to 0 if gin model
    mamba_layers = config["model"]["mamba_layers"]
    if args.model_type == "gin":
        mamba_layers = 0
        logger.info("Using standalone GIN baseline (mamba_layers=0)")

    # Initialize model
    base_model = GINMambaHybrid(
        node_features=dataset.num_node_features,
        d_model=config["model"]["d_model"],
        gin_layers=config["model"]["gin_layers"],
        mamba_state=config["model"]["mamba_state"],
        mamba_conv=config["model"]["mamba_conv"],
        mamba_layers=mamba_layers,
        dropout=config["model"]["dropout"],
        num_tasks=dataset.num_tasks,
    )

    model = ModelWrapper(base_model, ordering_func).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
        fused=torch.cuda.is_available()
    )

    logger.info("Computing pos_weight for BCE loss...")
    y_all = torch.cat([data.y for data in train_subset], dim=0) # shape (N, 12)
    valid_mask = ~torch.isnan(y_all)
    pos_counts = ((y_all == 1) & valid_mask).sum(dim=0)
    neg_counts = ((y_all == 0) & valid_mask).sum(dim=0)
    pos_weight = neg_counts.float() / pos_counts.clamp(min=1).float()
    pos_weight = pos_weight.to(device)
    logger.info(f"pos_weight: {pos_weight.tolist()}")

    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    best_val_roc_auc = -1.0
    best_model_path = (
        f"outputs/checkpoints/best_model_{args.model_type}_{args.ordering}.pt"
    )
    
    # Use command line epochs if provided, otherwise use config
    epochs = args.epochs if args.epochs is not None else config["training"]["epochs"]

    logger.info("Starting training...")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        val_roc_auc = val_metrics.get("roc_auc", 0.0)
        val_f1_score = val_metrics.get("f1_score", 0.0)

        logger.info(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val ROC-AUC: {val_roc_auc:.4f} | Val F1-Score: {val_f1_score:.4f}"
        )

        if val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = val_roc_auc
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"-> Saved new best model with Val ROC-AUC: {best_val_roc_auc:.4f}"
            )

    # Load best model for testing
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

    # Write results to JSON
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
