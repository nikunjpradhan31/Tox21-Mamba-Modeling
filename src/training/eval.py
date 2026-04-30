import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any

from src.training.metrics import compute_metrics
from src.training.threshold_optimization import find_optimal_threshold_per_task, apply_thresholds_to_predictions


@torch.no_grad()
def evaluate(
    model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluates the model on the given dataloader.

    Args:
        model: PyTorch model
        dataloader: DataLoader for validation/testing data
        criterion: Loss function (should be BCEWithLogitsLoss with reduction='none')
        device: Device to evaluate on

    Returns:
        Tuple of (average loss, dictionary of metrics)
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    all_logits = []
    all_labels = []

    for batch in dataloader:
        if hasattr(batch, "y") and hasattr(batch, "x"):
            labels = batch.y.to(device)
            batch = batch.to(device)
            outputs = model(batch)
        elif isinstance(batch, dict):
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
        elif isinstance(batch, (tuple, list)):
            inputs, labels = batch[:-1], batch[-1]
            labels = labels.to(device)
            if len(inputs) == 1:
                outputs = model(inputs[0].to(device))
            else:
                outputs = model(*[inp.to(device) for inp in inputs])
        else:
            raise ValueError("Unexpected batch format")

        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        if logits.shape != labels.shape:
            logits = logits.view(labels.shape)

        valid_mask = ~torch.isnan(labels)
        safe_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))

        loss_matrix = criterion(logits, safe_labels)
        masked_loss = loss_matrix[valid_mask]

        if masked_loss.numel() > 0:
            loss = masked_loss.mean()
            total_loss += loss.item()

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    if len(all_logits) == 0:
        return avg_loss, {"roc_auc": 0.0, "prc_auc": 0.0}

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Convert logits to probabilities
    probs = torch.sigmoid(all_logits)

    # Compute metrics (ignores NaNs)
    # Find optimal thresholds on validation set
    optimal_thresholds, optimal_f1_scores = find_optimal_threshold_per_task(all_labels, probs)
    
    # Apply optimal thresholds for F1-score calculation
    binary_preds = apply_thresholds_to_predictions(probs, optimal_thresholds)
    
    metrics = compute_metrics(all_labels, probs)
    
    # Compute F1 with optimal thresholds
    metrics_f1_optimal = compute_metrics(all_labels, binary_preds)
    metrics["f1_score_optimal"] = metrics_f1_optimal["f1_score"]
    metrics["optimal_thresholds"] = optimal_thresholds.tolist()

    return avg_loss, metrics
