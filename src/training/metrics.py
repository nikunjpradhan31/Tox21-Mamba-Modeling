import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def compute_metrics(y_true, y_pred):
    """
    Computes macro-averaged ROC-AUC, PRC-AUC, and F1-score for multi-task classification.
    Ignores NaN labels in ground truth.

    Args:
        y_true: Ground truth labels (numpy array or torch tensor) of shape (N, num_tasks).
                Contains NaNs for missing labels.
        y_pred: Predicted probabilities or logits (numpy array or torch tensor) of shape (N, num_tasks)

    Returns:
        Dictionary with macro-averaged metrics across valid tasks.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    roc_aucs = []
    prc_aucs = []
    f1_scores = []

    if len(y_true.shape) == 1:
        y_true = np.expand_dims(y_true, axis=-1)
        y_pred = np.expand_dims(y_pred, axis=-1)

    num_tasks = y_true.shape[1]

    for i in range(num_tasks):
        # Extract true labels and predictions for the i-th task
        task_y_true = y_true[:, i]
        task_y_pred = y_pred[:, i]

        # Filter out NaNs from ground truth
        valid_indices = ~np.isnan(task_y_true)
        valid_y_true = task_y_true[valid_indices]
        valid_y_pred = task_y_pred[valid_indices]

        # Also filter out NaNs from predictions if any (e.g., due to model instability)
        non_nan_pred = ~np.isnan(valid_y_pred)
        valid_y_true = valid_y_true[non_nan_pred]
        valid_y_pred = valid_y_pred[non_nan_pred]

        # Need at least one positive and one negative sample to compute metrics
        if len(np.unique(valid_y_true)) > 1 and len(valid_y_pred) > 0:
            roc_auc = roc_auc_score(valid_y_true, valid_y_pred)
            prc_auc = average_precision_score(valid_y_true, valid_y_pred)

            # Use input as binary predictions if they appear to be binary (0/1 values)
            if np.all(np.isin(valid_y_pred, [0, 1])):
                # Input is already binary predictions
                f1 = f1_score(valid_y_true, valid_y_pred, zero_division=0)
            else:
                # Input is probabilities, use 0.5 threshold
                valid_y_pred_binary = (valid_y_pred >= 0.5).astype(int)
                f1 = f1_score(valid_y_true, valid_y_pred_binary, zero_division=0)

            roc_aucs.append(roc_auc)
            prc_aucs.append(prc_auc)
            f1_scores.append(f1)

    return {
        "roc_auc": np.mean(roc_aucs) if roc_aucs else 0.0,
        "prc_auc": np.mean(prc_aucs) if prc_aucs else 0.0,
        "f1_score": np.mean(f1_scores) if f1_scores else 0.0,
    }
