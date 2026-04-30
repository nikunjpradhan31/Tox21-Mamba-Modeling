import numpy as np
import torch
from sklearn.metrics import f1_score


def find_optimal_threshold_per_task(y_true, y_pred_probs):
    """
    Find optimal thresholds that maximize F1-score for each task individually.
    
    Args:
        y_true: Ground truth labels (numpy array) of shape (N, num_tasks)
        y_pred_probs: Predicted probabilities (numpy array) of shape (N, num_tasks)
    
    Returns:
        optimal_thresholds: Array of optimal thresholds for each task
        optimal_f1_scores: Array of F1-scores at optimal thresholds
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.detach().cpu().numpy()
    
    num_tasks = y_true.shape[1]
    optimal_thresholds = np.zeros(num_tasks)
    optimal_f1_scores = np.zeros(num_tasks)
    
    for task_idx in range(num_tasks):
        task_y_true = y_true[:, task_idx]
        task_y_pred = y_pred_probs[:, task_idx]
        
        # Filter out NaN labels
        valid_mask = ~np.isnan(task_y_true)
        valid_y_true = task_y_true[valid_mask]
        valid_y_pred = task_y_pred[valid_mask]
        
        # Filter out NaN predictions
        non_nan_pred = ~np.isnan(valid_y_pred)
        valid_y_true = valid_y_true[non_nan_pred]
        valid_y_pred = valid_y_pred[non_nan_pred]
        
        if len(np.unique(valid_y_true)) > 1 and len(valid_y_pred) > 10:
            thresholds = np.linspace(0.1, 0.9, 50)
            f1_scores = []
            
            for thresh in thresholds:
                pred_binary = (valid_y_pred >= thresh).astype(int)
                f1 = f1_score(valid_y_true, pred_binary, zero_division=0)
                f1_scores.append(f1)
            
            if len(f1_scores) > 0:
                best_thresh_idx = np.argmax(f1_scores)
                optimal_thresholds[task_idx] = thresholds[best_thresh_idx]
                optimal_f1_scores[task_idx] = f1_scores[best_thresh_idx]
            else:
                optimal_thresholds[task_idx] = 0.5
                optimal_f1_scores[task_idx] = 0.0
        else:
            optimal_thresholds[task_idx] = 0.5
            optimal_f1_scores[task_idx] = 0.0
    
    return optimal_thresholds, optimal_f1_scores


def apply_thresholds_to_predictions(y_pred_probs, thresholds):
    """
    Apply task-specific thresholds to probability predictions.
    
    Args:
        y_pred_probs: Predicted probabilities (numpy array) of shape (N, num_tasks)
        thresholds: Thresholds to apply, shape (num_tasks,)
    
    Returns:
        Binary predictions with task-specific thresholds
    """
    if isinstance(y_pred_probs, torch.Tensor):
        y_pred_probs = y_pred_probs.detach().cpu().numpy()
    
    thresholds = np.array(thresholds)
    if len(thresholds.shape) == 1:
        thresholds = thresholds.reshape(1, -1)
    
    return (y_pred_probs >= thresholds).astype(int)