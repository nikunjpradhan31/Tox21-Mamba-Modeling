import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Union, Tuple


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Trains the model for one epoch.

    Args:
        model: PyTorch model
        dataloader: DataLoader yielding training batches
        optimizer: Optimizer
        criterion: Loss function (should be BCEWithLogitsLoss with reduction='none')
        device: Device to train on

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch in dataloader:
        if hasattr(batch, "y") and hasattr(batch, "x"):
            # PyTorch Geometric Batch object
            labels = batch.y.to(device)
            batch = batch.to(device)
            outputs = model(batch)
        elif isinstance(batch, dict):
            # Assume dict contains inputs and 'labels'
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
        elif isinstance(batch, (tuple, list)):
            # Assume last element is labels, rest are inputs
            inputs, labels = batch[:-1], batch[-1]
            labels = labels.to(device)
            if len(inputs) == 1:
                outputs = model(inputs[0].to(device))
            else:
                outputs = model(*[inp.to(device) for inp in inputs])
        else:
            raise ValueError("Unexpected batch format")

        # Extract logits from model outputs
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        if logits.shape != labels.shape:
            logits = logits.view(labels.shape)

        # Create mask for valid labels (non-NaN)
        valid_mask = ~torch.isnan(labels)
        # Replace NaNs in labels with 0 to safely compute loss
        safe_labels = torch.where(valid_mask, labels, torch.zeros_like(labels))
        # Compute element-wise loss
        loss_matrix = criterion(logits, safe_labels)
        # Apply mask and compute mean loss over valid elements
        masked_loss = loss_matrix[valid_mask]

        if masked_loss.numel() > 0:
            loss = masked_loss.mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_batches if num_batches > 0 else 0.0


class Trainer:
    """
    Convenience Trainer class that wraps the training loop.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(self.device)

    def train_epoch(self, dataloader: DataLoader) -> float:
        return train_epoch(
            self.model, dataloader, self.optimizer, self.criterion, self.device
        )
