"""Per-epoch training and prediction collection.

These functions are deliberately small and side-effect free (apart from the
optimizer step) so the runner can compose them and the smoke test can exercise
them on a tiny model.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return the average loss per sample."""
    model.train()
    running_loss = 0.0
    seen = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.reshape(-1).long().to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        seen += images.size(0)
    return running_loss / max(seen, 1)


@torch.no_grad()
def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_score)`` over ``loader``.

    ``y_true`` has shape ``(n_samples, 1)`` and ``y_score`` holds softmax
    probabilities of shape ``(n_samples, n_classes)``, matching the metric
    functions.
    """
    model.eval()
    all_scores = []
    all_labels = []
    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        all_scores.append(probabilities.cpu().numpy())
        all_labels.append(targets.reshape(-1, 1).numpy())
    return np.concatenate(all_labels, axis=0), np.concatenate(all_scores, axis=0)
