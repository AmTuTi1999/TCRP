"""Classification evaluation metrics for TCRP experiments (TC-03).

All functions operate on plain Python lists or numpy arrays of predictions
and ground-truth labels. No dependency on scikit-learn is required.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader


def accuracy(y_pred: np.ndarray | list, y_true: np.ndarray | list) -> float:
    """Overall classification accuracy."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    return float((y_pred == y_true).mean())


def macro_f1(y_pred: np.ndarray | list, y_true: np.ndarray | list) -> float:
    """Macro-averaged F1 score (unweighted mean over classes)."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def per_class_accuracy(
    y_pred: np.ndarray | list, y_true: np.ndarray | list
) -> dict[int, float]:
    """Per-class recall (accuracy conditioned on true class)."""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    result = {}
    for c in np.unique(y_true):
        mask = y_true == c
        result[int(c)] = float((y_pred[mask] == c).mean()) if mask.any() else 0.0
    return result


def confusion_matrix(
    y_pred: np.ndarray | list, y_true: np.ndarray | list, n_classes: int | None = None
) -> np.ndarray:
    """Confusion matrix of shape (n_classes, n_classes).

    Entry [i, j] = number of samples with true label i predicted as j.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    if n_classes is None:
        n_classes = int(max(y_pred.max(), y_true.max())) + 1
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=False):
        cm[int(t), int(p)] += 1
    return cm


@torch.no_grad()
def evaluate_all(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
    class_weights: Tensor | None = None,
) -> dict:
    """Run model over loader and return full classification metrics.

    Args:
        model: TCRPClassifier (or any module whose forward returns TCRPOutput
               with y_hat of shape (B, C)).
        loader: DataLoader yielding (x, y) batches.
        device: Target device (defaults to CPU).
        class_weights: Optional (C,) tensor for weighted cross-entropy reporting.

    Returns:
        Dict with keys: accuracy, macro_f1, per_class_accuracy, confusion_matrix,
        cross_entropy, n_samples.
    """
    if device is None:
        device = torch.device("cpu")
    model.eval()
    model.to(device)

    all_pred, all_true, all_logits = [], [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        logits = out.y_hat  # (B, C)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_pred.extend(preds.tolist())
        all_true.extend(y.tolist())
        all_logits.append(logits.cpu())

    y_pred = np.array(all_pred)
    y_true = np.array(all_true)
    logits_cat = torch.cat(all_logits, dim=0)

    ce = torch.nn.functional.cross_entropy(
        logits_cat,
        torch.tensor(y_true, dtype=torch.long),
        weight=class_weights,
    ).item()

    n_classes = logits_cat.shape[1]
    return {
        "accuracy": accuracy(y_pred, y_true),
        "macro_f1": macro_f1(y_pred, y_true),
        "per_class_accuracy": per_class_accuracy(y_pred, y_true),
        "confusion_matrix": confusion_matrix(y_pred, y_true, n_classes),
        "cross_entropy": ce,
        "n_samples": len(y_true),
    }
