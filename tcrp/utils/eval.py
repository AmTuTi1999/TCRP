"""Evaluation helpers shared across training and comparison scripts."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from tcrp.dataset.preprocessing import inverse_transform
from tcrp.model.tcrp_forecaster.components.bottleneck import alignment_loss


def forward_y_hat(model: nn.Module, x: Tensor) -> Tensor:
    """Return y_hat regardless of model type.

    AdversarialTCRPForecaster returns a plain 2-tuple (TCRPOutput, A_align).
    All other models return a NamedTuple whose first field is y_hat.
    NamedTuples inherit from tuple, so we use type() not isinstance().
    """
    out = model(x)
    if type(out) is tuple:          # plain tuple → adversarial: (TCRPOutput, A_align)
        return out[0].y_hat
    return out.y_hat                # NamedTuple (TCRPOutput, BaselineOutput, …)


def eval_denorm(
    model: nn.Module,
    loader: DataLoader,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> dict:
    """Evaluate model on loader, returning denormalised MSE / MAE / RMSE."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            y_hat = forward_y_hat(model, x.to(device))
            preds.append(inverse_transform(y_hat.cpu(), mean, std))
            targets.append(inverse_transform(y, mean, std))
    p = torch.cat(preds)
    t = torch.cat(targets)
    mse = float(torch.mean((p - t) ** 2))
    mae = float(torch.mean(torch.abs(p - t)))
    return {"mse": round(mse, 6), "mae": round(mae, 6), "rmse": round(mse ** 0.5, 6)}


def compute_cas(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Mean Concept Alignment Score (CAS) across all batches and concepts."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = model(x)
            if type(out) is tuple:   # adversarial wrapper
                out = out[0]
            total += alignment_loss(out.A, out.C).item() * x.shape[0]
            count += x.shape[0]
    return total / max(count, 1)


def gather_segments(
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 5,
) -> Tensor:
    """Collect raw input tensors from up to max_batches for purity diagnostics."""
    xs = []
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        xs.append(x.to(device))
    return torch.cat(xs, dim=0)
