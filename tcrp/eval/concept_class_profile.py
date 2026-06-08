"""Concept relevance profiles per class for TCRPClassifier (TC-04).

For each class c, computes the mean concept relevance vector over all test
samples predicted as class c. Positive mean relevance means the concept
pushed the model toward that class; negative means it pushed away.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from tcrp.model.classifier import TCRPClassifier


@torch.no_grad()
def class_concept_profiles(
    model: TCRPClassifier,
    loader: DataLoader,
    concept_names: list[str],
    device: torch.device | None = None,
) -> dict[int, dict[str, float]]:
    """Compute mean concept relevance per predicted class.

    For each correctly predicted sample, the concept relevance vector is the
    pooled concept activation h ∈ R^K. We take the mean per class so each
    entry reflects the "concept fingerprint" of that class.

    Args:
        model: Trained TCRPClassifier.
        loader: DataLoader yielding (x, y) pairs.
        concept_names: List of K concept names (in order matching model.config.K).
        device: Computation device.

    Returns:
        Dict mapping class_idx → {concept_name: mean_relevance}.
        Only classes with at least one correctly predicted sample are included.
    """
    if device is None:
        device = torch.device("cpu")
    model.eval()
    model.to(device)

    K = model.config.K
    C = model.config.C
    assert (
        len(concept_names) == K
    ), f"concept_names length {len(concept_names)} != K={K}"

    # Accumulate h vectors per predicted class (correct predictions only)
    accum: dict[int, list[np.ndarray]] = {c: [] for c in range(C)}

    for x, y_true in loader:
        x = x.to(device)
        out = model(x)
        preds = out.y_hat.argmax(dim=-1).cpu()
        h = out.h.cpu().numpy()  # (B, K)
        y_true_np = np.asarray(y_true)

        for i in range(len(preds)):
            p = int(preds[i])
            if p == int(y_true_np[i]):  # correct predictions only
                accum[p].append(h[i])

    profiles: dict[int, dict[str, float]] = {}
    for c in range(C):
        if not accum[c]:
            continue
        mean_h = np.mean(accum[c], axis=0)  # (K,)
        profiles[c] = {name: float(mean_h[k]) for k, name in enumerate(concept_names)}

    return profiles


@torch.no_grad()
def all_samples_concept_matrix(
    model: TCRPClassifier,
    loader: DataLoader,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return full (N, K) concept activation matrix, predictions, and true labels.

    Useful for downstream analysis: clustering, CAS computation, bypass ratio.

    Returns:
        H       : (N, K) concept activations (pooled h vectors)
        y_pred  : (N,) predicted class indices
        y_true  : (N,) true class indices
    """
    if device is None:
        device = torch.device("cpu")
    model.eval()
    model.to(device)

    H_list, pred_list, true_list = [], [], []
    for x, y_true in loader:
        x = x.to(device)
        out = model(x)
        H_list.append(out.h.cpu().numpy())
        pred_list.append(out.y_hat.argmax(dim=-1).cpu().numpy())
        true_list.append(np.asarray(y_true))

    return (
        np.concatenate(H_list, axis=0),
        np.concatenate(pred_list, axis=0),
        np.concatenate(true_list, axis=0),
    )
