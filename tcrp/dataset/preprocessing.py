"""
Phase 10 · Preprocessing utilities.

T-20: zscore_normalise, inverse_transform, check_no_leakage
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def zscore_normalise(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit z-score statistics on train, then transform all three splits.

    Returns:
        (train_norm, val_norm, test_norm, mean, std)
        mean and std have shape (V,) for multivariate or () for 1-D input.
    """
    train = np.asarray(train, dtype=np.float32)
    val = np.asarray(val, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)

    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)

    return (
        (train - mean) / std,
        (val - mean) / std,
        (test - mean) / std,
        mean,
        std,
    )


def inverse_transform(
    y_pred: Tensor,
    mean: np.ndarray | Tensor,
    std: np.ndarray | Tensor,
) -> Tensor:
    """Undo z-score normalisation for metric computation.

    Args:
        y_pred: Normalised predictions, shape (B, H) or (B, H, V).
        mean: Per-channel mean fitted on the training split.
        std: Per-channel std fitted on the training split.
    """
    if isinstance(mean, np.ndarray):
        mean = torch.from_numpy(mean.astype(np.float32))
    if isinstance(std, np.ndarray):
        std = torch.from_numpy(std.astype(np.float32))

    mean = mean.to(y_pred.device)
    std = std.to(y_pred.device)

    return y_pred * std + mean


def check_no_leakage(
    train_indices,
    val_indices,
    test_indices,
) -> None:
    """Assert that train, val, and test index sets are pairwise disjoint.

    Raises:
        AssertionError: If any pair of splits shares one or more indices.
    """
    train_set = set(int(i) for i in train_indices)
    val_set = set(int(i) for i in val_indices)
    test_set = set(int(i) for i in test_indices)

    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set

    assert not overlap_tv, (
        f"Train/val overlap: {len(overlap_tv)} shared indices "
        f"(e.g. {sorted(overlap_tv)[:5]})"
    )
    assert not overlap_tt, (
        f"Train/test overlap: {len(overlap_tt)} shared indices "
        f"(e.g. {sorted(overlap_tt)[:5]})"
    )
    assert not overlap_vt, (
        f"Val/test overlap: {len(overlap_vt)} shared indices "
        f"(e.g. {sorted(overlap_vt)[:5]})"
    )
