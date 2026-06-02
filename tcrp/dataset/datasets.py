"""
Phase 10 · Dataset loaders for time-series benchmarks.

T-19: TimeSeriesDataset and get_loaders

Supported datasets: ETTh1, ETTm2, Weather, ExchangeRate, GEFCOM2014
Split convention: train 60% / val 20% / test 20% (TimesNet convention).
Normalisation statistics are always fit on the training portion only.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# Per-dataset metadata: CSV filename, target column for univariate mode,
# and the name of any date/timestamp column to drop.
DATASET_META: dict = {
    "ETTh1": {
        "filename": "ETTh1.csv",
        "target": "OT",
        "date_col": "date",
    },
    "ETTm2": {
        "filename": "ETTm2.csv",
        "target": "OT",
        "date_col": "date",
    },
    "Weather": {
        "filename": "weather.csv",
        "target": "WetBulbCelsius",
        "date_col": "date",
    },
    "ExchangeRate": {
        # No explicit date column; last column used as univariate target.
        "filename": "exchange_rate.csv",
        "target": None,
        "date_col": None,
    },
    "GEFCOM2014": {
        "filename": "gefcom2014.csv",
        "target": "LOAD",
        "date_col": "date",
    },
}

# Fractional boundaries for each split.
_SPLIT_BOUNDS = {
    "train": (0.0, 0.6),
    "val":   (0.6, 0.8),
    "test":  (0.8, 1.0),
}


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for univariate or multivariate time series.

    Normalisation (z-score, per channel) is always fitted on the train
    portion (first 60%) so val/test statistics do not leak into training.

    Args:
        path: Path to the CSV file.
        split: One of 'train', 'val', 'test'.
        T: Look-back window length.
        H: Forecast horizon.
        normalise: Apply per-channel z-score normalisation (default True).
        target_col: Column name for univariate mode. When None and
                    univariate=True, the last numeric column is used.
        univariate: If True, return 1-D tensors (T,)/(H,) for the target
                    channel. If False, return (T, V)/(H, V) for all channels.
    """

    mean: np.ndarray
    std: np.ndarray

    def __init__(
        self,
        path: str,
        split: str,
        T: int,
        H: int,
        normalise: bool = True,
        target_col: Optional[str] = None,
        univariate: bool = True,
    ) -> None:
        if split not in _SPLIT_BOUNDS:
            raise ValueError(
                f"split must be one of {list(_SPLIT_BOUNDS)}, got '{split}'"
            )

        self.T = T
        self.H = H
        self.normalise = normalise
        self.univariate = univariate

        # ── Load CSV ────────────────────────────────────────────────────────
        df = pd.read_csv(path)

        # Drop timestamp columns (date/time) by name or dtype.
        date_like_cols = [
            c for c in df.columns
            if c.lower() in ("date", "datetime", "timestamp", "time")
        ]
        df = df.drop(columns=date_like_cols, errors="ignore")
        df = df.select_dtypes(include=[np.number])

        if df.empty:
            raise ValueError(f"No numeric columns found in '{path}'.")

        data = df.values.astype(np.float32)  # (N, V)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        N, V = data.shape

        # ── Split boundaries ────────────────────────────────────────────────
        train_end = int(N * 0.6)
        val_end   = int(N * 0.8)

        split_slices = {
            "train": slice(0,         train_end),
            "val":   slice(train_end, val_end),
            "test":  slice(val_end,   N),
        }

        train_data = data[split_slices["train"]]

        # ── Normalisation statistics (train only) ───────────────────────────
        self.mean = train_data.mean(axis=0)               # (V,)
        self.std  = train_data.std(axis=0)                # (V,)
        self.std  = np.where(self.std == 0.0, 1.0, self.std)

        split_data = data[split_slices[split]].copy()

        if normalise:
            split_data = (split_data - self.mean) / self.std

        # ── Channel selection for univariate mode ───────────────────────────
        if univariate:
            if target_col is not None:
                col_names = list(df.columns)
                if target_col not in col_names:
                    raise ValueError(
                        f"target_col '{target_col}' not found in {col_names}"
                    )
                col_idx = col_names.index(target_col)
            else:
                col_idx = V - 1  # last column by convention
            split_data = split_data[:, col_idx : col_idx + 1]
            # Narrow statistics to match the single output channel so that
            # inverse_transform(y_hat, mean, std) broadcasts correctly against (B, H).
            self.mean = self.mean[col_idx : col_idx + 1]
            self.std  = self.std[col_idx : col_idx + 1]

        self.data = split_data  # (M, 1) univariate | (M, V) multivariate

        M = len(self.data)
        if M < T + H:
            raise ValueError(
                f"Split '{split}' contains only {M} timesteps; "
                f"need at least T+H = {T}+{H} = {T + H}. "
                "Reduce T/H or use a larger dataset."
            )

        self.n_samples = M - T - H + 1

    # ── Dataset protocol ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        x = self.data[idx       : idx + self.T]
        y = self.data[idx + self.T : idx + self.T + self.H]

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)

        if self.univariate:
            # Squeeze the singleton channel dimension.
            x_t = x_t.squeeze(-1)  # (T,)
            y_t = y_t.squeeze(-1)  # (H,)

        return x_t, y_t


def get_loaders(
    dataset_name: str,
    T: int = 336,
    H: int = 96,
    normalise: bool = True,
    univariate: bool = True,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:
    """Build DataLoader wrappers for train/val/test splits.

    Args:
        dataset_name: One of the keys in DATASET_META.
        data_root: Directory that contains the dataset CSV files.
        T: Look-back window length.
        H: Forecast horizon.
        normalise: Apply z-score normalisation.
        univariate: Return single-channel tensors if True.
        batch_size: Batch size for all loaders.
        num_workers: Worker processes for DataLoader.

    Returns:
        Dict with keys 'train', 'val', 'test' (DataLoader) and
        'mean', 'std' (np.ndarray, train-split normalisation statistics
        for use with inverse_transform).
    """
    if dataset_name not in DATASET_META:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: {list(DATASET_META)}"
        )

    meta = DATASET_META[dataset_name]
    path = os.path.join("tcrp/data/raw", meta["filename"])
    target_col = meta["target"] if univariate else None

    def _make(split: str) -> TimeSeriesDataset:
        return TimeSeriesDataset(
            path=path,
            split=split,
            T=T,
            H=H,
            normalise=normalise,
            target_col=target_col,
            univariate=univariate,
        )

    train_ds = _make("train")
    val_ds   = _make("val")
    test_ds  = _make("test")

    return {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        # Expose train-split statistics for inverse_transform.
        "mean": train_ds.mean,
        "std":  train_ds.std,
    }
