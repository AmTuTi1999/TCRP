"""Classification dataset loaders for TCRP experiments (TC-02).

Supported datasets and their local paths:
  ECG5000     — tcrp/data/raw/ECG5000/ECG5000_{TRAIN,TEST}.txt
  MIT-BIH     — tcrp/data/raw/CRWU/mitbih_{train,test}.csv
  CWRU        — requires download (see CWRU_DOWNLOAD note below)
  Sleep-EDF   — requires download (see SLEEP_EDF_DOWNLOAD note below)
  UCI-HAR     — requires download (see UCI_HAR_DOWNLOAD note below)
  Ethanol     — requires download (UCR archive)
  SP500       — requires download (Yahoo Finance / NBER)
  FX          — tcrp/data/raw/HISTDATA_COM_ASCII_EURUSD_T202512/

All loaders return (x, y) pairs:
  x : Tensor (T,)  for univariate series
  y : int          class label (0-indexed)

Preprocessing: z-score normalised using training-split statistics only.
For imbalanced datasets (MIT-BIH, Sleep-EDF): class weights are exposed
via the `class_weights` attribute for use in weighted cross-entropy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ── Download notes (printed when data is missing) ───────────────────────────

_CWRU_DOWNLOAD = """
CWRU Bearing dataset not found. Download from:
  https://engineering.case.edu/bearingdatacenter/download-data-file
Place the raw .mat files in tcrp/data/raw/CWRU/ and run:
  python scripts/prepare_cwru.py
"""

_SLEEP_EDF_DOWNLOAD = """
Sleep-EDF Cassette dataset not found. Download from Physionet:
  pip install wfdb
  python -c "import wfdb; wfdb.dl_database('sleep-edfx', 'tcrp/data/raw/Sleep-EDF')"
Then run:  python scripts/prepare_sleep_edf.py
"""

_UCI_HAR_DOWNLOAD = """
UCI-HAR dataset not found. Download from:
  https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
Extract to tcrp/data/raw/UCI-HAR/
"""

_ETHANOL_DOWNLOAD = """
EthanolConcentration dataset not found. Download from UCR Archive:
  http://www.timeseriesclassification.com/description.php?Dataset=EthanolConcentration
Place .arff files in tcrp/data/raw/EthanolConcentration/
"""

_SP500_DOWNLOAD = """
S&P 500 dataset not found. Obtain daily OHLCV data (e.g. via yfinance):
  pip install yfinance
  python scripts/prepare_sp500.py
Outputs: tcrp/data/raw/SP500/sp500_log_returns.csv
         tcrp/data/raw/SP500/sp500_labels_recession.csv
         tcrp/data/raw/SP500/sp500_labels_vix_regime.csv
"""

DATA_ROOT = Path("tcrp/data/raw")


# ── ECG5000 ─────────────────────────────────────────────────────────────────


class ECG5000Dataset(Dataset):
    """ECG5000 arrhythmia classification dataset.

    Source: UCR Time Series Archive.
    T=140, C=5 (normal sinus, R-on-T PVC, PVC, paced beat, others).
    Labels in file are 1–5; mapped to 0–4.

    Args:
        split: 'train' or 'test'.
        data_root: Path to the directory containing ECG5000_{TRAIN,TEST}.txt.
        mean: Pre-computed training mean for normalisation (supplied for test split).
        std:  Pre-computed training std for normalisation.
    """

    T: int = 140
    C: int = 5

    def __init__(
        self,
        split: str,
        data_root: str | Path = DATA_ROOT / "ECG5000",
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        """Load and z-score normalise the requested split from ECG5000 text files."""
        path = Path(data_root) / f"ECG5000_{split.upper()}.txt"
        if not path.exists():
            raise FileNotFoundError(f"ECG5000 file not found: {path}")

        raw = np.loadtxt(path, dtype=np.float32)
        labels = raw[:, 0].astype(np.int64) - 1  # 1-5 → 0-4
        series = raw[:, 1:]  # (N, T)

        if mean is None:
            self.mean = series.mean(axis=0, keepdims=True)
            self.std = series.std(axis=0, keepdims=True)
            self.std = np.where(self.std == 0, 1.0, self.std)
        else:
            self.mean = mean
            self.std = std

        self.x = torch.from_numpy((series - self.mean) / self.std)
        self.y = torch.from_numpy(labels)

        counts = np.bincount(labels, minlength=self.C).astype(np.float32)
        self.class_weights = torch.from_numpy(1.0 / np.where(counts == 0, 1.0, counts))

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """Return (x, y) pair at index idx."""
        return self.x[idx], self.y[idx].item()


# ── MIT-BIH ─────────────────────────────────────────────────────────────────


class MITBIHDataset(Dataset):
    """MIT-BIH Arrhythmia heartbeat classification dataset.

    Pre-segmented heartbeat windows from Kaggle (Kachuee et al.):
      mitbih_train.csv / mitbih_test.csv
    T=187, C=5 AAMI classes (N=0, S=1, V=2, F=3, Q=4).
    Labels are already 0-indexed in the last column.

    Heavily imbalanced: N dominates. class_weights are inverse class
    frequencies for use in weighted cross-entropy.
    """

    T: int = 187
    C: int = 5

    def __init__(
        self,
        split: str,
        data_root: str | Path = DATA_ROOT / "CRWU",
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        """Load and z-score normalise the requested split from MIT-BIH CSV files."""
        fname = "mitbih_train.csv" if split == "train" else "mitbih_test.csv"
        path = Path(data_root) / fname
        if not path.exists():
            raise FileNotFoundError(f"MIT-BIH file not found: {path}")

        df = pd.read_csv(path, header=None)
        series = df.iloc[:, :-1].values.astype(np.float32)
        labels = df.iloc[:, -1].values.astype(np.int64)

        if mean is None:
            self.mean = series.mean(axis=0, keepdims=True)
            self.std = series.std(axis=0, keepdims=True)
            self.std = np.where(self.std == 0, 1.0, self.std)
        else:
            self.mean = mean
            self.std = std

        self.x = torch.from_numpy((series - self.mean) / self.std)
        self.y = torch.from_numpy(labels)

        counts = np.bincount(labels, minlength=self.C).astype(np.float32)
        self.class_weights = torch.from_numpy(1.0 / np.where(counts == 0, 1.0, counts))

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """Return (x, y) pair at index idx."""
        return self.x[idx], self.y[idx].item()


# ── Generic NPZ loader (CWRU, Sleep-EDF, UCI-HAR, Ethanol, SP500) ───────────


class NPZClassificationDataset(Dataset):
    """Generic loader for pre-processed classification datasets stored as .npz.

    Expected keys in the .npz file:
      x_train, y_train  — training split
      x_val,   y_val    — validation split (optional; falls back to 10% of train)
      x_test,  y_test   — test split

    x arrays shape: (N, T) for univariate.
    y arrays shape: (N,) with integer labels 0..C-1.

    Pre-processing scripts that produce this format are in scripts/prepare_*.py.
    """

    def __init__(
        self,
        npz_path: str | Path,
        split: str,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> None:
        """Load and z-score normalise the requested split from an .npz file."""
        path = Path(npz_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        data = np.load(path, allow_pickle=False)

        x_key = f"x_{split}"
        y_key = f"y_{split}"
        if x_key not in data:
            raise KeyError(f"Key '{x_key}' not in {path}. Available: {list(data)}")

        series = data[x_key].astype(np.float32)
        labels = data[y_key].astype(np.int64)
        n_classes = int(labels.max()) + 1

        if mean is None:
            train_series = data["x_train"].astype(np.float32)
            self.mean = train_series.mean(axis=0, keepdims=True)
            self.std = train_series.std(axis=0, keepdims=True)
            self.std = np.where(self.std == 0, 1.0, self.std)
        else:
            self.mean = mean
            self.std = std

        self.x = torch.from_numpy((series - self.mean) / self.std)
        self.y = torch.from_numpy(labels)

        counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
        self.class_weights = torch.from_numpy(1.0 / np.where(counts == 0, 1.0, counts))

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """Return (x, y) pair at index idx."""
        return self.x[idx], self.y[idx].item()


# ── FX Regime dataset (EXP-C08) ─────────────────────────────────────────────


class FXRegimeDataset(Dataset):
    """FX trend/mean-reversion regime classification from tick data.

    Uses EURUSD tick data from tcrp/data/raw/HISTDATA_COM_ASCII_EURUSD_T202512/.
    Resamples to daily mid-prices, computes log-returns, then labels each
    T=21 day window by rolling Hurst exponent:
      0 — trending       (H > 0.65)
      1 — mean-reverting (H < 0.35)
      2 — random walk    (0.35 ≤ H ≤ 0.65)

    Labels are analytically derived — no human annotation required.
    """

    T: int = 21
    C: int = 3

    def __init__(
        self,
        split: str,
        data_root: str | Path = DATA_ROOT / "HISTDATA_COM_ASCII_EURUSD_T202512",
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        hurst_window: int = 21,
        hurst_hi: float = 0.65,
        hurst_lo: float = 0.35,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        stride: int = 1,
    ) -> None:
        """Build sliding windows with Hurst-derived labels from EURUSD tick data."""
        log_ret = self._load_log_returns(data_root, hurst_window)

        windows, labels = _build_hurst_windows(
            log_ret, self.T, stride, hurst_window, hurst_hi, hurst_lo
        )

        N = len(windows)
        n_train = int(N * train_frac)
        n_val = int(N * val_frac)
        slices = {
            "train": slice(0, n_train),
            "val": slice(n_train, n_train + n_val),
            "test": slice(n_train + n_val, N),
        }
        if split not in slices:
            raise ValueError(f"split must be train/val/test, got '{split}'")

        series = windows[slices[split]]
        labs = labels[slices[split]]

        if mean is None:
            self.mean = windows[slices["train"]].mean(axis=0, keepdims=True)
            self.std = windows[slices["train"]].std(axis=0, keepdims=True)
            self.std = np.where(self.std == 0, 1.0, self.std)
        else:
            self.mean = mean
            self.std = std

        self.x = torch.from_numpy((series - self.mean) / self.std)
        self.y = torch.from_numpy(labs)

        counts = np.bincount(labs, minlength=self.C).astype(np.float32)
        self.class_weights = torch.from_numpy(1.0 / np.where(counts == 0, 1.0, counts))

    def _load_log_returns(self, data_root: str | Path, hurst_window: int) -> np.ndarray:
        """Load daily log-returns from tick CSV; fall back to exchange_rate.csv."""
        min_rows = self.T + hurst_window // 2 + 2
        csv_path = Path(data_root) / "DAT_ASCII_EURUSD_T_202512.csv"
        if csv_path.exists():
            df = pd.read_csv(
                csv_path,
                header=None,
                names=["datetime", "bid", "ask", "vol"],
                parse_dates=["datetime"],
            )
            df["mid"] = (df["bid"] + df["ask"]) / 2.0
            df["datetime"] = pd.to_datetime(
                df["datetime"], format="%Y%m%d %H%M%S%f", errors="coerce"
            )
            df = df.dropna(subset=["datetime"])
            daily = df.set_index("datetime")["mid"].resample("D").last().dropna()
            log_ret = np.log(daily / daily.shift(1)).dropna().values.astype(np.float32)
            if len(log_ret) >= min_rows:
                return log_ret

        # Tick data insufficient — fall back to exchange_rate.csv (column 0: AUD/USD).
        # This file ships with the repo and covers 1990-2010 (~7500 daily rows).
        fallback = DATA_ROOT / "exchange_rate.csv"
        if not fallback.exists():
            raise FileNotFoundError(
                f"FX tick data at {csv_path} has fewer than {min_rows} daily rows "
                f"and fallback {fallback} was not found."
            )
        prices = pd.read_csv(fallback).iloc[:, 1].values.astype(np.float32)
        return np.log(prices[1:] / prices[:-1])

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """Return (x, y) pair at index idx."""
        return self.x[idx], self.y[idx].item()


def _hurst_rs(x: np.ndarray) -> float:
    """Hurst exponent via R/S analysis."""
    n = len(x)
    if n < 4:
        return 0.5
    mean_x = x.mean()
    dev = np.cumsum(x - mean_x)
    R = dev.max() - dev.min()
    S = x.std(ddof=1) + 1e-10
    return float(np.log(R / S) / np.log(n))


def _build_hurst_windows(
    returns: np.ndarray,
    T: int,
    stride: int,
    hurst_window: int,
    hurst_hi: float,
    hurst_lo: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide T-length windows over returns and assign Hurst-based labels."""
    windows, labels = [], []
    hurst_half = hurst_window // 2
    for start in range(0, len(returns) - T - hurst_half, stride):
        w = returns[start : start + T]
        # Use the window itself for Hurst estimation
        h = _hurst_rs(w)
        if h > hurst_hi:
            label = 0  # trending
        elif h < hurst_lo:
            label = 1  # mean-reverting
        else:
            label = 2  # random walk
        windows.append(w)
        labels.append(label)
    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64)


# ── EthanolConcentration ─────────────────────────────────────────────────────


class EthanolConcentrationDataset(Dataset):
    """EthanolConcentration spectroscopy classification dataset (UCR Archive).

    Near-infrared spectra of water-ethanol solutions sampled from 44 whisky
    bottles at four concentrations: 35%, 38%, 40%, 45%.
    T=1751, C=4 classes (E35→0, E38→1, E40→2, E45→3).

    The dataset has 3 spectral channels — repeat readings of the same bottle.
    All 3 are loaded and averaged per sample (noise reduction), giving a single
    univariate series of length T=1751 per sample.

    The raw TRAIN split is divided 80/20 into train/val (stratified by class),
    so the val split is independent of the TEST files.

    Source: http://www.timeseriesclassification.com/description.php?Dataset=EthanolConcentration
    Files expected at data_root/EthanolConcentrationDimension{1,2,3}_{TRAIN,TEST}.arff
    """

    T: int = 1751
    C: int = 4
    _LABEL_MAP: dict[str, int] = {"E35": 0, "E38": 1, "E40": 2, "E45": 3}

    def __init__(
        self,
        split: str,
        data_root: str | Path = DATA_ROOT / "EthanolConcentration",
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        val_frac: float = 0.2,
        seed: int = 42,
    ) -> None:
        """Load all 3 spectral channels, average them, and return the requested split.

        For split='train' or 'val': reads EthanolConcentrationDimension{1,2,3}_TRAIN.arff
        and performs a stratified 80/20 split.
        For split='test': reads EthanolConcentrationDimension{1,2,3}_TEST.arff.
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got '{split}'")

        arff_split = "TEST" if split == "test" else "TRAIN"
        series, labels = self._load_averaged(Path(data_root), arff_split)

        if split in ("train", "val"):
            series, labels = self._stratified_split(
                series, labels, val_frac, seed, split
            )

        if mean is None:
            self.mean = series.mean(axis=0, keepdims=True)
            self.std = series.std(axis=0, keepdims=True)
            self.std = np.where(self.std == 0, 1.0, self.std)
        else:
            self.mean = mean
            self.std = std

        self.x = torch.from_numpy((series - self.mean) / self.std)
        self.y = torch.from_numpy(labels)

        counts = np.bincount(labels, minlength=self.C).astype(np.float32)
        self.class_weights = torch.from_numpy(1.0 / np.where(counts == 0, 1.0, counts))

    def _load_averaged(
        self, data_root: Path, arff_split: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load all 3 dimension ARFF files and return their per-sample average."""
        channels = []
        labels = None
        for dim in (1, 2, 3):
            fname = f"EthanolConcentrationDimension{dim}_{arff_split}.arff"
            path = data_root / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"EthanolConcentration file not found: {path}\n"
                    f"Download from http://www.timeseriesclassification.com/"
                    f"description.php?Dataset=EthanolConcentration\n"
                    f"and extract to {data_root}"
                )
            series_d, labels_d = self._parse_arff(path)
            channels.append(series_d)
            if labels is None:
                labels = labels_d

        averaged = np.mean(np.stack(channels, axis=0), axis=0).astype(np.float32)
        return averaged, labels

    def _stratified_split(
        self,
        series: np.ndarray,
        labels: np.ndarray,
        val_frac: float,
        seed: int,
        which: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stratified split of the TRAIN data into train/val subsets."""
        rng = np.random.default_rng(seed)
        train_idx, val_idx = [], []
        for c in np.unique(labels):
            idx = np.where(labels == c)[0]
            idx = rng.permutation(idx)
            n_val = max(1, int(len(idx) * val_frac))
            val_idx.extend(idx[:n_val].tolist())
            train_idx.extend(idx[n_val:].tolist())

        chosen = np.array(train_idx if which == "train" else val_idx)
        chosen.sort()
        return series[chosen], labels[chosen]

    def _parse_arff(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Parse ARFF: skip comments and @-directives, read CSV data block.

        Last column is the class label; all preceding columns are signal values.
        """
        rows, labels_raw = [], []
        in_data = False
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("%"):
                    continue
                if line.lower() == "@data":
                    in_data = True
                    continue
                if line.startswith("@"):
                    continue
                if in_data:
                    parts = line.split(",")
                    rows.append([float(v) for v in parts[:-1]])
                    labels_raw.append(parts[-1].strip())

        series = np.array(rows, dtype=np.float32)
        labels = np.array([self._LABEL_MAP[lbl] for lbl in labels_raw], dtype=np.int64)
        return series, labels

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """Return (x, y) pair at index idx."""
        return self.x[idx], self.y[idx].item()


# ── Loader factory ───────────────────────────────────────────────────────────

DATASET_REGISTRY: dict[str, type] = {
    "ECG5000": ECG5000Dataset,
    "MITBIH": MITBIHDataset,
    "Ethanol": EthanolConcentrationDataset,
    "FX": FXRegimeDataset,
}

_NPZ_PATHS: dict[str, Path] = {
    "CWRU": DATA_ROOT / "CWRU" / "cwru_classification.npz",
    "SleepEDF": DATA_ROOT / "Sleep-EDF" / "sleep_edf.npz",
    "UCIHAR": DATA_ROOT / "UCI-HAR" / "uci_har.npz",
    "SP500_A": DATA_ROOT / "SP500" / "sp500_recession.npz",
    "SP500_B": DATA_ROOT / "SP500" / "sp500_vix_regime.npz",
}

_NPZ_DOWNLOAD_NOTES: dict[str, str] = {
    "CWRU": _CWRU_DOWNLOAD,
    "SleepEDF": _SLEEP_EDF_DOWNLOAD,
    "UCIHAR": _UCI_HAR_DOWNLOAD,
    "SP500_A": _SP500_DOWNLOAD,
    "SP500_B": _SP500_DOWNLOAD,
}


def build_classification_loaders(
    dataset_name: str,
    batch_size: int = 64,
    num_workers: int = 0,
    weighted_sampling: bool = False,
    **dataset_kwargs,
) -> dict:
    """Build train/val/test DataLoaders for a classification dataset.

    Args:
        dataset_name: One of the keys in DATASET_REGISTRY or _NPZ_PATHS.
        batch_size: DataLoader batch size.
        num_workers: Worker processes.
        weighted_sampling: If True, use WeightedRandomSampler on train split
            to balance class distribution.
        **dataset_kwargs: Passed through to the dataset constructor.

    Returns:
        Dict with 'train', 'val', 'test' DataLoaders and 'class_weights' Tensor.
    """
    pin = torch.cuda.is_available()

    if dataset_name in DATASET_REGISTRY:
        cls = DATASET_REGISTRY[dataset_name]
        train_ds = cls(split="train", **dataset_kwargs)
        # Pass normalisation stats from train to val/test
        val_ds = cls(
            split="val" if _has_val_split(cls) else "test",
            mean=train_ds.mean,
            std=train_ds.std,
            **dataset_kwargs,
        )
        test_ds = cls(
            split="test", mean=train_ds.mean, std=train_ds.std, **dataset_kwargs
        )
    elif dataset_name in _NPZ_PATHS:
        npz = _NPZ_PATHS[dataset_name]
        if not npz.exists():
            raise FileNotFoundError(_NPZ_DOWNLOAD_NOTES[dataset_name])
        train_ds = NPZClassificationDataset(npz, "train")
        val_ds = NPZClassificationDataset(
            npz, "val", mean=train_ds.mean, std=train_ds.std
        )
        test_ds = NPZClassificationDataset(
            npz, "test", mean=train_ds.mean, std=train_ds.std
        )
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY) + list(_NPZ_PATHS)}"
        )

    if weighted_sampling:
        sample_weights = train_ds.class_weights[train_ds.y]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(train_ds), replacement=True
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
        )

    return {
        "train": train_loader,
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
        ),
        "class_weights": train_ds.class_weights,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
    }


def _has_val_split(cls: type) -> bool:
    """Return True for dataset classes that support a 'val' split."""
    return cls in (
        FXRegimeDataset,
        NPZClassificationDataset,
        EthanolConcentrationDataset,
    )
