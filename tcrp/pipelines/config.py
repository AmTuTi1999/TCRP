"""PipelineConfig — flat dataclass that covers dataset, model, and training settings."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class PipelineConfig:
    # ── Dataset ────────────────────────────────────────────────────────
    dataset: str                                   # ETTh1 | ETTm2 | Weather | ExchangeRate
    data_root: str = "tcrp/data/raw"
    univariate: bool = True
    # Overrides DATASET_META target; None → DATASET_META default (or last col)
    target_col: Optional[str] = None

    # ── Model ───────────────────────────────────────────────────────────
    T: int = 336                                   # look-back window
    H: int = 96                                    # forecast horizon
    L: int = 20                                    # segment length
    stride: int = 5                                # segmentation stride
    d: int = 64                                    # encoder hidden dim
    periods: List[int] = field(default_factory=lambda: [24, 168])
    alpha: float = 5.0                             # monotonicity temperature
    beta: float = 5.0                              # curvature temperature
    lambda1: float = 0.1                           # alignment loss weight
    lambda2: float = 1e-4                          # projection regularisation weight
    probabilistic: bool = False
    include_gbm: bool = False

    # ── Training ────────────────────────────────────────────────────────
    lr: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    lr_patience: int = 5                           # ReduceLROnPlateau patience
    lr_factor: float = 0.5                         # ReduceLROnPlateau factor
    grad_clip: float = 1.0                         # 0 = disabled
    num_workers: int = 0
    seed: int = 42

    # ── I/O ─────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    run_name: str = ""                             # auto-generated if empty

    @property
    def K(self) -> int:
        return 4 + len(self.periods)


def load_config(path: str | Path) -> PipelineConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)
