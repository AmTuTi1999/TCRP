"""PipelineConfig — flat dataclass that covers dataset, model, and training settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from omegaconf import DictConfig

from tcrp.model.utils.types import TCRPConfig


@dataclass
class PipelineConfig:
    """Flat dataclass covering dataset, model, and training pipeline settings."""

    # ── Dataset ────────────────────────────────────────────────────────
    dataset: str  # ETTh1 | ETTm2 | Weather | ExchangeRate
    data_root: str = "tcrp/data/raw"
    univariate: bool = True
    # Overrides DATASET_META target; None → DATASET_META default (or last col)
    target_col: str | None = None

    # ── Model ───────────────────────────────────────────────────────────
    T: int = 336  # look-back window (data pipeline only)
    H: int = 96  # forecast horizon
    L: int = 20  # segment length
    stride: int = 5  # segmentation stride
    periods: list[int] = field(default_factory=lambda: [24, 168])
    k_max: int = 2  # ACF lags in concept vector
    alpha: float = 5.0  # monotonicity sigmoid sharpness
    beta: float = 5.0  # curvature sigmoid sharpness
    lambda1: float = 0.1  # alignment loss weight
    lambda2: float = 1e-4  # projection regularisation weight
    lambda3: float = 0.01  # stability loss weight (adversarial only)
    probabilistic: bool = False
    include_gbm: bool = False
    adversarial: bool = False  # enable adversarial (GRL) training
    alpha_max: float = 1.0  # max GRL reversal strength
    warmup_epochs: int = 20  # epochs before GRL activates
    # ── Encoder ─────────────────────────────────────────────────────────
    encoder_in_ch: int = 1
    encoder_hidden: int = 64  # latent dim d (encoder output = projection input)
    tcn_encoder_n_layers: int = 4
    tcn_encoder_kernel_size: int = 3
    tcn_encoder_use_weight_norm: bool = True
    lstm_encoder_bidirectional: bool = False
    lstm_encoder_dropout: float = 0.0
    lstm_encoder_pooling: str = "last"  # "last" | "mean"
    # ── Attention pool ───────────────────────────────────────────────────
    attention_hidden: int = 32
    attention_temp: float = 1.0
    # ── Baseline models ─────────────────────────────────────────────────
    model_type: str = "tcrp"  # tcrp | nbeats | lstm | tcn
    baseline_hidden: int = 256  # hidden/d_model dim for baseline models
    baseline_layers: int = (
        3  # depth (lstm layers / tcn blocks / nbeats blocks-per-stack)
    )

    # ── Training ────────────────────────────────────────────────────────
    lr: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    lr_patience: int = 5  # ReduceLROnPlateau patience
    lr_factor: float = 0.5  # ReduceLROnPlateau factor
    grad_clip: float = 1.0  # 0 = disabled
    num_workers: int = 0
    seed: int = 42

    # ── I/O ─────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    run_name: str = ""  # auto-generated if empty

    @property
    def K(self) -> int:
        """Return the total number of concept dimensions."""
        return 16 + self.k_max + len(self.periods)


def tcrp_config_from_hydra(cfg: DictConfig) -> TCRPConfig:  # type: ignore[name-defined]
    """Build a TCRPConfig from a Hydra DictConfig with nested groups.

    Expects ``cfg.models`` for model params, ``cfg.H`` for forecast horizon.
    """
    m = cfg.models
    return TCRPConfig(
        H=int(cfg.H),
        L=int(m.L),
        stride=int(m.stride),
        d=int(m.encoder_hidden),
        K=0,
        periods=list(cfg.datasets.periods),
        k_max=int(m.k_max),
        alpha=float(m.alpha),
        beta=float(m.beta),
        gamma=float(m.gamma),
        gamma_j=float(m.gamma_j),
        jump_threshold=float(m.jump_threshold),
        train_std=float(m.train_std),
        lambda1=float(m.lambda1),
        lambda2=float(m.lambda2),
        lambda3=float(m.lambda3),
        probabilistic=bool(m.probabilistic),
        adversarial=bool(m.adversarial),
        alpha_max=float(m.alpha_max),
        warmup_epochs=int(m.warmup_epochs),
        encoder_in_ch=int(m.encoder_in_ch),
        encoder_hidden=int(m.encoder_hidden),
        tcn_encoder_n_layers=int(m.tcn_encoder_n_layers),
        tcn_encoder_kernel_size=int(m.tcn_encoder_kernel_size),
        tcn_encoder_use_weight_norm=bool(m.tcn_encoder_use_weight_norm),
        lstm_encoder_bidirectional=bool(m.lstm_encoder_bidirectional),
        lstm_encoder_dropout=float(m.lstm_encoder_dropout),
        lstm_encoder_pooling=str(m.lstm_encoder_pooling),
        attention_hidden=int(m.attention_hidden),
        attention_temp=float(m.attention_temp),
    )


def load_config(path: str | Path) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file.

    Two composition modes are supported:

    **``base:``** (single-file inheritance) — the base file is loaded first,
    then the current file's keys override it::

        base: ../datasets/etth1.yaml
        H: 192
        run_name: etth1_H192

    **``defaults:``** (multi-file composition, Hydra-style) — each entry names
    a YAML file relative to the config root (``configs/``).  Files are merged
    in order; later entries and the current file's own keys win::

        defaults:
          - models/tcrp
          - datasets/etth1
          - trainers/tcrp_trainer
        H: 96
    """
    path = Path(path).resolve()
    raw: dict = yaml.safe_load(path.read_text()) or {}

    if "base" in raw:
        base_path = (path.parent / raw.pop("base")).resolve()
        base: dict = yaml.safe_load(base_path.read_text()) or {}
        base.update(raw)
        raw = base

    elif "defaults" in raw:
        defaults = raw.pop("defaults")
        # Resolve relative to the configs/ root (parent of the entry-point file)
        configs_root = path.parent
        merged: dict = {}
        for entry in defaults:
            ref = entry if isinstance(entry, str) else next(iter(entry.values()))
            ref_path = (configs_root / f"{ref}.yaml").resolve()
            sub = yaml.safe_load(ref_path.read_text()) or {}
            merged.update(sub)
        merged.update(raw)
        raw = merged

    return PipelineConfig(**raw)
