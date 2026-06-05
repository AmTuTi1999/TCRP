"""Model output and configuration dataclasses for TCRP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from torch import Tensor


class TCRPOutput(NamedTuple):
    """Named tuple holding all outputs from a TCRPForecaster forward pass."""

    y_hat: Tensor
    h: Tensor
    A: Tensor
    C: Tensor
    eta: Tensor


@dataclass
class TCRPConfig:
    """Full configuration for building and training a TCRPForecaster."""

    H: int
    L: int
    stride: int
    d: int
    K: int  # recomputed in __post_init__
    periods: list[int]
    alpha: float
    beta: float
    gamma: float  # stochasticity (xi) vol-scale
    gamma_j: float  # jump sigmoid sharpness
    k_max: int  # ACF lags
    jump_threshold: float  # jump detection threshold (σ)
    train_std: float  # dataset std for σ̃ normalisation
    lambda1: float
    lambda2: float
    lambda3: float
    lambda4: float
    probabilistic: bool
    adversarial: bool
    alpha_max: float
    warmup_epochs: int

    encoder_in_ch: int
    encoder_hidden: int

    tcn_encoder_n_layers: int
    tcn_encoder_kernel_size: int
    tcn_encoder_use_weight_norm: bool

    lstm_encoder_bidirectional: bool
    lstm_encoder_dropout: float
    lstm_encoder_pooling: str

    attention_hidden: int
    attention_temp: float

    def __post_init__(self):
        """Derive K from k_max and periods after field assignment."""
        self.K = 16 + self.k_max + len(self.periods)
