"""Component-level configuration dataclasses for TCRP model building blocks."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TCNEncoderConfig:
    """Configuration for the TCN segment encoder."""

    in_ch: int = 1
    hidden: int = 64
    n_layers: int = 4
    kernel_size: int = 3
    use_weight_norm: bool = True


@dataclass
class LSTMEncoderConfig:
    """Configuration for the LSTM segment encoder."""

    in_ch: int = 1
    hidden: int = 64
    n_layers: int = 2
    bidirectional: bool = False
    dropout: float = 0.0
    pooling: str = "last"


@dataclass
class HorizonDecoderConfig:
    """Configuration for the linear horizon decoder."""

    K: int = 20
    H: int = 96


@dataclass
class GaussianDecoderConfig:
    """Configuration for the Gaussian probabilistic decoder."""

    K: int = 20
    H: int = 96


@dataclass
class SegmenterConfig:
    """Configuration for the sliding-window segmenter."""

    L: int = 20
    stride: int = 5


@dataclass
class ConceptScorerConfig:
    """Configuration for the analytic concept scorer."""

    alpha: float = 5.0
    beta: float = 5.0
    periods: list[int] = field(default_factory=lambda: [24, 168])
    gamma: float = 0.5
    gamma_j: float = 2.0
    k_max: int = 2
    jump_threshold: float = 3.0
    train_std: float = 1.0


@dataclass
class ConceptAttentionPoolConfig:
    """Configuration for the additive concept attention pool."""

    K: int = 20
    hidden: int = 32
    temp: float = 1.0
