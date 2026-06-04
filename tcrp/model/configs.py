"""Component-level configuration dataclasses for TCRP model building blocks."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class TCNEncoderConfig:
    in_ch:           int  = 1
    hidden:          int  = 64
    n_layers:        int  = 4
    kernel_size:     int  = 3
    use_weight_norm: bool = True


@dataclass
class LSTMEncoderConfig:
    in_ch:         int   = 1
    hidden:        int   = 64
    n_layers:      int   = 2
    bidirectional: bool  = False
    dropout:       float = 0.0
    pooling:       str   = 'last'


@dataclass
class HorizonDecoderConfig:
    K: int = 20
    H: int = 96


@dataclass
class GaussianDecoderConfig:
    K: int = 20
    H: int = 96


@dataclass
class SegmenterConfig:
    L:      int = 20
    stride: int = 5


@dataclass
class ConceptScorerConfig:
    alpha:          float     = 5.0
    beta:           float     = 5.0
    periods:        List[int] = field(default_factory=lambda: [24, 168])
    gamma:          float     = 0.5
    gamma_j:        float     = 2.0
    k_max:          int       = 2
    jump_threshold: float     = 3.0
    train_std:      float     = 1.0


@dataclass
class ConceptAttentionPoolConfig:
    K:      int   = 20
    hidden: int   = 32
    temp:   float = 1.0
