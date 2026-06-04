
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, NamedTuple

from torch import Tensor


class TCRPOutput(NamedTuple):
    y_hat: Tensor
    h: Tensor
    A: Tensor
    C: Tensor
    eta: Tensor


@dataclass
class TCRPConfig:
    H: int
    L: int
    stride: int
    d: int
    K: int                                       # recomputed in __post_init__
    periods: List[int] 
    alpha: float 
    beta: float 
    gamma: float                                # stochasticity (xi) vol-scale
    gamma_j: float                               # jump sigmoid sharpness
    k_max: int                                   # ACF lags
    jump_threshold: float                        # jump detection threshold (σ)
    train_std: float                            # dataset std for σ̃ normalisation
    lambda1: float 
    lambda2: float
    lambda3: float
    probabilistic: bool
    adversarial: bool
    alpha_max: float 
    warmup_epochs: int

    encoder_in_ch:           int  
    encoder_hidden:          int

    tcn_encoder_n_layers:        int
    tcn_encoder_kernel_size:     int
    tcn_encoder_use_weight_norm: bool


    lstm_encoder_bidirectional: bool
    lstm_encoder_dropout: float
    lstm_encoder_pooling: str

    attention_hidden: int
    attention_temp: float 


    def __post_init__(self):
        # K = 16 fixed + k_max ACF lags + len(periods) periodicity scores
        self.K = 16 + self.k_max + len(self.periods)