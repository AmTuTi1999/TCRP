"""
Phase 6 · TCRP Forecaster Assembly

Assembles segmentation, encoding, concept projection, scoring, attention, and decoding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from .segmentation import Segmenter
from .encoder import TCNEncoder
from .bottleneck import ConceptProjection
from .aggregation import ConceptAttentionPool
from .decoder import HorizonDecoder, GaussianDecoder
from tcrp.concepts import ConceptScorer


class TCRPOutput(NamedTuple):
    y_hat: Tensor
    h: Tensor
    A: Tensor
    C: Tensor
    eta: Tensor


@dataclass
class TCRPConfig:
    T: int
    H: int
    L: int = 20
    stride: int = 5
    d: int = 64
    K: int = 20                                         # recomputed in __post_init__
    periods: List[int] = field(default_factory=lambda: [24, 168])
    alpha: float = 5.0
    beta: float = 5.0
    gamma: float = 0.5                                  # stochasticity (xi) vol-scale
    gamma_j: float = 2.0                                # jump sigmoid sharpness
    k_max: int = 2                                      # ACF lags
    jump_threshold: float = 3.0                         # jump detection threshold (σ)
    train_std: float = 1.0                              # dataset std for σ̃ normalisation
    lambda1: float = 0.1
    lambda2: float = 1e-4
    probabilistic: bool = False

    def __post_init__(self):
        # K = 16 fixed + k_max ACF lags + len(periods) periodicity scores
        self.K = 16 + self.k_max + len(self.periods)


class TCRPForecaster(nn.Module):
    """TCRP forecaster combining all TCRP components."""

    def __init__(self, config: TCRPConfig):
        super().__init__()
        self.config = config

        self.segmenter = Segmenter(L=config.L, stride=config.stride)
        self.encoder = TCNEncoder(in_ch=1, hidden=config.d, n_layers=4, kernel_size=3)
        self.projection = ConceptProjection(d=config.d, K=config.K)
        self.scorer = ConceptScorer(
            alpha=config.alpha,
            beta=config.beta,
            periods=config.periods,
            gamma=config.gamma,
            gamma_j=config.gamma_j,
            k_max=config.k_max,
            jump_threshold=config.jump_threshold,
            train_std=config.train_std,
        )
        self.pool = ConceptAttentionPool(K=config.K, hidden=32, temp=1.0)
        self.decoder = GaussianDecoder(config.K, config.H) if config.probabilistic else HorizonDecoder(config.K, config.H)

    def forward(self, x: Tensor) -> TCRPOutput:
        """Forward pass through the full TCRP model.

        Args:
            x: Input tensor of shape (B, T)

        Returns:
            TCRPOutput(y_hat, h, A, C, eta)
        """
        if x.dim() != 2:
            raise ValueError(f"Expected input shape (B, T), got {tuple(x.shape)}")

        B, T = x.shape
        segments = self.segmenter(x)
        # segments: (B, N, L)

        Z = self.encoder(segments)
        # Z: (B, N, d)

        A = self.projection(Z)
        # A: (B, N, K)

        with torch.no_grad():
            flat_segments = segments.reshape(-1, segments.shape[-1])
            C_flat = self.scorer(flat_segments)
            C = C_flat.view(B, segments.shape[1], self.config.K)

        h, eta = self.pool(A)

        if self.config.probabilistic:
            mu, sigma = self.decoder(h)
            y_hat = mu
            # Attach sigma for downstream use if needed
            self.last_sigma = sigma
        else:
            y_hat = self.decoder(h)
            self.last_sigma = None

        return TCRPOutput(y_hat=y_hat, h=h, A=A, C=C, eta=eta)
