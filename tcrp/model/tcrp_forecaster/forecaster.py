"""
TCRP Forecaster Assembly

Assembles segmentation, encoding, concept projection, scoring, attention, and decoding.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .components.segmentation import Segmenter
from .components.encoder import TCNEncoder
from .components.bottleneck import ConceptProjection
from .components.aggregation import ConceptAttentionPool
from .components.decoder import HorizonDecoder, GaussianDecoder
from tcrp.concepts import ConceptScorer
from ..utils.types import TCRPOutput, TCRPConfig


class TCRPForecaster(nn.Module):
    """TCRP forecaster combining all TCRP components."""

    def __init__(self, config: TCRPConfig):
        super().__init__()
        self.config = config

        self.segmenter = Segmenter(L=config.L, stride=config.stride)
        self.encoder = TCNEncoder(
            in_ch=config.encoder_in_ch,
            hidden=config.encoder_hidden,
            n_layers=config.tcn_encoder_n_layers,
            kernel_size=config.tcn_encoder_kernel_size,
            use_weight_norm=config.tcn_encoder_use_weight_norm,
        )
        self.projection = ConceptProjection(d=config.encoder_hidden, K=config.K)
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
        self.pool = ConceptAttentionPool(K=config.K, hidden=config.attention_hidden, temp=config.attention_temp)
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

        B, _ = x.shape
        segments = self.segmenter(x) # segments: (B, N, L)
        Z = self.encoder(segments) # Z: (B, N, d)
        A = self.projection(Z) # A: (B, N, K)

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
