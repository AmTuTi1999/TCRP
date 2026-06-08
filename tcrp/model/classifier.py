"""TCRPClassifier — TCRP adapted for time-series classification.

Architecture delta from TCRPForecaster (tasks_classification.md TC-01):
  - Decoder: nn.Linear(K, C) outputting C class logits, not H horizon values
  - Loss: F.cross_entropy applied externally (in trainer)
  - Relevance init uses predicted class logit for k_star, not forecast horizon
All other components (segmentation, TCN encoder, bottleneck, concept scorer,
attention pool) are identical to the forecaster.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

from tcrp.concepts import ConceptScorer
from tcrp.model.tcrp_forecaster.components.aggregation import ConceptAttentionPool
from tcrp.model.tcrp_forecaster.components.bottleneck import ConceptProjection
from tcrp.model.tcrp_forecaster.components.encoder import TCNEncoder
from tcrp.model.tcrp_forecaster.components.segmentation import Segmenter
from tcrp.model.utils.types import TCRPOutput


@dataclass
class TCRPClassConfig:
    """Configuration for TCRPClassifier.

    Mirrors TCRPConfig fields but replaces the forecast horizon H with the
    number of output classes C. K is derived automatically from k_max and periods.
    """

    C: int
    L: int = 20
    stride: int = 5
    K: int = 0
    periods: list = field(default_factory=list)
    k_max: int = 2
    alpha: float = 5.0
    beta: float = 5.0
    gamma: float = 0.5
    gamma_j: float = 2.0
    jump_threshold: float = 3.0
    train_std: float = 1.0
    lambda1: float = 0.1
    lambda2: float = 1e-4
    lambda3: float = 0.01
    lambda4: float = 0.1
    adversarial: bool = False
    alpha_max: float = 1.0
    warmup_epochs: int = 20
    encoder_in_ch: int = 1
    encoder_hidden: int = 64
    tcn_encoder_n_layers: int = 4
    tcn_encoder_kernel_size: int = 3
    tcn_encoder_use_weight_norm: bool = True
    attention_hidden: int = 32
    attention_temp: float = 1.0

    def __post_init__(self):
        """Derive K from k_max and the number of period concepts."""
        self.K = 16 + self.k_max + len(self.periods)

    @property
    def d(self) -> int:
        """Encoder output dimension (alias for encoder_hidden, used by TCRPAnalyser)."""
        return self.encoder_hidden


class ClassifierDecoder(nn.Module):
    """Linear classification decoder: logits = linear(h) where h ∈ R^K."""

    def __init__(self, K: int, C: int) -> None:
        """Initialise decoder with K input concepts and C output classes."""
        super().__init__()
        self.K = K
        self.C = C
        self.linear = nn.Linear(K, C, bias=True)
        self.Theta = self.linear.weight

    def forward(self, h: Tensor) -> Tensor:
        """Map pooled concept vector (B, K) → class logits (B, C)."""
        return self.linear(h)


class TCRPClassifier(nn.Module):
    """TCRP adapted for time-series classification.

    Identical pipeline to TCRPForecaster except the decoder maps K concepts
    to C class logits, and the loss (cross-entropy) is applied externally.

    The conservation theorem still holds: for the predicted class k_star,
    sum_t R_x_t = f_{k_star}(x) = predicted class logit.
    """

    def __init__(self, config: TCRPClassConfig) -> None:
        """Initialise classifier from a TCRPClassConfig."""
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
        self.pool = ConceptAttentionPool(
            K=config.K, hidden=config.attention_hidden, temp=config.attention_temp
        )
        self.decoder = ClassifierDecoder(K=config.K, C=config.C)

    def forward(self, x: Tensor) -> TCRPOutput:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T).

        Returns:
            TCRPOutput where y_hat is (B, C) class logits (not softmax probabilities).
        """
        if x.dim() != 2:
            raise ValueError(f"Expected (B, T), got {tuple(x.shape)}")

        B, _ = x.shape
        segments = self.segmenter(x)  # (B, N, L)
        Z = self.encoder(segments)  # (B, N, d)
        A = self.projection(Z)  # (B, N, K)

        with torch.no_grad():
            flat = segments.reshape(-1, segments.shape[-1])
            C_flat = self.scorer(flat)
            C = C_flat.view(B, segments.shape[1], self.config.K)

        h, eta = self.pool(A)  # h: (B, K), eta: (B, N, K)
        y_hat = self.decoder(h)  # (B, C) logits

        return TCRPOutput(y_hat=y_hat, h=h, A=A, C=C, eta=eta)

    def predict(self, x: Tensor) -> Tensor:
        """Return predicted class indices (B,)."""
        with torch.no_grad():
            out = self.forward(x)
        return out.y_hat.argmax(dim=-1)
