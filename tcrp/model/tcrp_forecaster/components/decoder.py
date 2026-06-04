"""Phase 6 · Horizon Decoder.

Implements linear horizon decoder and Gaussian probabilistic decoder.
"""

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class BaseDecoder(nn.Module, ABC):
    """Abstract base class for all TCRP horizon decoders.

    All decoders map a pooled concept vector `h (B, K)` to a forecast.
    """

    @abstractmethod
    def forward(self, h: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Map pooled concepts (B, K) → forecast (B, H) or (mu, sigma)."""


class HorizonDecoder(BaseDecoder):
    """Linear horizon decoder: y_hat = Theta @ h (no non-linearity)."""

    def __init__(self, K: int, H: int):
        """Initialize HorizonDecoder with concept dim K and forecast horizon H."""
        super().__init__()
        self.K = K
        self.H = H
        self.linear = nn.Linear(K, H, bias=True)

        # Expose decoder weights as Theta for analysis
        self.Theta = self.linear.weight

    def forward(self, h: Tensor) -> Tensor:
        """Map pooled concept vector (B, K) to forecast (B, H)."""
        if h.dim() != 2:
            raise ValueError(f"Expected h shape (B, K), got {tuple(h.shape)}")
        if h.shape[1] != self.K:
            raise ValueError(f"Expected K={self.K}, got {h.shape[1]}")
        return self.linear(h)


class GaussianDecoder(BaseDecoder):
    """Probabilistic Gaussian decoder with mu and log-sigma heads."""

    def __init__(self, K: int, H: int):
        """Initialize GaussianDecoder with concept dim K and forecast horizon H."""
        super().__init__()
        self.K = K
        self.H = H
        self.mu_head = nn.Linear(K, H, bias=True)
        self.log_sigma_head = nn.Linear(K, H, bias=True)

        # Expose mu_head weights for analysis
        self.Theta = self.mu_head.weight

    def forward(self, h: Tensor) -> tuple[Tensor, Tensor]:
        """Map pooled concept vector (B, K) to (mu, sigma) forecast distributions."""
        if h.dim() != 2:
            raise ValueError(f"Expected h shape (B, K), got {tuple(h.shape)}")
        if h.shape[1] != self.K:
            raise ValueError(f"Expected K={self.K}, got {h.shape[1]}")

        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)
        sigma = torch.exp(log_sigma).clamp(min=1e-6)
        return mu, sigma

    @staticmethod
    def nll_loss(mu: Tensor, sigma: Tensor, y_true: Tensor) -> Tensor:
        """Negative log-likelihood under independent Gaussian outputs."""
        if mu.shape != y_true.shape:
            raise ValueError("mu and y_true must have the same shape")
        if sigma.shape != y_true.shape:
            raise ValueError("sigma and y_true must have the same shape")

        var = sigma * sigma
        log_term = 0.5 * torch.log(2 * math.pi * var)
        sq_term = 0.5 * ((y_true - mu) ** 2) / var
        nll = log_term + sq_term
        return nll.mean()
