"""Phase 5 · Temporal Concept Aggregation.

Implements additive attention pooling per Eq.10.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConceptAttentionPool(nn.Module):
    """Additive attention pooling over segments for concept activations.

    Args:
        K: number of concepts (dim of A_n)
        hidden: hidden dimension for additive attention
        temp: temperature scaling applied before softmax (divide by temp)
    """

    def __init__(self, K: int, hidden: int = 32, temp: float = 1.0):
        """Initialize ConceptAttentionPool with concept dimension, hidden size, and temperature."""
        super().__init__()
        self.K = K
        self.hidden = hidden
        self.temp = float(temp)

        # U: maps K -> hidden (implements U @ A_n)
        self.U = nn.Linear(K, hidden, bias=False)
        # v: (hidden,)
        self.v = nn.Parameter(torch.randn(hidden))

        # last attention weights for analysis
        self.last_eta: Tensor | None = None

    def forward(self, A: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            A: Tensor of shape (B, N, K)

        Returns:
            h: pooled vector (B, K)
            eta: attention weights (B, N)
        """
        if A.dim() != 3:
            raise ValueError(f"Expected A shape (B, N, K), got {tuple(A.shape)}")

        B, N, K = A.shape
        if K != self.K:
            raise ValueError(f"Expected K={self.K}, got {K}")

        # U @ A_n -> shape (B, N, hidden)
        Ux = torch.tanh(self.U(A))

        # e_n = v^T tanh(U @ A_n) -> sum over hidden
        e = (Ux * self.v).sum(dim=-1)  # (B, N)

        # temperature scaling then softmax over N
        logits = e / self.temp
        eta = F.softmax(logits, dim=1)  # (B, N)

        # store last eta for analysis (detach to avoid holding graph)
        self.last_eta = eta.detach()

        # pooled vector h = sum_n eta_n * A_n  -> (B, K)
        h = (eta.unsqueeze(-1) * A).sum(dim=1)

        return h, eta
