"""
Phase 4 · Concept Projection Bottleneck

Implements ConceptProjection and alignment loss per T-08 and T-09.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConceptProjection(nn.Module):
    """Linear concept projection: projects d -> K via a learnable linear map.

    Weight matrix rows correspond to concept directions w_k.
    """

    def __init__(self, d: int, K: int):
        super().__init__()
        self.d = d
        self.K = K
        self.linear = nn.Linear(d, K, bias=True)

    def init_from_pca(self, Z_calib: Tensor) -> None:
        """Initialize rows of the linear weight to top-K PCA directions.

        Z_calib: (B, N, d) calibration batch.
        """
        if Z_calib.dim() != 3:
            raise ValueError("Z_calib must have shape (B, N, d)")

        X = Z_calib.reshape(-1, self.d).to(dtype=self.linear.weight.dtype, device=self.linear.weight.device)

        # Center
        mean = X.mean(dim=0, keepdim=True)
        Xc = X - mean

        # If there are fewer samples than dimensions, SVD still works but handle small sample cases
        if Xc.numel() == 0:
            raise ValueError("Empty calibration data for PCA initialization")

        # Compute SVD on centered data: Xc = U S Vh  => principal components are rows of Vh
        # Use torch.linalg.svd for stability
        try:
            U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        except RuntimeError:
            # Fallback to cpu SVD if GPU SVD fails
            U, S, Vh = torch.linalg.svd(Xc.cpu(), full_matrices=False)
            Vh = Vh.to(self.linear.weight.device)

        comps = Vh[: min(self.K, Vh.shape[0]), :]

        # If fewer components than K, pad with zeros
        if comps.shape[0] < self.K:
            pad = torch.zeros(self.K - comps.shape[0], self.d, device=comps.device, dtype=comps.dtype)
            comps = torch.cat([comps, pad], dim=0)

        with torch.no_grad():
            # Linear.weight has shape (K, d)
            self.linear.weight.copy_(comps)
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, Z: Tensor) -> Tensor:
        """Project input Z (B, N, d) to activations A (B, N, K).

        Returns A only — downstream layers should use A instead of Z.
        """
        if Z.dim() != 3:
            raise ValueError(f"Expected Z shape (B, N, d), got {tuple(Z.shape)}")

        B, N, d = Z.shape
        if d != self.d:
            raise ValueError(f"Expected last dim d={self.d}, got {d}")

        Z_flat = Z.reshape(B * N, d)
        A_flat = self.linear(Z_flat)
        A = A_flat.view(B, N, self.K)
        return A


def alignment_loss(A: Tensor, C: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute alignment loss: sum_k (1 - corr_k)^2 where corr_k is Pearson corr.

    A: (B, N, K) learned activations
    C: (B, N, K) analytic scores

    Degenerate concepts in C (zero variance) are skipped.
    """
    if A.shape != C.shape:
        raise ValueError("A and C must have the same shape")

    B, N, K = A.shape
    loss = torch.tensor(0.0, device=A.device, dtype=A.dtype)
    valid = 0

    # Flatten over B,N for each concept
    for k in range(K):
        a = A[:, :, k].reshape(-1)
        c = C[:, :, k].reshape(-1)

        c_mean = c.mean()
        c_cent = c - c_mean
        c_std = torch.sqrt(torch.mean(c_cent * c_cent))

        # Skip degenerate C with zero variance
        if c_std.item() == 0.0:
            continue

        a_mean = a.mean()
        a_cent = a - a_mean
        a_std = torch.sqrt(torch.mean(a_cent * a_cent))

        denom = (a_std * c_std).clamp(min=eps)
        corr = torch.mean(a_cent * c_cent) / denom

        term = (1.0 - corr) ** 2
        loss = loss + term
        valid += 1

    if valid == 0:
        return torch.tensor(0.0, device=A.device, dtype=A.dtype)

    return loss


def stability_loss(A: Tensor, C: Tensor) -> Tensor:
    """Penalise deviation between consecutive-segment deltas of A and C.

    Encourages concept activations to change in step with analytic scores
    across the segment sequence.

    A, C: (B, N, K)
    """
    if A.shape[1] < 2:
        return A.new_zeros(())
    dA = A[:, 1:, :] - A[:, :-1, :]  # (B, N-1, K)
    dC = C[:, 1:, :] - C[:, :-1, :]  # (B, N-1, K)
    return F.mse_loss(dA, dC)


def _unit_tests() -> None:
    # Basic unit tests
    d = 8
    K = 3
    B = 2
    N = 4

    Z = torch.randn(B, N, d)
    proj = ConceptProjection(d, K)
    proj.init_from_pca(Z)

    A = proj(Z)
    assert A.shape == (B, N, K)

    # alignment loss zero when A == C
    loss = alignment_loss(A, A)
    if not torch.isclose(loss, torch.tensor(0.0, device=loss.device), atol=1e-6):
        raise AssertionError(f"alignment_loss(A,A) != 0, got {loss}")

    print("bottleneck unit tests passed")


if __name__ == "__main__":
    _unit_tests()
