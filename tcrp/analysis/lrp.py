"""Layer-wise relevance propagation utilities for TCRP."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv1d, Linear


def lrp_linear_eps(layer: Linear, a_in: Tensor, R_out: Tensor, eps: float = 1e-6) -> Tensor:
    """LRP-ε for a fully connected linear layer."""
    if a_in.dim() != 2 or R_out.dim() != 2:
        raise ValueError("a_in and R_out must be 2D tensors")

    w = layer.weight  # shape (out, in)
    z = a_in @ w.T
    z_eps = z + eps * torch.sign(z)
    s = R_out / z_eps
    c = s @ w
    R_in = a_in * c
    return R_in


def lrp_gamma_conv(
    layer: Conv1d,
    a_in: Tensor,
    R_out: Tensor,
    gamma: float = 0.25,
    eps: float = 1e-6,
) -> Tensor:
    """LRP-γ for a causally-padded Conv1d (paper Appendix C, Eq. 13 variant).

    Algorithm:
        1. w_γ = w + γ · clamp(w, min=0)
        2. Recompute forward z = conv(pad(a_in), w_γ)   [bias excluded]
        3. Stabilise: z_s = z + ε · sign(z)
        4. s = R_out / z_s
        5. c_padded = conv_transpose(s, w_γ)   [adjoint of step 2]
        6. Strip causal padding: c = c_padded[..., pad:]
        7. R_in = a_in * c

    Args:
        layer: Conv1d with padding==0 (causal padding is applied externally).
        a_in:  Pre-padding input activations, shape (B, C_in, L).
        R_out: Output relevance, shape (B, C_out, L_out).
        gamma: LRP-γ positive-weight boost (default 0.25).
        eps:   Denominator stabiliser (default 1e-6).

    Returns:
        R_in: Input relevance, shape (B, C_in, L).
    """
    if a_in.dim() != 3 or R_out.dim() != 3:
        raise ValueError("a_in and R_out must be 3D tensors")

    if layer.padding != (0,):
        raise ValueError(
            "lrp_gamma_conv expects layer.padding==0; causal padding must be applied externally"
        )

    kernel_size = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
    dilation    = layer.dilation[0]    if isinstance(layer.dilation,    tuple) else layer.dilation
    stride      = layer.stride[0]      if isinstance(layer.stride,      tuple) else layer.stride
    pad = (kernel_size - 1) * dilation

    # Step 1 — gamma-modified weights
    w = layer.weight                          # (C_out, C_in, K)
    w_gamma = w + gamma * w.clamp(min=0)

    # Step 2 — forward with gamma weights (no bias; bias excluded from LRP formula)
    a_padded = F.pad(a_in, (pad, 0)) if pad > 0 else a_in
    z = F.conv1d(a_padded, w_gamma, bias=None, dilation=dilation, stride=stride, padding=0)

    # Step 3 — stabilise denominator (LRP-ε)
    z_sign = z.sign()
    z_sign[z_sign == 0] = 1           # treat exact zero as positive
    z_s = z + eps * z_sign

    # Step 4 — scale by incoming relevance
    s = R_out / z_s                           # (B, C_out, L_out)

    # Step 5 — adjoint (transposed conv) to spread relevance back to inputs
    c_padded = F.conv_transpose1d(s, w_gamma, bias=None, dilation=dilation, stride=stride, padding=0)
    # shape: (B, C_in, L + pad)

    # Step 6 — strip causal zero-padding on the left
    c = c_padded[..., pad:] if pad > 0 else c_padded   # (B, C_in, L)

    # Step 7 — element-wise product with input activations
    return a_in * c


def lrp_relu(a_in: Tensor, R_out: Tensor) -> Tensor:
    """LRP pass-through for ReLU layers."""
    if a_in.shape != R_out.shape:
        raise ValueError("a_in and R_out must have the same shape")
    return R_out


def lrp_mean_pool(a_in: Tensor, R_out: Tensor) -> Tensor:
    """Distribute relevance equally across the temporal dimension after mean pooling."""
    if a_in.dim() != 3 or R_out.dim() != 2:
        raise ValueError("a_in must be 3D and R_out must be 2D")

    B, d, T = a_in.shape
    if R_out.shape != (B, d):
        raise ValueError(f"Expected R_out shape {(B, d)}, got {tuple(R_out.shape)}")

    return R_out.unsqueeze(-1).expand(-1, -1, T) / float(T)
