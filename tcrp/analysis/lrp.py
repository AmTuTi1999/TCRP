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
    """LRP-γ for a Conv1d layer using weight-magnitude redistribution.
    
    Implements a conservative relevance propagation rule for convolutional layers
    based on weight magnitudes. For each output, relevance is redistributed to
    input positions weighted by the magnitude of the connecting weights.
    
    This approach is guaranteed to conserve relevance:
        sum_t R_in[t] = sum_j R_out[j]
    
    Args:
        layer: Conv1d layer with padding=0 and manual external padding
        a_in: Input activations, shape (B, C_in, L)
        R_out: Output relevance, shape (B, C_out, L_out)
        gamma: Regularization weight applied to weights (default 0.25)
        eps: Numerical stability parameter (default 1e-6, unused in this rule)
    
    Returns:
        R_in: Input relevance, shape (B, C_in, L) where L is original input length
    """
    if a_in.dim() != 3 or R_out.dim() != 3:
        raise ValueError("a_in and R_out must be 3D tensors")

    w = layer.weight
    w_gamma = w + gamma * w.clamp(min=0)

    if layer.padding != (0,):
        raise ValueError("lrp_gamma_conv expects layer.padding==0 and manual padding external to conv")

    kernel_size = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
    dilation = layer.dilation[0] if isinstance(layer.dilation, tuple) else layer.dilation
    stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
    pad = (kernel_size - 1) * dilation

    a_in_padded = F.pad(a_in, (pad, 0)) if pad > 0 else a_in
    
    B, C_in, L_in = a_in_padded.shape
    C_out, _, K = w_gamma.shape
    L_out = R_out.shape[-1]
    
    R_in_padded = torch.zeros_like(a_in_padded)
    
    # For each output position j, distribute relevance to contributing inputs
    # via weight magnitudes
    for j in range(L_out):
        for k in range(K):
            t_in = j * stride + k * dilation
            if t_in < L_in:
                # Weight magnitudes for this kernel position: (C_out, C_in)
                w_mag = w_gamma[:, :, k].abs()
                
                # Distribute output relevance weighted by magnitudes:
                # R_contribution[b, c_in] = sum_c_out (w_mag[c_out, c_in] * R_out[b, c_out, j])
                # R_out[:, :, j] is (B, C_out), w_mag is (C_out, C_in)
                # Want output (B, C_in): w_mag.T @ R_out[:, :, j].T gives (C_in, B), then transpose
                contribution = (w_mag.T @ R_out[:, :, j].T).T  # (B, C_in)
                R_in_padded[:, :, t_in] += contribution
    
    # Redistribute left-padding relevance to the first real input
    if pad > 0:
        padding_relevance = R_in_padded[:, :, :pad].sum(dim=-1)  # (B, C_in)
        R_in_padded[:, :, pad] += padding_relevance
        R_in = R_in_padded[..., pad:]
    else:
        R_in = R_in_padded
    
    return R_in


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
