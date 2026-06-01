"""
T-06 · Dilated Causal TCN Block
Paper: Appendix B (encoder description)

Causal convolution blocks with dilated kernels for temporal feature extraction.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm
import warnings


class CausalDilatedBlock(nn.Module):
    """
    Dilated causal convolutional block with residual connection.
    
    Architecture:
        Conv1d → WeightNorm → ReLU → Conv1d → WeightNorm → ReLU
        with residual skip connection (optional 1×1 projection if in_ch != out_ch)
    
    Causality is enforced through left-only padding, ensuring that output at
    timestep t depends only on inputs at timesteps 0 to t (not t+1, t+2, ...).
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        use_weight_norm: bool = True,
    ):
        """
        Initialize CausalDilatedBlock.
        
        Args:
            in_channels: Number of input feature channels
            out_channels: Number of output feature channels
            kernel_size: Convolutional kernel size (must be > 1)
            dilation: Dilation factor for receptive field expansion (default: 1)
            use_weight_norm: Whether to apply weight normalization (default: True)
            
        Raises:
            ValueError: If kernel_size < 1 or dilation < 1
        """
        super().__init__()
        
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        if dilation < 1:
            raise ValueError(f"dilation must be >= 1, got {dilation}")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Compute causal padding: amount to pad on left to make output same-length as input
        # For a Conv1d with kernel_size k and dilation d:
        # receptive_field_size = (k - 1) * d + 1
        # To maintain causality, we pad left by (k - 1) * d
        self.causal_pad = (kernel_size - 1) * dilation
        
        # First conv layer: in_channels -> out_channels
        conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # We apply padding manually in forward
            bias=True,
        )
        self.conv1 = weight_norm(conv1) if use_weight_norm else conv1
        
        # Second conv layer: out_channels -> out_channels
        conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # We apply padding manually in forward
            bias=True,
        )
        self.conv2 = weight_norm(conv2) if use_weight_norm else conv2
        
        # Activation functions
        self.relu = nn.ReLU(inplace=False)
        
        # Residual projection: 1x1 conv if channels don't match
        if in_channels != out_channels:
            proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)
            self.residual_proj = weight_norm(proj) if use_weight_norm else proj
        else:
            self.residual_proj = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply causal dilated convolution with residual connection.
        
        Args:
            x: Input tensor of shape (B, C_in, T) where:
               - B: batch size
               - C_in: number of input channels (must equal in_channels)
               - T: sequence length
        
        Returns:
            Output tensor of shape (B, C_out, T) where:
            - C_out: number of output channels (out_channels)
            - T: same as input (causality preserves length)
            
        Causality Guarantee:
            output[:, :, t] depends only on input[:, :, 0:t+1]
            (not on input[:, :, t+1:])
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected input shape (B, C, T), got {tuple(x.shape)}"
            )
        
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        
        # Save input for residual connection
        residual = x
        
        # First conv block: Conv → ReLU
        # Apply left-only causal padding
        x = F.pad(x, (self.causal_pad, 0))  # pad left only
        x = self.conv1(x)
        x = self.relu(x)
        
        # Second conv block: Conv → ReLU
        x = F.pad(x, (self.causal_pad, 0))  # pad left only
        x = self.conv2(x)
        x = self.relu(x)
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        # Element-wise addition
        output = x + residual
        
        return output
    
    def compute_receptive_field(self) -> int:
        """
        Compute the receptive field size of this block.
        
        Returns:
            Receptive field size = (kernel_size - 1) * dilation + 1
            This indicates the number of input timesteps that influence one output timestep.
        """
        return (self.kernel_size - 1) * self.dilation + 1
    
    def extra_repr(self) -> str:
        """Return string representation of module parameters."""
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"dilation={self.dilation}, "
            f"causal_pad={self.causal_pad}, "
            f"receptive_field={self.compute_receptive_field()}"
        )


class TCNEncoder(nn.Module):
    """
    Stacked TCN encoder built from `CausalDilatedBlock`.

    Args:
        in_ch: number of input channels (must be 1 for input shaped (B, N, L))
        hidden: hidden feature dimension (d)
        n_layers: number of stacked causal blocks
        kernel_size: convolutional kernel size
        use_weight_norm: whether to apply weight normalization to convs

    Forward input: `segments` shaped (B, N, L). Output shape: (B, N, d).
    """

    def __init__(
        self,
        in_ch: int = 1,
        hidden: int = 64,
        n_layers: int = 4,
        kernel_size: int = 3,
        use_weight_norm: bool = True,
    ):
        super().__init__()

        self.in_ch = in_ch
        self.hidden = hidden
        self.n_layers = n_layers
        self.kernel_size = kernel_size

        # Dilations: [1, 2, 4, 8, ...]
        self.dilations = [2 ** i for i in range(n_layers)]

        # Build stacked causal blocks
        layers = []
        curr_in = in_ch
        for d in self.dilations:
            layers.append(
                CausalDilatedBlock(
                    curr_in, hidden, kernel_size=kernel_size, dilation=d, use_weight_norm=use_weight_norm
                )
            )
            curr_in = hidden

        self.tcn = nn.Sequential(*layers)

        # The total left-padding (sum of each block's causal_pad)
        # For kernel_size=3 and dilations [1,2,4,8]: total_causal_pad = (k-1)*sum(dilations) = 30
        self.total_causal_pad = (kernel_size - 1) * sum(self.dilations)

    def compute_receptive_field(self) -> int:
        """Return the effective receptive field measured in timesteps (without +1).

        Note: this follows the convention used elsewhere in this file where we
        consider the total left-pad required: (k-1)*sum(dilations).
        """
        return self.total_causal_pad

    def forward(self, segments: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            segments: Tensor of shape (B, N, L)

        Returns:
            Tensor of shape (B, N, d)
        """
        if segments.dim() != 3:
            raise ValueError(f"Expected input shape (B, N, L), got {tuple(segments.shape)}")

        B, N, L = segments.shape

        if self.in_ch != 1:
            raise NotImplementedError("TCNEncoder currently supports in_ch==1 for input shaped (B, N, L)")

        if L < self.total_causal_pad:
            warnings.warn(
                f"Input length L={L} is smaller than the stacked receptive field requirement "
                f"{self.total_causal_pad}. Results may be unreliable.",
                UserWarning,
            )

        # Flatten (B, N) into batch dimension so weights are shared across segments
        x = segments.reshape(B * N, self.in_ch, L)

        out = self.tcn(x)  # (B*N, hidden, L)

        # Mean-pool over time dimension
        z = out.mean(dim=-1)  # (B*N, hidden)

        # Restore (B, N, d)
        z = z.view(B, N, self.hidden)

        return z


def verify_causality(
    block: CausalDilatedBlock,
    batch_size: int = 2,
    seq_length: int = 100,
    num_tests: int = 10,
) -> bool:
    """
    Verify that a CausalDilatedBlock respects causality.
    
    This function creates input sequences and checks that modifying future timesteps
    does not affect past outputs.
    
    Args:
        block: CausalDilatedBlock instance to test
        batch_size: Batch size for test inputs
        seq_length: Sequence length for test inputs
        num_tests: Number of random modifications to test
        
    Returns:
        True if causality is verified, False otherwise
    """
    block.eval()
    device = next(block.parameters()).device
    
    with torch.no_grad():
        # Create a base input
        x = torch.randn(batch_size, block.in_channels, seq_length, device=device)
        
        # Get baseline output
        y_baseline = block(x)
        
        # Test: modifying future timesteps shouldn't affect past outputs
        for test_idx in range(num_tests):
            # Create a modified input where future timesteps are perturbed
            x_modified = x.clone()
            
            # Randomly choose a past timestep to check
            check_t = torch.randint(0, seq_length - 1, (1,)).item()
            
            # Perturb all future timesteps
            if check_t < seq_length - 1:
                x_modified[:, :, check_t + 1:] += torch.randn_like(x_modified[:, :, check_t + 1:]) * 0.1
            
            # Get modified output
            y_modified = block(x_modified)
            
            # Check if output at check_t is unchanged (up to numerical precision)
            max_diff = (y_baseline[:, :, :check_t + 1] - y_modified[:, :, :check_t + 1]).abs().max()
            
            if max_diff > 1e-5:
                print(f"Causality violation at test {test_idx}, timestep {check_t}: max_diff={max_diff}")
                return False
    
    return True
