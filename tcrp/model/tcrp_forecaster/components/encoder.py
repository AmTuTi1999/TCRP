"""
 Dilated Causal TCN Block

Causal convolution blocks with dilated kernels for temporal feature extraction.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm
import warnings



class BaseEncoder(nn.Module, ABC):
    """Abstract base class for all TCRP segment encoders.

    All concrete encoders map `(B, N, L)` segments to `(B, N, d)` representations,
    sharing weights across the N segment axis.
    """

    @abstractmethod
    def forward(self, segments: Tensor) -> Tensor:
        """Map segments (B, N, L) → latent representations (B, N, d)."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Encoder output dimension *d*."""


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
    
        self.relu = nn.ReLU(inplace=False)
    
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
        
        residual = x
        x = F.pad(x, (self.causal_pad, 0))  # pad left only
        x = self.conv1(x)
        x = self.relu(x)
        x = F.pad(x, (self.causal_pad, 0))  # pad left only
        x = self.conv2(x)
        x = self.relu(x)
        
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
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


class TCNEncoder(BaseEncoder):
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
        self.dilations = [2 ** i for i in range(n_layers)]

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

    @property
    def output_dim(self) -> int:
        return self.hidden

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

        x = segments.reshape(B * N, self.in_ch, L)
        out = self.tcn(x)  # (B*N, hidden, L)
        z = out.mean(dim=-1)  # (B*N, hidden)
        z = z.view(B, N, self.hidden)
        return z


class LSTMEncoder(BaseEncoder):
    """
    LSTM encoder for temporal segment encoding.

    Shares weights across all N segments by processing a flattened (B*N, L, 1)
    tensor, matching the TCNEncoder interface exactly.

    Args:
        in_ch: input channels (must be 1 for segments shaped (B, N, L))
        hidden: LSTM hidden state dimension (d)
        n_layers: number of stacked LSTM layers
        bidirectional: if True, use a bidirectional LSTM and project back to hidden
        dropout: dropout probability between LSTM layers (ignored when n_layers==1)
        pooling: 'last' uses the final hidden state; 'mean' mean-pools over timesteps

    Forward input: segments (B, N, L). Output: (B, N, d).
    """

    def __init__(
        self,
        in_ch: int = 1,
        hidden: int = 64,
        n_layers: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.0,
        pooling: str = "last",
    ):
        super().__init__()

        if pooling not in ("last", "mean"):
            raise ValueError(f"pooling must be 'last' or 'mean', got '{pooling}'")

        self.in_ch = in_ch
        self.hidden = hidden
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.pooling = pooling

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Project 2*hidden → hidden when bidirectional so output dim is always `hidden`
        self.proj = nn.Linear(2 * hidden, hidden, bias=False) if bidirectional else None

    @property
    def output_dim(self) -> int:
        return self.hidden


    def forward(self, segments: Tensor) -> Tensor:
        """
        Args:
            segments: (B, N, L)

        Returns:
            Tensor of shape (B, N, d)
        """
        if segments.dim() != 3:
            raise ValueError(f"Expected input shape (B, N, L), got {tuple(segments.shape)}")

        B, N, L = segments.shape

        # Treat each timestep as a scalar input; share weights across segments
        x = segments.reshape(B * N, L, 1)          # (B*N, L, 1)

        out, (h_n, _) = self.lstm(x)               # out: (B*N, L, D), h_n: (layers*dirs, B*N, hidden)

        if self.pooling == "last":
            if self.bidirectional:
                # h_n[-2]: last forward layer; h_n[-1]: last backward layer
                z = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B*N, 2*hidden)
            else:
                z = h_n[-1]                        # (B*N, hidden)
        else:  # mean
            z = out.mean(dim=1)                    # (B*N, D)

        if self.proj is not None:
            z = self.proj(z)                       # (B*N, hidden)

        return z.view(B, N, self.hidden)
