"""T-05 · Sliding Window Segmentation.

Paper: Eq. 7 (background notation)

Efficient segmentation of time series into overlapping windows using unfold.
"""

import torch
import torch.nn as nn
from torch import Tensor


class Segmenter(nn.Module):
    """Slides a window of fixed length over a time series, producing overlapping segments.

    This module uses PyTorch's efficient `unfold` operation, which does not copy data.
    """

    def __init__(self, L: int, stride: int):
        """Initialize Segmenter.

        Args:
            L: Segment/window length in timesteps
            stride: Step size between segment starts

        Raises:
            ValueError: If L <= 0 or stride <= 0
        """
        super().__init__()

        if L <= 0:
            raise ValueError(f"Segment length L must be positive, got {L}")
        if stride <= 0:
            raise ValueError(f"Stride must be positive, got {stride}")

        self.L = L
        self.stride = stride

        # These will be set in forward() when we know T
        self._start_indices: Tensor | None = None
        self._overlap_counts: Tensor | None = None
        self._T_cached: int | None = None

    @property
    def start_indices(self) -> Tensor | None:
        """Return start indices of all segments.

        Returns:
            Tensor of shape (N,) where N = floor((T - L) / stride) + 1
            Element n contains t_n = n * stride (0-indexed start position of segment n)
        """
        return self._start_indices

    @property
    def overlap_counts(self) -> Tensor | None:
        """Return overlap counts for each timestep.

        Returns:
            Tensor of shape (T,) where element t indicates how many segments
            contain timestep t. Values range from 1 (edges) to higher (center).
        """
        return self._overlap_counts

    def forward(self, x: Tensor) -> Tensor:
        """Segment univariate time series into overlapping windows.

        Args:
            x: Input tensor of shape (B, T) where:
               - B: batch size
               - T: time series length

        Returns:
            Segmented tensor of shape (B, N, L) where:
            - N: number of segments = floor((T - L) / stride) + 1
            - L: segment length

        Raises:
            ValueError: If T < L (not enough data to extract a single segment)

        Computation Notes:
            Uses `unfold` for efficient, zero-copy windowing.
            The unfold operation applies to the time dimension (dimension 1).
        """
        if x.dim() != 2:
            raise ValueError(f"Expected input shape (B, T), got {tuple(x.shape)}")

        B, T = x.shape

        # Validate that we have enough data
        if T < self.L:
            raise ValueError(
                f"Time series length T={T} must be >= segment length L={self.L}"
            )

        # Use unfold to extract sliding windows
        # unfold(dimension, size, step) -> applies windowing to the specified dimension
        # Input: (B, T) -> Output: (B, L, N)
        # We need to rearrange to get (B, N, L)
        segments = x.unfold(dimension=1, size=self.L, step=self.stride)
        # Shape after unfold: (B, N, L)

        # Compute and cache metadata
        N = segments.shape[1]

        # Compute start indices: t_n = n * stride for n in range(N)
        start_indices = torch.arange(N, dtype=torch.long, device=x.device) * self.stride
        self._start_indices = start_indices

        # Compute overlap counts: for each timestep t, count how many segments contain it
        overlap_counts = torch.zeros(T, dtype=torch.long, device=x.device)
        for n in range(N):
            start = n * self.stride
            end = start + self.L
            overlap_counts[start:end] += 1

        self._overlap_counts = overlap_counts.float()
        self._T_cached = T

        return segments

    def get_segment_info(self) -> dict:
        """Return metadata about the most recent segmentation.

        Returns:
            Dictionary with keys:
            - 'L': segment length
            - 'stride': stride
            - 'N': number of segments (if forward was called)
            - 'T': time series length (if forward was called)
            - 'start_indices': tensor of segment start positions
            - 'overlap_counts': tensor of overlap counts per timestep
        """
        return {
            "L": self.L,
            "stride": self.stride,
            "N": (
                self._start_indices.shape[0]
                if self._start_indices is not None
                else None
            ),
            "T": self._T_cached,
            "start_indices": self._start_indices,
            "overlap_counts": self._overlap_counts,
        }
