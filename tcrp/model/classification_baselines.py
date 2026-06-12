"""Baseline classification models for comparison with TCRPClassifier.

All models share the same interface:
  forward(x: Tensor) -> ClassificationOutput   where x is (B, T)

ClassificationOutput.y_hat has shape (B, C), compatible with evaluate_all().

Models  (Wang et al. 2017 Time Series Classification from Scratch)
------
MLPClassifier    — Three-layer MLP with dropout (MLP baseline)
LSTMClassifier   — Bidirectional LSTM + linear head
FCNClassifier    — Fully Convolutional Network (strong non-parametric baseline)
ResNetClassifier — 1D ResNet (state-of-the-art among non-attention baselines)
"""

from __future__ import annotations

from typing import NamedTuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ClassificationOutput(NamedTuple):
    """Output container for all baseline classifiers."""

    y_hat: Tensor  # (B, C) class logits


class MLPClassifier(nn.Module):
    """Three-layer MLP with progressive dropout (Wang et al. 2017 MLP baseline).

    Input is flattened directly — no temporal structure assumed.
    """

    def __init__(self, T: int, C: int, hidden: int = 500) -> None:
        """Build MLP with T input features, hidden units per layer, and C outputs."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(T, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, C),
        )

    def forward(self, x: Tensor) -> ClassificationOutput:
        """Return class logits for input x of shape (B, T)."""
        return ClassificationOutput(y_hat=self.net(x))  # (B, C)


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM encoder followed by a linear classification head."""

    def __init__(
        self,
        T: int,
        C: int,
        hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Build bidirectional LSTM with hidden units per direction and C outputs."""
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden * 2, C)

    def forward(self, x: Tensor) -> ClassificationOutput:
        """Return class logits using the final LSTM hidden state."""
        h, _ = self.lstm(x.unsqueeze(-1))  # (B, T, 2*hidden)
        logits = self.fc(h[:, -1, :])  # last step (B, C)
        return ClassificationOutput(y_hat=logits)


class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int) -> None:
        pad = kernel // 2
        super().__init__(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )


class FCNClassifier(nn.Module):
    """Fully Convolutional Network for time series (Wang et al. 2017).

    Three Conv→BN→ReLU blocks with global average pooling and a linear head.
    No pooling between blocks — maintains full temporal resolution until GAP.
    """

    def __init__(self, T: int, C: int) -> None:
        """Build FCN with three conv blocks and a linear head over C classes."""
        super().__init__()
        self.convs = nn.Sequential(
            _ConvBnRelu(1, 128, kernel=8),
            _ConvBnRelu(128, 256, kernel=5),
            _ConvBnRelu(256, 128, kernel=3),
        )
        self.fc = nn.Linear(128, C)

    def forward(self, x: Tensor) -> ClassificationOutput:
        """Return class logits via convolutional feature extraction and GAP."""
        h = self.convs(x.unsqueeze(1))  # (B, 128, T)
        h = h.mean(dim=-1)  # global average pooling → (B, 128)
        return ClassificationOutput(y_hat=self.fc(h))


class _ResBlock(nn.Module):
    """One residual block: three convolutions with a shortcut projection."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=8, padding=4, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.convs(x)
        # Trim/pad to match skip connection length if padding caused mismatch
        if out.shape[-1] != x.shape[-1]:
            out = out[..., : x.shape[-1]]
        return F.relu(out + self.skip(x))


class ResNetClassifier(nn.Module):
    """1D ResNet for time series classification (Wang et al. 2017).

    Three residual blocks (64→128→128 channels) with global average pooling.
    """

    def __init__(self, T: int, C: int) -> None:
        """Build ResNet with three residual blocks and a linear head over C classes."""
        super().__init__()
        self.blocks = nn.Sequential(
            _ResBlock(1, 64),
            _ResBlock(64, 128),
            _ResBlock(128, 128),
        )
        self.fc = nn.Linear(128, C)

    def forward(self, x: Tensor) -> ClassificationOutput:
        """Return class logits via residual feature extraction and GAP."""
        h = self.blocks(x.unsqueeze(1))  # (B, 128, T)
        h = h.mean(dim=-1)  # global average pooling → (B, 128)
        return ClassificationOutput(y_hat=self.fc(h))


class _NBeatsClsBlock(nn.Module):
    """N-BEATS block adapted for classification.

    Produces a backcast (subtracted from the residual input) and a logit
    contribution (accumulated across blocks into the final class scores).
    """

    def __init__(self, T: int, C: int, hidden: int, n_fc: int = 4) -> None:
        """Build one N-BEATS block with backcast and logit projection heads."""
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(T, hidden), nn.ReLU()]
        for _ in range(n_fc - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.fc = nn.Sequential(*layers)
        self.backcast_proj = nn.Linear(hidden, T)
        self.logit_proj = nn.Linear(hidden, C)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return (backcast, logit_contribution) for residual decomposition."""
        h = self.fc(x)
        return self.backcast_proj(h), self.logit_proj(h)  # (B, T), (B, C)


class NBeatsClassifier(nn.Module):
    """N-BEATS adapted for classification (Oreshkin et al. 2020 architecture).

    Each block decomposes the residual input via backcast subtraction and
    contributes class logits that are accumulated across all blocks.
    """

    def __init__(
        self,
        T: int,
        C: int,
        n_stacks: int = 2,
        n_blocks: int = 3,
        hidden: int = 256,
        n_fc: int = 4,
    ) -> None:
        """Build N-BEATS stack of n_stacks × n_blocks blocks with hidden units."""
        super().__init__()
        self.blocks = nn.ModuleList(
            [_NBeatsClsBlock(T, C, hidden, n_fc) for _ in range(n_stacks * n_blocks)]
        )

    def forward(self, x: Tensor) -> ClassificationOutput:
        """Accumulate logit contributions across all blocks via backcast residuals."""
        residual = x
        logits = x.new_zeros(x.shape[0], self.blocks[0].logit_proj.out_features)
        for block in self.blocks:
            backcast, contribution = block(residual)
            residual = residual - backcast
            logits = logits + contribution
        return ClassificationOutput(y_hat=logits)


BASELINE_MODELS = ("mlp", "lstm", "fcn", "resnet", "nbeats")


def build_baseline_classifier(model_type: str, T: int, C: int) -> nn.Module:
    """Construct a baseline classifier by name.

    Args:
        model_type: One of "mlp" | "lstm" | "fcn" | "resnet" | "nbeats".
        T: Input sequence length.
        C: Number of output classes.
    """
    mt = model_type.lower()
    if mt == "mlp":
        return MLPClassifier(T, C)
    if mt == "lstm":
        return LSTMClassifier(T, C)
    if mt == "fcn":
        return FCNClassifier(T, C)
    if mt == "resnet":
        return ResNetClassifier(T, C)
    if mt == "nbeats":
        return NBeatsClassifier(T, C)
    raise ValueError(
        f"Unknown baseline model_type '{model_type}'. Choose: {' | '.join(BASELINE_MODELS)}"
    )
