"""
Baseline forecasting models for comparison with TCRP.

All models share the same interface as TCRPForecaster:
  forward(x: Tensor) -> BaselineOutput   where x is (B, T)

BaselineOutput.y_hat has shape (B, H), so every existing evaluation helper
(_eval_denorm, validate, etc.) works without modification.

Models
------
NBeats         — Generic N-BEATS stack (Oreshkin et al., 2020)
LSTMForecaster — Multi-layer LSTM encoder + linear head
TCNForecaster  — Dilated causal TCN over the full input sequence + linear head
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BaselineOutput(NamedTuple):
    y_hat: Tensor


# ---------------------------------------------------------------------------
# N-BEATS  (generic stack, no trend/seasonality decomposition)
# ---------------------------------------------------------------------------

class _NBeatsBlock(nn.Module):
    def __init__(self, T: int, H: int, hidden: int, n_fc: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(T, hidden), nn.ReLU()]
        for _ in range(n_fc - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.fc = nn.Sequential(*layers)
        self.backcast_proj = nn.Linear(hidden, T)
        self.forecast_proj = nn.Linear(hidden, H)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.fc(x)                        # (B, hidden)
        return self.backcast_proj(h), self.forecast_proj(h)   # (B,T), (B,H)


class NBeats(nn.Module):
    """
    Generic N-BEATS with `n_stacks` stacks of `n_blocks` blocks each.
    Each block produces a backcast (subtracted from the residual) and a
    forecast (accumulated into the output).
    """

    def __init__(
        self,
        T: int,
        H: int,
        n_stacks: int = 2,
        n_blocks: int = 3,
        hidden: int = 256,
        n_fc: int = 4,
    ):
        super().__init__()
        self.H = H
        self.blocks = nn.ModuleList(
            [_NBeatsBlock(T, H, hidden, n_fc) for _ in range(n_stacks * n_blocks)]
        )

    def forward(self, x: Tensor) -> BaselineOutput:
        residual = x                           # (B, T)
        forecast = x.new_zeros(x.shape[0], self.H)
        for block in self.blocks:
            backcast, f = block(residual)
            residual = residual - backcast
            forecast = forecast + f
        return BaselineOutput(y_hat=forecast)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    """
    Multi-layer LSTM over the input sequence followed by a linear head.
    Takes the last-layer hidden state at the final time step as the summary.
    """

    def __init__(
        self,
        T: int,
        H: int,
        hidden: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden, H)

    def forward(self, x: Tensor) -> BaselineOutput:
        h, _ = self.lstm(x.unsqueeze(-1))      # (B, T, hidden)
        y_hat = self.fc(h[:, -1, :])           # (B, H)
        return BaselineOutput(y_hat=y_hat)


# ---------------------------------------------------------------------------
# TCN baseline  (full-sequence dilated causal conv, not segment-based)
# ---------------------------------------------------------------------------

class _CausalBlock(nn.Module):
    """Dilated causal conv + layer norm + residual (no weight_norm for simplicity)."""

    def __init__(self, d: int, kernel: int, dilation: int):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(d, d, kernel, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(d, d, kernel, dilation=dilation, padding=pad)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self._causal_trim = pad  # trim right to enforce causality after symmetric padding

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, d, T)
        trim = self._causal_trim
        out = self.conv1(x)
        if trim:
            out = out[:, :, :-trim]            # causal: drop future context
        out = F.gelu(self.norm1(out.transpose(1, 2)).transpose(1, 2))
        out = self.conv2(out)
        if trim:
            out = out[:, :, :-trim]
        out = self.norm2((out + x).transpose(1, 2)).transpose(1, 2)
        return out


class TCNForecaster(nn.Module):
    """
    Dilated causal TCN over the full input sequence.

    Architecture:
        linear input projection → stack of dilated causal blocks
        → last time step → linear forecast head

    Unlike the TCRP encoder (which operates on fixed-length segments),
    this model processes the raw T-length input directly.
    """

    def __init__(
        self,
        T: int,
        H: int,
        d_model: int = 64,
        n_layers: int = 4,
        kernel: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.blocks = nn.Sequential(
            *[_CausalBlock(d_model, kernel, dilation=2 ** i) for i in range(n_layers)]
        )
        self.fc = nn.Linear(d_model, H)

    def forward(self, x: Tensor) -> BaselineOutput:
        h = self.input_proj(x.unsqueeze(-1))   # (B, T, d)
        h = self.blocks(h.transpose(1, 2))     # (B, d, T)
        y_hat = self.fc(h[:, :, -1])           # (B, H) — last time step
        return BaselineOutput(y_hat=y_hat)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_baseline(
    model_type: str,
    T: int,
    H: int,
    hidden: int = 256,
    n_layers: int = 3,
) -> nn.Module:
    """
    Construct a baseline model by name.

    model_type : "nbeats" | "lstm" | "tcn"
    hidden     : hidden/d_model dimension (all models)
    n_layers   : depth  (lstm: num LSTM layers; tcn: num conv layers;
                         nbeats: n_blocks per stack, n_stacks fixed at 2)
    """
    mt = model_type.lower()
    if mt == "nbeats":
        return NBeats(T, H, n_stacks=2, n_blocks=n_layers, hidden=hidden)
    if mt == "lstm":
        return LSTMForecaster(T, H, hidden=hidden, n_layers=n_layers)
    if mt == "tcn":
        return TCNForecaster(T, H, d_model=hidden, n_layers=n_layers)
    raise ValueError(f"Unknown baseline model_type '{model_type}'. Choose: nbeats | lstm | tcn")
