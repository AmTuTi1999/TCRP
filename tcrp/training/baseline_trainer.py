"""Trainer for baseline models (NBeats, LSTM, TCN) — MSE only, no alignment loss."""
from __future__ import annotations

import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


class BaselineTrainer:
    """
    Minimal trainer for baseline forecasting models.

    Optimises MSE only — no concept alignment or regularisation.
    Shares the same fit / validate / checkpoint interface as Trainer so
    it can be dropped into the pipeline without any other changes.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        grad_clip: float = 1.0,
        es_patience: int = 10,
        checkpoint_path: str = "checkpoints/baseline_best.pt",
        device: torch.device | None = None,
    ):
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        self.optimizer = Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=lr_patience, factor=lr_factor
        )
        self._grad_clip  = grad_clip
        self._es_patience = es_patience

        self.checkpoint_path = checkpoint_path
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

        self.best_val_mse    = float("inf")
        self.epochs_no_improve = 0

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total, count = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            loss = F.mse_loss(self.model(x).y_hat, y)
            loss.backward()
            if self._grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip)
            self.optimizer.step()
            total += loss.item() * x.shape[0]
            count += x.shape[0]
        return total / count

    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        tot_mse = tot_mae = 0.0
        count = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model(x).y_hat
                tot_mse += F.mse_loss(y_hat, y, reduction="sum").item()
                tot_mae += F.l1_loss(y_hat, y, reduction="sum").item()
                count   += x.shape[0]
        return {"mse": tot_mse / count, "mae": tot_mae / count}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, max_epochs: int = 100) -> None:
        for epoch in range(1, max_epochs + 1):
            train_mse = self.train_epoch(train_loader)
            val_m     = self.validate(val_loader)
            self.scheduler.step(val_m["mse"])
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"{epoch:6d} | {train_mse:10.6f} | "
                f"{val_m['mse']:10.6f} | {val_m['mae']:10.6f} | {lr:10.2e}"
            )
            if val_m["mse"] < self.best_val_mse:
                self.best_val_mse      = val_m["mse"]
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                self.epochs_no_improve += 1
            if self.epochs_no_improve >= self._es_patience:
                print(f"Early stopping at epoch {epoch} ({self.epochs_no_improve} epochs without improvement)")
                break
