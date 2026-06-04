"""Trainer implementation for TCRP."""
import os
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from tcrp.model.tcrp_forecaster.components.bottleneck import alignment_loss
from tcrp.model.tcrp_forecaster.forecaster import TCRPConfig
from tcrp.training.losses import LossBundle, TCRPLoss


class Trainer:
    def __init__(self, model: nn.Module, config: TCRPConfig, device: torch.device | None = None):
        self.model = model
        self.config = config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=1e-3, weight_decay=0.0)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=5,
            factor=0.5,
        )
        self.criterion = TCRPLoss(lambda1=config.lambda1, lambda2=config.lambda2)

        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "best.pt")

        self.best_val_mse = float("inf")
        self.epochs_no_improve = 0

    def train_epoch(self, loader: DataLoader) -> LossBundle:
        self.model.train()
        total_forecast = 0.0
        total_align = 0.0
        total_reg = 0.0
        total_loss = 0.0
        count = 0

        for batch in loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)

            loss_bundle = self.criterion(output.y_hat, y, output.A, output.C, self.model.projection)
            loss_bundle.total_loss.backward()
            self.optimizer.step()

            batch_size = x.shape[0]
            total_forecast += loss_bundle.forecast_loss.item() * batch_size
            total_align += loss_bundle.align_loss.item() * batch_size
            total_reg += loss_bundle.reg_loss.item() * batch_size
            total_loss += loss_bundle.total_loss.item() * batch_size
            count += batch_size

        return LossBundle(
            forecast_loss=torch.tensor(total_forecast / count, device=self.device),
            align_loss=torch.tensor(total_align / count, device=self.device),
            reg_loss=torch.tensor(total_reg / count, device=self.device),
            total_loss=torch.tensor(total_loss / count, device=self.device),
        )

    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0
        total_align = 0.0
        count = 0

        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)
                total_mse += nn.functional.mse_loss(output.y_hat, y, reduction="sum").item()
                total_mae += nn.functional.l1_loss(output.y_hat, y, reduction="sum").item()
                total_align += alignment_loss(output.A, output.C).item() * x.shape[0]
                count += x.shape[0]
        return {
            "mse": total_mse / count,
            "mae": total_mae / count,
            "align_loss": total_align / count,
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, max_epochs: int = 100):
        for epoch in range(1, max_epochs + 1):
            train_losses = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            self.scheduler.step(val_metrics["mse"])

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"{epoch} | {train_losses.forecast_loss.item():.6f} | {val_metrics['mse']:.6f} | {val_metrics['align_loss']:.6f} | {lr:.6e}"
            )

            if val_metrics["mse"] < self.best_val_mse:
                self.best_val_mse = val_metrics["mse"]
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= 10:
                print(f"Early stopping at epoch {epoch} (no improvement for {self.epochs_no_improve} epochs)")
                break
