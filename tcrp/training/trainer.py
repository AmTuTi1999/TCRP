"""Trainer implementation for TCRP."""

import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from tcrp.model.tcrp_forecaster.components.bottleneck import (
    concept_magnitude_loss,
    weighted_alignment_loss,
)
from tcrp.model.tcrp_forecaster.forecaster import TCRPConfig
from tcrp.training.losses import LossBundle, TCRPLoss


class Trainer:
    """Base TCRP trainer: optimiser, scheduler, early stopping, and checkpointing."""

    def __init__(
        self, model: nn.Module, config: TCRPConfig, device: torch.device | None = None
    ):
        """Initialise optimiser, scheduler, loss function, and checkpoint path."""
        self.model = model
        self.config = config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=1e-3, weight_decay=0.0)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )
        self.criterion = TCRPLoss(
            lambda1=config.lambda1,
            lambda2=config.lambda2,
            lambda3=config.lambda3,
            lambda4=config.lambda4,
        )

        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "best.pt")

        self.best_val_mse = float("inf")
        self.epochs_no_improve = 0

    def train_epoch(self, loader: DataLoader) -> LossBundle:
        """Run one training epoch and return averaged loss components."""
        self.model.train()
        totals = {
            "forecast": 0.0,
            "align": 0.0,
            "mag": 0.0,
            "stab": 0.0,
            "reg": 0.0,
            "total": 0.0,
        }
        count = 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(x)
            lb = self.criterion(
                output.y_hat, y, output.A, output.C, self.model.projection
            )
            lb.total_loss.backward()
            self.optimizer.step()

            n = x.shape[0]
            totals["forecast"] += lb.forecast_loss.item() * n
            totals["align"] += lb.align_loss.item() * n
            totals["mag"] += lb.mag_loss.item() * n
            totals["stab"] += lb.stab_loss.item() * n
            totals["reg"] += lb.reg_loss.item() * n
            totals["total"] += lb.total_loss.item() * n
            count += n

        def t(k):
            return torch.tensor(totals[k] / count, device=self.device)

        return LossBundle(
            forecast_loss=t("forecast"),
            align_loss=t("align"),
            mag_loss=t("mag"),
            stab_loss=t("stab"),
            reg_loss=t("reg"),
            total_loss=t("total"),
        )

    def validate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate on loader and return MSE, MAE, alignment, and magnitude losses."""
        self.model.eval()
        totals = {"mse": 0.0, "mae": 0.0, "align": 0.0, "mag": 0.0}
        count = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                n = x.shape[0]
                totals["mse"] += nn.functional.mse_loss(
                    output.y_hat, y, reduction="sum"
                ).item()
                totals["mae"] += nn.functional.l1_loss(
                    output.y_hat, y, reduction="sum"
                ).item()
                totals["align"] += (
                    weighted_alignment_loss(output.A, output.C).item() * n
                )
                totals["mag"] += concept_magnitude_loss(output.A, output.C).item() * n
                count += n

        return {k: v / count for k, v in totals.items()}

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, max_epochs: int = 100
    ):
        """Train with early stopping, saving the best checkpoint by val MSE."""
        for epoch in range(1, max_epochs + 1):
            train_losses = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            self.scheduler.step(val_metrics["mse"])
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"{epoch:4d} | "
                f"loss {train_losses.total_loss.item():.5f} | "
                f"fore {train_losses.forecast_loss.item():.5f} | "
                f"aln {train_losses.align_loss.item():.5f} | "
                f"mag {train_losses.mag_loss.item():.5f} | "
                f"stb {train_losses.stab_loss.item():.5f} | "
                f"val_mse {val_metrics['mse']:.5f} | "
                f"val_mag {val_metrics['mag']:.5f} | "
                f"lr {lr:.2e}"
            )

            if val_metrics["mse"] < self.best_val_mse:
                self.best_val_mse = val_metrics["mse"]
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= 10:
                print(f"Early stopping at epoch {epoch}")
                break
