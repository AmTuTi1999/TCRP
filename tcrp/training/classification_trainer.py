"""Classification trainer for TCRPClassifier.

Adapts the base TCRP training loop for cross-entropy loss while keeping all
concept alignment, magnitude, stability, and regularisation terms.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from tcrp.model.classifier import TCRPClassConfig, TCRPClassifier
from tcrp.model.tcrp_forecaster.components.bottleneck import (
    ConceptProjection,
    concept_magnitude_loss,
    stability_loss,
    weighted_alignment_loss,
)
from tcrp.training.losses import LossBundle


class ClassificationLoss:
    """Combined loss for TCRPClassifier training.

    forecast_loss → cross_entropy (replaces MSE from forecasting)
    align_loss    → weighted Pearson alignment between A and C
    mag_loss      → magnitude alignment (mean|A_k| vs mean|C_k|)
    stab_loss     → consecutive-segment stability of A vs C
    reg_loss      → Frobenius norm of projection weights
    """

    def __init__(
        self,
        lambda1: float,
        lambda2: float,
        lambda3: float = 0.0,
        lambda4: float = 0.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        """Store loss weights and optional per-class weights."""
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)
        self.lambda3 = float(lambda3)
        self.lambda4 = float(lambda4)
        self.class_weights = class_weights  # (C,) tensor or None

    def __call__(
        self,
        logits: torch.Tensor,
        y_true: torch.Tensor,
        A: torch.Tensor,
        C: torch.Tensor,
        projection: ConceptProjection,
    ) -> LossBundle:
        """Compute combined cross-entropy and concept alignment losses."""
        device = logits.device
        weights = (
            self.class_weights.to(device) if self.class_weights is not None else None
        )

        forecast_loss = F.cross_entropy(logits, y_true, weight=weights)
        align_loss_val = weighted_alignment_loss(A, C)
        mag_loss_val = concept_magnitude_loss(A, C)
        stab_loss_val = stability_loss(A, C)
        reg_loss_val = torch.norm(projection.linear.weight, p="fro")

        total = (
            forecast_loss
            + self.lambda1 * align_loss_val
            + self.lambda4 * mag_loss_val
            + self.lambda3 * stab_loss_val
            + self.lambda2 * reg_loss_val
        )
        return LossBundle(
            forecast_loss=forecast_loss,
            align_loss=align_loss_val,
            mag_loss=mag_loss_val,
            stab_loss=stab_loss_val,
            reg_loss=reg_loss_val,
            total_loss=total,
        )


class ClassificationTrainer:
    """Trainer for TCRPClassifier: cross-entropy + concept alignment losses."""

    def __init__(
        self,
        model: TCRPClassifier,
        config: TCRPClassConfig,
        lr: float = 1e-3,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        grad_clip: float = 1.0,
        es_patience: int = 20,
        checkpoint_path: str = "checkpoints/classifier_best.pt",
        class_weights: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Set up optimiser, scheduler, and loss for classification training."""
        self.model = model
        self.config = config
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self._grad_clip = grad_clip
        self._es_patience = es_patience
        self.checkpoint_path = checkpoint_path

        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=0.0)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=lr_patience, factor=lr_factor
        )
        self.criterion = ClassificationLoss(
            lambda1=config.lambda1,
            lambda2=config.lambda2,
            lambda3=config.lambda3,
            lambda4=config.lambda4,
            class_weights=class_weights,
        )
        self.best_val_loss = float("inf")
        self.epochs_no_improve = 0

    def train_epoch(self, loader: DataLoader) -> LossBundle:
        """One training epoch with gradient clipping."""
        self.model.train()
        totals = {k: 0.0 for k in ("forecast", "align", "mag", "stab", "reg", "total")}
        count = 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(x)
            lb = self.criterion(out.y_hat, y, out.A, out.C, self.model.projection)
            lb.total_loss.backward()
            if self._grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self._grad_clip)
            self.optimizer.step()
            bs = x.shape[0]
            totals["forecast"] += lb.forecast_loss.item() * bs
            totals["align"] += lb.align_loss.item() * bs
            totals["mag"] += lb.mag_loss.item() * bs
            totals["stab"] += lb.stab_loss.item() * bs
            totals["reg"] += lb.reg_loss.item() * bs
            totals["total"] += lb.total_loss.item() * bs
            count += bs

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

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate on validation loader, returning CE loss and accuracy."""
        self.model.eval()
        total_ce, total_correct, count = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            total_ce += F.cross_entropy(out.y_hat, y, reduction="sum").item()
            total_correct += (out.y_hat.argmax(dim=-1) == y).sum().item()
            count += x.shape[0]
        return {"ce": total_ce / count, "accuracy": total_correct / count}

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, max_epochs: int = 100
    ) -> None:
        """Fit with early stopping, saving best checkpoint by validation CE."""
        import os

        os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)

        header = (
            f"\n{'Epoch':>6} | {'train_CE':>10} | {'val_CE':>10} | "
            f"{'val_acc':>8} | {'align':>10} | {'lr':>10}"
        )
        print(header)
        print("-" * 66)

        for epoch in range(1, max_epochs + 1):
            lb = self.train_epoch(train_loader)
            val_m = self.validate(val_loader)
            self.scheduler.step(val_m["ce"])
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"{epoch:6d} | {lb.forecast_loss.item():10.6f} | "
                f"{val_m['ce']:10.6f} | {val_m['accuracy']:8.4f} | "
                f"{lb.align_loss.item():10.6f} | {lr:10.2e}"
            )

            if val_m["ce"] < self.best_val_loss:
                self.best_val_loss = val_m["ce"]
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self._es_patience:
                print(f"Early stopping at epoch {epoch}.")
                break
