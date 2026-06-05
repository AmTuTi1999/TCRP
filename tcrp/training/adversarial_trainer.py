"""Phase T*-04 · Adversarial Trainer.

Extends Trainer with a two-path backward pass and GRL alpha scheduling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tcrp.model.tcrp_forecaster.components.adversarial import (
    AdversarialTCRPForecaster,
    grl_alpha_schedule,
)
from tcrp.model.tcrp_forecaster.components.bottleneck import (
    concept_magnitude_loss,
    stability_loss,
    weighted_alignment_loss,
)
from tcrp.model.tcrp_forecaster.forecaster import TCRPConfig
from tcrp.training.losses import LossBundle
from tcrp.training.trainer import Trainer


class AdversarialTrainer(Trainer):
    """Replaces the standard single-loss backward with a two-path backward.

    Path 1: L_forecast + L_reg  → decoder, projection, encoder (normal gradient)
    Path 2: L_align + L_stab    → projection + GRL → encoder (reversed gradient)

    All standard monitoring (early stopping, LR scheduling) is inherited
    from Trainer.  Additional per-epoch logging:
    - alpha value
    - concept purity score (every 5 epochs when diagnostics are available)
    """

    model: AdversarialTCRPForecaster  # narrowed type

    def __init__(
        self,
        model: AdversarialTCRPForecaster,
        config: TCRPConfig,
        device: torch.device | None = None,
    ):
        """Initialize AdversarialTrainer wrapping an AdversarialTCRPForecaster."""
        super().__init__(model, config, device)

    # ------------------------------------------------------------------
    # Core two-path training step
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> LossBundle:
        """Run one adversarial training epoch with the two-path backward pass."""
        self.model.train()
        tot_fc = tot_al = tot_mag = tot_stab = tot_reg = tot = 0.0
        count = 0

        for batch in loader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            forecast_output, A_align = self.model(x)
            C = forecast_output.C  # already detached (no_grad in forward)

            L_fc = F.mse_loss(forecast_output.y_hat, y)
            L_al = weighted_alignment_loss(A_align, C)
            L_mag = concept_magnitude_loss(A_align, C)
            L_stab = stability_loss(A_align, C)
            L_reg = self.config.lambda2 * self.model.base.projection.linear.weight.norm(
                "fro"
            )

            # Path 1: forecast + reg — normal gradients reach encoder
            (L_fc + L_reg).backward(retain_graph=True)

            # Path 2: alignment + magnitude + stability — reversed via GRL
            (
                self.config.lambda1 * L_al
                + self.config.lambda4 * L_mag
                + self.config.lambda3 * L_stab
            ).backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            bs = x.shape[0]
            tot_fc += L_fc.item() * bs
            tot_al += L_al.item() * bs
            tot_mag += L_mag.item() * bs
            tot_stab += L_stab.item() * bs
            tot_reg += L_reg.item() * bs
            tot += (
                L_fc + self.config.lambda1 * L_al + self.config.lambda4 * L_mag + L_reg
            ).item() * bs
            count += bs

        def t(v):
            return torch.tensor(v / count, device=self.device)

        return LossBundle(
            forecast_loss=t(tot_fc),
            align_loss=t(tot_al),
            mag_loss=t(tot_mag),
            stab_loss=t(tot_stab),
            reg_loss=t(tot_reg),
            total_loss=t(tot),
        )

    # ------------------------------------------------------------------
    # validate — delegates to parent; A_align not needed for val metrics
    # ------------------------------------------------------------------

    def validate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate the model on the validation loader."""
        self.model.eval()
        tot_mse = tot_mae = tot_align = tot_mag = 0.0
        count = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                forecast_output, _ = self.model(x)
                A, C = forecast_output.A, forecast_output.C
                tot_mse += F.mse_loss(forecast_output.y_hat, y, reduction="sum").item()
                tot_mae += F.l1_loss(forecast_output.y_hat, y, reduction="sum").item()
                tot_align += weighted_alignment_loss(A, C).item() * x.shape[0]
                tot_mag += concept_magnitude_loss(A, C).item() * x.shape[0]
                count += x.shape[0]

        return {
            "mse": tot_mse / count,
            "mae": tot_mae / count,
            "align_loss": tot_align / count,
            "mag": tot_mag / count,
        }

    # ------------------------------------------------------------------
    # fit — alpha scheduling + standard epoch loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
    ) -> None:
        """Train with GRL alpha scheduling and early stopping."""
        for epoch in range(1, max_epochs + 1):
            alpha = grl_alpha_schedule(
                epoch - 1,
                max_epochs,
                warmup_epochs=self.config.warmup_epochs,
                alpha_max=self.config.alpha_max,
            )
            self.model.set_alpha(alpha)

            train_bundle = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            self.scheduler.step(val_metrics["mse"])
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"{epoch:4d} | α={alpha:.4f} | "
                f"fc {train_bundle.forecast_loss.item():.5f} | "
                f"aln {train_bundle.align_loss.item():.5f} | "
                f"mag {train_bundle.mag_loss.item():.5f} | "
                f"stb {train_bundle.stab_loss.item():.5f} | "
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
