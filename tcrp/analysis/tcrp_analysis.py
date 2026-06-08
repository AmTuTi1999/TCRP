"""TCRP analysis pass for relevance propagation and explanation generation.

Supports both TCRPForecaster (forecasting) and TCRPClassifier (classification).

Forecasting mode  — call analyse(x, h_star=<step>)
Classification mode — call analyse(x, k_star=<class_idx>)
                      or analyse(x, k_star=None) to use each sample's
                      argmax-predicted class (default for classifiers).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from tcrp.analysis.lrp import lrp_gamma_conv, lrp_linear_eps, lrp_mean_pool, lrp_relu
from tcrp.model.tcrp_forecaster.components.encoder import CausalDilatedBlock, TCNEncoder
from tcrp.model.tcrp_forecaster.forecaster import TCRPForecaster


@dataclass
class TCRPExplanation:
    """Container for all LRP relevance tensors produced by TCRPAnalyser."""

    R_h: Tensor
    R_A: Tensor
    R_x: Tensor
    R_x_cond: Tensor
    eta: Tensor
    A: Tensor
    C: Tensor
    # Segmentation metadata — needed for segment-level visualisations.
    starts: Tensor  # (N,) integer start indices of each segment
    L: int  # segment length in timesteps
    # Classification-only: (B,) predicted or specified class per sample.
    # None for forecasting.
    k_stars: Tensor | None = None


class TCRPAnalyser:
    """Runs LRP-based relevance propagation through a trained TCRP model.

    Accepts both ``TCRPForecaster`` (forecasting) and ``TCRPClassifier``
    (classification). The analysis pass is identical in both cases; only the
    decoder relevance initialisation differs:

    - Forecasting: ``R_out = y_hat[:, h_star]`` — the logit at horizon step h_star.
    - Classification: ``R_out = y_hat[i, k_star_i]`` — the predicted class logit,
      one per sample. k_star can be fixed (same class for all samples) or
      per-sample (each sample's argmax).
    """

    def __init__(self, model: TCRPForecaster, eps: float = 1e-6):
        """Initialize TCRPAnalyser with a trained model and LRP epsilon."""
        self.model = model
        self.eps = eps

    # ── Public entry point ───────────────────────────────────────────────────

    def analyse(
        self,
        x: Tensor,
        h_star: int = 0,
        k_star: int | None = -1,
    ) -> TCRPExplanation:
        """Run the full LRP analysis pass and return a TCRPExplanation.

        Args:
            x:      Input tensor of shape (B, T).
            h_star: Horizon step to explain (forecasting only, 0-based).
            k_star: Class index to explain (classification only).
                    Pass an integer to explain a specific class for all samples.
                    Pass ``None`` to use each sample's argmax-predicted class.
                    The sentinel ``-1`` (default) means "auto-select based on
                    model type": forecaster uses h_star, classifier uses None.
        """
        if x.dim() != 2:
            raise ValueError(f"Expected x shape (B, T), got {tuple(x.shape)}")

        from tcrp.model.classifier import TCRPClassifier

        is_cls = isinstance(self.model, TCRPClassifier)

        with torch.no_grad():
            B, T = x.shape
            segments = self.model.segmenter(x)
            B, N, L = segments.shape

            flat_segments = segments.reshape(-1, L)
            C_flat = self.model.scorer(flat_segments)
            C = C_flat.view(B, N, self.model.config.K)

            Z, hidden_acts, final_act = self._encode_with_cache(segments)
            A = self.model.projection(Z)
            h, eta = self.model.pool(A)

            if is_cls:
                y_hat = self.model.decoder(h)  # (B, C_classes) logits
            elif getattr(self.model.config, "probabilistic", False):
                mu, _ = self.model.decoder(h)
                y_hat = mu
            else:
                y_hat = self.model.decoder(h)

        if is_cls:
            # Resolve k_stars: per-sample predicted class or fixed class index.
            _k = k_star if k_star != -1 else None
            if _k is None:
                k_stars = y_hat.argmax(dim=-1)  # (B,) predicted classes
            else:
                k_stars = torch.full((B,), _k, dtype=torch.long, device=x.device)
            R_h = self._decode_relevance_cls(h, y_hat, k_stars)
        else:
            k_stars = None
            R_h = self._decode_relevance(h, y_hat, h_star)

        R_A = eta.unsqueeze(-1) * R_h.unsqueeze(1)
        R_z = self._project_relevance(Z, A, R_A)
        R_s = self._encode_relevance(hidden_acts, final_act, R_z)
        R_x = self._assemble_relevance(R_s, self.model.segmenter.start_indices, T)
        R_x_cond = self._assemble_concept_conditional(
            R_s, R_A, self.model.segmenter.start_indices, T
        )

        return TCRPExplanation(
            R_h=R_h,
            R_A=R_A,
            R_x=R_x,
            R_x_cond=R_x_cond,
            eta=eta,
            A=A,
            C=C,
            starts=self.model.segmenter.start_indices,
            L=L,
            k_stars=k_stars,
        )

    # ── Encoder with activation cache ────────────────────────────────────────

    def _encode_with_cache(self, segments: Tensor) -> tuple[Tensor, list, Tensor]:
        if not isinstance(self.model.encoder, TCNEncoder):
            raise NotImplementedError(
                "TCRPAnalyser currently supports TCNEncoder only; "
                f"got {type(self.model.encoder).__name__}"
            )
        B, N, L = segments.shape
        x = segments.reshape(B * N, 1, L)
        hidden_acts = []
        current = x

        for block in self.model.encoder.tcn:
            if not isinstance(block, CausalDilatedBlock):
                raise RuntimeError("Expected CausalDilatedBlock in TCNEncoder stack")

            a0 = current
            pad = (block.kernel_size - 1) * block.dilation
            x1 = block.conv1(F.pad(a0, (pad, 0)))
            x1_relu = block.relu(x1)
            x2 = block.conv2(F.pad(x1_relu, (pad, 0)))
            x2_relu = block.relu(x2)
            residual = (
                block.residual_proj(a0) if block.residual_proj is not None else a0
            )
            out = x2_relu + residual

            hidden_acts.append((a0, x1, x1_relu, x2, x2_relu, residual, block))
            current = out

        # Use config.d if available (TCRPConfig); fall back to encoder_hidden
        # (TCRPClassConfig exposes d as a property, so both work via getattr).
        d = getattr(self.model.config, "d", self.model.config.encoder_hidden)
        Z = current.mean(dim=-1).view(B, N, d)
        return Z, hidden_acts, current

    # ── Decoder relevance initialisation ─────────────────────────────────────

    def _decode_relevance(self, h: Tensor, y_hat: Tensor, h_star: int) -> Tensor:
        """Forecasting: propagate relevance for horizon step h_star (same for all samples)."""
        theta = self.model.decoder.Theta  # (H, K)
        if h_star < 0 or h_star >= theta.shape[0]:
            raise ValueError(f"h_star={h_star} out of range [0, {theta.shape[0]})")
        theta_star = theta[h_star]  # (K,)
        z = (h * theta_star.unsqueeze(0)).sum(dim=1)
        denom = z + self.eps * torch.sign(z)
        R_out = y_hat[:, h_star]
        R_h = (h * theta_star.unsqueeze(0)) / denom.unsqueeze(1) * R_out.unsqueeze(1)
        return R_h

    def _decode_relevance_cls(
        self, h: Tensor, y_hat: Tensor, k_stars: Tensor
    ) -> Tensor:
        """Classification: propagate relevance for class k_stars (B,) — one per sample.

        Conservation: sum_k R_h[b, k] == y_hat[b, k_star_b] for every sample b.
        """
        theta = self.model.decoder.Theta  # (C, K)
        theta_per_sample = theta[k_stars]  # (B, K) — each sample's decoder row
        z = (h * theta_per_sample).sum(dim=1)  # (B,)
        denom = z + self.eps * torch.sign(z)
        R_out = y_hat.gather(1, k_stars.unsqueeze(1)).squeeze(1)  # (B,) predicted logit
        R_h = (h * theta_per_sample) / denom.unsqueeze(1) * R_out.unsqueeze(1)
        return R_h

    # ── Projection relevance ─────────────────────────────────────────────────

    def _project_relevance(self, Z: Tensor, A: Tensor, R_A: Tensor) -> Tensor:
        B, N, d = Z.shape
        R_A_flat = R_A.reshape(-1, self.model.config.K)
        Z_flat = Z.reshape(-1, d)
        R_z_flat = lrp_linear_eps(
            self.model.projection.linear, Z_flat, R_A_flat, eps=self.eps
        )
        return R_z_flat.view(B, N, d)

    # ── Encoder relevance (backward through TCN) ─────────────────────────────

    def _encode_relevance(
        self, hidden_acts: list, final_act: Tensor, R_z: Tensor
    ) -> Tensor:
        B, N, d = R_z.shape
        R = R_z.reshape(B * N, d)
        R_current = lrp_mean_pool(final_act.detach(), R)

        for a0, x1, x1_relu, x2, x2_relu, residual, block in reversed(hidden_acts):
            eps = self.eps
            abs_main = torch.abs(x2_relu)
            abs_res = torch.abs(residual)
            safe_total = (abs_main + abs_res).clamp(min=eps)
            R_main = R_current * abs_main / safe_total
            R_res = R_current - R_main

            R_main = lrp_relu(x2, R_main)
            R_main = lrp_gamma_conv(block.conv2, x1_relu, R_main, gamma=0.25, eps=eps)
            R_main = lrp_relu(x1, R_main)
            R_a0 = lrp_gamma_conv(block.conv1, a0, R_main, gamma=0.25, eps=eps)

            if block.residual_proj is not None:
                R_res_in = lrp_gamma_conv(
                    block.residual_proj, a0, R_res, gamma=0.25, eps=eps
                )
            else:
                R_res_in = R_res

            R_current = R_a0 + R_res_in

        return R_current.view(B, N, R_current.shape[-1])

    # ── Temporal assembly ─────────────────────────────────────────────────────

    def _assemble_relevance(self, R_s: Tensor, starts: Tensor, T: int) -> Tensor:
        B, N, L = R_s.shape
        R_x = torch.zeros(B, T, device=R_s.device, dtype=R_s.dtype)
        for n in range(N):
            start = starts[n].item()
            R_x[:, start : start + L] += R_s[:, n, :]
        return R_x

    def _assemble_concept_conditional(
        self, R_s: Tensor, R_A: Tensor, starts: Tensor, T: int
    ) -> Tensor:
        B, N, L = R_s.shape
        K = R_A.shape[-1]
        weight = R_A / (R_A.sum(dim=2, keepdim=True) + self.eps)
        R_x_cond = torch.zeros(B, K, T, device=R_s.device, dtype=R_s.dtype)
        for n in range(N):
            start = starts[n].item()
            weighted = weight[:, n, :].unsqueeze(-1) * R_s[:, n, :].unsqueeze(1)
            R_x_cond[:, :, start : start + L] += weighted
        return R_x_cond


# ── Conservation verification ────────────────────────────────────────────────


def verify_conservation(
    explanation: TCRPExplanation,
    y_hat: Tensor,
    h_star: int = 0,
    tol: float = 1e-4,
    target: Tensor | None = None,
) -> bool:
    """Verify that LRP relevance scores sum to the model output within tolerance.

    Args:
        explanation: Output of TCRPAnalyser.analyse().
        y_hat:       Raw model outputs (B, H) for forecasting or (B, C) for classification.
        h_star:      Horizon step (forecasting only; ignored when target is given).
        tol:         Absolute tolerance.
        target:      Pre-computed per-sample target tensor (B,). When provided,
                     overrides the h_star computation. Pass for classification where
                     each sample has its own k_star.
    """
    if target is None:
        target = y_hat[:, h_star]

    ok = True
    R_x = explanation.R_x
    R_x_cond = explanation.R_x_cond
    R_h = explanation.R_h

    x_sum = R_x.sum(dim=-1)
    if not torch.allclose(x_sum, target, atol=tol, rtol=0.0):
        print("Conservation violation: R_x sum", x_sum, "target", target)
        ok = False

    cond_sum = R_x_cond.sum(dim=1)
    if not torch.allclose(cond_sum, R_x, atol=tol, rtol=0.0):
        print("Conservation violation: R_x_cond sum", cond_sum, "R_x", R_x)
        ok = False

    h_sum = R_h.sum(dim=-1)
    if not torch.allclose(h_sum, target, atol=tol, rtol=0.0):
        print("Conservation violation: R_h sum", h_sum, "target", target)
        ok = False

    return ok
