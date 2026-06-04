"""TCRP analysis pass for relevance propagation and explanation generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from tcrp.analysis.lrp import lrp_gamma_conv, lrp_linear_eps, lrp_mean_pool, lrp_relu
from tcrp.model.tcrp_forecaster.components.bottleneck import ConceptProjection
from tcrp.model.tcrp_forecaster.components.decoder import HorizonDecoder, GaussianDecoder
from tcrp.model.tcrp_forecaster.components.encoder import TCNEncoder, LSTMEncoder, CausalDilatedBlock
from tcrp.model.tcrp_forecaster.forecaster import TCRPForecaster


@dataclass
class TCRPExplanation:
    R_h: Tensor
    R_A: Tensor
    R_x: Tensor
    R_x_cond: Tensor
    eta: Tensor
    A: Tensor
    C: Tensor


class TCRPAnalyser:
    def __init__(self, model: TCRPForecaster, eps: float = 1e-6):
        self.model = model
        self.eps = eps

    def analyse(self, x: Tensor, h_star: int = 0) -> TCRPExplanation:
        if x.dim() != 2:
            raise ValueError(f"Expected x shape (B, T), got {tuple(x.shape)}")

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

            if self.model.config.probabilistic:
                mu, sigma = self.model.decoder(h)
                y_hat = mu
            else:
                y_hat = self.model.decoder(h)

        R_h = self._decode_relevance(h, y_hat, h_star)
        R_A = eta.unsqueeze(-1) * R_h.unsqueeze(1)
        R_z = self._project_relevance(Z, A, R_A)
        R_s = self._encode_relevance(hidden_acts, final_act, R_z)
        R_x = self._assemble_relevance(R_s, self.model.segmenter.start_indices, self.model.segmenter.overlap_counts, T)
        R_x_cond = self._assemble_concept_conditional(R_s, R_A, self.model.segmenter.start_indices, self.model.segmenter.overlap_counts, T)

        return TCRPExplanation(
            R_h=R_h,
            R_A=R_A,
            R_x=R_x,
            R_x_cond=R_x_cond,
            eta=eta,
            A=A,
            C=C,
        )

    def _encode_with_cache(self, segments: Tensor) -> Tuple[Tensor, list, Tensor]:
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
            residual = block.residual_proj(a0) if block.residual_proj is not None else a0
            out = x2_relu + residual

            hidden_acts.append((a0, x1, x1_relu, x2, x2_relu, residual, block))
            current = out

        Z = current.mean(dim=-1).view(B, N, self.model.config.d)
        return Z, hidden_acts, current

    def _decode_relevance(self, h: Tensor, y_hat: Tensor, h_star: int) -> Tensor:
        theta = self.model.decoder.Theta
        if h_star < 0 or h_star >= theta.shape[0]:
            raise ValueError(f"Invalid h_star index {h_star}")

        theta_star = theta[h_star]
        z = (h * theta_star.unsqueeze(0)).sum(dim=1)
        denom = z + self.eps * torch.sign(z)
        R_out = y_hat[:, h_star]
        R_h = (h * theta_star.unsqueeze(0)) / denom.unsqueeze(1) * R_out.unsqueeze(1)
        return R_h

    def _project_relevance(self, Z: Tensor, A: Tensor, R_A: Tensor) -> Tensor:
        B, N, d = Z.shape
        R_A_flat = R_A.reshape(-1, self.model.config.K)
        Z_flat = Z.reshape(-1, d)
        R_z_flat = lrp_linear_eps(self.model.projection.linear, Z_flat, R_A_flat, eps=self.eps)
        return R_z_flat.view(B, N, d)

    def _encode_relevance(self, hidden_acts: list, final_act: Tensor, R_z: Tensor) -> Tensor:
        B, N, d = R_z.shape
        R = R_z.reshape(B * N, d)
        R_current = lrp_mean_pool(final_act.detach(), R)

        for a0, x1, x1_relu, x2, x2_relu, residual, block in reversed(hidden_acts):
            eps = self.eps
            # Split relevance between main path (x2_relu) and residual branch.
            # Use exact conservation: R_main + R_res = R_current.
            abs_main = torch.abs(x2_relu)
            abs_res  = torch.abs(residual)
            safe_total = (abs_main + abs_res).clamp(min=eps)
            R_main = R_current * abs_main / safe_total
            R_res  = R_current - R_main   # exact: no eps leak

            R_main = lrp_relu(x2, R_main)
            R_main = lrp_gamma_conv(block.conv2, x1_relu, R_main, gamma=0.25, eps=eps)
            R_main = lrp_relu(x1, R_main)
            R_a0 = lrp_gamma_conv(block.conv1, a0, R_main, gamma=0.25, eps=eps)

            if block.residual_proj is not None:
                R_res_in = lrp_gamma_conv(block.residual_proj, a0, R_res, gamma=0.25, eps=eps)
            else:
                R_res_in = R_res

            R_current = R_a0 + R_res_in

        return R_current.view(B, N, R_current.shape[-1])

    def _assemble_relevance(self, R_s: Tensor, starts: Tensor, overlap: Tensor, T: int) -> Tensor:
        B, N, L = R_s.shape
        R_x = torch.zeros(B, T, device=R_s.device, dtype=R_s.dtype)
        for n in range(N):
            start = starts[n].item()
            R_x[:, start:start + L] += R_s[:, n, :]
        # clamp(min=1) avoids divide-by-zero for timesteps not covered by any segment
        R_x = R_x / overlap.clamp(min=1).unsqueeze(0)
        return R_x

    def _assemble_concept_conditional(self, R_s: Tensor, R_A: Tensor, starts: Tensor, overlap: Tensor, T: int) -> Tensor:
        B, N, L = R_s.shape
        K = R_A.shape[-1]
        weight = R_A / (R_A.sum(dim=2, keepdim=True) + self.eps)
        safe_overlap = overlap.clamp(min=1)
        R_x_cond = torch.zeros(B, K, T, device=R_s.device, dtype=R_s.dtype)
        for n in range(N):
            start = starts[n].item()
            weighted = weight[:, n, :].unsqueeze(-1) * R_s[:, n, :].unsqueeze(1)
            R_x_cond[:, :, start:start + L] += weighted / safe_overlap[start:start + L].unsqueeze(0).unsqueeze(0)
        return R_x_cond


def verify_conservation(explanation: TCRPExplanation, y_hat: Tensor, h_star: int, tol: float = 1e-4) -> bool:
    ok = True
    R_x = explanation.R_x
    R_x_cond = explanation.R_x_cond
    R_h = explanation.R_h
    target = y_hat[:, h_star]

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
