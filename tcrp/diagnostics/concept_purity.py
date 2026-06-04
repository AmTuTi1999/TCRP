"""
Phase T*-05 · Concept Purity Diagnostic

Measures whether learned concept directions w_k are aligned with the genuine
analytic gradient direction, or contaminated by spurious encoder features.

The "mean gradient" of analytic score c_k w.r.t. encoder output z is computed
via least-squares linear regression (z → c_k), which equals the expected
gradient E[∂c_k/∂z] under a linear approximation of the encoder–scorer
relationship.  This is equivalent to the TCAV directional derivative approach
and does not require the scorer and encoder to share a computational graph.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from tcrp.concepts import ConceptScorer
from tcrp.model.tcrp_forecaster.components.adversarial import AdversarialTCRPForecaster


def concept_purity_score(
    model: AdversarialTCRPForecaster,
    concept_scorer: ConceptScorer,
    segments: Tensor,
    k: int,
) -> dict:
    """
    Computes cosine similarity between w_k (learned concept direction in
    encoder space) and the mean gradient direction of analytic score c_k
    w.r.t. encoder output z.

    segments: (N, L) — batch of 1-D time-series segments (validation set).
    k:        concept index.

    Returns dict with keys: concept, cosine_sim, pure, warning.

    Interpretation:
      > 0.7  : pure — concept direction genuinely aligned
      0.5–0.7: moderate contamination — monitor
      0.3–0.5: significant contamination — increase alpha_max or lambda1
      < 0.3  : severe contamination — adversarial training not converging
    """
    model.eval()
    with torch.no_grad():
        # Encoder expects (B, N_segs, L); treat each segment as a batch element
        # with one segment: (N, 1, L) → (N, 1, d) → squeeze → (N, d)
        z = model.base.encoder(segments.unsqueeze(1)).squeeze(1)  # (N, d)
        c = concept_scorer(segments)[:, k]     # (N,)

    # Mean gradient of c_k w.r.t. z via least-squares linear regression:
    # find w ∈ R^d such that z_centered @ w ≈ c_centered.
    # The solution w is the gradient direction E[∂c_k/∂z] under a linear model.
    z_c = z - z.mean(0)                        # (N, d)
    c_c = c - c.mean()                         # (N,)

    try:
        w_raw = torch.linalg.lstsq(z_c, c_c.unsqueeze(-1)).solution.squeeze(-1)  # (d,)
    except RuntimeError:
        # Fallback: closed-form normal equations (works when N < d)
        var = (z_c ** 2).mean(0).clamp(min=1e-8)     # (d,)
        w_raw = (z_c * c_c.unsqueeze(-1)).mean(0) / var   # (d,)

    mean_grad = F.normalize(w_raw.unsqueeze(0), dim=1)  # (1, d) unit vector

    w_k = F.normalize(
        model.base.projection.linear.weight[k].unsqueeze(0), dim=1
    )  # (1, d) unit vector

    cosine_sim = (w_k * mean_grad).sum().item()

    return {
        "concept":    concept_scorer.concept_names[k],
        "cosine_sim": cosine_sim,
        "pure":       cosine_sim > 0.7,
        "warning":    cosine_sim < 0.5,
    }


def concept_purity_report(
    model: AdversarialTCRPForecaster,
    concept_scorer: ConceptScorer,
    train_segments: Tensor,
    val_segments: Tensor,
) -> dict:
    """
    Runs concept_purity_score for all K concepts on both train and val segments.

    Returns a dict keyed by concept name with fields:
      cosine_train, cosine_val, purity_gap, pure_val, warning

    purity_gap = cosine_train - cosine_val.  A positive gap indicates spurious
    contamination persisting despite adversarial training.
    """
    report: dict = {}
    K = len(concept_scorer.concept_names)
    for k in range(K):
        tr = concept_purity_score(model, concept_scorer, train_segments, k)
        va = concept_purity_score(model, concept_scorer, val_segments,   k)
        gap = tr["cosine_sim"] - va["cosine_sim"]
        report[concept_scorer.concept_names[k]] = {
            "cosine_train": tr["cosine_sim"],
            "cosine_val":   va["cosine_sim"],
            "purity_gap":   gap,
            "pure_val":     va["pure"],
            "warning":      va["warning"] or (gap > 0.15),
        }
    return report


def log_purity_report(
    report: dict,
    log_path: str | os.PathLike,
    epoch: int,
) -> None:
    """Appends one row per concept to a CSV at log_path."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["epoch", "concept", "cosine_train", "cosine_val",
                             "purity_gap", "pure_val", "warning"])
        for concept, vals in report.items():
            writer.writerow([
                epoch, concept,
                f"{vals['cosine_train']:.6f}",
                f"{vals['cosine_val']:.6f}",
                f"{vals['purity_gap']:.6f}",
                vals["pure_val"],
                vals["warning"],
            ])
