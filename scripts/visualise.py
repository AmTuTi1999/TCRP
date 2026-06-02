"""T-28 · TCRP analysis visualiser

Four publication-style figures from a single look-back window:

    Plot 1 — R_x overlaid on raw input x          (shared x-axis)
    Plot 2 — Stacked area of R_x_cond per concept  (concept-conditional temporal maps)
    Plot 3 — R_h bar chart with sign               (concept relevance vector)
    Plot 4 — R_A heatmap (N × K)                  (segment × concept relevance)

Usage (run from project root):
    python scripts/visualise.py \\
        --config  tcrp/pipelines/configs/etth1.yaml \\
        [--checkpoint checkpoints/ETTh1_T336_H96_best.pt] \\
        [--sample-idx 0] \\
        [--h-star    0] \\
        [--run-id    my_run]

Output is written to  figures/{run_id}/  as four PNG files.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# ── headless-safe backend ─────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from tcrp.analysis.tcrp_analysis import TCRPAnalyser
from tcrp.dataset.datasets import DATASET_META
from tcrp.model.forecaster import TCRPConfig, TCRPForecaster
from tcrp.pipelines.config import PipelineConfig, load_config
from tcrp.pipelines.train import build_loaders

# ── concept colour palette ────────────────────────────────────────────────────
# Colours are keyed on concept-name *prefixes* that match the canonical ordering
# defined in ConceptScorer.concept_names (see tcrp/concepts/concept_vector.py).
_GROUP_COLORS: dict[str, str] = {
    "mu_signed":   "#1f77b4",   # trend — blue
    "mu_mag":      "#aec7e8",
    "kappa_signed":"#2ca02c",   # curvature — green
    "tau":         "#98df8a",
    "xi":          "#9467bd",   # stochasticity — purple
    "sigma_tilde": "#ff7f0e",   # volatility — orange
    "mu_v":        "#ffbb78",
    "psi":         "#d62728",
    "theta_hat":   "#8c564b",   # mean reversion — brown
    "z":           "#c49c94",
    "b_mu_tilde":  "#bcbd22",   # breaks — olive (must come before b_mu)
    "b_mu":        "#e6e61a",
    "b_sigma":     "#dbdb8d",
    "varsigma":    "#7f7f7f",   # shape — grey
    "kappa4":      "#c7c7c7",
    "j":           "#aaaaaa",
}
# ACF lags and periodicity terms fall back to a tab20 slice
_FALLBACK = plt.cm.tab20(np.linspace(0, 1, 20))


def _concept_color(name: str, idx: int) -> str:
    """Return a hex colour for a concept by name, falling back to tab20."""
    for prefix, color in _GROUP_COLORS.items():
        if name.startswith(prefix):
            return color
    return matplotlib.colors.to_hex(_FALLBACK[idx % 20])


# ── helper: strip gradient wrappers ──────────────────────────────────────────

def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


# ── plot helpers ──────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


# ── Plot 1: temporal relevance overlaid on raw input ─────────────────────────

def plot_temporal(
    x: np.ndarray,          # (T,)
    R_x: np.ndarray,        # (T,)
    run_id: str,
    h_star: int,
    out_dir: Path,
) -> None:
    T = len(x)
    t = np.arange(T)

    fig, (ax_x, ax_r) = plt.subplots(
        2, 1, figsize=(12, 5), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
    )
    fig.suptitle(
        f"{run_id} · temporal relevance  (h*={h_star})",
        fontsize=11, y=1.01,
    )

    # — top: raw input —
    ax_x.plot(t, x, color="#2c7bb6", lw=1.2, label="input  x")
    ax_x.set_ylabel("Normalised value", fontsize=9)
    ax_x.legend(fontsize=8, framealpha=0.4)
    ax_x.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # — bottom: R_x fill (positive/negative) —
    pos = np.clip(R_x, 0, None)
    neg = np.clip(R_x, None, 0)
    ax_r.fill_between(t, 0, pos, color="#2c7bb6", alpha=0.75, label="positive R_x")
    ax_r.fill_between(t, neg, 0, color="#d7191c", alpha=0.75, label="negative R_x")
    ax_r.axhline(0, color="black", lw=0.6, ls="--")
    ax_r.set_ylabel("R_x", fontsize=9)
    ax_r.set_xlabel("Timestep  t", fontsize=9)
    ax_r.legend(fontsize=8, framealpha=0.4)
    ax_r.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    _save(fig, out_dir / "plot1_temporal.png")


# ── Plot 2: stacked area of R_x_cond ─────────────────────────────────────────

def plot_concept_maps(
    R_x_cond: np.ndarray,   # (K, T)
    R_x: np.ndarray,        # (T,)
    concept_names: List[str],
    run_id: str,
    h_star: int,
    out_dir: Path,
) -> None:
    K, T = R_x_cond.shape
    t = np.arange(T)
    colors = [_concept_color(n, i) for i, n in enumerate(concept_names)]

    fig, ax = plt.subplots(figsize=(13, 4))
    fig.suptitle(
        f"{run_id} · concept-conditional temporal maps  (h*={h_star})",
        fontsize=11,
    )

    # Stacked positive contributions (upward)
    pos_floor = np.zeros(T)
    for k in range(K):
        contrib = np.clip(R_x_cond[k], 0, None)
        ax.fill_between(t, pos_floor, pos_floor + contrib,
                        color=colors[k], alpha=0.85)
        pos_floor += contrib

    # Stacked negative contributions (downward)
    neg_floor = np.zeros(T)
    for k in range(K):
        contrib = np.clip(R_x_cond[k], None, 0)
        ax.fill_between(t, neg_floor + contrib, neg_floor,
                        color=colors[k], alpha=0.85)
        neg_floor += contrib

    # Total R_x line on top
    ax.plot(t, R_x, color="black", lw=1.2, ls="-", label="R_x (total)", zorder=5)
    ax.axhline(0, color="black", lw=0.5, ls="--")

    # Legend: one patch per concept, placed outside to the right
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=colors[k], label=concept_names[k]) for k in range(K)]
    ax.legend(
        handles=patches,
        fontsize=6.5,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        framealpha=0.6,
    )
    ax.set_xlabel("Timestep  t", fontsize=9)
    ax.set_ylabel("R_x_cond", fontsize=9)

    _save(fig, out_dir / "plot2_concept_maps.png")


# ── Plot 3: R_h bar chart ─────────────────────────────────────────────────────

def plot_concept_bar(
    R_h: np.ndarray,        # (K,)
    concept_names: List[str],
    run_id: str,
    h_star: int,
    out_dir: Path,
) -> None:
    K = len(R_h)
    colors = [_concept_color(n, i) for i, n in enumerate(concept_names)]
    bar_colors = [
        c if R_h[i] >= 0 else _darken(c)
        for i, c in enumerate(colors)
    ]

    fig, ax = plt.subplots(figsize=(max(8, K * 0.55), 4))
    fig.suptitle(
        f"{run_id} · concept relevance  R_h  (h*={h_star})",
        fontsize=11,
    )

    x_pos = np.arange(K)
    bars = ax.bar(x_pos, R_h, color=bar_colors, edgecolor="white", linewidth=0.4)

    # Value annotations
    for bar, val in zip(bars, R_h):
        va = "bottom" if val >= 0 else "top"
        offset = 0.002 * (R_h.max() - R_h.min() + 1e-9)
        y = val + (offset if val >= 0 else -offset)
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f"{val:.3f}", ha="center", va=va, fontsize=6, rotation=90)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(concept_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("R_h", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    _save(fig, out_dir / "plot3_concept_bar.png")


def _darken(hex_color: str, factor: float = 0.65) -> str:
    """Return a darkened version of a hex colour (for negative bars)."""
    rgb = matplotlib.colors.to_rgb(hex_color)
    return matplotlib.colors.to_hex(tuple(c * factor for c in rgb))


# ── Plot 4: R_A heatmap ───────────────────────────────────────────────────────

def plot_segment_heatmap(
    R_A: np.ndarray,        # (N, K)
    concept_names: List[str],
    run_id: str,
    h_star: int,
    out_dir: Path,
) -> None:
    N, K = R_A.shape
    vmax = max(np.abs(R_A).max(), 1e-9)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # Collapse very long segment axes for readability
    max_rows = 80
    if N > max_rows:
        step = N // max_rows
        R_A = R_A[::step]
        y_label = f"Segment  (every {step}th)"
    else:
        y_label = "Segment  n"

    fig_h = max(4, min(12, R_A.shape[0] * 0.18))
    fig, ax = plt.subplots(figsize=(max(8, K * 0.5), fig_h))
    fig.suptitle(
        f"{run_id} · segment × concept relevance  R_A  (h*={h_star})",
        fontsize=11,
    )

    im = ax.imshow(R_A, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Relevance", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xticks(np.arange(K))
    ax.set_xticklabels(concept_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_xlabel("Concept  k", fontsize=9)

    # Thin grid lines
    ax.set_xticks(np.arange(K) - 0.5, minor=True)
    ax.set_yticks(np.arange(R_A.shape[0]) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    _save(fig, out_dir / "plot4_segment_hmap.png")


# ── checkpoint / data discovery ───────────────────────────────────────────────

def _find_checkpoint(cfg: PipelineConfig, explicit: Optional[str]) -> str:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {explicit}")
        return str(p)
    run_name = cfg.run_name or f"{cfg.dataset}_T{cfg.T}_H{cfg.H}"
    candidate = Path(cfg.checkpoint_dir) / f"{run_name}_best.pt"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(
        f"No checkpoint found at {candidate}.\n"
        f"Train first:  python -m tcrp.pipelines.train --config <yaml>\n"
        f"or pass      --checkpoint <path>"
    )


def _check_data(cfg: PipelineConfig) -> None:
    meta = DATASET_META.get(cfg.dataset)
    if meta is None:
        raise ValueError(f"Unknown dataset '{cfg.dataset}'")
    csv_path = Path(cfg.data_root) / meta["filename"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found: {csv_path}\n"
            f"Download {cfg.dataset} and place it at that path."
        )


# ── main pipeline ─────────────────────────────────────────────────────────────

def visualise(
    cfg: PipelineConfig,
    checkpoint: str,
    sample_idx: int = 0,
    h_star: int = 0,
    run_id: Optional[str] = None,
) -> Path:
    """Run analysis and produce all four figures.

    Returns the output directory path.
    """
    _check_data(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = run_id or cfg.run_name or f"{cfg.dataset}_T{cfg.T}_H{cfg.H}"
    out_dir = Path("figures") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────────────
    _, _, test_loader, _, _ = build_loaders(cfg)
    x_batch, _ = next(iter(test_loader))           # (B, T)

    if sample_idx >= x_batch.shape[0]:
        raise IndexError(
            f"--sample-idx {sample_idx} >= batch size {x_batch.shape[0]}. "
            f"Use a value in [0, {x_batch.shape[0] - 1}]."
        )

    # ── model ─────────────────────────────────────────────────────────────────
    model_cfg = TCRPConfig(
        T=cfg.T, H=cfg.H, L=cfg.L, stride=cfg.stride,
        d=cfg.d, periods=cfg.periods, k_max=cfg.k_max,
        alpha=cfg.alpha, beta=cfg.beta,
        lambda1=cfg.lambda1, lambda2=cfg.lambda2,
        probabilistic=cfg.probabilistic,
    )
    model = TCRPForecaster(model_cfg).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()

    concept_names: List[str] = model.scorer.concept_names

    if h_star < 0 or h_star >= cfg.H:
        raise ValueError(f"--h-star {h_star} out of range [0, {cfg.H})")

    # ── analysis (single sample only — LRP is memory-intensive) ───────────────
    x_single = x_batch[sample_idx].unsqueeze(0).to(device)  # (1, T)
    analyser = TCRPAnalyser(model, eps=1e-6)
    expl = analyser.analyse(x_single, h_star=h_star)

    x_np     = _np(x_batch[sample_idx])           # (T,)
    R_x      = _np(expl.R_x[0])                   # (T,)
    R_x_cond = _np(expl.R_x_cond[0])              # (K, T)
    R_h      = _np(expl.R_h[0])                   # (K,)
    R_A      = _np(expl.R_A[0])                   # (N, K)

    print(f"\nVisualising sample {sample_idx}  |  run={run_id}  |  h*={h_star}")
    print(f"  x        {x_np.shape}   [{x_np.min():.3f}, {x_np.max():.3f}]")
    print(f"  R_x      {R_x.shape}    [{R_x.min():.3e}, {R_x.max():.3e}]")
    print(f"  R_x_cond {R_x_cond.shape}")
    print(f"  R_h      {R_h.shape}")
    print(f"  R_A      {R_A.shape}")
    print(f"  output   {out_dir}/\n")

    # ── four plots ────────────────────────────────────────────────────────────
    plot_temporal(x_np, R_x, run_id, h_star, out_dir)
    plot_concept_maps(R_x_cond, R_x, concept_names, run_id, h_star, out_dir)
    plot_concept_bar(R_h, concept_names, run_id, h_star, out_dir)
    plot_segment_heatmap(R_A, concept_names, run_id, h_star, out_dir)

    return out_dir


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TCRP analysis visualiser — four publication figures from a checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",      required=True,
                        help="Path to YAML pipeline config")
    parser.add_argument("--checkpoint",  default=None,
                        help="Path to .pt checkpoint (auto-discovered if omitted)")
    parser.add_argument("--sample-idx",  type=int, default=0,
                        help="Index of test sample to visualise")
    parser.add_argument("--h-star",      type=int, default=0,
                        help="Forecast horizon step to explain (0-based)")
    parser.add_argument("--run-id",      default=None,
                        help="Output folder name under figures/ (derived from config if omitted)")
    args = parser.parse_args()

    cfg        = load_config(args.config)
    checkpoint = _find_checkpoint(cfg, args.checkpoint)

    out_dir = visualise(
        cfg, checkpoint,
        sample_idx=args.sample_idx,
        h_star=args.h_star,
        run_id=args.run_id,
    )
    print(f"Done. All figures in {out_dir}/")


if __name__ == "__main__":
    main()
