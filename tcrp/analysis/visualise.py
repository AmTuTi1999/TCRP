"""Publication-style figures for TCRP relevance explanations.

Four plots from a single TCRPExplanation:

    Plot 1 — R_x overlaid on raw input x          (shared x-axis)
    Plot 2 — Stacked area of R_x_cond per concept  (concept-conditional maps)
    Plot 3 — R_h bar chart with sign               (concept relevance vector)
    Plot 4 — R_A heatmap (N × K)                  (segment × concept relevance)

Entry point for callers::

    from tcrp.analysis.visualise import plot_explanation
    plot_explanation(explanation, x_batch, concept_names, run_id, h_star, out_dir)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

from tcrp.analysis.tcrp_analysis import TCRPExplanation

# ── concept colour palette ─────────────────────────────────────────────────────
_GROUP_COLORS: dict[str, str] = {
    "trend_direction": "#1f77b4",
    "trend_strength": "#aec7e8",
    "curvature": "#2ca02c",
    "convexity": "#98df8a",
    "stochasticity": "#9467bd",
    "volatility": "#ff7f0e",
    "vol_trend": "#ffbb78",
    "vol_ratio": "#d62728",
    "acf_lag": "#8c564b",
    "mean_reversion": "#e377c2",
    "z_score": "#c49c94",
    "break_mean": "#bcbd22",
    "break_vol": "#dbdb8d",
    "break_trend": "#e6e61a",
    "skewness": "#7f7f7f",
    "kurtosis": "#c7c7c7",
    "jump": "#aaaaaa",
    "period_": "#17becf",
}
_FALLBACK = plt.cm.tab20(np.linspace(0, 1, 20))


def _concept_color(name: str, idx: int) -> str:
    for prefix, color in _GROUP_COLORS.items():
        if name.startswith(prefix):
            return color
    return matplotlib.colors.to_hex(_FALLBACK[idx % 20])


def _darken(hex_color: str, factor: float = 0.65) -> str:
    rgb = matplotlib.colors.to_rgb(hex_color)
    return matplotlib.colors.to_hex(tuple(c * factor for c in rgb))


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    now = datetime.now()
    fig.text(
        0.99,
        0.005,
        now.strftime("%Y-%m-%d  %H:%M:%S"),
        ha="right",
        va="bottom",
        fontsize=6,
        color="#aaaaaa",
        transform=fig.transFigure,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}")


# ── Plot 1: temporal relevance overlaid on raw input ──────────────────────────


def plot_temporal(
    x: np.ndarray,
    R_x: np.ndarray,
    run_id: str,
    h_star: int,
    out_dir: Path,
    label: str | None = None,
) -> None:
    """Plot R_x temporal relevance overlaid on the raw input signal."""
    T = len(x)
    t = np.arange(T)
    tag = label if label is not None else f"h*={h_star}"

    fig, (ax_x, ax_r) = plt.subplots(
        2,
        1,
        figsize=(12, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
    )
    fig.suptitle(f"{run_id} · temporal relevance  ({tag})", fontsize=11, y=1.01)

    ax_x.plot(t, x, color="#2c7bb6", lw=1.2, label="input  x")
    ax_x.set_ylabel("Normalised value", fontsize=9)
    ax_x.legend(fontsize=8, framealpha=0.4)
    ax_x.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

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


# ── Plot 2: stacked area of R_x_cond ──────────────────────────────────────────


def plot_concept_maps(
    R_x_cond: np.ndarray,
    R_x: np.ndarray,
    concept_names: list[str],
    run_id: str,
    h_star: int,
    out_dir: Path,
    top_k: int = 10,
    label: str | None = None,
) -> None:
    """Plot stacked concept-conditional temporal relevance maps.

    Only the top-k concepts by mean absolute contribution are shown individually;
    the remainder are summed into a single grey "other" band so the dominant
    concepts are not obscured by many near-zero traces.
    """
    K, T = R_x_cond.shape
    t = np.arange(T)

    # Rank concepts by mean absolute contribution and keep top_k.
    mean_abs = np.abs(R_x_cond).mean(axis=1)
    top_idx = np.argsort(mean_abs)[::-1][:top_k]
    other_idx = np.argsort(mean_abs)[::-1][top_k:]

    # Build reduced arrays: top concepts + one "other" row.
    top_signals = R_x_cond[top_idx]
    top_names = [concept_names[i] for i in top_idx]
    top_colors = [_concept_color(concept_names[i], i) for i in top_idx]

    if len(other_idx):
        other_signal = R_x_cond[other_idx].sum(axis=0, keepdims=True)
        plot_signals = np.concatenate([top_signals, other_signal], axis=0)
        plot_names = top_names + ["other"]
        plot_colors = top_colors + ["#cccccc"]
    else:
        plot_signals = top_signals
        plot_names = top_names
        plot_colors = top_colors

    tag = label if label is not None else f"h*={h_star}"
    fig, ax = plt.subplots(figsize=(13, 4))
    fig.suptitle(
        f"{run_id} · concept-conditional temporal maps  ({tag}, top {top_k} shown)",
        fontsize=11,
    )

    pos_floor = np.zeros(T)
    for sig, col in zip(plot_signals, plot_colors, strict=False):
        contrib = np.clip(sig, 0, None)
        ax.fill_between(t, pos_floor, pos_floor + contrib, color=col, alpha=0.85)
        pos_floor += contrib

    neg_floor = np.zeros(T)
    for sig, col in zip(plot_signals, plot_colors, strict=False):
        contrib = np.clip(sig, None, 0)
        ax.fill_between(t, neg_floor + contrib, neg_floor, color=col, alpha=0.85)
        neg_floor += contrib

    ax.plot(t, R_x, color="black", lw=1.2, ls="-", label="R_x (total)", zorder=5)
    ax.axhline(0, color="black", lw=0.5, ls="--")

    patches = [
        mpatches.Patch(color=col, label=name)
        for col, name in zip(plot_colors, plot_names, strict=False)
    ]
    ax.legend(
        handles=patches,
        fontsize=7,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        framealpha=0.6,
    )
    ax.set_xlabel("Timestep  t", fontsize=9)
    ax.set_ylabel("R_x_cond", fontsize=9)

    _save(fig, out_dir / "plot2_concept_maps.png")


# ── Plot 3: R_h bar chart ──────────────────────────────────────────────────────


def plot_concept_bar(
    R_h: np.ndarray,
    concept_names: list[str],
    run_id: str,
    h_star: int,
    out_dir: Path,
    label: str | None = None,
) -> None:
    """Plot a bar chart of concept relevance scores R_h with sign colouring."""
    K = len(R_h)
    tag = label if label is not None else f"h*={h_star}"
    colors = [_concept_color(n, i) for i, n in enumerate(concept_names)]
    bar_colors = [c if R_h[i] >= 0 else _darken(c) for i, c in enumerate(colors)]

    fig, ax = plt.subplots(figsize=(max(8, K * 0.55), 4))
    fig.suptitle(f"{run_id} · concept relevance  R_h  ({tag})", fontsize=11)

    x_pos = np.arange(K)
    bars = ax.bar(x_pos, R_h, color=bar_colors, edgecolor="white", linewidth=0.4)

    for bar, val in zip(bars, R_h, strict=False):
        va = "bottom" if val >= 0 else "top"
        offset = 0.002 * (R_h.max() - R_h.min() + 1e-9)
        y = val + (offset if val >= 0 else -offset)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{val:.3f}",
            ha="center",
            va=va,
            fontsize=6,
            rotation=90,
        )

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(concept_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("R_h", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    _save(fig, out_dir / "plot3_concept_bar.png")


# ── Plot 4: R_A heatmap ────────────────────────────────────────────────────────


def plot_segment_heatmap(
    R_A: np.ndarray,
    concept_names: list[str],
    run_id: str,
    h_star: int,
    out_dir: Path,
    label: str | None = None,
) -> None:
    """Plot a diverging heatmap of segment-by-concept relevance scores R_A."""
    N, K = R_A.shape
    tag = label if label is not None else f"h*={h_star}"
    vmax = max(np.abs(R_A).max(), 1e-9)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    max_rows = 80
    if N > max_rows:
        step = N // max_rows
        R_A = R_A[::step]
        y_label = f"Segment  (every {step}th)"
    else:
        y_label = "Segment  n"

    fig_h = max(4, min(12, R_A.shape[0] * 0.18))
    fig, ax = plt.subplots(figsize=(max(8, K * 0.5), fig_h))
    fig.suptitle(f"{run_id} · segment × concept relevance  R_A  ({tag})", fontsize=11)

    im = ax.imshow(
        R_A, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Relevance", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xticks(np.arange(K))
    ax.set_xticklabels(concept_names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(y_label, fontsize=9)
    ax.set_xlabel("Concept  k", fontsize=9)

    ax.set_xticks(np.arange(K) - 0.5, minor=True)
    ax.set_yticks(np.arange(R_A.shape[0]) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    _save(fig, out_dir / "plot4_segment_hmap.png")


# ── Plot 5: segment concept map ───────────────────────────────────────────────


def plot_segment_concept_map(
    x: np.ndarray,
    R_A: np.ndarray,
    starts: np.ndarray,
    L: int,
    concept_names: list[str],
    run_id: str,
    h_star: int,
    out_dir: Path,
    highlight_seg: int | None = None,
    label: str | None = None,
    top_k: int = 5,
) -> None:
    """5×1 subfigure: for each top concept, x(t) with its highest-relevance segment highlighted.

    Each panel corresponds to one of the top-k concepts ranked by mean |R_A|.
    The segment where that concept has the highest |R_A[n, k]| is overlaid as a
    filled, bordered box on the time series so the temporal location is immediately
    visible.

    Args:
        x:             Raw input signal (T,).
        R_A:           Segment × concept relevance (N, K).
        starts:        Integer start index of each segment (N,).
        L:             Segment length in timesteps.
        concept_names: Ordered concept name strings.
        run_id:        String used in titles.
        h_star:        Horizon step being explained.
        out_dir:       Output directory.
        highlight_seg: Unused (kept for call-site compatibility).
        label:         Optional annotation string appended to plot titles.
        top_k:         Number of concepts / panels to draw (default 5).
    """
    T = len(x)
    N, K = R_A.shape
    t = np.arange(T)
    n_panels = min(top_k, K)
    tag = label if label is not None else f"h*={h_star}"

    colors = [_concept_color(n, i) for i, n in enumerate(concept_names)]

    # Top concepts by mean absolute relevance across segments.
    concept_mean_abs = np.abs(R_A).mean(axis=0)
    top_concept_idx = np.argsort(concept_mean_abs)[::-1][:n_panels]

    y_lo = x.min() - 0.05 * (x.max() - x.min())
    y_hi = x.max() + 0.05 * (x.max() - x.min())

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(13, 2.6 * n_panels),
        sharex=True,
        gridspec_kw={"hspace": 0.45},
    )
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        f"{run_id} · segment concept map  ({tag}, top {n_panels})",
        fontsize=11,
        y=1.01,
    )

    for panel, k in enumerate(top_concept_idx):
        ax = axes[panel]
        col = colors[k]

        # Best segment for this concept.
        best_seg = int(np.abs(R_A[:, k]).argmax())
        best_val = R_A[best_seg, k]
        hs = int(starts[best_seg])

        ax.plot(t, x, color="#444444", lw=1.0, zorder=3)

        # Highlight box.
        rect = mpatches.FancyBboxPatch(
            (hs, y_lo),
            L,
            y_hi - y_lo,
            boxstyle="square,pad=0",
            linewidth=1.8,
            edgecolor=col,
            facecolor=col,
            alpha=0.30,
            zorder=4,
        )
        ax.add_patch(rect)
        ax.annotate(
            f"seg {best_seg}  R_A={best_val:+.3f}",
            xy=(hs + L / 2, y_hi),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=col,
            fontweight="bold",
            zorder=5,
        )

        ax.set_xlim(0, T - 1)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("x(t)", fontsize=8)
        ax.set_title(
            f"[{panel + 1}] {concept_names[k]}  "
            f"(mean |R_A|={concept_mean_abs[k]:.3f})",
            fontsize=8.5,
            loc="left",
            color=col,
            fontweight="bold",
        )
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Timestep  t", fontsize=9)

    _save(fig, out_dir / "plot5_segment_concept_map.png")


# ── Plot 6: concept activation vs relevance diagnostic ───────────────────────


def plot_concept_signal_vs_usage(
    C: np.ndarray,
    R_A: np.ndarray,
    concept_names: list[str],
    run_id: str,
    h_star: int,
    out_dir: Path,
    label: str | None = None,
) -> None:
    """Four-panel diagnostic comparing analytic concept signal with model usage.

    A flat C row can mean two opposite things: consistently strong signal
    (high absolute mean, low std) or genuinely absent signal (low mean AND
    low std).  This plot separates those cases.

    Panel 1 — Mean |C| per concept (bar chart):
               How strongly each concept activates on average across segments.
               High bar = concept is consistently present in this window.

    Panel 2 — Std of C per concept (bar chart):
               How much each concept varies across segments.
               High bar = concept is discriminative between segments.
               Low bar does NOT mean absent — it may mean uniformly strong.

    Panel 3 — Raw heatmap of C (N × K, no z-scoring):
               Actual activation values per segment per concept.
               Lets you see whether a flat row is flat-high or flat-zero.

    Panel 4 — Raw heatmap of |R_A| (N × K, globally normalised to [0, 1]):
               LRP relevance per segment per concept, scaled by the global max
               so pale rows are genuinely low-relevance (not just rescaled).
               Compare with Panel 3: if a concept is strong in C but absent
               in R_A the model ignores it; if weak in C but present in R_A
               the model is amplifying a faint signal.

    Args:
        C:             Analytic concept activations (N, K).
        R_A:           LRP segment-concept relevances (N, K).
        concept_names: Ordered concept name strings.
        run_id:        String used in titles.
        h_star:        Horizon step being explained.
        out_dir:       Output directory.
        label:         Optional annotation string appended to plot titles.
    """
    N, K = C.shape
    colors = [_concept_color(n, i) for i, n in enumerate(concept_names)]
    x_pos = np.arange(K)

    mean_C = np.abs(C).mean(axis=0)  # (K,) — average absolute activation
    std_C = C.std(axis=0)  # (K,) — within-window variation

    # Normalise |R_A| by the global max so colours reflect absolute importance.
    # Per-column normalisation would make low-relevance concepts look equally
    # bright as dominant ones (xi, sigma_tilde), which is misleading.
    abs_RA = np.abs(R_A)
    global_max = abs_RA.max()
    RA_norm = abs_RA / (global_max if global_max > 1e-9 else 1.0)

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(max(10, K * 0.55), 16))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1.2, 2.0, 2.0], hspace=0.55)
    ax_mean = fig.add_subplot(gs[0])
    ax_std = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])
    ax_r = fig.add_subplot(gs[3], sharex=ax_c)

    tag = label if label is not None else f"h*={h_star}"
    fig.suptitle(f"{run_id} · concept signal vs model usage  ({tag})", fontsize=11)

    # ── Panel 1: mean |C| ─────────────────────────────────────────────────────
    ax_mean.bar(x_pos, mean_C, color=colors, edgecolor="white", linewidth=0.4)
    ax_mean.set_xticks(x_pos)
    ax_mean.set_xticklabels(concept_names, rotation=45, ha="right", fontsize=7)
    ax_mean.set_ylabel("Mean |C|", fontsize=8)
    ax_mean.set_title(
        "Mean absolute activation per concept  —  high = consistently present in this window",
        fontsize=8,
    )

    # ── Panel 2: std of C ─────────────────────────────────────────────────────
    ax_std.bar(x_pos, std_C, color=colors, edgecolor="white", linewidth=0.4)
    ax_std.set_xticks(x_pos)
    ax_std.set_xticklabels(concept_names, rotation=45, ha="right", fontsize=7)
    ax_std.set_ylabel("Std of C", fontsize=8)
    ax_std.set_title(
        "Within-window variation per concept  —  low std + high mean = uniform strong signal  "
        "(not absent)",
        fontsize=8,
    )

    # ── Panel 3: raw C heatmap ────────────────────────────────────────────────
    vmax_c = np.abs(C).max()
    norm_c = (
        TwoSlopeNorm(vmin=-vmax_c, vcenter=0.0, vmax=vmax_c) if vmax_c > 0 else None
    )
    im_c = ax_c.imshow(
        C.T,
        aspect="auto",
        cmap="RdBu_r",
        norm=norm_c,
        interpolation="nearest",
    )
    fig.colorbar(im_c, ax=ax_c, fraction=0.015, pad=0.01, label="C value")
    ax_c.set_yticks(x_pos)
    ax_c.set_yticklabels(concept_names, fontsize=7)
    ax_c.set_xlabel("Segment  n", fontsize=8)
    ax_c.set_title(
        "Raw C heatmap  —  flat-bright row = uniform strong signal;  "
        "flat-white row = absent signal",
        fontsize=8,
    )

    # ── Panel 4: column-normalised |R_A| heatmap ─────────────────────────────
    im_r = ax_r.imshow(
        RA_norm.T,
        aspect="auto",
        cmap="Oranges",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    fig.colorbar(im_r, ax=ax_r, fraction=0.015, pad=0.01, label="|R_A| (col-norm)")
    ax_r.set_yticks(x_pos)
    ax_r.set_yticklabels(concept_names, fontsize=7)
    ax_r.set_xlabel("Segment  n", fontsize=8)
    ax_r.set_title(
        "LRP relevances |R_A| (globally normalised to max)  —  "
        "pale = low absolute relevance;  compare rows with Panel 3 above",
        fontsize=8,
    )

    _save(fig, out_dir / "plot6_signal_vs_usage.png")


# ── Plot 7: concept values across segments (line plot) ───────────────────────


def plot_concept_lines(
    C: np.ndarray,
    concept_names: list[str],
    run_id: str,
    h_star: int,
    out_dir: Path,
    top_k: int = 10,
    label: str | None = None,
) -> None:
    """Line plot of analytic concept values C[n, k] across segments.

    One line per concept; x-axis is segment index n.  Only the top-k concepts
    by mean absolute value are drawn individually — the rest are omitted to keep
    the chart readable.

    Args:
        C:             Concept activations (N, K).
        concept_names: Ordered concept name strings.
        run_id:        String used in titles.
        h_star:        Horizon step / class index (used in title when label is None).
        out_dir:       Output directory.
        top_k:         Maximum number of concepts to draw individually.
        label:         Optional annotation string appended to the plot title.
    """
    N, K = C.shape
    seg_idx = np.arange(N)
    tag = label if label is not None else f"h*={h_star}"

    mean_abs = np.abs(C).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_k]

    fig, ax = plt.subplots(figsize=(max(8, N * 0.15 + 2), 4))
    fig.suptitle(
        f"{run_id} · concept values across segments  ({tag}"
        + (f", top {top_k} of {K}" if K > top_k else "")
        + ")",
        fontsize=11,
    )

    for k in top_idx:
        color = _concept_color(concept_names[k], int(k))
        ax.plot(
            seg_idx,
            C[:, k],
            lw=1.2,
            color=color,
            label=concept_names[k],
            alpha=0.85,
        )

    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_xlabel("Segment  n", fontsize=9)
    ax.set_ylabel("Concept value  C[n, k]", fontsize=9)
    ax.legend(
        fontsize=7,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        framealpha=0.6,
    )
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    _save(fig, out_dir / "plot7_concept_lines.png")


# ── Plot 8: top concepts at different segments ────────────────────────────────


def plot_top_segment_concepts(
    x: np.ndarray,
    R_A: np.ndarray,
    starts: np.ndarray,
    L: int,
    concept_names: list[str],
    run_id: str,
    h_star: int,
    out_dir: Path,
    top_k: int = 5,
    label: str | None = None,
) -> None:
    """5×1 subfigure: for each top segment, x(t) with the segment window highlighted.

    Selects the top_k most relevant segments by total |R_A| and shows the full
    time series in each panel with the segment window overlaid as a coloured box
    whose colour reflects that segment's dominant concept.

    Args:
        x:             Raw input signal (T,).
        R_A:           Segment × concept relevance (N, K).
        starts:        Integer start index of each segment (N,).
        L:             Segment length in timesteps.
        concept_names: Ordered concept name strings.
        run_id:        String used in titles.
        h_star:        Horizon step / class index.
        out_dir:       Output directory.
        top_k:         Number of top segments / panels (default 5).
        label:         Optional annotation string appended to plot titles.
    """
    T = len(x)
    N, K = R_A.shape
    t = np.arange(T)
    n_panels = min(top_k, N)
    tag = label if label is not None else f"h*={h_star}"

    seg_total_abs = np.abs(R_A).sum(axis=1)
    top_seg_idx = np.argsort(seg_total_abs)[::-1][:n_panels]

    colors = [_concept_color(n, i) for i, n in enumerate(concept_names)]

    y_lo = x.min() - 0.05 * (x.max() - x.min())
    y_hi = x.max() + 0.05 * (x.max() - x.min())

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(13, 2.6 * n_panels),
        sharex=True,
        gridspec_kw={"hspace": 0.45},
    )
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        f"{run_id} · segment concept maps  ({tag}, top {n_panels} segments)",
        fontsize=11,
        y=1.01,
    )

    for panel, seg in enumerate(top_seg_idx):
        ax = axes[panel]
        dom_k = int(np.abs(R_A[seg]).argmax())
        dom_val = R_A[seg, dom_k]
        col = colors[dom_k]
        hs = int(starts[seg])

        ax.plot(t, x, color="#444444", lw=1.0, zorder=3)

        rect = mpatches.FancyBboxPatch(
            (hs, y_lo),
            L,
            y_hi - y_lo,
            boxstyle="square,pad=0",
            linewidth=1.8,
            edgecolor=col,
            facecolor=col,
            alpha=0.30,
            zorder=4,
        )
        ax.add_patch(rect)
        ax.annotate(
            f"seg {seg}  t={hs}–{hs + L - 1}  R_A={dom_val:+.3f}",
            xy=(hs + L / 2, y_hi),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=col,
            fontweight="bold",
            zorder=5,
        )

        ax.set_xlim(0, T - 1)
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel("x(t)", fontsize=8)
        ax.set_title(
            f"[{panel + 1}] Segment {seg}  "
            f"(total |R_A|={seg_total_abs[seg]:.3f})  ·  "
            f"dominant: {concept_names[dom_k]}",
            fontsize=8.5,
            loc="left",
            color=col,
            fontweight="bold",
        )
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Timestep  t", fontsize=9)

    _save(fig, out_dir / "plot8_top_segment_concepts.png")


# ── High-level entry point ─────────────────────────────────────────────────────


def plot_explanation(
    explanation: TCRPExplanation,
    x_batch: torch.Tensor,
    concept_names: list[str],
    run_id: str,
    h_star: int,
    out_dir: Path,
    sample_idx: int = 0,
    highlight_seg: int | None = None,
    label: str | None = None,
) -> Path:
    """Produce all six TCRP analysis figures for one sample.

    Args:
        explanation:   TCRPExplanation returned by TCRPAnalyser.analyse().
        x_batch:       Raw input batch (B, T); sample_idx selects the sample.
        concept_names: Ordered list of concept name strings.
        run_id:        String identifier used in plot titles and the output path.
        h_star:        Horizon step (forecasting) or class index (classification),
                       used only when ``label`` is not given.
        out_dir:       Directory where PNGs are written (created if absent).
        sample_idx:    Which sample in the batch to visualise (default 0).
        highlight_seg: Segment to drill down on in plot 5 (default: highest relevance).
        label:         Override the ``h*=…`` / ``k*=…`` tag in all plot titles.
                       E.g. ``"k*=2 (V-class)"`` for a classification explanation.

    Returns:
        out_dir as a resolved Path.
    """
    out_dir = Path(out_dir)

    x_np = _np(x_batch[sample_idx])
    R_x = _np(explanation.R_x[sample_idx])
    R_x_cond = _np(explanation.R_x_cond[sample_idx])
    R_h = _np(explanation.R_h[sample_idx])
    R_A = _np(explanation.R_A[sample_idx])
    C = _np(explanation.C[sample_idx])
    starts = _np(explanation.starts).astype(int)
    L = explanation.L

    tag = label if label is not None else f"h*={h_star}"
    print(f"\nVisualising sample {sample_idx}  |  run={run_id}  |  {tag}")
    print(f"  x        {x_np.shape}   [{x_np.min():.3f}, {x_np.max():.3f}]")
    print(f"  R_x      {R_x.shape}    [{R_x.min():.3e}, {R_x.max():.3e}]")
    print(f"  R_x_cond {R_x_cond.shape}")
    print(f"  R_h      {R_h.shape}")
    print(f"  R_A      {R_A.shape}")
    print(f"  C        {C.shape}")
    print(f"  output   {out_dir}/\n")

    plot_temporal(x_np, R_x, run_id, h_star, out_dir, label=label)
    plot_concept_maps(
        R_x_cond, R_x, concept_names, run_id, h_star, out_dir, label=label
    )
    plot_concept_bar(R_h, concept_names, run_id, h_star, out_dir, label=label)
    plot_segment_heatmap(R_A, concept_names, run_id, h_star, out_dir, label=label)
    plot_segment_concept_map(
        x_np,
        R_A,
        starts,
        L,
        concept_names,
        run_id,
        h_star,
        out_dir,
        highlight_seg=highlight_seg,
        label=label,
    )
    plot_concept_signal_vs_usage(
        C, R_A, concept_names, run_id, h_star, out_dir, label=label
    )
    plot_concept_lines(C, concept_names, run_id, h_star, out_dir, label=label)
    plot_top_segment_concepts(
        x_np, R_A, starts, L, concept_names, run_id, h_star, out_dir, label=label
    )

    return out_dir
