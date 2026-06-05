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
    "mu_signed": "#1f77b4",
    "mu_mag": "#aec7e8",
    "kappa_signed": "#2ca02c",
    "tau": "#98df8a",
    "xi": "#9467bd",
    "sigma_tilde": "#ff7f0e",
    "mu_v": "#ffbb78",
    "psi": "#d62728",
    "theta_hat": "#8c564b",
    "z": "#c49c94",
    "b_mu_tilde": "#bcbd22",
    "b_mu": "#e6e61a",
    "b_sigma": "#dbdb8d",
    "varsigma": "#7f7f7f",
    "kappa4": "#c7c7c7",
    "j": "#aaaaaa",
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
) -> None:
    """Plot R_x temporal relevance overlaid on the raw input signal."""
    T = len(x)
    t = np.arange(T)

    fig, (ax_x, ax_r) = plt.subplots(
        2,
        1,
        figsize=(12, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
    )
    fig.suptitle(f"{run_id} · temporal relevance  (h*={h_star})", fontsize=11, y=1.01)

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
) -> None:
    """Plot stacked concept-conditional temporal relevance maps."""
    K, T = R_x_cond.shape
    t = np.arange(T)
    colors = [_concept_color(n, i) for i, n in enumerate(concept_names)]

    fig, ax = plt.subplots(figsize=(13, 4))
    fig.suptitle(
        f"{run_id} · concept-conditional temporal maps  (h*={h_star})", fontsize=11
    )

    pos_floor = np.zeros(T)
    for k in range(K):
        contrib = np.clip(R_x_cond[k], 0, None)
        ax.fill_between(t, pos_floor, pos_floor + contrib, color=colors[k], alpha=0.85)
        pos_floor += contrib

    neg_floor = np.zeros(T)
    for k in range(K):
        contrib = np.clip(R_x_cond[k], None, 0)
        ax.fill_between(t, neg_floor + contrib, neg_floor, color=colors[k], alpha=0.85)
        neg_floor += contrib

    ax.plot(t, R_x, color="black", lw=1.2, ls="-", label="R_x (total)", zorder=5)
    ax.axhline(0, color="black", lw=0.5, ls="--")

    patches = [
        mpatches.Patch(color=colors[k], label=concept_names[k]) for k in range(K)
    ]
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


# ── Plot 3: R_h bar chart ──────────────────────────────────────────────────────


def plot_concept_bar(
    R_h: np.ndarray,
    concept_names: list[str],
    run_id: str,
    h_star: int,
    out_dir: Path,
) -> None:
    """Plot a bar chart of concept relevance scores R_h with sign colouring."""
    K = len(R_h)
    colors = [_concept_color(n, i) for i, n in enumerate(concept_names)]
    bar_colors = [c if R_h[i] >= 0 else _darken(c) for i, c in enumerate(colors)]

    fig, ax = plt.subplots(figsize=(max(8, K * 0.55), 4))
    fig.suptitle(f"{run_id} · concept relevance  R_h  (h*={h_star})", fontsize=11)

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
) -> None:
    """Plot a diverging heatmap of segment-by-concept relevance scores R_A."""
    N, K = R_A.shape
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
    fig.suptitle(
        f"{run_id} · segment × concept relevance  R_A  (h*={h_star})", fontsize=11
    )

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
) -> None:
    """Two-panel figure showing dominant concept per time region and a segment drill-down.

    Panel 1 — time series x(t) with background coloured by the dominant concept
    at each timestep (weighted vote over all segments covering that step).
    The selected segment is overlaid with a bold rectangle.

    Panel 2 — horizontal bar chart of R_A[highlight_seg, :] showing all concept
    relevances for the chosen segment; the dominant concept bar is outlined.

    Args:
        x:             Raw input signal (T,).
        R_A:           Segment × concept relevance (N, K).
        starts:        Integer start index of each segment (N,).
        L:             Segment length in timesteps.
        concept_names: Ordered concept name strings.
        run_id:        String used in titles.
        h_star:        Horizon step being explained.
        out_dir:       Output directory.
        highlight_seg: Segment index to drill down on.  Defaults to the segment
                       with the highest total absolute relevance.
    """
    T = len(x)
    N, K = R_A.shape
    t = np.arange(T)

    # ── per-timestep dominant concept ─────────────────────────────────────────
    # For each timestep accumulate |R_A[n, k]| from every segment covering it.
    C_t = np.zeros((T, K), dtype=np.float32)
    for n in range(N):
        s = int(starts[n])
        C_t[s : s + L] += np.abs(R_A[n])

    covered = C_t.sum(axis=1) > 0  # timesteps touched by at least one segment
    dominant = C_t.argmax(axis=1)  # (T,) index of dominant concept

    # ── choose segment to highlight ───────────────────────────────────────────
    if highlight_seg is None:
        highlight_seg = int(np.abs(R_A).sum(axis=1).argmax())

    colors = [_concept_color(n, i) for i, n in enumerate(concept_names)]

    # ── figure layout ─────────────────────────────────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(13, 7), gridspec_kw={"height_ratios": [3, 2], "hspace": 0.35}
    )
    fig.suptitle(f"{run_id} · segment concept map  (h*={h_star})", fontsize=11, y=1.01)

    # ── Panel 1: time series + coloured background ────────────────────────────
    # Draw one axvspan per run of the same dominant concept.
    if covered.any():
        prev_t = 0
        prev_c = dominant[0]
        for ti in range(1, T + 1):
            cur_c = dominant[ti] if ti < T and covered[ti] else -1
            if ti == T or cur_c != prev_c or not covered[ti]:
                if covered[prev_t]:
                    ax_top.axvspan(
                        prev_t, ti - 1, alpha=0.22, color=colors[prev_c], linewidth=0
                    )
                prev_t = ti
                prev_c = cur_c if ti < T else -1

    ax_top.plot(t, x, color="black", lw=1.1, zorder=3, label="input  x")

    # Highlight the chosen segment with a bold box.
    hs = int(starts[highlight_seg])
    y_lo, y_hi = ax_top.get_ylim()
    # Use data range for the box height (recomputed after plotting).
    y_lo = x.min() - 0.05 * (x.max() - x.min())
    y_hi = x.max() + 0.05 * (x.max() - x.min())
    dom_k = int(np.abs(R_A[highlight_seg]).argmax())
    rect = mpatches.FancyBboxPatch(
        (hs, y_lo),
        L,
        y_hi - y_lo,
        boxstyle="square,pad=0",
        linewidth=2,
        edgecolor=colors[dom_k],
        facecolor=colors[dom_k],
        alpha=0.35,
        zorder=4,
    )
    ax_top.add_patch(rect)
    ax_top.annotate(
        f"seg {highlight_seg}",
        xy=(hs + L / 2, y_hi),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        fontsize=7,
        color=colors[dom_k],
        zorder=5,
    )
    ax_top.set_xlim(0, T - 1)
    ax_top.set_ylim(y_lo, y_hi)
    ax_top.set_ylabel("Normalised value", fontsize=9)
    ax_top.set_xlabel("Timestep  t", fontsize=9)

    # Legend: one patch per concept that actually appears as dominant.
    dominant_set = sorted(set(dominant[covered]))
    legend_patches = [
        mpatches.Patch(color=colors[k], label=concept_names[k], alpha=0.7)
        for k in dominant_set
    ]
    ax_top.legend(
        handles=legend_patches,
        fontsize=6.5,
        ncol=max(1, len(dominant_set) // 4),
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        framealpha=0.6,
        title="Dominant\nconcept",
        title_fontsize=6.5,
    )

    # ── Panel 2: concept breakdown for highlighted segment ────────────────────
    seg_R = R_A[highlight_seg]  # (K,)
    bar_colors = [colors[k] for k in range(K)]
    y_pos = np.arange(K)

    bars = ax_bot.barh(y_pos, seg_R, color=bar_colors, edgecolor="white", linewidth=0.4)

    # Outline the dominant-concept bar.
    bars[dom_k].set_edgecolor("black")
    bars[dom_k].set_linewidth(1.8)

    # Value annotations.
    for bar, val in zip(bars, seg_R, strict=False):
        ha = "left" if val >= 0 else "right"
        offset = 0.002 * (np.abs(seg_R).max() + 1e-9)
        x_ann = val + (offset if val >= 0 else -offset)
        ax_bot.text(
            x_ann,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha=ha,
            fontsize=6.5,
        )

    ax_bot.axvline(0, color="black", lw=0.8)
    ax_bot.set_yticks(y_pos)
    ax_bot.set_yticklabels(concept_names, fontsize=7)
    ax_bot.set_xlabel("R_A  (concept relevance)", fontsize=9)
    ax_bot.set_title(
        f"Segment {highlight_seg}  (t={hs}–{hs + L - 1})  ·  "
        f"dominant: {concept_names[dom_k]}",
        fontsize=9,
    )

    _save(fig, out_dir / "plot5_segment_concept_map.png")


# ── Plot 6: concept activation vs relevance diagnostic ───────────────────────


def plot_concept_signal_vs_usage(
    C: np.ndarray,
    R_A: np.ndarray,
    concept_names: list[str],
    run_id: str,
    h_star: int,
    out_dir: Path,
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

    Panel 4 — Raw heatmap of |R_A| (N × K, column-normalised to [0, 1]):
               LRP relevance per segment per concept, scaled for comparison.
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
    """
    N, K = C.shape
    colors = [_concept_color(n, i) for i, n in enumerate(concept_names)]
    x_pos = np.arange(K)

    mean_C = np.abs(C).mean(axis=0)  # (K,) — average absolute activation
    std_C = C.std(axis=0)  # (K,) — within-window variation

    # Column-normalise |R_A| to [0, 1] so the heatmap is readable regardless
    # of the overall relevance scale.
    abs_RA = np.abs(R_A)
    col_max = abs_RA.max(axis=0, keepdims=True)
    RA_norm = abs_RA / np.where(col_max < 1e-9, 1.0, col_max)

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(max(10, K * 0.55), 16))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1.2, 2.0, 2.0], hspace=0.55)
    ax_mean = fig.add_subplot(gs[0])
    ax_std = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])
    ax_r = fig.add_subplot(gs[3], sharex=ax_c)

    fig.suptitle(
        f"{run_id} · concept signal vs model usage  (h*={h_star})", fontsize=11
    )

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
        "LRP relevances |R_A| (each concept scaled to its own max)  —  "
        "compare rows with Panel 3 above",
        fontsize=8,
    )

    _save(fig, out_dir / "plot6_signal_vs_usage.png")


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
) -> Path:
    """Produce all five TCRP analysis figures for one sample.

    Args:
        explanation:   TCRPExplanation returned by TCRPAnalyser.analyse().
        x_batch:       Raw input batch (B, T); sample_idx selects the sample.
        concept_names: Ordered list of concept name strings.
        run_id:        String identifier used in plot titles and the output path.
        h_star:        Horizon step that was explained (0-based).
        out_dir:       Directory where PNGs are written (created if absent).
        sample_idx:    Which sample in the batch to visualise (default 0).
        highlight_seg: Segment to drill down on in plot 5 (default: highest relevance).

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

    print(f"\nVisualising sample {sample_idx}  |  run={run_id}  |  h*={h_star}")
    print(f"  x        {x_np.shape}   [{x_np.min():.3f}, {x_np.max():.3f}]")
    print(f"  R_x      {R_x.shape}    [{R_x.min():.3e}, {R_x.max():.3e}]")
    print(f"  R_x_cond {R_x_cond.shape}")
    print(f"  R_h      {R_h.shape}")
    print(f"  R_A      {R_A.shape}")
    print(f"  C        {C.shape}")
    print(f"  output   {out_dir}/\n")

    plot_temporal(x_np, R_x, run_id, h_star, out_dir)
    plot_concept_maps(R_x_cond, R_x, concept_names, run_id, h_star, out_dir)
    plot_concept_bar(R_h, concept_names, run_id, h_star, out_dir)
    plot_segment_heatmap(R_A, concept_names, run_id, h_star, out_dir)
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
    )
    plot_concept_signal_vs_usage(C, R_A, concept_names, run_id, h_star, out_dir)

    return out_dir
