"""
viz.py
------
Shared plotting helpers for all notebooks.

All functions return a matplotlib Figure so you can either:
  - display inline:  fig = plot_X(...); plt.show()
  - save to disk:    fig.savefig("results/figures/exp1_reconstruction.png", dpi=150)

Plots defined here:

  A. Refusal direction
       plot_direction_norms()       — how strong is r at each layer
       plot_layer_alignment()       — cosine sim between two direction dicts

  B. Safety neurons
       plot_change_score_heatmap()  — neuron change scores across all layers
       plot_neuron_layer_dist()     — how top-k neurons distribute across layers

  C. Thesis core  ← the plots that go in your results chapter
       plot_reconstruction_curve()  — variance explained vs k (Experiment 1)
       plot_direction_survival()    — cosine sim before/after ablation (Experiment 2)
       plot_causal_effect_bar()     — causal effect C for increasing k (Experiment 2)

  D. Behavioral
       plot_refusal_rates()         — bar chart comparing conditions
       plot_layer_sensitivity()     — refusal rate when ablating each layer

Style: clean, thesis-ready. No chart junk. All fonts large enough for a PDF.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from typing import Optional


# ── Shared style ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi":       150,
    "font.size":        12,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "lines.linewidth":  2.0,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
})

# Colorblind-friendly palette
COLORS = {
    "base":        "#4878CF",   # blue
    "instruct":    "#D65F5F",   # red
    "original":    "#4878CF",
    "ablated":     "#D65F5F",
    "patched":     "#6ACC65",   # green
    "neutral":     "#888888",
}


def _save_or_return(fig, save_path: Optional[str]) -> plt.Figure:
    """Save figure if path given, always return the figure."""
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved → {save_path}")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# A. Refusal direction plots
# ═════════════════════════════════════════════════════════════════════════════

def plot_direction_norms(
    directions: dict[int, torch.Tensor],
    title: str = "Refusal Direction Strength by Layer",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the L2 norm of the refusal direction at each layer.

    Useful sanity check: layers where r is near-zero are unlikely
    to be causally important. Peaks indicate where refusal is most
    strongly represented.

    Args:
        directions: output of compute_refusal_direction(normalize=False)
                    (use un-normalized directions so norms are meaningful)
    """
    layers = sorted(directions.keys())
    norms  = [directions[l].float().norm().item() for l in layers]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(layers, norms, color=COLORS["base"], marker="o", markersize=4)
    ax.fill_between(layers, norms, alpha=0.15, color=COLORS["base"])
    ax.set_xlabel("Layer")
    ax.set_ylabel("‖r‖  (L2 norm)")
    ax.set_title(title)
    ax.set_xlim(layers[0], layers[-1])
    return _save_or_return(fig, save_path)


def plot_layer_alignment(
    similarities: dict[int, float],
    title: str = "Refusal Direction Alignment Before vs After Ablation",
    ylabel: str = "Cosine Similarity",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot cosine similarity between two direction dicts across layers.

    Primary use: Experiment 2 — shows which layers retained / lost the
    refusal direction after safety neuron ablation.

    Args:
        similarities: output of direction_alignment()
    """
    layers = sorted(similarities.keys())
    sims   = [similarities[l] for l in layers]

    fig, ax = plt.subplots(figsize=(9, 4))
    colors  = [COLORS["instruct"] if s < 0.5 else COLORS["base"] for s in sims]
    ax.bar(layers, sims, color=colors, width=0.7, alpha=0.85)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", label="Perfect alignment")
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":",  label="0.5 threshold")
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=10)
    return _save_or_return(fig, save_path)


# ═════════════════════════════════════════════════════════════════════════════
# B. Safety neuron plots
# ═════════════════════════════════════════════════════════════════════════════

def plot_change_score_heatmap(
    scores: dict[int, torch.Tensor],
    top_k_neurons: int = 100,
    title: str = "Neuron Change Scores (Base → Instruct)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of change scores across layers and top-k neurons per layer.

    Rows = layers, columns = top neurons within each layer (ranked by score).
    Color intensity = change score magnitude.

    Useful for seeing whether safety neurons cluster in specific layers
    or spread uniformly.

    Args:
        scores:          output of compute_change_scores()
        top_k_neurons:   how many neurons to show per layer (for readability)
    """
    layers = sorted(scores.keys())
    matrix = []
    for l in layers:
        layer_scores = scores[l].float()
        top_vals, _  = layer_scores.topk(min(top_k_neurons, len(layer_scores)))
        matrix.append(top_vals.numpy())

    matrix = np.array(matrix)   # [n_layers, top_k_neurons]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel(f"Top-{top_k_neurons} neurons per layer (ranked)")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Change score", shrink=0.8)
    return _save_or_return(fig, save_path)


def plot_neuron_layer_dist(
    safety_neurons: list[tuple[int, int]],
    n_layers: int,
    title: str = "Distribution of Top Safety Neurons Across Layers",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart: how many of the top-k safety neurons fall in each layer.

    Helps answer: is safety localized to specific layers or distributed?
    Paper 2 finds neurons cluster in mid-to-late layers.

    Args:
        safety_neurons: output of get_top_safety_neurons()
        n_layers:       total number of layers (for x-axis range)
    """
    counts = [0] * n_layers
    for layer_idx, _ in safety_neurons:
        if layer_idx < n_layers:
            counts[layer_idx] += 1

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(n_layers), counts, color=COLORS["instruct"], alpha=0.8, width=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Number of safety neurons")
    ax.set_title(title)
    ax.set_xlim(-0.5, n_layers - 0.5)
    return _save_or_return(fig, save_path)


# ═════════════════════════════════════════════════════════════════════════════
# C. Thesis core plots
# ═════════════════════════════════════════════════════════════════════════════

def plot_reconstruction_curve(
    curve: dict[int, float],
    random_curve: Optional[dict[int, float]] = None,
    title: str = "Refusal Direction Reconstructed by Safety Neurons",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot variance explained in the refusal direction vs number of neurons used.

    This is your central Experiment 1 result figure.

    x-axis: k (number of safety neurons)
    y-axis: fraction of refusal direction r explained

    Optionally overlay a random-neuron baseline to show safety neurons
    are special (not just any k neurons would work).

    Args:
        curve:        output of top_neuron_reconstruction() — safety neurons
        random_curve: same format, but using randomly selected neurons
                      (run top_neuron_reconstruction on a random neuron list)

    Interpretation guide (add to thesis caption):
        Steep rise → small neuron subset suffices to reconstruct r
        Gap between safety and random curves → neurons are specifically aligned
        Plateau < 1.0 → some of r comes from non-MLP sources (attention, etc.)
    """
    ks    = sorted(curve.keys())
    vals  = [curve[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, vals, color=COLORS["instruct"], marker="o", markersize=5,
            label="Safety neurons")

    if random_curve is not None:
        rks   = sorted(random_curve.keys())
        rvals = [random_curve[k] for k in rks]
        ax.plot(rks, rvals, color=COLORS["neutral"], marker="s", markersize=5,
                linestyle="--", label="Random neurons (baseline)")

    ax.set_xlabel("Number of neurons (k)")
    ax.set_ylabel("Variance explained in refusal direction r")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=10)
    return _save_or_return(fig, save_path)


def plot_direction_survival(
    original_sims: dict[int, float],
    ablated_sims: dict[int, float],
    title: str = "Refusal Direction Survival After Neuron Ablation",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare per-layer direction alignment before and after safety neuron ablation.

    This is your central Experiment 2 result figure.

    Each pair of bars at layer l shows:
      Blue  = cosine similarity of original r with itself (always 1.0, reference)
      Red   = cosine similarity of ablated r with original r

    Large drop (blue→red) at layer l → safety neurons were generating r there
    Small drop               at layer l → r persists; neurons not its source there

    Args:
        original_sims: direction_alignment(original_dirs, original_dirs) — all 1.0
                        OR pass the ablated directions compared to original:
                        direction_alignment(original_dirs, ablated_dirs)
        ablated_sims:  direction_alignment(original_dirs, ablated_dirs)

    Tip: in practice you'll usually just pass ablated_sims and draw a
    horizontal reference line at 1.0 for the "before" baseline.
    """
    layers  = sorted(set(original_sims) & set(ablated_sims))
    x       = np.arange(len(layers))
    width   = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, [original_sims[l] for l in layers],
           width, color=COLORS["original"], alpha=0.85, label="Before ablation")
    ax.bar(x + width/2, [ablated_sims[l]  for l in layers],
           width, color=COLORS["ablated"],  alpha=0.85, label="After ablation")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity with original refusal direction")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontsize=8)
    ax.set_ylim(-0.1, 1.15)
    ax.axhline(1.0, color="gray", linewidth=0.7, linestyle="--")
    ax.legend(fontsize=10)
    return _save_or_return(fig, save_path)


def plot_causal_effect_bar(
    causal_effects: dict[int, float],
    title: str = "Causal Effect of Top-k Safety Neurons on Alignment",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of causal effect C for increasing k.

    Run compute_causal_effect() for k = [50, 100, 250, 500, ...] and
    pass the results here.

    x-axis: k (number of neurons patched)
    y-axis: causal effect C (fraction of alignment recovered)

    This answers: how many neurons do you need to recover most of safety?
    Matches Paper 2's Figure showing ~5% of neurons suffice.

    Args:
        causal_effects: dict mapping k → C (float in [0, 1])
    """
    ks   = sorted(causal_effects.keys())
    vals = [causal_effects[k] for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(ks)), vals, color=COLORS["patched"], alpha=0.85, width=0.6)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Number of patched neurons (k)")
    ax.set_ylabel("Causal effect C")
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", label="Full alignment")
    ax.legend(fontsize=10)

    # Label bars
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.0%}", ha="center", va="bottom", fontsize=10)

    return _save_or_return(fig, save_path)


# ═════════════════════════════════════════════════════════════════════════════
# D. Behavioral plots
# ═════════════════════════════════════════════════════════════════════════════

def plot_refusal_rates(
    conditions: dict[str, float],
    title: str = "Refusal Rate by Condition",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart comparing refusal rates across experimental conditions.

    Args:
        conditions: dict mapping condition label → refusal rate
                    e.g. {
                        "Instruct (baseline)":        0.92,
                        "After direction ablation":   0.11,
                        "After neuron ablation":      0.18,
                        "After both ablations":       0.08,
                        "Base model (no alignment)":  0.03,
                    }
    """
    labels = list(conditions.keys())
    values = list(conditions.values())

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(labels) + 1.5))
    bars = ax.barh(labels, values, color=COLORS["instruct"], alpha=0.8, height=0.55)
    ax.set_xlabel("Refusal rate")
    ax.set_title(title)
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.invert_yaxis()   # highest refusal at top

    # Label bars
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.0%}", va="center", fontsize=10)

    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_layer_sensitivity(
    sensitivity: dict[int, float],
    baseline_rate: float,
    title: str = "Refusal Rate When Ablating Each Layer Individually",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Line plot of refusal rate when the intervention is applied at each
    layer independently.

    Dip below baseline → that layer is important for refusal.
    Flat line           → intervention has no layer-specific effect.

    Args:
        sensitivity:   output of layer_sensitivity()
        baseline_rate: refusal rate with no intervention (draw as reference)
    """
    layers = sorted(sensitivity.keys())
    rates  = [sensitivity[l] for l in layers]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(layers, rates, color=COLORS["instruct"], marker="o", markersize=4,
            label="Per-layer ablation")
    ax.axhline(baseline_rate, color=COLORS["base"], linewidth=1.5,
               linestyle="--", label=f"Baseline ({baseline_rate:.0%})")
    ax.fill_between(layers, rates, baseline_rate,
                    where=[r < baseline_rate for r in rates],
                    alpha=0.15, color=COLORS["instruct"],
                    label="Refusal reduction")

    ax.set_xlabel("Layer where ablation is applied")
    ax.set_ylabel("Refusal rate")
    ax.set_title(title)
    ax.set_xlim(layers[0], layers[-1])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=10)
    return _save_or_return(fig, save_path)