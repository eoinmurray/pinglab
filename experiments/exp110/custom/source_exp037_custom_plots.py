"""Presentation of saved measurements only; no simulation or aggregation.

The owning experiment calls these local definitions from its explicit stages.
Inputs and outputs retain that stage’s validated scientific contract; importing
this module does not start computation or select a scientific run.
"""

from pathlib import Path

import matplotlib.pyplot as plt

import matplotlib.ticker as mticker

from experiments import theme

from experiments.exp110.custom.source_exp037_custom_figures import save_figure

from matplotlib.patches import Patch

from experiments.exp110.custom.source_exp037_recipe import MODELS

MODEL_COLORS = {"coba": theme.DEEP_RED, "ping": theme.INK_BLACK}

MODEL_MARKERS = {"coba": "s", "ping": "D"}


def plot_perturbation_curves(
    data: dict,
    out_path: Path,
    run_id: str,
    *,
    axes=None,
) -> None:
    """Plot saved across-seed means and sample-SD envelopes."""
    theme.apply()
    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3), sharey=True)
    fig = axes[0].figure
    use_pct = data["use_pct"]

    # Left panel: drop (as % of spikes dropped)
    ax_drop = axes[0]
    for model in MODELS:
        row = data["panels"]["drop"][model]
        xs, means, lo, hi = (row[k] for k in ("x", "mean", "lo", "hi"))
        ax_drop.plot(
            xs,
            means,
            marker=MODEL_MARKERS[model],
            markersize=5,
            linewidth=1.4,
            color=MODEL_COLORS[model],
            label=model.upper(),
            zorder=3 if model == "coba" else 2,
        )
        ax_drop.fill_between(
            xs,
            lo,
            hi,
            color=MODEL_COLORS[model],
            alpha=0.15,
            linewidth=0,
        )
    ax_drop.set_xlabel("Spike-deletion probability (%)", fontsize=theme.SIZE_LABEL)
    ax_drop.set_title(
        "Drop — Bernoulli deletion", fontsize=theme.SIZE_LABEL, loc="left", pad=4
    )
    ax_drop.set_xlim(-2, 102)
    ax_drop.axhline(10.0, ls="--", color=theme.MUTED, lw=0.7, alpha=0.6)
    ax_drop.text(
        0.02,
        12,
        "chance",
        transform=ax_drop.get_yaxis_transform(),
        fontsize=theme.SIZE_ANNOTATION,
        color=theme.MUTED,
        va="bottom",
    )

    # Right panel: add
    ax_add = axes[1]
    if use_pct:
        for model in MODELS:
            row = data["panels"]["add"][model]
            xs, means, lo, hi = (row[k] for k in ("x", "mean", "lo", "hi"))
            ax_add.plot(
                xs,
                means,
                marker=MODEL_MARKERS[model],
                markersize=5,
                linewidth=1.4,
                color=MODEL_COLORS[model],
                label=model.upper(),
                zorder=3 if model == "coba" else 2,
            )
            ax_add.fill_between(
                xs,
                lo,
                hi,
                color=MODEL_COLORS[model],
                alpha=0.15,
                linewidth=0,
            )
        ax_add.set_xlabel(
            "Nominal added rate / baseline E rate (%)"
            if data.get("relative_test_baseline")
            else "Nominal added rate / reference E rate (%)",
            fontsize=theme.SIZE_LABEL,
        )
        ax_add.set_title(
            "Add — Bernoulli insertion",
            fontsize=theme.SIZE_LABEL,
            loc="left",
            pad=4,
        )
        maximum = max(max(row["x"]) for row in data["panels"]["add"].values())
        margin = max(1.0, 0.03 * maximum)
        ax_add.set_xlim(-margin, maximum + margin)
    else:
        for model in MODELS:
            row = data["panels"]["add"][model]
            xs, means, lo, hi = (row[k] for k in ("x", "mean", "lo", "hi"))
            ax_add.plot(
                xs,
                means,
                marker=MODEL_MARKERS[model],
                markersize=5,
                linewidth=1.4,
                color=MODEL_COLORS[model],
                label=model.upper(),
                zorder=3 if model == "coba" else 2,
            )
            ax_add.fill_between(
                xs,
                lo,
                hi,
                color=MODEL_COLORS[model],
                alpha=0.15,
                linewidth=0,
            )
        ax_add.set_xlabel("Nominal added rate (Hz / neuron)", fontsize=theme.SIZE_LABEL)
        ax_add.set_title(
            "Add — Bernoulli insertion",
            fontsize=theme.SIZE_LABEL,
            loc="left",
            pad=4,
        )
    ax_add.axhline(10.0, ls="--", color=theme.MUTED, lw=0.7, alpha=0.6)

    for ax in axes:
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=theme.SIZE_TICK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.grid(True, axis="y", alpha=0.15, linewidth=0.5)
    ax_drop.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    legend_handles, legend_labels = ax_add.get_legend_handles_labels()
    legend_handles.append(
        Patch(
            facecolor=theme.INK_BLACK,
            edgecolor="none",
            alpha=0.15,
            label="±1 SD band",
        )
    )
    legend_labels.append("±1 SD band")
    ax_add.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        fontsize=theme.SIZE_LEGEND,
        frameon=False,
    )
    # The writeup caption carries the takeaway rather than a figure title.
    if standalone:
        theme.label_panels(axes)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)  # line/curve plot: SVG + PDF
        plt.close(fig)
