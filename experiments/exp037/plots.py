"""Presentation of saved measurements only; no simulation or aggregation."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from experiments.helpers import theme
from experiments.helpers.figsave import save_figure

from .recipe import EI_RASTER_N_E_PLOT, EI_RASTER_N_I_PLOT, MODELS

MODEL_COLORS = {"coba": theme.DEEP_RED, "ping": theme.INK_BLACK}
MODEL_MARKERS = {"coba": "s", "ping": "D"}


def plot_perturbation_rasters(
    samples: list[dict], out_path: Path, run_id: str, level_fmt: str, title: str
) -> None:
    """Stacked single-trial rasters across perturbation levels for one mode."""
    theme.apply()
    n = len(samples)
    n_e = EI_RASTER_N_E_PLOT
    n_i = EI_RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(6.9, 3.88),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples)):
        ax.scatter(
            s["e_t"],
            s["e_n"],
            s=2.0,
            c=theme.INK_BLACK,
            marker="|",
            linewidths=0.4,
        )
        ax.scatter(
            s["i_t"],
            s["i_n"],
            s=2.0,
            c=theme.DEEP_RED,
            marker="|",
            linewidths=0.4,
        )
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, s["t_ms"])
        ax.text(
            1.012,
            0.5,
            level_fmt.format(level=s["level"]) + f"\nE = {s['e_rate_hz']:.1f} Hz",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=theme.SIZE_LABEL,
        )
        # The writeup caption carries the takeaway rather than a figure title.
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.subplots_adjust(left=0.07, right=0.78, bottom=0.12, top=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG + PDF
    plt.close(fig)


def plot_perturbation_curves(
    data: dict,
    out_path: Path,
    run_id: str,
) -> None:
    """Plot saved across-seed means and sample-SD envelopes."""
    theme.apply()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3), sharey=True)
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
            "Nominal added rate / reference E rate (%)",
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
    ax_add.legend(
        loc="upper right",
        fontsize=theme.SIZE_LEGEND,
        frameon=False,
    )
    # The writeup caption carries the takeaway rather than a figure title.
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)  # line/curve plot: SVG + PDF
    plt.close(fig)
