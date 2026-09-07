"""Render retained summaries and raster samples; no simulation or measurement."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import theme
from experiments.helpers.figsave import save_figure


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _compound_raster_panel(ax, s: dict, title: str) -> None:
    """One raster panel — E (black) below, I (red) above."""
    n_e, n_i, gap = s["e"].shape[1], s["i"].shape[1], 6
    T = s["e"].shape[0]
    t_axis = np.arange(T) * s["dt"]
    e_t, e_n = np.where(s["e"])
    i_t, i_n = np.where(s["i"])
    ax.scatter(t_axis[e_t], e_n, s=1.5, c=theme.INK_BLACK, marker="|", linewidths=0.4)
    ax.scatter(
        t_axis[i_t],
        i_n + n_e + gap,
        s=1.5,
        c=theme.DEEP_RED,
        marker="|",
        linewidths=0.4,
    )
    ax.set_ylim(-2, n_e + n_i + gap + 2)
    ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
    ax.set_yticklabels(["E", "I"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, s["t_ms"])
    ax.set_title(title, fontsize=theme.SIZE_LABEL)
    _despine(ax)


def _compound_sweep_panel(
    ax,
    rows: list[dict],
    *,
    xlabel: str,
    title: str,
    xlim: tuple[float, float],
    legend_loc: str,
) -> None:
    """One sweep panel — E (black), I (red), accuracy (grey, twin axis) vs σ."""
    sig = [r["sigma_ms"] for r in rows]
    e_means = [r["e_rate_hz"]["mean"] for r in rows]
    i_means = [r["i_rate_hz"]["mean"] for r in rows]
    a_means = [r["acc"]["mean"] for r in rows]

    ax.plot(
        sig, e_means, marker="o", ms=3, lw=1.4, color=theme.INK_BLACK, label="E rate"
    )
    # Realised mean I-spike rate is shown on the same Hz axis. Count-preserving
    # replay keeps this rate fixed; it does not measure inhibitory conductance.
    ax.plot(
        sig,
        i_means,
        marker="s",
        ms=3,
        lw=1.4,
        color=theme.DEEP_RED,
        ls="-",
        label="realised I",
    )
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("firing rate (Hz)", color=theme.INK_BLACK, fontsize=theme.SIZE_LABEL)
    ax.tick_params(axis="y", labelcolor=theme.INK_BLACK)
    ax.set_title(title, fontsize=theme.SIZE_LABEL)

    ax_acc = ax.twinx()
    ax_acc.plot(
        sig, a_means, marker="s", ms=3, lw=1.4, color=theme.GREY_MID, label="accuracy"
    )
    ax_acc.set_ylabel("accuracy (%)", color=theme.GREY_MID, fontsize=theme.SIZE_LABEL)
    ax_acc.tick_params(axis="y", labelcolor=theme.GREY_MID)
    ax_acc.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax_acc.spines["top"].set_visible(False)

    # Self-identify all three traces; combine both axes' handles into one legend.
    h_rate, l_rate = ax.get_legend_handles_labels()
    h_acc, l_acc = ax_acc.get_legend_handles_labels()
    ax.legend(
        h_rate + h_acc,
        l_rate + l_acc,
        loc=legend_loc,
        frameon=False,
        fontsize=theme.SIZE_ANNOTATION,
    )
def fig_rhythm_compound(
    cyc_rows: list[dict],
    cell_rows: list[dict],
    raster_cyc: dict,
    raster_cell: dict,
    out_path: Path,
) -> None:
    """2×2 compound comparing independent-spike and fixed-window jitter.

    Top row: illustrative single-trial rasters. Bottom row: full sweep means.
    The figure reports realised spike rates without claiming measured synchrony
    or matched postsynaptic conductance.
    """
    theme.apply()
    prev_bbox = plt.rcParams["savefig.bbox"]
    plt.rcParams["savefig.bbox"] = "standard"
    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 4.5))
    shared_sweep_xlim = (0.0, 14.0)

    _compound_raster_panel(
        axes[0, 0],
        raster_cell,
        "Independent-spike jitter",
    )
    _compound_raster_panel(
        axes[0, 1],
        raster_cyc,
        "Fixed-window group jitter",
    )
    _compound_sweep_panel(
        axes[1, 0],
        cell_rows,
        xlabel="jitter σ (ms)",
        title="Independent-spike jitter",
        xlim=shared_sweep_xlim,
        legend_loc="center right",
    )
    _compound_sweep_panel(
        axes[1, 1],
        cyc_rows,
        xlabel="jitter σ (ms)",
        title="Fixed-window group jitter",
        xlim=shared_sweep_xlim,
        legend_loc="center left",
    )
    handles, labels = axes[1, 0].get_legend_handles_labels()
    legend = axes[1, 0].get_legend()
    handles = legend.legend_handles
    labels = [text.get_text() for text in legend.get_texts()]
    for ax in axes[1]:
        ax.get_legend().remove()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=theme.SIZE_LEGEND)
    for ax in axes[0]:
        ax.set_xlabel("time (ms)")
    # The caption carries the findings.
    theme.label_panels(axes.flat)
    fig.tight_layout(rect=(0, 0.07, 1, 0.98), h_pad=1.5, w_pad=1.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png",))  # dense rasters: PNG, not SVG
    plt.close(fig)
    plt.rcParams["savefig.bbox"] = prev_bbox
