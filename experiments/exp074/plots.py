from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools"), str(REPO / "tools/snnsim")]

import matplotlib.pyplot as plt
import numpy as np
from experiments.exp074.recipe import DT_MS, N_E, N_I, N_INPUT, T_MS
from experiments.helpers import theme


def plot_rasters(raster_path: Path, out_path: Path) -> dict[str, int]:
    """Plot the exact input and resulting E/I spikes for one aligned trial."""
    theme.apply()
    dt_ms = DT_MS
    with np.load(raster_path) as events:
        input_t, input_cell = events["input_t"], events["input_cell"]
        e_t, e_cell = events["e_t"], events["e_cell"]
        i_t, i_cell = events["i_t"], events["i_cell"]
    panels = [
        ("INPUT", input_t, input_cell, theme.GREY_MID, N_INPUT),
        ("E", e_t, e_cell, theme.INK_BLACK, N_E),
        ("I", i_t, i_cell, theme.DEEP_RED, N_I),
    ]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.5, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.35, 0.9], "hspace": 0.16},
    )
    for ax, (label, times, cells, colour, size) in zip(axes, panels):
        ax.scatter(
            times * dt_ms,
            cells,
            s=2.0,
            marker=".",
            linewidths=0,
            color=colour,
            alpha=0.7,
            rasterized=True,
        )
        ax.set_ylim(-1, size)
        ax.set_ylabel(label, rotation=0, ha="right", va="center")
        ax.text(
            0.995,
            0.92,
            f"{size} cells · {len(times):,} spikes",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=theme.SIZE_ANNOTATION,
            color=colour,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlim(0, T_MS)
    axes[-1].set_xlabel("time (ms)")
    fig.align_ylabels(axes)
    theme.label_panels(axes)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
