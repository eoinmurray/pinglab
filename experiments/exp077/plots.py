from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools"), str(REPO / "tools/snnsim")]

import matplotlib.pyplot as plt
import numpy as np
from experiments.exp077.recipe import DT_MS
from experiments.helpers import theme


def render_rasters(results: dict[str, dict[str, np.ndarray]], out: Path) -> None:
    theme.apply()
    fig, axes = plt.subplots(len(results), 2, figsize=(7.2, 6.8), sharex=True)
    for row, (name, rec) in enumerate(results.items()):
        for col, key in enumerate(("population_0", "population_2")):
            t, cell = np.nonzero(rec[key][:, 0])
            axes[row, col].scatter(
                t * DT_MS,
                cell,
                s=3,
                color=theme.INK_BLACK if col == 0 else theme.DEEP_RED,
                linewidths=0,
            )
            axes[row, col].set_ylabel(name.replace("_", " "), fontsize=7)
            if row == 0:
                axes[row, col].set_title(
                    "circuit A · E" if col == 0 else "circuit B · E"
                )
    for ax in axes[-1]:
        ax.set_xlabel("time (ms)")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
