from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pinglab.plots.styles import save_both


def save_weights_heatmap(
    weights: np.ndarray,
    path: Path | str,
    *,
    layer_bounds: list[tuple[int, int, str]] | None = None,
    title: str = "Weights heatmap",
) -> None:
    matrix = np.asarray(weights, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("weights must be a 2D matrix")

    def _plot() -> None:
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(
            matrix,
            origin="upper",
            aspect="auto",
            interpolation="nearest",
            cmap="Greys",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="weight")

        ax.set_title(title)
        ax.set_xlabel("Source neuron id")
        ax.set_ylabel("Target neuron id")

        n_rows, n_cols = matrix.shape
        if layer_bounds:
            for start, stop, label in layer_bounds:
                s = int(start)
                e = int(stop)
                if e <= s:
                    continue
                ax.axhline(s - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
                ax.axvline(s - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
                center = (s + e - 1) * 0.5
                ax.text(
                    n_cols + 1.0,
                    center,
                    str(label),
                    va="center",
                    ha="left",
                    fontsize=9,
                    clip_on=False,
                )
                ax.text(
                    center,
                    -2.0,
                    str(label),
                    va="bottom",
                    ha="center",
                    fontsize=9,
                    clip_on=False,
                )
            if layer_bounds:
                final_stop = int(layer_bounds[-1][1])
                ax.axhline(final_stop - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
                ax.axvline(final_stop - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

        ax.set_xlim(-0.5, n_cols - 0.5)
        ax.set_ylim(n_rows - 0.5, -0.5)
        fig.tight_layout()

    save_both(path, _plot)
