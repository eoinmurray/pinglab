from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pinglab.plots.styles import save_both


def plot_line(times: np.ndarray, values: np.ndarray, out_path: Path, *, title: str) -> None:
    def plot_fn() -> None:
        plt.figure(figsize=(6, 6))
        if times.size and values.size:
            plt.plot(times, values, linewidth=1.5)
        else:
            plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    save_both(out_path, plot_fn)


def plot_weight_histograms(
    W_ee: np.ndarray,
    W_ei: np.ndarray,
    W_ie: np.ndarray,
    W_ii: np.ndarray,
    out_path: Path,
    *,
    title: str,
) -> None:
    def plot_fn() -> None:
        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
        blocks = [
            ("W_ee", W_ee),
            ("W_ei", W_ei),
            ("W_ie", W_ie),
            ("W_ii", W_ii),
        ]
        for ax, (label, block) in zip(axes.flat, blocks):
            data = block.ravel()
            ax.hist(data, bins=50, alpha=0.8)
            ax.set_title(label)
            ax.grid(True, alpha=0.2)
        if title:
            fig.suptitle(title)
        plt.tight_layout()

    save_both(out_path, plot_fn)
