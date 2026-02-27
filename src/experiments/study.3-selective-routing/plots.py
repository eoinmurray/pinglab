from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pinglab.plots.styles import save_both


def _apply_font() -> None:
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.monospace"] = [
        "Menlo",
        "Monaco",
        "DejaVu Sans Mono",
        "Courier New",
        "monospace",
    ]
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.titlesize"] = 21
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13


def save_e2_spikes_vs_delay(
    path: Path,
    delay_ms: list[float],
    e1_spikes: list[int],
    e2_spikes: list[int],
) -> None:
    x = np.asarray(delay_ms, dtype=float)
    y1 = np.asarray(e1_spikes, dtype=float)
    y = np.asarray(e2_spikes, dtype=float)

    def _plot() -> None:
        _apply_font()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(x, y1, marker="o", linewidth=1.6, markersize=4, label="E_A")
        ax.plot(x, y, marker="o", linewidth=1.6, markersize=4, label="E_B")
        ax.set_title("Target spikes vs delay")
        ax.set_xlabel("E_src->E_B delay (ms)")
        ax.set_ylabel("Spikes")
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()

    save_both(path, _plot)
