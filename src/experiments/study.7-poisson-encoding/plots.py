from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pinglab.plots.styles import save_both


def save_placeholder_plot(path: Path) -> None:
    x = np.linspace(0.0, 1.0, 200)
    y = 0.5 + 0.4 * np.sin(2.0 * np.pi * 5.0 * x)

    def _plot() -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(x, y, linewidth=2.0)
        ax.set_title("Placeholder")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()

    save_both(path, _plot)


def save_total_spikes_vs_neuron_id(
    path: Path,
    neuron_ids: np.ndarray,
    spike_counts: np.ndarray,
) -> None:
    x = neuron_ids.astype(np.int64)
    y = spike_counts.astype(np.int64)

    def _plot() -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(x, y, linewidth=1.6, marker="o", markersize=3.0)
        ax.set_title("Total spikes vs neuron id")
        ax.set_xlabel("Neuron id")
        ax.set_ylabel("Total spikes")
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylim(0.0, max(1.0, float(y.max()) * 1.05))
        fig.tight_layout()

    save_both(path, _plot)
