from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from pinglab.plots.styles import save_both

from .model import LocalConfig


def run_experiment(config: LocalConfig, data_path: Path) -> None:
    """Placeholder experiment. Replace with real analysis."""

    def plot_fn() -> None:
        plt.figure(figsize=(6, 6))
        plt.text(0.5, 0.5, "TODO: add plots", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()

    save_both(data_path / "placeholder.png", plot_fn)
