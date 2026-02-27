from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pinglab.plots.styles import save_both


def save_mean_input_plot(path: Path, *, t_ms: np.ndarray, mean_input: np.ndarray) -> None:
    def _plot() -> None:
        plt.rcParams["font.family"] = "monospace"
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(t_ms, mean_input, linewidth=1.8)
        ax.set_title("Mean E input over time")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Mean E input")
        ax.grid(alpha=0.25)
        fig.tight_layout()

    save_both(path, _plot)
