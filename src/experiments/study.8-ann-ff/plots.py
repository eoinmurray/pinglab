from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pinglab.plots.styles import save_both

def save_line(path: Path, x, y, *, title: str, xlabel: str, ylabel: str) -> None:
    def _plot() -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        marker = "o" if len(x) <= 20 else None
        markersize = 5 if marker else 0
        ax.plot(x, y, linewidth=1.5, marker=marker, markersize=markersize)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if len(x) <= 20:
            ax.set_xticks(x)
        fig.tight_layout()

    save_both(path, _plot)
