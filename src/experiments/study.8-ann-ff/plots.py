from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pinglab.plots.styles import save_both

def save_line(path: Path, x, y) -> None:
    def _plot() -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(x, y, linewidth=2.0)
        ax.set_title("Placeholder")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()

    save_both(path, _plot)
