from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools"), str(REPO / "tools/snnsim")]

import matplotlib.pyplot as plt
import numpy as np
from experiments.exp075.recipe import N_CLASSES
from experiments.helpers import theme


def plot_training(metrics: dict, out_path: Path) -> None:
    theme.apply()
    rows = metrics["epochs"]
    epochs = np.asarray([row["ep"] for row in rows])
    train_loss = np.asarray([row["loss"] for row in rows])
    test_loss = np.asarray([row["test_loss"] for row in rows])
    accuracy = np.asarray([row["acc"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
    axes[0].plot(
        epochs,
        train_loss,
        marker="o",
        color=theme.INK_BLACK,
        label="train",
    )
    axes[0].plot(
        epochs,
        test_loss,
        marker="o",
        color=theme.DEEP_RED,
        label="validation",
    )
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].legend(frameon=False)

    axes[1].plot(
        epochs,
        accuracy,
        marker="o",
        color=theme.INK_BLACK,
    )
    axes[1].axhline(
        100.0 / N_CLASSES,
        color=theme.DEEP_RED,
        linestyle="--",
        linewidth=1.0,
        label="chance",
    )
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation accuracy (%)")
    axes[1].set_ylim(0, max(30.0, float(accuracy.max()) * 1.15))
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.set_xticks(epochs)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    theme.label_panels(axes)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
