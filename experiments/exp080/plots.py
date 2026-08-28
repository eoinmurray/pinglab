"""Render retained values without simulation or statistical estimation."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import theme


def plot_training(records: list[dict[str, Any]], output: Path) -> None:
    theme.apply()
    fig, axis = plt.subplots(figsize=(5.1, 3.1), constrained_layout=True)
    for record in records:
        axis.plot(
            [row["epoch"] for row in record["history"]],
            [row["validation_accuracy"] for row in record["history"]],
            label=f"Seed {record['seed']}",
        )
    axis.set(xlabel="Epoch", ylabel="Mixed-rate validation accuracy", ylim=(0, 1))
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.savefig(output / "training_history.svg", metadata={"Date": None})
    plt.close(fig)


def plot_psychometric(decision: dict[str, Any], output: Path, cfg: dict) -> None:
    theme.apply()
    rows = decision["rows"]
    rates = np.asarray([row["rate_hz"] for row in rows])
    accuracy = np.asarray([row["accuracy"] for row in rows])
    lower = np.asarray([row["minimum_seed_accuracy"] for row in rows])
    upper = np.asarray([row["maximum_seed_accuracy"] for row in rows])
    fig, axis = plt.subplots(figsize=(5.1, 3.2), constrained_layout=True)
    axis.plot(rates, accuracy, color=theme.INK_BLACK, marker="o")
    axis.fill_between(
        rates, lower, upper, color=theme.INK_BLACK, alpha=0.14, linewidth=0
    )
    axis.axhline(cfg["chance_accuracy"], color="#777777", linestyle=":", label="Chance")
    axis.axhline(
        cfg["useful_accuracy"],
        color="#777777",
        linestyle="--",
        label="Practical criterion",
    )
    if decision["criterion_crossed"]:
        axis.axvline(
            decision["r_train_hz"],
            color=theme.DEEP_RED,
            linewidth=1.0,
            label="Selected floor",
        )
    axis.set_xscale("log")
    axis.set_xticks(cfg["rates_hz"])
    axis.set_xticklabels([f"{rate:g}" for rate in cfg["rates_hz"]])
    axis.set(
        xlabel="Maximum-pixel encoding rate (Hz)",
        ylabel="Held-out accuracy",
        ylim=(0, 1),
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=7)
    fig.savefig(output / "psychometric.svg", metadata={"Date": None})
    plt.close(fig)


def plot_feature_images(
    image: np.ndarray, features: np.ndarray, rates: np.ndarray, output: Path
) -> None:
    theme.apply()
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.0), constrained_layout=True)
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Input image")
    for axis, feature, rate in zip(axes[1:], features, rates, strict=True):
        axis.imshow(feature.reshape(28, 28), cmap="magma", vmin=0, vmax=65)
        axis.set_title(f"{rate:g} Hz")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.savefig(output / "feature_images.png", dpi=240)
    plt.close(fig)
