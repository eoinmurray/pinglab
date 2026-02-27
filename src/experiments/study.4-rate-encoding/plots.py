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
    plt.rcParams["legend.fontsize"] = 12


def save_input_signal_plot(
    path: Path,
    t_ms: np.ndarray,
    message: np.ndarray,
    input_current: np.ndarray,
) -> None:
    def _plot() -> None:
        _apply_font()
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        ax.plot(t_ms, message, linewidth=1.4, label="Message envelope")
        ax.plot(t_ms, input_current, linewidth=1.0, linestyle="--", alpha=0.75, label="Injected current")
        ax.set_title("Message signal")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Current")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.25)
        fig.tight_layout()

    save_both(path, _plot)


def save_layer_population_rates_plot(
    path: Path,
    t_ms: np.ndarray,
    rates_by_layer: dict[str, np.ndarray],
) -> None:
    def _plot() -> None:
        _apply_font()
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        max_rate = 0.0
        for rate in rates_by_layer.values():
            if rate.size:
                max_rate = max(max_rate, float(np.max(rate)))
        offset_step = max(1.0, 1.2 * max_rate)

        x_max = float(t_ms[-1]) if t_ms.size else 0.0
        items = list(rates_by_layer.items())
        n_layers = len(items)
        for idx, (label, rate) in enumerate(items):
            offset = (n_layers - 1 - idx) * offset_step
            ax.hlines(
                y=offset,
                xmin=0.0,
                xmax=x_max,
                linewidth=0.8,
                linestyles="--",
                alpha=0.35,
            )
            ax.plot(t_ms, rate + offset, linewidth=1.2)
            ax.text(
                x_max * 1.005,
                offset + 0.5 * offset_step,
                label,
                fontsize=10,
                va="center",
                ha="left",
            )
        ax.set_title("Stacked population rates")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Rate (stacked)")
        ax.set_yticks([])
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()

    save_both(path, _plot)


def save_decoded_envelopes_plot(
    path: Path,
    t_ms: np.ndarray,
    message_ref: np.ndarray,
    decoded_by_layer: dict[str, np.ndarray],
) -> None:
    def _plot() -> None:
        _apply_font()
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.plot(t_ms, message_ref, linewidth=1.8, label="Message (ref)")
        for label, decoded in decoded_by_layer.items():
            ax.plot(t_ms, decoded, linewidth=1.2, label=f"{label} decoded")
        ax.set_title("Decoded envelopes vs message")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Normalized amplitude")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.25)
        fig.tight_layout()

    save_both(path, _plot)

