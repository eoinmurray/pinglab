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
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12


def save_input_mean_vs_neuron_plot(
    path: Path,
    *,
    neuron_ids: np.ndarray,
    input_mean: np.ndarray,
) -> None:
    def _plot() -> None:
        _apply_font()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(neuron_ids, input_mean, linewidth=1.6)
        ax.set_title("Tonic input mean vs E neuron id")
        ax.set_xlabel("E neuron id")
        ax.set_ylabel("Input mean")
        ax.grid(alpha=0.25)
        fig.tight_layout()

    save_both(path, _plot)


def save_neuron_phase_vs_id_plot(
    path: Path,
    *,
    neuron_ids: np.ndarray,
    mean_phase: np.ndarray,
    input_mean: np.ndarray,
) -> None:
    def _plot() -> None:
        _apply_font()
        fig, ax = plt.subplots(figsize=(8, 8))
        valid = np.isfinite(mean_phase) & np.isfinite(input_mean)
        if np.any(valid):
            x = neuron_ids[valid]
            y = mean_phase[valid]
            v = input_mean[valid].astype(float)
            vmin = float(np.min(v))
            vmax = float(np.max(v))
            if vmax > vmin:
                v_norm = (v - vmin) / (vmax - vmin)
            else:
                v_norm = np.zeros_like(v, dtype=float)
            sizes = 8.0 + 28.0 * v_norm
            ax.scatter(x, y, s=sizes, alpha=0.75)
        ax.set_title("E neuron id vs mean phase")
        ax.set_xlabel("E neuron id")
        ax.set_ylabel("Mean phase in cycle")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.25)
        fig.tight_layout()

    save_both(path, _plot)


def save_true_vs_decoded_input_plot(
    path: Path,
    *,
    true_input: np.ndarray,
    decoded_input: np.ndarray,
) -> None:
    def _plot() -> None:
        _apply_font()
        fig, ax = plt.subplots(figsize=(8, 8))
        valid = np.isfinite(true_input) & np.isfinite(decoded_input)
        ax.scatter(true_input[valid], decoded_input[valid], s=10, alpha=0.7)
        if np.sum(valid) >= 2:
            lo = float(min(np.min(true_input[valid]), np.min(decoded_input[valid])))
            hi = float(max(np.max(true_input[valid]), np.max(decoded_input[valid])))
            ax.plot([lo, hi], [lo, hi], linewidth=1.0, linestyle="--", alpha=0.6)
            r = float(np.corrcoef(true_input[valid], decoded_input[valid])[0, 1])
            ax.text(0.02, 0.98, f"r={r:.3f}", transform=ax.transAxes, ha="left", va="top")
        ax.set_title("True tonic vs decoded tonic")
        ax.set_xlabel("True tonic input")
        ax.set_ylabel("Decoded tonic input")
        ax.grid(alpha=0.25)
        fig.tight_layout()

    save_both(path, _plot)
