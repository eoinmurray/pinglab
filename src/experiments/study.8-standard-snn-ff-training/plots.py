from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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
        # Only set explicit ticks when there are few points (e.g. per-epoch).
        # For per-iteration series with many points, let matplotlib auto-tick.
        if len(x) <= 20:
            ax.set_xticks(x)
        fig.tight_layout()

    save_both(path, _plot)


def save_raster_grid(
    path: Path,
    spike_tensors,   # list[Tensor[T, n_out]] — one per digit class
    *,
    dt: float,
    digit_labels=None,
    suptitle: str | None = None,
) -> None:
    """2×5 grid of output-layer rasters, one panel per digit class.

    Args:
        path: Output file path (without extension / light/dark suffix).
        spike_tensors: List of 10 tensors, each [T, n_out].  Values are 0/1.
        dt: Timestep in ms (used for x-axis scaling).
        digit_labels: Optional list of 10 digit class labels (default 0–9).
        suptitle: Optional overall figure title.
    """
    if digit_labels is None:
        digit_labels = list(range(len(spike_tensors)))

    n_out = spike_tensors[0].shape[1]
    T = spike_tensors[0].shape[0]
    t_ms = np.arange(T) * dt

    def _plot() -> None:
        fig, axes = plt.subplots(2, 5, figsize=(6, 6), sharex=True, sharey=True)
        axes = axes.flatten()

        for idx, (spikes, label) in enumerate(zip(spike_tensors, digit_labels)):
            ax = axes[idx]
            arr = spikes.numpy() if hasattr(spikes, "numpy") else np.asarray(spikes)
            # Build scatter coordinates from dense tensor
            t_idx, n_idx = np.nonzero(arr)
            ax.scatter(t_ms[t_idx], n_idx, s=3, marker=".", rasterized=True)
            ax.set_title(f"digit {label}", fontsize=8)
            ax.set_yticks(range(n_out))
            ax.set_yticklabels([str(i) for i in range(n_out)], fontsize=5)
            ax.set_ylim(-0.5, n_out - 0.5)
            ax.tick_params(axis="x", labelsize=6)

        for ax in axes:
            ax.set_xlabel("ms", fontsize=6)

        if suptitle:
            fig.suptitle(suptitle, fontsize=9)
        fig.tight_layout()

    save_both(path, _plot)


def save_raster_layers(
    path: Path,
    spikes,          # Tensor[T, N_E] — full E population
    *,
    dt: float,
    pop_idx: dict,   # population_index from compile_graph
    title: str | None = None,
    n_hid_sample: int = 80,  # max hidden neurons to display
) -> None:
    """Raster showing hidden and output layer activity for a single sample.

    E_in (784 neurons) is omitted — it is just the rate-coded image.
    E_hid is subsampled to n_hid_sample evenly-spaced neurons for readability.
    E_out (10 neurons) is shown in full.

    Args:
        path: Output file path (without extension / light/dark suffix).
        spikes: [T, N_E] spike tensor (0/1 floats).
        dt: Timestep in ms.
        pop_idx: ``plan["population_index"]`` dict from ``compile_graph``.
        title: Optional plot title.
        n_hid_sample: How many hidden neurons to subsample for display.
    """
    arr = spikes.numpy() if hasattr(spikes, "numpy") else np.asarray(spikes)
    T = arr.shape[0]
    t_ms = np.arange(T) * dt

    hid_start = int(pop_idx["E_hid"]["start"])
    hid_stop  = int(pop_idx["E_hid"]["stop"])
    out_start = int(pop_idx["E_out"]["start"])
    out_stop  = int(pop_idx["E_out"]["stop"])

    n_hid = hid_stop - hid_start
    n_out = out_stop - out_start

    # Subsample hidden neurons evenly
    hid_sample = np.linspace(hid_start, hid_stop - 1, min(n_hid_sample, n_hid), dtype=int)
    n_hid_shown = len(hid_sample)

    def _plot() -> None:
        fig, ax = plt.subplots(figsize=(6, 6))

        cursor = 0  # running y-offset in display space

        # ── Hidden layer (subsampled) ───────────────────────────────────────
        hid_arr = arr[:, hid_sample]           # [T, n_hid_shown]
        t_idx, n_idx = np.nonzero(hid_arr)
        ax.scatter(t_ms[t_idx], cursor + n_idx, s=2, marker=".", rasterized=True,
                   label=f"E_hid (n={n_hid_shown}/{n_hid})")
        cursor += n_hid_shown

        # separator
        ax.axhline(cursor, linewidth=0.7, linestyle="--", alpha=0.4)

        # ── Output layer ────────────────────────────────────────────────────
        out_arr = arr[:, out_start:out_stop]   # [T, n_out]
        t_idx, n_idx = np.nonzero(out_arr)
        ax.scatter(t_ms[t_idx], cursor + n_idx, s=5, marker="|", rasterized=True,
                   label="E_out")

        # label each output neuron
        for i in range(n_out):
            ax.text(t_ms[-1] * 1.01, cursor + i, str(i), fontsize=6,
                    va="center", ha="left", clip_on=False)

        cursor += n_out

        # ── Layer labels (left margin) ──────────────────────────────────────
        ax.text(-t_ms[-1] * 0.02, n_hid_shown / 2, "E_hid",
                fontsize=7, va="center", ha="right", clip_on=False)
        ax.text(-t_ms[-1] * 0.02, n_hid_shown + n_out / 2, "E_out",
                fontsize=7, va="center", ha="right", clip_on=False)

        ax.set_xlim(0, t_ms[-1])
        ax.set_ylim(cursor + 1, -1)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron (display index)")
        ax.set_yticks([])
        if title:
            ax.set_title(title)
        fig.tight_layout()

    save_both(path, _plot)
