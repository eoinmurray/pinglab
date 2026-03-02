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
        if len(x) <= 20:
            ax.set_xticks(x)
        fig.tight_layout()

    save_both(path, _plot)


def save_raster_grid(
    path: Path,
    spike_tensors,
    *,
    dt: float,
    digit_labels=None,
    suptitle: str | None = None,
) -> None:
    """2x5 grid of output-layer rasters, one panel per digit class."""
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
    spikes,
    *,
    dt: float,
    pop_idx: dict,
    title: str | None = None,
    n_hid_sample: int = 80,
) -> None:
    """Raster showing E_hid, I_global (if present), and E_out activity."""
    arr = spikes.numpy() if hasattr(spikes, "numpy") else np.asarray(spikes)
    T = arr.shape[0]
    t_ms = np.arange(T) * dt

    hid_start = int(pop_idx["E_hid"]["start"])
    hid_stop  = int(pop_idx["E_hid"]["stop"])
    out_start = int(pop_idx["E_out"]["start"])
    out_stop  = int(pop_idx["E_out"]["stop"])

    has_i = "I_global" in pop_idx
    if has_i:
        i_start = int(pop_idx["I_global"]["start"])
        i_stop  = int(pop_idx["I_global"]["stop"])
        n_i = i_stop - i_start

    n_hid = hid_stop - hid_start
    n_out = out_stop - out_start

    hid_sample = np.linspace(hid_start, hid_stop - 1, min(n_hid_sample, n_hid), dtype=int)
    n_hid_shown = len(hid_sample)

    def _plot() -> None:
        fig, ax = plt.subplots(figsize=(6, 6))

        cursor = 0
        label_positions = {}

        # I_global on top (if present)
        if has_i:
            i_arr = arr[:, i_start:i_stop]
            t_idx, n_idx = np.nonzero(i_arr)
            ax.scatter(t_ms[t_idx], cursor + n_idx, s=2, marker=".", rasterized=True,
                       color="red", label=f"I_global (n={n_i})")
            label_positions["I_global"] = cursor + n_i / 2
            cursor += n_i
            ax.axhline(cursor, linewidth=0.7, linestyle="--", alpha=0.4)

        # E_hid
        hid_arr = arr[:, hid_sample]
        t_idx, n_idx = np.nonzero(hid_arr)
        ax.scatter(t_ms[t_idx], cursor + n_idx, s=2, marker=".", rasterized=True,
                   color="C0", label=f"E_hid (n={n_hid_shown}/{n_hid})")
        label_positions["E_hid"] = cursor + n_hid_shown / 2
        cursor += n_hid_shown

        ax.axhline(cursor, linewidth=0.7, linestyle="--", alpha=0.4)

        # E_out
        out_arr = arr[:, out_start:out_stop]
        t_idx, n_idx = np.nonzero(out_arr)
        ax.scatter(t_ms[t_idx], cursor + n_idx, s=5, marker="|", rasterized=True,
                   color="C0", label="E_out")

        for i in range(n_out):
            ax.text(t_ms[-1] * 1.01, cursor + i, str(i), fontsize=6,
                    va="center", ha="left", clip_on=False)

        label_positions["E_out"] = cursor + n_out / 2
        cursor += n_out

        for lbl, y_pos in label_positions.items():
            ax.text(-t_ms[-1] * 0.02, y_pos, lbl,
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


def save_confusion_matrix(
    path: Path,
    labels: list[int],
    preds: list[int],
    *,
    title: str = "Confusion Matrix",
) -> None:
    """Save a 10x10 confusion matrix heatmap."""
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(labels, preds):
        cm[t][p] += 1

    def _plot() -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(10):
            for j in range(10):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > thresh else "black")

        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()

    save_both(path, _plot)
