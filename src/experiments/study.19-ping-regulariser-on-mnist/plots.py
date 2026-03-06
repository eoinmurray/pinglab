from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from pinglab.plots.styles import save_both


def save_stacked_lines(
    path: Path,
    x,
    series: dict[str, list],
    *,
    suptitle: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Save vertically stacked subplots, one per series, sharing the x-axis."""
    n = len(series)
    def _plot() -> None:
        fig, axes = plt.subplots(n, 1, figsize=(6, 2.5 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, (label, y) in zip(axes, series.items()):
            ax.plot(x, y, linewidth=1.0)
            ax.set_ylabel(ylabel)
            ax.set_title(label)
        axes[-1].set_xlabel(xlabel)
        fig.suptitle(suptitle, y=1.01)
        fig.tight_layout()

    save_both(path, _plot)


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
    """Grid of output-layer rasters, one panel per sample."""
    if digit_labels is None:
        digit_labels = list(range(len(spike_tensors)))

    n_out = spike_tensors[0].shape[1]
    T = spike_tensors[0].shape[0]
    t_ms = np.arange(T) * dt
    n = len(spike_tensors)

    # Choose grid layout based on number of samples
    if n <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 2, 5

    def _plot() -> None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(6, 6), sharex=True, sharey=True)
        axes = axes.flatten()

        for idx, (spikes, label) in enumerate(zip(spike_tensors, digit_labels)):
            ax = axes[idx]
            arr = spikes.numpy() if hasattr(spikes, "numpy") else np.asarray(spikes)
            t_idx, n_idx = np.nonzero(arr)
            ax.scatter(t_ms[t_idx], n_idx, s=3, marker=".", rasterized=True)
            ax.set_title(str(label), fontsize=8)
            ax.set_yticks(range(n_out))
            ax.set_yticklabels([str(i) for i in range(n_out)], fontsize=5)
            ax.set_ylim(-0.5, n_out - 0.5)
            ax.tick_params(axis="x", labelsize=6)

        # Hide unused panels
        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        for ax in axes[:n]:
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
    n_hid_sample: int = 0,
    n_input_sample: int = 0,
    input_ext: np.ndarray | None = None,
    n_input: int = 0,
) -> None:
    """Raster showing spike activity across layers.

    Order (top to bottom): E_in, E_hid, E_out, I_global.
    E_in is shown only when ``input_ext`` is provided (Poisson current array).
    """
    arr = spikes.numpy() if hasattr(spikes, "numpy") else np.asarray(spikes)
    T = arr.shape[0]
    t_ms = np.arange(T) * dt

    in_start = int(pop_idx["E_in"]["start"])
    in_stop  = int(pop_idx["E_in"]["stop"])
    hid_start = int(pop_idx["E_hid"]["start"])
    hid_stop  = int(pop_idx["E_hid"]["stop"])
    out_start = int(pop_idx["E_out"]["start"])
    out_stop  = int(pop_idx["E_out"]["stop"])

    has_i = "I_global" in pop_idx
    if has_i:
        i_start = int(pop_idx["I_global"]["start"])
        i_stop  = int(pop_idx["I_global"]["stop"])
        n_i = i_stop - i_start

    n_in = in_stop - in_start
    n_hid = hid_stop - hid_start
    n_out = out_stop - out_start

    in_sample = np.arange(n_in)
    hid_sample = np.arange(hid_start, hid_stop)
    n_in_shown = n_in
    n_hid_shown = n_hid

    def _plot() -> None:
        fig, ax = plt.subplots(figsize=(6, 6))

        cursor = 0
        label_positions = {}

        # E_in (input spikes from external current array)
        if input_ext is not None and n_input > 0:
            in_arr = (input_ext[:, :n_input] != 0).astype(np.float32)
            in_arr = in_arr[:, in_sample]
            t_idx, n_idx = np.nonzero(in_arr)
            ax.scatter(t_ms[t_idx], cursor + n_idx, s=0.5, marker=".", rasterized=True,
                       color="C2", label=f"E_in (n={n_in})")
            label_positions["E_in"] = cursor + n_in_shown / 2
            cursor += n_in_shown
            ax.axhline(cursor, linewidth=0.7, linestyle="--", alpha=0.4)

        # E_hid
        hid_arr = arr[:, hid_sample]
        t_idx, n_idx = np.nonzero(hid_arr)
        ax.scatter(t_ms[t_idx], cursor + n_idx, s=2, marker=".", rasterized=True,
                   color="C0", label=f"E_hid (n={n_hid})")
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

        ax.axhline(cursor, linewidth=0.7, linestyle="--", alpha=0.4)

        # I_global at bottom (if present)
        if has_i:
            i_arr = arr[:, i_start:i_stop]
            t_idx, n_idx = np.nonzero(i_arr)
            ax.scatter(t_ms[t_idx], cursor + n_idx, s=2, marker=".", rasterized=True,
                       color="red", label=f"I_global (n={n_i})")
            label_positions["I_global"] = cursor + n_i / 2
            cursor += n_i

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


def save_input_raster(
    path: Path,
    ext: np.ndarray,
    *,
    dt: float,
    n_input: int,
    n_show: int = 100,
    title: str | None = None,
) -> None:
    """Raster of Poisson input spike trains for a subset of input neurons."""
    arr = ext[:, :n_input]  # [T, n_input]
    T = arr.shape[0]
    t_ms = np.arange(T) * dt

    # Sample evenly-spaced input neurons
    indices = np.linspace(0, n_input - 1, min(n_show, n_input), dtype=int)
    arr_sub = arr[:, indices]

    def _plot() -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        t_idx, n_idx = np.nonzero(arr_sub)
        ax.scatter(t_ms[t_idx], n_idx, s=0.3, marker=".", rasterized=True, color="C0")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(f"Input neuron (sampled {len(indices)}/{n_input})")
        ax.set_xlim(0, t_ms[-1])
        ax.set_ylim(-0.5, len(indices) - 0.5)
        if title:
            ax.set_title(title)
        fig.tight_layout()

    save_both(path, _plot)


def save_voltage_traces(
    path: Path,
    voltage_trace: np.ndarray,
    *,
    dt: float,
    out_start: int,
    out_stop: int,
    title: str | None = None,
    v_th: float = -50.0,
) -> None:
    """Plot membrane voltage traces for output neurons over time."""
    n_out = out_stop - out_start
    T = voltage_trace.shape[0]
    t_ms = np.arange(T) * dt
    out_v = voltage_trace[:, out_start:out_stop]  # [T, n_out]

    def _plot() -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        for i in range(n_out):
            ax.plot(t_ms, out_v[:, i], linewidth=0.8, label=str(i), alpha=0.8)
        ax.axhline(v_th, linewidth=0.7, linestyle="--", color="grey", alpha=0.5, label="V_th")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Membrane voltage (mV)")
        ax.legend(fontsize=6, ncol=2, loc="lower right", title="Neuron", title_fontsize=6)
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
    n_classes: int = 10,
    class_labels: list[str] | None = None,
) -> None:
    """Save a confusion matrix heatmap."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t][p] += 1

    if class_labels is None:
        class_labels = [str(i) for i in range(n_classes)]

    def _plot() -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_labels, fontsize=8)
        ax.set_yticklabels(class_labels, fontsize=8)

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > thresh else "black")

        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()

    save_both(path, _plot)
