"""Plot generation for study.16 — loads NPZ data, produces all artifacts.

Can be run standalone:
    uv run python plots.py --data-dir data/<run_id>
    uv run python plots.py  # auto-discovers latest run
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from pinglab.plots.styles import save_both


# ── individual plot functions ─────────────────────────────────────────────


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
        fig, axes = plt.subplots(n, 1, figsize=(6, 2.5 * n + 0.5), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, (label, y) in zip(axes, series.items()):
            ax.plot(x, y, linewidth=1.0)
            ax.set_ylabel(ylabel)
            ax.set_title(label)
        axes[-1].set_xlabel(xlabel)
        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

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


# ── main entry point ──────────────────────────────────────────────────────


def main(
    data_dir: Path | str,
    artifacts_dir: Path | str | None = None,
) -> None:
    """Generate all plots from saved NPZ data.

    Args:
        data_dir: Path to a run data directory containing NPZ files + config.
        artifacts_dir: Where to write plot PNGs. Defaults to the standard
            ``_artifacts/study.16-e-prop/`` location.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from pinglab.io.compiler import compile_graph
    from pinglab.io.graph_renderer import save_graph_diagram

    data_dir = Path(data_dir)
    if artifacts_dir is None:
        from settings import ARTIFACTS_ROOT
        artifacts_dir = ARTIFACTS_ROOT / Path(__file__).parent.name
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config_path = data_dir / "config.json"
    with config_path.open(encoding="utf-8") as f:
        spec = json.load(f)

    plan = compile_graph(spec)
    pop_idx = plan["population_index"]
    out_start = int(pop_idx["E_out"]["start"])
    out_stop = int(pop_idx["E_out"]["stop"])
    n_input = int(pop_idx["E_in"]["stop"] - pop_idx["E_in"]["start"])
    sim_cfg = spec.get("sim", {})
    dt = float(sim_cfg.get("dt_ms", 1.0))
    v_th = float(spec.get("biophysics", {}).get("V_th", -50.0))
    epochs = int(spec.get("meta", {}).get("epochs", 5))

    # Graph diagram
    save_graph_diagram(spec, artifacts_dir / "graph")

    # Load training metrics
    tm = np.load(data_dir / "training_metrics.npz")
    all_iter_losses = tm["all_iter_losses"]
    all_iter_accs = tm["all_iter_accs"]
    test_losses = tm["test_losses"]
    test_accuracies = tm["test_accuracies"]

    iters_x = list(range(1, len(all_iter_losses) + 1))
    epochs_x = list(range(1, len(test_losses) + 1))

    save_line(artifacts_dir / "loss_train", x=iters_x, y=all_iter_losses.tolist(),
              title="Train Loss", xlabel="Iteration", ylabel="Cross-entropy loss")
    save_line(artifacts_dir / "loss_test", x=epochs_x, y=test_losses.tolist(),
              title="Test Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    save_line(artifacts_dir / "accuracy", x=iters_x, y=(all_iter_accs * 100).tolist(),
              title="Train Accuracy", xlabel="Iteration", ylabel="Accuracy (%)")

    # Firing rates
    rates_series = {}
    for name in ("E_in", "E_hid", "E_out", "I_global"):
        key = f"rates_{name}"
        if key in tm and len(tm[key]) > 0:
            rates_series[name] = tm[key].tolist()
    if rates_series:
        save_stacked_lines(
            artifacts_dir / "firing_rates",
            x=iters_x,
            series=rates_series,
            suptitle="Mean Firing Rate per Layer",
            xlabel="Iteration",
            ylabel="Hz",
        )

    # Grad norms
    grad_series = {}
    for name in ("W_ee", "W_ei", "W_ie"):
        key = f"grad_norms_{name}"
        if key in tm and len(tm[key]) > 0:
            grad_series[name] = tm[key].tolist()
    if grad_series:
        save_stacked_lines(
            artifacts_dir / "grad_norms",
            x=iters_x,
            series=grad_series,
            suptitle="Gradient Norm per Weight Matrix",
            xlabel="Iteration",
            ylabel="||grad||",
        )

    # Load inference data
    inf = np.load(data_dir / "inference_data.npz")
    all_preds = inf["all_preds"]
    all_labels = inf["all_labels"]
    class_accs = inf["class_accs"]
    available_digits = inf["available_digits"]

    # Per-class accuracy
    save_line(
        artifacts_dir / "accuracy_per_class",
        x=list(range(10)),
        y=class_accs.tolist(),
        title="Per-class Test Accuracy",
        xlabel="Digit",
        ylabel="Accuracy (%)",
    )

    # Confusion matrix
    save_confusion_matrix(
        artifacts_dir / "confusion",
        all_labels.tolist(),
        all_preds.tolist(),
        title="Confusion Matrix",
    )

    # Raster plots
    out_spike_tensors = []
    for d in available_digits:
        full_spikes = inf[f"full_spikes_{d:02d}"]
        out_spike_tensors.append(full_spikes[:, out_start:out_stop])

    save_raster_grid(
        artifacts_dir / "raster_output_all_all",
        out_spike_tensors,
        dt=dt,
        digit_labels=available_digits.tolist(),
        suptitle="Output layer spikes per digit class (trained PING, e-prop)",
    )

    for d in available_digits:
        save_raster_layers(
            artifacts_dir / f"raster_layers_digit_{d:02d}",
            inf[f"full_spikes_{d:02d}"],
            dt=dt,
            pop_idx=pop_idx,
            input_ext=inf[f"input_{d:02d}"],
            n_input=n_input,
            title=f"All layers — digit {d}",
        )

    for d in available_digits:
        save_input_raster(
            artifacts_dir / f"raster_input_digit_{d:02d}",
            inf[f"input_{d:02d}"],
            dt=dt,
            n_input=n_input,
            title=f"Poisson input — digit {d}",
        )

    for d in available_digits:
        save_voltage_traces(
            artifacts_dir / f"voltage_output_digit_{d:02d}",
            inf[f"voltage_{d:02d}"],
            dt=dt,
            out_start=out_start,
            out_stop=out_stop,
            title=f"Output neuron voltages — digit {d}",
            v_th=v_th,
        )

    # Copy metadata files to artifacts
    for name in ("config.json", "results.json", "train.log"):
        src = data_dir / name
        if src.exists():
            shutil.copy2(src, artifacts_dir / name)

    print(f"Plots saved to {artifacts_dir}")


def _find_latest_run(experiment_dir: Path) -> Path:
    """Find the most recently modified run data directory."""
    data_root = experiment_dir / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"No data directory at {data_root}")
    runs = [d for d in data_root.iterdir() if d.is_dir() and d.name != "MNIST"]
    if not runs:
        raise FileNotFoundError(f"No run directories in {data_root}")
    return max(runs, key=lambda d: d.stat().st_mtime)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate plots from saved run data")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Path to run data dir (default: auto-discover latest)")
    parser.add_argument("--artifacts-dir", type=Path, default=None,
                        help="Where to write plots (default: _artifacts/study.16-e-prop/)")
    args = parser.parse_args()

    experiment_dir = Path(__file__).parent.resolve()
    if args.data_dir is None:
        data_dir = _find_latest_run(experiment_dir)
        print(f"Auto-discovered latest run: {data_dir}")
    else:
        data_dir = args.data_dir

    main(data_dir=data_dir, artifacts_dir=args.artifacts_dir)
