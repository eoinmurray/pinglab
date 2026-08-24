"""EXP094: frozen temporal decoders for trained COBA and PING networks.

One preselected MNIST digit and one Poisson encoding are replayed through the
canonical seed-42 COBA and PING checkpoints. Alternative temporal evidence
functions and output mappings are then computed post hoc from the same output
trajectories. No training occurs.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import theme  # noqa: E402
from helpers.checkpoints import (  # noqa: E402
    public_provenance,
    resolve_checkpoint,
)
from helpers.datasets import load_mnist_split  # noqa: E402
from helpers.paths import artifacts_and_figures, log_runner_event  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp094"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
DESIGN_ASSETS = REPO / "artifacts" / "data" / SLUG
SNN_TOOL = REPO / "tools" / "snn" / "tool.py"
TRAINING_ROOT = Path(
    os.environ.get(
        "PINGLAB_TRAINING_ROOT",
        REPO
        / "runs"
        / "restored"
        / "gold-2"
        / "state"
        / "checkpoints"
        / "current-repair-exp022"
        / "cells",
    )
)
CELLS = {
    "coba": "coba__canonical__seed42",
    "ping": "ping__canonical__seed42",
}
GOLD_STAR_PUBLICATION = "ggs-production-composite-20260821-6d9c38eb"
UPSTREAM_CAMPAIGN = "ggs-fr-repair-20260820-ac6f4988"
CHECKPOINT_ROLE = "final_epoch"
EXPECTED_CHECKPOINTS = {
    "coba": "860d7c9842bc60351603fe08b89a4e78c34e3ae7f19ed588562e02e4eeefffe3",
    "ping": "afe3bce49a89c2dbdac4f986bc3ca65bda91db385b2d58914c9c765075d78a0f",
}
LABEL = 4
SEED = 94
DT_MS = 0.1
DURATION_MS = 200.0
INPUT_RATE_HZ = 25.0
LEAK_TAU_MS = 25.0
WINDOW_MS = 25.0
SOFTMAX_TEMPERATURE = 4.0
SCREEN_PER_CLASS = 10
SCREEN_BATCH_SIZE = 25
SCREEN_SEED = 9400
N_INPUT = 784
N_CLASSES = 10
HAND_ASSETS = (
    "decoder_pipeline.svg",
    "z_mean.svg",
    "z_cumulative.svg",
    "z_leaky.svg",
    "z_window.svg",
    "z_vote.svg",
    "p_softmax.svg",
    "p_softened.svg",
    "p_sigmoid.svg",
)
SCALE = {
    "dataset": "mnist",
    "models": list(CELLS),
    "training_family": "canonical",
    "gold_star_publication": GOLD_STAR_PUBLICATION,
    "seed": SEED,
    "label": LABEL,
    "dt_ms": DT_MS,
    "t_ms": DURATION_MS,
    "input_rate_hz": INPUT_RATE_HZ,
    "trials": 1,
    "screen_trials": N_CLASSES * SCREEN_PER_CLASS,
}


def require_training_bank() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    missing = []
    for model, cell in CELLS.items():
        directory = TRAINING_ROOT / cell
        cell_missing = []
        for name in ("config.json", "metrics.json", "weights_final.pth"):
            if not (directory / name).is_file():
                cell_missing.append(str(directory / name))
        missing.extend(cell_missing)
        if not cell_missing:
            records[model] = resolve_checkpoint(directory, CHECKPOINT_ROLE)
            if records[model]["sha256"] != EXPECTED_CHECKPOINTS[model]:
                raise RuntimeError(
                    f"{cell} does not match the checkpoint published by "
                    f"{GOLD_STAR_PUBLICATION}: {records[model]['sha256']}"
                )
    if missing:
        raise SystemExit(
            "exp094 requires the canonical seed-42 exp022 COBA/PING checkpoint bank "
            "from the latest gold-star publication view. "
            "Set PINGLAB_TRAINING_ROOT to the restored cells directory. Missing:\n  "
            + "\n  ".join(missing)
        )
    return records


def encode_digit(pixels: np.ndarray, *, generator: torch.Generator) -> np.ndarray:
    steps = int(round(DURATION_MS / DT_MS))
    probability = torch.as_tensor(pixels, dtype=torch.float32) * INPUT_RATE_HZ * DT_MS / 1000.0
    spikes = torch.rand(steps, 1, N_INPUT, generator=generator) < probability.reshape(1, 1, -1)
    return spikes.to(torch.float32).numpy()


def encode_digits(pixels: np.ndarray, *, generator: torch.Generator) -> np.ndarray:
    """Encode a batch as one shared (time, trial, input) Poisson tensor."""
    steps = int(round(DURATION_MS / DT_MS))
    probability = (
        torch.as_tensor(pixels, dtype=torch.float32)
        * INPUT_RATE_HZ
        * DT_MS
        / 1000.0
    )
    spikes = torch.rand(
        steps, len(pixels), N_INPUT, generator=generator
    ) < probability.unsqueeze(0)
    return spikes.to(torch.float32).numpy()


def _dense(raster: np.lib.npyio.NpzFile, prefix: str, width: int) -> np.ndarray:
    values = np.zeros((int(raster["T"]), width), dtype=np.float32)
    keep = raster[f"{prefix}_trial"] == 0
    values[raster[f"{prefix}_t"][keep], raster[f"{prefix}_cell"][keep]] = 1.0
    return values


def _dense_trials(
    raster: np.lib.npyio.NpzFile, prefix: str, width: int
) -> np.ndarray:
    values = np.zeros(
        (int(raster["T"]), int(raster["n_trials"]), width), dtype=np.float32
    )
    values[
        raster[f"{prefix}_t"],
        raster[f"{prefix}_trial"],
        raster[f"{prefix}_cell"],
    ] = 1.0
    return values


def run_model(
    model: str,
    input_spikes: np.ndarray,
    checkpoint: dict[str, Any],
) -> dict[str, np.ndarray]:
    directory = TRAINING_ROOT / CELLS[model]
    work = ARTIFACTS / model
    work.mkdir(parents=True, exist_ok=True)
    input_path = work / "input.npz"
    np.savez_compressed(input_path, input_spikes=input_spikes)

    def simulate(name: str, *, readout: str | None = None) -> Path:
        output = work / name
        command = [
            "uv", "run", "python", str(SNN_TOOL), "sim",
            "--load-config", str(directory / "config.json"),
            "--load-weights", str(checkpoint["path"]),
            "--device", "auto",
            "--n-in", str(N_INPUT),
            "--input-file", str(input_path),
            "--outputs", "rasters",
            "--out-dir", str(output),
        ]
        if readout is not None:
            command[command.index("--n-in"):command.index("--n-in")] = [
                "--readout", readout
            ]
        subprocess.run(command, cwd=REPO, check=True)
        return output / "rasters.npz"

    with np.load(simulate("native_mem_mean")) as raster:
        e = _dense(raster, "e", int(raster["n_e"]))
        i = _dense(raster, "i", int(raster["n_i"]))
        native_out = _dense(raster, "out", N_CLASSES)
    with np.load(simulate("counterfactual_spike_count", readout="spike-count")) as raster:
        out = _dense(raster, "out", N_CLASSES)
    w_out = load_output_weights(model, checkpoint)
    config = json.loads((directory / "config.json").read_text())
    pre_voltage = replay_pre_reset_voltage(
        e, native_out, w_out, float(config.get("tau_out_ms", 2.0)), DT_MS
    )
    return {"e": e, "i": i, "out": out, "pre_voltage": pre_voltage}


def load_output_weights(model: str, checkpoint: dict[str, Any]) -> np.ndarray:
    directory = TRAINING_ROOT / CELLS[model]
    weights_dir = ARTIFACTS / model / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv", "run", "python", str(SNN_TOOL), "dump-weights",
            "--load-config", str(directory / "config.json"),
            "--load-weights", str(checkpoint["path"]),
            "--device", "auto",
            "--out-dir", str(weights_dir),
        ],
        cwd=REPO,
        check=True,
    )
    with np.load(weights_dir / "weights_dump.npz") as dumped:
        keys = sorted(
            (key for key in dumped.files if key.startswith("W_ff_") and key.endswith("_trained")),
            key=lambda key: int(key.split("_")[2]),
        )
        return dumped[keys[-1]].copy()


def run_screen_batch(
    model: str,
    input_path: Path,
    checkpoint: dict[str, Any],
    batch_number: int,
    w_out: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run both readouts once for a batch and return dense trial trajectories."""
    directory = TRAINING_ROOT / CELLS[model]
    work = ARTIFACTS / "screen" / f"batch_{batch_number:02d}" / model
    work.mkdir(parents=True, exist_ok=True)

    def simulate(name: str, *, readout: str | None = None) -> Path:
        output = work / name
        command = [
            "uv", "run", "python", str(SNN_TOOL), "sim",
            "--load-config", str(directory / "config.json"),
            "--load-weights", str(checkpoint["path"]),
            "--device", "auto",
            "--n-in", str(N_INPUT),
            "--input-file", str(input_path),
            "--outputs", "rasters",
            "--out-dir", str(output),
        ]
        if readout is not None:
            command[command.index("--n-in"):command.index("--n-in")] = [
                "--readout", readout
            ]
        subprocess.run(command, cwd=REPO, check=True)
        return output / "rasters.npz"

    with np.load(simulate("native_mem_mean")) as raster:
        e = _dense_trials(raster, "e", int(raster["n_e"]))
        i = _dense_trials(raster, "i", int(raster["n_i"]))
        native_out = _dense_trials(raster, "out", N_CLASSES)
    with np.load(
        simulate("counterfactual_spike_count", readout="spike-count")
    ) as raster:
        out = _dense_trials(raster, "out", N_CLASSES)
    config = json.loads((directory / "config.json").read_text())
    pre_voltage = replay_pre_reset_voltage_trials(
        e, native_out, w_out, float(config.get("tau_out_ms", 2.0)), DT_MS
    )
    return {"i": i, "out": out, "pre_voltage": pre_voltage}


def replay_pre_reset_voltage(
    e_spikes: np.ndarray,
    out_spikes: np.ndarray,
    w_out: np.ndarray,
    tau_out_ms: float,
    dt_ms: float,
) -> np.ndarray:
    beta = float(np.exp(-dt_ms / tau_out_ms))
    scale = (1.0 - beta) / dt_ms
    post_reset = np.zeros(out_spikes.shape[1], dtype=np.float64)
    pre_reset = np.zeros_like(out_spikes, dtype=np.float64)
    weights = np.asarray(w_out, dtype=np.float64)
    for t in range(len(e_spikes)):
        drive = np.einsum(
            "...i,io->...o",
            np.asarray(e_spikes[t], dtype=np.float64),
            weights,
            optimize=True,
        )
        if not np.isfinite(drive).all():
            raise RuntimeError("non-finite native readout drive during voltage replay")
        value = beta * post_reset + scale * drive
        pre_reset[t] = value
        post_reset = value - out_spikes[t]
    return pre_reset


def replay_pre_reset_voltage_trials(
    e_spikes: np.ndarray,
    out_spikes: np.ndarray,
    w_out: np.ndarray,
    tau_out_ms: float,
    dt_ms: float,
) -> np.ndarray:
    beta = float(np.exp(-dt_ms / tau_out_ms))
    scale = (1.0 - beta) / dt_ms
    post_reset = np.zeros(out_spikes.shape[1:], dtype=np.float64)
    pre_reset = np.zeros_like(out_spikes, dtype=np.float64)
    weights = np.asarray(w_out, dtype=np.float64)
    for t in range(len(e_spikes)):
        drive = np.einsum(
            "...i,io->...o",
            np.asarray(e_spikes[t], dtype=np.float64),
            weights,
            optimize=True,
        )
        if not np.isfinite(drive).all():
            raise RuntimeError("non-finite native readout drive during voltage replay")
        value = beta * post_reset + scale * drive
        pre_reset[t] = value
        post_reset = value - out_spikes[t]
    return pre_reset


def cumulative_mean(values: np.ndarray) -> np.ndarray:
    return np.cumsum(values, axis=0) / np.arange(1, len(values) + 1)[:, None]


def cumulative_count(spikes: np.ndarray) -> np.ndarray:
    return np.cumsum(spikes, axis=0)


def leaky_count(spikes: np.ndarray, retention: float) -> np.ndarray:
    result = np.zeros_like(spikes, dtype=np.float64)
    state = np.zeros(spikes.shape[1], dtype=np.float64)
    for t, row in enumerate(spikes):
        state = retention * state + row
        result[t] = state
    return result


def window_count(spikes: np.ndarray, steps: int) -> np.ndarray:
    cumulative = np.concatenate(
        [np.zeros((1, spikes.shape[1])), np.cumsum(spikes, axis=0)], axis=0
    )
    result = np.empty_like(spikes, dtype=np.float64)
    for t in range(len(spikes)):
        lo = max(0, t + 1 - steps)
        result[t] = cumulative[t + 1] - cumulative[lo]
    return result


def ping_cycle_boundaries(i_spikes: np.ndarray, dt_ms: float) -> np.ndarray:
    counts = i_spikes.sum(axis=1)
    active = np.flatnonzero(counts > 0)
    if len(active) == 0:
        return np.array([0, len(i_spikes)], dtype=int)
    gap = max(1, int(round(5.0 / dt_ms)))
    splits = np.flatnonzero(np.diff(active) > gap) + 1
    groups = np.split(active, splits)
    centres = np.array([group[np.argmax(counts[group])] for group in groups], dtype=int)
    if len(centres) < 2:
        return np.array([0, len(i_spikes)], dtype=int)
    mids = ((centres[:-1] + centres[1:]) // 2).astype(int)
    return np.unique(np.concatenate(([0], mids, [len(i_spikes)])))


def cumulative_bin_votes(spikes: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    votes = np.zeros(spikes.shape[1], dtype=np.float64)
    result = np.zeros_like(spikes, dtype=np.float64)
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        counts = spikes[start:stop].sum(axis=0)
        votes[int(np.argmax(counts))] += 1.0
        result[start:stop] = votes
    return result


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = values / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def decoder_arrays(
    recordings: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    boundaries = ping_cycle_boundaries(recordings["ping"]["i"], DT_MS)
    retention = float(np.exp(-DT_MS / LEAK_TAU_MS))
    window_steps = int(round(WINDOW_MS / DT_MS))
    decoded: dict[str, dict[str, np.ndarray]] = {}
    for model, arrays in recordings.items():
        counts = cumulative_count(arrays["out"])
        decoded[model] = {
            "mean": cumulative_mean(arrays["pre_voltage"]),
            "cumulative": counts,
            "leaky": leaky_count(arrays["out"], retention),
            "window": window_count(arrays["out"], window_steps),
            "vote": cumulative_bin_votes(arrays["out"], boundaries),
            "softmax": softmax(counts),
            "softened": softmax(counts, SOFTMAX_TEMPERATURE),
            "sigmoid": sigmoid(counts),
        }
    return decoded, boundaries


DECODER_ORDER = (
    "mean",
    "cumulative",
    "leaky",
    "window",
    "vote",
    "softmax",
    "softened",
    "sigmoid",
)
DECODER_LABELS = (
    "Mean",
    "Cumulative",
    "Leaky",
    "Window",
    "Vote",
    "Softmax",
    "Softened",
    "Sigmoid",
)


def balanced_test_indices(labels: np.ndarray, per_class: int) -> np.ndarray:
    """Choose fixed, outcome-blind official-test indices in class order."""
    return np.concatenate(
        [np.flatnonzero(labels == label)[:per_class] for label in range(N_CLASSES)]
    ).astype(np.int64)


def screen_predictions(
    recordings: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Return one final prediction per model, trial, and decoder."""
    retention = float(np.exp(-DT_MS / LEAK_TAU_MS))
    window_steps = int(round(WINDOW_MS / DT_MS))
    trials = recordings["ping"]["out"].shape[1]
    ping_boundaries = [
        ping_cycle_boundaries(recordings["ping"]["i"][:, trial], DT_MS)
        for trial in range(trials)
    ]
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for model, arrays in recordings.items():
        counts = arrays["out"].sum(axis=0)
        leaky = np.zeros((trials, N_CLASSES), dtype=np.float64)
        state = np.zeros_like(leaky)
        for row in arrays["out"]:
            state = retention * state + row
        leaky[:] = state
        window = arrays["out"][-window_steps:].sum(axis=0)
        votes = np.zeros((trials, N_CLASSES), dtype=np.float64)
        for trial, boundaries in enumerate(ping_boundaries):
            for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
                winner = int(np.argmax(arrays["out"][start:stop, trial].sum(axis=0)))
                votes[trial, winner] += 1.0
        evidence = {
            "mean": arrays["pre_voltage"].mean(axis=0),
            "cumulative": counts,
            "leaky": leaky,
            "window": window,
            "vote": votes,
            "softmax": softmax(counts),
            "softened": softmax(counts, SOFTMAX_TEMPERATURE),
            "sigmoid": sigmoid(counts),
        }
        predictions[model] = {
            name: values.argmax(axis=1).astype(np.int64)
            for name, values in evidence.items()
        }
    return predictions


def screening_summary(
    labels: np.ndarray,
    predictions: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": int(len(labels)), "models": {}}
    for model, rows in predictions.items():
        native_correct = rows["mean"] == labels
        model_summary: dict[str, Any] = {}
        for name in DECODER_ORDER:
            predicted = rows[name]
            correct = predicted == labels
            model_summary[name] = {
                "accuracy": float(correct.mean()),
                "native_agreement": float((predicted == rows["mean"]).mean()),
                "transitions": {
                    "correct_to_correct": int(np.sum(native_correct & correct)),
                    "correct_to_wrong": int(np.sum(native_correct & ~correct)),
                    "wrong_to_correct": int(np.sum(~native_correct & correct)),
                    "wrong_to_wrong": int(np.sum(~native_correct & ~correct)),
                },
                "accuracy_by_class": [
                    float(correct[labels == label].mean())
                    for label in range(N_CLASSES)
                ],
            }
        summary["models"][model] = model_summary
    return summary


def _competitor(values: np.ndarray, true_class: int) -> int:
    final = values[-1].copy()
    final[true_class] = -np.inf
    return int(np.argmax(final))


PLOT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.linewidth": 0.75,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "svg.fonttype": "none",
}


def _finish_editorial_figure(
    fig: plt.Figure,
    axes: np.ndarray,
    *,
    ylabel: str,
    legend: list[Line2D],
    readout: str,
) -> None:
    for axis, model in zip(axes, ("COBA", "PING"), strict=True):
        axis.set_title(model, loc="left", pad=12)
        axis.text(
            0,
            1.025,
            readout,
            transform=axis.transAxes,
            color=theme.MUTED,
            fontsize=7.5,
            va="bottom",
        )
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color=theme.RULE_WARM, linewidth=0.55)
        axis.margins(x=0)
    fig.supylabel(ylabel, x=0.012, fontsize=9)
    fig.supxlabel("Time (ms)", y=0.025, fontsize=9)
    fig.legend(
        handles=legend,
        loc="upper right",
        bbox_to_anchor=(0.99, 1.015),
        frameon=False,
        ncol=len(legend),
        handlelength=2.2,
        columnspacing=1.4,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.22, top=0.76, wspace=0.24)


def plot_decoder_pair(
    decoded: dict[str, dict[str, np.ndarray]],
    key: str,
    ylabel: str,
    path: Path,
    *,
    boundaries: np.ndarray | None = None,
) -> None:
    theme.apply()
    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.65), sharex=True)
        time_ms = np.arange(len(decoded["coba"][key])) * DT_MS
        for axis, model in zip(axes, ("coba", "ping"), strict=True):
            values = decoded[model][key]
            competitor = _competitor(values, LABEL)
            axis.plot(
                time_ms,
                values[:, competitor],
                color=theme.INK_BLACK,
                lw=1.15,
                zorder=2,
            )
            axis.plot(
                time_ms,
                values[:, LABEL],
                color=theme.DEEP_RED,
                lw=1.8,
                zorder=3,
            )
            if boundaries is not None:
                for boundary in boundaries[1:-1]:
                    axis.axvline(
                        boundary * DT_MS,
                        color=theme.RULE_WARM,
                        lw=0.55,
                        zorder=1,
                    )
        legend = [
            Line2D([], [], color=theme.DEEP_RED, lw=1.8, label="true class 4"),
            Line2D(
                [],
                [],
                color=theme.INK_BLACK,
                lw=1.15,
                label="strongest final competitor",
            ),
        ]
        readout = "native mem-mean" if key == "mean" else "inference-only spike decoder"
        _finish_editorial_figure(
            fig, axes, ylabel=ylabel, legend=legend, readout=readout
        )
        fig.savefig(path, format="svg", facecolor="white", bbox_inches="tight")
        plt.close(fig)


def plot_mapping_pair(
    decoded: dict[str, dict[str, np.ndarray]], key: str, ylabel: str, path: Path
) -> None:
    theme.apply()
    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.65), sharex=True, sharey=True)
        time_ms = np.arange(len(decoded["coba"][key])) * DT_MS
        for axis, model in zip(axes, ("coba", "ping"), strict=True):
            values = decoded[model][key]
            for class_index in range(N_CLASSES):
                if class_index == LABEL:
                    continue
                axis.plot(
                    time_ms,
                    values[:, class_index],
                    color=theme.GREY_MID,
                    lw=0.7,
                    alpha=0.32,
                    zorder=1,
                )
            axis.plot(
                time_ms,
                values[:, LABEL],
                color=theme.DEEP_RED,
                lw=1.8,
                zorder=3,
            )
            axis.axhline(0.5, color=theme.GREY_LIGHT, lw=0.6, ls="--", zorder=0)
            axis.set_ylim(-0.025, 1.025)
        legend = [
            Line2D([], [], color=theme.DEEP_RED, lw=1.8, label="true class 4"),
            Line2D([], [], color=theme.GREY_MID, lw=0.8, alpha=0.55, label="other classes"),
        ]
        _finish_editorial_figure(
            fig,
            axes,
            ylabel=ylabel,
            legend=legend,
            readout="same cumulative-count evidence",
        )
        fig.savefig(path, format="svg", facecolor="white", bbox_inches="tight")
        plt.close(fig)


def plot_screen_accuracy(summary: dict[str, Any], path: Path) -> None:
    """Headline matrix: absolute accuracy and change from native decoding."""
    values = np.asarray(
        [
            [summary["models"][model][name]["accuracy"] for name in DECODER_ORDER]
            for model in ("coba", "ping")
        ]
    )
    theme.apply()
    with plt.rc_context(PLOT_RC):
        fig, axis = plt.subplots(figsize=(8.2, 2.65))
        image = axis.imshow(values, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        for row in range(2):
            baseline = values[row, 0]
            for column in range(len(DECODER_ORDER)):
                delta = values[row, column] - baseline
                label = f"{values[row, column] * 100:.0f}%"
                if column:
                    label += f"\n{delta * 100:+.0f} pp"
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if values[row, column] < 0.3 else theme.INK_BLACK,
                )
        axis.set_xticks(range(len(DECODER_ORDER)), DECODER_LABELS, rotation=28, ha="right")
        axis.set_yticks((0, 1), ("COBA", "PING"))
        axis.tick_params(length=0)
        axis.axvline(4.5, color="white", lw=3)
        axis.text(2.0, -0.82, "evidence rule z", ha="center", fontsize=8, color=theme.MUTED)
        axis.text(6.0, -0.82, "mapping p on cumulative z", ha="center", fontsize=8, color=theme.MUTED)
        colorbar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.025)
        colorbar.set_label("Accuracy")
        fig.subplots_adjust(left=0.11, right=0.94, bottom=0.31, top=0.76)
        fig.savefig(path, format="svg", facecolor="white", bbox_inches="tight")
        plt.close(fig)


def plot_screen_transitions(summary: dict[str, Any], path: Path) -> None:
    """Show how alternatives preserve, break, or repair native decisions."""
    categories = (
        "correct_to_correct",
        "correct_to_wrong",
        "wrong_to_correct",
        "wrong_to_wrong",
    )
    labels = (
        "correct stays correct",
        "correct becomes wrong",
        "wrong becomes correct",
        "wrong stays wrong",
    )
    colors = ("#2f6b4f", theme.DEEP_RED, "#d8a52b", theme.GREY_MID)
    theme.apply()
    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), sharey=True)
        x = np.arange(len(DECODER_ORDER))
        for axis, model in zip(axes, ("coba", "ping"), strict=True):
            bottom = np.zeros(len(DECODER_ORDER))
            for category, label, color in zip(categories, labels, colors, strict=True):
                values = np.asarray(
                    [
                        summary["models"][model][name]["transitions"][category]
                        for name in DECODER_ORDER
                    ]
                )
                axis.bar(x, values, bottom=bottom, color=color, width=0.76, label=label)
                bottom += values
            axis.set_title(model.upper(), loc="left")
            axis.set_xticks(x, DECODER_LABELS, rotation=32, ha="right")
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color=theme.RULE_WARM, linewidth=0.55)
            axis.set_axisbelow(True)
        axes[0].set_ylabel("Screening images")
        fig.legend(
            *axes[0].get_legend_handles_labels(),
            loc="upper right",
            bbox_to_anchor=(0.99, 1.01),
            frameon=False,
            ncol=2,
            fontsize=8,
        )
        fig.subplots_adjust(left=0.08, right=0.985, bottom=0.28, top=0.72, wspace=0.18)
        fig.savefig(path, format="svg", facecolor="white", bbox_inches="tight")
        plt.close(fig)


def plot_screen_classes(summary: dict[str, Any], path: Path) -> None:
    """Reveal decoder changes that are concentrated in particular digits."""
    theme.apply()
    with plt.rc_context(PLOT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.25), sharex=True, sharey=True)
        last_image = None
        for axis, model in zip(axes, ("coba", "ping"), strict=True):
            accuracy = np.asarray(
                [
                    summary["models"][model][name]["accuracy_by_class"]
                    for name in DECODER_ORDER
                ]
            ).T
            delta = (accuracy - accuracy[:, [0]]) * 100.0
            last_image = axis.imshow(
                delta, cmap="RdBu_r", vmin=-100.0, vmax=100.0, aspect="auto"
            )
            for digit in range(N_CLASSES):
                for decoder in range(len(DECODER_ORDER)):
                    value = int(round(delta[digit, decoder]))
                    axis.text(
                        decoder,
                        digit,
                        f"{value:+d}" if decoder else "0",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color=(
                            "white" if abs(value) >= 60 else theme.INK_BLACK
                        ),
                    )
            axis.set_title(model.upper(), loc="left")
            axis.set_xticks(
                range(len(DECODER_ORDER)), DECODER_LABELS, rotation=32, ha="right"
            )
            axis.set_yticks(range(N_CLASSES), [str(value) for value in range(N_CLASSES)])
            axis.tick_params(length=0)
            axis.axvline(4.5, color="white", lw=2)
        axes[0].set_ylabel("MNIST digit")
        assert last_image is not None
        colorbar = fig.colorbar(last_image, ax=axes, fraction=0.025, pad=0.025)
        colorbar.set_label("Accuracy change from native (percentage points)")
        fig.subplots_adjust(left=0.08, right=0.9, bottom=0.22, top=0.92, wspace=0.12)
        fig.savefig(path, format="svg", facecolor="white", bbox_inches="tight")
        plt.close(fig)


def run_screen(
    pixels: np.ndarray,
    labels: np.ndarray,
    checkpoints: dict[str, dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    """Evaluate a fixed screening sample while keeping peak memory bounded."""
    collected: dict[str, dict[str, list[np.ndarray]]] = {
        model: {name: [] for name in DECODER_ORDER} for model in CELLS
    }
    weights = {
        model: load_output_weights(model, checkpoints[model]) for model in CELLS
    }
    generator = torch.Generator().manual_seed(SCREEN_SEED)
    screen_root = ARTIFACTS / "screen"
    screen_root.mkdir(parents=True, exist_ok=True)
    for batch_number, start in enumerate(range(0, len(labels), SCREEN_BATCH_SIZE)):
        stop = min(start + SCREEN_BATCH_SIZE, len(labels))
        input_path = screen_root / f"input_{batch_number:02d}.npz"
        input_spikes = encode_digits(pixels[start:stop], generator=generator)
        np.savez(input_path, input_spikes=input_spikes)
        del input_spikes
        recordings = {
            model: run_screen_batch(
                model,
                input_path,
                checkpoints[model],
                batch_number,
                weights[model],
            )
            for model in CELLS
        }
        batch_predictions = screen_predictions(recordings)
        for model in CELLS:
            for name in DECODER_ORDER:
                collected[model][name].append(batch_predictions[model][name])
        input_path.unlink()
    return {
        model: {
            name: np.concatenate(parts)
            for name, parts in model_predictions.items()
        }
        for model, model_predictions in collected.items()
    }


def main() -> None:
    checkpoints = require_training_bank()
    run_id = next_run_id(SLUG)
    log_runner_event(SLUG, "started", run_id=run_id)
    started = time.monotonic()
    coba_config = json.loads((TRAINING_ROOT / CELLS["coba"] / "config.json").read_text())
    _, x_test, _, y_test = load_mnist_split(max_samples=int(coba_config["max_samples"]))
    sample_index = int(np.flatnonzero(y_test == LABEL)[0])
    pixels = np.asarray(x_test[sample_index], dtype=np.float32)
    input_spikes = encode_digit(pixels, generator=torch.Generator().manual_seed(SEED))
    recordings = {
        model: run_model(model, input_spikes, checkpoints[model]) for model in CELLS
    }
    decoded, boundaries = decoder_arrays(recordings)
    screen_indices = balanced_test_indices(y_test, SCREEN_PER_CLASS)
    screen_labels = np.asarray(y_test[screen_indices], dtype=np.int64)
    screen = run_screen(
        np.asarray(x_test[screen_indices], dtype=np.float32),
        screen_labels,
        checkpoints,
    )
    screen_summary = screening_summary(screen_labels, screen)

    with published_run(SLUG, run_id, scale=SCALE) as (_scratch, staging):
        for name in HAND_ASSETS:
            shutil.copy2(DESIGN_ASSETS / name, staging / name)
        plot_decoder_pair(decoded, "mean", "mean pre-reset voltage", staging / "measured_z_mean.svg")
        plot_decoder_pair(decoded, "cumulative", "cumulative output spikes", staging / "measured_z_cumulative.svg")
        plot_decoder_pair(decoded, "leaky", "leaky spike evidence", staging / "measured_z_leaky.svg")
        plot_decoder_pair(decoded, "window", "spikes in window", staging / "measured_z_window.svg")
        plot_decoder_pair(decoded, "vote", "cumulative votes", staging / "measured_z_vote.svg", boundaries=boundaries)
        plot_mapping_pair(decoded, "softmax", "softmax class share", staging / "measured_p_softmax.svg")
        plot_mapping_pair(decoded, "softened", "softened class share", staging / "measured_p_softened.svg")
        plot_mapping_pair(decoded, "sigmoid", "independent sigmoid score", staging / "measured_p_sigmoid.svg")
        plot_screen_accuracy(screen_summary, staging / "screen_accuracy.svg")
        plot_screen_transitions(screen_summary, staging / "screen_transitions.svg")
        plot_screen_classes(screen_summary, staging / "screen_classes.svg")
        np.savez_compressed(
            staging / "measurements.npz",
            pixels=pixels,
            input_spikes=input_spikes,
            cycle_boundaries=boundaries,
            **{
                f"{model}_{name}": values
                for model, rows in decoded.items()
                for name, values in rows.items()
            },
        )
        np.savez_compressed(
            staging / "screen_predictions.npz",
            indices=screen_indices,
            labels=screen_labels,
            **{
                f"{model}_{name}": values
                for model, rows in screen.items()
                for name, values in rows.items()
            },
        )
        payload = {
            "status": "complete",
            "training_source": {
                "publication_run_id": GOLD_STAR_PUBLICATION,
                "upstream_campaign_id": UPSTREAM_CAMPAIGN,
                "cells": CELLS,
            },
            "selection": {"label": LABEL, "official_test_index": sample_index, "outcome_blind": True},
            "shared_input": {"seed": SEED, "identical_between_models": True},
            "checkpoint_provenance": {
                model: public_provenance(record) for model, record in checkpoints.items()
            },
            "decoder": {
                "native_readout": "mem-mean",
                "spike_readout_intervention": "spike-count",
                "shared_hidden_drive": True,
                "leak_tau_ms": LEAK_TAU_MS,
                "window_ms": WINDOW_MS,
                "softmax_temperature": SOFTMAX_TEMPERATURE,
                "p_input": "the same cumulative output-spike count vector for every mapping",
                "cycle_boundaries_ms": (boundaries * DT_MS).tolist(),
            },
            "final_winners": {
                model: {name: int(values[-1].argmax()) for name, values in rows.items()}
                for model, rows in decoded.items()
            },
            "screen": {
                "design": {
                    "sample": "first ten official-test examples per class",
                    "selection_is_outcome_blind": True,
                    "per_class": SCREEN_PER_CLASS,
                    "batch_size": SCREEN_BATCH_SIZE,
                    "poisson_seed": SCREEN_SEED,
                    "interpretation": "screening result, not a population estimate",
                },
                **screen_summary,
            },
            "duration_s": round(time.monotonic() - started, 3),
        }
        (staging / "numbers.json").write_text(json.dumps(payload, indent=2) + "\n")
    log_runner_event(SLUG, "completed", run_id=run_id)
    print(f"exp094 complete: {run_id}")


if __name__ == "__main__":
    main()
