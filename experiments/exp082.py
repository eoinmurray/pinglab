"""EXP082: streaming inference with variable-rate-trained PING weights.

This is the successor to exp048.  It consumes the planned exp022 PING cells
trained with per-presentation variable input rates and the output-LIF
`spike-count` readout.  It reports three protocols:

1. a stream whose presentation duration and input rate both vary;
2. a 200-ms input-rate psychometric curve; and
3. a presentation-duration by input-rate accuracy map.

A matched 200-ms stream is retained as an internal validation artifact.

The runner is intentionally executable only after exp022 has produced all
three variable-rate checkpoints.  Until then it fails before creating a run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp022 import (  # noqa: E402
    training_run_cell,
    training_run_values,
)
from helpers import theme  # noqa: E402
from helpers.checkpoints import (  # noqa: E402
    cache_tag,
    checkpoint_policy,
    checkpoint_provenance,
    resolve_checkpoint,
)
from helpers.datasets import load_mnist_split  # noqa: E402
from helpers.paths import (  # noqa: E402
    artifacts_and_figures,
    log_runner_event,
    runner_paths,
)
from helpers.run_dirs import (
    finalize_prepared_run,  # noqa: E402
    preserve_active_view,  # noqa: E402
)
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402

SLUG = "exp082"
ANALYSIS_PURPOSE = "deployment_performance"
CHECKPOINT_POLICY = checkpoint_policy(ANALYSIS_PURPOSE)
CHECKPOINT_ROLE = CHECKPOINT_POLICY["role"]
RUN_PATHS = runner_paths(SLUG)
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
SNN_TOOL = REPO / "tools" / "snn" / "tool.py"
TRAINING_ROOT = Path(
    os.environ.get("PINGLAB_TRAINING_ROOT", REPO / "temp" / "experiments" / "exp022")
)

SEEDS = training_run_values("TR-06", "seed")
SMOKE = os.environ.get("PINGLAB_SMOKE") == "1"
_TR06_RATE_SETS = training_run_values("TR-06", "input_rates_hz")
if len(_TR06_RATE_SETS) != 1:
    raise ValueError(f"TR-06 must register one input-rate set, got {_TR06_RATE_SETS}")
TRAINING_RATES_HZ = tuple(_TR06_RATE_SETS[0])
PSYCHOMETRIC_RATES_HZ = (
    (0.5, 5.0, 25.0) if SMOKE else TRAINING_RATES_HZ
)
DURATIONS_MS = (50.0, 200.0) if SMOKE else (25.0, 50.0, 100.0, 200.0)
MATCHED_DURATION_MS = 200.0
MATCHED_RATE_HZ = 5.0
N_CLASSES = 10
N_INPUT = 784
N_HEADLINE_DIGITS = 5
STREAMS_PER_CELL = int(
    os.environ.get("PINGLAB_EXP082_STREAMS_PER_CELL", 1 if SMOKE else 40)
)
DIGITS_PER_STREAM = int(
    os.environ.get("PINGLAB_EXP082_DIGITS_PER_STREAM", 3 if SMOKE else 5)
)
STREAM_BATCH_SIZE = int(
    os.environ.get("PINGLAB_EXP082_STREAM_BATCH_SIZE", 1 if SMOKE else 5)
)
DT_MS = 0.1

if STREAMS_PER_CELL < 1 or DIGITS_PER_STREAM < 1 or STREAM_BATCH_SIZE < 1:
    raise ValueError("exp082 stream, digit, and stream-batch counts must be positive")

EVALUATION_PROFILE = (
    "smoke"
    if SMOKE
    else (
        "pilot"
        if "PINGLAB_EXP082_STREAMS_PER_CELL" in os.environ
        or "PINGLAB_EXP082_DIGITS_PER_STREAM" in os.environ
        else "production"
    )
)

VARIABLE_STREAM = (
    (200.0, 0.5),
    (50.0, 25.0),
    (100.0, 2.0),
    (25.0, 10.0),
    (200.0, 5.0),
)
SINGLE_TRIAL_TRANSITION_WINDOW_MS = (91.5, 94.5)
CLASS_PROBABILITY_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)

SCALE = {
    "dataset": "mnist",
    "t_ms": MATCHED_DURATION_MS,
    "dt_ms": DT_MS,
    "seeds": len(SEEDS),
    "cells": len(DURATIONS_MS) * len(PSYCHOMETRIC_RATES_HZ),
    "grid": f"{len(DURATIONS_MS)} duration × {len(PSYCHOMETRIC_RATES_HZ)} rate",
}

MEASUREMENTS_FILE = "measurements.npz"


def training_cell_name(seed: int) -> str:
    return training_run_cell("TR-06", seed=seed)["name"]


def training_dir(seed: int) -> Path:
    return TRAINING_ROOT / training_cell_name(seed)


def require_training_bank() -> None:
    if RUN_PATHS.isolated and not os.environ.get("PINGLAB_TRAINING_ROOT"):
        raise RuntimeError("isolated exp082 requires explicit PINGLAB_TRAINING_ROOT")
    missing = []
    for seed in SEEDS:
        directory = training_dir(seed)
        cell_missing = []
        for filename in ("config.json", "metrics.json", "weights.pth", "weights_final.pth"):
            if not (directory / filename).exists():
                cell_missing.append(str(directory / filename))
        missing.extend(cell_missing)
        if not cell_missing:
            resolve_checkpoint(directory, CHECKPOINT_ROLE)
    if missing:
        joined = "\n  ".join(missing)
        raise SystemExit(
            "exp082 requires the exp022 variable-rate training bank:\n  " + joined
        )


def load_eval(seed: int) -> tuple[Path, dict[str, Any], np.ndarray, np.ndarray]:
    directory = training_dir(seed)
    config = json.loads((directory / "config.json").read_text())
    readout = config.get("readout_mode", config.get("readout"))
    if readout != "spike-count":
        raise SystemExit(
            f"{directory} has readout {readout!r}; exp082 requires output-LIF 'spike-count'"
        )
    _, x_test, _, y_test = load_mnist_split(max_samples=int(config["max_samples"]))
    return directory, config, x_test, y_test


def encode_segment(
    pixels: np.ndarray,
    duration_ms: float,
    rate_hz: float,
    generator: torch.Generator,
) -> torch.Tensor:
    steps = int(round(duration_ms / DT_MS))
    images = torch.as_tensor(pixels, dtype=torch.float32).reshape(-1, N_INPUT)
    probability = images * rate_hz * DT_MS / 1000.0
    return (
        torch.rand(steps, len(images), N_INPUT, generator=generator)
        < probability.unsqueeze(0)
    ).to(torch.float32)


def encode_stream(
    pixels: np.ndarray,
    conditions: tuple[tuple[float, float], ...],
    generator: torch.Generator,
) -> torch.Tensor:
    if len(pixels) != len(conditions):
        raise ValueError("one (duration, rate) condition is required per digit")
    return torch.cat(
        [
            encode_segment(pixels[i : i + 1], duration, rate, generator)
            for i, (duration, rate) in enumerate(conditions)
        ],
        dim=0,
    )


def run_spikes(
    directory: Path,
    input_spikes: torch.Tensor,
    tag: str,
    *,
    reset_steps: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    checkpoint = resolve_checkpoint(directory, CHECKPOINT_ROLE)
    out_dir = (ARTIFACTS / "stream" / f"{directory.name}__{cache_tag(checkpoint)}" / tag).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "input.npz"
    readout_reset = np.zeros(len(input_spikes), dtype=np.bool_)
    for step in reset_steps:
        if not 0 <= step < len(readout_reset):
            raise ValueError(f"readout reset step {step} is outside the input stream")
        readout_reset[step] = True
    np.savez_compressed(
        input_path,
        input_spikes=input_spikes.cpu().numpy(),
        readout_reset=readout_reset,
    )
    subprocess.run(
        [
            "uv", "run", "python", str(SNN_TOOL), "sim",
            "--load-config", str((directory / "config.json").resolve()),
            "--load-weights", str(checkpoint["path"]),
            # Saved training configs record the accelerator used on Wilkes.
            # Inference must select the current host rather than inheriting
            # that non-portable device identity.
            "--device", "auto",
            "--n-in", str(N_INPUT),
            "--input-file", str(input_path),
            "--outputs", "rasters",
            "--out-dir", str(out_dir),
        ],
        cwd=REPO,
        check=True,
    )
    raster = np.load(out_dir / "rasters.npz")
    total_steps = int(raster["T"])

    def dense(prefix: str, width: int) -> np.ndarray:
        keep = raster[f"{prefix}_trial"] == 0
        values = np.zeros((total_steps, width), dtype=np.int8)
        values[raster[f"{prefix}_t"][keep], raster[f"{prefix}_cell"][keep]] = 1
        return values

    required = {"out_trial", "out_t", "out_cell"}
    missing = sorted(required - set(raster.files))
    if missing:
        raise RuntimeError(
            "exp082 requires tools/snn spike-count output-spike rasters; "
            f"missing {missing} in {out_dir / 'rasters.npz'}"
        )
    return (
        dense("e", int(raster["n_e"])),
        dense("i", int(raster["n_i"])),
        dense("out", N_CLASSES),
    )


def run_spike_summary(
    directory: Path,
    input_spikes: torch.Tensor,
    tag: str,
    *,
    reset_steps: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Run a batched statistical stream without retaining full rasters."""
    checkpoint = resolve_checkpoint(directory, CHECKPOINT_ROLE)
    work_root = ARTIFACTS / ".work"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{tag}__", dir=work_root) as raw:
        out_dir = Path(raw).resolve()
        input_path = out_dir / "input.npz"
        readout_reset = np.zeros(len(input_spikes), dtype=np.bool_)
        for step in reset_steps:
            if not 0 <= step < len(readout_reset):
                raise ValueError(f"readout reset step {step} is outside the input stream")
            readout_reset[step] = True
        np.savez_compressed(
            input_path,
            input_spikes=input_spikes.cpu().numpy(),
            readout_reset=readout_reset,
        )
        subprocess.run(
            [
                "uv", "run", "python", str(SNN_TOOL), "sim",
                "--load-config", str((directory / "config.json").resolve()),
                "--load-weights", str(checkpoint["path"]),
                "--device", "auto",
                "--n-in", str(N_INPUT),
                "--input-file", str(input_path),
                "--outputs", "spike_summary",
                "--out-dir", str(out_dir),
            ],
            cwd=REPO,
            check=True,
        )
        with np.load(out_dir / "spike_summary.npz") as summary:
            return {key: summary[key].copy() for key in summary.files}


def spike_count_logits(
    spikes_out: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    """Output-LIF spike counts over exactly ``[start, stop)``."""
    if stop <= start:
        raise ValueError("spike-count window must contain at least one timestep")
    return spikes_out[start:stop].sum(axis=0)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def output_activity_summary(
    spikes_out: np.ndarray,
    boundaries: list[int],
) -> dict[str, Any]:
    """Summarize output activity over declared presentation windows."""
    counts = [
        spikes_out[start:stop].sum(axis=0)
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    per_presentation = np.asarray(counts, dtype=np.int64)
    totals = per_presentation.sum(axis=1)
    return {
        "n_presentations": len(counts),
        "total_output_spikes": int(totals.sum()),
        "spikes_per_presentation": totals.tolist(),
        "silent_presentations": int((totals == 0).sum()),
        "silent_fraction": float((totals == 0).mean()),
        "class_spike_totals": per_presentation.sum(axis=0).tolist(),
    }


def single_trial_from_stream(
    stream: dict[str, Any], segment_index: int = 0,
) -> dict[str, Any]:
    """Extract one independently readable presentation from a stream result."""
    start = int(stream["boundaries"][segment_index])
    stop = int(stream["boundaries"][segment_index + 1])
    trial = {
        "conditions": [stream["conditions"][segment_index]],
        "pixels": np.asarray(stream["pixels"])[segment_index : segment_index + 1],
        "labels": [stream["labels"][segment_index]],
        "predictions": [stream["predictions"][segment_index]],
        "correct": [stream["correct"][segment_index]],
        "boundaries": [0, stop - start],
        **{
            key: np.asarray(stream[key])[start:stop]
            for key in ("spikes_e", "spikes_i", "spikes_out", "probabilities")
        },
    }
    trial["output_activity"] = output_activity_summary(
        trial["spikes_out"], trial["boundaries"]
    )
    return trial


def first_correct_trial_from_stream(stream: dict[str, Any]) -> dict[str, Any]:
    """Select the first successful presentation for explanatory figures."""
    try:
        segment_index = list(stream["correct"]).index(1)
    except ValueError as error:
        raise RuntimeError("matched stream contains no correctly classified trial") from error
    return single_trial_from_stream(stream, segment_index)


def pick_digits(x_test: np.ndarray, y_test: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = rng.permutation(N_CLASSES)[:n]
    indices = [int(rng.choice(np.flatnonzero(y_test == label))) for label in classes]
    return x_test[indices], y_test[indices]


def evaluate_stream(
    directory: Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
    conditions: tuple[tuple[float, float], ...],
    seed: int,
    tag: str,
) -> dict[str, Any]:
    pixels, labels = pick_digits(x_test, y_test, len(conditions), seed)
    encoded = encode_stream(pixels, conditions, torch.Generator().manual_seed(seed + 1))
    segment_steps = [int(round(duration / DT_MS)) for duration, _ in conditions]
    reset_steps = tuple(np.cumsum([0, *segment_steps[:-1]]).tolist())
    spikes_e, spikes_i, spikes_out = run_spikes(
        directory, encoded, tag, reset_steps=reset_steps
    )
    probabilities = np.zeros((len(spikes_e), N_CLASSES), dtype=np.float32)
    predictions = []
    correct = []
    boundaries = [0]
    cursor = 0
    for label, (duration, _) in zip(labels, conditions, strict=True):
        steps = int(round(duration / DT_MS))
        stop = cursor + steps
        for timestep in range(cursor, stop):
            probabilities[timestep] = softmax(
                spike_count_logits(spikes_out, cursor, timestep + 1)
            )
        prediction = int(np.argmax(spike_count_logits(spikes_out, cursor, stop)))
        predictions.append(prediction)
        correct.append(int(prediction == label))
        cursor = stop
        boundaries.append(cursor)
    return {
        "conditions": [list(item) for item in conditions],
        "pixels": pixels,
        "labels": labels.tolist(),
        "predictions": predictions,
        "correct": correct,
        "boundaries": boundaries,
        "spikes_e": spikes_e,
        "spikes_i": spikes_i,
        "spikes_out": spikes_out,
        "probabilities": probabilities,
        "output_activity": output_activity_summary(spikes_out, boundaries),
    }


def evaluate_cell(seed: int, duration_ms: float, rate_hz: float) -> dict[str, Any]:
    directory, _, x_test, y_test = load_eval(seed)
    checkpoint = resolve_checkpoint(directory, CHECKPOINT_ROLE)
    condition_path = condition_result_path(seed, duration_ms, rate_hz, checkpoint)
    condition_dir = condition_path.parent
    if condition_path.is_file():
        try:
            cached = json.loads(condition_path.read_text())
        except (OSError, json.JSONDecodeError):
            cached = None
        if (
            isinstance(cached, dict)
            and cached.get("n_total") == STREAMS_PER_CELL * DIGITS_PER_STREAM
            and cached.get("stream_batch_size") == STREAM_BATCH_SIZE
        ):
            return cached

    rng = np.random.default_rng(82_000 + seed + int(duration_ms * 10) + int(rate_hz * 100))
    n_correct = 0
    n_total = 0
    n_output_spikes = 0
    n_silent_presentations = 0
    n_e_spikes = 0
    n_i_spikes = 0
    total_duration_s = 0.0
    class_spike_totals = np.zeros(N_CLASSES, dtype=np.int64)
    conditions = tuple((duration_ms, rate_hz) for _ in range(DIGITS_PER_STREAM))
    reset_steps = tuple(
        digit_index * int(round(duration_ms / DT_MS))
        for digit_index in range(DIGITS_PER_STREAM)
    )
    encoded_streams: list[torch.Tensor] = []
    label_streams: list[np.ndarray] = []

    def flush_batch(first_stream_index: int) -> None:
        nonlocal n_correct, n_total, n_output_spikes, n_silent_presentations
        nonlocal n_e_spikes, n_i_spikes, total_duration_s, class_spike_totals
        if not encoded_streams:
            return
        batched = torch.cat(encoded_streams, dim=1)
        summary = run_spike_summary(
            directory,
            batched,
            f"cell_d{duration_ms:g}_r{rate_hz:g}_s{first_stream_index}",
            reset_steps=reset_steps,
        )
        output_counts = np.asarray(summary["out_counts"], dtype=np.int64)
        labels = np.stack(label_streams)
        if output_counts.shape != (*labels.shape, N_CLASSES):
            raise RuntimeError(
                f"unexpected output-count shape {output_counts.shape}; "
                f"expected {(*labels.shape, N_CLASSES)}"
            )
        n_correct += int((output_counts.argmax(axis=2) == labels).sum())
        n_total += int(labels.size)
        totals = output_counts.sum(axis=2)
        n_output_spikes += int(totals.sum())
        n_silent_presentations += int((totals == 0).sum())
        class_spike_totals += output_counts.sum(axis=(0, 1))
        n_e_spikes += int(np.asarray(summary["e_counts"]).sum())
        n_i_spikes += int(np.asarray(summary["i_counts"]).sum())
        total_duration_s += labels.size * duration_ms / 1000.0
        encoded_streams.clear()
        label_streams.clear()

    first_stream_index = 0
    for stream_index in range(STREAMS_PER_CELL):
        indices = rng.choice(len(y_test), DIGITS_PER_STREAM, replace=False)
        encoded_streams.append(
            encode_stream(
                x_test[indices],
                conditions,
                torch.Generator().manual_seed(82_000 + seed * 100 + stream_index),
            )
        )
        label_streams.append(y_test[indices])
        if len(encoded_streams) == STREAM_BATCH_SIZE:
            flush_batch(first_stream_index)
            first_stream_index = stream_index + 1
    flush_batch(first_stream_index)

    result = {
        "seed": seed,
        "duration_ms": duration_ms,
        "rate_hz": rate_hz,
        "stream_batch_size": STREAM_BATCH_SIZE,
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": n_correct / n_total,
        "output_spikes_per_presentation": n_output_spikes / n_total,
        "silent_fraction": n_silent_presentations / n_total,
        "class_spike_totals": class_spike_totals.tolist(),
        "rate_e_hz": n_e_spikes / (1024 * total_duration_s),
        "rate_i_hz": n_i_spikes / (256 * total_duration_s),
    }
    condition_dir.mkdir(parents=True, exist_ok=True)
    temporary = condition_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(condition_path)
    return result


def _number_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _number_from_tag(value: str) -> float:
    return float(value.replace("p", "."))


def condition_job_id(seed: int, duration_ms: float, rate_hz: float) -> str:
    return f"seed{seed}__d{_number_tag(duration_ms)}__r{_number_tag(rate_hz)}"


def parse_condition_job_id(job_id: str) -> tuple[int, float, float]:
    parts = job_id.split("__")
    if (
        len(parts) != 3
        or not parts[0].startswith("seed")
        or not parts[1].startswith("d")
        or not parts[2].startswith("r")
    ):
        raise ValueError(f"invalid exp082 condition job: {job_id}")
    return (
        int(parts[0].removeprefix("seed")),
        _number_from_tag(parts[1].removeprefix("d")),
        _number_from_tag(parts[2].removeprefix("r")),
    )


def condition_result_path(
    seed: int,
    duration_ms: float,
    rate_hz: float,
    checkpoint: dict[str, Any] | None = None,
) -> Path:
    directory = training_dir(seed)
    resolved = checkpoint or resolve_checkpoint(directory, CHECKPOINT_ROLE)
    return (
        ARTIFACTS
        / "conditions"
        / f"{directory.name}__{cache_tag(resolved)}"
        / f"d{duration_ms:g}_r{rate_hz:g}.json"
    )


def infer_jobs() -> list[str]:
    return [
        condition_job_id(seed, duration, rate)
        for duration in DURATIONS_MS
        for rate in PSYCHOMETRIC_RATES_HZ
        for seed in SEEDS
    ]


def job_is_done(job_id: str) -> bool:
    seed, duration, rate = parse_condition_job_id(job_id)
    path = condition_result_path(seed, duration, rate)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("n_total") == STREAMS_PER_CELL * DIGITS_PER_STREAM
        and payload.get("stream_batch_size") == STREAM_BATCH_SIZE
    )


def run_infer_job(job_id: str) -> None:
    seed, duration, rate = parse_condition_job_id(job_id)
    evaluate_cell(seed, duration, rate)


def grid_output_preflight(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report and reject a wholly silent grid-level output readout."""
    n_presentations = sum(int(row["n_total"]) for row in rows)
    n_silent = sum(
        round(float(row["silent_fraction"]) * int(row["n_total"]))
        for row in rows
    )
    total_spikes = sum(
        float(row["output_spikes_per_presentation"]) * int(row["n_total"])
        for row in rows
    )
    summary = {
        "n_presentations": n_presentations,
        "total_output_spikes": int(round(total_spikes)),
        "silent_presentations": n_silent,
        "silent_fraction": n_silent / n_presentations,
    }
    if summary["total_output_spikes"] == 0:
        raise RuntimeError(
            "exp082 scientific preflight failed: output readout is silent "
            "across the complete duration-rate grid"
        )
    return summary


def plot_stream(result: dict[str, Any], path: Path, run_id: str) -> None:
    theme.apply()
    spikes_e = result["spikes_e"]
    spikes_i = result["spikes_i"]
    probabilities = result["probabilities"]
    time_ms = np.arange(len(spikes_e)) * DT_MS
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 5.4), sharex=True, constrained_layout=True)
    e_t, e_n = np.nonzero(spikes_e[:, :200])
    i_t, i_n = np.nonzero(spikes_i[:, :64])
    axes[0].scatter(e_t * DT_MS, e_n, s=1, color=theme.INK_BLACK)
    axes[1].scatter(i_t * DT_MS, i_n, s=2, color=theme.DEEP_RED)
    for label in range(N_CLASSES):
        axes[2].plot(time_ms, probabilities[:, label], lw=0.9, label=str(label))
    for boundary in result["boundaries"][1:-1]:
        for axis in axes:
            axis.axvline(boundary * DT_MS, color=theme.GREY_MID, lw=0.7, ls=":")
    axes[0].set_ylabel("E neuron")
    axes[1].set_ylabel("I neuron")
    axes[2].set(xlabel="time (ms)", ylabel="class probability", ylim=(0, 1))
    axes[2].legend(ncol=5, frameon=False, fontsize=7)
    stamp_figure(fig, run_id)
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)


def plot_stream_headline(
    result: dict[str, Any],
    path: Path,
    run_id: str,
    *,
    annotate_final_counts: bool = False,
) -> None:
    """Exp048-Figure-1-style streaming headline for one or more trials."""
    theme.apply()
    conditions = result["conditions"]
    boundaries = np.asarray(result["boundaries"], dtype=int)
    starts = boundaries[:-1]
    stops = boundaries[1:]
    starts_ms = starts * DT_MS
    stops_ms = stops * DT_MS
    total_ms = stops_ms[-1]
    time_ms = np.arange(len(result["spikes_e"])) * DT_MS
    labels = result["labels"]
    predictions = result["predictions"]

    fig = plt.figure(figsize=(6.9, 5.33), dpi=150)
    grid = fig.add_gridspec(
        4, 1, height_ratios=[1.35, 2.2, 1.2, 2.0], hspace=0.18,
    )

    thumbnail_axis = fig.add_subplot(grid[0])
    thumbnail_axis.set(xlim=(0, total_ms), ylim=(0, 1))
    thumbnail_axis.set_xticks([])
    thumbnail_axis.set_yticks([])
    for spine in thumbnail_axis.spines.values():
        spine.set_visible(False)
    rates = np.asarray([condition[1] for condition in conditions], dtype=float)
    log_rates = np.log(rates)
    for index, ((duration_ms, rate_hz), start_ms, stop_ms) in enumerate(
        zip(conditions, starts_ms, stops_ms, strict=True)
    ):
        width = (stop_ms - start_ms) / total_ms * 0.88
        left = thumbnail_axis.get_position().x0 + (
            start_ms / total_ms
            * (thumbnail_axis.get_position().x1 - thumbnail_axis.get_position().x0)
        )
        inset = fig.add_axes([  # ty: ignore[no-matching-overload]
            left,
            thumbnail_axis.get_position().y0 + 0.005,
            width,
            thumbnail_axis.get_position().height - 0.02,
        ])
        alpha = 1.0
        if log_rates.max() > log_rates.min():
            alpha = 0.2 + 0.8 * (
                np.log(rate_hz) - log_rates.min()
            ) / (log_rates.max() - log_rates.min())
        inset.imshow(
            np.asarray(result["pixels"])[index].reshape(28, 28),
            cmap="Greys",
            interpolation="nearest",
            aspect="auto",
            alpha=alpha,
        )
        inset.set_xticks([])
        inset.set_yticks([])
        colour = (
            theme.INK_BLACK
            if predictions[index] == labels[index]
            else theme.DEEP_RED
        )
        inset.text(
            0.05,
            0.95,
            f"{labels[index]}→{predictions[index]}",
            transform=inset.transAxes,
            ha="left",
            va="top",
            fontsize=theme.SIZE_LABEL,
            color="white",
            weight="bold",
            bbox=dict(
                facecolor=colour,
                edgecolor="none",
                boxstyle="round,pad=0.2",
                alpha=0.95,
            ),
        )
        thumbnail_axis.text(
            (start_ms + stop_ms) / 2,
            1.02 if index % 2 == 0 else 1.15,
            f"{duration_ms:g} ms · {rate_hz:g} Hz",
            transform=thumbnail_axis.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=theme.SIZE_LABEL - 2,
            color=theme.MUTED,
            clip_on=False,
        )

    raster_specs = (
        (result["spikes_e"][:, :200], "E cell", theme.INK_BLACK, 2.0, 1024),
        (result["spikes_i"][:, :64], "I cell", theme.DEEP_RED, 2.0, 256),
    )
    for row, (spikes, label, colour, size, population) in enumerate(
        raster_specs, start=1
    ):
        axis = fig.add_subplot(grid[row])
        spike_times, neurons = np.nonzero(spikes)
        axis.scatter(
            spike_times * DT_MS,
            neurons,
            s=size,
            c=colour,
            marker="|",
            linewidths=0.4,
        )
        axis.set(xlim=(0, total_ms), ylim=(0, spikes.shape[1]))
        axis.set_yticks([0, spikes.shape[1]], ["0", f"{population}"])
        axis.set_ylabel(label, fontsize=theme.SIZE_LABEL)
        axis.tick_params(axis="x", labelbottom=False)
        axis.spines[["top", "right"]].set_visible(False)
        for boundary in starts_ms[1:]:
            axis.axvline(boundary, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.7)

    evidence_axis = fig.add_subplot(grid[3])
    probabilities = result["probabilities"]
    for class_index in range(N_CLASSES):
        evidence_axis.plot(
            time_ms,
            probabilities[:, class_index],
            color=theme.GREY_MID,
            lw=0.6,
            alpha=0.45,
        )
    final_counts = np.asarray(result["spikes_out"]).sum(axis=0).astype(int)
    final_winner = int(final_counts.argmax())
    if annotate_final_counts and final_winner not in set(labels):
        evidence_axis.plot(
            time_ms,
            probabilities[:, final_winner],
            color=theme.INK_BLACK,
            lw=1.5,
        )
    for index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        evidence_axis.plot(
            time_ms[start:stop],
            probabilities[start:stop, labels[index]],
            color=theme.DEEP_RED,
            lw=2.2,
        )
    for boundary in starts_ms[1:]:
        evidence_axis.axvline(
            boundary, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.7,
        )
    evidence_axis.axhline(0.5, color=theme.GREY_MID, lw=0.5, ls="--", alpha=0.6)
    evidence_axis.set(
        xlim=(0, total_ms),
        ylim=(0, 1),
        xlabel="time (ms)",
        ylabel=r"$p_c(u)$ · Eq. (2)",
    )
    evidence_axis.set_yticks(CLASS_PROBABILITY_TICKS)
    evidence_axis.spines[["top", "right"]].set_visible(False)
    if annotate_final_counts:
        true_class = int(labels[0])
        other_counts = final_counts.copy()
        other_counts[final_winner] = -1
        runner_up_class = int(other_counts.argmax())
        margin = int(final_counts[final_winner] - final_counts[runner_up_class])
        if true_class == final_winner:
            summary = (
                f"correct class {true_class}: {final_counts[true_class]} spikes · "
                f"runner-up {runner_up_class}: {final_counts[runner_up_class]} spikes · "
                f"margin {margin}"
            )
        else:
            summary = (
                f"true {true_class}: {final_counts[true_class]} spikes · "
                f"winner {final_winner}: {final_counts[final_winner]} spikes · "
                f"margin {margin}"
            )
        evidence_axis.text(
            0.01,
            0.98,
            summary,
            transform=evidence_axis.transAxes,
            ha="left",
            va="top",
            fontsize=theme.SIZE_ANNOTATION,
            color=theme.INK_BLACK,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82),
        )

    stamp_figure(fig, run_id)
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)


def plot_variable_headline(result: dict[str, Any], path: Path, run_id: str) -> None:
    """Plot the variable-condition stream used as the exp048 successor."""
    plot_stream_headline(result, path, run_id)


def plot_single_trial(result: dict[str, Any], path: Path, run_id: str) -> None:
    """Plot one selected presentation to explain spike-count evidence."""
    plot_stream_headline(result, path, run_id, annotate_final_counts=True)


def plot_single_trial_transition(
    result: dict[str, Any], path: Path, run_id: str,
) -> None:
    """Resolve the output spikes behind the selected evidence transition."""
    theme.apply()
    start_ms, stop_ms = SINGLE_TRIAL_TRANSITION_WINDOW_MS
    start = int(round(start_ms / DT_MS))
    stop = int(round(stop_ms / DT_MS)) + 1
    spikes_out = np.asarray(result["spikes_out"])
    counts = spikes_out.cumsum(axis=0)
    probabilities = np.asarray(result["probabilities"])
    time_ms = np.arange(len(spikes_out)) * DT_MS
    true_class = int(result["labels"][0])
    winner = int(counts[-1].argmax())

    fig, axes = plt.subplots(
        3, 1, figsize=(6.9, 4.8), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": (1.0, 1.4, 1.7)},
    )
    spike_times, spike_classes = np.nonzero(spikes_out[start:stop])
    spike_times_ms = (spike_times + start) * DT_MS
    spike_colours = [
        theme.DEEP_RED if class_index == true_class
        else theme.INK_BLACK if class_index == winner
        else theme.GREY_MID
        for class_index in spike_classes
    ]
    axes[0].scatter(
        spike_times_ms, spike_classes, c=spike_colours,
        marker="|", s=48, linewidths=1.2,
    )
    axes[0].set_ylabel("output class")
    axes[0].set_yticks(range(N_CLASSES))

    for class_index in range(N_CLASSES):
        colour = theme.GREY_MID
        width = 0.7
        alpha = 0.45
        if class_index == winner:
            colour, width, alpha = theme.INK_BLACK, 1.6, 1.0
        if class_index == true_class:
            colour, width, alpha = theme.DEEP_RED, 2.0, 1.0
        axes[1].step(
            time_ms[start:stop], counts[start:stop, class_index],
            where="post", color=colour, lw=width, alpha=alpha,
        )
        axes[2].step(
            time_ms[start:stop], probabilities[start:stop, class_index],
            where="post", color=colour, lw=width, alpha=alpha,
        )
    axes[1].set_ylabel(r"$z_c(u)$ · Eq. (1)")
    axes[2].set(
        xlabel="time (ms)", ylabel=r"$p_c(u)$ · Eq. (2)", ylim=(0, 1),
    )
    axes[2].set_yticks(CLASS_PROBABILITY_TICKS)
    axes[2].axhline(0.5, color=theme.GREY_MID, lw=0.5, ls="--", alpha=0.6)
    for axis in axes:
        axis.set_xlim(start_ms, stop_ms)
        axis.spines[["top", "right"]].set_visible(False)
    title = (
        f"true and winning class {true_class} (red)"
        if true_class == winner
        else f"true class {true_class} (red) · eventual winner {winner} (black)"
    )
    axes[0].set_title(title, loc="left", fontsize=theme.SIZE_LABEL)
    stamp_figure(fig, run_id)
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)


def plot_psychometric(rows: list[dict[str, Any]], path: Path, run_id: str) -> None:
    theme.apply()
    plt.rcParams["svg.hashsalt"] = "pinglab-exp082"
    rates = sorted({row["rate_hz"] for row in rows})
    means = []
    sems = []
    for rate in rates:
        values = np.asarray([row["accuracy"] for row in rows if row["rate_hz"] == rate])
        means.append(float(values.mean()))
        sems.append(float(values.std(ddof=1) / np.sqrt(len(values))))
    fig, axis = plt.subplots(figsize=(6.5, 3.66), constrained_layout=True)
    axis.errorbar(rates, means, yerr=sems, color=theme.INK_BLACK, marker="o", capsize=3)
    axis.set_xscale("log")
    axis.set_xticks(rates)
    axis.set_xticklabels([f"{rate:g}" for rate in rates])
    axis.set(xlabel="maximum-pixel input rate (Hz)", ylabel="accuracy", ylim=(0, 1))
    axis.spines[["top", "right"]].set_visible(False)
    stamp_figure(fig, run_id)
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


def plot_duration_rate_summary(
    rows: list[dict[str, Any]], path: Path, run_id: str,
) -> None:
    """Exp048-Figure-2-style duration×rate map plus the 200-ms psychometric."""
    theme.apply()
    durations = list(DURATIONS_MS)
    rates = list(PSYCHOMETRIC_RATES_HZ)
    grid = np.zeros((len(rates), len(durations)), dtype=np.float32)
    sem = np.zeros(len(rates), dtype=np.float32)
    for rate_index, rate in enumerate(rates):
        for duration_index, duration in enumerate(durations):
            values = np.asarray([
                row["accuracy"] for row in rows
                if row["rate_hz"] == rate and row["duration_ms"] == duration
            ])
            grid[rate_index, duration_index] = values.mean()
            if duration == MATCHED_DURATION_MS:
                sem[rate_index] = values.std(ddof=1) / np.sqrt(len(values))
    fig, (map_axis, curve_axis) = plt.subplots(
        1, 2, figsize=(6.5, 3.25), constrained_layout=True,
        gridspec_kw={"width_ratios": (1.15, 1)},
    )
    image = map_axis.imshow(grid, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
    map_axis.set_xticks(range(len(durations)), [f"{value:g}" for value in durations])
    map_axis.set_yticks(range(len(rates)), [f"{value:g}" for value in rates])
    map_axis.set(xlabel="presentation = readout (ms)", ylabel="input rate (Hz)")
    fig.colorbar(image, ax=map_axis, label="accuracy")
    curve_axis.errorbar(rates, grid[:, -1], yerr=sem, color=theme.INK_BLACK, marker="o", capsize=3)
    curve_axis.set_xscale("log")
    curve_axis.set_xticks(rates, [f"{value:g}" for value in rates])
    curve_axis.set(xlabel="input rate (Hz)", ylabel="accuracy at 200 ms", ylim=(0, 1))
    curve_axis.spines[["top", "right"]].set_visible(False)
    stamp_figure(fig, run_id)
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)


def save_measurements(
    matched: dict[str, Any],
    variable: dict[str, Any],
    single_trial: dict[str, Any] | None = None,
) -> None:
    """Save the array-valued results needed to reproduce the stream figures."""
    results = [("matched", matched), ("variable", variable)]
    if single_trial is not None:
        results.append(("single_trial", single_trial))
    np.savez_compressed(
        FIGURES / MEASUREMENTS_FILE,
        **{
            f"{name}_{key}": result[key]
            for name, result in results
            for key in (
                "pixels", "spikes_e", "spikes_i", "spikes_out", "probabilities",
            )
        },
    )


def replot_results(numbers_path: Path, measurements_path: Path) -> None:
    """Regenerate every exp082 figure from saved inference measurements."""
    payload = json.loads(numbers_path.read_text())
    with np.load(measurements_path) as arrays:
        pixels_by_stream: dict[str, np.ndarray] = {}
        if all(f"{name}_pixels" in arrays for name in ("matched", "variable")):
            pixels_by_stream = {
                name: arrays[f"{name}_pixels"] for name in ("matched", "variable")
            }
        else:
            _, x_test, _, y_test = load_mnist_split()
            pixels_by_stream = {
                name: pick_digits(x_test, y_test, N_HEADLINE_DIGITS, seed)[0]
                for name, seed in (("matched", 82), ("variable", 83))
            }
        streams = {
            name: {
                **payload[f"{name}_stream"],
                "pixels": pixels_by_stream[name],
                **{
                    key: arrays[f"{name}_{key}"]
                    for key in (
                        "spikes_e", "spikes_i", "spikes_out", "probabilities",
                    )
                },
            }
            for name in ("matched", "variable")
        }
    run_id = payload.get("run_id", "replot")
    single_trial = first_correct_trial_from_stream(streams["matched"])
    plot_single_trial(
        single_trial, FIGURES / "single_trial.png", run_id
    )
    plot_single_trial_transition(
        single_trial, FIGURES / "single_trial_transition.png", run_id
    )
    plot_stream(streams["matched"], FIGURES / "matched_stream.png", run_id)
    plot_variable_headline(
        streams["variable"], FIGURES / "variable_stream.png", run_id
    )
    rows = payload["grid_per_seed"]
    psychometric = payload["duration_200ms_psychometric"]
    plot_psychometric(psychometric, FIGURES / "psychometric_200ms.svg", run_id)
    plot_duration_rate_summary(rows, FIGURES / "duration_rate_summary.png", run_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replot",
        action="store_true",
        help="regenerate figures from numbers.json and measurements.npz",
    )
    return parser.parse_args()


@preserve_active_view(SLUG)
def main() -> None:
    args = parse_args()
    if args.replot:
        replot_results(FIGURES / "numbers.json", FIGURES / MEASUREMENTS_FILE)
        return
    require_training_bank()
    run_id = next_run_id(SLUG)
    log_runner_event(SLUG, "started", run_id=run_id)
    prepare_run_dirs(SLUG, run_id, wipe=False, make_artifacts=True, scale=SCALE, host="local")
    started = time.monotonic()
    directory, config, x_test, y_test = load_eval(SEEDS[0])

    matched_conditions = tuple(
        (MATCHED_DURATION_MS, MATCHED_RATE_HZ) for _ in range(N_HEADLINE_DIGITS)
    )
    matched = evaluate_stream(directory, x_test, y_test, matched_conditions, 82, "matched")
    variable = evaluate_stream(directory, x_test, y_test, VARIABLE_STREAM, 83, "variable")
    single_trial = first_correct_trial_from_stream(matched)
    save_measurements(matched, variable, single_trial)
    plot_single_trial(single_trial, FIGURES / "single_trial.png", run_id)
    plot_single_trial_transition(
        single_trial, FIGURES / "single_trial_transition.png", run_id
    )
    plot_stream(matched, FIGURES / "matched_stream.png", run_id)
    plot_variable_headline(variable, FIGURES / "variable_stream.png", run_id)

    # Pending tools/snn support is deliberately encountered here, after the two
    # single-stream figures prove checkpoint and readout compatibility.
    rows = [
        evaluate_cell(seed, duration, rate)
        for duration in DURATIONS_MS
        for rate in PSYCHOMETRIC_RATES_HZ
        for seed in SEEDS
    ]
    output_preflight = grid_output_preflight(rows)
    psychometric = [row for row in rows if row["duration_ms"] == MATCHED_DURATION_MS]
    plot_psychometric(psychometric, FIGURES / "psychometric_200ms.svg", run_id)
    plot_duration_rate_summary(rows, FIGURES / "duration_rate_summary.png", run_id)

    payload = {
        "status": "complete",
        "run_id": run_id,
        "profile": EVALUATION_PROFILE,
        "training_source": "exp022 variable-rate streaming training",
        "training_cells": [training_cell_name(seed) for seed in SEEDS],
        "checkpoint_policy": CHECKPOINT_POLICY,
        "checkpoint_provenance": checkpoint_provenance(
            [training_dir(seed) for seed in SEEDS], CHECKPOINT_ROLE
        ),
        "readout": {
            "mode": "spike-count",
            "definition": "total output-LIF spikes over the matched presentation window",
            "reported_activity": "output spike rate in Hz may be derived as count divided by window duration in seconds",
        },
        "config": {
            "seeds": list(SEEDS),
            "training_rates_hz": list(TRAINING_RATES_HZ),
            "psychometric_rates_hz": list(PSYCHOMETRIC_RATES_HZ),
            "durations_ms": list(DURATIONS_MS),
            "matched_duration_ms": MATCHED_DURATION_MS,
            "matched_rate_hz": MATCHED_RATE_HZ,
            "streams_per_cell": STREAMS_PER_CELL,
            "digits_per_stream": DIGITS_PER_STREAM,
            "stream_batch_size": STREAM_BATCH_SIZE,
            "digits_per_seed_cell": STREAMS_PER_CELL * DIGITS_PER_STREAM,
            "dt_ms": float(config["dt"]),
        },
        "matched_stream": {key: value for key, value in matched.items() if not isinstance(value, np.ndarray)},
        "variable_stream": {key: value for key, value in variable.items() if not isinstance(value, np.ndarray)},
        "single_trial": {
            key: value
            for key, value in single_trial.items()
            if not isinstance(value, np.ndarray)
        },
        "scientific_preflight": {
            "matched_stream": matched["output_activity"],
            "variable_stream": variable["output_activity"],
            "evaluation_grid": output_preflight,
        },
        "grid_per_seed": rows,
        "duration_200ms_psychometric": psychometric,
        "duration_s": time.monotonic() - started,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(payload, indent=2) + "\n")
    log_runner_event(SLUG, "completed", run_id=run_id, quantitative_rows=len(rows))
    print(f"exp082 complete: {run_id}")
    finalize_prepared_run(SLUG, run_id)


if __name__ == "__main__":
    main()
