"""EXP082: streaming inference with variable-rate-trained PING weights.

This is the successor to exp048.  It consumes the planned exp022 PING cells
trained with per-presentation variable input rates and the output-LIF
`spike-count` readout.  It evaluates four protocols:

1. a matched 200-ms presentation/readout stream;
2. a stream whose presentation duration and input rate both vary;
3. a 200-ms input-rate psychometric curve; and
4. a presentation-duration by input-rate accuracy map.

The runner is intentionally executable only after exp022 has produced all
three variable-rate checkpoints.  Until then it fails before creating a run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
    os.environ.get("PINGLAB_EXP082_STREAMS_PER_CELL", 1 if SMOKE else 20)
)
DIGITS_PER_STREAM = int(
    os.environ.get("PINGLAB_EXP082_DIGITS_PER_STREAM", 3 if SMOKE else 5)
)
DT_MS = 0.1

if STREAMS_PER_CELL < 1 or DIGITS_PER_STREAM < 1:
    raise ValueError("exp082 stream and digit counts must both be positive")

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
    rng = np.random.default_rng(82_000 + seed + int(duration_ms * 10) + int(rate_hz * 100))
    n_correct = 0
    n_total = 0
    n_output_spikes = 0
    n_silent_presentations = 0
    n_e_spikes = 0
    n_i_spikes = 0
    total_duration_s = 0.0
    class_spike_totals = np.zeros(N_CLASSES, dtype=np.int64)
    for stream_index in range(STREAMS_PER_CELL):
        indices = rng.choice(len(y_test), DIGITS_PER_STREAM, replace=False)
        conditions = tuple(
            (duration_ms, rate_hz) for _ in range(DIGITS_PER_STREAM)
        )
        spikes = encode_stream(
            x_test[indices], conditions,
            torch.Generator().manual_seed(82_000 + seed * 100 + stream_index),
        )
        spikes_e, spikes_i, spikes_out = run_spikes(
            directory,
            spikes,
            f"cell_d{duration_ms:g}_r{rate_hz:g}_s{stream_index}",
            reset_steps=tuple(
                digit_index * int(round(duration_ms / DT_MS))
                for digit_index in range(DIGITS_PER_STREAM)
            ),
        )
        segment_steps = int(round(duration_ms / DT_MS))
        for digit_index, label in enumerate(y_test[indices]):
            start = digit_index * segment_steps
            stop = start + segment_steps
            logits = spike_count_logits(spikes_out, start, stop)
            n_correct += int(np.argmax(logits) == label)
            n_total += 1
            spike_total = int(logits.sum())
            n_output_spikes += spike_total
            n_silent_presentations += int(spike_total == 0)
            class_spike_totals += logits.astype(np.int64)
        n_e_spikes += int(spikes_e.sum())
        n_i_spikes += int(spikes_i.sum())
        total_duration_s += DIGITS_PER_STREAM * duration_ms / 1000.0
    return {
        "seed": seed,
        "duration_ms": duration_ms,
        "rate_hz": rate_hz,
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": n_correct / n_total,
        "output_spikes_per_presentation": n_output_spikes / n_total,
        "silent_fraction": n_silent_presentations / n_total,
        "class_spike_totals": class_spike_totals.tolist(),
        "rate_e_hz": n_e_spikes / (1024 * total_duration_s),
        "rate_i_hz": n_i_spikes / (256 * total_duration_s),
    }


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


def save_measurements(matched: dict[str, Any], variable: dict[str, Any]) -> None:
    """Save the array-valued results needed to reproduce the stream figures."""
    np.savez_compressed(
        FIGURES / MEASUREMENTS_FILE,
        **{
            f"{name}_{key}": result[key]
            for name, result in (("matched", matched), ("variable", variable))
            for key in ("spikes_e", "spikes_i", "spikes_out", "probabilities")
        },
    )


def replot_results(numbers_path: Path, measurements_path: Path) -> None:
    """Regenerate every exp082 figure from saved inference measurements."""
    payload = json.loads(numbers_path.read_text())
    with np.load(measurements_path) as arrays:
        streams = {
            name: {
                **payload[f"{name}_stream"],
                **{
                    key: arrays[f"{name}_{key}"]
                    for key in ("spikes_e", "spikes_i", "spikes_out", "probabilities")
                },
            }
            for name in ("matched", "variable")
        }
    run_id = payload.get("run_id", "replot")
    plot_stream(streams["matched"], FIGURES / "matched_stream.png", run_id)
    plot_stream(streams["variable"], FIGURES / "variable_stream.png", run_id)
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
    save_measurements(matched, variable)
    plot_stream(matched, FIGURES / "matched_stream.png", run_id)
    plot_stream(variable, FIGURES / "variable_stream.png", run_id)

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
            "digits_per_seed_cell": STREAMS_PER_CELL * DIGITS_PER_STREAM,
            "dt_ms": float(config["dt"]),
        },
        "matched_stream": {key: value for key, value in matched.items() if not isinstance(value, np.ndarray)},
        "variable_stream": {key: value for key, value in variable.items() if not isinstance(value, np.ndarray)},
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


if __name__ == "__main__":
    main()
