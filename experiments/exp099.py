"""EXP099: replay a trained PING checkpoint across pixel-rate drive."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import theme  # noqa: E402
from helpers.checkpoints import public_provenance, resolve_checkpoint  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.datasets import load_mnist_split  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.rhythmicity import (  # noqa: E402
    iei_histogram,
    population_event_times,
    rhythmicity_scalars,
    spike_autocorrelogram,
)
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp099"
UPSTREAM_PUBLICATION = "ggs-production-composite-20260821-6d9c38eb"
UPSTREAM_CELL = "ping__canonical__seed42"
CHECKPOINT_ROLE = "final_epoch"
EXPECTED_CHECKPOINT_SHA256 = (
    "afe3bce49a89c2dbdac4f986bc3ca65bda91db385b2d58914c9c765075d78a0f"
)
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
SNN_TOOL = REPO / "tools" / "snn" / "tool.py"
DT_MS, T_MS, BURN_MS = 0.1, 400.0, 100.0
N_INPUT, N_E, N_I = 784, 1_024, 256
MNIST_SPLIT, IMAGE_INDEX, EXPECTED_LABEL = "official_test", 0, 7
POISSON_SEED = 9900
INPUT_RATES_HZ = tuple(float(value) for value in np.linspace(0.0, 100.0, 40))
VIDEO_FPS = 10
CORRELATION_CELLS, CORRELATION_BIN_MS = 20, 5.0
SCALE = {
    "upstream_publication": UPSTREAM_PUBLICATION,
    "upstream_cell": UPSTREAM_CELL,
    "checkpoint_role": CHECKPOINT_ROLE,
    "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    "dataset": "mnist",
    "dataset_split": MNIST_SPLIT,
    "image_index": IMAGE_INDEX,
    "expected_label": EXPECTED_LABEL,
    "poisson_seed": POISSON_SEED,
    "input_rates_hz": list(INPUT_RATES_HZ),
    "dt_ms": DT_MS,
    "t_ms": T_MS,
    "burn_ms": BURN_MS,
    "n_input": N_INPUT,
    "n_e": N_E,
    "n_i": N_I,
    "video_fps": VIDEO_FPS,
}


def require_checkpoint() -> tuple[Path, dict]:
    directory = TRAINING_ROOT / UPSTREAM_CELL
    checkpoint = resolve_checkpoint(directory, CHECKPOINT_ROLE)
    if checkpoint["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"{UPSTREAM_CELL} checkpoint does not match {UPSTREAM_PUBLICATION}: "
            f"{checkpoint['sha256']}"
        )
    return directory, checkpoint


def load_test_image() -> tuple[np.ndarray, int]:
    _, test_pixels, _, test_labels = load_mnist_split()
    pixels = test_pixels[IMAGE_INDEX].astype(np.float32)
    label = int(test_labels[IMAGE_INDEX])
    if label != EXPECTED_LABEL:
        raise RuntimeError(f"MNIST test image {IMAGE_INDEX} is {label}, expected 7")
    return pixels, label


def paired_input_bank(pixels: np.ndarray) -> dict[float, np.ndarray]:
    steps = round(T_MS / DT_MS)
    draws = np.random.default_rng(POISSON_SEED).random(
        (steps, N_INPUT), dtype=np.float32
    )
    return {
        rate: (draws < pixels[None, :] * rate * DT_MS / 1_000.0)[:, None, :].astype(
            np.uint8
        )
        for rate in INPUT_RATES_HZ
    }


def _dense(raster: np.lib.npyio.NpzFile, prefix: str, width: int) -> np.ndarray:
    values = np.zeros((int(raster["T"]), width), dtype=np.uint8)
    keep = raster[f"{prefix}_trial"] == 0
    values[raster[f"{prefix}_t"][keep], raster[f"{prefix}_cell"][keep]] = 1
    return values


def replay(
    directory: Path, checkpoint: dict, input_spikes: np.ndarray, work: Path
) -> dict[str, np.ndarray]:
    work.mkdir(parents=True, exist_ok=True)
    input_path, output = work / "input.npz", work / "replay"
    raster_path = output / "rasters.npz"
    if not raster_path.is_file():
        np.savez_compressed(input_path, input_spikes=input_spikes)
        subprocess.run(
            [
                "uv",
                "run",
                "python",
                str(SNN_TOOL),
                "sim",
                "--load-config",
                str(directory / "config.json"),
                "--load-weights",
                str(checkpoint["path"]),
                "--device",
                "auto",
                "--n-in",
                str(N_INPUT),
                "--input-file",
                str(input_path),
                "--outputs",
                "rasters",
                "--out-dir",
                str(output),
            ],
            cwd=REPO,
            check=True,
        )
    with np.load(raster_path) as raster:
        return {
            "e": _dense(raster, "e", int(raster["n_e"])),
            "i": _dense(raster, "i", int(raster["n_i"])),
        }


def _isi_cv(spikes: np.ndarray) -> float | None:
    values = []
    for cell in range(spikes.shape[1]):
        times = np.flatnonzero(spikes[:, cell])
        if len(times) >= 4:
            intervals = np.diff(times)
            values.append(float(intervals.std(ddof=1) / intervals.mean()))
    return None if not values else float(np.median(values))


def _pairwise_correlation(spikes: np.ndarray) -> float | None:
    bin_steps = round(CORRELATION_BIN_MS / DT_MS)
    trimmed = spikes[: len(spikes) // bin_steps * bin_steps, :CORRELATION_CELLS]
    counts = trimmed.reshape(-1, bin_steps, CORRELATION_CELLS).sum(axis=1)
    counts = counts[:, counts.std(axis=0) > 0]
    if counts.shape[1] < 2:
        return None
    matrix = np.corrcoef(counts, rowvar=False)
    values = matrix[np.triu_indices_from(matrix, 1)]
    values = values[np.isfinite(values)]
    return None if not values.size else float(values.mean())


def _spectrum(spikes: np.ndarray) -> tuple[float | None, float | None]:
    counts = spikes.sum(axis=1).astype(float)
    if counts.std() == 0:
        return None, None
    power = np.abs(np.fft.rfft((counts - counts.mean()) * np.hanning(len(counts)))) ** 2
    frequencies = np.fft.rfftfreq(len(counts), DT_MS / 1_000.0)
    keep = (frequencies >= 5.0) & (frequencies <= 100.0)
    selected = power[keep]
    if not selected.size or not np.any(selected > 0):
        return None, None
    index, median = int(np.argmax(selected)), float(np.median(selected))
    return float(frequencies[keep][index]), (
        None if median == 0 else float(selected[index] / median)
    )


def _e_i_lag(e: np.ndarray, i: np.ndarray) -> float | None:
    bin_steps = round(1.0 / DT_MS)
    e_counts = (
        e[: len(e) // bin_steps * bin_steps].reshape(-1, bin_steps, N_E).sum((1, 2))
    )
    i_counts = (
        i[: len(i) // bin_steps * bin_steps].reshape(-1, bin_steps, N_I).sum((1, 2))
    )
    if e_counts.std() == 0 or i_counts.std() == 0:
        return None
    correlation = np.correlate(
        e_counts - e_counts.mean(), i_counts - i_counts.mean(), mode="full"
    )
    lags = np.arange(-len(e_counts) + 1, len(e_counts))
    keep = np.abs(lags) <= 20
    # np.correlate(E, I) returns a negative raw lag when E precedes I.
    return float(-lags[keep][np.argmax(correlation[keep])])


def describe(rate: float, arrays: dict[str, np.ndarray]) -> dict:
    burn = round(BURN_MS / DT_MS)
    e, i = arrays["e"][burn:], arrays["i"][burn:]
    duration_s = len(e) * DT_MS / 1_000.0
    frequency, prominence = _spectrum(e)
    ac_lags, ac = spike_autocorrelogram(e, DT_MS, 100.0, 1.0)
    iei_lags, iei = iei_histogram(population_event_times(e, DT_MS), 100.0, 1.0)
    rhythmicity = rhythmicity_scalars(ac_lags, ac, iei_lags, iei, 1.0)["contrast"]
    return {
        "input_rate_hz": rate,
        "e_rate_hz": float(e.sum() / N_E / duration_s),
        "i_rate_hz": float(i.sum() / N_I / duration_s),
        "median_e_isi_cv": _isi_cv(e),
        "mean_e_pairwise_correlation": _pairwise_correlation(e),
        "rhythmicity_contrast": None if rhythmicity is None else float(rhythmicity),
        "dominant_frequency_hz": frequency,
        "peak_to_median_power": prominence,
        "e_leads_i_lag_ms": _e_i_lag(e, i),
    }


def plot_summary(rows: list[dict], out: Path) -> None:
    theme.apply()
    rates = [row["input_rate_hz"] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.0), sharex=True)
    axes[0].plot(
        rates,
        [row["e_rate_hz"] for row in rows],
        "o-",
        label="E",
        color=theme.INK_BLACK,
    )
    axes[0].plot(
        rates, [row["i_rate_hz"] for row in rows], "o-", label="I", color=theme.DEEP_RED
    )
    axes[0].set_ylabel("rate (Hz)")
    axes[0].legend(frameon=False)
    axes[1].plot(
        rates,
        [row["mean_e_pairwise_correlation"] for row in rows],
        label="correlation",
        color=theme.DEEP_RED,
    )
    axes[1].plot(
        rates,
        [row["median_e_isi_cv"] for row in rows],
        label="ISI CV",
        color=theme.INK_BLACK,
    )
    axes[1].set_ylabel("AI diagnostics")
    axes[1].legend(frameon=False)
    axes[2].plot(
        rates,
        [row["dominant_frequency_hz"] for row in rows],
        "o-",
        color=theme.DEEP_RED,
    )
    axes[2].axhspan(30, 80, color=theme.GREY_LIGHT, alpha=0.4)
    axes[2].set(xlabel="maximum pixel rate D (Hz)", ylabel="dominant frequency (Hz)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def render_video(
    pixels: np.ndarray,
    rows: list[dict],
    recordings: list[dict[str, np.ndarray]],
    out: Path,
    poster: Path,
) -> None:
    theme.apply()
    fig = plt.figure(figsize=(10.5, 5.8))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 2.4))
    image_axis, text_axis = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])
    raster_axis = fig.add_subplot(grid[:, 1])
    image_axis.imshow(pixels.reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
    image_axis.set_title("MNIST test image 0 · label 7")
    image_axis.axis("off")

    def update(frame: int):
        raster_axis.clear()
        arrays, row = recordings[frame], rows[frame]
        e_t, e_cell = np.nonzero(arrays["e"][:, :100])
        i_t, i_cell = np.nonzero(arrays["i"][:, :40])
        raster_axis.scatter(e_t * DT_MS, e_cell, s=1.2, color=theme.INK_BLACK)
        raster_axis.scatter(i_t * DT_MS, i_cell + 105, s=1.2, color=theme.DEEP_RED)
        raster_axis.axvline(BURN_MS, color=theme.GREY_MID, ls="--", lw=0.8)
        raster_axis.set(
            xlim=(0, T_MS),
            ylim=(-2, 147),
            xlabel="biological time (ms)",
            ylabel="sampled cells",
            title=f"trained PING replay · D = {row['input_rate_hz']:.1f} Hz",
        )
        raster_axis.set_yticks((49.5, 124.5), ("E", "I"))
        raster_axis.spines[["top", "right"]].set_visible(False)
        text_axis.clear()
        text_axis.axis("off")
        frequency = row["dominant_frequency_hz"]
        text_axis.text(
            0,
            1,
            "upstream: gamma-gated-sparsity\ncheckpoint: canonical seed42 · final epoch\n\n"
            f"E rate  {row['e_rate_hz']:.1f} Hz\nI rate  {row['i_rate_hz']:.1f} Hz\n"
            f"ISI CV  {row['median_e_isi_cv'] if row['median_e_isi_cv'] is not None else '—'}\n"
            f"pair corr  {row['mean_e_pairwise_correlation'] if row['mean_e_pairwise_correlation'] is not None else '—'}\n"
            f"rhythm  {'unresolved' if frequency is None else f'{frequency:.1f} Hz'}",
            va="top",
            fontsize=9,
        )
        return []

    update(0)
    fig.tight_layout()
    fig.savefig(poster, dpi=160)
    movie = animation.FuncAnimation(
        fig, update, frames=len(rows), interval=1_000 / VIDEO_FPS
    )
    writer = animation.FFMpegWriter(
        fps=VIDEO_FPS,
        codec="libx264",
        bitrate=2_400,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    movie.save(out, writer=writer, dpi=120)
    plt.close(fig)


def main() -> None:
    meta = parse_meta(sys.argv)
    if meta.runpod:
        raise SystemExit(
            "exp099 uses bounded local checkpoint replay; RunPod is unsupported"
        )
    if meta.plot_only:
        raise SystemExit("exp099 plot-only replay is not implemented")
    directory, checkpoint = require_checkpoint()
    pixels, label = load_test_image()
    inputs = paired_input_bank(pixels)
    started, run_id = time.monotonic(), next_run_id(SLUG)
    with published_run(SLUG, run_id, scale=SCALE) as (scratch, staging):
        rows, recordings = [], []
        for index, rate in enumerate(INPUT_RATES_HZ):
            print(f"[replay] D={rate:.2f} Hz")
            arrays = replay(
                directory, checkpoint, inputs[rate], scratch / f"condition-{index:02d}"
            )
            rows.append(describe(rate, arrays))
            recordings.append(arrays)
        np.savez_compressed(
            staging / "paired_inputs.npz",
            **{
                f"D_{index:02d}": inputs[rate]
                for index, rate in enumerate(INPUT_RATES_HZ)
            },
        )
        plot_summary(rows, staging / "trained_pixel_transition.svg")
        render_video(
            pixels,
            rows,
            recordings,
            staging / "trained_pixel_transition.mp4",
            staging / "trained_pixel_transition_poster.png",
        )
        (staging / "protocol.json").write_text(json.dumps(SCALE, indent=2) + "\n")
        write_numbers(
            staging,
            run_id=run_id,
            duration_s=time.monotonic() - started,
            payload={
                "question": "Does maximum pixel rate move a trained PING checkpoint through AI-like activity into PING?",
                "config": SCALE,
                "upstream_checkpoint": public_provenance(checkpoint),
                "image": {"split": MNIST_SPLIT, "index": IMAGE_INDEX, "label": label},
                "conditions": rows,
            },
        )
    print(f"exp099 complete: {run_id}")


if __name__ == "__main__":
    main()
