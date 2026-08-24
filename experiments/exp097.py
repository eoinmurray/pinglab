"""EXP097: scout the recurrent-conductance state portrait of a PING cycle."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation, patches
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "assets" / "exp097"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "snn"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from execution import ExecutionSpec, simulate  # noqa: E402
from experiments import exp083  # noqa: E402
from tools import snnlang as snn  # noqa: E402, TID251

from helpers import theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp097"
DT_MS = 0.1
T_MS = 500.0
DYNAMIC_T_MS = 1_200.0
BURN_MS = 100.0
TAU_GABA_MS = 2.0
NETWORK_SEED = 83
TRIAL_SEEDS = exp083.TRIAL_SEEDS
INPUT_RATE_HZ = 50.0
VISUAL_SEED = 8302
N_E = 800
N_I = 200
CONSTANT_INPUT_RASTER_CHANNELS = tuple(np.linspace(0, exp083.N_INPUT - 1, 24, dtype=int))
E_RASTER_CELLS = tuple(np.linspace(0, N_E - 1, 40, dtype=int))
I_RASTER_CELLS = tuple(np.linspace(0, N_I - 1, 20, dtype=int))
SCALE = {
    "dt_ms": DT_MS,
    "t_ms": T_MS,
    "burn_ms": BURN_MS,
    "n_e": N_E,
    "n_i": N_I,
    "input_rate_hz": INPUT_RATE_HZ,
    "tau_gaba_ms": TAU_GABA_MS,
    "network_seed": NETWORK_SEED,
    "trial_seeds": list(TRIAL_SEEDS),
}


def author_network() -> snn.Bundle:
    net = snn.Network("exp097_large_ping", dt=DT_MS * snn.ms)
    drive = net.input("drive", shape=("time", "batch", exp083.N_INPUT), signal_type="spikes", unit="spike")
    cell = snn.components.ping(net, name="ping", n_e=N_E, n_i=N_I, source=drive, tau_gaba=TAU_GABA_MS * snn.ms)
    net.expose(cell.E.spikes, cell.I.spikes, name="population")
    return snn.compile(net, target="tools/snn")


def make_inputs() -> np.ndarray:
    steps = round(T_MS / DT_MS)
    probability = INPUT_RATE_HZ * DT_MS / 1_000.0
    trials = []
    for seed in TRIAL_SEEDS:
        rng = np.random.default_rng(seed)
        trials.append(rng.random((steps, exp083.N_INPUT), dtype=np.float32) < probability)
    return np.stack(trials, axis=1).astype(np.uint8)


def input_rate_schedule(kind: str) -> np.ndarray:
    """Return the commanded per-channel rate for a transient-drive video."""
    t = np.arange(round(DYNAMIC_T_MS / DT_MS)) * DT_MS
    rate = np.zeros_like(t)
    active = (t >= 200.0) & (t < 1_000.0)
    if kind == "ramp":
        rising = active & (t < 600.0)
        falling = active & (t >= 600.0)
        rate[rising] = INPUT_RATE_HZ * (t[rising] - 200.0) / 400.0
        rate[falling] = INPUT_RATE_HZ * (1_000.0 - t[falling]) / 400.0
    else:
        raise ValueError(f"unknown input-rate schedule: {kind}")
    return rate


def make_scheduled_inputs(kind: str) -> tuple[np.ndarray, np.ndarray]:
    rates = input_rate_schedule(kind)
    rng = np.random.default_rng(VISUAL_SEED)
    probability = rates[:, None] * DT_MS / 1_000.0
    spikes = rng.random((len(rates), exp083.N_INPUT), dtype=np.float32) < probability
    return spikes[:, None, :].astype(np.uint8), rates


def native_raster_events(spikes: np.ndarray, left: int, right: int, trial: int, channels: tuple[int, ...]) -> tuple[list[float], list[int]]:
    """Retain native timestamps for a fixed sample of channels or cells."""
    selected = spikes[left:right, trial][:, np.asarray(channels)]
    steps, rows = np.nonzero(selected)
    return (steps.astype(float) * DT_MS).round(3).tolist(), rows.astype(int).tolist()


def population_rate_hz(spikes: np.ndarray, edges: np.ndarray, trial: int) -> list[float]:
    """Calculate per-neuron rates in display bins, independent of population size."""
    n_cells = spikes.shape[2]
    rates = []
    for left, right in zip(edges[:-1], edges[1:]):
        duration_s = max((right - left) * DT_MS / 1_000.0, DT_MS / 1_000.0)
        rates.append(float(spikes[left:right, trial].sum()) / n_cells / duration_s)
    return np.asarray(rates).round(4).tolist()


def build_display_state(
    recordings: dict[str, np.ndarray],
    inputs: np.ndarray,
    trial: int,
    left: int,
    right: int,
    title: str,
    seed: int | None = None,
    command_rate_hz: np.ndarray | None = None,
    input_label: str = "input 24/128",
) -> dict:
    """Sample one measured interval into the shared 300-frame visual grammar."""
    e = recordings["e_spikes"]
    i = recordings["i_spikes"]
    ge_cells = recordings["g_e_to_i"]
    ge = ge_cells.mean(axis=2)
    gi = recordings["g_i_to_e"].mean(axis=2)
    ve = recordings["v_e"].mean(axis=2)
    vi = recordings["v_i"].mean(axis=2)
    edges = np.linspace(left, right, 301, dtype=int)
    sample = edges[:-1]
    input_channels = CONSTANT_INPUT_RASTER_CHANNELS
    input_raster_times, input_raster_rows = native_raster_events(inputs, left, right, trial, input_channels)
    e_raster_times, e_raster_rows = native_raster_events(e, left, right, trial, E_RASTER_CELLS)
    i_raster_times, i_raster_rows = native_raster_events(i, left, right, trial, I_RASTER_CELLS)
    state = {
        "status": "measured",
        "title": title,
        "trial": trial,
        "seed": TRIAL_SEEDS[trial] if seed is None else seed,
        "input_label": input_label,
        "dt_ms": DT_MS,
        "interval_ms": round((right - left) * DT_MS, 3),
        "time_ms": ((sample - left) * DT_MS).round(3).tolist(),
        "ge": ge[sample, trial].round(8).tolist(),
        "gi": gi[sample, trial].round(8).tolist(),
        "ve": ve[sample, trial].round(6).tolist(),
        "vi": vi[sample, trial].round(6).tolist(),
        "e_spikes": [int(e[edges[k]:edges[k + 1], trial].sum()) for k in range(len(sample))],
        "i_spikes": [int(i[edges[k]:edges[k + 1], trial].sum()) for k in range(len(sample))],
        "input_spikes": [int(inputs[edges[k]:edges[k + 1], trial].sum()) for k in range(len(sample))],
        "input_raster_times_ms": input_raster_times,
        "input_raster_rows": input_raster_rows,
        "input_raster_count": len(input_channels),
        "e_raster_times_ms": e_raster_times,
        "e_raster_rows": e_raster_rows,
        "i_raster_times_ms": i_raster_times,
        "i_raster_rows": i_raster_rows,
        "input_rate_hz": population_rate_hz(inputs, edges, trial),
        "e_rate_hz": population_rate_hz(e, edges, trial),
        "i_rate_hz": population_rate_hz(i, edges, trial),
        "ge_cells": ge_cells[sample, trial, :13].round(8).T.tolist(),
    }
    if command_rate_hz is not None:
        state["command_rate_hz"] = command_rate_hz[sample].round(4).tolist()
    return state


def detect_cycles(e_spikes: np.ndarray) -> list[np.ndarray]:
    burn = round(BURN_MS / DT_MS)
    cycles = []
    for trial in range(e_spikes.shape[1]):
        fraction = e_spikes[:, trial].mean(axis=1)
        smooth = gaussian_filter1d(fraction.astype(float), sigma=5.0)
        peaks, _ = find_peaks(smooth, distance=round(12.0 / DT_MS), prominence=2.0 / 80.0)
        peaks = peaks[peaks >= burn]
        cycles.append(peaks)
    return cycles


def phase_series(peaks: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    phase = np.full(steps, np.nan)
    next_ms = np.full(steps, np.nan)
    for left, right in zip(peaks[:-1], peaks[1:]):
        phase[left:right] = np.arange(right - left) / (right - left)
        next_ms[left:right] = (right - np.arange(left, right)) * DT_MS
    return phase, next_ms


def circular_error(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    delta = np.abs(predicted - actual)
    return np.minimum(delta, 1.0 - delta)


def held_out_errors(states: np.ndarray, cycles: list[np.ndarray]) -> dict:
    phases = []
    next_times = []
    keep = []
    stride = round(1.0 / DT_MS)
    for trial, peaks in enumerate(cycles):
        phase, next_ms = phase_series(peaks, states.shape[0])
        valid = np.flatnonzero(np.isfinite(phase))[::stride]
        phases.append(phase[valid])
        next_times.append(next_ms[valid])
        keep.append(valid)
    errors = {"two_phase": [], "four_phase": [], "two_timing": [], "four_timing": []}
    for held in range(states.shape[1]):
        train_trials = [i for i in range(states.shape[1]) if i != held]
        for dims, prefix in ((2, "two"), (4, "four")):
            train_x = np.concatenate([states[keep[i], i, :dims] for i in train_trials])
            train_phase = np.concatenate([phases[i] for i in train_trials])
            train_next = np.concatenate([next_times[i] for i in train_trials])
            mean = train_x.mean(axis=0)
            scale = train_x.std(axis=0)
            scale[scale == 0] = 1.0
            tree = cKDTree((train_x - mean) / scale)
            _, index = tree.query((states[keep[held], held, :dims] - mean) / scale)
            errors[f"{prefix}_phase"].extend(circular_error(train_phase[index], phases[held]))
            errors[f"{prefix}_timing"].extend(np.abs(train_next[index] - next_times[held]))
    return {key: float(np.median(value)) for key, value in errors.items()}


def analyse(recordings: dict[str, np.ndarray], inputs: np.ndarray) -> tuple[dict, dict]:
    e = recordings["e_spikes"]
    i = recordings["i_spikes"]
    ge_cells = recordings["g_e_to_i"]
    gi_cells = recordings["g_i_to_e"]
    ge = ge_cells.mean(axis=2)
    gi = gi_cells.mean(axis=2)
    ve = recordings["v_e"].mean(axis=2)
    vi = recordings["v_i"].mean(axis=2)
    cycles = detect_cycles(e)

    areas = []
    orientations = []
    periods = []
    lags = []
    per_trial = []
    for trial, peaks in enumerate(cycles):
        trial_areas = []
        for left, right in zip(peaks[:-1], peaks[1:]):
            x, y = ge[left:right, trial], gi[left:right, trial]
            area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
            areas.append(float(area))
            trial_areas.append(float(area))
            orientations.append(int(np.sign(area)))
            periods.append((right - left) * DT_MS)
            e_peak = left + int(np.argmax(e[left:right, trial].sum(axis=1)))
            i_peak = left + int(np.argmax(i[left:right, trial].sum(axis=1)))
            lags.append((i_peak - e_peak) * DT_MS)
        per_trial.append({"trial": trial, "seed": TRIAL_SEEDS[trial], "cycles": len(trial_areas), "median_period_ms": float(np.median(np.diff(peaks)) * DT_MS), "orientation": int(np.sign(np.median(trial_areas)))})

    modal = 1 if orientations.count(1) >= orientations.count(-1) else -1
    orientation_fraction = float(np.mean(np.asarray(orientations) == modal))
    states = np.stack([ge, gi, ve, vi], axis=2)
    prediction = held_out_errors(states, cycles)
    burn = round(BURN_MS / DT_MS)
    ge_post = ge_cells[burn:]
    gi_post = gi_cells[burn:]
    ge_residual_fraction = float(np.std(ge_post - ge_post.mean(axis=2, keepdims=True)) / np.std(ge_post))
    gi_residual_fraction = float(np.std(gi_post - gi_post.mean(axis=2, keepdims=True)) / np.std(gi_post))

    trial_frequency = [1_000.0 / row["median_period_ms"] for row in per_trial]
    display_trial = int(np.argmin(np.abs(trial_frequency - np.median(trial_frequency))))
    display_peaks = cycles[display_trial]
    starts = display_peaks[:6]
    left, right = int(starts[0]), int(starts[-1])
    state = build_display_state(recordings, inputs, display_trial, left, right, "Simulated recurrent E–I cycle at 50 Hz input")
    result = {
        "cycles_total": len(areas),
        "cycles_per_trial": [len(p) - 1 for p in cycles],
        "modal_orientation": "counter-clockwise" if modal > 0 else "clockwise",
        "orientation_consistency": orientation_fraction,
        "median_signed_area_uS2": float(np.median(areas)),
        "median_period_ms": float(np.median(periods)),
        "median_frequency_hz": float(1_000.0 / np.median(periods)),
        "median_e_to_i_lag_ms": float(np.median(lags)),
        "ge_target_residual_sd_fraction": ge_residual_fraction,
        "gi_target_residual_sd_fraction": gi_residual_fraction,
        "ge_mean_across_target_sd_uS": float(np.std(ge_post, axis=2).mean()),
        "gi_mean_across_target_sd_uS": float(np.std(gi_post, axis=2).mean()),
        "prediction": prediction,
        "per_trial": per_trial,
        "display_trial": display_trial,
    }
    return result, state


def plot_result(state: dict, result: dict, out: Path) -> None:
    theme.apply()
    t = np.asarray(state["time_ms"])
    ge = np.asarray(state["ge"])
    gi = np.asarray(state["gi"])
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0))
    axes[0].plot(ge, gi, color=theme.DEEP_RED, lw=1.4)
    axes[0].scatter(ge[::20], gi[::20], s=9, color=theme.INK_BLACK)
    axes[0].set(xlabel="$g_E$ (µS)", ylabel="$g_I$ (µS)", title="simulated joint state")
    axes[1].plot(t, ge, color=theme.INK_BLACK, label="$g_E$")
    axes[1].plot(t, gi, color=theme.DEEP_RED, label="$g_I$")
    axes[1].set(xlabel="time (ms)", ylabel="mean conductance (µS)", title="four cycles")
    axes[1].legend(frameon=False)
    pred = result["prediction"]
    axes[2].bar([0, 1], [pred["two_phase"], pred["four_phase"]], color=[theme.INK_BLACK, theme.DEEP_RED])
    axes[2].set_xticks([0, 1], ["$g_E,g_I$", "+ voltage"])
    axes[2].set(ylabel="median circular error (cycles)", title="held-out phase")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def render_measured_animation(state: dict, out: Path, poster: Path) -> None:
    """Render the simulated multi-cycle state in the design schematic's visual grammar."""
    theme.apply()
    t = np.asarray(state["time_ms"])
    ge = np.asarray(state["ge"])
    gi = np.asarray(state["gi"])
    ve = np.asarray(state["ve"])
    vi = np.asarray(state["vi"])
    e_spikes = np.asarray(state["e_spikes"])
    i_spikes = np.asarray(state["i_spikes"])
    input_raster_times = np.asarray(state["input_raster_times_ms"])
    input_raster_rows = np.asarray(state["input_raster_rows"])
    e_raster_times = np.asarray(state["e_raster_times_ms"])
    e_raster_rows = np.asarray(state["e_raster_rows"])
    i_raster_times = np.asarray(state["i_raster_times_ms"])
    i_raster_rows = np.asarray(state["i_raster_rows"])
    input_rate = np.asarray(state["input_rate_hz"])
    e_rate = np.asarray(state["e_rate_hz"])
    i_rate = np.asarray(state["i_rate_hz"])
    command_rate = np.asarray(state["command_rate_hz"]) if "command_rate_hz" in state else None
    transition_metrics = state.get("transition_metrics")
    if transition_metrics is not None:
        metric_arrays = {key: np.asarray([np.nan if value is None else value for value in values], dtype=float) for key, values in transition_metrics.items()}
    else:
        metric_arrays = None
    ge_low, ge_high = float(ge.min()), float(ge.max())
    gi_low, gi_high = float(gi.min()), float(gi.max())
    ve_low, ve_high = float(ve.min()), float(ve.max())
    vi_low, vi_high = float(vi.min()), float(vi.max())

    fig = plt.figure(figsize=(14.0, 6.1), facecolor="#f4f7f9")
    grid = fig.add_gridspec(2, 3, left=0.055, right=0.98, bottom=0.20, top=0.90, hspace=0.48, wspace=0.30)
    engine = fig.add_subplot(grid[0, 0])
    traces = fig.add_subplot(grid[0, 1])
    phase = fig.add_subplot(grid[0, 2])
    voltage_engine = fig.add_subplot(grid[1, 0])
    voltages = fig.add_subplot(grid[1, 1])
    spikes = fig.add_subplot(grid[1, 2])
    fig.suptitle(state["title"], fontsize=15, fontweight="bold", color=theme.INK_BLACK)
    time_text = fig.text(0.5, 0.055, "", ha="center", color=theme.INK_BLACK, fontsize=8)

    engine.set(xlim=(0, 10), ylim=(0, 10), title="1 · Local conductance engine")
    engine.axis("off")
    cylinder_specs = ((1.2, theme.DEEP_RED, "$g_E$ · E→I"), (6.2, theme.INK_BLACK, "$g_I$ · I→E"))
    fills = []
    plungers = []
    for x, colour, label in cylinder_specs:
        engine.add_patch(patches.FancyBboxPatch((x, 1.7), 2.5, 6.4, boxstyle="round,pad=0.08", facecolor="white", edgecolor=theme.GREY_MID, linewidth=1.5))
        fill = patches.Rectangle((x + 0.65, 2.3), 1.2, 0.4, facecolor=colour, alpha=0.88)
        plunger = patches.Rectangle((x + 0.35, 2.65), 1.8, 0.22, facecolor=colour)
        engine.add_patch(fill)
        engine.add_patch(plunger)
        engine.text(x + 1.25, 1.15, label, ha="center", fontsize=10, fontweight="bold")
        fills.append(fill)
        plungers.append(plunger)
    engine.annotate("E volley triggers $g_E$", (6.0, 8.7), (4.0, 8.7), arrowprops={"arrowstyle": "->", "color": theme.DEEP_RED}, ha="center", fontsize=8)
    engine.annotate("$g_I$ suppresses E", (3.9, 4.4), (6.0, 4.4), arrowprops={"arrowstyle": "->", "color": theme.INK_BLACK}, ha="center", fontsize=8)
    e_lamp = patches.Circle((4.9, 7.3), 0.24, facecolor=theme.DEEP_RED, alpha=0.12)
    i_lamp = patches.Circle((4.9, 2.7), 0.24, facecolor=theme.INK_BLACK, alpha=0.12)
    engine.add_patch(e_lamp)
    engine.add_patch(i_lamp)

    voltage_engine.set(xlim=(0, 10), ylim=(0, 10), title="4 · Local membrane-voltage engine")
    voltage_engine.axis("off")
    voltage_fills = []
    voltage_plungers = []
    for x, colour, label in ((1.2, theme.DEEP_RED, "$V_E$ · E cells"), (6.2, theme.INK_BLACK, "$V_I$ · I cells")):
        voltage_engine.add_patch(patches.FancyBboxPatch((x, 1.7), 2.5, 6.4, boxstyle="round,pad=0.08", facecolor="white", edgecolor=theme.GREY_MID, linewidth=1.5))
        fill = patches.Rectangle((x + 0.65, 2.3), 1.2, 0.4, facecolor=colour, alpha=0.88)
        plunger = patches.Rectangle((x + 0.35, 2.65), 1.8, 0.22, facecolor=colour)
        voltage_engine.add_patch(fill)
        voltage_engine.add_patch(plunger)
        voltage_engine.text(x + 1.25, 1.15, label, ha="center", fontsize=10, fontweight="bold")
        voltage_fills.append(fill)
        voltage_plungers.append(plunger)
    voltage_engine.text(5.0, 8.75, "up = depolarized", ha="center", fontsize=9)

    traces.plot(t, ge, color=theme.DEEP_RED, lw=1.5, label="$g_E$")
    traces.plot(t, gi, color=theme.INK_BLACK, lw=1.5, label="$g_I$")
    e_volleys, _ = find_peaks(e_rate, distance=max(1, len(t) // 12), prominence=max(float(e_rate.max()) * 0.2, 1e-9))
    i_volleys, _ = find_peaks(i_rate, distance=max(1, len(t) // 12), prominence=max(float(i_rate.max()) * 0.2, 1e-9))
    e_times = t[e_volleys]
    i_times = t[i_volleys]
    traces.scatter(e_times, np.full_like(e_times, ge_high * 1.08), marker="|", s=55, color=theme.DEEP_RED, label="E volley")
    traces.scatter(i_times, np.full_like(i_times, ge_high * 1.15), marker="|", s=55, color=theme.INK_BLACK, label="I volley")
    cursor = traces.axvline(t[0], color=theme.GREY_MID, lw=1.2)
    trace_ge_dot, = traces.plot([t[0]], [ge[0]], "o", color=theme.DEEP_RED, ms=6)
    trace_gi_dot, = traces.plot([t[0]], [gi[0]], "o", color=theme.INK_BLACK, ms=6)
    traces.set(xlabel="biological time (ms)", ylabel="mean conductance (µS)", title="2 · Conductance traces", xlim=(t[0], t[-1]))
    traces.legend(frameon=False, ncol=2, fontsize=8)

    phase.plot(ge, gi, color=theme.GREY_MID, lw=1.0, alpha=0.45)
    trail, = phase.plot([], [], color=theme.DEEP_RED, lw=2.3)
    phase_dot, = phase.plot([ge[0]], [gi[0]], "o", color=theme.INK_BLACK, ms=7)
    phase.set(xlabel="$g_E$ (µS)", ylabel="$g_I$ (µS)", title="3 · Joint conductance state", xlim=(ge_low, ge_high), ylim=(gi_low, gi_high))
    metrics_text = phase.text(0.98, 0.97, "", transform=phase.transAxes, ha="right", va="top", fontsize=7, linespacing=1.35, bbox={"facecolor": "white", "edgecolor": theme.GREY_MID, "alpha": 0.88, "pad": 3})
    metrics_text.set_visible(metric_arrays is not None)

    voltages.plot(t, ve, color=theme.DEEP_RED, lw=1.15, label="$V_E$")
    voltages.plot(t, vi, color=theme.INK_BLACK, lw=1.15, label="$V_I$")
    voltage_cursor = voltages.axvline(t[0], color=theme.GREY_MID, lw=1.2)
    ve_dot, = voltages.plot([t[0]], [ve[0]], "o", color=theme.DEEP_RED, ms=5)
    vi_dot, = voltages.plot([t[0]], [vi[0]], "o", color=theme.INK_BLACK, ms=5)
    voltage_pad = 0.06 * max(float(max(ve.max(), vi.max()) - min(ve.min(), vi.min())), 1.0)
    voltages.set(xlabel="biological time (ms)", ylabel="population mean voltage (mV)", title="5 · Membrane-voltage traces", xlim=(t[0], t[-1]), ylim=(min(ve.min(), vi.min()) - voltage_pad, max(ve.max(), vi.max()) + voltage_pad))
    voltages.legend(frameon=False, ncol=2, fontsize=8)

    input_y = 2.08 + 0.52 * input_raster_rows / max(int(state["input_raster_count"]) - 1, 1)
    e_y = 1.08 + 0.52 * e_raster_rows / max(len(E_RASTER_CELLS) - 1, 1)
    i_y = 0.08 + 0.52 * i_raster_rows / max(len(I_RASTER_CELLS) - 1, 1)
    spikes.scatter(input_raster_times, input_y, marker="|", s=10, linewidths=0.55, color=theme.GREY_MID, alpha=0.72)
    spikes.scatter(e_raster_times, e_y, marker="|", s=10, linewidths=0.55, color=theme.DEEP_RED, alpha=0.72)
    spikes.scatter(i_raster_times, i_y, marker="|", s=10, linewidths=0.55, color=theme.INK_BLACK, alpha=0.72)
    for rate, base, colour in ((input_rate, 2.68, theme.GREY_MID), (e_rate, 1.68, theme.DEEP_RED), (i_rate, 0.68, theme.INK_BLACK)):
        envelope = base + 0.18 * rate / max(float(rate.max()), 1.0)
        spikes.plot(t, envelope, color=colour, lw=0.9, alpha=0.85)
    if command_rate is not None:
        command_line = 2.68 + 0.18 * command_rate / max(float(command_rate.max()), 1.0)
        spikes.plot(t, command_line, color=theme.INK_BLACK, lw=1.0, ls="--", alpha=0.8)
        spikes.text(0.99, 0.98, "dashed = commanded input rate", transform=spikes.transAxes, ha="right", va="top", fontsize=7)
    spike_cursor = spikes.axvline(t[0], color=theme.DEEP_RED, lw=1.4, alpha=0.75)
    spikes.set_yticks([0.34, 1.34, 2.34], ["I 20/200", "E 40/800", state["input_label"]])
    spikes.set(xlabel="biological time (ms)", title="6 · Sampled spikes and population rates", xlim=(t[0], t[-1]), ylim=(-0.05, 2.95))
    for axis in (traces, phase, voltages, spikes):
        axis.spines[["top", "right"]].set_visible(False)

    def update(frame: int):
        ge_fraction = (ge[frame] - ge_low) / max(ge_high - ge_low, 1e-12)
        gi_fraction = (gi[frame] - gi_low) / max(gi_high - gi_low, 1e-12)
        for fill, plunger, fraction in zip(fills, plungers, (ge_fraction, gi_fraction)):
            height = 0.4 + 4.2 * fraction
            fill.set_y(2.3)
            fill.set_height(height)
            plunger.set_y(2.7 + height)
        ve_fraction = (ve[frame] - ve_low) / max(ve_high - ve_low, 1e-12)
        vi_fraction = (vi[frame] - vi_low) / max(vi_high - vi_low, 1e-12)
        for fill, plunger, fraction in zip(voltage_fills, voltage_plungers, (ve_fraction, vi_fraction)):
            height = 0.4 + 4.2 * fraction
            fill.set_y(2.3)
            fill.set_height(height)
            plunger.set_y(2.7 + height)
        e_lamp.set_alpha(0.95 if e_spikes[frame] > 0 else 0.12)
        i_lamp.set_alpha(0.95 if i_spikes[frame] > 0 else 0.12)
        cursor.set_xdata([t[frame], t[frame]])
        trace_ge_dot.set_data([t[frame]], [ge[frame]])
        trace_gi_dot.set_data([t[frame]], [gi[frame]])
        start = max(0, frame - 70)
        trail.set_data(ge[start:frame + 1], gi[start:frame + 1])
        phase_dot.set_data([ge[frame]], [gi[frame]])
        voltage_cursor.set_xdata([t[frame], t[frame]])
        ve_dot.set_data([t[frame]], [ve[frame]])
        vi_dot.set_data([t[frame]], [vi[frame]])
        spike_cursor.set_xdata([t[frame], t[frame]])
        command = "" if command_rate is None else f"  ·  command {command_rate[frame]:.1f} Hz/channel"
        time_text.set_text(f"simulation seed {state['seed']}  ·  biological time {t[frame]:.1f} ms{command}  ·  playback 30 fps")
        if metric_arrays is not None:
            cv = metric_arrays["isi_cv"][frame]
            corr = metric_arrays["pairwise_correlation"][frame]
            frequency = metric_arrays["gamma_frequency_hz"][frame]
            power = metric_arrays["gamma_power_fraction"][frame]
            metrics_text.set_text(
                "trailing 300 ms\n"
                + f"ISI CV  {'—' if np.isnan(cv) else f'{cv:.2f}'}\n"
                + f"pair corr  {'—' if np.isnan(corr) else f'{corr:.3f}'}\n"
                + f"gamma f  {'unresolved' if np.isnan(frequency) else f'{frequency:.1f} Hz'}\n"
                + f"gamma power  {'—' if np.isnan(power) else f'{power:.2f}'}"
            )
        return [*fills, *plungers, *voltage_fills, *voltage_plungers, e_lamp, i_lamp, cursor, trace_ge_dot, trace_gi_dot, trail, phase_dot, metrics_text, voltage_cursor, ve_dot, vi_dot, spike_cursor, time_text]

    update(0)
    fig.savefig(poster, dpi=150, facecolor=fig.get_facecolor())
    movie = animation.FuncAnimation(fig, update, frames=len(t), interval=1000 / 30, blit=False)
    writer = animation.FFMpegWriter(fps=30, codec="libx264", bitrate=2400, extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    movie.save(out, writer=writer, dpi=120)
    plt.close(fig)


def recording_arrays(result) -> dict[str, np.ndarray]:
    return {
        "e_spikes": result.recordings["population_0"].cpu().numpy(),
        "i_spikes": result.recordings["population_1"].cpu().numpy(),
        "v_e": result.recordings["ping_E.voltage"].cpu().numpy(),
        "v_i": result.recordings["ping_I.voltage"].cpu().numpy(),
        "g_e_to_i": result.recordings["ping_E_to_I.conductance"].cpu().numpy(),
        "g_i_to_e": result.recordings["ping_I_to_E.conductance"].cpu().numpy(),
    }


def transient_summary(recordings: dict[str, np.ndarray], command_rate: np.ndarray) -> dict:
    def mean_rate(spikes: np.ndarray, start_ms: float, end_ms: float) -> float:
        left, right = round(start_ms / DT_MS), round(end_ms / DT_MS)
        return float(spikes[left:right].sum() / spikes.shape[1] / spikes.shape[2] / ((end_ms - start_ms) / 1_000.0))

    peaks = detect_cycles(recordings["e_spikes"])[0] * DT_MS
    return {
        "command_peak_hz": float(command_rate.max()),
        "e_rate_pre_hz": mean_rate(recordings["e_spikes"], 0.0, 200.0),
        "e_rate_active_hz": mean_rate(recordings["e_spikes"], 200.0, 1_000.0),
        "e_rate_post_hz": mean_rate(recordings["e_spikes"], 1_000.0, 1_200.0),
        "i_rate_pre_hz": mean_rate(recordings["i_spikes"], 0.0, 200.0),
        "i_rate_active_hz": mean_rate(recordings["i_spikes"], 200.0, 1_000.0),
        "i_rate_post_hz": mean_rate(recordings["i_spikes"], 1_000.0, 1_200.0),
        "active_e_volleys": int(np.sum((peaks >= 200.0) & (peaks < 1_000.0))),
        "post_e_volleys": int(np.sum(peaks >= 1_000.0)),
    }


def main() -> None:
    meta = parse_meta(sys.argv)
    if meta.runpod:
        raise SystemExit("exp097 is a bounded local scout; RunPod is not supported")
    started = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {run_id}")
    with published_run(SLUG, run_id, scale=SCALE, plot_only=meta.plot_only) as (scratch, staging):
        bundle = author_network()
        visual_inputs = make_inputs()
        result = simulate(ExecutionSpec(kind="simulate", executor="graph", graph=bundle.graph, inputs={"drive": torch.from_numpy(visual_inputs).float()}, seed=NETWORK_SEED, recording="full"))
        arrays = recording_arrays(result)
        analysis, state = analyse(arrays, visual_inputs)
        np.savez_compressed(scratch / "recordings.npz", **arrays)
        plot_result(state, analysis, staging / "measured_cycle.svg")
        render_measured_animation(state, staging / "measured_engine.mp4", staging / "measured_engine_poster.png")
        (staging / "animation_state.json").write_text(json.dumps(state, separators=(",", ":")) + "\n")
        (staging / "ping_engine_state.js").write_text("window.EXP097_MEASURED_STATE=" + json.dumps(state, separators=(",", ":")) + ";\n")
        del result, arrays

        transient_results = {}
        for kind, filename, title in (
            ("ramp", "input_ramp_engine", "Linear 0–50–0 Hz input-drive ramp"),
        ):
            dynamic_visual_inputs, command_rate = make_scheduled_inputs(kind)
            dynamic_result = simulate(ExecutionSpec(kind="simulate", executor="graph", graph=bundle.graph, inputs={"drive": torch.from_numpy(dynamic_visual_inputs).float()}, seed=NETWORK_SEED, recording="full"))
            dynamic_arrays = recording_arrays(dynamic_result)
            dynamic_state = build_display_state(
                dynamic_arrays,
                dynamic_visual_inputs,
                0,
                0,
                len(command_rate),
                title,
                seed=VISUAL_SEED,
                command_rate_hz=command_rate,
            )
            render_measured_animation(dynamic_state, staging / f"{filename}.mp4", staging / f"{filename}_poster.png")
            (staging / f"{filename}_state.json").write_text(json.dumps(dynamic_state, separators=(",", ":")) + "\n")
            transient_results[kind] = transient_summary(dynamic_arrays, command_rate)
            del dynamic_result, dynamic_arrays, dynamic_state, dynamic_visual_inputs
        analysis["drive_transients"] = transient_results
        shutil.copy2(ASSETS / "ping_engine.css", staging / "ping_engine.css")
        shutil.copy2(ASSETS / "ping_engine.js", staging / "ping_engine.js")
        shutil.copy2(ASSETS / "ping_engine_storyboard.svg", staging / "ping_engine_storyboard.svg")
        bundle.write(staging / "network.bundle", visualise=True)
        shutil.copy2(staging / "network.bundle" / "reports" / "circuit.svg", staging / "network.svg")
        (staging / "protocol.json").write_text(json.dumps(SCALE, indent=2) + "\n")
        write_numbers(staging, run_id=run_id, duration_s=time.monotonic() - started, payload={"question": "Do recurrent E and I conductances form a useful two-variable portrait of a stochastic PING cycle?", "results": analysis})
        (staging / "SCIENTIFIC_COLLECTION_STATE.md").write_text(f"""# Exp097 ScientificCollectionState

## Registration

- Writing: `writings/exp097.typ`
- Collection: `snnlang`
- Scientific role: exploratory recurrent-state visualization scout
- Lifecycle status: `ExpScout`; retrospectively reconstructed plan and completed
  `ScoutExecution`
- Implementation: `experiments/exp097.py`; tests: `experiments/tests/test_exp097.py`
- Hard dependencies: `experiments/exp083.py`, `tools/snn`, and `tools/snnlang`
- Scout execution: `{run_id}`
- Writing metadata: title `Can a PING cycle be seen as a running engine?`,
  status `complete`, order 11
- Simulation results: `numbers.json`, `measured_cycle.svg`, `measured_engine.mp4`,
  `measured_engine_poster.png`, `animation_state.json`, `input_ramp_engine.mp4`,
  `input_ramp_engine_poster.png`, `input_ramp_engine_state.json`
- Simulation-result web animation: `ping_engine_state.js`, `ping_engine.js`, `ping_engine.css`

## PublicationView

- Current local view uses ad-hoc run `{run_id}` and the two video results.
- No campaign evidence has been accepted or activated as a gold-star view.

## Execution

- Local execution used the frozen 800-E, 200-I network, network seed 83, input
  seeds 8300--8304, 50 Hz/channel drive, 0.1 ms timestep, 2 ms inhibitory
  decay, and five 500 ms trials with a 100 ms transient exclusion.
- Full recordings preserve E/I spikes, E/I voltages, E-to-I AMPA conductance,
  and I-to-E GABA conductance in the run scratch artifact.
- The animation state contains five simulated cycles from the trial selected by
  the frozen median-frequency rule. It is downsampled for display; analyses use
  native-resolution recordings.
- A second simulation uses the same network and input realization with a linear
  0--50--0 Hz/channel drive between two 200 ms silent periods.
## Scientific disposition

- Revise: the conductance plane is coherent but mean voltage improves held-out
  phase and next-volley prediction.
- The constant-drive rhythm is below gamma, and the scout is specific to one
  network realization and its tested operating points.

## Campaign readiness and blockers

- Not campaign-ready: no prospectively frozen plan predates this execution.
- A new scout must first locate a gamma-frequency operating point before a
  multi-realization `ExpStudyPlan` would be scientifically warranted.
- Campaign construction, evidence acceptance, activation, and publication each
  remain separate user review gates.
""")
    print(f"exp097 complete: {run_id}")


if __name__ == "__main__":
    main()
