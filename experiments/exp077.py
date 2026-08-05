"""Experiment 077 through Step 2: empirical pixel-response calibration.

The complete staged contract lives in ``writings/exp077.typ``.  Only Step 1 is
implemented here: the validated local probe and its empirical response library.
Later steps remain explicit hard stops.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import theme  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_id import next_run_id, persist  # noqa: E402

SLUG = "exp077"
N_STEPS = 7
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

PRESENTATION_MS = 200.0
DT_MS = 0.1
N_TIMESTEPS = int(round(PRESENTATION_MS / DT_MS))
TRAINING_RATES_HZ = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 25.0)
PROBE_US = 1.2
SEED = 42
SEEDS = (42, 43, 44)
PROBE_CONDUCTANCES_US = (0.6, 1.2, 2.4)
INTENSITY_LEVELS = np.arange(256, dtype=np.uint16)

# Locked before inspecting the Step 2 pilot.  Each candidate compares two
# independent, equally sized blocks.  The deterministic subset spans zero,
# low, middle, and full intensity; every probe; every registered seed; and five
# rates spanning the registered grid.  A candidate passes only when both the
# 95th-percentile and worst normalized discrepancies pass for mean and unbiased
# sample variance.  Absolute floors prevent near-zero conditions from making a
# relative metric ill-conditioned.
PILOT_CANDIDATE_K = (64, 128, 256, 512)
PILOT_MAX_K = max(PILOT_CANDIDATE_K)
PILOT_INTENSITIES = (0, 16, 64, 128, 192, 255)
PILOT_RATES_HZ = (0.25, 1.0, 3.0, 10.0, 25.0)
PILOT_MEAN_ABS_TOL_MV = 0.15
PILOT_MEAN_REL_TOL = 0.10
PILOT_VARIANCE_ABS_TOL_MV2 = 0.25
PILOT_VARIANCE_REL_TOL = 0.25
PILOT_P95_LIMIT = 1.0
PILOT_WORST_LIMIT = 2.0
COUNT_VALIDATION_DRAWS = 6_000
COUNT_VALIDATION_RATE_HZ = 25.0
COUNT_VALIDATION_INTENSITY = 0.8
EARLY_SPIKE_MS = 20.0
LATE_SPIKE_MS = 180.0
SPIKE_TIME_GRID_MS = np.linspace(0.0, PRESENTATION_MS - DT_MS, 101)

PARAMETERS = {
    "C_m_nF": 1.0,
    "g_L_uS": 0.05,
    "E_L_mV": -65.0,
    "E_e_mV": 0.0,
    "tau_ampa_ms": 2.0,
    "probe_uS": PROBE_US,
}

STAGE_NAMES: dict[int, str] = {
    1: "generate and validate filter-matched pixel features",
    2: "generate the empirical pixel-response library",
    3: "calculate and test the dependent linear-filter prediction",
    4: "construct and validate complete sampled feature images",
    5: "train the mixed-rate nonlinear and linear decoders",
    6: "evaluate held-out psychometric curves and select thresholds",
    7: "write the variable-rate training-range decision",
}


def _not_implemented(step: int) -> None:
    raise NotImplementedError(
        f"exp077 Step {step} is specified but not implemented: "
        f"{STAGE_NAMES[step]}. Follow writings/exp077.typ and register the "
        "completed stage in IMPLEMENTED_STEPS."
    )


def encode_poisson(
    intensities: np.ndarray, rate_hz: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw independent Bernoulli spike trains with p = rate * dt * intensity."""
    values = np.asarray(intensities, dtype=np.float64)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("pixel intensities must lie in [0, 1]")
    probability = rate_hz * (DT_MS / 1000.0) * values
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("Bernoulli encoder probability must lie in [0, 1]")
    return rng.random((N_TIMESTEPS, *values.shape)) < probability


def probe_spikes(
    spikes: np.ndarray, probe_uS: float = PROBE_US
) -> dict[str, np.ndarray | float]:
    """Apply the registered decay-then-add AMPA and non-spiking membrane probe."""
    spike_array = np.asarray(spikes, dtype=np.float64)
    if spike_array.shape[0] != N_TIMESTEPS:
        raise ValueError(f"expected {N_TIMESTEPS} timesteps")
    channel_shape = spike_array.shape[1:]
    g = np.zeros(channel_shape, dtype=np.float64)
    v = np.full(channel_shape, PARAMETERS["E_L_mV"], dtype=np.float64)
    g_trace = np.empty_like(spike_array)
    v_trace = np.empty_like(spike_array)
    ampa_decay = math.exp(-DT_MS / PARAMETERS["tau_ampa_ms"])

    for timestep, incoming in enumerate(spike_array):
        g = g * ampa_decay + probe_uS * incoming
        g_total = PARAMETERS["g_L_uS"] + g
        v_inf = (
            PARAMETERS["g_L_uS"] * PARAMETERS["E_L_mV"] + g * PARAMETERS["E_e_mV"]
        ) / g_total
        v = v_inf + (v - v_inf) * np.exp(-DT_MS * g_total / PARAMETERS["C_m_nF"])
        g_trace[timestep] = g
        v_trace[timestep] = v

    z = np.mean(v_trace - PARAMETERS["E_L_mV"], axis=0)
    return {"conductance_uS": g_trace, "voltage_mV": v_trace, "z_mV": z}


def single_spike_train(spike_time_ms: float) -> np.ndarray:
    spikes = np.zeros(N_TIMESTEPS, dtype=np.float64)
    index = int(round(spike_time_ms / DT_MS))
    if not 0 <= index < N_TIMESTEPS:
        raise ValueError("spike time must fall within the presentation")
    spikes[index] = 1.0
    return spikes


def _validation_record(ok: bool, **evidence: Any) -> dict[str, Any]:
    return {"ok": bool(ok), **evidence}


def tools_snn_reference(spikes: np.ndarray) -> dict[str, Any]:
    """Run the shared engine in a subprocess and return its probe trajectory."""
    code = r"""
import json
import math
import sys
import torch
from tools.snn import models

spikes = json.loads(sys.stdin.read())
torch.set_default_dtype(torch.float64)
v = torch.tensor([models.E_L])
ref = torch.zeros(1)
g = torch.zeros(1)
weight = torch.tensor([[1.2]])
decay = math.exp(-0.1 / models.tau_ampa)
conductance = []
voltage = []
def zero_spike(value):
    return torch.zeros_like(value)
for incoming in spikes:
    g = models.exp_synapse(g, torch.tensor([[incoming]]), weight, decay)
    v, _, ref = models.lif_step_expeuler(
        v, ref, g, None, models.C_m_E, models.g_L_E, 1, zero_spike,
        dt_override=0.1,
    )
    conductance.append(float(g.item()))
    voltage.append(float(v.item()))
print(json.dumps({
    "conductance_uS": conductance,
    "voltage_mV": voltage,
    "parameters": {
        "C_m_nF": models.C_m_E,
        "g_L_uS": models.g_L_E,
        "E_L_mV": models.E_L,
        "E_e_mV": models.E_e,
        "tau_ampa_ms": models.tau_ampa,
    },
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        input=json.dumps(np.asarray(spikes, dtype=float).tolist()),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def validate_probe() -> dict[str, dict[str, Any]]:
    """Run every predeclared focused validation and return its evidence."""
    probability_grid = np.asarray(TRAINING_RATES_HZ) * DT_MS / 1000.0
    probability_ok = bool(np.all((probability_grid >= 0.0) & (probability_grid <= 1.0)))

    rng = np.random.default_rng(SEED)
    pixels = np.full((COUNT_VALIDATION_DRAWS, 4), COUNT_VALIDATION_INTENSITY)
    # Shape: time, draw, independent pixel channel.
    encoded = encode_poisson(pixels, COUNT_VALIDATION_RATE_HZ, rng)
    counts = encoded.sum(axis=0).astype(np.float64)
    p = COUNT_VALIDATION_RATE_HZ * DT_MS / 1000.0 * COUNT_VALIDATION_INTENSITY
    expected_mean = N_TIMESTEPS * p
    expected_variance = N_TIMESTEPS * p * (1.0 - p)
    empirical_mean = float(counts.mean())
    empirical_variance = float(counts.var(ddof=1))
    mean_se = math.sqrt(expected_variance / counts.size)
    variance_rel_error = abs(empirical_variance - expected_variance) / expected_variance
    count_ok = (
        abs(empirical_mean - expected_mean) <= 5.0 * mean_se
        and variance_rel_error < 0.08
    )

    corr = np.corrcoef(counts, rowvar=False)
    off_diagonal = corr[np.triu_indices(corr.shape[0], k=1)]
    max_abs_correlation = float(np.max(np.abs(off_diagonal)))
    independence_ok = max_abs_correlation < 0.05

    synaptic_spikes = np.zeros(N_TIMESTEPS)
    synaptic_spikes[0] = 1.0
    synaptic = probe_spikes(synaptic_spikes)
    g_trace = np.asarray(synaptic["conductance_uS"])
    expected_g0 = PROBE_US
    expected_g1 = PROBE_US * math.exp(-DT_MS / PARAMETERS["tau_ampa_ms"])
    synapse_error = float(
        max(abs(g_trace[0] - expected_g0), abs(g_trace[1] - expected_g1))
    )
    synapse_ok = synapse_error < 1e-12

    g = PROBE_US
    v0 = PARAMETERS["E_L_mV"]
    g_total = PARAMETERS["g_L_uS"] + g
    v_inf = (
        PARAMETERS["g_L_uS"] * PARAMETERS["E_L_mV"] + g * PARAMETERS["E_e_mV"]
    ) / g_total
    expected_v1 = v_inf + (v0 - v_inf) * math.exp(
        -DT_MS * g_total / PARAMETERS["C_m_nF"]
    )
    observed_v1 = float(np.asarray(synaptic["voltage_mV"])[0])
    voltage_error = abs(observed_v1 - expected_v1)
    voltage_ok = voltage_error < 1e-12

    replay_a = encode_poisson(np.asarray([0.2, 0.8]), 25.0, np.random.default_rng(SEED))
    replay_b = encode_poisson(np.asarray([0.2, 0.8]), 25.0, np.random.default_rng(SEED))
    replay_ok = bool(np.array_equal(replay_a, replay_b))

    comparison_spikes = np.zeros(N_TIMESTEPS, dtype=np.float64)
    comparison_spikes[[0, 57, 431, 1000, 1764]] = 1.0
    local = probe_spikes(comparison_spikes)
    engine = tools_snn_reference(comparison_spikes)
    engine_g = engine["conductance_uS"]
    engine_v = engine["voltage_mV"]
    engine_g_error = float(
        np.max(np.abs(np.asarray(local["conductance_uS"]) - engine_g))
    )
    engine_v_error = float(np.max(np.abs(np.asarray(local["voltage_mV"]) - engine_v)))
    parameters_match = engine["parameters"] == {
        key: value for key, value in PARAMETERS.items() if key != "probe_uS"
    }
    engine_ok = engine_g_error < 1e-12 and engine_v_error < 1e-11 and parameters_match

    early_z = float(probe_spikes(single_spike_train(EARLY_SPIKE_MS))["z_mV"])
    late_z = float(probe_spikes(single_spike_train(LATE_SPIKE_MS))["z_mV"])
    timing_difference = early_z - late_z
    timing_ok = early_z > late_z > 0.0

    validations = {
        "encoder_probability_bounds": _validation_record(
            probability_ok,
            minimum=float(probability_grid.min()),
            maximum=float(probability_grid.max()),
        ),
        "spike_count_moments": _validation_record(
            count_ok,
            draws=COUNT_VALIDATION_DRAWS,
            channels=4,
            expected_mean=expected_mean,
            empirical_mean=empirical_mean,
            mean_standard_error=mean_se,
            expected_variance=expected_variance,
            empirical_variance=empirical_variance,
            variance_relative_error=variance_rel_error,
        ),
        "pixel_independence": _validation_record(
            independence_ok,
            maximum_absolute_count_correlation=max_abs_correlation,
            threshold=0.05,
        ),
        "ampa_decay_then_add": _validation_record(
            synapse_ok,
            observed_first_uS=float(g_trace[0]),
            expected_first_uS=expected_g0,
            observed_second_uS=float(g_trace[1]),
            expected_second_uS=expected_g1,
            maximum_absolute_error_uS=synapse_error,
        ),
        "exponential_euler_one_step": _validation_record(
            voltage_ok,
            observed_mV=observed_v1,
            expected_mV=expected_v1,
            absolute_error_mV=voltage_error,
            tolerance_mV=1e-12,
        ),
        "deterministic_replay": _validation_record(replay_ok, seed=SEED),
        "tools_snn_uncoupled_cell_agreement": _validation_record(
            engine_ok,
            conductance_maximum_absolute_error_uS=engine_g_error,
            voltage_maximum_absolute_error_mV=engine_v_error,
            conductance_tolerance_uS=1e-12,
            voltage_tolerance_mV=1e-11,
            target_parameters_match=parameters_match,
            shared_functions=["models.exp_synapse", "models.lif_step_expeuler"],
        ),
        "spike_timing_sensitivity": _validation_record(
            timing_ok,
            early_spike_ms=EARLY_SPIKE_MS,
            late_spike_ms=LATE_SPIKE_MS,
            early_z_mV=early_z,
            late_z_mV=late_z,
            early_minus_late_z_mV=timing_difference,
        ),
    }
    if not all(record["ok"] for record in validations.values()):
        failed = [name for name, record in validations.items() if not record["ok"]]
        raise RuntimeError(f"Step 1 validation failed: {', '.join(failed)}")
    return validations


def make_plot_record() -> dict[str, Any]:
    cases = {
        "no_spike": np.zeros(N_TIMESTEPS),
        "early_spike": single_spike_train(EARLY_SPIKE_MS),
        "late_spike": single_spike_train(LATE_SPIKE_MS),
    }
    time_ms = np.arange(N_TIMESTEPS, dtype=np.float64) * DT_MS
    traces: dict[str, Any] = {"time_ms": time_ms.tolist(), "cases": {}}
    for name, spikes in cases.items():
        result = probe_spikes(spikes)
        traces["cases"][name] = {
            "spikes": spikes.astype(int).tolist(),
            "conductance_uS": np.asarray(result["conductance_uS"]).tolist(),
            "voltage_mV": np.asarray(result["voltage_mV"]).tolist(),
            "z_mV": float(result["z_mV"]),
        }
    timing_z = [
        float(probe_spikes(single_spike_train(t))["z_mV"]) for t in SPIKE_TIME_GRID_MS
    ]
    traces["timing_curve"] = {
        "spike_time_ms": SPIKE_TIME_GRID_MS.tolist(),
        "z_mV": timing_z,
    }
    return traces


def plot_probe(record: dict[str, Any], path: Path) -> None:
    time_ms = np.asarray(record["time_ms"])
    cases = record["cases"]
    colors = {
        "no_spike": theme.FAINT,
        "early_spike": theme.INK_BLACK,
        "late_spike": theme.DEEP_RED,
    }
    labels = {
        "no_spike": "No spike",
        "early_spike": "Early: 20 ms",
        "late_spike": "Late: 180 ms",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), constrained_layout=True)
    ax_spike, ax_g, ax_v, ax_timing = axes.flat
    for name, case in cases.items():
        linestyle = ":" if name == "no_spike" else "-"
        zorder = 4 if name == "no_spike" else 2
        spikes = np.asarray(case["spikes"])
        spike_times = time_ms[spikes > 0]
        if spike_times.size:
            ax_spike.vlines(
                spike_times, 0.0, 1.0, color=colors[name], lw=2.2, label=labels[name]
            )
        elif name == "no_spike":
            ax_spike.plot([], [], color=colors[name], lw=2.2, label=labels[name])
        ax_g.plot(
            time_ms,
            case["conductance_uS"],
            color=colors[name],
            lw=1.7,
            ls=linestyle,
            zorder=zorder,
            label=labels[name],
        )
        ax_v.plot(
            time_ms,
            case["voltage_mV"],
            color=colors[name],
            lw=1.7,
            ls=linestyle,
            zorder=zorder,
            label=labels[name],
        )
    ax_spike.set(title="A  Matched input counts", ylabel="Input spike")
    ax_spike.set_ylim(-0.05, 1.12)
    ax_spike.set_yticks([0, 1])
    ax_spike.legend(frameon=False, ncol=3, fontsize=8, loc="upper center")
    ax_g.set(title="B  AMPA filter", ylabel="Conductance (μS)")
    ax_v.set(
        title="C  Subthreshold membrane", xlabel="Time (ms)", ylabel="Voltage (mV)"
    )
    timing = record["timing_curve"]
    ax_timing.plot(
        timing["spike_time_ms"], timing["z_mV"], color=theme.INK_BLACK, lw=2.2
    )
    ax_timing.scatter(
        [EARLY_SPIKE_MS, LATE_SPIKE_MS],
        [cases["early_spike"]["z_mV"], cases["late_spike"]["z_mV"]],
        color=[theme.INK_BLACK, theme.DEEP_RED],
        zorder=3,
    )
    ax_timing.set(
        title="D  Equal count, different feature",
        xlabel="Single-spike time (ms)",
        ylabel="Mean voltage feature z (mV)",
    )
    for ax in axes.flat:
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    ax_g.set_xlabel("Time (ms)")
    fig.suptitle(
        "Finite-window timing survives AMPA and membrane filtering",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def simulate_condition_grid(
    intensities: tuple[int, ...] | np.ndarray,
    rates_hz: tuple[float, ...],
    probes_uS: tuple[float, ...],
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate z for a probe × rate × intensity grid without storing traces."""
    intensity_array = np.asarray(intensities, dtype=np.float64) / 255.0
    probabilities = (
        np.asarray(rates_hz, dtype=np.float64)[:, None]
        * (DT_MS / 1000.0)
        * intensity_array[None, :]
    )
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("condition grid produces an invalid Bernoulli probability")
    probability_grid = np.broadcast_to(
        probabilities[None, :, :, None],
        (len(probes_uS), len(rates_hz), len(intensity_array), draws),
    )
    probe_grid = np.asarray(probes_uS, dtype=np.float64)[:, None, None, None]
    g = np.zeros(probability_grid.shape, dtype=np.float64)
    v = np.full(probability_grid.shape, PARAMETERS["E_L_mV"], dtype=np.float64)
    v_sum = np.zeros(probability_grid.shape, dtype=np.float64)
    ampa_decay = math.exp(-DT_MS / PARAMETERS["tau_ampa_ms"])
    for _ in range(N_TIMESTEPS):
        spikes = rng.random(probability_grid.shape) < probability_grid
        g = g * ampa_decay + probe_grid * spikes
        g_total = PARAMETERS["g_L_uS"] + g
        v_inf = (
            PARAMETERS["g_L_uS"] * PARAMETERS["E_L_mV"]
            + g * PARAMETERS["E_e_mV"]
        ) / g_total
        v = v_inf + (v - v_inf) * np.exp(
            -DT_MS * g_total / PARAMETERS["C_m_nF"]
        )
        v_sum += v - PARAMETERS["E_L_mV"]
    return (v_sum / N_TIMESTEPS).astype(np.float32)


def _step2_rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, 77, 2, stream]))


def _normalised_difference(
    first: np.ndarray, second: np.ndarray, absolute_floor: float, relative: float
) -> np.ndarray:
    denominator = np.maximum(
        absolute_floor, relative * np.maximum(np.abs(first), np.abs(second))
    )
    return np.abs(first - second) / denominator


def run_step2_pilot() -> dict[str, Any]:
    """Execute the locked, bounded draw-count convergence pilot."""
    largest = 2 * PILOT_MAX_K
    by_seed = []
    for seed in SEEDS:
        by_seed.append(
            simulate_condition_grid(
                PILOT_INTENSITIES,
                PILOT_RATES_HZ,
                PROBE_CONDUCTANCES_US,
                largest,
                _step2_rng(seed, 0),
            )
        )
    values = np.stack(by_seed, axis=0)
    nonzero = np.asarray(PILOT_INTENSITIES) > 0
    trajectory = []
    selected: int | None = None
    for candidate in PILOT_CANDIDATE_K:
        first = values[..., :candidate]
        second = values[..., candidate : 2 * candidate]
        mean_a = first.mean(axis=-1)
        mean_b = second.mean(axis=-1)
        var_a = first.var(axis=-1, ddof=1)
        var_b = second.var(axis=-1, ddof=1)
        mean_error = _normalised_difference(
            mean_a[..., nonzero],
            mean_b[..., nonzero],
            PILOT_MEAN_ABS_TOL_MV,
            PILOT_MEAN_REL_TOL,
        )
        variance_error = _normalised_difference(
            var_a[..., nonzero],
            var_b[..., nonzero],
            PILOT_VARIANCE_ABS_TOL_MV2,
            PILOT_VARIANCE_REL_TOL,
        )
        metrics = {
            "K": candidate,
            "mean_p95_normalised_error": float(np.quantile(mean_error, 0.95)),
            "mean_maximum_normalised_error": float(mean_error.max()),
            "variance_p95_normalised_error": float(
                np.quantile(variance_error, 0.95)
            ),
            "variance_maximum_normalised_error": float(variance_error.max()),
        }
        metrics["passed"] = bool(
            metrics["mean_p95_normalised_error"] <= PILOT_P95_LIMIT
            and metrics["mean_maximum_normalised_error"] <= PILOT_WORST_LIMIT
            and metrics["variance_p95_normalised_error"] <= PILOT_P95_LIMIT
            and metrics["variance_maximum_normalised_error"] <= PILOT_WORST_LIMIT
        )
        trajectory.append(metrics)
        if selected is None and metrics["passed"]:
            selected = candidate
    zero_exact = bool(np.all(values[:, :, :, 0, :] == 0.0))
    return {
        "candidate_K": list(PILOT_CANDIDATE_K),
        "hard_maximum_K": PILOT_MAX_K,
        "evaluation_condition_count": int(
            len(SEEDS)
            * len(PROBE_CONDUCTANCES_US)
            * len(PILOT_RATES_HZ)
            * len(PILOT_INTENSITIES)
        ),
        "trajectory": trajectory,
        "selected_K": selected,
        "passed": selected is not None and zero_exact,
        "zero_intensity_exact": zero_exact,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_step2_pilot(pilot: dict[str, Any], path: Path) -> None:
    """Plot the locked convergence trajectory, including a failed hard stop."""
    rows = pilot["trajectory"]
    candidate = np.asarray([row["K"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.66), constrained_layout=True)
    series = (
        ("mean_p95_normalised_error", "95th percentile", "o", "-"),
        ("mean_maximum_normalised_error", "Maximum", "s", "--"),
    )
    for key, label, marker, linestyle in series:
        axes[0].plot(
            candidate,
            [row[key] for row in rows],
            color=theme.INK_BLACK,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
    variance_series = (
        ("variance_p95_normalised_error", "95th percentile", "o", "-"),
        ("variance_maximum_normalised_error", "Maximum", "s", "--"),
    )
    for key, label, marker, linestyle in variance_series:
        axes[1].plot(
            candidate,
            [row[key] for row in rows],
            color=theme.DEEP_RED,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
    for ax, title in zip(axes, ("A  Conditional mean", "B  Sample variance")):
        ax.axhline(PILOT_P95_LIMIT, color=theme.FAINT, linestyle=":", linewidth=1.4)
        ax.axhline(PILOT_WORST_LIMIT, color=theme.FAINT, linestyle="-.", linewidth=1.4)
        ax.set(
            xscale="log",
            xticks=candidate,
            xticklabels=[str(value) for value in candidate],
            xlabel="Draws per comparison block, K",
            ylabel="Normalized discrepancy",
            title=title,
        )
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)


def record_step2_pilot(pilot: dict[str, Any], duration_s: float) -> None:
    """Freeze the pilot outcome without erasing the completed Step 1 evidence."""
    outcome_path = FIGURES / "step2_pilot_outcome.json"
    outcome_path.write_text(json.dumps(pilot, indent=2) + "\n")
    plot_step2_pilot(pilot, FIGURES / "response_library.png")
    numbers_path = FIGURES / "numbers.json"
    numbers = json.loads(numbers_path.read_text())
    numbers["step"] = 2
    numbers["status"] = "killed_at_locked_convergence_gate"
    numbers["scope"] = (
        "Step 1 remains complete; Step 2 stopped at its predeclared draw-count "
        "pilot; no final library, decoder, held-out test, threshold, or PING run"
    )
    numbers["step2"] = {
        "status": "failed_locked_pilot",
        "pilot": pilot,
        "pilot_duration_s": round(duration_s, 1),
        "final_library_generated": False,
        "later_steps_run": False,
        "paid_compute_usd": 0.0,
    }
    numbers_path.write_text(json.dumps(numbers, indent=2) + "\n")
    protocol_path = FIGURES / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol.update(
        {
            "attempted_through_step": 2,
            "step2_status": "failed_locked_pilot",
            "step2_selected_K": None,
            "step2_final_library_generated": False,
            "later_steps": "not run",
        }
    )
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    reproducer = {
        "command": (
            "EXP077_THROUGH_STEP=2 EXP077_STEP2_PILOT_ONLY=1 "
            "uv run python experiments/exp077.py"
        ),
        "paid_compute": False,
        "expected_outcome": "locked convergence failure; no final library",
        "expected_outputs": [
            "artifacts/data/exp077/step2_pilot_outcome.json",
            "artifacts/data/exp077/response_library.png",
            "artifacts/data/exp077/numbers.json",
        ],
    }
    (FIGURES / "reproducer.json").write_text(json.dumps(reproducer, indent=2) + "\n")
    (FIGURES / "provenance.json").write_text(
        json.dumps(_git_metadata(), indent=2) + "\n"
    )
    manifest = {
        "status": "pilot_failed_no_library",
        "selected_K": None,
        "library_generated": False,
        "library_shape": None,
        "library_storage_bytes": 0,
        "dtype": None,
        "ordered_axes": ["seed", "probe_uS", "rate_hz", "intensity", "draw"],
        "seed_recipe": (
            "numpy.random.SeedSequence([registered_seed, 77, 2, stream]); "
            "conditions occupy fixed probe-rate-intensity array indices"
        ),
        "pilot_outcome_sha256": sha256_file(outcome_path),
        "response_library_figure_sha256": sha256_file(
            FIGURES / "response_library.png"
        ),
        "locked_protocol_sha256": sha256_file(
            FIGURES / "step2_pilot_protocol.json"
        ),
        "regeneration_command": reproducer["command"],
    }
    (FIGURES / "step2_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


def _git_metadata() -> dict[str, Any]:
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=REPO, text=True
    )
    source = Path(__file__).read_bytes()
    return {
        "git_sha_at_execution": sha,
        "worktree_dirty_at_execution": bool(status),
        "runner_sha256": hashlib.sha256(source).hexdigest(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": sys.platform,
    }


def step_1() -> None:
    started = time.perf_counter()
    theme.apply()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    run_id = next_run_id(SLUG)
    validations = validate_probe()
    plot_record = make_plot_record()
    plot_probe(plot_record, FIGURES / "probe_dynamics.svg")

    protocol = {
        "experiment": SLUG,
        "implemented_step": 1,
        "presentation_ms": PRESENTATION_MS,
        "dt_ms": DT_MS,
        "timesteps": N_TIMESTEPS,
        "training_rates_hz": list(TRAINING_RATES_HZ),
        "probe_uS": PROBE_US,
        "seed": SEED,
        "equations": "writings/exp077.typ, Equations 2-7",
        "update_order": "AMPA decay, spike kick, exponential-Euler membrane update",
        "feature": "mean_t(v(t) - E_L)",
        "later_steps": "not run",
    }
    provenance = _git_metadata()
    reproducer = {
        "command": "EXP077_THROUGH_STEP=1 uv run python experiments/exp077.py",
        "paid_compute": False,
        "expected_outputs": [
            "artifacts/data/exp077/numbers.json",
            "artifacts/data/exp077/probe_dynamics.svg",
            "artifacts/data/exp077/protocol.json",
            "artifacts/data/exp077/provenance.json",
        ],
    }
    (FIGURES / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (FIGURES / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (FIGURES / "reproducer.json").write_text(json.dumps(reproducer, indent=2) + "\n")
    duration = time.perf_counter() - started
    write_numbers(
        FIGURES,
        run_id=run_id,
        duration_s=duration,
        payload={
            "step": 1,
            "status": "complete",
            "scope": "filter-matched feature generation only; no decoder or PING claim",
            "parameters": PARAMETERS,
            "protocol": protocol,
            "validations": validations,
            "plot_data": plot_record,
            "all_validations_passed": all(row["ok"] for row in validations.values()),
            "paid_compute_usd": 0.0,
            "reproducer": reproducer["command"],
        },
    )
    persist(SLUG, run_id)
    print(
        f"exp077 Step 1 complete: {len(validations)}/{len(validations)} validations passed"
    )
    print(f"Artifacts: {FIGURES.relative_to(REPO)}")


def step_2() -> None:
    started = time.perf_counter()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    pilot = run_step2_pilot()
    record_step2_pilot(pilot, time.perf_counter() - started)
    if not pilot["passed"]:
        raise RuntimeError(
            "Step 2 draw-count pilot did not pass by the locked maximum K; "
            "the final library was not generated"
        )
    print(f"exp077 Step 2 pilot selected K={pilot['selected_K']}")
    if os.environ.get("EXP077_STEP2_PILOT_ONLY") == "1":
        return
    raise RuntimeError("Step 2 final-library generation is not yet enabled")


def step_3() -> None:
    _not_implemented(3)


def step_4() -> None:
    _not_implemented(4)


def step_5() -> None:
    _not_implemented(5)


def step_6() -> None:
    _not_implemented(6)


def step_7() -> None:
    _not_implemented(7)


STAGE_FUNCTIONS: dict[int, Callable[[], None]] = {
    1: step_1,
    2: step_2,
    3: step_3,
    4: step_4,
    5: step_5,
    6: step_6,
    7: step_7,
}

IMPLEMENTED_STEPS: frozenset[int] = frozenset({1, 2})


def requested_through_step() -> int:
    raw = os.environ.get("EXP077_THROUGH_STEP", str(N_STEPS))
    try:
        step = int(raw)
    except ValueError as exc:
        raise SystemExit(
            "EXP077_THROUGH_STEP must be an integer from 1 through 7"
        ) from exc
    if step not in STAGE_NAMES:
        raise SystemExit("EXP077_THROUGH_STEP must be an integer from 1 through 7")
    return step


def main() -> None:
    through_step = requested_through_step()
    for step in range(1, through_step + 1):
        if step not in IMPLEMENTED_STEPS:
            _not_implemented(step)
        STAGE_FUNCTIONS[step]()


if __name__ == "__main__":
    main()
