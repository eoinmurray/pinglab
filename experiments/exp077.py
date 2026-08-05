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
    _not_implemented(2)


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

IMPLEMENTED_STEPS: frozenset[int] = frozenset({1})


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
