"""Experiment 080: filter-matched MNIST calibration and decoding.

The complete staged contract lives in ``writings/exp080.typ``. Steps 1--4
calibrate and validate the feature generator. Steps 5--7 train frozen decoders,
evaluate held-out psychometric curves, and select the later PING rate range.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import theme  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_id import next_run_id, persist  # noqa: E402

SLUG = "exp080"
N_STEPS = 7
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

PRESENTATION_MS = 200.0
DT_MS = 0.1
N_TIMESTEPS = int(round(PRESENTATION_MS / DT_MS))
TRAINING_RATES_HZ = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 25.0)
# Steps 1--4 retain their authenticated calibration table on TRAINING_RATES_HZ.
# Decoder training and held-out evaluation give the added sparse rates exactly
# the same sampling status as every previously registered rate.
DECODER_RATES_HZ = (
    0.01,
    0.05,
    0.1,
    *TRAINING_RATES_HZ,
)
EXPANDED_RATE_PROTOCOL_SHA256 = (
    "07e083cfe6d44bd172c18da2be8fe36cafc321025a86f3a567813f8916bb67aa"
)
PRE_EXPANSION_EXACT_MODAL_COST_USD = 4.2940396
PROBE_US = 1.2
SEED = 42
SEEDS = (42, 43, 44)
PROBE_CONDUCTANCES_US = (0.6, 1.2, 2.4)
INTENSITY_LEVELS = np.arange(256, dtype=np.uint16)
LIBRARY_K = 2048
LIBRARY_SHAPE = (
    len(SEEDS),
    len(PROBE_CONDUCTANCES_US),
    len(TRAINING_RATES_HZ),
    len(INTENSITY_LEVELS),
    LIBRARY_K,
)
LIBRARY_AXIS_ORDER = ("seed", "probe_uS", "rate_hz", "intensity", "draw")
LIBRARY_CHUNK_INTENSITIES = 8
LIBRARY_SCRATCH = REPO / "temp" / SLUG / "response_library.float32.npy"
LIBRARY_PROGRESS = REPO / "temp" / SLUG / "response_library.progress.json"
LIBRARY_STREAM = 1
DIAGNOSTIC_K = 512
REPLAY_CHUNK_INDICES = (0,)
DIRECT_VALIDATION_INTENSITIES = (64, 128, 255)
DIRECT_VALIDATION_RATES_HZ = (0.25, 3.0, 25.0)
MONOTONIC_Z = 3.0
BOOTSTRAP_CANDIDATE_K = (64, 128, 256, 512, 1024, 2048)
BOOTSTRAP_REPETITIONS = 200
BOOTSTRAP_PASS_FREQUENCY = 0.95
LIBRARY_SHA256 = "5184788979b5fa7fa9a3f38936399b8e2724d63914019f1306a7171981b36783"
AMENDMENT_PATH = FIGURES / "step3_step4_exploratory_amendment.json"
FREQUENCY_BOUNDS_HZ = (1e-4, 1e6)
FREQUENCY_GRID_POINTS = (8193, 16385, 32769)
FREQUENCY_WIDE_BOUNDS_HZ = (1e-5, 1e7)
QUADRATURE_REL_TOL = 0.002
GAIN_OPERATING_RATES_HZ = (0.25, 3.0, 25.0)
GAIN_FREQUENCIES_HZ = (1.0, 10.0, 100.0)
GAIN_MODULATION_FRACTION = 0.01
GAIN_REL_TOL = 0.03
STEP4_RATES: dict[str, float] = {"low": 0.25, "transitional": 3.0, "high": 25.0}
STEP4_IMAGE_INDICES = tuple(range(16))
STEP4_DIAGNOSTIC_INDICES: dict[str, int] = {"low": 0, "transitional": 1, "high": 2}
STEP4_REPLICATES = 8
STEP4_SEED = 42

TRAIN_INDICES = (0, 55_000)
VALIDATION_INDICES = (55_000, 60_000)
DECODER_BATCH_SIZE = 256
DECODER_LEARNING_RATE = 0.001
DECODER_MAX_EPOCHS = 15
DECODER_HIDDEN = 1024
LINEAR_WEIGHT_DECAYS = (1e-5, 1e-4, 1e-3)
CHANCE_ACCURACY = 0.10
USEFUL_ACCURACY = 0.50
EVALUATION_DRAWS = 3
BOOTSTRAP_REPETITIONS_HELDOUT = 2000
CONFIDENCE_LEVEL = 0.95


def savefig_atomic(fig: Any, path: Path, **kwargs: Any) -> None:
    """Write a figure atomically so the live site builder never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}.", suffix=path.suffix, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        fig.savefig(temporary_path, **kwargs)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


# Locked before inspecting the Step 2 pilot.  Each candidate compares two
# independent, equally sized blocks.  The deterministic subset spans zero,
# low, middle, and full intensity; every probe; every registered seed; and five
# rates spanning the registered grid.  A candidate passes only when both the
# 95th-percentile and worst normalized discrepancies pass for mean and unbiased
# sample variance.  Absolute floors prevent near-zero conditions from making a
# relative metric ill-conditioned.
# The original predeclared pilot used (64, 128, 256, 512) and is preserved in
# step2_pilot_outcome.json.  After that gate failed, the user explicitly
# authorized this follow-up extension without changing any tolerances.
PILOT_CANDIDATE_K = (1024, 2048)
PILOT_MAX_K = max(PILOT_CANDIDATE_K)
PILOT_OUTCOME_NAME = "step2_pilot_extension_outcome.json"
PILOT_FIGURE_NAME = "response_library_extension.png"
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
    2: "generate the empirical pixel-response table",
    3: "calculate and test the dependent linear-filter prediction",
    4: "construct and validate complete sampled feature images",
    5: "train the mixed-rate nonlinear and linear decoders",
    6: "evaluate held-out psychometric curves and select thresholds",
    7: "write the variable-rate training-range decision",
}


def _not_implemented(step: int) -> None:
    raise NotImplementedError(
        f"exp080 Step {step} is specified but not implemented: "
        f"{STAGE_NAMES[step]}. Follow writings/exp080.typ and register the "
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
        raise ValueError("per-timestep spike probability must lie in [0, 1]")
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
    fig, axes = plt.subplots(
        2, 2, figsize=(8.2, 5.9), constrained_layout=True, sharex=True
    )
    ax_spikes, ax_g, ax_v, ax_timing = axes.flat
    for name, color, label in (
        ("early_spike", theme.INK_BLACK, "Spike at 20 ms"),
        ("late_spike", theme.DEEP_RED, "Spike at 180 ms"),
    ):
        spikes = np.asarray(cases[name]["spikes"])
        ax_spikes.vlines(time_ms[spikes > 0], 0, 1, color=color, lw=2, label=label)
        ax_g.plot(
            time_ms,
            cases[name]["conductance_uS"],
            color=color,
            lw=1.8,
            label=label,
        )
        ax_v.plot(time_ms, cases[name]["voltage_mV"], color=color, lw=2, label=label)
        spike_time = EARLY_SPIKE_MS if name == "early_spike" else LATE_SPIKE_MS
        ax_v.axvline(spike_time, color=color, lw=0.8, alpha=0.45)
    ax_spikes.set(
        title="A  One early or late input spike",
        xlabel="Time (ms)",
        ylabel="Input spike",
        ylim=(-0.05, 1.12),
    )
    ax_spikes.set_yticks((0, 1))
    ax_spikes.legend(frameon=False, fontsize=8)
    ax_g.set(
        title="B  AMPA synapse response",
        xlabel="Time (ms)",
        ylabel="Conductance (μS)",
    )
    ax_v.axhline(PARAMETERS["E_L_mV"], color=theme.FAINT, lw=0.8)
    ax_v.set(
        title="C  Subthreshold membrane response",
        xlabel="Time (ms)",
        ylabel="Membrane voltage (mV)",
    )
    ax_v.legend(frameon=False, fontsize=8)
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
        title="D  Timing survives window averaging",
        xlabel="Single-spike time (ms)",
        ylabel="Mean voltage feature z (mV)",
    )
    for ax in axes.flat:
        ax.set_xlim(0, PRESENTATION_MS)
        ax.tick_params(axis="x", labelbottom=True)
        ax.spines[["top", "right"]].set_visible(False)
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
            PARAMETERS["g_L_uS"] * PARAMETERS["E_L_mV"] + g * PARAMETERS["E_e_mV"]
        ) / g_total
        v = v_inf + (v - v_inf) * np.exp(-DT_MS * g_total / PARAMETERS["C_m_nF"])
        v_sum += v - PARAMETERS["E_L_mV"]
    return (v_sum / N_TIMESTEPS).astype(np.float32)


def _step2_rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, 77, 2, stream]))


def _library_chunk_rng(seed: int, chunk_index: int) -> np.random.Generator:
    """Return the independently reproducible RNG for one full-table chunk."""
    return np.random.default_rng(
        np.random.SeedSequence([seed, 77, 2, LIBRARY_STREAM, chunk_index])
    )


def _library_chunks() -> list[tuple[int, int, int]]:
    return [
        (index, start, min(start + LIBRARY_CHUNK_INTENSITIES, len(INTENSITY_LEVELS)))
        for index, start in enumerate(
            range(0, len(INTENSITY_LEVELS), LIBRARY_CHUNK_INTENSITIES)
        )
    ]


def _open_library(mode: Literal["r", "r+"] = "r") -> np.memmap:
    return np.load(LIBRARY_SCRATCH, mmap_mode=mode)


def generate_full_library() -> dict[str, Any]:
    """Generate the selected-K empirical response table into a resumable float32 memmap."""
    LIBRARY_SCRATCH.parent.mkdir(parents=True, exist_ok=True)
    if LIBRARY_SCRATCH.exists():
        library = _open_library("r+")
        if library.shape != LIBRARY_SHAPE or library.dtype != np.float32:
            raise RuntimeError(
                "existing response-table scratch array has wrong metadata"
            )
    else:
        library = np.lib.format.open_memmap(
            LIBRARY_SCRATCH, mode="w+", dtype=np.float32, shape=LIBRARY_SHAPE
        )
    progress: dict[str, Any]
    if LIBRARY_PROGRESS.exists():
        progress = dict(json.loads(LIBRARY_PROGRESS.read_text()))
    else:
        progress = {"completed": [], "chunk_checksums": {}}
    completed_list: list[str] = list(progress.get("completed", []))
    chunk_checksums: dict[str, str] = dict(progress.get("chunk_checksums", {}))
    completed = set(completed_list)
    started = time.perf_counter()
    for seed_index, seed in enumerate(SEEDS):
        for chunk_index, start, stop in _library_chunks():
            key = f"seed={seed}:intensity={start}:{stop}"
            if key in completed:
                continue
            values = simulate_condition_grid(
                INTENSITY_LEVELS[start:stop],
                TRAINING_RATES_HZ,
                PROBE_CONDUCTANCES_US,
                LIBRARY_K,
                _library_chunk_rng(seed, chunk_index),
            )
            library[seed_index, :, :, start:stop, :] = values
            library.flush()
            checksum = hashlib.sha256(values.tobytes(order="C")).hexdigest()
            completed_list.append(key)
            chunk_checksums[key] = checksum
            progress["completed"] = completed_list
            progress["chunk_checksums"] = chunk_checksums
            progress["shape"] = list(LIBRARY_SHAPE)
            progress["dtype"] = "float32"
            progress["axis_order"] = list(LIBRARY_AXIS_ORDER)
            progress["updated_at_utc"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            LIBRARY_PROGRESS.write_text(json.dumps(progress, indent=2) + "\n")
            print(f"completed {key}", flush=True)
    return {
        "duration_s": time.perf_counter() - started,
        "completed_chunks": len(completed_list),
        "chunk_checksums": chunk_checksums,
    }


def _summary_statistics(library: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(library, dtype=np.float64)
    pooled = values.transpose(1, 2, 3, 0, 4).reshape(
        len(PROBE_CONDUCTANCES_US), len(TRAINING_RATES_HZ), 256, -1
    )
    return {
        "mean": pooled.mean(axis=-1),
        "variance": pooled.var(axis=-1, ddof=1),
        "standard_deviation": pooled.std(axis=-1, ddof=1),
        "zero_fraction": (pooled == 0.0).mean(axis=-1),
        "per_seed_mean": values.mean(axis=-1),
        "per_seed_variance": values.var(axis=-1, ddof=1),
    }


def plot_response_distributions(path: Path) -> dict[str, Any]:
    """Plot recorded low-to-high-rate feature distributions at one condition."""
    library = np.load(LIBRARY_SCRATCH, mmap_mode="r")
    probe_uS = 1.2
    intensity = 255
    rates_hz = (0.25, 3.0, 25.0)
    probe_index = PROBE_CONDUCTANCES_US.index(probe_uS)

    theme.apply()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), constrained_layout=True)
    records: list[dict[str, Any]] = []
    panel_labels = ("A", "B", "C")
    for axis, panel, rate_hz in zip(axes, panel_labels, rates_hz):
        rate_index = TRAINING_RATES_HZ.index(rate_hz)
        values = np.asarray(
            library[:, probe_index, rate_index, intensity, :], dtype=np.float64
        ).reshape(-1)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        zero_fraction = float(np.mean(values == 0.0))
        upper = float(values.max())
        bins = np.linspace(0.0, upper * 1.001, 61)
        axis.hist(
            values,
            bins=bins,
            weights=np.full(values.size, 1.0 / values.size),
            color=theme.INK_BLACK,
            alpha=0.62,
        )
        x = np.linspace(0.0, upper * 1.001, 600)
        bin_width = float(bins[1] - bins[0])
        gaussian = (
            bin_width
            * np.exp(-0.5 * ((x - mean) / sd) ** 2)
            / (sd * math.sqrt(2.0 * math.pi))
        )
        axis.plot(
            x,
            gaussian,
            color=theme.DEEP_RED,
            linestyle="--",
        )
        axis.set_yscale("log")
        axis.set_title(
            f"{panel}  {rate_hz:g} Hz  (E[N]={rate_hz * PRESENTATION_MS / 1000.0:g})\n"
            f"No-spike fraction: {zero_fraction:.3f}"
        )
        axis.set_xlabel("Feature z (mV)")
        axis.spines[["top", "right"]].set_visible(False)
        records.append(
            {
                "rate_hz": rate_hz,
                "expected_spikes": rate_hz * PRESENTATION_MS / 1000.0,
                "sample_count": int(values.size),
                "mean_mV": mean,
                "standard_deviation_mV": sd,
                "zero_fraction": zero_fraction,
                "skewness": float(np.mean(((values - mean) / sd) ** 3)),
            }
        )
    axes[0].set_ylabel("Probability per bin (log scale)")
    savefig_atomic(fig, path, format=path.suffix.lstrip("."), metadata={"Date": None})
    plt.close(fig)
    return {
        "status": "complete",
        "source": str(LIBRARY_SCRATCH.relative_to(REPO)),
        "uses_recorded_samples_only": True,
        "probe_uS": probe_uS,
        "intensity": intensity,
        "records": records,
        "figure_path": str(path.relative_to(REPO)),
        "figure_sha256": sha256_file(path),
        "paid_compute_usd": 0.0,
    }


def refresh_response_distributions() -> None:
    """Regenerate the response-distribution diagnostic from recorded samples."""
    figure_path = FIGURES / "response_distributions.svg"
    outcome = plot_response_distributions(figure_path)
    outcome_path = FIGURES / "response_distributions.json"
    outcome_path.write_text(json.dumps(outcome, indent=2) + "\n")

    provenance_path = FIGURES / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["response_distributions"] = {
        **_git_metadata(),
        "outcome_sha256": sha256_file(outcome_path),
        "figure_sha256": sha256_file(figure_path),
        "paid_compute_usd": 0.0,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    reproducer_path = FIGURES / "reproducer.json"
    reproducer = json.loads(reproducer_path.read_text())
    reproducer["response_distributions"] = {
        "command": "uv run python -c 'from experiments.exp080 import refresh_response_distributions; refresh_response_distributions()'",
        "uses_recorded_arrays_only": True,
        "paid_compute": False,
    }
    reproducer_path.write_text(json.dumps(reproducer, indent=2) + "\n")


def _representative_distributions(library: np.ndarray) -> list[dict[str, Any]]:
    records = []
    probe_index = PROBE_CONDUCTANCES_US.index(1.2)
    for label, rate, intensity in zip(
        ("low", "transitional", "high"),
        DIRECT_VALIDATION_RATES_HZ,
        DIRECT_VALIDATION_INTENSITIES,
    ):
        rate_index = TRAINING_RATES_HZ.index(rate)
        values = np.asarray(
            library[:, probe_index, rate_index, intensity, :], dtype=np.float64
        ).reshape(-1)
        std = values.std(ddof=1)
        skew = (
            0.0 if std == 0.0 else float(np.mean(((values - values.mean()) / std) ** 3))
        )
        records.append(
            {
                "label": label,
                "probe_uS": 1.2,
                "rate_hz": rate,
                "intensity": intensity,
                "count": int(values.size),
                "mean_mV": float(values.mean()),
                "standard_deviation_mV": float(std),
                "zero_fraction": float(np.mean(values == 0.0)),
                "quantiles_mV": {
                    str(q): float(np.quantile(values, q))
                    for q in (0.0, 0.25, 0.5, 0.75, 0.95, 1.0)
                },
                "skewness": skew,
                "distinct_float32_values": int(np.unique(values).size),
            }
        )
    return records


def validate_full_library(library: np.ndarray) -> dict[str, Any]:
    """Run the predeclared full-response-table validation suite."""
    validations: dict[str, Any] = {}
    metadata_ok = bool(
        library.shape == LIBRARY_SHAPE
        and library.dtype == np.float32
        and np.all(np.isfinite(library))
    )
    validations["grid_shape_dtype_finite"] = _validation_record(
        metadata_ok, shape=list(library.shape), dtype=str(library.dtype)
    )
    minimum, maximum = float(library.min()), float(library.max())
    validations["physical_bounds"] = _validation_record(
        bool(minimum >= 0.0 and maximum <= 65.0),
        minimum_mV=minimum,
        maximum_mV=maximum,
    )
    zero_exact = bool(np.all(library[:, :, :, 0, :] == 0.0))
    validations["zero_intensity_rest"] = _validation_record(zero_exact)

    replay_rows = []
    for seed_index, seed in enumerate(SEEDS):
        for chunk_index in REPLAY_CHUNK_INDICES:
            _, start, stop = _library_chunks()[chunk_index]
            replay = simulate_condition_grid(
                INTENSITY_LEVELS[start:stop],
                TRAINING_RATES_HZ,
                PROBE_CONDUCTANCES_US,
                LIBRARY_K,
                _library_chunk_rng(seed, chunk_index),
            )
            expected = np.asarray(library[seed_index, :, :, start:stop, :])
            replay_rows.append(
                {
                    "seed": seed,
                    "intensity_start": start,
                    "intensity_stop": stop,
                    "exact": bool(np.array_equal(replay, expected)),
                }
            )
    validations["deterministic_chunk_replay"] = _validation_record(
        all(row["exact"] for row in replay_rows), chunks=replay_rows
    )
    seed_hashes = [
        hashlib.sha256(np.asarray(library[index]).tobytes()).hexdigest()
        for index in range(len(SEEDS))
    ]
    condition_hashes = [
        hashlib.sha256(np.asarray(library[0, 1, index, 128]).tobytes()).hexdigest()
        for index in range(len(TRAINING_RATES_HZ))
    ]
    streams_unique = len(set(seed_hashes)) == len(seed_hashes) and len(
        set(condition_hashes)
    ) == len(condition_hashes)
    validations["independent_streams"] = _validation_record(
        streams_unique, seed_sha256=seed_hashes, condition_sha256=condition_hashes
    )

    summaries = _summary_statistics(library)
    moment_rows = []
    for probe_index, probe in enumerate(PROBE_CONDUCTANCES_US):
        for rate, intensity in zip(
            DIRECT_VALIDATION_RATES_HZ, DIRECT_VALIDATION_INTENSITIES
        ):
            rate_index = TRAINING_RATES_HZ.index(rate)
            raw = np.asarray(
                library[:, probe_index, rate_index, intensity, :],
                dtype=np.float64,
            ).reshape(-1)
            mean_error = abs(
                float(
                    raw.mean() - summaries["mean"][probe_index, rate_index, intensity]
                )
            )
            variance_error = abs(
                float(
                    raw.var(ddof=1)
                    - summaries["variance"][probe_index, rate_index, intensity]
                )
            )
            moment_rows.append(
                {
                    "probe_uS": probe,
                    "rate_hz": rate,
                    "intensity": intensity,
                    "mean_absolute_error_mV": mean_error,
                    "variance_absolute_error_mV2": variance_error,
                }
            )
    validations["independent_moment_recomputation"] = _validation_record(
        all(
            row["mean_absolute_error_mV"] <= 1e-12
            and row["variance_absolute_error_mV2"] <= 1e-12
            for row in moment_rows
        ),
        conditions=moment_rows,
        tolerance=1e-12,
    )
    pooled_count = len(SEEDS) * LIBRARY_K
    mean = summaries["mean"]
    std = summaries["standard_deviation"]
    se = std / math.sqrt(pooled_count)
    intensity_diff = np.diff(mean, axis=2)
    intensity_tol = MONOTONIC_Z * np.sqrt(se[:, :, 1:] ** 2 + se[:, :, :-1] ** 2)
    rate_diff = np.diff(mean, axis=1)
    rate_tol = MONOTONIC_Z * np.sqrt(se[:, 1:, :] ** 2 + se[:, :-1, :] ** 2)
    intensity_violations = int(np.sum(intensity_diff < -intensity_tol))
    rate_violations = int(np.sum(rate_diff < -rate_tol))
    intensity_violation_rows = []
    for probe_index, rate_index, lower_intensity in np.argwhere(
        intensity_diff < -intensity_tol
    ):
        difference = float(intensity_diff[probe_index, rate_index, lower_intensity])
        tolerance = float(intensity_tol[probe_index, rate_index, lower_intensity])
        intensity_violation_rows.append(
            {
                "probe_uS": PROBE_CONDUCTANCES_US[int(probe_index)],
                "rate_hz": TRAINING_RATES_HZ[int(rate_index)],
                "lower_intensity": int(lower_intensity),
                "higher_intensity": int(lower_intensity) + 1,
                "difference_mV": difference,
                "negative_tolerance_mV": -tolerance,
                "excess_mV": -difference - tolerance,
            }
        )
    intensity_violation_rows.sort(key=lambda row: row["excess_mV"], reverse=True)
    rate_violation_rows = []
    for probe_index, lower_rate_index, intensity in np.argwhere(rate_diff < -rate_tol):
        difference = float(rate_diff[probe_index, lower_rate_index, intensity])
        tolerance = float(rate_tol[probe_index, lower_rate_index, intensity])
        rate_violation_rows.append(
            {
                "probe_uS": PROBE_CONDUCTANCES_US[int(probe_index)],
                "lower_rate_hz": TRAINING_RATES_HZ[int(lower_rate_index)],
                "higher_rate_hz": TRAINING_RATES_HZ[int(lower_rate_index) + 1],
                "intensity": int(intensity),
                "difference_mV": difference,
                "negative_tolerance_mV": -tolerance,
                "excess_mV": -difference - tolerance,
            }
        )
    rate_violation_rows.sort(key=lambda row: row["excess_mV"], reverse=True)
    validations["monotonic_means"] = _validation_record(
        intensity_violations == 0 and rate_violations == 0,
        standard_error_multiplier=MONOTONIC_Z,
        intensity_comparison_count=int(intensity_diff.size),
        rate_comparison_count=int(rate_diff.size),
        intensity_violations=intensity_violations,
        rate_violations=rate_violations,
        intensity_violation_conditions=intensity_violation_rows,
        rate_violation_conditions=rate_violation_rows,
    )

    direct_rows = []
    for probe_index, probe in enumerate(PROBE_CONDUCTANCES_US):
        direct = simulate_condition_grid(
            DIRECT_VALIDATION_INTENSITIES,
            DIRECT_VALIDATION_RATES_HZ,
            (probe,),
            LIBRARY_K,
            _step2_rng(SEED, 700 + probe_index),
        )[0]
        for rate_offset, rate in enumerate(DIRECT_VALIDATION_RATES_HZ):
            library_rate = TRAINING_RATES_HZ.index(rate)
            for intensity_offset, intensity in enumerate(DIRECT_VALIDATION_INTENSITIES):
                observed = np.asarray(
                    library[:, probe_index, library_rate, intensity, :],
                    dtype=np.float64,
                ).reshape(-1)
                fresh = direct[rate_offset, intensity_offset].astype(np.float64)
                mean_delta = abs(float(observed.mean() - fresh.mean()))
                mean_limit = 4.0 * math.sqrt(
                    observed.var(ddof=1) / observed.size
                    + fresh.var(ddof=1) / fresh.size
                )
                variance_delta = abs(float(observed.var(ddof=1) - fresh.var(ddof=1)))
                variance_limit = max(
                    0.25, 0.25 * max(observed.var(ddof=1), fresh.var(ddof=1))
                )
                direct_rows.append(
                    {
                        "probe_uS": probe,
                        "rate_hz": rate,
                        "intensity": intensity,
                        "mean_delta_mV": mean_delta,
                        "mean_limit_mV": mean_limit,
                        "variance_delta_mV2": variance_delta,
                        "variance_limit_mV2": variance_limit,
                        "passed": bool(
                            mean_delta <= mean_limit
                            and variance_delta <= variance_limit
                        ),
                    }
                )
    validations["fresh_direct_simulation"] = _validation_record(
        all(row["passed"] for row in direct_rows), conditions=direct_rows
    )

    subset64 = np.asarray(library[:, :, :, (64, 128, 255), :], dtype=np.float64)
    subset32 = subset64.astype(np.float32).astype(np.float64)
    mean_error = float(np.max(np.abs(subset64.mean(axis=-1) - subset32.mean(axis=-1))))
    variance_error = float(
        np.max(np.abs(subset64.var(axis=-1, ddof=1) - subset32.var(axis=-1, ddof=1)))
    )
    validations["float32_storage"] = _validation_record(
        mean_error <= 1e-6 and variance_error <= 1e-5,
        maximum_mean_error_mV=mean_error,
        maximum_variance_error_mV2=variance_error,
        mean_tolerance_mV=1e-6,
        variance_tolerance_mV2=1e-5,
    )
    distributions = _representative_distributions(library)
    low = distributions[0]
    validations["low_rate_empirical_structure"] = _validation_record(
        low["zero_fraction"] > 0.0
        and low["skewness"] > 0.0
        and low["distinct_float32_values"] > 1,
        representative=distributions,
    )
    return {
        "validations": validations,
        "summaries": summaries,
        "distributions": distributions,
    }


def _normalised_difference(
    first: np.ndarray, second: np.ndarray, absolute_floor: float, relative: float
) -> np.ndarray:
    denominator = np.maximum(
        absolute_floor, relative * np.maximum(np.abs(first), np.abs(second))
    )
    return np.abs(first - second) / denominator


def run_bootstrap_stability() -> dict[str, Any]:
    """Run the locked repeated paired-resampling K-stability diagnostic."""
    protocol_path = FIGURES / "step2_bootstrap_stability_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["candidate_K"] != list(BOOTSTRAP_CANDIDATE_K):
        raise RuntimeError("bootstrap candidates differ from the locked protocol")
    if protocol["repetitions"] != BOOTSTRAP_REPETITIONS:
        raise RuntimeError("bootstrap repetitions differ from the locked protocol")
    if sha256_file(LIBRARY_SCRATCH) != protocol["source_library_sha256"]:
        raise RuntimeError(
            "source empirical response table does not match the locked SHA-256"
        )

    library = _open_library("r")
    rate_indices = [TRAINING_RATES_HZ.index(rate) for rate in PILOT_RATES_HZ]
    intensity_indices = [
        int(np.where(INTENSITY_LEVELS == intensity)[0][0])
        for intensity in PILOT_INTENSITIES
        if intensity > 0
    ]
    source = np.asarray(
        library[:, :, rate_indices, :, :][:, :, :, intensity_indices, :]
    )
    trajectory: list[dict[str, Any]] = []
    started = time.perf_counter()
    for candidate in BOOTSTRAP_CANDIDATE_K:
        rows = []
        for repetition in range(BOOTSTRAP_REPETITIONS):
            samples = []
            for block in (0, 1):
                rng = np.random.default_rng(
                    np.random.SeedSequence([77, 2, 2048, repetition, candidate, block])
                )
                indices = rng.integers(
                    0, LIBRARY_K, size=(*source.shape[:-1], candidate)
                )
                samples.append(np.take_along_axis(source, indices, axis=-1))
            first, second = samples
            mean_a = first.mean(axis=-1)
            mean_b = second.mean(axis=-1)
            variance_a = first.var(axis=-1, ddof=1)
            variance_b = second.var(axis=-1, ddof=1)
            mean_error = _normalised_difference(
                mean_a,
                mean_b,
                PILOT_MEAN_ABS_TOL_MV,
                PILOT_MEAN_REL_TOL,
            )
            variance_error = _normalised_difference(
                variance_a,
                variance_b,
                PILOT_VARIANCE_ABS_TOL_MV2,
                PILOT_VARIANCE_REL_TOL,
            )
            metrics = {
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
            rows.append(metrics)
        metric_names = [name for name in rows[0] if name != "passed"]
        trajectory.append(
            {
                "K": candidate,
                "pass_count": sum(row["passed"] for row in rows),
                "pass_frequency": float(np.mean([row["passed"] for row in rows])),
                "metric_distributions": {
                    name: {
                        "median": float(np.median([row[name] for row in rows])),
                        "p05": float(np.quantile([row[name] for row in rows], 0.05)),
                        "p95": float(np.quantile([row[name] for row in rows], 0.95)),
                        "minimum": float(min(row[name] for row in rows)),
                        "maximum": float(max(row[name] for row in rows)),
                    }
                    for name in metric_names
                },
            }
        )
        print(
            f"bootstrap K={candidate}: pass frequency "
            f"{trajectory[-1]['pass_frequency']:.3f}",
            flush=True,
        )
    final_frequency = float(trajectory[-1]["pass_frequency"])
    return {
        "status": "complete",
        "classification": "post-hoc diagnostic",
        "candidate_K": list(BOOTSTRAP_CANDIDATE_K),
        "repetitions": BOOTSTRAP_REPETITIONS,
        "condition_seed_count": int(np.prod(source.shape[:-1])),
        "trajectory": trajectory,
        "K2048_pass_frequency": final_frequency,
        "K2048_typically_stable": bool(final_frequency >= BOOTSTRAP_PASS_FREQUENCY),
        "decision_threshold": BOOTSTRAP_PASS_FREQUENCY,
        "run_independent_K4096": bool(final_frequency < BOOTSTRAP_PASS_FREQUENCY),
        "duration_s": round(time.perf_counter() - started, 1),
        "source_library_sha256": protocol["source_library_sha256"],
        "protocol_sha256": sha256_file(protocol_path),
        "paid_compute_usd": 0.0,
    }


def record_bootstrap_stability(outcome: dict[str, Any]) -> None:
    """Write the locked post-hoc stability diagnostic outcome."""
    path = FIGURES / "step2_bootstrap_stability_outcome.json"
    path.write_text(json.dumps(outcome, indent=2) + "\n")
    numbers_path = FIGURES / "numbers.json"
    numbers = json.loads(numbers_path.read_text())
    numbers["step2"]["bootstrap_stability"] = outcome
    numbers_path.write_text(json.dumps(numbers, indent=2) + "\n")
    manifest_path = FIGURES / "step2_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["bootstrap_stability_protocol_sha256"] = outcome["protocol_sha256"]
    manifest["bootstrap_stability_outcome_sha256"] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


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
            "variance_p95_normalised_error": float(np.quantile(variance_error, 0.95)),
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
    positions = np.arange(len(candidate))
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.66), constrained_layout=True)
    series = (
        ("mean_p95_normalised_error", "95th percentile", "o", "-"),
        ("mean_maximum_normalised_error", "Maximum", "s", "--"),
    )
    for key, label, marker, linestyle in series:
        axes[0].plot(
            positions,
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
            positions,
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
            xticks=positions,
            xticklabels=[str(value) for value in candidate],
            xlabel="Draws per comparison block, K",
            ylabel="Normalized discrepancy",
            title=title,
        )
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    savefig_atomic(fig, path, dpi=240, facecolor="white")
    plt.close(fig)


def record_step2_pilot(pilot: dict[str, Any], duration_s: float) -> None:
    """Freeze the pilot outcome without erasing the completed Step 1 evidence."""
    outcome_path = FIGURES / PILOT_OUTCOME_NAME
    outcome_path.write_text(json.dumps(pilot, indent=2) + "\n")
    plot_step2_pilot(pilot, FIGURES / PILOT_FIGURE_NAME)
    numbers_path = FIGURES / "numbers.json"
    numbers = json.loads(numbers_path.read_text())
    numbers["step"] = 2
    numbers["status"] = (
        "extended_pilot_converged_library_pending"
        if pilot["passed"]
        else "killed_at_extended_convergence_gate"
    )
    numbers["scope"] = (
        "Step 1 remains complete; the authorized Step 2 pilot extension "
        "converged but no final empirical response table, decoder, held-out test, threshold, "
        "or PING run was generated"
    )
    original_pilot = json.loads((FIGURES / "step2_pilot_outcome.json").read_text())
    combined_pilot = {
        "candidate_K": original_pilot["candidate_K"] + pilot["candidate_K"],
        "hard_maximum_K": pilot["hard_maximum_K"],
        "evaluation_condition_count": pilot["evaluation_condition_count"],
        "trajectory": original_pilot["trajectory"] + pilot["trajectory"],
        "selected_K": pilot["selected_K"],
        "passed": pilot["passed"],
        "zero_intensity_exact": (
            original_pilot["zero_intensity_exact"] and pilot["zero_intensity_exact"]
        ),
    }
    plot_step2_pilot(combined_pilot, FIGURES / "response_library.png")
    numbers["step2"] = {
        "status": "extension_converged" if pilot["passed"] else "extension_failed",
        "original_pilot": original_pilot,
        "extension_pilot": pilot,
        "combined_pilot": combined_pilot,
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
            "step2_status": (
                "extended_pilot_converged"
                if pilot["passed"]
                else "extended_pilot_failed"
            ),
            "step2_selected_K": pilot["selected_K"],
            "step2_final_library_generated": False,
            "later_steps": "not run",
        }
    )
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    reproducer = {
        "command": (
            "EXP080_THROUGH_STEP=2 EXP080_STEP2_PILOT_ONLY=1 "
            "uv run python experiments/exp080.py"
        ),
        "paid_compute": False,
        "expected_outcome": "evaluate the authorized K=1024 and K=2048 extension",
        "expected_outputs": [
            f"artifacts/data/exp080/{PILOT_OUTCOME_NAME}",
            f"artifacts/data/exp080/{PILOT_FIGURE_NAME}",
            "artifacts/data/exp080/numbers.json",
        ],
    }
    (FIGURES / "reproducer.json").write_text(json.dumps(reproducer, indent=2) + "\n")
    (FIGURES / "provenance.json").write_text(
        json.dumps(_git_metadata(), indent=2) + "\n"
    )
    manifest = {
        "status": (
            "pilot_converged_library_pending"
            if pilot["passed"]
            else "extended_pilot_failed_no_library"
        ),
        "selected_K": pilot["selected_K"],
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
        "response_library_figure_sha256": sha256_file(FIGURES / "response_library.png"),
        "locked_protocol_sha256": sha256_file(
            FIGURES / "step2_pilot_extension_protocol.json"
        ),
        "regeneration_command": reproducer["command"],
    }
    (FIGURES / "step2_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def plot_full_library(
    summaries: dict[str, np.ndarray],
    distributions: list[dict[str, Any]],
    library: np.ndarray,
    path: Path,
) -> None:
    """Render every rate-intensity observation in the complete Step 2 evidence."""
    mean = summaries["mean"]
    standard_deviation = summaries["standard_deviation"]
    expected_count = (
        np.asarray(TRAINING_RATES_HZ)[:, None]
        * (INTENSITY_LEVELS.astype(np.float64)[None, :] / 255.0)
        * (PRESENTATION_MS / 1000.0)
    )
    count_flat = expected_count.reshape(-1)
    colors = (theme.INK_BLACK, theme.DEEP_RED, theme.ELECTRIC_CYAN)
    fig, (ax_mean, ax_std) = plt.subplots(
        1, 2, figsize=(8.2, 3.25), constrained_layout=True
    )
    for probe_index, (probe, color) in enumerate(zip(PROBE_CONDUCTANCES_US, colors)):
        for ax, values in (
            (ax_mean, mean[probe_index]),
            (ax_std, standard_deviation[probe_index]),
        ):
            ax.scatter(
                count_flat,
                values.reshape(-1),
                color=color,
                s=5,
                alpha=0.18,
                edgecolors="none",
                label=f"{probe:g} μS",
            )
    ax_mean.set(title="A  Signal grows with input drive", ylabel="Mean feature z (mV)")
    ax_std.set(title="B  Trial-to-trial variability", ylabel="Feature SD (mV)")
    for ax in (ax_mean, ax_std):
        ax.set_xscale("symlog", linthresh=0.02)
        ax.set_xlim(-0.002, 5.5)
        ax.set_xticks((0, 0.01, 0.1, 1, 5), labels=("0", "0.01", "0.1", "1", "5"))
        ax.set_xlabel("Expected input spikes")
    ax_mean.legend(frameon=False, fontsize=7, title="Probe conductance")
    ax_mean.legend(frameon=False, fontsize=7, loc="upper left")
    for ax in (ax_mean, ax_std):
        ax.grid(alpha=0.14)
        ax.spines[["top", "right"]].set_visible(False)
    savefig_atomic(fig, path, dpi=240, facecolor="white")
    plt.close(fig)


def refresh_full_library_figure() -> None:
    """Replot the diagnostic Step 2 moments without rerunning any simulation."""
    library = _open_library("r")
    diagnostic = library[..., :DIAGNOSTIC_K]
    summaries = _summary_statistics(diagnostic)
    distributions = _representative_distributions(diagnostic)
    summary_path = FIGURES / "response_table_diagnostic_summary.npz"
    np.savez_compressed(
        summary_path,
        mean=summaries["mean"],
        standard_deviation=summaries["standard_deviation"],
        zero_fraction=summaries["zero_fraction"],
    )
    plot_full_library(
        summaries,
        distributions,
        diagnostic,
        FIGURES / "response_library.png",
    )
    manifest_path = FIGURES / "step2_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["response_library_figure_sha256"] = sha256_file(
        FIGURES / "response_library.png"
    )
    manifest.pop("visual_k_convergence_figure_sha256", None)
    manifest.pop("visual_k_selection_sha256", None)
    manifest["diagnostic_draws_per_condition_per_seed"] = DIAGNOSTIC_K
    manifest["diagnostic_summary_sha256"] = sha256_file(summary_path)
    manifest["role"] = "mean/SD plotting and consistency validation only"
    manifest["ann_inputs"] = (
        "fresh direct Poisson-synapse-membrane simulation per presentation"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def record_full_library(
    generation: dict[str, Any], validation: dict[str, Any]
) -> dict[str, Any]:
    """Publish authenticated summaries while retaining the raw memmap in scratch."""
    library = _open_library("r")
    summaries = validation["summaries"]
    distributions = validation["distributions"]
    summary_path = FIGURES / "response_library_summary.npz"
    np.savez_compressed(summary_path, **summaries)
    diagnostic = library[..., :DIAGNOSTIC_K]
    plot_full_library(
        _summary_statistics(diagnostic),
        _representative_distributions(diagnostic),
        diagnostic,
        FIGURES / "response_library.png",
    )
    library_sha256 = sha256_file(LIBRARY_SCRATCH)
    progress = json.loads(LIBRARY_PROGRESS.read_text())
    payload_bytes = int(np.prod(LIBRARY_SHAPE) * np.dtype(np.float32).itemsize)
    manifest = {
        "status": "complete",
        "selected_K": LIBRARY_K,
        "diagnostic_draws_per_condition_per_seed": DIAGNOSTIC_K,
        "role": "mean/SD plotting and consistency validation only",
        "ann_inputs": "fresh direct Poisson-synapse-membrane simulation per presentation",
        "convergence_rule_changed": False,
        "library_generated": True,
        "library_shape": list(LIBRARY_SHAPE),
        "library_value_count": int(np.prod(LIBRARY_SHAPE)),
        "library_payload_bytes": payload_bytes,
        "library_file_bytes": LIBRARY_SCRATCH.stat().st_size,
        "dtype": "float32",
        "ordered_axes": list(LIBRARY_AXIS_ORDER),
        "condition_arrays": {
            "seeds": list(SEEDS),
            "probe_conductances_uS": list(PROBE_CONDUCTANCES_US),
            "rates_hz": list(TRAINING_RATES_HZ),
            "intensities": INTENSITY_LEVELS.astype(int).tolist(),
        },
        "seed_recipe": (
            "numpy.random.SeedSequence([registered_seed, 77, 2, 1, "
            "intensity_chunk_index]); fixed probe-rate-intensity-draw indices"
        ),
        "chunking": {
            "intensities_per_chunk": LIBRARY_CHUNK_INTENSITIES,
            "chunks_per_seed": len(_library_chunks()),
            "total_chunks": len(SEEDS) * len(_library_chunks()),
        },
        "library_storage": str(LIBRARY_SCRATCH.relative_to(REPO)),
        "library_sha256": library_sha256,
        "chunk_sha256": progress["chunk_checksums"],
        "summary_storage": str(summary_path.relative_to(REPO)),
        "summary_sha256": sha256_file(summary_path),
        "pilot_outcome_sha256": sha256_file(
            FIGURES / "step2_pilot_extension_outcome.json"
        ),
        "locked_protocol_sha256": sha256_file(
            FIGURES / "step2_pilot_extension_protocol.json"
        ),
        "generation_command": (
            "EXP080_THROUGH_STEP=2 EXP080_STEP2_FULL=1 "
            "uv run python experiments/exp080.py"
        ),
        "validation_command": (
            "EXP080_THROUGH_STEP=2 EXP080_STEP2_VALIDATE_ONLY=1 "
            "uv run python experiments/exp080.py"
        ),
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
    }
    (FIGURES / "step2_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    validation_records = validation["validations"]
    numbers = json.loads((FIGURES / "numbers.json").read_text())
    numbers.update(
        {
            "step": 2,
            "status": "step2_library_complete",
            "scope": (
                "Empirical response table only; no decoder, held-out test, "
                "threshold selection, or PING run"
            ),
        }
    )
    original_pilot = json.loads((FIGURES / "step2_pilot_outcome.json").read_text())
    extension_pilot = json.loads(
        (FIGURES / "step2_pilot_extension_outcome.json").read_text()
    )
    combined_pilot = {
        "candidate_K": original_pilot["candidate_K"] + extension_pilot["candidate_K"],
        "hard_maximum_K": extension_pilot["hard_maximum_K"],
        "evaluation_condition_count": extension_pilot["evaluation_condition_count"],
        "trajectory": original_pilot["trajectory"] + extension_pilot["trajectory"],
        "selected_K": extension_pilot["selected_K"],
        "passed": extension_pilot["passed"],
        "zero_intensity_exact": bool(
            original_pilot["zero_intensity_exact"]
            and extension_pilot["zero_intensity_exact"]
        ),
    }
    if "step2" in numbers:
        step2 = numbers["step2"]
    else:
        committed_numbers = json.loads(
            subprocess.check_output(
                ["git", "show", "HEAD:artifacts/data/exp080/numbers.json"],
                cwd=REPO,
                text=True,
            )
        )
        step2 = committed_numbers.get(
            "step2",
            {
                "original_pilot": original_pilot,
                "extension_pilot": extension_pilot,
                "combined_pilot": combined_pilot,
            },
        )
        numbers["step2"] = step2
    step2.update(
        {
            "status": "library_complete",
            "final_library_generated": True,
            "selected_K": LIBRARY_K,
            "library_shape": list(LIBRARY_SHAPE),
            "dtype": "float32",
            "payload_bytes": payload_bytes,
            "file_bytes": LIBRARY_SCRATCH.stat().st_size,
            "library_sha256": library_sha256,
            "generation_duration_s": round(generation["duration_s"], 1),
            "validation_count": len(validation_records),
            "all_validations_passed": all(
                row["ok"] for row in validation_records.values()
            ),
            "validations": validation_records,
            "representative_distributions": distributions,
            "later_steps_run": False,
            "paid_compute_usd": 0.0,
        }
    )
    (FIGURES / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")
    protocol = json.loads((FIGURES / "protocol.json").read_text())
    protocol.update(
        {
            "attempted_through_step": 2,
            "step2_status": "library_complete",
            "step2_selected_K": LIBRARY_K,
            "step2_final_library_generated": True,
            "step2_shape": list(LIBRARY_SHAPE),
            "step2_dtype": "float32",
            "later_steps": "not run",
        }
    )
    (FIGURES / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    reproducer = {
        "command": manifest["generation_command"],
        "validation_command": manifest["validation_command"],
        "paid_compute": False,
        "expected_library_sha256": library_sha256,
        "expected_outputs": [
            "temp/exp080/response_library.float32.npy",
            "artifacts/data/exp080/response_library_summary.npz",
            "artifacts/data/exp080/step2_manifest.json",
            "artifacts/data/exp080/response_library.png",
        ],
    }
    (FIGURES / "reproducer.json").write_text(json.dumps(reproducer, indent=2) + "\n")
    provenance = _git_metadata()
    provenance.update(
        {
            "library_sha256": library_sha256,
            "generation_duration_s": round(generation["duration_s"], 1),
            "paid_compute_usd": 0.0,
        }
    )
    (FIGURES / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return manifest


def _generation_timing_from_scratch() -> dict[str, Any]:
    """Recover the completed local generation interval from filesystem metadata."""
    birth_epoch = int(
        subprocess.check_output(
            ["stat", "--printf=%W", str(LIBRARY_SCRATCH)], text=True
        ).strip()
    )
    modified_epoch = LIBRARY_SCRATCH.stat().st_mtime
    if birth_epoch <= 0 or modified_epoch < birth_epoch:
        raise RuntimeError("cannot recover full-response-table generation timing")
    return {
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(birth_epoch)),
        "completed_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(modified_epoch)
        ),
        "duration_s": round(modified_epoch - birth_epoch, 1),
        "method": "filesystem birth time to final raw-response-table modification time",
    }


def record_full_library_failure(
    generation: dict[str, Any], validation: dict[str, Any]
) -> dict[str, Any]:
    """Record a killed full-response-table attempt without weakening a failed check."""
    manifest = record_full_library(generation, validation)
    failed = [name for name, row in validation["validations"].items() if not row["ok"]]
    timing = _generation_timing_from_scratch()
    failure = {
        "status": "killed_at_required_validation",
        "failed_validations": failed,
        "validation_rule_changed": False,
        "generation_timing": timing,
        "library_sha256": manifest["library_sha256"],
        "validations": validation["validations"],
        "representative_distributions": validation["distributions"],
        "paid_compute_usd": 0.0,
        "later_steps_run": False,
    }
    failure_path = FIGURES / "step2_library_validation_failure.json"
    failure_path.write_text(json.dumps(failure, indent=2) + "\n")
    manifest.update(
        {
            "status": "validation_failed",
            "validation_passed": False,
            "failed_validations": failed,
            "failure_record": str(failure_path.relative_to(REPO)),
            "failure_record_sha256": sha256_file(failure_path),
            "generation_timing": timing,
        }
    )
    (FIGURES / "step2_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    numbers = json.loads((FIGURES / "numbers.json").read_text())
    numbers["status"] = "killed_at_full_library_validation"
    numbers["scope"] = (
        "Full empirical response table generated, but Step 2 killed at its required "
        "monotonicity validation; no later stage ran"
    )
    numbers["step2"].update(
        {
            "status": "validation_failed",
            "generation_duration_s": timing["duration_s"],
            "generation_timing": timing,
            "all_validations_passed": False,
            "failed_validations": failed,
        }
    )
    (FIGURES / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")
    protocol = json.loads((FIGURES / "protocol.json").read_text())
    protocol["step2_status"] = "full_library_validation_failed"
    protocol["later_steps"] = "not run"
    (FIGURES / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    reproducer = json.loads((FIGURES / "reproducer.json").read_text())
    reproducer["expected_outcome"] = (
        "reproduce the authenticated empirical response table and the required monotonicity failure"
    )
    reproducer["failure_record"] = str(failure_path.relative_to(REPO))
    (FIGURES / "reproducer.json").write_text(json.dumps(reproducer, indent=2) + "\n")
    provenance = json.loads((FIGURES / "provenance.json").read_text())
    provenance["generation_timing"] = timing
    provenance["generation_duration_s"] = timing["duration_s"]
    provenance["validation_status"] = "failed"
    (FIGURES / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return failure


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
        "equations": "writings/exp080.typ, Equations 2-7",
        "update_order": "AMPA decay, spike kick, exponential-Euler membrane update",
        "feature": "mean_t(v(t) - E_L)",
        "later_steps": "not run",
    }
    provenance = _git_metadata()
    reproducer = {
        "command": "EXP080_THROUGH_STEP=1 uv run python experiments/exp080.py",
        "paid_compute": False,
        "expected_outputs": [
            "artifacts/data/exp080/numbers.json",
            "artifacts/data/exp080/probe_dynamics.svg",
            "artifacts/data/exp080/protocol.json",
            "artifacts/data/exp080/provenance.json",
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
        f"exp080 Step 1 complete: {len(validations)}/{len(validations)} validations passed"
    )
    print(f"Artifacts: {FIGURES.relative_to(REPO)}")


def step_2() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    if os.environ.get("EXP080_STEP2_PILOT_ONLY") == "1":
        started = time.perf_counter()
        pilot = run_step2_pilot()
        record_step2_pilot(pilot, time.perf_counter() - started)
        if not pilot["passed"]:
            raise RuntimeError(
                "Step 2 draw-count pilot did not pass by the locked maximum K; "
                "the final empirical response table was not generated"
            )
        print(f"exp080 Step 2 pilot selected K={pilot['selected_K']}")
        return
    selected = json.loads((FIGURES / "step2_pilot_extension_outcome.json").read_text())[
        "selected_K"
    ]
    if selected != LIBRARY_K:
        raise RuntimeError(f"registered selected K is {selected}, expected {LIBRARY_K}")
    validate_only = os.environ.get("EXP080_STEP2_VALIDATE_ONLY") == "1"
    if not validate_only and os.environ.get("EXP080_STEP2_FULL") != "1":
        raise RuntimeError(
            "Step 2 full-response-table generation requires EXP080_STEP2_FULL=1"
        )
    if validate_only:
        if not LIBRARY_SCRATCH.exists():
            raise RuntimeError("response-table scratch array does not exist")
        generation = {"duration_s": 0.0, "completed_chunks": None}
    else:
        generation = generate_full_library()
    library = _open_library("r")
    validation = validate_full_library(library)
    failed = [name for name, row in validation["validations"].items() if not row["ok"]]
    if failed:
        record_full_library_failure(generation, validation)
        raise RuntimeError(
            f"Step 2 full-response-table validation failed: {', '.join(failed)}"
        )
    manifest = record_full_library(generation, validation)
    print(
        f"exp080 Step 2 complete: {manifest['library_shape']} {manifest['dtype']} "
        f"sha256={manifest['library_sha256']}"
    )


def verify_authenticated_library() -> np.memmap:
    """Open the locked Step 2 response table only after authenticating its bytes."""
    if not LIBRARY_SCRATCH.exists():
        raise RuntimeError(f"authenticated response table is absent: {LIBRARY_SCRATCH}")
    observed = sha256_file(LIBRARY_SCRATCH)
    if observed != LIBRARY_SHA256:
        raise RuntimeError(f"response-table SHA-256 mismatch: {observed}")
    library = np.load(LIBRARY_SCRATCH, mmap_mode="r")
    if library.shape != LIBRARY_SHAPE or library.dtype != np.float32:
        raise RuntimeError(
            f"response-table contract mismatch: shape={library.shape}, dtype={library.dtype}"
        )
    return library


def linear_operating_point(
    lambda_hz: np.ndarray | float, probe_uS: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    """Equations 11--12 in the registered uS, nF, mV, ms units."""
    lam = np.asarray(lambda_hz, dtype=np.float64)
    probe = np.asarray(probe_uS, dtype=np.float64)
    mean_g = (lam / 1000.0) * probe * PARAMETERS["tau_ampa_ms"]
    mean_v = (
        PARAMETERS["g_L_uS"] * PARAMETERS["E_L_mV"] + mean_g * PARAMETERS["E_e_mV"]
    ) / (PARAMETERS["g_L_uS"] + mean_g)
    return mean_g, mean_v


def complete_transfer(
    frequency_hz: np.ndarray,
    lambda_hz: np.ndarray | float,
    probe_uS: np.ndarray | float,
) -> np.ndarray:
    """Equations 13--15, with angular frequency expressed in rad/ms."""
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    omega = 2.0 * np.pi * frequency / 1000.0
    argument = omega * PRESENTATION_MS / 2.0
    averaging = np.exp(-1j * argument) * np.sinc(argument / np.pi)
    return averaging * synapse_membrane_transfer(frequency_hz, lambda_hz, probe_uS)


def synapse_membrane_transfer(
    frequency_hz: np.ndarray,
    lambda_hz: np.ndarray | float,
    probe_uS: np.ndarray | float,
) -> np.ndarray:
    """Equation 13 before applying the finite presentation-window average."""
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    omega = 2.0 * np.pi * frequency / 1000.0
    mean_g, mean_v = linear_operating_point(lambda_hz, probe_uS)
    synapse = np.asarray(probe_uS) / (1j * omega + 1.0 / PARAMETERS["tau_ampa_ms"])
    membrane = (PARAMETERS["E_e_mV"] - mean_v) / (
        1j * omega * PARAMETERS["C_m_nF"] + PARAMETERS["g_L_uS"] + mean_g
    )
    return synapse * membrane


def predicted_linear_variance(
    lambda_hz: np.ndarray,
    probe_uS: np.ndarray,
    *,
    bounds_hz: tuple[float, float] = FREQUENCY_BOUNDS_HZ,
    grid_points: int = FREQUENCY_GRID_POINTS[-1],
) -> np.ndarray:
    """Numerically integrate Equation 18 for aligned operating points."""
    lam = np.asarray(lambda_hz, dtype=np.float64).reshape(-1)
    probe = np.asarray(probe_uS, dtype=np.float64).reshape(-1)
    if lam.shape != probe.shape:
        raise ValueError("lambda_hz and probe_uS must have matching shapes")
    result = np.zeros_like(lam)
    positive = lam > 0.0
    frequencies = np.geomspace(bounds_hz[0], bounds_hz[1], grid_points)
    # Equation 18 is even.  Converting d(angular frequency in rad/ms) to
    # frequency in Hz contributes 2*pi/1000, leaving the factor 2/1000 below.
    for start in range(0, int(np.count_nonzero(positive)), 128):
        indices = np.flatnonzero(positive)[start : start + 128]
        transfer = complete_transfer(
            frequencies[None, :], lam[indices, None], probe[indices, None]
        )
        integrand = np.abs(transfer) ** 2 * (lam[indices, None] / 1000.0)
        integral = np.trapezoid(integrand, frequencies, axis=1)
        zero_transfer = (
            np.abs(
                complete_transfer(
                    np.asarray([[0.0]]), lam[indices, None], probe[indices, None]
                )[:, 0]
            )
            ** 2
        )
        low_tail = zero_transfer * (lam[indices] / 1000.0) * bounds_hz[0]
        result[indices] = (2.0 / 1000.0) * (integral + low_tail)
    return result


def numerical_sinusoidal_gain(
    rate_hz: float, probe_uS: float, frequency_hz: float
) -> float:
    """Measure the local deterministic gain of the registered numerical probe."""
    burn_steps = int(round(2000.0 / DT_MS))
    duration_ms = max(2000.0, 20.0 * 1000.0 / frequency_hz)
    measure_steps = int(round(duration_ms / DT_MS))
    total_steps = burn_steps + measure_steps
    time_ms = np.arange(total_steps, dtype=np.float64) * DT_MS
    phase = 2.0 * np.pi * frequency_hz * time_ms / 1000.0
    rate_per_ms = (rate_hz / 1000.0) * (1.0 + GAIN_MODULATION_FRACTION * np.sin(phase))
    g = rate_hz / 1000.0 * probe_uS * PARAMETERS["tau_ampa_ms"]
    _, v = linear_operating_point(rate_hz, probe_uS)
    v_value = float(v)
    decay = math.exp(-DT_MS / PARAMETERS["tau_ampa_ms"])
    measured = np.empty(measure_steps, dtype=np.float64)
    for index in range(total_steps):
        # Exact constant-input update over dt for dg/dt=-g/tau+w*lambda(t).
        g = g * decay + probe_uS * rate_per_ms[index] * PARAMETERS["tau_ampa_ms"] * (
            1.0 - decay
        )
        total_g = PARAMETERS["g_L_uS"] + g
        v_inf = (
            PARAMETERS["g_L_uS"] * PARAMETERS["E_L_mV"] + g * PARAMETERS["E_e_mV"]
        ) / total_g
        v_value = v_inf + (v_value - v_inf) * math.exp(
            -DT_MS * total_g / PARAMETERS["C_m_nF"]
        )
        if index >= burn_steps:
            measured[index - burn_steps] = v_value
    fit_phase = phase[burn_steps:]
    design = np.column_stack(
        [np.ones(measure_steps), np.sin(fit_phase), np.cos(fit_phase)]
    )
    coefficients = np.linalg.lstsq(design, measured, rcond=None)[0]
    output_amplitude = float(np.hypot(coefficients[1], coefficients[2]))
    input_amplitude_hz = rate_hz * GAIN_MODULATION_FRACTION
    return output_amplitude / input_amplitude_hz


def _drive_regime(expected_spikes: float) -> str:
    if expected_spikes < 0.1:
        return "low"
    if expected_spikes < 1.0:
        return "transitional"
    return "high"


def calculate_step3(library: np.ndarray) -> dict[str, Any]:
    """Calculate the complete registered Step 3 diagnostic."""
    probes, rates, levels = np.meshgrid(
        np.asarray(PROBE_CONDUCTANCES_US),
        np.asarray(TRAINING_RATES_HZ),
        np.arange(256),
        indexing="ij",
    )
    intensities = levels / 255.0
    lambdas = rates * intensities
    flat_lambda = lambdas.reshape(-1)
    flat_probe = probes.reshape(-1)
    predictions: dict[int, np.ndarray] = {}
    for points in FREQUENCY_GRID_POINTS:
        predictions[points] = predicted_linear_variance(
            flat_lambda, flat_probe, grid_points=points
        ).reshape(lambdas.shape)
    primary = predictions[FREQUENCY_GRID_POINTS[-1]]
    previous = predictions[FREQUENCY_GRID_POINTS[-2]]
    denominator = np.maximum(np.abs(primary), np.finfo(float).tiny)
    refinement_relative = np.where(
        primary > 0, np.abs(primary - previous) / denominator, 0
    )
    wide = predicted_linear_variance(
        flat_lambda,
        flat_probe,
        bounds_hz=FREQUENCY_WIDE_BOUNDS_HZ,
        grid_points=FREQUENCY_GRID_POINTS[-1],
    ).reshape(lambdas.shape)
    bound_relative = np.where(primary > 0, np.abs(primary - wide) / denominator, 0)
    empirical_by_seed = np.var(np.asarray(library, dtype=np.float64), axis=-1, ddof=1)
    empirical = np.mean(empirical_by_seed, axis=0)
    ratio = np.divide(
        primary, empirical, out=np.full_like(primary, np.nan), where=empirical > 0
    )

    gain_checks: list[dict[str, Any]] = []
    for probe in PROBE_CONDUCTANCES_US:
        for label, rate in zip(("low", "middle", "high"), GAIN_OPERATING_RATES_HZ):
            for frequency in GAIN_FREQUENCIES_HZ:
                analytical = float(
                    np.abs(complete_transfer(np.asarray([frequency]), rate, probe)[0])
                )
                # The sinusoidal check is on membrane gain before finite-window
                # averaging, so divide H by the averaging response.
                argument = np.pi * frequency * PRESENTATION_MS / 1000.0
                averaging = abs(float(np.sinc(argument / np.pi)))
                analytical_unaveraged = (
                    analytical / averaging
                    if averaging > 1e-12
                    else float(
                        np.abs(
                            complete_transfer(np.asarray([frequency]), rate, probe)[0]
                        )
                    )
                )
                if averaging <= 1e-12:
                    mean_g, mean_v = linear_operating_point(rate, probe)
                    omega = 2.0 * np.pi * frequency / 1000.0
                    analytical_unaveraged = float(
                        abs(
                            probe
                            / (1j * omega + 1 / PARAMETERS["tau_ampa_ms"])
                            * (PARAMETERS["E_e_mV"] - mean_v)
                            / (
                                1j * omega * PARAMETERS["C_m_nF"]
                                + PARAMETERS["g_L_uS"]
                                + mean_g
                            )
                        )
                    )
                # Analytical input is spikes/ms; express gain per spikes/s.
                analytical_per_hz = analytical_unaveraged / 1000.0
                numerical = numerical_sinusoidal_gain(rate, probe, frequency)
                relative_error = abs(numerical - analytical_per_hz) / analytical_per_hz
                gain_checks.append(
                    {
                        "drive": label,
                        "rate_hz": rate,
                        "probe_uS": probe,
                        "frequency_hz": frequency,
                        "analytical_gain_mV_per_hz": analytical_per_hz,
                        "numerical_gain_mV_per_hz": numerical,
                        "relative_error": relative_error,
                        "passed": relative_error <= GAIN_REL_TOL,
                    }
                )

    summaries: list[dict[str, Any]] = []
    expected = lambdas * PRESENTATION_MS / 1000.0
    for probe_index, probe in enumerate(PROBE_CONDUCTANCES_US):
        for regime in ("low", "transitional", "high"):
            mask = np.vectorize(_drive_regime)(expected[probe_index]) == regime
            values = ratio[probe_index][mask]
            values = values[np.isfinite(values) & (values > 0)]
            summaries.append(
                {
                    "probe_uS": probe,
                    "drive_regime": regime,
                    "condition_count": int(values.size),
                    "median_predicted_empirical_ratio": float(np.median(values)),
                    "median_absolute_log2_ratio": float(
                        np.median(np.abs(np.log2(values)))
                    ),
                    "fraction_in_agreement_band": float(
                        np.mean((values >= 0.5) & (values <= 2.0))
                    ),
                }
            )
    return {
        "stationary_mean_conductance_uS": linear_operating_point(lambdas, probes)[0],
        "stationary_mean_voltage_mV": linear_operating_point(lambdas, probes)[1],
        "predicted_variance_mV2": primary,
        "empirical_variance_by_seed_mV2": empirical_by_seed,
        "empirical_variance_mV2": empirical,
        "ratio": ratio,
        "residual_mV2": primary - empirical,
        "expected_spikes": expected,
        "quadrature": {
            "maximum_refinement_relative_change": float(np.max(refinement_relative)),
            "maximum_bound_relative_change": float(np.max(bound_relative)),
            "refinement_passed": bool(
                np.max(refinement_relative) <= QUADRATURE_REL_TOL
            ),
            "bound_sensitivity_passed": bool(
                np.max(bound_relative) <= QUADRATURE_REL_TOL
            ),
        },
        "gain_checks": gain_checks,
        "gain_checks_passed": all(row["passed"] for row in gain_checks),
        "agreement_summaries": summaries,
    }


def plot_step3(record: dict[str, Any], path: Path) -> None:
    """Show the registered linear system as a minimal pair of Bode plots."""
    theme.apply()
    fig, (ax_g, ax_h) = plt.subplots(
        1, 2, figsize=(8.2, 3.25), constrained_layout=True, sharey=True
    )
    frequency = np.geomspace(0.1, 200.0, 1400)
    probe = 1.2
    reference = float(
        np.abs(synapse_membrane_transfer(np.asarray([0.0]), 0.25, probe)[0])
    )
    styles = (
        (theme.INK_BLACK, "0.25 Hz drive"),
        (theme.DEEP_RED, "3 Hz drive"),
        (theme.ELECTRIC_CYAN, "25 Hz drive"),
    )
    for rate, (color, label) in zip(GAIN_OPERATING_RATES_HZ, styles):
        g_magnitude = np.abs(synapse_membrane_transfer(frequency, rate, probe))
        h_magnitude = np.abs(complete_transfer(frequency, rate, probe))
        ax_g.semilogx(
            frequency,
            20.0 * np.log10(np.maximum(g_magnitude / reference, 1e-8)),
            color=color,
            linewidth=1.8,
            label=label,
        )
        ax_h.semilogx(
            frequency,
            20.0 * np.log10(np.maximum(h_magnitude / reference, 1e-8)),
            color=color,
            linewidth=1.8,
        )
    ax_g.set(
        title="A  Synapse + membrane, |Gλ(f)|",
        xlabel="Frequency (Hz)",
        ylabel="Gain relative to low-drive DC (dB)",
        ylim=(-90, 4),
    )
    ax_h.set(
        title="B  After 200 ms averaging, |Hλ(f)|",
        xlabel="Frequency (Hz)",
    )
    ax_g.legend(frameon=False, fontsize=7.5, title="Nominal 1.2 μS probe")
    for axis in (ax_g, ax_h):
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def calculate_step3_empirical_comparison() -> dict[str, Any]:
    """Compare stored stationary predictions with the stored empirical table."""
    with np.load(FIGURES / "step3_linear_filter_arrays.npz") as analytical:
        predicted_mean = analytical["stationary_mean_voltage_mV"] - PARAMETERS["E_L_mV"]
        predicted_sd = np.sqrt(analytical["predicted_variance_mV2"])
    with np.load(FIGURES / "response_library_summary.npz") as empirical:
        empirical_mean = empirical["mean"]
        empirical_sd = empirical["standard_deviation"]

    def summarize(
        predicted: np.ndarray, observed: np.ndarray
    ) -> dict[str, float | int]:
        valid = (
            np.isfinite(predicted)
            & np.isfinite(observed)
            & ((predicted > 0.0) | (observed > 0.0))
        )
        positive_observed = valid & (observed > 0.0)
        return {
            "condition_count": int(np.count_nonzero(valid)),
            "pearson_r": float(np.corrcoef(predicted[valid], observed[valid])[0, 1]),
            "mean_absolute_error_mV": float(
                np.mean(np.abs(predicted[valid] - observed[valid]))
            ),
            "median_predicted_empirical_ratio": float(
                np.median(predicted[positive_observed] / observed[positive_observed])
            ),
        }

    return {
        "comparison": "stationary analytical approximation versus finite 200 ms empirical response table",
        "source_arrays": {
            "analytical": "artifacts/data/exp080/step3_linear_filter_arrays.npz",
            "empirical": "artifacts/data/exp080/response_library_summary.npz",
        },
        "mean": summarize(predicted_mean, empirical_mean),
        "standard_deviation": summarize(predicted_sd, empirical_sd),
        "predicted_mean_mV": predicted_mean,
        "empirical_mean_mV": empirical_mean,
        "predicted_sd_mV": predicted_sd,
        "empirical_sd_mV": empirical_sd,
    }


def plot_step3_empirical_comparison(record: dict[str, Any], path: Path) -> None:
    """Plot analytical predictions directly against stored empirical moments."""
    theme.apply()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.45), constrained_layout=True)
    colors = (theme.INK_BLACK, theme.DEEP_RED, theme.ELECTRIC_CYAN)
    panels = (
        (
            record["predicted_mean_mV"],
            record["empirical_mean_mV"],
            "A  Mean feature",
        ),
        (
            record["predicted_sd_mV"],
            record["empirical_sd_mV"],
            "B  Feature SD",
        ),
    )
    for axis, (predicted, observed, title) in zip(axes, panels, strict=True):
        limit = float(max(np.max(predicted), np.max(observed))) * 1.03
        axis.plot([0.0, limit], [0.0, limit], color="#777777", linewidth=1.0)
        for probe_index, (probe, color) in enumerate(
            zip(PROBE_CONDUCTANCES_US, colors, strict=True)
        ):
            axis.scatter(
                predicted[probe_index].reshape(-1),
                observed[probe_index].reshape(-1),
                s=5,
                alpha=0.24,
                linewidths=0,
                color=color,
                label=f"{probe:g} μS",
                rasterized=True,
            )
        axis.set(
            title=title,
            xlabel="Analytical prediction (mV)",
            ylabel="Empirical value (mV)",
            xlim=(0.0, limit),
            ylim=(0.0, limit),
            aspect="equal",
        )
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7.5, title="Probe conductance")
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)


def refresh_step3_empirical_comparison() -> None:
    """Create the Step 3 comparison from recorded arrays without rerunning science."""
    record = calculate_step3_empirical_comparison()
    figure_path = FIGURES / "linear_filter_empirical_comparison.svg"
    plot_step3_empirical_comparison(record, figure_path)
    outcome = {
        key: value for key, value in record.items() if not isinstance(value, np.ndarray)
    }
    outcome.update(
        {
            "status": "complete",
            "classification": "post-hoc comparison derived only from recorded Step 2 and Step 3 arrays",
            "paid_compute_usd": 0.0,
            "figure_path": str(figure_path.relative_to(REPO)),
            "figure_sha256": sha256_file(figure_path),
            "source_sha256": {
                "analytical": sha256_file(FIGURES / "step3_linear_filter_arrays.npz"),
                "empirical": sha256_file(FIGURES / "response_library_summary.npz"),
            },
        }
    )
    comparison_path = FIGURES / "step3_empirical_comparison.json"
    comparison_path.write_text(json.dumps(outcome, indent=2) + "\n")

    step3_path = FIGURES / "step3_outcome.json"
    step3 = json.loads(step3_path.read_text())
    step3["empirical_comparison"] = {
        "path": str(comparison_path.relative_to(REPO)),
        "sha256": sha256_file(comparison_path),
        "figure_path": str(figure_path.relative_to(REPO)),
        "figure_sha256": sha256_file(figure_path),
    }
    step3_path.write_text(json.dumps(step3, indent=2) + "\n")

    manifest_path = FIGURES / "step2_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["exploratory_continuation"]["step3_outcome_sha256"] = sha256_file(
        step3_path
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    provenance_path = FIGURES / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["step3_empirical_comparison"] = {
        **_git_metadata(),
        "comparison_sha256": sha256_file(comparison_path),
        "figure_sha256": sha256_file(figure_path),
        "paid_compute_usd": 0.0,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    reproducer_path = FIGURES / "reproducer.json"
    reproducer = json.loads(reproducer_path.read_text())
    reproducer["step3_empirical_comparison"] = {
        "command": "uv run python -c 'from experiments.exp080 import refresh_step3_empirical_comparison; refresh_step3_empirical_comparison()'",
        "uses_recorded_arrays_only": True,
        "paid_compute": False,
    }
    reproducer_path.write_text(json.dumps(reproducer, indent=2) + "\n")
    record_steps5_7_publication_contract()


def step_3() -> None:
    started = time.perf_counter()
    library = verify_authenticated_library()
    record = calculate_step3(library)
    if (
        not record["quadrature"]["refinement_passed"]
        or not record["quadrature"]["bound_sensitivity_passed"]
    ):
        raise RuntimeError("Step 3 locked quadrature convergence check failed")
    if not record["gain_checks_passed"]:
        raise RuntimeError("Step 3 locked sinusoidal gain validation failed")
    arrays_path = FIGURES / "step3_linear_filter_arrays.npz"
    np.savez_compressed(
        arrays_path,
        stationary_mean_conductance_uS=record.pop("stationary_mean_conductance_uS"),
        stationary_mean_voltage_mV=record.pop("stationary_mean_voltage_mV"),
        predicted_variance_mV2=record.pop("predicted_variance_mV2"),
        empirical_variance_by_seed_mV2=record.pop("empirical_variance_by_seed_mV2"),
        empirical_variance_mV2=record.pop("empirical_variance_mV2"),
        ratio=record.pop("ratio"),
        residual_mV2=record.pop("residual_mV2"),
        expected_spikes=record.pop("expected_spikes"),
    )
    with np.load(arrays_path) as arrays:
        plot_record = {**record, **{name: arrays[name] for name in arrays.files}}
        plot_step3(plot_record, FIGURES / "linear_filter.svg")
    outcome = {
        "status": "complete",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "classification": "post-hoc exploratory diagnostic after preserved Step 2 failure",
        "library_sha256": LIBRARY_SHA256,
        "amendment_sha256": sha256_file(AMENDMENT_PATH),
        "calibration_point_count": int(np.prod((3, 12, 256))),
        **record,
        "arrays_path": (
            str(arrays_path.relative_to(REPO))
            if arrays_path.is_relative_to(REPO)
            else arrays_path.name
        ),
        "arrays_sha256": sha256_file(arrays_path),
        "figure_path": "artifacts/data/exp080/linear_filter.svg",
        "figure_sha256": sha256_file(FIGURES / "linear_filter.svg"),
        "runtime_s": round(time.perf_counter() - started, 3),
        "paid_compute_usd": 0.0,
    }
    (FIGURES / "step3_outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")
    quadrature = record["quadrature"]
    gain_checks = record["gain_checks"]
    assert isinstance(quadrature, dict)
    assert isinstance(gain_checks, list)
    print(
        "exp080 Step 3 complete: "
        f"max quadrature refinement={quadrature['maximum_refinement_relative_change']:.3g}, "
        f"gain checks={len(gain_checks)}/{len(gain_checks)}"
    )


def _step4_rng(
    stream: int, probe_index: int, rate_index: int, image_index: int, replicate: int
) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(
            [STEP4_SEED, 77, 4, stream, probe_index, rate_index, image_index, replicate]
        )
    )


def sample_library_image(
    library: np.ndarray,
    image_uint8: np.ndarray,
    probe_index: int,
    rate_index: int,
    image_index: int,
    replicate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample table values for the Step 4 diagnostic comparison only."""
    pixels = np.asarray(image_uint8, dtype=np.uint8).reshape(-1)
    rng = _step4_rng(1, probe_index, rate_index, image_index, replicate)
    draws = rng.integers(0, LIBRARY_K, size=pixels.size)
    values = library[0, probe_index, rate_index, pixels, draws]
    return np.asarray(values, dtype=np.float32), draws


def _probe_spikes_features(spikes: np.ndarray, probe_uS: float) -> np.ndarray:
    """Memory-bounded Step 1 probe returning features without full traces."""
    spike_array = np.asarray(spikes, dtype=np.float64)
    if spike_array.shape[0] != N_TIMESTEPS:
        raise ValueError(f"expected {N_TIMESTEPS} timesteps")
    g = np.zeros(spike_array.shape[1:], dtype=np.float64)
    v = np.full(spike_array.shape[1:], PARAMETERS["E_L_mV"], dtype=np.float64)
    total = np.zeros_like(v)
    decay = math.exp(-DT_MS / PARAMETERS["tau_ampa_ms"])
    for incoming in spike_array:
        g = g * decay + probe_uS * incoming
        total_g = PARAMETERS["g_L_uS"] + g
        v_inf = (
            PARAMETERS["g_L_uS"] * PARAMETERS["E_L_mV"] + g * PARAMETERS["E_e_mV"]
        ) / total_g
        v = v_inf + (v - v_inf) * np.exp(-DT_MS * total_g / PARAMETERS["C_m_nF"])
        total += v - PARAMETERS["E_L_mV"]
    return total / N_TIMESTEPS


def direct_feature_replicates(
    image_uint8: np.ndarray,
    rate_hz: float,
    probe_uS: float,
    probe_index: int,
    rate_index: int,
    image_index: int,
) -> np.ndarray:
    """Fresh direct Step 1 simulations for all locked replicates of one image."""
    pixels = np.asarray(image_uint8, dtype=np.uint8).reshape(-1)
    active = np.flatnonzero(pixels)
    output = np.zeros((STEP4_REPLICATES, pixels.size), dtype=np.float32)
    if active.size == 0:
        return output
    intensities = pixels[active].astype(np.float64) / 255.0
    spike_blocks = []
    for replicate in range(STEP4_REPLICATES):
        rng = _step4_rng(2, probe_index, rate_index, image_index, replicate)
        spike_blocks.append(encode_poisson(intensities, rate_hz, rng))
    # time x replicate x active pixel
    spikes = np.stack(spike_blocks, axis=1)
    features = _probe_spikes_features(spikes, probe_uS)
    output[:, active] = features.astype(np.float32)
    return output


def load_locked_mnist_training() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load only the official MNIST training partition; never instantiate test."""
    from torchvision.datasets import MNIST

    dataset = MNIST(root="/tmp/mnist", train=True, download=True)
    images = dataset.data.numpy().astype(np.uint8, copy=False)
    labels = dataset.targets.numpy().astype(np.int64, copy=False)
    if images.shape != (60000, 28, 28) or labels.shape != (60000,):
        raise RuntimeError(f"unexpected MNIST training contract: {images.shape}")
    raw_root = Path(dataset.raw_folder)
    raw_hashes = {
        path.name: sha256_file(path) for path in sorted(raw_root.glob("train-*-ubyte"))
    }
    return (
        images,
        labels,
        {
            "source": "torchvision.datasets.MNIST official training partition",
            "torchvision": __import__("torchvision").__version__,
            "image_shape": list(images.shape),
            "label_shape": list(labels.shape),
            "raw_file_sha256": raw_hashes,
            "official_test_partition_loaded": False,
        },
    )


def _relative_difference(first: float, second: float, floor: float = 1e-12) -> float:
    return abs(first - second) / max((abs(first) + abs(second)) / 2.0, floor)


def compare_feature_condition(
    library_values: np.ndarray, direct_values: np.ndarray, regime: str
) -> dict[str, Any]:
    """Apply the locked Step 4 metrics to one probe-rate condition."""
    lib = np.asarray(library_values, dtype=np.float64)
    direct = np.asarray(direct_values, dtype=np.float64)
    if lib.shape != direct.shape:
        raise ValueError("response-table and direct arrays must have matching shapes")
    lib_mean = float(np.mean(lib))
    direct_mean = float(np.mean(direct))
    lib_variance = float(np.var(lib, ddof=1))
    direct_variance = float(np.var(direct, ddof=1))
    lib_image_means = np.mean(lib, axis=-1)
    direct_image_means = np.mean(direct, axis=-1)
    lib_image_variances = np.var(lib, axis=-1, ddof=1)
    direct_image_variances = np.var(direct, axis=-1, ddof=1)
    # Average the eight independent replicates before measuring spatial signal.
    lib_spatial = np.mean(lib, axis=1).reshape(-1)
    direct_spatial = np.mean(direct, axis=1).reshape(-1)
    correlation = float(np.corrcoef(lib_spatial, direct_spatial)[0, 1])
    thresholds = {
        "pooled_mean_relative_difference": 0.1,
        "pooled_variance_relative_difference": 0.2,
        "zero_fraction_absolute_difference": 0.03,
        "image_mean_relative_difference_median": 0.2,
        "image_variance_relative_difference_median": 0.35,
        "spatial_mean_correlation_minimum": {
            "low": 0.2,
            "transitional": 0.75,
            "high": 0.9,
        }[regime],
    }
    metrics = {
        "library_pooled_mean_mV": lib_mean,
        "direct_pooled_mean_mV": direct_mean,
        "pooled_mean_relative_difference": _relative_difference(lib_mean, direct_mean),
        "library_pooled_variance_mV2": lib_variance,
        "direct_pooled_variance_mV2": direct_variance,
        "pooled_variance_relative_difference": _relative_difference(
            lib_variance, direct_variance
        ),
        "library_zero_fraction": float(np.mean(lib == 0.0)),
        "direct_zero_fraction": float(np.mean(direct == 0.0)),
        "zero_fraction_absolute_difference": float(
            abs(np.mean(lib == 0.0) - np.mean(direct == 0.0))
        ),
        "paired_mean_absolute_difference_mV": float(np.mean(np.abs(lib - direct))),
        "image_mean_relative_difference_median": float(
            np.median(
                np.abs(lib_image_means - direct_image_means)
                / np.maximum(
                    (np.abs(lib_image_means) + np.abs(direct_image_means)) / 2.0, 1e-12
                )
            )
        ),
        "image_variance_relative_difference_median": float(
            np.median(
                np.abs(lib_image_variances - direct_image_variances)
                / np.maximum(
                    (np.abs(lib_image_variances) + np.abs(direct_image_variances))
                    / 2.0,
                    1e-12,
                )
            )
        ),
        "spatial_mean_correlation": correlation,
    }
    passed = {
        name: metrics[name] <= limit
        for name, limit in thresholds.items()
        if name != "spatial_mean_correlation_minimum"
    }
    passed["spatial_mean_correlation_minimum"] = (
        correlation >= thresholds["spatial_mean_correlation_minimum"]
    )
    return {
        "metrics": metrics,
        "thresholds": thresholds,
        "checks": passed,
        "passed": all(passed.values()),
    }


def plot_step4(
    images: np.ndarray,
    library_values: np.ndarray,
    direct_values: np.ndarray,
    path: Path,
) -> None:
    """Show intuitive response-table and direct feature-image examples only."""
    theme.apply()
    fig, axes = plt.subplots(3, 3, figsize=(8.0, 5.8), constrained_layout=True)
    nominal_probe_index = PROBE_CONDUCTANCES_US.index(1.2)
    for row, (regime, rate) in enumerate(STEP4_RATES.items()):
        image_index = STEP4_DIAGNOSTIC_INDICES[regime]
        position = STEP4_IMAGE_INDICES.index(image_index)
        library_image = library_values[nominal_probe_index, row, position, 0].reshape(
            28, 28
        )
        direct_image = direct_values[nominal_probe_index, row, position, 0].reshape(
            28, 28
        )
        axes[row, 0].imshow(images[image_index], cmap="gray", vmin=0, vmax=255)
        axes[row, 1].imshow(library_image, cmap="magma", vmin=0, vmax=65)
        axes[row, 2].imshow(direct_image, cmap="magma", vmin=0, vmax=65)
        axes[row, 0].set_ylabel(
            f"{rate:g} Hz",
            fontsize=10,
            rotation=0,
            ha="right",
            va="center",
            labelpad=18,
        )
    for column, title in enumerate(
        ("Original intensity", "Empirical response table", "Fresh direct simulation")
    ):
        axes[0, column].set_title(title, fontsize=9)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        "Empirical response table versus fresh direct simulation",
        fontsize=11,
        fontweight="bold",
    )
    savefig_atomic(fig, path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def step_4() -> None:
    started = time.perf_counter()
    library = verify_authenticated_library()
    images, labels, dataset_record = load_locked_mnist_training()
    shape = (
        len(PROBE_CONDUCTANCES_US),
        len(STEP4_RATES),
        len(STEP4_IMAGE_INDICES),
        STEP4_REPLICATES,
        784,
    )
    library_values = np.empty(shape, dtype=np.float32)
    direct_values = np.empty(shape, dtype=np.float32)
    draw_indices = np.empty(shape, dtype=np.uint16)
    for probe_index, probe in enumerate(PROBE_CONDUCTANCES_US):
        for rate_index, (regime, rate) in enumerate(STEP4_RATES.items()):
            for image_position, image_index in enumerate(STEP4_IMAGE_INDICES):
                for replicate in range(STEP4_REPLICATES):
                    values, draws = sample_library_image(
                        library,
                        images[image_index],
                        probe_index,
                        TRAINING_RATES_HZ.index(rate),
                        image_index,
                        replicate,
                    )
                    library_values[
                        probe_index, rate_index, image_position, replicate
                    ] = values
                    draw_indices[probe_index, rate_index, image_position, replicate] = (
                        draws
                    )
                direct_values[probe_index, rate_index, image_position] = (
                    direct_feature_replicates(
                        images[image_index],
                        rate,
                        probe,
                        probe_index,
                        TRAINING_RATES_HZ.index(rate),
                        image_index,
                    )
                )
            print(f"Step 4 direct: {probe:g} uS {regime}, 16 images complete")
    condition_records: list[dict[str, Any]] = []
    for probe_index, probe in enumerate(PROBE_CONDUCTANCES_US):
        for rate_index, (regime, rate) in enumerate(STEP4_RATES.items()):
            comparison = compare_feature_condition(
                library_values[probe_index, rate_index],
                direct_values[probe_index, rate_index],
                regime,
            )
            condition_records.append(
                {
                    "probe_uS": probe,
                    "drive_regime": regime,
                    "rate_hz": rate,
                    "comparison": comparison,
                }
            )

    replay_values, replay_draws = sample_library_image(
        library, images[0], 0, TRAINING_RATES_HZ.index(0.25), 0, 0
    )
    direct_replay = direct_feature_replicates(
        images[0], 0.25, 0.6, 0, TRAINING_RATES_HZ.index(0.25), 0
    )
    validations = {
        "all_conditions_passed": all(
            row["comparison"]["passed"] for row in condition_records
        ),
        "finite_and_bounded": bool(
            np.all(np.isfinite(library_values))
            and np.all(np.isfinite(direct_values))
            and np.all((library_values >= 0) & (library_values <= 65))
            and np.all((direct_values >= 0) & (direct_values <= 65))
        ),
        "library_deterministic_replay": bool(
            np.array_equal(replay_values, library_values[0, 0, 0, 0])
            and np.array_equal(replay_draws, draw_indices[0, 0, 0, 0])
        ),
        "direct_deterministic_replay": bool(
            np.array_equal(direct_replay, direct_values[0, 0, 0])
        ),
        "pixel_and_image_stream_independence": bool(
            not np.array_equal(draw_indices[0, 0, 0, 0], draw_indices[0, 0, 0, 1])
            and not np.array_equal(
                library_values[0, 0, 0, 0], library_values[0, 0, 1, 0]
            )
            and not np.array_equal(direct_values[0, 0, 0, 0], direct_values[0, 0, 0, 1])
        ),
        "low_rate_zero_heavy": bool(np.mean(library_values[:, 0] == 0) > 0.9),
        "higher_rate_spatial_structure": bool(
            min(
                row["comparison"]["metrics"]["spatial_mean_correlation"]
                for row in condition_records
                if row["drive_regime"] == "high"
            )
            >= 0.9
        ),
        "held_out_test_partition_sealed": True,
    }
    arrays_path = FIGURES / "step4_feature_comparison_arrays.npz"
    np.savez_compressed(
        arrays_path,
        image_indices=np.asarray(STEP4_IMAGE_INDICES),
        image_labels=labels[np.asarray(STEP4_IMAGE_INDICES)],
        original_images=images[np.asarray(STEP4_IMAGE_INDICES)],
        library_values=library_values,
        direct_values=direct_values,
        draw_indices=draw_indices,
    )
    figure_path = FIGURES / "feature_images.png"
    plot_step4(images, library_values, direct_values, figure_path)
    outcome = {
        "status": "complete" if all(validations.values()) else "validation_failed",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "classification": "post-hoc exploratory feature-construction validation after preserved Step 2 failure",
        "library_sha256": LIBRARY_SHA256,
        "amendment_sha256": sha256_file(AMENDMENT_PATH),
        "dataset": dataset_record,
        "decoder_train_indices": [0, 54999],
        "validation_indices": [55000, 59999],
        "comparison_image_indices": list(STEP4_IMAGE_INDICES),
        "diagnostic_image_indices": STEP4_DIAGNOSTIC_INDICES,
        "condition_records": condition_records,
        "validations": validations,
        "arrays_path": str(arrays_path.relative_to(REPO)),
        "arrays_sha256": sha256_file(arrays_path),
        "figure_path": str(figure_path.relative_to(REPO)),
        "figure_sha256": sha256_file(figure_path),
        "runtime_s": round(time.perf_counter() - started, 3),
        "paid_compute_usd": 0.0,
        "interpretation": "feature construction only; no decoder, held-out test, threshold selection, or PING run",
    }
    (FIGURES / "step4_outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")
    if not all(validations.values()):
        failed = [name for name, passed in validations.items() if not passed]
        raise RuntimeError(f"Step 4 validation failed: {', '.join(failed)}")
    print(f"exp080 Step 4 complete: {len(condition_records)}/9 conditions passed")


def record_steps3_4_publication_contract() -> None:
    """Extend cumulative exp080 metadata without erasing Steps 1--2 history."""
    step3_path = FIGURES / "step3_outcome.json"
    step4_path = FIGURES / "step4_outcome.json"
    step3_record = json.loads(step3_path.read_text())
    step4_record = json.loads(step4_path.read_text())
    numbers_path = FIGURES / "numbers.json"
    numbers = json.loads(numbers_path.read_text())
    numbers.update(
        {
            "step": 4,
            "status": "killed_at_step4_validation",
            "scope": (
                "Exploratory Steps 3-4 followed the preserved Step 2 failure; "
                "Step 4 stopped at locked low-rate image-level checks"
            ),
            "step3": step3_record,
            "step4": step4_record,
            "later_steps_run": False,
            "paid_compute_usd": 0.0,
        }
    )
    numbers_path.write_text(json.dumps(numbers, indent=2) + "\n")

    protocol_path = FIGURES / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol.update(
        {
            "attempted_through_step": 4,
            "step2_status": "full_library_validation_failed",
            "steps3_4_classification": "explicitly authorized post-hoc exploratory amendment",
            "step3_status": step3_record["status"],
            "step4_status": step4_record["status"],
            "later_steps": "Steps 5-7 not run",
            "held_out_test_partition": "sealed",
        }
    )
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

    manifest_path = FIGURES / "step2_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["exploratory_continuation"] = {
        "original_step2_status_unchanged": manifest["status"],
        "amendment_path": "artifacts/data/exp080/step3_step4_exploratory_amendment.json",
        "amendment_sha256": sha256_file(AMENDMENT_PATH),
        "step3_outcome": str(step3_path.relative_to(REPO)),
        "step3_outcome_sha256": sha256_file(step3_path),
        "step4_outcome": str(step4_path.relative_to(REPO)),
        "step4_outcome_sha256": sha256_file(step4_path),
        "steps5_7_run": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    provenance_path = FIGURES / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["exploratory_steps3_4"] = {
        **_git_metadata(),
        "library_sha256_verified": LIBRARY_SHA256,
        "amendment_sha256": sha256_file(AMENDMENT_PATH),
        "step3_outcome_sha256": sha256_file(step3_path),
        "step4_outcome_sha256": sha256_file(step4_path),
        "linear_filter_figure_sha256": sha256_file(FIGURES / "linear_filter.svg"),
        "feature_images_figure_sha256": sha256_file(FIGURES / "feature_images.png"),
        "paid_compute_usd": 0.0,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    reproducer_path = FIGURES / "reproducer.json"
    reproducer = json.loads(reproducer_path.read_text())
    reproducer["exploratory_steps3_4"] = {
        "step3_command": "uv run python -c 'from experiments import exp080; exp080.step_3()'",
        "step4_command": "uv run python -c 'from experiments import exp080; exp080.step_4()'",
        "step4_expected_outcome": "locked low-rate image-level validation failure",
        "metadata_command": "uv run python -c 'from experiments import exp080; exp080.record_steps3_4_publication_contract()'",
        "expected_library_sha256": LIBRARY_SHA256,
        "expected_outputs": [
            "artifacts/data/exp080/linear_filter.svg",
            "artifacts/data/exp080/step3_outcome.json",
            "artifacts/data/exp080/feature_images.png",
            "artifacts/data/exp080/step4_outcome.json",
        ],
        "paid_compute": False,
    }
    reproducer_path.write_text(json.dumps(reproducer, indent=2) + "\n")


def refresh_primary_result_figures() -> None:
    """Redraw the four primary figures from authenticated recorded artifacts."""
    numbers = json.loads((FIGURES / "numbers.json").read_text())
    plot_probe(numbers["plot_data"], FIGURES / "probe_dynamics.svg")
    refresh_full_library_figure()

    step3_path = FIGURES / "step3_outcome.json"
    step3_record = json.loads(step3_path.read_text())
    with np.load(FIGURES / "step3_linear_filter_arrays.npz") as arrays:
        plot_step3(
            {**step3_record, **{name: arrays[name] for name in arrays.files}},
            FIGURES / "linear_filter.svg",
        )
    step3_record["figure_sha256"] = sha256_file(FIGURES / "linear_filter.svg")
    step3_path.write_text(json.dumps(step3_record, indent=2) + "\n")

    step4_path = FIGURES / "step4_outcome.json"
    step4_record = json.loads(step4_path.read_text())
    with np.load(FIGURES / "step4_feature_comparison_arrays.npz") as arrays:
        plot_step4(
            arrays["original_images"],
            arrays["library_values"],
            arrays["direct_values"],
            FIGURES / "feature_images.png",
        )
    step4_record["figure_sha256"] = sha256_file(FIGURES / "feature_images.png")
    step4_path.write_text(json.dumps(step4_record, indent=2) + "\n")
    record_steps3_4_publication_contract()


def _decoder_seed(*parts: int) -> int:
    """Stable 63-bit seed derived from the registered experimental coordinates."""
    digest = hashlib.sha256(
        ":".join(str(part) for part in (77, *parts)).encode()
    ).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _torch_device() -> Any:
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def direct_feature_batch_torch(
    images_uint8: Any,
    rates_hz: Any,
    probe_uS: float,
    generator: Any,
) -> Any:
    """Generate exact fresh features without storing the Bernoulli spike train.

    This is the registered decay-then-add synapse and exponential-Euler membrane
    update, evaluated for all pixels in a batch. Random draws are fresh for every
    call and deterministic for a fixed generator state.
    """
    import torch

    images = images_uint8.to(dtype=torch.float32).reshape(-1, 784) / 255.0
    rates = rates_hz.to(device=images.device, dtype=torch.float32).reshape(-1, 1)
    probability = images * rates * (DT_MS / 1000.0)
    g = torch.zeros_like(images)
    v = torch.full_like(images, PARAMETERS["E_L_mV"])
    total = torch.zeros_like(images)
    decay = math.exp(-DT_MS / PARAMETERS["tau_ampa_ms"])
    for _ in range(N_TIMESTEPS):
        incoming = (
            torch.rand(
                images.shape,
                device=images.device,
                dtype=images.dtype,
                generator=generator,
            )
            < probability
        )
        g = g * decay + probe_uS * incoming
        total_g = PARAMETERS["g_L_uS"] + g
        v_inf = (
            PARAMETERS["g_L_uS"] * PARAMETERS["E_L_mV"] + g * PARAMETERS["E_e_mV"]
        ) / total_g
        v = v_inf + (v - v_inf) * torch.exp(-DT_MS * total_g / PARAMETERS["C_m_nF"])
        total += v - PARAMETERS["E_L_mV"]
    return total / N_TIMESTEPS


def _make_decoders(device: Any, seed: int) -> tuple[Any, list[Any]]:
    import torch

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    nonlinear = torch.nn.Sequential(
        torch.nn.Linear(784, DECODER_HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(DECODER_HIDDEN, 10),
    ).to(device)
    linear = [torch.nn.Linear(784, 10).to(device) for _ in LINEAR_WEIGHT_DECAYS]
    return nonlinear, linear


def _classification_metrics(logits: Any, labels: Any) -> tuple[Any, int]:
    import torch

    loss = torch.nn.functional.cross_entropy(logits, labels)
    correct = int((logits.argmax(dim=1) == labels).sum().item())
    return loss, correct


def _dataset_batches(
    indices: np.ndarray,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> list[np.ndarray]:
    ordered = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(ordered)
    return [
        ordered[start : start + batch_size]
        for start in range(0, len(ordered), batch_size)
    ]


def train_decoder_condition(
    *,
    images: np.ndarray,
    labels: np.ndarray,
    probe_uS: float,
    seed: int,
    epochs: int,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    batch_size: int = DECODER_BATCH_SIZE,
    output_dir: Path,
) -> dict[str, Any]:
    """Train the nonlinear decoder and matched regularized linear candidates."""
    import torch

    device = _torch_device()
    nonlinear, linear_models = _make_decoders(device, seed)
    nonlinear_optimizer = torch.optim.Adam(
        nonlinear.parameters(), lr=DECODER_LEARNING_RATE
    )
    linear_optimizers = [
        torch.optim.Adam(
            model.parameters(), lr=DECODER_LEARNING_RATE, weight_decay=decay
        )
        for model, decay in zip(linear_models, LINEAR_WEIGHT_DECAYS, strict=True)
    ]
    feature_generator = torch.Generator(device=device)
    history: list[dict[str, Any]] = []
    best_nonlinear = (-math.inf, 0, None)
    best_linear = [(-math.inf, 0, None) for _ in linear_models]
    started = time.perf_counter()
    sampled_counts = {str(rate): 0 for rate in DECODER_RATES_HZ}

    for epoch in range(1, epochs + 1):
        nonlinear.train()
        for model in linear_models:
            model.train()
        train_loss = 0.0
        train_correct = 0
        linear_train_loss = [0.0 for _ in linear_models]
        linear_train_correct = [0 for _ in linear_models]
        seen = 0
        batches = _dataset_batches(
            train_indices, batch_size, _decoder_seed(seed, epoch, 1), True
        )
        rate_rng = np.random.default_rng(_decoder_seed(seed, epoch, 2))
        feature_generator.manual_seed(_decoder_seed(seed, epoch, 3))
        for batch_indices in batches:
            rate_positions = rate_rng.integers(
                0, len(DECODER_RATES_HZ), len(batch_indices)
            )
            sampled_rates = np.asarray(DECODER_RATES_HZ)[rate_positions]
            for rate in sampled_rates:
                sampled_counts[str(float(rate))] += 1
            batch_images = torch.as_tensor(images[batch_indices], device=device)
            batch_labels = torch.as_tensor(labels[batch_indices], device=device)
            rate_tensor = torch.as_tensor(sampled_rates, device=device)
            features = direct_feature_batch_torch(
                batch_images, rate_tensor, probe_uS, feature_generator
            )
            nonlinear_optimizer.zero_grad(set_to_none=True)
            nonlinear_logits = nonlinear(features)
            loss, correct = _classification_metrics(nonlinear_logits, batch_labels)
            loss.backward()
            nonlinear_optimizer.step()
            train_loss += float(loss.item()) * len(batch_indices)
            train_correct += correct
            for position, (model, optimizer) in enumerate(
                zip(linear_models, linear_optimizers, strict=True)
            ):
                optimizer.zero_grad(set_to_none=True)
                logits = model(features.detach())
                linear_loss, linear_correct = _classification_metrics(
                    logits, batch_labels
                )
                linear_loss.backward()
                optimizer.step()
                linear_train_loss[position] += float(linear_loss.item()) * len(
                    batch_indices
                )
                linear_train_correct[position] += linear_correct
            seen += len(batch_indices)

        nonlinear.eval()
        for model in linear_models:
            model.eval()
        validation_loss = 0.0
        validation_correct = 0
        linear_validation_loss = [0.0 for _ in linear_models]
        linear_validation_correct = [0 for _ in linear_models]
        validation_seen = 0
        feature_generator.manual_seed(_decoder_seed(seed, epoch, 4))
        validation_rate_rng = np.random.default_rng(_decoder_seed(seed, epoch, 5))
        with torch.no_grad():
            for batch_indices in _dataset_batches(
                validation_indices, batch_size, _decoder_seed(seed, epoch, 6), False
            ):
                rate_positions = validation_rate_rng.integers(
                    0, len(DECODER_RATES_HZ), len(batch_indices)
                )
                sampled_rates = np.asarray(DECODER_RATES_HZ)[rate_positions]
                batch_images = torch.as_tensor(images[batch_indices], device=device)
                batch_labels = torch.as_tensor(labels[batch_indices], device=device)
                features = direct_feature_batch_torch(
                    batch_images,
                    torch.as_tensor(sampled_rates, device=device),
                    probe_uS,
                    feature_generator,
                )
                logits = nonlinear(features)
                loss, correct = _classification_metrics(logits, batch_labels)
                validation_loss += float(loss.item()) * len(batch_indices)
                validation_correct += correct
                for position, model in enumerate(linear_models):
                    linear_loss, linear_correct = _classification_metrics(
                        model(features), batch_labels
                    )
                    linear_validation_loss[position] += float(linear_loss.item()) * len(
                        batch_indices
                    )
                    linear_validation_correct[position] += linear_correct
                validation_seen += len(batch_indices)
        nonlinear_accuracy = validation_correct / validation_seen
        if nonlinear_accuracy > best_nonlinear[0]:
            best_nonlinear = (
                nonlinear_accuracy,
                epoch,
                {
                    name: value.detach().cpu()
                    for name, value in nonlinear.state_dict().items()
                },
            )
        for position, model in enumerate(linear_models):
            accuracy = linear_validation_correct[position] / validation_seen
            if accuracy > best_linear[position][0]:
                best_linear[position] = (
                    accuracy,
                    epoch,
                    {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                )
        row = {
            "epoch": epoch,
            "nonlinear": {
                "train_loss": train_loss / seen,
                "train_accuracy": train_correct / seen,
                "validation_loss": validation_loss / validation_seen,
                "validation_accuracy": nonlinear_accuracy,
            },
            "linear": [
                {
                    "weight_decay": decay,
                    "train_loss": linear_train_loss[position] / seen,
                    "train_accuracy": linear_train_correct[position] / seen,
                    "validation_loss": linear_validation_loss[position]
                    / validation_seen,
                    "validation_accuracy": linear_validation_correct[position]
                    / validation_seen,
                }
                for position, decay in enumerate(LINEAR_WEIGHT_DECAYS)
            ],
        }
        history.append(row)
        print(
            f"probe={probe_uS:g} seed={seed} epoch={epoch}/{epochs} "
            f"nonlinear_val={nonlinear_accuracy:.4f} "
            f"linear_val={max(item['validation_accuracy'] for item in row['linear']):.4f}",
            flush=True,
        )

    selected_linear_index = int(np.argmax([record[0] for record in best_linear]))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "decoders.pt"
    torch.save(
        {
            "nonlinear": best_nonlinear[2],
            "linear": best_linear[selected_linear_index][2],
            "configuration": {
                "probe_uS": probe_uS,
                "seed": seed,
                "hidden": DECODER_HIDDEN,
                "learning_rate": DECODER_LEARNING_RATE,
                "batch_size": batch_size,
                "rate_grid_hz": DECODER_RATES_HZ,
                "linear_weight_decay": LINEAR_WEIGHT_DECAYS[selected_linear_index],
            },
        },
        checkpoint,
    )
    result = {
        "status": "complete",
        "probe_uS": probe_uS,
        "seed": seed,
        "device": str(device),
        "epochs_run": epochs,
        "selected_nonlinear_epoch": best_nonlinear[1],
        "selected_nonlinear_validation_accuracy": best_nonlinear[0],
        "selected_linear_epoch": best_linear[selected_linear_index][1],
        "selected_linear_validation_accuracy": best_linear[selected_linear_index][0],
        "selected_linear_weight_decay": LINEAR_WEIGHT_DECAYS[selected_linear_index],
        "rate_sample_counts": sampled_counts,
        "history": history,
        "runtime_s": time.perf_counter() - started,
        "checkpoint": checkpoint.name,
    }
    (output_dir / "training.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def write_expanded_rate_training_protocol() -> Path:
    """Freeze the sparse-rate retraining design before any new training outcomes."""
    protocol = {
        "status": "frozen_before_expanded_rate_retraining",
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": _git_metadata(),
        "purpose": "give 0.01, 0.05, and 0.1 Hz equal training and evaluation status",
        "predecessor_rate_grid_hz": list(TRAINING_RATES_HZ),
        "added_rates_hz": [0.01, 0.05, 0.1],
        "decoder_rate_grid_hz": list(DECODER_RATES_HZ),
        "training_rate_distribution": {
            "form": "uniform categorical per image presentation",
            "probability_per_rate": 1.0 / len(DECODER_RATES_HZ),
        },
        "unchanged_design": {
            "training_indices": [0, 54_999],
            "validation_indices": [55_000, 59_999],
            "probe_conductances_uS": list(PROBE_CONDUCTANCES_US),
            "decoder_seeds": list(SEEDS),
            "epochs": DECODER_MAX_EPOCHS,
            "batch_size": DECODER_BATCH_SIZE,
            "learning_rate": DECODER_LEARNING_RATE,
            "hidden_units": DECODER_HIDDEN,
            "linear_weight_decays": list(LINEAR_WEIGHT_DECAYS),
            "fresh_direct_simulation_per_presentation": True,
            "validation_selects_checkpoint_and_linear_weight_decay": True,
        },
        "evaluation": {
            "same_rate_grid_as_training": True,
            "direct_simulation_draws": EVALUATION_DRAWS,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS_HELDOUT,
            "chance_accuracy": CHANCE_ACCURACY,
            "useful_accuracy": USEFUL_ACCURACY,
            "threshold_rules_unchanged": True,
        },
        "scope": (
            "Steps 1-4 retain the authenticated 0.25-25 Hz empirical calibration; "
            "all decoder inputs use fresh direct simulation"
        ),
    }
    path = FIGURES / "expanded_rate_training_protocol.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    return path


def verify_expanded_rate_training_protocol() -> dict[str, Any]:
    """Fail closed unless the committed sparse-rate training design matches code."""
    path = FIGURES / "expanded_rate_training_protocol.json"
    protocol: dict[str, Any]
    if path.exists():
        if sha256_file(path) != EXPANDED_RATE_PROTOCOL_SHA256:
            raise RuntimeError("expanded-rate training protocol hash mismatch")
        protocol = json.loads(path.read_text())
    else:
        remote_hash = os.environ.get("EXP080_FROZEN_TRAINING_PROTOCOL_SHA256")
        if remote_hash != EXPANDED_RATE_PROTOCOL_SHA256:
            raise RuntimeError(
                "expanded-rate training requires a frozen training protocol"
            )
        protocol = {
            "status": "frozen_before_expanded_rate_retraining",
            "decoder_rate_grid_hz": list(DECODER_RATES_HZ),
            "training_rate_distribution": {
                "probability_per_rate": 1.0 / len(DECODER_RATES_HZ)
            },
        }
    checks = {
        "status": protocol.get("status") == "frozen_before_expanded_rate_retraining",
        "rate_grid": protocol.get("decoder_rate_grid_hz") == list(DECODER_RATES_HZ),
        "uniform_probability": protocol.get("training_rate_distribution", {}).get(
            "probability_per_rate"
        )
        == 1.0 / len(DECODER_RATES_HZ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"expanded-rate training protocol mismatch: {checks}")
    return protocol


def run_step5_stage(stage: str) -> None:
    """Run a local smoke or one remote training cell without accessing test data."""
    images, labels, dataset = load_locked_mnist_training()
    stage_settings = {
        "smoke": (128, 64, 2),
        "pilot": (1000, 500, 2),
        "full": (55_000, 5_000, DECODER_MAX_EPOCHS),
    }
    if stage not in stage_settings:
        raise ValueError(f"unknown Step 5 stage: {stage}")
    if stage == "full":
        verify_expanded_rate_training_protocol()
    train_count, validation_count, epochs = stage_settings[stage]
    probe = float(os.environ.get("EXP080_PROBE_US", str(PROBE_US)))
    seed = int(os.environ.get("EXP080_SEED", str(SEED)))
    output_dir = FIGURES / "step5" / stage / f"probe-{probe:g}" / f"seed-{seed}"
    result = train_decoder_condition(
        images=images,
        labels=labels,
        probe_uS=probe,
        seed=seed,
        epochs=epochs,
        train_indices=np.arange(TRAIN_INDICES[0], TRAIN_INDICES[0] + train_count),
        validation_indices=np.arange(
            VALIDATION_INDICES[0], VALIDATION_INDICES[0] + validation_count
        ),
        batch_size=min(DECODER_BATCH_SIZE, train_count),
        output_dir=output_dir,
    )
    result["dataset"] = dataset
    result["train_indices"] = [TRAIN_INDICES[0], TRAIN_INDICES[0] + train_count - 1]
    result["validation_indices"] = [
        VALIDATION_INDICES[0],
        VALIDATION_INDICES[0] + validation_count - 1,
    ]
    (output_dir / "training.json").write_text(json.dumps(result, indent=2) + "\n")


def step_5() -> None:
    run_step5_stage(os.environ.get("EXP080_STAGE", "smoke"))


def finalize_step5() -> dict[str, Any]:
    """Aggregate the nine full training cells and draw the registered history."""
    records: list[dict[str, Any]] = []
    for probe in PROBE_CONDUCTANCES_US:
        for seed in SEEDS:
            directory = FIGURES / "step5" / "full" / f"probe-{probe:g}" / f"seed-{seed}"
            training_path = directory / "training.json"
            modal_path = directory / "modal.json"
            checkpoint_path = directory / "decoders.pt"
            if not all(
                path.exists() for path in (training_path, modal_path, checkpoint_path)
            ):
                raise RuntimeError(
                    f"incomplete Step 5 cell: probe={probe:g}, seed={seed}"
                )
            training = json.loads(training_path.read_text())
            modal = json.loads(modal_path.read_text())
            records.append(
                {
                    "probe_uS": probe,
                    "seed": seed,
                    "training_path": str(training_path.relative_to(REPO)),
                    "training_sha256": sha256_file(training_path),
                    "checkpoint_path": str(checkpoint_path.relative_to(REPO)),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "selected_nonlinear_epoch": training["selected_nonlinear_epoch"],
                    "selected_nonlinear_validation_accuracy": training[
                        "selected_nonlinear_validation_accuracy"
                    ],
                    "selected_linear_epoch": training["selected_linear_epoch"],
                    "selected_linear_validation_accuracy": training[
                        "selected_linear_validation_accuracy"
                    ],
                    "selected_linear_weight_decay": training[
                        "selected_linear_weight_decay"
                    ],
                    "rate_sample_counts": training["rate_sample_counts"],
                    "history": training["history"],
                    "runtime_s": training["runtime_s"],
                    "modal_elapsed_s": modal["elapsed_s"],
                    "estimated_cost_usd": modal["estimated_cost_usd"],
                }
            )
    theme.apply()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)
    colors = {0.6: "#4477AA", 1.2: "#222222", 2.4: "#CC6677"}
    for probe in PROBE_CONDUCTANCES_US:
        probe_records = [record for record in records if record["probe_uS"] == probe]
        epochs = np.asarray([row["epoch"] for row in probe_records[0]["history"]])
        nonlinear = np.asarray(
            [
                [row["nonlinear"]["validation_accuracy"] for row in record["history"]]
                for record in probe_records
            ]
        )
        linear = np.asarray(
            [
                [
                    max(item["validation_accuracy"] for item in row["linear"])
                    for row in record["history"]
                ]
                for record in probe_records
            ]
        )
        for axis, values, title in (
            (axes[0], nonlinear, "Nonlinear decoder"),
            (axes[1], linear, "Linear decoder"),
        ):
            mean = np.mean(values, axis=0)
            low = np.min(values, axis=0)
            high = np.max(values, axis=0)
            axis.plot(epochs, mean, color=colors[probe], label=f"{probe:g} μS")
            axis.fill_between(
                epochs, low, high, color=colors[probe], alpha=0.16, linewidth=0
            )
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Validation accuracy")
            axis.set_ylim(0, 1)
    axes[1].legend(frameon=False, title="Probe")
    figure_path = FIGURES / "step5_training_history.svg"
    savefig_atomic(fig, figure_path, bbox_inches="tight")
    plt.close(fig)
    killed_cost = json.loads(
        (FIGURES / "step5" / "pilot" / "attempt-002.json").read_text()
    )["estimated_cost_usd"]
    pilot_cost = json.loads(
        (
            FIGURES / "step5" / "pilot" / "probe-1.2" / "seed-42" / "modal.json"
        ).read_text()
    )["estimated_cost_usd"]
    outcome = {
        "status": "complete",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": records,
        "input_diagnostics": {
            "rate_sampling": "uniform categorical over the registered grid for every presentation",
            "per_cell_rate_sample_counts_recorded": True,
            "fresh_direct_features": True,
            "feature_bounds_mV": [0.0, 65.0],
            "firing_rate": "not applicable: the registered pixel probe is explicitly subthreshold and non-spiking",
        },
        "figure_path": str(figure_path.relative_to(REPO)),
        "figure_sha256": sha256_file(figure_path),
        "successful_step5_estimated_cost_usd": sum(
            float(record["estimated_cost_usd"]) for record in records
        ),
        "killed_attempt_estimated_cost_usd": killed_cost,
        "successful_pilot_estimated_cost_usd": pilot_cost,
    }
    outcome["cumulative_estimated_cost_usd"] = (
        outcome["successful_step5_estimated_cost_usd"]
        + outcome["killed_attempt_estimated_cost_usd"]
        + outcome["successful_pilot_estimated_cost_usd"]
    )
    path = FIGURES / "step5_outcome.json"
    path.write_text(json.dumps(outcome, indent=2) + "\n")
    return outcome


def recover_step6_metadata() -> dict[str, Any]:
    """Validate a complete returned tensor after the post-write path failure."""
    protocol_path = FIGURES / "frozen_evaluation_protocol.json"
    arrays_path = FIGURES / "step6" / "held_out_correctness.npz"
    modal_path = FIGURES / "step6" / "modal.json"
    _, labels, dataset = load_held_out_mnist_test(protocol_path)
    with np.load(arrays_path) as arrays:
        checks = {
            "nonlinear_shape": arrays["nonlinear_correct"].shape
            == (3, len(DECODER_RATES_HZ), 3, 3, 10_000),
            "linear_shape": arrays["linear_correct"].shape
            == (len(DECODER_RATES_HZ), 3, 3, 10_000),
            "rate_grid": np.array_equal(arrays["rates_hz"], DECODER_RATES_HZ),
            "probe_grid": np.array_equal(arrays["probes_uS"], PROBE_CONDUCTANCES_US),
            "decoder_seeds": np.array_equal(arrays["decoder_seeds"], SEEDS),
            "labels": np.array_equal(arrays["labels"], labels),
            "boolean_correctness": (
                arrays["nonlinear_correct"].dtype == np.bool_
                and arrays["linear_correct"].dtype == np.bool_
            ),
        }
    if not all(checks.values()):
        raise RuntimeError(f"returned Step 6 tensor validation failed: {checks}")
    modal_record = json.loads(modal_path.read_text())
    outcome = {
        "status": "complete_after_post_compute_metadata_recovery",
        "protocol_sha256": sha256_file(protocol_path),
        "dataset": dataset,
        "device": "cuda",
        "runtime_s": modal_record["elapsed_s"],
        "arrays_path": str(arrays_path.relative_to(REPO)),
        "arrays_sha256": sha256_file(arrays_path),
        "validation_checks": checks,
        "preserved_failure": {
            "stage": "post-compute metadata formatting",
            "scientific_tensor_completed": True,
            "remote_error": modal_record["error"],
            "modal_artifact_payload_sha256": modal_record["artifact_sha256"],
        },
    }
    (FIGURES / "step6" / "evaluation.json").write_text(
        json.dumps(outcome, indent=2) + "\n"
    )
    return outcome


def write_frozen_evaluation_protocol() -> Path:
    """Freeze all choices and hashes before any official test-set access."""
    step5_path = FIGURES / "step5_outcome.json"
    if not step5_path.exists():
        raise RuntimeError("Step 5 must be finalized before freezing evaluation")
    step5 = json.loads(step5_path.read_text())
    models = [
        {
            "probe_uS": record["probe_uS"],
            "seed": record["seed"],
            "path": record["checkpoint_path"],
            "sha256": record["checkpoint_sha256"],
            "selected_nonlinear_epoch": record["selected_nonlinear_epoch"],
            "selected_linear_epoch": record["selected_linear_epoch"],
            "linear_weight_decay": record["selected_linear_weight_decay"],
        }
        for record in step5["records"]
    ]
    protocol = {
        "status": "frozen_before_test_access",
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": _git_metadata(),
        "step5_outcome_path": str(step5_path.relative_to(REPO)),
        "step5_outcome_sha256": sha256_file(step5_path),
        "models": models,
        "rate_grid_hz": list(DECODER_RATES_HZ),
        "probe_conductances_uS": list(PROBE_CONDUCTANCES_US),
        "decoder_seeds": list(SEEDS),
        "direct_simulation_draws": EVALUATION_DRAWS,
        "direct_simulation_seed_recipe": (
            "sha256-derived 63-bit seed of [77,6,probe_index,rate_index,draw,batch_start]"
        ),
        "same_held_out_images_for_every_decoder_seed": True,
        "bootstrap": {
            "repetitions": BOOTSTRAP_REPETITIONS_HELDOUT,
            "confidence_level": CONFIDENCE_LEVEL,
            "bound": "one-sided lower percentile",
            "independently_resampled_axes": [
                "held-out images",
                "direct-simulation draws",
                "decoder seeds",
            ],
        },
        "criteria": {
            "chance_accuracy": CHANCE_ACCURACY,
            "useful_accuracy": USEFUL_ACCURACY,
            "r_decode": "lowest observed rate with lower bound strictly above chance",
            "r_train": "lowest observed rate with lower bound at least useful accuracy",
            "interpolation_is_observation": False,
        },
        "dataset": {
            "training_indices": [0, 54_999],
            "validation_indices": [55_000, 59_999],
            "official_test_size": 10_000,
            "official_test_loaded_before_freeze": False,
        },
    }
    path = FIGURES / "frozen_evaluation_protocol.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    return path


def load_held_out_mnist_test(
    protocol_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the official test partition only after authenticating the freeze file."""
    if not protocol_path.exists():
        raise RuntimeError("held-out test access requires a frozen evaluation protocol")
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("status") != "frozen_before_test_access":
        raise RuntimeError("evaluation protocol is not frozen")
    for model in protocol["models"]:
        path = REPO / model["path"]
        if not path.exists() or sha256_file(path) != model["sha256"]:
            raise RuntimeError(f"frozen decoder hash mismatch: {path}")
    from torchvision.datasets import MNIST

    dataset = MNIST(root="/tmp/mnist", train=False, download=True)
    images = dataset.data.numpy().astype(np.uint8, copy=False)
    labels = dataset.targets.numpy().astype(np.int64, copy=False)
    if images.shape != (10_000, 28, 28) or labels.shape != (10_000,):
        raise RuntimeError(f"unexpected MNIST held-out contract: {images.shape}")
    raw_hashes = {
        path.name: sha256_file(path)
        for path in sorted(Path(dataset.raw_folder).glob("t10k-*-ubyte"))
    }
    return (
        images,
        labels,
        {
            "source": "torchvision.datasets.MNIST official held-out test partition",
            "image_shape": list(images.shape),
            "label_shape": list(labels.shape),
            "raw_file_sha256": raw_hashes,
        },
    )


def _load_frozen_models(
    protocol: dict[str, Any], device: Any
) -> dict[tuple[float, int], tuple[Any, Any]]:
    import torch

    models: dict[tuple[float, int], tuple[Any, Any]] = {}
    for record in protocol["models"]:
        checkpoint = torch.load(
            REPO / record["path"], map_location=device, weights_only=True
        )
        nonlinear, linear_candidates = _make_decoders(device, record["seed"])
        nonlinear.load_state_dict(checkpoint["nonlinear"])
        linear = linear_candidates[0]
        linear.load_state_dict(checkpoint["linear"])
        nonlinear.eval()
        linear.eval()
        models[(float(record["probe_uS"]), int(record["seed"]))] = (nonlinear, linear)
    return models


def evaluate_frozen_decoders(protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    """Evaluate frozen models with shared fresh direct draws on held-out MNIST."""
    import torch

    protocol = json.loads(protocol_path.read_text())
    images, labels, dataset = load_held_out_mnist_test(protocol_path)
    device = _torch_device()
    models = _load_frozen_models(protocol, device)
    nonlinear_correct = np.empty(
        (
            len(PROBE_CONDUCTANCES_US),
            len(DECODER_RATES_HZ),
            EVALUATION_DRAWS,
            len(SEEDS),
            len(labels),
        ),
        dtype=np.bool_,
    )
    linear_correct = np.empty(
        (len(DECODER_RATES_HZ), EVALUATION_DRAWS, len(SEEDS), len(labels)),
        dtype=np.bool_,
    )
    started = time.perf_counter()
    label_tensor = torch.as_tensor(labels, device=device)
    for probe_index, probe in enumerate(PROBE_CONDUCTANCES_US):
        for rate_index, rate in enumerate(DECODER_RATES_HZ):
            for draw in range(EVALUATION_DRAWS):
                for start in range(0, len(labels), DECODER_BATCH_SIZE):
                    stop = min(start + DECODER_BATCH_SIZE, len(labels))
                    generator = torch.Generator(device=device).manual_seed(
                        _decoder_seed(6, probe_index, rate_index, draw, start)
                    )
                    batch_images = torch.as_tensor(images[start:stop], device=device)
                    batch_rates = torch.full((stop - start,), rate, device=device)
                    features = direct_feature_batch_torch(
                        batch_images, batch_rates, probe, generator
                    )
                    for seed_index, seed in enumerate(SEEDS):
                        nonlinear, linear = models[(probe, seed)]
                        with torch.no_grad():
                            nonlinear_prediction = nonlinear(features).argmax(dim=1)
                            nonlinear_correct[
                                probe_index, rate_index, draw, seed_index, start:stop
                            ] = (
                                (nonlinear_prediction == label_tensor[start:stop])
                                .cpu()
                                .numpy()
                            )
                            if probe == PROBE_US:
                                linear_prediction = linear(features).argmax(dim=1)
                                linear_correct[
                                    rate_index, draw, seed_index, start:stop
                                ] = (
                                    (linear_prediction == label_tensor[start:stop])
                                    .cpu()
                                    .numpy()
                                )
                print(
                    f"held-out probe={probe:g} rate={rate:g} draw={draw + 1}/{EVALUATION_DRAWS}",
                    flush=True,
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "held_out_correctness.npz"
    np.savez_compressed(
        arrays_path,
        nonlinear_correct=nonlinear_correct,
        linear_correct=linear_correct,
        rates_hz=np.asarray(DECODER_RATES_HZ),
        probes_uS=np.asarray(PROBE_CONDUCTANCES_US),
        decoder_seeds=np.asarray(SEEDS),
        labels=labels,
    )
    outcome = {
        "status": "complete",
        "protocol_sha256": sha256_file(protocol_path),
        "dataset": dataset,
        "device": str(device),
        "runtime_s": time.perf_counter() - started,
        "arrays_path": str(arrays_path.relative_to(REPO)),
        "arrays_sha256": sha256_file(arrays_path),
    }
    (output_dir / "evaluation.json").write_text(json.dumps(outcome, indent=2) + "\n")
    return outcome


def _hierarchical_lower_bound(
    values: np.ndarray, seed: int
) -> tuple[float, float, float]:
    """Bootstrap images, direct draws, and decoder seeds independently."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("correctness array must have draw x seed x image axes")
    rng = np.random.default_rng(seed)
    estimates = np.empty(BOOTSTRAP_REPETITIONS_HELDOUT, dtype=np.float64)
    for repetition in range(BOOTSTRAP_REPETITIONS_HELDOUT):
        draws = rng.integers(0, array.shape[0], array.shape[0])
        seeds = rng.integers(0, array.shape[1], array.shape[1])
        image_indices = rng.integers(0, array.shape[2], array.shape[2])
        selected = array[np.ix_(draws, seeds, image_indices)]
        estimates[repetition] = np.mean(selected)
    alpha = 1.0 - CONFIDENCE_LEVEL
    return (
        float(np.mean(array)),
        float(np.quantile(estimates, alpha)),
        float(np.quantile(estimates, 1.0 - alpha)),
    )


def analyze_held_out_evaluation() -> dict[str, Any]:
    """Calculate frozen intervals, thresholds, and the concise psychometric plot."""
    arrays_path = FIGURES / "step6" / "held_out_correctness.npz"
    with np.load(arrays_path) as arrays:
        nonlinear_correct = arrays["nonlinear_correct"]
        linear_correct = arrays["linear_correct"]
    nonlinear_rows: list[dict[str, Any]] = []
    for probe_index, probe in enumerate(PROBE_CONDUCTANCES_US):
        for rate_index, rate in enumerate(DECODER_RATES_HZ):
            mean, lower, upper = _hierarchical_lower_bound(
                nonlinear_correct[probe_index, rate_index],
                _decoder_seed(7, probe_index, rate_index, 1),
            )
            nonlinear_rows.append(
                {
                    "probe_uS": probe,
                    "rate_hz": rate,
                    "accuracy": mean,
                    "lower_95_one_sided": lower,
                    "upper_95": upper,
                }
            )
    linear_rows: list[dict[str, Any]] = []
    for rate_index, rate in enumerate(DECODER_RATES_HZ):
        mean, lower, upper = _hierarchical_lower_bound(
            linear_correct[rate_index], _decoder_seed(7, rate_index, 2)
        )
        linear_rows.append(
            {
                "probe_uS": PROBE_US,
                "rate_hz": rate,
                "accuracy": mean,
                "lower_95_one_sided": lower,
                "upper_95": upper,
            }
        )

    def threshold(
        rows: list[dict[str, Any]], criterion: float, strict: bool
    ) -> float | None:
        for row in rows:
            lower = float(row["lower_95_one_sided"])
            if (lower > criterion) if strict else (lower >= criterion):
                return float(row["rate_hz"])
        return None

    thresholds: dict[str, Any] = {}
    for probe in PROBE_CONDUCTANCES_US:
        rows = [row for row in nonlinear_rows if row["probe_uS"] == probe]
        thresholds[str(probe)] = {
            "r_decode_hz": threshold(rows, CHANCE_ACCURACY, True),
            "r_train_hz": threshold(rows, USEFUL_ACCURACY, False),
        }
    thresholds["linear_1.2"] = {
        "r_decode_hz": threshold(linear_rows, CHANCE_ACCURACY, True),
        "r_train_hz": threshold(linear_rows, USEFUL_ACCURACY, False),
    }
    theme.apply()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.25), constrained_layout=True)
    colors = {0.6: "#4477AA", 1.2: "#222222", 2.4: "#CC6677"}
    nominal = [row for row in nonlinear_rows if row["probe_uS"] == PROBE_US]
    for rows, label, color, linestyle in (
        (nominal, "Nonlinear", "#222222", "-"),
        (linear_rows, "Linear", "#999999", "--"),
    ):
        rates = np.asarray([row["rate_hz"] for row in rows])
        mean = np.asarray([row["accuracy"] for row in rows])
        low = np.asarray([row["lower_95_one_sided"] for row in rows])
        high = np.asarray([row["upper_95"] for row in rows])
        axes[0].plot(rates, mean, label=label, color=color, linestyle=linestyle)
        axes[0].fill_between(rates, low, high, color=color, alpha=0.15, linewidth=0)
    for probe in PROBE_CONDUCTANCES_US:
        rows = [row for row in nonlinear_rows if row["probe_uS"] == probe]
        rates = np.asarray([row["rate_hz"] for row in rows])
        mean = np.asarray([row["accuracy"] for row in rows])
        low = np.asarray([row["lower_95_one_sided"] for row in rows])
        high = np.asarray([row["upper_95"] for row in rows])
        axes[1].plot(rates, mean, label=f"{probe:g} μS", color=colors[probe])
        axes[1].fill_between(
            rates, low, high, color=colors[probe], alpha=0.15, linewidth=0
        )
    for axis, title in zip(
        axes, ("Decoder comparison at 1.2 μS", "Conductance sensitivity"), strict=True
    ):
        axis.axhline(CHANCE_ACCURACY, color="#777777", linewidth=0.8, linestyle=":")
        axis.axhline(USEFUL_ACCURACY, color="#777777", linewidth=0.8, linestyle="--")
        axis.set_xscale("log")
        display_ticks = (0.01, 0.1, 0.5, 1.0, 5.0, 25.0)
        axis.set_xticks(display_ticks)
        axis.set_xticklabels([f"{rate:g}" for rate in display_ticks])
        axis.set_ylim(0, 1)
        axis.set_xlabel("Encoding rate (Hz)")
        axis.set_ylabel("Held-out accuracy")
        axis.set_title(title)
        axis.legend(frameon=False)
    figure_path = FIGURES / "psychometric.svg"
    savefig_atomic(fig, figure_path, bbox_inches="tight")
    plt.close(fig)
    outcome = {
        "status": "complete",
        "confidence": {
            "level": CONFIDENCE_LEVEL,
            "side": "one-sided lower percentile bound",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS_HELDOUT,
            "resampled_axes": [
                "held-out images",
                "direct-simulation draws",
                "decoder seeds",
            ],
        },
        "nonlinear": nonlinear_rows,
        "linear_nominal": linear_rows,
        "thresholds": thresholds,
        "figure_path": str(figure_path.relative_to(REPO)),
        "figure_sha256": sha256_file(figure_path),
        "arrays_sha256": sha256_file(arrays_path),
    }
    (FIGURES / "step6_outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")
    return outcome


def step_6() -> None:
    protocol_path = FIGURES / "frozen_evaluation_protocol.json"
    evaluate_frozen_decoders(protocol_path, FIGURES / "step6")


def step_7() -> None:
    step6_path = FIGURES / "step6_outcome.json"
    outcome = (
        json.loads(step6_path.read_text())
        if step6_path.exists()
        else analyze_held_out_evaluation()
    )
    nominal = outcome["thresholds"][str(PROBE_US)]
    floors = {
        str(probe): outcome["thresholds"][str(probe)]["r_train_hz"]
        for probe in PROBE_CONDUCTANCES_US
    }
    observed_floors = [float(value) for value in floors.values() if value is not None]
    floor_indices = [DECODER_RATES_HZ.index(value) for value in observed_floors]
    sensitive = bool(floor_indices and max(floor_indices) - min(floor_indices) > 1)
    recommendation = (
        {
            "form": "plausible conductance-sensitive floor range",
            "floor_range_hz": [min(observed_floors), max(observed_floors)],
            "ceiling_hz": 25.0,
        }
        if sensitive and observed_floors
        else {
            "form": "nominal floor to registered ceiling",
            "floor_hz": nominal["r_train_hz"],
            "ceiling_hz": 25.0,
        }
    )
    step5 = json.loads((FIGURES / "step5_outcome.json").read_text())
    modal_evaluation = json.loads((FIGURES / "step6" / "modal.json").read_text())
    estimated_total_cost = float(step5["cumulative_estimated_cost_usd"]) + float(
        modal_evaluation["estimated_cost_usd"]
    )
    billing_path = FIGURES / "modal_billing.json"
    if not billing_path.exists():
        raise RuntimeError("provider billing report must be captured before Step 7")
    billing_rows = json.loads(billing_path.read_text())
    relevant_billing = [
        row
        for row in billing_rows
        if row.get("description") in {"pinglab-exp080", "pinglab-exp080-evaluation"}
    ]
    exact_total_cost = sum(float(row["cost"]) for row in relevant_billing)
    protocol_path = FIGURES / "frozen_evaluation_protocol.json"
    decision = {
        "status": "complete",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "r_decode_hz": nominal["r_decode_hz"],
        "r_train_hz": nominal["r_train_hz"],
        "conductance_floors_hz": floors,
        "conductance_sensitive_by_registered_rule": sensitive,
        "recommendation": recommendation,
        "uncertainty": outcome["confidence"],
        "criteria": {
            "chance_accuracy": CHANCE_ACCURACY,
            "useful_accuracy": USEFUL_ACCURACY,
        },
        "pass_fail": {
            "nominal_r_decode_observed": nominal["r_decode_hz"] is not None,
            "nominal_r_train_observed": nominal["r_train_hz"] is not None,
            "all_conductance_r_train_observed": all(
                value is not None for value in floors.values()
            ),
            "spend_within_40_usd_ceiling": exact_total_cost <= 40.0,
            "held_out_protocol_frozen_before_access": True,
        },
        "hashes": {
            "frozen_protocol_sha256": sha256_file(protocol_path),
            "step5_outcome_sha256": sha256_file(FIGURES / "step5_outcome.json"),
            "step6_outcome_sha256": sha256_file(FIGURES / "step6_outcome.json"),
            "held_out_arrays_sha256": outcome["arrays_sha256"],
            "psychometric_figure_sha256": outcome["figure_sha256"],
            "models": [
                {
                    "probe_uS": model["probe_uS"],
                    "seed": model["seed"],
                    "sha256": model["sha256"],
                }
                for model in json.loads(protocol_path.read_text())["models"]
            ],
        },
        "compute": {
            "provider": "Modal",
            "total_exact_cost_usd": exact_total_cost,
            "total_estimated_cost_usd": estimated_total_cost,
            "exact_provider_billing": True,
            "billing_report_path": str(billing_path.relative_to(REPO)),
            "billing_report_sha256": sha256_file(billing_path),
            "resource_rows": relevant_billing,
        },
        "scope": "decoder-relative thresholds only; no absolute information limit or PING accuracy claim",
    }
    (FIGURES / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")


def capture_modal_billing() -> Path:
    """Capture the provider's exact per-app resource charges for this experiment."""
    modal_executable = Path(sys.executable).with_name("modal")
    completed = subprocess.run(
        [
            str(modal_executable),
            "billing",
            "report",
            "--for",
            "today",
            "--resolution",
            "h",
            "--show-resources",
            "--json",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = json.loads(completed.stdout)
    relevant = [
        row
        for row in rows
        if row.get("description") in {"pinglab-exp080", "pinglab-exp080-evaluation"}
    ]
    if not relevant:
        raise RuntimeError("Modal billing report contained no exp080 rows")
    path = FIGURES / "modal_billing.json"
    path.write_text(json.dumps(relevant, indent=2) + "\n")
    return path


def record_expanded_rate_outcome() -> Path:
    """Summarize the completed sparse-rate extension from recorded artifacts."""
    step5 = json.loads((FIGURES / "step5_outcome.json").read_text())
    step6 = json.loads((FIGURES / "step6_outcome.json").read_text())
    decision = json.loads((FIGURES / "decision.json").read_text())
    billing = json.loads((FIGURES / "modal_billing.json").read_text())
    total_exact = sum(float(row["cost"]) for row in billing)
    nominal_rows = [
        row
        for row in step6["nonlinear"]
        if row["probe_uS"] == PROBE_US and row["rate_hz"] in (0.01, 0.05, 0.1)
    ]
    outcome = {
        "status": "complete",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_protocol_path": "artifacts/data/exp080/expanded_rate_training_protocol.json",
        "training_protocol_sha256": sha256_file(
            FIGURES / "expanded_rate_training_protocol.json"
        ),
        "added_rates_hz": [0.01, 0.05, 0.1],
        "complete_decoder_rate_grid_hz": list(DECODER_RATES_HZ),
        "equal_training_probability_per_rate": 1.0 / len(DECODER_RATES_HZ),
        "training": {
            "cell_count": len(step5["records"]),
            "parallel_wall_time_s": max(
                float(
                    json.loads(
                        (REPO / record["training_path"])
                        .with_name("modal.json")
                        .read_text()
                    )["parallel_dispatch_elapsed_s"]
                )
                for record in step5["records"]
            ),
            "aggregate_gpu_time_s": sum(
                float(record["modal_elapsed_s"]) for record in step5["records"]
            ),
            "selected_nonlinear_validation_accuracy_range": [
                min(
                    float(record["selected_nonlinear_validation_accuracy"])
                    for record in step5["records"]
                ),
                max(
                    float(record["selected_nonlinear_validation_accuracy"])
                    for record in step5["records"]
                ),
            ],
        },
        "held_out_evaluation": {
            "runtime_s": json.loads(
                (FIGURES / "step6" / "evaluation.json").read_text()
            )["runtime_s"],
            "nominal_added_rate_rows": nominal_rows,
            "r_decode_hz": decision["r_decode_hz"],
            "r_train_hz": decision["r_train_hz"],
            "conductance_floors_hz": decision["conductance_floors_hz"],
        },
        "compute": {
            "provider": "Modal",
            "pre_expansion_exact_cost_usd": PRE_EXPANSION_EXACT_MODAL_COST_USD,
            "cumulative_exact_cost_usd": total_exact,
            "expansion_exact_cost_usd": total_exact
            - PRE_EXPANSION_EXACT_MODAL_COST_USD,
            "provider_billing_path": "artifacts/data/exp080/modal_billing.json",
            "provider_billing_sha256": sha256_file(FIGURES / "modal_billing.json"),
        },
        "preserved_infrastructure_failures_before_training": 3,
        "scope": "decoder-relative exploratory extension; no PING accuracy claim",
    }
    path = FIGURES / "expanded_rate_outcome.json"
    path.write_text(json.dumps(outcome, indent=2) + "\n")
    return path


def record_steps5_7_publication_contract() -> None:
    """Extend cumulative metadata with the completed frozen decoder study."""
    step5_path = FIGURES / "step5_outcome.json"
    step6_path = FIGURES / "step6_outcome.json"
    decision_path = FIGURES / "decision.json"
    freeze_path = FIGURES / "frozen_evaluation_protocol.json"
    step5 = json.loads(step5_path.read_text())
    step6 = json.loads(step6_path.read_text())
    decision = json.loads(decision_path.read_text())
    expanded_path = FIGURES / "expanded_rate_outcome.json"
    expanded = json.loads(expanded_path.read_text()) if expanded_path.exists() else None

    numbers_path = FIGURES / "numbers.json"
    numbers = json.loads(numbers_path.read_text())
    numbers.update(
        {
            "step": 7,
            "status": "complete",
            "scope": "filter-matched decoder-relative MNIST rate thresholds",
            "step5": step5,
            "step6": step6,
            "step7": decision,
            "later_steps_run": True,
            "paid_compute_usd": decision["compute"]["total_exact_cost_usd"],
            "expanded_rate_extension": expanded,
        }
    )
    numbers_path.write_text(json.dumps(numbers, indent=2) + "\n")

    protocol_path = FIGURES / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol.update(
        {
            "attempted_through_step": 7,
            "steps5_7_classification": "explicitly authorized continuation with Step 4 failure preserved",
            "step5_status": step5["status"],
            "step6_status": step6["status"],
            "step7_status": decision["status"],
            "held_out_test_partition": "accessed only after committed frozen protocol",
            "frozen_evaluation_protocol_sha256": sha256_file(freeze_path),
            "expanded_rate_training_protocol_sha256": EXPANDED_RATE_PROTOCOL_SHA256,
        }
    )
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")

    manifest_path = FIGURES / "step2_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["exploratory_continuation"]["steps5_7_run"] = True
    manifest["exploratory_continuation"].update(
        {
            "step5_outcome": str(step5_path.relative_to(REPO)),
            "step5_outcome_sha256": sha256_file(step5_path),
            "frozen_evaluation_protocol": str(freeze_path.relative_to(REPO)),
            "frozen_evaluation_protocol_sha256": sha256_file(freeze_path),
            "step6_outcome": str(step6_path.relative_to(REPO)),
            "step6_outcome_sha256": sha256_file(step6_path),
            "decision": str(decision_path.relative_to(REPO)),
            "decision_sha256": sha256_file(decision_path),
            "expanded_rate_outcome": str(expanded_path.relative_to(REPO)),
            "expanded_rate_outcome_sha256": sha256_file(expanded_path),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    provenance_path = FIGURES / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["steps5_7"] = {
        **_git_metadata(),
        "step5_outcome_sha256": sha256_file(step5_path),
        "frozen_evaluation_protocol_sha256": sha256_file(freeze_path),
        "held_out_correctness_sha256": sha256_file(
            FIGURES / "step6" / "held_out_correctness.npz"
        ),
        "step6_outcome_sha256": sha256_file(step6_path),
        "decision_sha256": sha256_file(decision_path),
        "paid_compute_usd": decision["compute"]["total_exact_cost_usd"],
        "expanded_rate_outcome_sha256": sha256_file(expanded_path),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    reproducer_path = FIGURES / "reproducer.json"
    reproducer = json.loads(reproducer_path.read_text())
    reproducer["steps5_7"] = {
        "smoke_command": "EXP080_STAGE=smoke uv run python -c 'from experiments.exp080 import step_5; step_5()'",
        "modal_training_command": "uv run python experiments/exp080_modal.py --stage full --probe <probe> --seed <seed> --live",
        "freeze_command": "uv run python -c 'from experiments.exp080 import write_frozen_evaluation_protocol; write_frozen_evaluation_protocol()'",
        "modal_evaluation_command": "uv run python experiments/exp080_evaluate_modal.py --live",
        "analysis_command": "uv run python -c 'from experiments.exp080 import analyze_held_out_evaluation, step_7; analyze_held_out_evaluation(); step_7()'",
        "checkpoint_count": len(step5["records"]),
        "expected_decision_sha256": sha256_file(decision_path),
        "paid_compute": True,
        "expanded_rate_outcome_command": "uv run python -c 'from experiments.exp080 import record_expanded_rate_outcome; record_expanded_rate_outcome()'",
    }
    reproducer_path.write_text(json.dumps(reproducer, indent=2) + "\n")


STAGE_FUNCTIONS: dict[int, Callable[[], None]] = {
    1: step_1,
    2: step_2,
    3: step_3,
    4: step_4,
    5: step_5,
    6: step_6,
    7: step_7,
}

IMPLEMENTED_STEPS: frozenset[int] = frozenset({1, 2, 3, 4, 5, 6, 7})


def requested_through_step() -> int:
    raw = os.environ.get("EXP080_THROUGH_STEP", str(N_STEPS))
    try:
        step = int(raw)
    except ValueError as exc:
        raise SystemExit(
            "EXP080_THROUGH_STEP must be an integer from 1 through 7"
        ) from exc
    if step not in STAGE_NAMES:
        raise SystemExit("EXP080_THROUGH_STEP must be an integer from 1 through 7")
    return step


def main() -> None:
    through_step = requested_through_step()
    for step in range(1, through_step + 1):
        if step not in IMPLEMENTED_STEPS:
            _not_implemented(step)
        STAGE_FUNCTIONS[step]()


if __name__ == "__main__":
    main()
