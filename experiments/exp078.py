"""Experiment 078 — graph-native Arnold tongue of two coupled PING circuits.

The runner is deliberately gated and resumable.  Calibration, tolerance
registration and the coupling pilot are completed and frozen before any
primary-grid result is inspected.  Dense neuron state is transient for the
primary grid; exact input/population spikes and decimated state summaries are
kept for every cell, while neuron-level voltage/conductance is retained for the
predeclared representative traces.  This avoids a scientifically useless
multi-gigabyte dense-state archive without weakening the locking analysis.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath
from scipy import signal as sp_signal
from scipy.ndimage import gaussian_filter1d

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import snnlang as snn  # noqa: E402, TID251

from helpers import modal_backend, theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp078"
FOLLOWUP_MODE = os.environ.get("EXP078_FOLLOWUP_MODE", "")
DT_MS = 0.1
T_MS = 5_500.0 if FOLLOWUP_MODE == "panel" else 3_000.0
BURN_MS = 500.0
N_INPUT = 800 if FOLLOWUP_MODE else 80
N_E = 800 if FOLLOWUP_MODE else 80
N_I = 200 if FOLLOWUP_MODE else 20
INPUT_WEIGHT = 0.2
COUPLING_REFERENCE = 1.0
DELAY_MS = 0.1
NETWORK_SEED = 78_000
TRIALS = 10 if FOLLOWUP_MODE == "panel" else (1 if FOLLOWUP_MODE == "benchmark" else 5)
RATE_GRID_HZ = (60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0)
TARGET_DETUNINGS_HZ = (-6.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
PILOT_COUPLINGS = (0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.125)
PRIMARY_NONZERO_LEVELS = 10
SMOOTH_SIGMA_MS = 5.0
BAND_HZ = (25.0, 90.0)
PEAK_BAND_HZ = (25.0, 80.0)
STATE_DECIMATE_MS = 1.0
FOLLOWUP_JOBS = {
    f"{'m1' if detuning < 0 else 'p1'}_k{int(round(coupling * 1000)):03d}": {
        "detuning_index": 4 if detuning < 0 else 8,
        "target_detuning_hz": detuning,
        "rate_a_hz": 93.5 if detuning < 0 else 98.5,
        "rate_b_hz": 98.5 if detuning < 0 else 93.5,
        "coupling": coupling,
    }
    for detuning in (-1.0, 1.0)
    for coupling in (0.0, 0.016, 0.024)
}
FOLLOWUP_JOBS["benchmark"] = {
    "detuning_index": 4,
    "target_detuning_hz": -1.0,
    "rate_a_hz": 93.5,
    "rate_b_hz": 98.5,
    "coupling": 0.016,
}
FOLLOWUP_SCRATCH = Path(os.environ.get(
    "PINGLAB_ARTIFACTS_ROOT", str(REPO / "temp" / "experiments" / "exp078-followup")
))

SCALE = {
    "stage": "complete gated Arnold-tongue sweep",
    "dt_ms": DT_MS,
    "t_ms": T_MS,
    "burn_ms": BURN_MS,
    "n_input_per_circuit": N_INPUT,
    "n_e_per_circuit": N_E,
    "n_i_per_circuit": N_I,
    "calibration_rates_hz": list(RATE_GRID_HZ),
    "target_detunings_hz": list(TARGET_DETUNINGS_HZ),
    "trials_per_cell": TRIALS,
    "primary_coupling_levels": PRIMARY_NONZERO_LEVELS + 1,
    "execution_host": "local",
}

POPULATION_KEYS = {
    "a_e": "population_0",
    "a_i": "population_1",
    "b_e": "population_2",
    "b_i": "population_3",
}


@dataclass(frozen=True)
class TrialMetrics:
    valid: bool
    invalid_reasons: tuple[str, ...]
    rate_a_e_hz: float
    rate_a_i_hz: float
    rate_b_e_hz: float
    rate_b_i_hz: float
    frequency_a_hz: float
    frequency_b_hz: float
    frequency_difference_hz: float
    phase_slope_rad_s: float
    phase_slips: int
    phase_locking_value: float
    circular_mean_phase_rad: float

    def as_dict(self) -> dict:
        row = dict(self.__dict__)
        row["invalid_reasons"] = list(self.invalid_reasons)
        return row


def author_network(*, coupling: float = COUPLING_REFERENCE) -> snn.Bundle:
    """Compile the fixed Lowet-style two-circuit topology."""
    net = snn.Network("lowet_two_ping", dt=DT_MS * snn.ms)
    drives = {
        name: net.input(
            f"drive_{name}",
            shape=("time", "batch", N_INPUT),
            signal_type="spikes",
            unit="spike",
        )
        for name in ("a", "b")
    }
    circuits = {}
    for name in ("a", "b"):
        circuit = snn.components.ping(
            net,
            name=name,
            n_e=N_E,
            n_i=N_I,
            source=None,
        )
        net.connect(
            drives[name],
            circuit.E.excitatory,
            name=f"drive_{name}_to_{name}_E",
            synapse=snn.AMPA(tau=2 * snn.ms),
            weight=snn.Constant(INPUT_WEIGHT),
            constraint=snn.NonNegative(),
            delay=DELAY_MS * snn.ms,
        )
        circuits[name] = circuit

    for source, target in (("a", "b"), ("b", "a")):
        net.connect(
            circuits[source].E.spikes,
            circuits[target].E.excitatory,
            name=f"{source}_E_to_{target}_E",
            synapse=snn.AMPA(tau=2 * snn.ms),
            weight=snn.Constant(coupling),
            constraint=snn.NonNegative(),
            connection="feedback",
            delay=DELAY_MS * snn.ms,
        )
        net.connect(
            circuits[source].E.spikes,
            circuits[target].I.excitatory,
            name=f"{source}_E_to_{target}_I",
            synapse=snn.AMPA(tau=2 * snn.ms),
            weight=snn.Constant(coupling),
            constraint=snn.NonNegative(),
            connection="feedback",
            delay=DELAY_MS * snn.ms,
        )

    net.expose(
        circuits["a"].E.spikes,
        circuits["a"].I.spikes,
        circuits["b"].E.spikes,
        circuits["b"].I.spikes,
        name="population",
    )
    return snn.compile(net, target="tools/snn")


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _json_load(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256_bytes(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


def _float_tag(value: float) -> str:
    return f"{value:+08.3f}".replace("+", "p").replace("-", "m").replace(".", "d")


def _condition_key(stage: str, detuning_index: int, coupling: float) -> str:
    return f"{stage}_d{detuning_index:02d}_k{_float_tag(coupling)}"


def _trial_seed(detuning_index: int, trial: int, circuit: int) -> int:
    """Stable input seed, paired across every coupling at one detuning."""
    return 78_000_000 + 10_000 * detuning_index + 10 * trial + circuit


def make_inputs(
    rate_a_hz: float,
    rate_b_hz: float,
    *,
    detuning_index: int,
    trials: int = TRIALS,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Generate private Poisson streams and their complete seed ledger."""
    n_steps = round(T_MS / DT_MS)
    values: dict[str, list[np.ndarray]] = {"drive_a": [], "drive_b": []}
    ledger: list[dict] = []
    for trial in range(trials):
        row = {"detuning_index": detuning_index, "trial": trial}
        for circuit, (name, rate_hz) in enumerate(
            (("drive_a", rate_a_hz), ("drive_b", rate_b_hz))
        ):
            seed = _trial_seed(detuning_index, trial, circuit)
            rng = np.random.default_rng(seed)
            p_step = rate_hz * DT_MS / 1000.0
            spikes = (rng.random((n_steps, N_INPUT), dtype=np.float32) < p_step)
            values[name].append(spikes)
            row[f"{name}_seed"] = seed
            row[f"{name}_rate_hz"] = rate_hz
        ledger.append(row)
    return {
        name: np.stack(rows, axis=1).astype(np.uint8)
        for name, rows in values.items()
    }, ledger


def _bundle_dir(root: Path, coupling: float) -> Path:
    path = root / "bundles" / f"k_{_float_tag(coupling)}.bundle"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        author_network(coupling=coupling).write(path, visualise=False)
    return path


def _run_graph_cli(
    bundle_dir: Path,
    input_path: Path,
    output_dir: Path,
) -> float:
    cmd = [
        sys.executable,
        str(REPO / "tools/snn/tool.py"),
        "sim",
        "--bundle",
        str(bundle_dir),
        "--executor",
        "graph",
        "--device",
        "auto",
        "--recording",
        "observables" if FOLLOWUP_MODE else "full",
        "--input-file",
        str(input_path),
        "--t-ms",
        str(T_MS),
        "--n-batch",
        str(TRIALS),
        "--seed",
        str(NETWORK_SEED),
        "--out-dir",
        str(output_dir),
    ]
    env = dict(os.environ)
    env.setdefault("PINGLAB_NO_COMPILE", "1")
    started = time.monotonic()
    subprocess.run(cmd, cwd=REPO, env=env, check=True)
    return time.monotonic() - started


def _population_rate(spikes: np.ndarray, n_cells: int) -> np.ndarray:
    rate_hz = spikes.sum(axis=-1) / n_cells * (1000.0 / DT_MS)
    return gaussian_filter1d(rate_hz.astype(np.float64), SMOOTH_SIGMA_MS / DT_MS)


def _peak_frequency(rate_hz: np.ndarray) -> tuple[float, bool]:
    frequencies, power = sp_signal.welch(
        rate_hz,
        fs=1000.0 / DT_MS,
        nperseg=len(rate_hz),
        detrend="constant",
        scaling="density",
    )
    mask = (frequencies >= PEAK_BAND_HZ[0]) & (frequencies <= PEAK_BAND_HZ[1])
    if not np.any(mask) or not np.all(np.isfinite(power[mask])):
        return float("nan"), False
    band_power = power[mask]
    peak_index = int(np.argmax(band_power))
    peak = float(frequencies[mask][peak_index])
    # A numerical maximum always exists.  Require it to rise above the band
    # median so featureless low-rate noise is not called a gamma oscillation.
    resolved = bool(band_power[peak_index] > np.median(band_power) * 1.25)
    return peak, resolved


def _phase_metrics(rate_a: np.ndarray, rate_b: np.ndarray) -> tuple[float, int, float, float, np.ndarray]:
    sample_hz = 1000.0 / DT_MS
    sos = sp_signal.butter(4, BAND_HZ, btype="bandpass", fs=sample_hz, output="sos")
    filtered_a = sp_signal.sosfiltfilt(sos, rate_a)
    filtered_b = sp_signal.sosfiltfilt(sos, rate_b)
    wrapped = np.angle(sp_signal.hilbert(filtered_a)) - np.angle(sp_signal.hilbert(filtered_b))
    wrapped = np.angle(np.exp(1j * wrapped))
    unwrapped = np.unwrap(wrapped)
    seconds = np.arange(len(unwrapped), dtype=np.float64) * DT_MS / 1000.0
    slope = float(np.polyfit(seconds, unwrapped, 1)[0])
    slips = int(np.floor(abs(unwrapped[-1] - unwrapped[0]) / (2.0 * np.pi)))
    plv = float(abs(np.mean(np.exp(1j * wrapped))))
    circular_mean = float(np.angle(np.mean(np.exp(1j * wrapped))))
    return slope, slips, plv, circular_mean, unwrapped


def analyse_recordings(path: Path) -> tuple[list[TrialMetrics], dict[str, np.ndarray]]:
    burn = round(BURN_MS / DT_MS)
    metrics: list[TrialMetrics] = []
    trace_cache: dict[str, list[np.ndarray]] = {
        "rate_a": [], "rate_b": [], "phase": [], "frequency_difference": []
    }
    with np.load(path) as recordings:
        arrays = {name: recordings[key] for name, key in POPULATION_KEYS.items()}
        all_recorded_state_finite = all(
            np.all(np.isfinite(recordings[key])) for key in recordings.files
        )
        for trial in range(arrays["a_e"].shape[1]):
            finite = all_recorded_state_finite and all(
                np.all(np.isfinite(array[:, trial])) for array in arrays.values()
            )
            rates = {
                name: float(array[burn:, trial].mean() * (1000.0 / DT_MS))
                for name, array in arrays.items()
            }
            rate_a = _population_rate(arrays["a_e"][:, trial], N_E)[burn:]
            rate_b = _population_rate(arrays["b_e"][:, trial], N_E)[burn:]
            frequency_a, peak_a = _peak_frequency(rate_a)
            frequency_b, peak_b = _peak_frequency(rate_b)
            slope, slips, plv, circular_mean, phase = _phase_metrics(rate_a, rate_b)
            reasons: list[str] = []
            if not finite:
                reasons.append("non-finite recorded state")
            for name, value in rates.items():
                if value < 1.0:
                    reasons.append(f"{name} firing rate below 1 Hz")
            if not peak_a:
                reasons.append("circuit A has no resolved 25–80 Hz peak")
            if not peak_b:
                reasons.append("circuit B has no resolved 25–80 Hz peak")
            if not all(np.isfinite(x) for x in (slope, plv, circular_mean)):
                reasons.append("non-finite phase estimator")
            metrics.append(TrialMetrics(
                valid=not reasons,
                invalid_reasons=tuple(reasons),
                rate_a_e_hz=rates["a_e"],
                rate_a_i_hz=rates["a_i"],
                rate_b_e_hz=rates["b_e"],
                rate_b_i_hz=rates["b_i"],
                frequency_a_hz=frequency_a,
                frequency_b_hz=frequency_b,
                frequency_difference_hz=abs(frequency_a - frequency_b),
                phase_slope_rad_s=slope,
                phase_slips=slips,
                phase_locking_value=plv,
                circular_mean_phase_rad=circular_mean,
            ))
            trace_cache["rate_a"].append(rate_a.astype(np.float32))
            trace_cache["rate_b"].append(rate_b.astype(np.float32))
            trace_cache["phase"].append(phase.astype(np.float32))
            window = round(250.0 / DT_MS)
            phase_delta = np.gradient(gaussian_filter1d(phase, window / 8.0))
            trace_cache["frequency_difference"].append(
                (phase_delta / (2.0 * np.pi * DT_MS / 1000.0)).astype(np.float32)
            )
    return metrics, {name: np.stack(rows) for name, rows in trace_cache.items()}


def _archive_condition(
    archive_path: Path,
    inputs: dict[str, np.ndarray],
    recording_path: Path,
) -> dict:
    """Keep exact binary events plus 1-ms population/projection state means."""
    decimate = round(STATE_DECIMATE_MS / DT_MS)
    payload: dict[str, np.ndarray] = {
        "drive_a_packed": np.packbits(inputs["drive_a"], axis=0),
        "drive_b_packed": np.packbits(inputs["drive_b"], axis=0),
        "event_steps": np.array(inputs["drive_a"].shape[0], dtype=np.int32),
    }
    with np.load(recording_path) as recordings:
        for label, key in POPULATION_KEYS.items():
            payload[f"{label}_spikes_packed"] = np.packbits(
                recordings[key].astype(np.uint8), axis=0
            )
        for key in recordings.files:
            if key in POPULATION_KEYS.values():
                continue
            array = recordings[key]
            usable = len(array) - len(array) % decimate
            mean = array[:usable].reshape(
                usable // decimate, decimate, array.shape[1], array.shape[2]
            ).mean(axis=(1, 3))
            payload[f"state_mean__{key}"] = mean.astype(np.float32)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive_path, **payload)
    return {
        "file": archive_path.name,
        "bytes": archive_path.stat().st_size,
        "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "contract": "exact packed input/population spikes; 1-ms population-mean voltage and projection conductance",
    }


def run_condition(
    scratch: Path,
    *,
    stage: str,
    detuning_index: int,
    rate_a_hz: float,
    rate_b_hz: float,
    coupling: float,
    keep_archive: bool,
) -> dict:
    key = _condition_key(stage, detuning_index, coupling)
    condition = scratch / "conditions" / key
    summary_path = condition / "summary.json"
    archive_path = condition / "compact.npz"
    if summary_path.exists() and (not keep_archive or archive_path.exists()):
        return _json_load(summary_path)

    condition.mkdir(parents=True, exist_ok=True)
    inputs, seed_rows = make_inputs(
        rate_a_hz, rate_b_hz, detuning_index=detuning_index
    )
    input_path = condition / "inputs.npz"
    np.savez_compressed(input_path, **inputs)
    sim_dir = condition / "sim"
    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    runtime_s = _run_graph_cli(_bundle_dir(scratch, coupling), input_path, sim_dir)
    execution_metrics = _json_load(sim_dir / "metrics.json")
    recording_path = sim_dir / "recordings.npz"
    trials, trace_cache = analyse_recordings(recording_path)
    np.savez_compressed(condition / "traces.npz", **trace_cache)
    archive = _archive_condition(archive_path, inputs, recording_path) if keep_archive else None
    dense_bytes = recording_path.stat().st_size
    summary = {
        "key": key,
        "stage": stage,
        "detuning_index": detuning_index,
        "rates_hz": {"a": rate_a_hz, "b": rate_b_hz},
        "coupling": coupling,
        "network_seed": NETWORK_SEED,
        "input_sha256": _sha256_bytes(inputs["drive_a"], inputs["drive_b"]),
        "seed_ledger": seed_rows,
        "runtime_s": runtime_s,
        "execution_metrics": execution_metrics,
        "dense_recording_bytes": dense_bytes,
        "archive": archive,
        "trials": [trial.as_dict() for trial in trials],
    }
    _json_dump(summary_path, summary)
    # Dense recordings and dense inputs are reproducibly represented by the
    # compact archive and ledger.  Keeping them per cell would exceed 25 GB.
    shutil.rmtree(sim_dir)
    input_path.unlink()
    return summary


def followup_cell_done(job_id: str) -> bool:
    if job_id not in FOLLOWUP_JOBS:
        return False
    spec = FOLLOWUP_JOBS[job_id]
    key = _condition_key("finite_size", spec["detuning_index"], spec["coupling"])
    return (FOLLOWUP_SCRATCH / "conditions" / key / "summary.json").exists()


def run_followup_cell(job_id: str) -> None:
    if not FOLLOWUP_MODE:
        raise RuntimeError("finite-size jobs require EXP078_FOLLOWUP_MODE")
    try:
        spec = FOLLOWUP_JOBS[job_id]
    except KeyError as exc:
        raise ValueError(f"unknown exp078 finite-size job: {job_id!r}") from exc
    summary = run_condition(
        FOLLOWUP_SCRATCH,
        stage="finite_size",
        detuning_index=spec["detuning_index"],
        rate_a_hz=spec["rate_a_hz"],
        rate_b_hz=spec["rate_b_hz"],
        coupling=spec["coupling"],
        keep_archive=True,
    )
    summary["target_detuning_hz"] = spec["target_detuning_hz"]
    summary["followup_mode"] = FOLLOWUP_MODE
    _json_dump(
        FOLLOWUP_SCRATCH / "conditions" / summary["key"] / "summary.json",
        summary,
    )


def run_followup_via_modal(meta: object) -> None:
    jobs = meta.only_cells or sorted(name for name in FOLLOWUP_JOBS if name != "benchmark")
    unknown = sorted(set(jobs) - set(FOLLOWUP_JOBS))
    if unknown:
        raise SystemExit(f"unknown exp078 finite-size jobs: {unknown}")
    benchmark_only = jobs == ["benchmark"]
    mode = "benchmark" if benchmark_only else "panel"
    modal_backend.dispatch(
        slug="exp078-followup",
        runner=SLUG,
        job_ids=jobs,
        live=meta.live,
        local_collect_dir=FOLLOWUP_SCRATCH,
        ledger_path=FOLLOWUP_SCRATCH / "compute_ledgers" / f"{mode}.json",
        timeout_s=1800 if benchmark_only else 7200,
        extra_env={"EXP078_FOLLOWUP_MODE": mode},
        is_done_name="followup_cell_done",
        run_job_name="run_followup_cell",
    )


def publish_followup(
    panel_root: Path,
    destination: Path,
    *,
    modal_ledger: Path | None = None,
) -> dict:
    """Publish the registered finite-size panel from completed condition jobs."""
    tolerances = _json_load(REPO / "artifacts/data/exp078/locking_tolerances.json")
    cells = []
    coupled_trials = []
    archive_dir = destination / "finite_size_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for job_id, spec in sorted(FOLLOWUP_JOBS.items()):
        if job_id == "benchmark":
            continue
        key = _condition_key("finite_size", spec["detuning_index"], spec["coupling"])
        condition = panel_root / "conditions" / key
        summary = _json_load(condition / "summary.json")
        trials = []
        for trial in summary["trials"]:
            row = {**trial, "locked": classify_trial(trial, tolerances)}
            if spec["coupling"] > 0:
                row["phase_sign_correct"] = bool(
                    np.sign(trial["circular_mean_phase_rad"])
                    == np.sign(spec["target_detuning_hz"])
                )
                coupled_trials.append(row)
            trials.append(row)
        archive_name = f"{job_id}.npz"
        shutil.copy2(condition / "compact.npz", archive_dir / archive_name)
        cells.append({
            "job_id": job_id,
            **spec,
            "runtime_s": summary["runtime_s"],
            "execution_metrics": summary["execution_metrics"],
            "valid_trials": sum(row["valid"] for row in trials),
            "locked_trials": sum(row["locked"] for row in trials),
            "phase_sign_correct_trials": (
                sum(row["phase_sign_correct"] for row in trials)
                if spec["coupling"] > 0 else None
            ),
            "mean_phase_rad": float(np.mean([row["circular_mean_phase_rad"] for row in trials])),
            "mean_plv": float(np.mean([row["phase_locking_value"] for row in trials])),
            "archive": archive_name,
            "trials": trials,
        })
    negative = next(row for row in cells if row["job_id"] == "m1_k016")
    positive = next(row for row in cells if row["job_id"] == "p1_k016")
    result = {
        "schema": "pinglab.exp078.finite-size-followup/v1",
        "status": "complete",
        "config": {
            "n_input_per_circuit": 800,
            "n_e_per_circuit": 800,
            "n_i_per_circuit": 200,
            "dt_ms": 0.1,
            "t_ms": 5500.0,
            "burn_ms": 500.0,
            "trials_per_cell": 10,
            "target_detunings_hz": [-1.0, 1.0],
            "couplings": [0.0, 0.016, 0.024],
            "recording": "observables",
            "execution_host": "local CPU after L40S benchmark rejection",
        },
        "locking_tolerances_inherited_from_exp078": tolerances,
        "cells": cells,
        "conclusion": {
            "passed": bool(
                len(coupled_trials) == 40
                and all(row["valid"] and row["locked"] and row["phase_sign_correct"] for row in coupled_trials)
            ),
            "coupled_trials": len(coupled_trials),
            "valid_coupled_trials": sum(row["valid"] for row in coupled_trials),
            "locked_coupled_trials": sum(row["locked"] for row in coupled_trials),
            "phase_sign_correct_trials": sum(row["phase_sign_correct"] for row in coupled_trials),
            "disputed_negative_cell": {
                "locked_trials": negative["locked_trials"],
                "phase_sign_correct_trials": negative["phase_sign_correct_trials"],
                "mean_phase_rad": negative["mean_phase_rad"],
                "mean_plv": negative["mean_plv"],
            },
            "mirrored_positive_cell": {
                "locked_trials": positive["locked_trials"],
                "phase_sign_correct_trials": positive["phase_sign_correct_trials"],
                "mean_phase_rad": positive["mean_phase_rad"],
                "mean_plv": positive["mean_plv"],
            },
        },
        "runtime": {
            "total_local_cell_s": float(sum(row["runtime_s"] for row in cells)),
            "modal_benchmark": _json_load(modal_ledger) if modal_ledger and modal_ledger.exists() else None,
        },
    }
    _json_dump(destination / "finite_size_followup.json", result)
    return result


def materialize_parameter_tensors(scratch: Path, couplings: list[float]) -> dict:
    """Retain the exact graph-executor tensors once per frozen coupling level."""
    root = scratch / "parameter_tensors"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    invariant_names: set[str] | None = None
    invariant_hashes: dict[str, str] | None = None
    cross_names = {
        "a_E_to_b_E.weight",
        "a_E_to_b_I.weight",
        "b_E_to_a_E.weight",
        "b_E_to_a_I.weight",
    }
    for coupling in couplings:
        filename = f"k_{_float_tag(coupling)}.npz"
        parameter_path = root / filename
        if not parameter_path.exists():
            probe = root / f"probe_{_float_tag(coupling)}"
            probe.mkdir(exist_ok=True)
            input_path = probe / "inputs.npz"
            zeros = np.zeros((1, 1, N_INPUT), dtype=np.uint8)
            np.savez_compressed(input_path, drive_a=zeros, drive_b=zeros)
            output = probe / "output"
            _run_graph_cli(_bundle_dir(scratch, coupling), input_path, output)
            shutil.copy2(output / "parameters.npz", parameter_path)
            shutil.rmtree(probe)

        bundle = author_network(coupling=coupling)
        expected = {row["id"] for row in bundle.graph["parameters"]}
        parameter_rows = {}
        with np.load(parameter_path) as parameters:
            if set(parameters.files) != expected:
                raise RuntimeError(f"parameter archive {filename} does not match its graph")
            for name in sorted(parameters.files):
                array = parameters[name]
                # GraphExecutor stores fan-in-normalised runtime weights.
                stored_coupling = (
                    np.asarray(coupling, dtype=array.dtype)
                    / np.asarray(N_E, dtype=array.dtype)
                )
                if name in cross_names and not np.all(array == stored_coupling):
                    raise RuntimeError(
                        f"{name} does not equal fan-in-normalised K={coupling}"
                    )
                parameter_rows[name] = {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": _sha256_bytes(array),
                }
        current_invariant = expected - cross_names
        current_hashes = {
            name: parameter_rows[name]["sha256"] for name in current_invariant
        }
        if invariant_names is None:
            invariant_names = current_invariant
            invariant_hashes = current_hashes
        elif current_invariant != invariant_names or current_hashes != invariant_hashes:
            raise RuntimeError("non-coupling parameter tensors changed across K levels")
        rows.append({
            "coupling": coupling,
            "file": filename,
            "bytes": parameter_path.stat().st_size,
            "sha256": hashlib.sha256(parameter_path.read_bytes()).hexdigest(),
            "parameters": parameter_rows,
        })
    return {
        "schema": "pinglab.exp078.parameter-tensors/v1",
        "network_seed": NETWORK_SEED,
        "coupling_parameter_names": sorted(cross_names),
        "invariant_parameter_names": sorted(invariant_names or ()),
        "verified_invariant_across_couplings": True,
        "rows": rows,
    }


def _valid_trials(condition: dict) -> list[dict]:
    return [trial for trial in condition["trials"] if trial["valid"]]


def run_calibration(scratch: Path) -> dict:
    """Freeze the contiguous valid input-rate interval before coupling."""
    frozen_path = scratch / "registration" / "calibration.json"
    if frozen_path.exists():
        return _json_load(frozen_path)
    rows = []
    for index, rate_hz in enumerate(RATE_GRID_HZ):
        print(f"[calibration {index + 1}/{len(RATE_GRID_HZ)}] {rate_hz:g} Hz")
        condition = run_condition(
            scratch,
            stage="calibration",
            detuning_index=100 + index,
            rate_a_hz=rate_hz,
            rate_b_hz=rate_hz,
            coupling=0.0,
            keep_archive=False,
        )
        trials = _valid_trials(condition)
        frequencies = [
            trial[name]
            for trial in trials
            for name in ("frequency_a_hz", "frequency_b_hz")
        ]
        rows.append({
            "rate_hz": rate_hz,
            "valid_trials": len(trials),
            "total_trials": len(condition["trials"]),
            "median_frequency_hz": float(np.median(frequencies)) if frequencies else None,
            "frequency_iqr_hz": (
                float(np.subtract(*np.percentile(frequencies, [75, 25])))
                if frequencies else None
            ),
            "condition_key": condition["key"],
        })
    # Longest contiguous interval that is fully valid and non-decreasing.  The
    # 0.4-Hz spectral grid makes equal adjacent medians legitimate plateaus.
    candidates: list[list[dict]] = []
    current: list[dict] = []
    for row in rows:
        # Peak switching at a rate boundary can satisfy the per-trial validity
        # rule yet make the calibration estimator unusable.  Register at most
        # two 0.4-Hz spectral bins of within-rate IQR before coupling is viewed.
        stable = row["frequency_iqr_hz"] is not None and row["frequency_iqr_hz"] <= 0.8
        good = row["valid_trials"] == row["total_trials"] and stable
        monotonic = not current or row["median_frequency_hz"] >= current[-1]["median_frequency_hz"]
        if good and monotonic:
            current.append(row)
        else:
            if current:
                candidates.append(current)
            current = [row] if good else []
    if current:
        candidates.append(current)
    interval = max(candidates, key=len, default=[])
    if len(interval) < 3:
        raise RuntimeError("calibration did not find a three-rate valid monotonic interval")
    result = {
        "status": "frozen before coupled pilot",
        "rows": rows,
        "operating_interval_hz": [interval[0]["rate_hz"], interval[-1]["rate_hz"]],
        "operating_rows": interval,
    }
    _json_dump(frozen_path, result)
    return result


def register_tolerances_from_scratch(scratch: Path, calibration: dict) -> dict:
    frozen_path = scratch / "registration" / "locking_tolerances.json"
    if frozen_path.exists():
        return _json_load(frozen_path)
    # Use every fully valid equal-drive trial in the frozen operating interval.
    frequency_difference = []
    phase_slope = []
    phase_slips = []
    source_keys = []
    operating_rates = {row["rate_hz"] for row in calibration["operating_rows"]}
    for index, rate_hz in enumerate(RATE_GRID_HZ):
        if rate_hz not in operating_rates:
            continue
        key = _condition_key("calibration", 100 + index, 0.0)
        condition = _json_load(scratch / "conditions" / key / "summary.json")
        source_keys.append(key)
        for trial in _valid_trials(condition):
            frequency_difference.append(trial["frequency_difference_hz"])
            phase_slope.append(abs(trial["phase_slope_rad_s"]))
            phase_slips.append(trial["phase_slips"])
    if len(frequency_difference) < TRIALS:
        raise RuntimeError("too few valid calibration repeats to register tolerances")
    # The 95th percentile is registered, with one spectral bin as a hard floor.
    spectral_bin_hz = 1.0 / ((T_MS - BURN_MS) / 1000.0)
    result = {
        "status": "frozen before coupled pilot",
        "source_condition_keys": source_keys,
        "sample_count": len(frequency_difference),
        "frequency_difference_hz": max(
            spectral_bin_hz, float(np.quantile(frequency_difference, 0.95, method="higher"))
        ),
        "absolute_phase_slope_rad_s": max(
            2.0 * np.pi * spectral_bin_hz,
            float(np.quantile(phase_slope, 0.95, method="higher")),
        ),
        "phase_slips": int(np.quantile(phase_slips, 0.95, method="higher")),
        "estimator": "95th percentile (higher) of valid uncoupled equal-drive trials",
        "spectral_bin_hz": spectral_bin_hz,
    }
    _json_dump(frozen_path, result)
    return result


def select_rate_pairs(calibration: dict) -> list[dict]:
    """Map signed target detunings onto symmetric calibrated input rates."""
    rows = calibration["operating_rows"]
    input_rates = np.array([row["rate_hz"] for row in rows], dtype=float)
    frequencies = np.array([row["median_frequency_hz"] for row in rows], dtype=float)
    unique_frequencies, unique_indices = np.unique(frequencies, return_index=True)
    unique_rates = input_rates[unique_indices]
    if len(unique_frequencies) < 2:
        raise RuntimeError("calibration frequency curve cannot resolve detuning")
    centre_frequency = float((unique_frequencies[0] + unique_frequencies[-1]) / 2.0)
    pairs = []
    for index, target in enumerate(TARGET_DETUNINGS_HZ):
        target_a = centre_frequency + target / 2.0
        target_b = centre_frequency - target / 2.0
        if target_a < unique_frequencies[0] or target_a > unique_frequencies[-1]:
            raise RuntimeError(f"target detuning {target:g} Hz exceeds calibrated circuit-A range")
        if target_b < unique_frequencies[0] or target_b > unique_frequencies[-1]:
            raise RuntimeError(f"target detuning {target:g} Hz exceeds calibrated circuit-B range")
        pairs.append({
            "detuning_index": index,
            "target_detuning_hz": target,
            "rate_a_hz": float(np.interp(target_a, unique_frequencies, unique_rates)),
            "rate_b_hz": float(np.interp(target_b, unique_frequencies, unique_rates)),
        })
    return pairs


def classify_trial(trial: dict, tolerances: dict) -> bool:
    return bool(
        trial["valid"]
        and trial["frequency_difference_hz"] <= tolerances["frequency_difference_hz"]
        and abs(trial["phase_slope_rad_s"]) <= tolerances["absolute_phase_slope_rad_s"]
        and trial["phase_slips"] <= tolerances["phase_slips"]
    )


def run_pilot(
    scratch: Path,
    rate_pairs: list[dict],
    tolerances: dict,
) -> dict:
    frozen_path = scratch / "registration" / "coupling_pilot.json"
    if frozen_path.exists():
        return _json_load(frozen_path)
    pilot_indices = (0, len(rate_pairs) // 2, len(rate_pairs) - 1)
    cells = []
    for coupling in PILOT_COUPLINGS:
        for index in pilot_indices:
            pair = rate_pairs[index]
            print(
                f"[pilot] target Δf={pair['target_detuning_hz']:+g} Hz, K={coupling:g}"
            )
            condition = run_condition(
                scratch,
                stage="pilot",
                detuning_index=pair["detuning_index"],
                rate_a_hz=pair["rate_a_hz"],
                rate_b_hz=pair["rate_b_hz"],
                coupling=coupling,
                keep_archive=False,
            )
            valid = _valid_trials(condition)
            cells.append({
                "condition_key": condition["key"],
                "target_detuning_hz": pair["target_detuning_hz"],
                "coupling": coupling,
                "valid_fraction": len(valid) / TRIALS,
                "locked_fraction": sum(classify_trial(t, tolerances) for t in valid) / len(valid) if valid else None,
            })
    # The upper bound is the first valid level locking all three pilot
    # detunings.  Lower levels then retain the uncoupled/partial transition
    # rather than wasting the primary grid deep in the saturated regime.
    safe = []
    for coupling in PILOT_COUPLINGS[1:]:
        rows = [cell for cell in cells if cell["coupling"] == coupling]
        if rows and all(
            cell["valid_fraction"] >= 0.8
            and cell["locked_fraction"] is not None
            and cell["locked_fraction"] >= 0.8
            for cell in rows
        ):
            safe.append(coupling)
    if not safe:
        raise RuntimeError("pilot found no valid level locking all three anchor detunings")
    maximum = min(safe)
    result = {
        "status": "frozen before primary sweep",
        "candidate_couplings": list(PILOT_COUPLINGS),
        "pilot_detuning_indices": list(pilot_indices),
        "cells": cells,
        "selected_maximum_coupling": maximum,
        "selection_rule": "smallest candidate with >=80% valid and locked trials at all three pilot detunings",
    }
    _json_dump(frozen_path, result)
    return result


def freeze_primary_grid(
    scratch: Path,
    rate_pairs: list[dict],
    pilot: dict,
    tolerances: dict,
) -> dict:
    path = scratch / "registration" / "primary_grid.json"
    if path.exists():
        return _json_load(path)
    couplings = np.concatenate((
        np.array([0.0]),
        np.linspace(
            pilot["selected_maximum_coupling"] / PRIMARY_NONZERO_LEVELS,
            pilot["selected_maximum_coupling"],
            PRIMARY_NONZERO_LEVELS,
        ),
    ))
    cells = [
        {
            "detuning_index": pair["detuning_index"],
            "target_detuning_hz": pair["target_detuning_hz"],
            "rate_a_hz": pair["rate_a_hz"],
            "rate_b_hz": pair["rate_b_hz"],
            "coupling": float(coupling),
        }
        for pair in rate_pairs
        for coupling in couplings
    ]
    registration = {
        "schema": "pinglab.exp078.primary-grid/v1",
        "status": "frozen before primary sweep",
        "created_unix_s": time.time(),
        "network_seed": NETWORK_SEED,
        "trial_count": TRIALS,
        "target_detunings_hz": list(TARGET_DETUNINGS_HZ),
        "couplings": [float(x) for x in couplings],
        "tolerances": tolerances,
        "representative_trace_policy": {
            "zero_detuning": "target 0 Hz at maximum registered coupling",
            "transition_detuning_index": 11,
            "transition_target_detuning_hz": 4.0,
            "inside": "first coupling with locked fraction >=0.8",
            "outside": "same detuning at immediately preceding coupling",
            "trial": 0,
        },
        "cells": cells,
    }
    canonical = json.dumps(registration, sort_keys=True, separators=(",", ":")).encode()
    registration["sha256"] = hashlib.sha256(canonical).hexdigest()
    _json_dump(path, registration)
    return registration


def run_primary(scratch: Path, grid: dict, tolerances: dict) -> list[dict]:
    results = []
    total = len(grid["cells"])
    for number, cell in enumerate(grid["cells"], start=1):
        print(
            f"[primary {number}/{total}] Δf target={cell['target_detuning_hz']:+g} Hz, "
            f"K={cell['coupling']:.4g}"
        )
        condition = run_condition(
            scratch,
            stage="primary",
            detuning_index=cell["detuning_index"],
            rate_a_hz=cell["rate_a_hz"],
            rate_b_hz=cell["rate_b_hz"],
            coupling=cell["coupling"],
            keep_archive=True,
        )
        valid = _valid_trials(condition)
        trial_rows = []
        for trial_index, trial in enumerate(condition["trials"]):
            row = dict(trial)
            row["trial"] = trial_index
            row["locked"] = classify_trial(trial, tolerances)
            trial_rows.append(row)
        results.append({
            **cell,
            "condition_key": condition["key"],
            "runtime_s": condition["runtime_s"],
            "dense_recording_bytes": condition["dense_recording_bytes"],
            "archive": condition["archive"],
            "measured_detuning_hz": float(np.mean([
                trial["frequency_a_hz"] - trial["frequency_b_hz"]
                for trial in valid
            ])) if valid else None,
            "valid_fraction": len(valid) / TRIALS,
            "locked_fraction": sum(row["locked"] for row in trial_rows) / len(valid) if valid else None,
            "trials": trial_rows,
        })
    return attach_natural_detunings(results)


def attach_natural_detunings(results: list[dict]) -> list[dict]:
    """Join every coupled trial to its paired K=0 natural-frequency control."""
    controls = {
        row["detuning_index"]: [
            trial["frequency_a_hz"] - trial["frequency_b_hz"]
            for trial in row["trials"]
        ]
        for row in results
        if row["coupling"] == 0.0
    }
    if len(controls) != len(TARGET_DETUNINGS_HZ):
        raise RuntimeError("missing K=0 natural-detuning controls")
    for row in results:
        natural = controls[row["detuning_index"]]
        if len(natural) != len(row["trials"]):
            raise RuntimeError("paired K=0 control has a different trial count")
        for trial, value in zip(row["trials"], natural):
            trial["natural_detuning_hz"] = value
        row["measured_detuning_hz"] = float(np.mean(natural))
    return results


def _matrix(results: list[dict], couplings: list[float], field: str) -> np.ndarray:
    matrix = np.full((len(couplings), len(TARGET_DETUNINGS_HZ)), np.nan)
    coupling_index = {round(value, 12): i for i, value in enumerate(couplings)}
    for row in results:
        values = [trial[field] for trial in row["trials"] if trial["valid"]]
        if field == "locked":
            value = row["locked_fraction"]
        elif values:
            value = float(np.mean(values))
        else:
            value = np.nan
        matrix[coupling_index[round(row["coupling"], 12)], row["detuning_index"]] = value
    return matrix


def _measured_detunings(results: list[dict]) -> np.ndarray:
    values = np.full(len(TARGET_DETUNINGS_HZ), np.nan)
    for row in results:
        if row["coupling"] == 0.0:
            values[row["detuning_index"]] = row["measured_detuning_hz"]
    if not np.all(np.isfinite(values)):
        raise RuntimeError("primary K=0 cells did not produce every measured detuning")
    return values


def _centred_locked_widths(locking: np.ndarray, measured: np.ndarray) -> list[float]:
    centre = int(np.argmin(np.abs(measured)))
    widths = []
    for row in locking:
        locked = np.isfinite(row) & (row >= 0.8)
        if not locked[centre]:
            widths.append(0.0)
            continue
        left = centre
        right = centre
        while left > 0 and locked[left - 1]:
            left -= 1
        while right + 1 < len(locked) and locked[right + 1]:
            right += 1
        widths.append(float(measured[right] - measured[left]))
    return widths


def evaluate_reproduction(results: list[dict], couplings: list[float]) -> dict:
    locking = _matrix(results, couplings, "locked")
    measured = _measured_detunings(results)
    widths = _centred_locked_widths(locking, measured)
    successive_increases = 0
    longest_increase_run = 0
    previous = widths[1]
    for width in widths[2:]:
        if width > previous + 1e-9:
            successive_increases += 1
            longest_increase_run = max(longest_increase_run, successive_increases)
        else:
            successive_increases = 0
        previous = width

    phase_cells = []
    for row in results:
        if row["coupling"] == 0 or row["locked_fraction"] is None or row["locked_fraction"] < 0.8:
            continue
        natural_detuning = measured[row["detuning_index"]]
        if abs(natural_detuning) < 0.4:
            continue
        locked_trials = [trial for trial in row["trials"] if trial["locked"]]
        if not locked_trials:
            continue
        circular = np.angle(np.mean(np.exp(1j * np.array([
            trial["circular_mean_phase_rad"] for trial in locked_trials
        ]))))
        phase_cells.append({
            "detuning_index": row["detuning_index"],
            "coupling": row["coupling"],
            "measured_detuning_hz": natural_detuning,
            "circular_mean_phase_rad": float(circular),
            "faster_circuit_leads": bool(np.sign(circular) == np.sign(natural_detuning)),
        })
    phase_lead_fraction = (
        float(np.mean([cell["faster_circuit_leads"] for cell in phase_cells]))
        if phase_cells else 0.0
    )
    tongue_ok = longest_increase_run >= 3
    phase_ok = bool(phase_cells) and all(
        cell["faster_circuit_leads"] for cell in phase_cells
    )
    valid_ok = all(row["valid_fraction"] > 0 for row in results)
    return {
        "passed": tongue_ok and phase_ok and valid_ok,
        "criteria": {
            "contiguous_centred_width_increases_across_three_successive_nonzero_levels": tongue_ok,
            "faster_circuit_leads_in_every_locked_nonzero_detuning_cell": phase_ok,
            "every_grid_cell_has_a_valid_trial": valid_ok,
        },
        "centred_locked_widths_hz": widths,
        "longest_successive_increase_run": longest_increase_run,
        "phase_lead_fraction": phase_lead_fraction,
        "phase_cells": phase_cells,
    }


def _axis_edges(values: np.ndarray) -> np.ndarray:
    mid = (values[:-1] + values[1:]) / 2.0
    return np.concatenate((
        [values[0] - (mid[0] - values[0])],
        mid,
        [values[-1] + (values[-1] - mid[-1])],
    ))


def plot_calibration(calibration: dict, out_path: Path) -> None:
    theme.apply()
    rows = calibration["rows"]
    x = np.array([row["rate_hz"] for row in rows])
    y = np.array([row["median_frequency_hz"] for row in rows])
    error = np.array([row["frequency_iqr_hz"] for row in rows]) / 2.0
    lo, hi = calibration["operating_interval_hz"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.axvspan(lo, hi, color=theme.GREY_LIGHT, alpha=0.7, label="frozen operating interval")
    ax.errorbar(x, y, yerr=error, marker="o", color=theme.INK_BLACK, capsize=3)
    ax.set(xlabel="input rate (Hz per channel)", ylabel="uncoupled gamma peak (Hz)")
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_locking_map(results: list[dict], couplings: list[float], out_path: Path) -> None:
    theme.apply()
    measured = _measured_detunings(results)
    locking = _matrix(results, couplings, "locked")
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    mesh = ax.pcolormesh(
        _axis_edges(measured), _axis_edges(np.asarray(couplings)), locking,
        vmin=0, vmax=1, cmap="magma", shading="flat",
    )
    fig.colorbar(mesh, ax=ax, label="locked fraction of valid trials")
    ax.axvline(0, color="white", linewidth=0.8, alpha=0.7)
    ax.set(xlabel="measured uncoupled detuning $f_A^0-f_B^0$ (Hz)", ylabel="coupling K")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_supporting_maps(results: list[dict], couplings: list[float], out_path: Path) -> None:
    theme.apply()
    measured = _measured_detunings(results)
    x_edges = _axis_edges(measured)
    y_edges = _axis_edges(np.asarray(couplings))
    panels = [
        ("frequency_difference_hz", "$|f_A-f_B|$ (Hz)", "viridis", None),
        ("phase_slope_rad_s", "phase slope (rad/s)", "coolwarm", "symmetric"),
        ("phase_slips", "phase slips", "viridis", None),
        ("phase_locking_value", "phase-locking value", "magma", (0, 1)),
        ("circular_mean_phase_rad", "circular mean phase (rad)", "twilight", (-np.pi, np.pi)),
        ("valid", "valid fraction", "Greys", (0, 1)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), sharex=True, sharey=True)
    for ax, (field, title, cmap, limits) in zip(axes.flat, panels):
        if field == "valid":
            matrix = np.full((len(couplings), len(TARGET_DETUNINGS_HZ)), np.nan)
            for row in results:
                i = couplings.index(row["coupling"])
                matrix[i, row["detuning_index"]] = row["valid_fraction"]
        else:
            matrix = _matrix(results, couplings, field)
        kwargs = {}
        if limits == "symmetric":
            bound = float(np.nanmax(np.abs(matrix)))
            kwargs = {"vmin": -bound, "vmax": bound}
        elif isinstance(limits, tuple):
            kwargs = {"vmin": limits[0], "vmax": limits[1]}
        mesh = ax.pcolormesh(x_edges, y_edges, matrix, cmap=cmap, shading="flat", **kwargs)
        fig.colorbar(mesh, ax=ax, shrink=0.82)
        ax.set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel("measured $Delta f_0$ (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel("coupling K")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def select_representatives(results: list[dict], grid: dict) -> list[dict]:
    policy = grid["representative_trace_policy"]
    zero = next(
        row for row in results
        if row["target_detuning_hz"] == 0.0 and row["coupling"] == max(grid["couplings"])
    )
    transition_rows = sorted(
        (row for row in results if row["detuning_index"] == policy["transition_detuning_index"]),
        key=lambda row: row["coupling"],
    )
    inside_index = next(
        (i for i, row in enumerate(transition_rows) if row["locked_fraction"] is not None and row["locked_fraction"] >= 0.8),
        None,
    )
    if inside_index is None or inside_index == 0:
        raise RuntimeError("predeclared +4 Hz trace location has no inside/outside transition")
    return [
        {"label": "zero detuning", **zero},
        {"label": "inside tongue", **transition_rows[inside_index]},
        {"label": "immediately outside", **transition_rows[inside_index - 1]},
    ]


def plot_representative_traces(scratch: Path, selected: list[dict], out_path: Path) -> None:
    theme.apply()
    trial = 0
    burn = round(BURN_MS / DT_MS)
    display_steps = round(750.0 / DT_MS)
    fig, axes = plt.subplots(len(selected), 4, figsize=(14.0, 7.5), sharex="col")
    for row_index, row in enumerate(selected):
        condition = scratch / "conditions" / row["condition_key"]
        with np.load(condition / "compact.npz") as packed, np.load(condition / "traces.npz") as traces:
            n_steps = int(packed["event_steps"])
            a = np.unpackbits(packed["a_e_spikes_packed"], axis=0)[:n_steps, trial]
            b = np.unpackbits(packed["b_e_spikes_packed"], axis=0)[:n_steps, trial]
            start, stop = burn, burn + display_steps
            for spikes, offset, colour in ((a, 0, theme.INK_BLACK), (b, N_E + 4, theme.DEEP_RED)):
                times, cells = np.nonzero(spikes[start:stop])
                axes[row_index, 0].scatter(
                    (times + start) * DT_MS, cells + offset, s=1.1,
                    color=colour, linewidths=0, rasterized=True,
                )
            post_t = np.arange(traces["rate_a"].shape[1]) * DT_MS + BURN_MS
            axes[row_index, 1].plot(post_t, traces["rate_a"][trial], color=theme.INK_BLACK, lw=0.8)
            axes[row_index, 1].plot(post_t, traces["rate_b"][trial], color=theme.DEEP_RED, lw=0.8)
            axes[row_index, 2].plot(post_t, traces["phase"][trial], color=theme.INK_BLACK, lw=0.8)
            axes[row_index, 3].plot(post_t, traces["frequency_difference"][trial], color=theme.INK_BLACK, lw=0.8)
        axes[row_index, 0].set_ylabel(
            f"{row['label']}\nΔf₀={row['measured_detuning_hz']:+.1f} Hz\nK={row['coupling']:.3f}"
        )
    titles = ("E rasters (A black, B red)", "smoothed E rates", "unwrapped relative phase", "instantaneous Δf")
    for ax, title in zip(axes[0], titles):
        ax.set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel("time (ms)")
    axes[0, 1].set_xlim(BURN_MS, BURN_MS + 750.0)
    axes[0, 2].set_xlim(BURN_MS, BURN_MS + 750.0)
    axes[0, 3].set_xlim(BURN_MS, BURN_MS + 750.0)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_network_diagram(bundle: snn.Bundle, svg_path: Path, png_path: Path) -> None:
    """Render the authored graph as a legible two-circuit scientific schematic."""
    graph = bundle.graph
    population_sizes = {row["id"]: row["size"] for row in graph["populations"]}
    projection_ids = {row["id"] for row in graph["projections"]}
    expected = {
        "a_E_to_I", "a_I_to_E", "b_E_to_I", "b_I_to_E",
        "a_E_to_b_E", "a_E_to_b_I", "b_E_to_a_E", "b_E_to_a_I",
    }
    if not expected <= projection_ids:
        raise ValueError(f"diagram graph is missing projections: {sorted(expected - projection_ids)}")

    theme.apply()
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 6)
    ax.axis("off")

    ink = theme.INK_BLACK
    red = theme.DEEP_RED
    muted = theme.GREY_MID
    border = "#B9C3CF"
    panel = "#F7F8FA"
    coupling = "#53657D"
    positions = {
        "drive_a": (2.25, 5.15), "a_I": (1.15, 2.65), "a_E": (3.35, 2.65),
        "drive_b": (7.75, 5.15), "b_E": (6.65, 2.65), "b_I": (8.85, 2.65),
    }

    def card(name: str, title: str, subtitle: str, *, width: float = 1.45) -> None:
        x, y = positions[name]
        box = FancyBboxPatch(
            (x - width / 2, y - 0.55), width, 1.1,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor="white", edgecolor=border, linewidth=1.5, zorder=4,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.13, title, ha="center", va="center", fontsize=10.5,
                fontweight="semibold", color=ink, zorder=5)
        ax.text(x, y - 0.22, subtitle, ha="center", va="center", fontsize=7.5,
                color=muted, zorder=5)

    for center, label in ((2.25, "CIRCUIT A"), (7.75, "CIRCUIT B")):
        ax.add_patch(FancyBboxPatch(
            (center - 2.05, 1.15), 4.1, 3.0,
            boxstyle="round,pad=0.1,rounding_size=0.16",
            facecolor=panel, edgecolor="#D8DEE7", linewidth=1.0, zorder=0,
        ))
        ax.text(center, 3.88, label, ha="center", va="center", fontsize=9,
                fontweight="bold", color=muted)

    card("drive_a", "drive A", "independent Poisson\n80 channels", width=2.25)
    card("drive_b", "drive B", "independent Poisson\n80 channels", width=2.25)
    for name, label in (("a_E", "E_A"), ("a_I", "I_A"), ("b_E", "E_B"), ("b_I", "I_B")):
        card(
            name,
            f"${label}$",
            f"{population_sizes[name]} neurons",
            width=1.55,
        )

    def arrow(start, end, *, color=ink, style="-", rad=0.0, heads="-|>", width=1.8, z=2):
        patch = FancyArrowPatch(
            start, end, arrowstyle=heads, mutation_scale=13,
            connectionstyle=f"arc3,rad={rad}", color=color, linewidth=width,
            linestyle=style, shrinkA=5, shrinkB=5, zorder=z,
        )
        ax.add_patch(patch)

    def routed_arrow(points, *, color=coupling, style="--", width=1.8):
        path = MplPath(
            points,
            [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1),
        )
        ax.add_patch(FancyArrowPatch(
            path=path, arrowstyle="-|>", mutation_scale=13, color=color,
            linewidth=width, linestyle=style, shrinkA=4, shrinkB=4, zorder=1,
            joinstyle="round",
        ))

    # Independent input and local PING loops.
    arrow((2.25, 4.58), (3.15, 3.23), width=1.6)
    arrow((7.75, 4.58), (6.85, 3.23), width=1.6)
    arrow((2.50, 2.91), (2.00, 2.91), width=2.0)
    arrow((2.00, 2.39), (2.50, 2.39), color=red, heads="-[", width=2.2)
    arrow((7.50, 2.91), (8.00, 2.91), width=2.0)
    arrow((8.00, 2.39), (7.50, 2.39), color=red, heads="-[", width=2.2)

    # Lowet-style reciprocal cross-circuit excitation.
    arrow((4.15, 2.95), (5.85, 2.95), color=coupling, style="--",
          heads="<|-|>", width=2.0)
    ax.text(5.0, 3.13, "reciprocal E→E", ha="center", va="bottom",
            fontsize=8.5, color=coupling)
    routed_arrow([(3.35, 2.02), (3.35, 0.72), (8.85, 0.72), (8.85, 2.02)])
    routed_arrow([(6.65, 2.02), (6.65, 0.22), (1.15, 0.22), (1.15, 2.02)])
    ax.text(6.1, 0.88, "$E_A$ → $I_B$", ha="center", va="bottom",
            fontsize=8.0, color=coupling)
    ax.text(3.9, 0.38, "$E_B$ → $I_A$", ha="center", va="bottom",
            fontsize=8.0, color=coupling)

    ax.plot([], [], color=ink, linewidth=2, label="excitatory")
    ax.plot([], [], color=red, linewidth=2.2, label="inhibitory")
    ax.plot([], [], color=coupling, linewidth=2, linestyle="--", label="cross-circuit")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              frameon=False, fontsize=8.5, handlelength=2.4)

    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    meta = parse_meta(sys.argv, allow_dispatch=True)
    if meta.modal:
        run_followup_via_modal(meta)
        return
    if meta.runpod:
        raise SystemExit("exp078 finite-size follow-up currently supports Modal, not RunPod")
    started = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {run_id}")

    with published_run(
        SLUG,
        run_id,
        scale=SCALE,
        plot_only=meta.plot_only,
    ) as (scratch, staging):
        registration_dir = scratch / "registration"
        results_path = scratch / "primary_results.json"

        calibration = run_calibration(scratch)
        tolerances = register_tolerances_from_scratch(scratch, calibration)
        rate_pairs = select_rate_pairs(calibration)
        pilot = run_pilot(scratch, rate_pairs, tolerances)
        grid = freeze_primary_grid(scratch, rate_pairs, pilot, tolerances)
        parameter_tensors = materialize_parameter_tensors(scratch, grid["couplings"])
        if meta.plot_only or meta.skip_training:
            if not results_path.exists():
                raise RuntimeError("no cached primary sweep; run exp078 without lifecycle flags first")
            results_doc = _json_load(results_path)
            if results_doc["grid_sha256"] != grid["sha256"]:
                raise RuntimeError("cached primary results do not match the frozen grid")
            results = results_doc["cells"]
        else:
            results = run_primary(scratch, grid, tolerances)
        results = attach_natural_detunings(results)
        results_doc = {
            "schema": "pinglab.exp078.results/v1",
            "grid_sha256": grid["sha256"],
            "cells": results,
        }
        _json_dump(results_path, results_doc)

        couplings = grid["couplings"]
        conclusion = evaluate_reproduction(results, couplings)
        selected = select_representatives(results, grid)

        bundle = author_network()
        scratch_bundle = scratch / "network.bundle"
        if scratch_bundle.exists():
            shutil.rmtree(scratch_bundle)
        bundle.write(scratch_bundle, visualise=True)

        shutil.copytree(scratch_bundle, staging / "network.bundle", dirs_exist_ok=True)
        render_network_diagram(
            bundle,
            staging / "network.svg",
            staging / "network.png",
        )

        plot_calibration(calibration, staging / "calibration.png")
        plot_locking_map(results, couplings, staging / "locking_map.png")
        plot_supporting_maps(results, couplings, staging / "supporting_maps.png")
        plot_representative_traces(scratch, selected, staging / "representative_traces.png")

        # Publish the complete registered design and machine-readable result.
        for name in (
            "calibration.json", "locking_tolerances.json", "coupling_pilot.json",
            "primary_grid.json",
        ):
            shutil.copy2(registration_dir / name, staging / name)
        shutil.copy2(results_path, staging / "results.json")
        _json_dump(staging / "conclusion.json", conclusion)
        _json_dump(staging / "parameter_tensors.json", parameter_tensors)
        published_parameters = staging / "parameter_tensors"
        published_parameters.mkdir(exist_ok=True)
        for row in parameter_tensors["rows"]:
            shutil.copy2(
                scratch / "parameter_tensors" / row["file"],
                published_parameters / row["file"],
            )

        # The exact packed event/state archive is small enough to publish.  It
        # is the evidence behind results.json, not an ignored local cache.
        archive_dir = staging / "condition_archives"
        archive_dir.mkdir(exist_ok=True)
        archive_bytes = 0
        seed_ledger = []
        for row in results:
            condition = scratch / "conditions" / row["condition_key"]
            source = condition / "compact.npz"
            target = archive_dir / f"{row['condition_key']}.npz"
            shutil.copy2(source, target)
            archive_bytes += target.stat().st_size
            summary = _json_load(condition / "summary.json")
            if row["coupling"] == 0.0:
                seed_ledger.extend(summary["seed_ledger"])
        _json_dump(staging / "seed_ledger.json", {
            "schema": "pinglab.exp078.seed-ledger/v1",
            "network_seed": NETWORK_SEED,
            "pairing_rule": "each detuning/trial input pair is reused across all coupling levels",
            "rows": seed_ledger,
        })
        representatives_dir = staging / "representatives"
        representatives_dir.mkdir(exist_ok=True)
        for row in selected:
            source = scratch / "conditions" / row["condition_key"]
            shutil.copy2(source / "traces.npz", representatives_dir / f"{row['label'].replace(' ', '_')}_traces.npz")

        runtimes = [row["runtime_s"] for row in results]
        dense_bytes = sum(row["dense_recording_bytes"] for row in results)
        benchmark = {
            "completed_cells": len(results),
            "completed_trials": len(results) * TRIALS,
            "median_cell_runtime_s": float(np.median(runtimes)),
            "total_cell_runtime_s": float(np.sum(runtimes)),
            "projected_dense_recording_bytes": dense_bytes,
            "published_compact_archive_bytes": archive_bytes,
            "compression_ratio": dense_bytes / archive_bytes,
        }
        _json_dump(staging / "benchmark.json", benchmark)

        graph = bundle.graph
        write_numbers(
            staging,
            run_id=run_id,
            duration_s=time.monotonic() - started,
            payload={
                "stage": "complete gated Arnold-tongue sweep",
                "graph": {
                    "name": graph["name"],
                    "digest": bundle.manifest["graph_digest"],
                    "populations": len(graph["populations"]),
                    "projections": len(graph["projections"]),
                    "observables": len(graph["observables"]),
                },
                "config": {
                    **SCALE,
                    "input_weight": INPUT_WEIGHT,
                    "coupling_reference": COUPLING_REFERENCE,
                    "delay_ms": DELAY_MS,
                },
                "simulation_executed": True,
                "registration": {
                    "calibrated_rate_interval_hz": calibration["operating_interval_hz"],
                    "locking_tolerances": tolerances,
                    "pilot_maximum_coupling": pilot["selected_maximum_coupling"],
                    "primary_grid_sha256": grid["sha256"],
                    "realized_parameter_tensor_sets": len(parameter_tensors["rows"]),
                    "parameters_verified_invariant_across_couplings": parameter_tensors[
                        "verified_invariant_across_couplings"
                    ],
                },
                "benchmark": benchmark,
                "valid_trial_fraction": float(np.mean([
                    trial["valid"] for row in results for trial in row["trials"]
                ])),
                "conclusion": conclusion,
                "representative_conditions": [
                    {
                        "label": row["label"],
                        "condition_key": row["condition_key"],
                        "measured_detuning_hz": row["measured_detuning_hz"],
                        "coupling": row["coupling"],
                    }
                    for row in selected
                ],
                "finite_size_followup": (
                    "consider reduced 800/200 confirmation because 80/20 did not pass"
                    if not conclusion["passed"]
                    else "not required by the prespecified 80/20 pass criteria"
                ),
            },
        )


if __name__ == "__main__":
    main()
