"""Experiment 078 — graph-native reciprocal gamma coupling.

The calibration grid, deterministic selection, sweep, and acceptance thresholds
below are registered before execution. Coupling variants differ only in graph data.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "snn"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from execution import ExecutionSpec, plan_graph, simulate  # noqa: E402
from tools import snnlang as snn  # noqa: E402, TID251

from helpers import theme  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp078"
SEED = 78
DT_MS = 0.1
N_E, N_I, N_INPUT = 40, 10, 16
STEPS = 18_000
TRANSIENT_STEPS = 4_000
ANALYSIS_SECONDS = (STEPS - TRANSIENT_STEPS) * DT_MS / 1000.0
GAMMA_BAND_HZ = (30.0, 80.0)
PHASE_BAND_HZ = (25.0, 90.0)
SMOOTH_MS = 5.0

# Registered bounded calibration: input settings only; within-circuit weights
# remain the reusable component defaults. Candidate order breaks score ties.
CALIBRATION_GRID = tuple(
    {"input_weight": weight, "rate_a_hz": rate_a, "rate_b_hz": rate_b}
    for weight in (0.20, 0.30, 0.40, 0.55)
    for rate_a, rate_b in ((180.0, 150.0), (220.0, 180.0), (260.0, 210.0))
)
COUPLING_STRENGTHS = (0.20, 0.50, 1.00, 2.00, 4.00)
DELAY_LABELS = ("short", "intermediate", "half_period")

ACTIVITY = {
    "min_e_rate_hz": 2.0,
    "max_e_rate_hz": 180.0,
    "min_i_rate_hz": 2.0,
    "max_i_rate_hz": 250.0,
    "min_active_fraction": 0.25,
    "max_active_fraction": 1.0,
    "min_peak_prominence": 2.0,
}
LOCKING = {
    "max_frequency_difference_fraction_of_baseline": 0.50,
    "min_plv_gain": 0.15,
    "min_coherence_gain": 0.10,
    "min_half_window_plv": 0.55,
    "max_half_window_phase_offset_difference_rad": 0.60,
}
HYPOTHESIS = (
    "Moderate reciprocal I-to-E GABA coupling entrains two independently driven, "
    "active, 5–15% detuned PING circuits, whereas zero coupling permits phase drift "
    "and the largest coupling may suppress activity."
)
SUCCESS = (
    "At least one moderate registered condition is active and locked by every "
    "frequency, PLV, coherence, and stable-phase threshold, with a contiguous "
    "strength/delay neighbour supporting the transition."
)
KILL = (
    "Kill if calibration finds no valid pair, coupling only suppresses/invalidates "
    "activity, or no registered condition meets every locking threshold."
)
GOAL_PROMPT = (REPO / "experiments" / "exp078_goal.txt").read_text()


@dataclass(frozen=True)
class Variant:
    name: str
    strength: float
    delay_label: str
    delay_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("smoke", "calibrate", "sweep"), required=True
    )
    return parser.parse_args()


def independent_inputs(
    rate_a_hz: float, rate_b_hz: float, *, steps: int = STEPS
) -> dict[str, torch.Tensor]:
    generators = (
        torch.Generator().manual_seed(SEED + 101),
        torch.Generator().manual_seed(SEED + 202),
    )
    a = (
        torch.rand((steps, 1, N_INPUT), generator=generators[0])
        < rate_a_hz * DT_MS / 1000.0
    ).float()
    b = (
        torch.rand((steps, 1, N_INPUT), generator=generators[1])
        < rate_b_hz * DT_MS / 1000.0
    ).float()
    return {"drive_a": a, "drive_b": b}


def author_graph(
    *, input_weight: float, coupling_strength: float = 0.0, delay_ms: float = 0.1
) -> snn.Bundle:
    net = snn.Network("two_ping_gamma_coupling", dt=DT_MS * snn.ms)
    drives = {
        name: net.input(
            f"drive_{name}",
            shape=("time", "batch", N_INPUT),
            signal_type="spikes",
            unit="spike",
        )
        for name in ("a", "b")
    }
    cells = {}
    for name in ("a", "b"):
        cell = snn.components.ping(net, name=name, n_e=N_E, n_i=N_I, source=None)
        net.connect(
            drives[name],
            cell.E.excitatory,
            name=f"{name}_input",
            synapse=snn.AMPA(tau=2 * snn.ms),
            weight=snn.Normal(input_weight, 0.03),
            constraint=snn.NonNegative(),
        )
        cells[name] = cell
    if coupling_strength > 0:
        for source, target in (("a", "b"), ("b", "a")):
            net.connect(
                cells[source].I.spikes,
                cells[target].E.inhibitory,
                name=f"{source}_I_to_{target}_E",
                synapse=snn.GABA(tau=9 * snn.ms),
                weight=snn.Constant(coupling_strength),
                constraint=snn.NonNegative(),
                connection="feedback",
                delay=delay_ms * snn.ms,
            )
    net.expose(
        cells["a"].E.spikes,
        cells["a"].I.spikes,
        cells["b"].E.spikes,
        cells["b"].I.spikes,
        name="population",
    )
    return snn.compile(net, target="tools/snn")


def gaussian_rate(spikes: np.ndarray) -> np.ndarray:
    population_hz = spikes.mean(axis=(1, 2)) * 1000.0 / DT_MS
    sigma = SMOOTH_MS / DT_MS
    width = int(6 * sigma) | 1
    kernel = signal.windows.gaussian(width, sigma)
    kernel /= kernel.sum()
    return np.convolve(population_hz, kernel, mode="same")


def spectrum(rate: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    fs = 1000.0 / DT_MS
    frequency, power = signal.welch(
        rate, fs=fs, nperseg=min(8192, len(rate)), noverlap=min(4096, len(rate) // 2)
    )
    mask = (frequency >= GAMMA_BAND_HZ[0]) & (frequency <= GAMMA_BAND_HZ[1])
    band_power = power[mask]
    if not np.any(mask) or band_power.size == 0 or float(np.median(band_power)) <= 0:
        return frequency, power, math.nan, 0.0
    peak_i = int(np.argmax(band_power))
    return (
        frequency,
        power,
        float(frequency[mask][peak_i]),
        float(band_power[peak_i] / np.median(band_power)),
    )


def circular_metrics(rate_a: np.ndarray, rate_b: np.ndarray) -> dict[str, float]:
    fs = 1000.0 / DT_MS
    sos = signal.butter(4, PHASE_BAND_HZ, btype="bandpass", fs=fs, output="sos")
    filtered_a = signal.sosfiltfilt(sos, rate_a)
    filtered_b = signal.sosfiltfilt(sos, rate_b)
    phase = np.angle(signal.hilbert(filtered_a)) - np.angle(signal.hilbert(filtered_b))
    vector = np.exp(1j * phase)
    midpoint = len(phase) // 2
    halves = [vector[:midpoint], vector[midpoint:]]
    half_offsets = [float(np.angle(np.mean(x))) for x in halves]
    offset_delta = float(
        abs(np.angle(np.exp(1j * (half_offsets[0] - half_offsets[1]))))
    )
    f_coh, coh = signal.coherence(rate_a, rate_b, fs=fs, nperseg=min(8192, len(rate_a)))
    coh_mask = (f_coh >= GAMMA_BAND_HZ[0]) & (f_coh <= GAMMA_BAND_HZ[1])
    centered_a, centered_b = rate_a - rate_a.mean(), rate_b - rate_b.mean()
    corr = signal.correlate(centered_a, centered_b, mode="full", method="fft")
    denom = np.linalg.norm(centered_a) * np.linalg.norm(centered_b)
    corr = corr / denom if denom else np.zeros_like(corr)
    lags = signal.correlation_lags(len(centered_a), len(centered_b), mode="full")
    window = abs(lags * DT_MS) <= 50.0
    peak = int(np.argmax(corr[window]))
    return {
        "plv": float(abs(np.mean(vector))),
        "mean_phase_difference_rad": float(np.angle(np.mean(vector))),
        "half_1_plv": float(abs(np.mean(halves[0]))),
        "half_2_plv": float(abs(np.mean(halves[1]))),
        "half_phase_offset_difference_rad": offset_delta,
        # Frozen after uncoupled calibration: a maximum-over-band statistic
        # saturated at 0.992 and could not support the declared gain threshold.
        "gamma_coherence": float(np.mean(coh[coh_mask])) if np.any(coh_mask) else 0.0,
        "cross_correlation_peak": float(corr[window][peak]),
        "cross_correlation_lag_ms": float(lags[window][peak] * DT_MS),
    }


def analyse(recordings: dict[str, torch.Tensor]) -> tuple[dict, dict[str, np.ndarray]]:
    arrays = {key: value.detach().cpu().numpy() for key, value in recordings.items()}
    post = {
        key: arrays[key][TRANSIENT_STEPS:]
        for key in ("population_0", "population_1", "population_2", "population_3")
    }
    rates = {key: gaussian_rate(value) for key, value in post.items()}
    populations = {}
    valid = True
    for circuit, e_key, i_key in (
        ("a", "population_0", "population_1"),
        ("b", "population_2", "population_3"),
    ):
        freq, power, peak_hz, prominence = spectrum(rates[e_key])
        e_rate = float(post[e_key].sum() / (N_E * ANALYSIS_SECONDS))
        i_rate = float(post[i_key].sum() / (N_I * ANALYSIS_SECONDS))
        e_active = float(np.mean(post[e_key].sum(axis=(0, 1)) > 0))
        i_active = float(np.mean(post[i_key].sum(axis=(0, 1)) > 0))
        finite = bool(
            np.isfinite(arrays[f"{circuit}_E.voltage"]).all()
            and np.isfinite(arrays[f"{circuit}_I.voltage"]).all()
        )
        active = bool(
            finite
            and ACTIVITY["min_e_rate_hz"] <= e_rate <= ACTIVITY["max_e_rate_hz"]
            and ACTIVITY["min_i_rate_hz"] <= i_rate <= ACTIVITY["max_i_rate_hz"]
            and e_active >= ACTIVITY["min_active_fraction"]
            and i_active >= ACTIVITY["min_active_fraction"]
        )
        spectral = bool(
            np.isfinite(peak_hz)
            and GAMMA_BAND_HZ[0] <= peak_hz <= GAMMA_BAND_HZ[1]
            and prominence >= ACTIVITY["min_peak_prominence"]
        )
        valid &= active and spectral
        populations[circuit] = {
            "e_rate_hz": e_rate,
            "i_rate_hz": i_rate,
            "e_active_fraction": e_active,
            "i_active_fraction": i_active,
            "dominant_frequency_hz": peak_hz,
            "peak_prominence": prominence,
            "finite": finite,
            "active": active,
            "spectrally_valid": spectral,
            "silent": e_rate < ACTIVITY["min_e_rate_hz"]
            or i_rate < ACTIVITY["min_i_rate_hz"],
            "saturated": e_rate > ACTIVITY["max_e_rate_hz"]
            or i_rate > ACTIVITY["max_i_rate_hz"],
        }
        rates[f"{circuit}_spectrum_frequency"] = freq
        rates[f"{circuit}_spectrum_power"] = power
    synchrony = circular_metrics(rates["population_0"], rates["population_2"])
    synchrony["frequency_difference_hz"] = abs(
        populations["a"]["dominant_frequency_hz"]
        - populations["b"]["dominant_frequency_hz"]
    )
    archived_rates = {f"rate_{key}": value for key, value in rates.items()}
    return {"populations": populations, "synchrony": synchrony, "valid": valid}, {
        **arrays,
        **archived_rates,
    }


def run_condition(
    settings: dict, *, coupling_strength: float = 0.0, delay_ms: float = 0.1
) -> tuple[dict, dict[str, np.ndarray], snn.Bundle]:
    inputs = independent_inputs(settings["rate_a_hz"], settings["rate_b_hz"])
    bundle = author_graph(
        input_weight=settings["input_weight"],
        coupling_strength=coupling_strength,
        delay_ms=delay_ms,
    )
    result = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=bundle.graph,
            inputs=inputs,
            seed=SEED,
        )
    )
    metrics, arrays = analyse(result.recordings)
    metrics["runtime_s"] = result.metrics["simulate_s"]
    metrics["peak_python_bytes"] = result.metrics["peak_python_bytes"]
    metrics["graph_digest"] = bundle.manifest["graph_digest"]
    return metrics, arrays, bundle


def calibration_score(row: dict) -> float:
    if not row["metrics"]["valid"] or not 0.05 <= row["detuning_fraction"] <= 0.15:
        return math.inf
    pops = row["metrics"]["populations"]
    return abs(row["detuning_fraction"] - 0.10) + 0.01 / min(
        pops["a"]["peak_prominence"], pops["b"]["peak_prominence"]
    )


def registration() -> dict:
    return {
        "hypothesis": HYPOTHESIS,
        "success_criterion": SUCCESS,
        "kill_criterion": KILL,
        "seed": SEED,
        "simulation": {
            "dt_ms": DT_MS,
            "duration_ms": STEPS * DT_MS,
            "transient_ms": TRANSIENT_STEPS * DT_MS,
            "analysis_seconds": ANALYSIS_SECONDS,
        },
        "populations": {
            "excitatory_per_circuit": N_E,
            "inhibitory_per_circuit": N_I,
            "input_channels_per_circuit": N_INPUT,
        },
        "independent_input_generators": [SEED + 101, SEED + 202],
        "calibration_grid": list(CALIBRATION_GRID),
        "calibration_selection": "Minimum registered score among valid active/spectral candidates with 5–15% detuning; score is |detuning-10%| + 0.01/min peak prominence; grid order breaks exact ties.",
        "within_circuit_parameters": "snnlang PING component defaults, frozen",
        "coupling_strengths": list(COUPLING_STRENGTHS),
        "delay_labels": list(DELAY_LABELS),
        "activity_thresholds": ACTIVITY,
        "locking_thresholds": LOCKING,
        "coherence_definition": "Arithmetic mean of magnitude-squared coherence bins from 30 through 80 Hz; frozen after uncoupled calibration and before any coupling sweep because a maximum-over-band statistic saturated at 0.992.",
        "locked_definition": "active plus <=50% baseline frequency difference, PLV gain >=0.15, mean-band coherence gain >=0.10, both half-window PLV >=0.55, and half-window phase-offset drift <=0.60 rad",
        "contiguity_definition": "A locked condition has at least one active locked orthogonal neighbour in the registered strength x delay grid; baseline is excluded.",
        "suppression_definition": "Either circuit violates the registered minimum E/I rates or active-cell fraction; suppression is never locking.",
    }


def smoke() -> None:
    settings = CALIBRATION_GRID[0]
    inputs = independent_inputs(settings["rate_a_hz"], settings["rate_b_hz"], steps=300)
    bundle = author_graph(input_weight=settings["input_weight"])
    first = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=bundle.graph,
            inputs=inputs,
            seed=SEED,
        )
    )
    second = simulate(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=bundle.graph,
            inputs=inputs,
            seed=SEED,
        )
    )
    keys = {
        "population_0",
        "population_1",
        "population_2",
        "population_3",
        "a_E.voltage",
        "a_I.voltage",
        "b_E.voltage",
        "b_I.voltage",
    }
    assert keys <= first.recordings.keys()
    assert all(
        torch.equal(first.recordings[key], second.recordings[key])
        for key in first.recordings
    )
    coupled = author_graph(
        input_weight=settings["input_weight"], coupling_strength=1.0, delay_ms=0.5
    )
    plan = plan_graph(coupled.graph)
    cross = [p for p in plan.projections if p.id in {"a_I_to_b_E", "b_I_to_a_E"}]
    assert len(cross) == 2 and {p.delay_steps for p in cross} == {5}
    print(
        json.dumps(
            {
                "finite": all(
                    torch.isfinite(x).all() for x in first.recordings.values()
                ),
                "recordings": sorted(keys),
                "reproducible": True,
                "coupling_graph_only": True,
                "delay_steps": 5,
            },
            indent=2,
        )
    )


def calibrate() -> None:
    started = time.monotonic()
    run_id = next_run_id(SLUG)
    with published_run(
        SLUG,
        run_id,
        scale={
            "stage": "calibration",
            "candidates": len(CALIBRATION_GRID),
            "seed": SEED,
        },
    ) as (_scratch, staging):
        rows = []
        for index, settings in enumerate(CALIBRATION_GRID):
            metrics, _arrays, _bundle = run_condition(settings)
            fa = metrics["populations"]["a"]["dominant_frequency_hz"]
            fb = metrics["populations"]["b"]["dominant_frequency_hz"]
            detuning = (
                abs(fa - fb) / ((fa + fb) / 2)
                if np.isfinite(fa + fb) and fa + fb
                else math.inf
            )
            row = {
                "index": index,
                "settings": settings,
                "metrics": metrics,
                "detuning_fraction": detuning,
            }
            row["selection_score"] = calibration_score(row)
            rows.append(row)
            print(index, settings, metrics["valid"], fa, fb, detuning)
        valid = [row for row in rows if math.isfinite(row["selection_score"])]
        selected = (
            min(valid, key=lambda row: (row["selection_score"], row["index"]))
            if valid
            else None
        )
        (staging / "registration.json").write_text(
            json.dumps(registration(), indent=2) + "\n"
        )
        (staging / "calibration_candidates.json").write_text(
            json.dumps(rows, indent=2) + "\n"
        )
        (staging / "calibration_selection.json").write_text(
            json.dumps(selected, indent=2) + "\n"
        )
        (staging / "goal.txt").write_text(GOAL_PROMPT)
        (staging / "reproduce.sh").write_text(
            "#!/bin/sh\nuv run python experiments/exp078.py --stage calibrate\n"
        )
        activity = [
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": "Completed the bounded uncoupled calibration locally; selection used only registered input settings and no cross-coupling.",
            }
        ]
        (staging / "activity_log.json").write_text(
            json.dumps(activity, indent=2) + "\n"
        )
        payload = {
            "stage": "calibration",
            "registration": registration(),
            "calibration": {
                "candidate_count": len(rows),
                "valid_candidate_count": len(valid),
                "selected": selected,
            },
            "exit": {
                "calibration_pass": selected is not None,
                "sweep_performed": False,
                "paid_compute_usd": 0.0,
            },
        }
        write_numbers(
            staging,
            run_id=run_id,
            duration_s=time.monotonic() - started,
            payload=payload,
        )
    if selected is None:
        raise SystemExit("KILLED: no registered calibration candidate passed")


def variants(selected: dict) -> list[Variant]:
    mean_frequency = np.mean(
        [
            selected["metrics"]["populations"][c]["dominant_frequency_hz"]
            for c in ("a", "b")
        ]
    )
    half_period = round((500.0 / mean_frequency) / DT_MS) * DT_MS
    delays = {"short": 0.1, "intermediate": 2.0, "half_period": half_period}
    return [Variant("uncoupled", 0.0, "none", 0.0)] + [
        Variant(f"w{strength:g}_{label}", strength, label, delays[label])
        for strength in COUPLING_STRENGTHS
        for label in DELAY_LABELS
    ]


def locked(metrics: dict, baseline: dict) -> bool:
    sync, base = metrics["synchrony"], baseline["synchrony"]
    return bool(
        metrics["valid"]
        and sync["frequency_difference_hz"]
        <= LOCKING["max_frequency_difference_fraction_of_baseline"]
        * base["frequency_difference_hz"]
        and sync["plv"] - base["plv"] >= LOCKING["min_plv_gain"]
        and sync["gamma_coherence"] - base["gamma_coherence"]
        >= LOCKING["min_coherence_gain"]
        and min(sync["half_1_plv"], sync["half_2_plv"])
        >= LOCKING["min_half_window_plv"]
        and sync["half_phase_offset_difference_rad"]
        <= LOCKING["max_half_window_phase_offset_difference_rad"]
    )


def sweep() -> None:
    source = REPO / "artifacts" / "data" / SLUG / "calibration_selection.json"
    if not source.exists():
        raise SystemExit("run registered calibration first")
    selected = json.loads(source.read_text())
    if selected is None:
        raise SystemExit("KILLED: registered calibration did not pass")
    settings = selected["settings"]
    started = time.monotonic()
    run_id = next_run_id(SLUG)
    sweep_variants = variants(selected)
    with published_run(
        SLUG,
        run_id,
        scale={
            "stage": "registered_sweep",
            "variants": len(sweep_variants),
            "seed": SEED,
        },
    ) as (_scratch, staging):
        variants_dir = staging / "variants"
        variants_dir.mkdir()
        rows, arrays_by_name = [], {}
        for variant in sweep_variants:
            metrics, arrays, bundle = run_condition(
                settings,
                coupling_strength=variant.strength,
                delay_ms=max(variant.delay_ms, DT_MS),
            )
            bundle.write(variants_dir / f"{variant.name}.bundle", visualise=True)
            np.savez_compressed(
                variants_dir / f"{variant.name}-recordings.npz", **arrays
            )
            row = {"variant": asdict(variant), "metrics": metrics}
            rows.append(row)
            arrays_by_name[variant.name] = arrays
            print(variant.name, metrics["valid"], metrics["synchrony"])
        baseline = rows[0]["metrics"]
        for row in rows:
            row["locked"] = (
                False if row is rows[0] else locked(row["metrics"], baseline)
            )
        grid = {
            (row["variant"]["strength"], row["variant"]["delay_label"]): row
            for row in rows[1:]
        }
        for row in rows[1:]:
            strength, label = row["variant"]["strength"], row["variant"]["delay_label"]
            si, di = COUPLING_STRENGTHS.index(strength), DELAY_LABELS.index(label)
            neighbours = []
            for sj, dj in ((si - 1, di), (si + 1, di), (si, di - 1), (si, di + 1)):
                if 0 <= sj < len(COUPLING_STRENGTHS) and 0 <= dj < len(DELAY_LABELS):
                    neighbours.append(grid[(COUPLING_STRENGTHS[sj], DELAY_LABELS[dj])])
            row["contiguous_support"] = bool(
                row["locked"] and any(n["locked"] for n in neighbours)
            )
        winners = [
            row
            for row in rows[1:]
            if row["locked"]
            and row["contiguous_support"]
            and row["variant"]["strength"] < max(COUPLING_STRENGTHS)
        ]
        success = bool(winners)
        render_figures(rows, arrays_by_name, staging)
        shutil.copy2(
            variants_dir / "w1_intermediate.bundle/reports/circuit.svg",
            staging / "representative_graph.svg",
        )
        np.savez_compressed(
            staging / "inputs.npz",
            **{
                k: v.numpy()
                for k, v in independent_inputs(
                    settings["rate_a_hz"], settings["rate_b_hz"]
                ).items()
            },
        )
        (staging / "registration.json").write_text(
            json.dumps(registration(), indent=2) + "\n"
        )
        (staging / "calibration_candidates.json").write_text(
            (REPO / "artifacts/data/exp078/calibration_candidates.json").read_text()
        )
        (staging / "calibration_selection.json").write_text(
            json.dumps(selected, indent=2) + "\n"
        )
        (staging / "sweep_table.json").write_text(json.dumps(rows, indent=2) + "\n")
        (staging / "goal.txt").write_text(GOAL_PROMPT)
        (staging / "reproduce.sh").write_text(
            "#!/bin/sh\nuv run python experiments/exp078.py --stage calibrate\nuv run python experiments/exp078.py --stage sweep\n"
        )
        activity = [
            {
                "timestamp": "2026-08-05T13:34:17Z",
                "event": "Killed the first complete sweep publication after rate traces overwrote spike-array artifact keys and raster rendering failed; atomic staging prevented invalid evidence publication.",
            },
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": "Completed the unchanged pre-registered local coupling sweep after an experiment-side artifact-key fix, without simulator edits or paid compute.",
            },
        ]
        (staging / "activity_log.json").write_text(
            json.dumps(activity, indent=2) + "\n"
        )
        payload = {
            "stage": "registered_sweep",
            "registration": registration(),
            "calibration": {"selected": selected},
            "sweep": {
                "condition_count": len(rows),
                "rows": rows,
                "locked_count": sum(row["locked"] for row in rows),
                "contiguously_supported_count": sum(
                    row.get("contiguous_support", False) for row in rows
                ),
                "winning_conditions": winners,
            },
            "exit": {
                "success": success,
                "kill_fired": not success,
                "simulator_edits": 0,
                "paid_compute_usd": 0.0,
            },
            "activity": activity,
        }
        write_numbers(
            staging,
            run_id=run_id,
            duration_s=time.monotonic() - started,
            payload=payload,
        )


def render_figures(
    rows: list[dict], arrays: dict[str, dict[str, np.ndarray]], out: Path
) -> None:
    theme.apply()
    representative = [rows[0]]
    locked_rows = [row for row in rows[1:] if row["locked"]]
    representative.append(locked_rows[0] if locked_rows else rows[len(rows) // 2])
    representative.append(rows[-1])
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.2), sharex=True)
    for row_i, row in enumerate(representative):
        name = row["variant"]["name"]
        rec = arrays[name]
        for col, key in enumerate(("population_0", "population_2")):
            sample = rec[key][TRANSIENT_STEPS:]
            t, cell = np.nonzero(sample[:, 0])
            axes[row_i, col].scatter(
                t * DT_MS,
                cell,
                s=1.5,
                linewidths=0,
                color=theme.INK_BLACK if col == 0 else theme.DEEP_RED,
            )
            axes[row_i, col].set_ylabel(name.replace("_", " "), fontsize=7)
    axes[0, 0].set_title("circuit A · E")
    axes[0, 1].set_title("circuit B · E")
    for ax in axes[-1]:
        ax.set_xlabel("post-transient time (ms)")
    fig.tight_layout()
    fig.savefig(out / "matched_rasters.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True)
    for ax, row in zip(axes, representative):
        rec = arrays[row["variant"]["name"]]
        ax.plot(
            np.arange(len(rec["population_0"]) - TRANSIENT_STEPS) * DT_MS,
            gaussian_rate(rec["population_0"][TRANSIENT_STEPS:]),
            label="A",
            color=theme.INK_BLACK,
        )
        ax.plot(
            np.arange(len(rec["population_2"]) - TRANSIENT_STEPS) * DT_MS,
            gaussian_rate(rec["population_2"][TRANSIENT_STEPS:]),
            label="B",
            color=theme.DEEP_RED,
            alpha=0.8,
        )
        ax.set_ylabel(row["variant"]["name"].replace("_", " "), fontsize=7)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("post-transient time (ms)")
    fig.tight_layout()
    fig.savefig(out / "population_rates.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    metrics = (
        ("frequency_difference_hz", "frequency difference (Hz)"),
        ("plv", "phase-locking value"),
        ("gamma_coherence", "gamma coherence"),
        ("mean_phase_difference_rad", "phase offset (rad)"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    for ax, (metric, label) in zip(axes.flat, metrics):
        matrix = np.full((len(COUPLING_STRENGTHS), len(DELAY_LABELS)), np.nan)
        for row in rows[1:]:
            i = COUPLING_STRENGTHS.index(row["variant"]["strength"])
            j = DELAY_LABELS.index(row["variant"]["delay_label"])
            matrix[i, j] = row["metrics"]["synchrony"][metric]
        image = ax.imshow(matrix, aspect="auto", origin="lower")
        ax.set_xticks(
            range(len(DELAY_LABELS)),
            ["short", "intermediate", "half period"],
            rotation=20,
            ha="right",
        )
        ax.set_yticks(
            range(len(COUPLING_STRENGTHS)), [str(x) for x in COUPLING_STRENGTHS]
        )
        ax.set_ylabel("coupling strength")
        ax.set_title(label)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "coupling_heatmaps.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    {"smoke": smoke, "calibrate": calibrate, "sweep": sweep}[args.stage]()


if __name__ == "__main__":
    main()
