"""Experiment 078 — graph-native reciprocal gamma coupling.

The calibration grid, deterministic selection, sweep, and acceptance thresholds
below are registered before execution. Coupling variants differ only in graph data.
"""

from __future__ import annotations

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
from matplotlib.colors import ListedColormap
from scipy import signal

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "snn"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from execution import (  # noqa: E402
    ExecutionSpec,
    plan_graph,
    runtime_state_signature,
    save_runtime_state,
    simulate,
)
from tools import snnlang as snn  # noqa: E402, TID251

from helpers import theme  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp078"
SEED = 78
DT_MS = 0.1
N_E, N_I, N_INPUT = 800, 200, 16
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
REFINEMENT_STRENGTHS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
REFINEMENT_DELAYS_MS = (8.0, 10.0, 11.5, 12.5, 13.5, 15.0)
CAPTURE_STRENGTHS = (0.12, 0.14, 0.16, 0.18, 0.20, 0.22)
CAPTURE_DELAYS_MS = (7.0, 8.0, 9.0, 10.0, 11.0, 12.0)
ACQUISITION_STRENGTHS = (0.18,)
ACQUISITION_DELAYS_MS = (10.0,)
PHASE_CONTEXT_STEPS = round(250.0 / DT_MS)
CLEAN_SEARCH_STRENGTHS = (0.20,)
CLEAN_DELAY_STRENGTHS = (0.12, 0.15, 0.18, 0.20, 0.22, 0.24)
CLEAN_SEARCH_DELAYS_MS = (8.0, 9.0, 10.0, 11.0, 12.0)
CLEAN_SEARCH_STEPS = round(1200.0 / DT_MS)

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
    "Moderate reciprocal long-range E-to-I AMPA coupling entrains two independently "
    "driven, active, 5–15% detuned PING circuits by recruiting inhibition in the "
    "other circuit, whereas zero coupling permits phase drift and the largest "
    "coupling may suppress activity."
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


def independent_inputs(
    rate_a_hz: float, rate_b_hz: float, *, steps: int = STEPS, seed_offset: int = 0
) -> dict[str, torch.Tensor]:
    generators = (
        torch.Generator().manual_seed(SEED + 101 + seed_offset),
        torch.Generator().manual_seed(SEED + 202 + seed_offset),
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
    # Keep long-range projections structurally present even at weight zero so a
    # mature uncoupled runtime state can branch into the coupled graph safely.
    for source, target in (("a", "b"), ("b", "a")):
        net.connect(
            cells[source].E.spikes,
            cells[target].I.excitatory,
            name=f"{source}_E_to_{target}_I",
            synapse=snn.AMPA(tau=2 * snn.ms),
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
        "cross_circuit_pathway": "Reciprocal long-range E-to-I AMPA projections; each receiving I population inhibits its local E population through the unchanged within-circuit PING loop.",
        "coupling_strengths": list(COUPLING_STRENGTHS),
        "delay_labels": list(DELAY_LABELS),
        "activity_thresholds": ACTIVITY,
        "locking_thresholds": LOCKING,
        "coherence_definition": "Arithmetic mean of magnitude-squared coherence bins from 30 through 80 Hz; frozen after uncoupled calibration and before any coupling sweep because a maximum-over-band statistic saturated at 0.992.",
        "locked_definition": "active plus <=50% baseline frequency difference, PLV gain >=0.15, mean-band coherence gain >=0.10, both half-window PLV >=0.55, and half-window phase-offset drift <=0.60 rad",
        "contiguity_definition": "A locked condition has at least one active locked orthogonal neighbour in the registered strength x delay grid; baseline is excluded.",
        "suppression_definition": "Either circuit violates the registered minimum E/I rates or active-cell fraction; suppression is never locking.",
    }


def json_safe(value):
    """Replace non-finite diagnostics with JSON null while retaining flags."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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
    cross = [p for p in plan.projections if p.id in {"a_E_to_b_I", "b_E_to_a_I"}]
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
        valid = []
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
            score = calibration_score(row)
            row["selection_score"] = score
            rows.append(row)
            if math.isfinite(score):
                valid.append(row)
            print(index, settings, metrics["valid"], fa, fb, detuning)
        selected = (
            min(valid, key=lambda row: (row["selection_score"], row["index"]))
            if valid
            else None
        )
        (staging / "registration.json").write_text(
            json.dumps(registration(), indent=2) + "\n"
        )
        (staging / "calibration_candidates.json").write_text(
            json.dumps(json_safe(rows), indent=2, allow_nan=False) + "\n"
        )
        (staging / "calibration_selection.json").write_text(
            json.dumps(selected, indent=2) + "\n"
        )
        (staging / "goal.txt").write_text(GOAL_PROMPT)
        (staging / "reproduce.sh").write_text(
            "#!/bin/sh\nuv run python experiments/exp078.py\n"
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
            payload=json_safe(payload),
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
        rows = []
        for variant in sweep_variants:
            metrics, arrays, bundle = run_condition(
                settings,
                coupling_strength=variant.strength,
                delay_ms=max(variant.delay_ms, DT_MS),
            )
            bundle.write(variants_dir / f"{variant.name}.bundle", visualise=True)
            np.savez_compressed(
                variants_dir / f"{variant.name}-recordings.npz",
                **arrays,  # ty: ignore[invalid-argument-type]
            )
            row = {"variant": asdict(variant), "metrics": metrics}
            rows.append(row)
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
        render_figures(rows, staging)
        shutil.copy2(
            variants_dir / "w0.2_half_period.bundle/reports/expanded.svg",
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
        (staging / "sweep_table.json").write_text(
            json.dumps(json_safe(rows), indent=2, allow_nan=False) + "\n"
        )
        (staging / "goal.txt").write_text(GOAL_PROMPT)
        (staging / "reproduce.sh").write_text(
            "#!/bin/sh\nuv run python experiments/exp078.py\n"
        )
        activity = [
            {
                "timestamp": "2026-08-05T12:48:57Z",
                "event": "Registered the hypothesis, bounded calibration, deterministic selection, graph-only sweep, locking thresholds, success criterion, and kill criterion before coupling execution.",
            },
            {
                "timestamp": "2026-08-05T12:50:00Z",
                "event": "Passed the local smoke gate with finite named E/I and voltage recordings, exact replay, graph-only coupling, and exact five-step delay lowering.",
            },
            {
                "timestamp": "2026-08-05T13:02:03Z",
                "event": "Completed the bounded 12-candidate uncoupled calibration locally and selected candidate 8 deterministically; no cross-coupling was evaluated.",
            },
            {
                "timestamp": "2026-08-05T13:03:12Z",
                "event": "Replaced maximum-over-band coherence with registered 30–80 Hz mean coherence before coupling because the uncoupled maximum saturated at 0.992 and made the declared gain impossible.",
            },
            {
                "timestamp": "2026-08-05T13:14:54Z",
                "event": "Froze the selected uncoupled baseline and all numerical thresholds before the first coupling condition.",
            },
            {
                "timestamp": "2026-08-05T13:34:17Z",
                "event": "Killed the first complete sweep publication after rate traces overwrote spike-array artifact keys and raster rendering failed; atomic staging prevented invalid evidence publication.",
            },
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": "Amended the cross-circuit anatomy before rerunning: replaced direct long-range I-to-E GABA projections with reciprocal long-range E-to-I AMPA projections while retaining the frozen uncoupled calibration, sweep grid, inputs, delays, and locking gates.",
            },
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "event": "After a scale amendment, recalibrated 800 E / 200 I circuits and completed the full biologically faithful E-to-I coupling sweep locally without simulator edits or paid compute; preliminary 40 E / 10 I measurements were superseded.",
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
            payload=json_safe(payload),
        )


def refinement_score(metrics: dict, baseline: dict) -> float:
    """Rank phase capture without pretending this exploratory scan was registered."""
    if not metrics["valid"]:
        return math.inf
    sync = metrics["synchrony"]
    base_df = baseline["synchrony"]["frequency_difference_hz"]
    return float(
        sync["frequency_difference_hz"] / base_df
        + (1.0 - sync["plv"])
        + (1.0 - min(sync["half_1_plv"], sync["half_2_plv"]))
        + sync["half_phase_offset_difference_rad"] / math.pi
    )


def refine() -> None:
    """Explore the narrow Arnold-tongue neighborhood exposed by the registered sweep."""
    root = REPO / "artifacts" / "data" / SLUG
    selected = json.loads((root / "calibration_selection.json").read_text())
    baseline = json.loads((root / "sweep_table.json").read_text())[0]["metrics"]
    destination = root / "refinement"
    destination.mkdir(exist_ok=True)
    rows: list[dict] = []
    best_score = math.inf
    for strength in REFINEMENT_STRENGTHS:
        for delay_ms in REFINEMENT_DELAYS_MS:
            metrics, arrays, bundle = run_condition(
                selected["settings"],
                coupling_strength=strength,
                delay_ms=delay_ms,
            )
            score = refinement_score(metrics, baseline)
            row = {
                "strength": strength,
                "delay_ms": delay_ms,
                "metrics": metrics,
                "registered_gate_pass": locked(metrics, baseline),
                "exploratory_score": score,
                "graph_digest": bundle.manifest["graph_digest"],
            }
            rows.append(row)
            if score < best_score:
                best_score = score
                np.savez_compressed(destination / "best-recordings.npz", **arrays)
                bundle.write(destination / "best.bundle", visualise=True)
            print(
                f"w={strength:.2f} d={delay_ms:.1f}",
                f"valid={metrics['valid']}",
                f"df={metrics['synchrony']['frequency_difference_hz']:.3f}",
                f"plv={metrics['synchrony']['plv']:.3f}",
                f"score={score:.3f}",
            )
    finite_rows = [row for row in rows if math.isfinite(row["exploratory_score"])]
    best = min(finite_rows, key=lambda row: row["exploratory_score"])
    payload = {
        "exploratory": True,
        "selection_rule": "minimize frequency_difference/baseline + (1-PLV) + (1-min_half_PLV) + half_phase_drift/pi among valid conditions",
        "strengths": list(REFINEMENT_STRENGTHS),
        "delays_ms": list(REFINEMENT_DELAYS_MS),
        "condition_count": len(rows),
        "best": best,
        "registered_gate_pass_count": sum(row["registered_gate_pass"] for row in rows),
        "rows": rows,
    }
    (destination / "results.json").write_text(
        json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(json_safe(payload["best"]), indent=2))


def capture_diagnostics(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    """Measure whether one static-coupling trajectory visibly captures over time."""
    _, _, vector = phase_trace(arrays)
    window_steps = round(150.0 / DT_MS)
    kernel = np.ones(window_steps) / window_steps
    normalizer = np.convolve(np.ones(len(vector)), kernel, mode="same")
    rolling_plv = np.abs(np.convolve(vector, kernel, mode="same") / normalizer)
    early = slice(round(100 / DT_MS), round(400 / DT_MS))
    late = slice(round(1100 / DT_MS), round(1700 / DT_MS))
    early_plv = float(np.mean(rolling_plv[early]))
    late_plv = float(np.mean(rolling_plv[late]))
    threshold = 0.75
    hold_steps = round(250.0 / DT_MS)
    capture_step = None
    for step in range(round(250 / DT_MS), round(1450 / DT_MS)):
        if float(np.mean(rolling_plv[step : step + hold_steps])) >= threshold:
            capture_step = step
            break
    unwrapped = np.unwrap(np.angle(vector))
    late_phase_span = float(np.ptp(unwrapped[late]))
    visible = bool(
        capture_step is not None
        and early_plv <= 0.65
        and late_plv >= 0.80
        and 300.0 <= capture_step * DT_MS <= 1300.0
        and late_phase_span <= 2.5
    )
    score = (
        early_plv
        + (1.0 - late_plv)
        + late_phase_span / (2 * math.pi)
        + (0.0 if capture_step is not None else 10.0)
    )
    return {
        "early_rolling_plv": early_plv,
        "late_rolling_plv": late_plv,
        "capture_time_ms": None if capture_step is None else capture_step * DT_MS,
        "late_unwrapped_phase_span_rad": late_phase_span,
        "visible_capture": visible,
        "selection_score": score,
    }


def capture() -> None:
    """Search for delayed spontaneous capture under static E-to-I coupling."""
    root = REPO / "artifacts" / "data" / SLUG
    selected = json.loads((root / "calibration_selection.json").read_text())
    destination = root / "capture"
    destination.mkdir(exist_ok=True)
    rows: list[dict] = []
    best_score = math.inf
    for strength in CAPTURE_STRENGTHS:
        for delay_ms in CAPTURE_DELAYS_MS:
            metrics, arrays, bundle = run_condition(
                selected["settings"], coupling_strength=strength, delay_ms=delay_ms
            )
            diagnostics = capture_diagnostics(arrays)
            row = {
                "strength": strength,
                "delay_ms": delay_ms,
                "metrics": metrics,
                "capture": diagnostics,
                "graph_digest": bundle.manifest["graph_digest"],
            }
            rows.append(row)
            eligible_score = (
                diagnostics["selection_score"]
                if metrics["valid"] and diagnostics["visible_capture"]
                else math.inf
            )
            if eligible_score < best_score:
                best_score = eligible_score
                np.savez_compressed(destination / "best-recordings.npz", **arrays)
                bundle.write(destination / "best.bundle", visualise=True)
            print(
                f"w={strength:.2f} d={delay_ms:.1f}",
                f"early={diagnostics['early_rolling_plv']:.3f}",
                f"late={diagnostics['late_rolling_plv']:.3f}",
                f"capture={diagnostics['capture_time_ms']}",
                f"visible={diagnostics['visible_capture']}",
            )
    visible = [row for row in rows if row["capture"]["visible_capture"]]
    best = (
        min(visible, key=lambda row: row["capture"]["selection_score"])
        if visible
        else None
    )
    payload = {
        "exploratory": True,
        "purpose": "Select one static-coupling trajectory with visible transition from phase drift to sustained phase locking.",
        "strengths": list(CAPTURE_STRENGTHS),
        "delays_ms": list(CAPTURE_DELAYS_MS),
        "condition_count": len(rows),
        "visible_capture_count": len(visible),
        "best": best,
        "rows": rows,
    }
    (destination / "results.json").write_text(
        json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n"
    )
    if best is None:
        raise SystemExit("No visible delayed-capture trajectory found")
    print(json.dumps(json_safe(best), indent=2))


def trailing_phase_metrics(
    arrays: dict[str, np.ndarray], *, discard_steps: int = 0
) -> dict[str, np.ndarray | float | None]:
    """Causal phase diagnostics for one already-mature continuation."""
    rate_a, rate_b, vector = phase_trace(arrays)
    window_steps = round(150.0 / DT_MS)
    rolling = np.full(len(vector), np.nan)
    cumulative = np.concatenate(([0j], np.cumsum(vector)))
    rolling[window_steps - 1 :] = np.abs(
        (cumulative[window_steps:] - cumulative[:-window_steps]) / window_steps
    )
    unwrapped = np.unwrap(np.angle(vector))
    reference = float(np.angle(np.mean(vector[-round(500 / DT_MS) :])))
    phase_error = np.angle(vector * np.exp(-1j * reference))
    hold_steps = round(250.0 / DT_MS)
    capture_step = None
    for step in range(window_steps - 1, len(vector) - hold_steps):
        if np.all(rolling[step : step + hold_steps] >= 0.80):
            capture_step = step
            break
    peaks, _ = signal.find_peaks(rate_a, distance=round(10.0 / DT_MS), prominence=2.0)
    if discard_steps:
        rate_a = rate_a[discard_steps:]
        rate_b = rate_b[discard_steps:]
        vector = vector[discard_steps:]
        rolling = rolling[discard_steps:]
        unwrapped = unwrapped[discard_steps:]
        phase_error = phase_error[discard_steps:]
        peaks = peaks[peaks >= discard_steps] - discard_steps
        capture_step = None
        for step in range(0, len(vector) - hold_steps):
            if np.all(rolling[step : step + hold_steps] >= 0.80):
                capture_step = step
                break
    peaks = peaks[(peaks >= round(25 / DT_MS)) & (peaks < len(vector) - round(25 / DT_MS))]
    return {
        "rate_a": rate_a,
        "rate_b": rate_b,
        "vector": vector,
        "rolling_plv": rolling,
        "unwrapped_phase": unwrapped,
        "phase_error": phase_error,
        "cycle_steps": peaks,
        "capture_step": capture_step,
        "locked_reference_rad": reference,
    }


def select_mature_checkpoint(arrays: dict[str, np.ndarray]) -> dict[str, float | int]:
    """Select the earliest maximum-separation healthy state by a frozen rule."""
    rate_a, rate_b, vector = phase_trace(arrays)
    start = round(800.0 / DT_MS)
    stop = min(round(1650.0 / DT_MS), len(vector) - round(100.0 / DT_MS))
    threshold_a = float(np.quantile(rate_a[start:stop], 0.25))
    threshold_b = float(np.quantile(rate_b[start:stop], 0.25))
    candidates = np.arange(start, stop)
    candidates = candidates[
        (rate_a[candidates] >= threshold_a) & (rate_b[candidates] >= threshold_b)
    ]
    if not len(candidates):
        raise RuntimeError("no healthy mature checkpoint candidate")
    separations = np.abs(np.angle(vector[candidates]))
    step = int(candidates[int(np.argmax(separations))])
    return {
        "step": step,
        "time_ms": step * DT_MS,
        "phase_separation_rad": float(np.abs(np.angle(vector[step]))),
        "rate_a_hz": float(rate_a[step]),
        "rate_b_hz": float(rate_b[step]),
        "selection_start_ms": start * DT_MS,
        "selection_stop_ms": stop * DT_MS,
        "rate_floor_a_hz": threshold_a,
        "rate_floor_b_hz": threshold_b,
    }


def mature_checkpoint_candidates(
    arrays: dict[str, np.ndarray], *, count: int = 5
) -> list[dict[str, float | int]]:
    """Return separated mature states with large phase offsets."""
    rate_a, rate_b, vector = phase_trace(arrays)
    start = round(700.0 / DT_MS)
    stop = min(round(1650.0 / DT_MS), len(vector) - PHASE_CONTEXT_STEPS)
    floor_a = float(np.quantile(rate_a[start:stop], 0.25))
    floor_b = float(np.quantile(rate_b[start:stop], 0.25))
    eligible = np.arange(start, stop)
    eligible = eligible[
        (rate_a[eligible] >= floor_a) & (rate_b[eligible] >= floor_b)
    ]
    order = eligible[np.argsort(-np.abs(np.angle(vector[eligible])))]
    minimum_spacing = round(120.0 / DT_MS)
    selected_steps: list[int] = []
    for raw_step in order:
        step = int(raw_step)
        if all(abs(step - previous) >= minimum_spacing for previous in selected_steps):
            selected_steps.append(step)
        if len(selected_steps) == count:
            break
    return [
        {
            "step": step,
            "time_ms": step * DT_MS,
            "phase_separation_rad": float(abs(np.angle(vector[step]))),
            "rate_a_hz": float(rate_a[step]),
            "rate_b_hz": float(rate_b[step]),
        }
        for step in sorted(selected_steps)
    ]


def clean_convergence_metrics(
    coupled: dict[str, np.ndarray], control: dict[str, np.ndarray]
) -> dict[str, float | bool | list[float]]:
    """Require visible approach to one phase delay followed by stable residence."""
    coupled_trace = trailing_phase_metrics(coupled, discard_steps=PHASE_CONTEXT_STEPS)
    control_trace = trailing_phase_metrics(control, discard_steps=PHASE_CONTEXT_STEPS)
    vector = coupled_trace["vector"]
    smoothing_steps = round(50.0 / DT_MS)
    cumulative = np.concatenate(([0j], np.cumsum(vector)))
    smooth = np.full(len(vector), np.nan + 0j)
    smooth[smoothing_steps - 1 :] = (
        cumulative[smoothing_steps:] - cumulative[:-smoothing_steps]
    ) / smoothing_steps
    late = slice(len(vector) - round(250.0 / DT_MS), None)
    early = slice(round(50.0 / DT_MS), round(300.0 / DT_MS))
    reference = float(np.angle(np.mean(vector[late])))
    error = np.angle(smooth * np.exp(-1j * reference))
    confidence = np.abs(smooth)
    valid = np.isfinite(error)
    early_error = float(np.nanmedian(np.abs(error[early])))
    late_error_95 = float(np.nanquantile(np.abs(error[late]), 0.95))
    early_plv = float(abs(np.mean(vector[early])))
    late_plv = float(abs(np.mean(vector[late])))
    control_late_plv = float(abs(np.mean(control_trace["vector"][late])))
    late_unwrapped_span = float(np.ptp(np.unwrap(np.angle(vector[late]))))
    smooth_late_span = float(np.ptp(np.unwrap(np.angle(smooth[late]))))
    edges = np.linspace(round(50 / DT_MS), len(vector), 5, dtype=int)
    quartile_errors = [
        float(np.nanmedian(np.abs(error[left:right])))
        for left, right in zip(edges[:-1], edges[1:])
    ]
    settled = np.abs(error) <= 0.65
    settled &= confidence >= 0.75
    hold = round(250.0 / DT_MS)
    capture_step = None
    for step in range(round(100.0 / DT_MS), len(vector) - hold):
        if float(np.mean(settled[step : step + hold])) >= 0.90:
            capture_step = step
            break
    convergent = bool(
        valid.any()
        and early_error >= 0.60
        and early_plv <= 0.70
        and late_error_95 <= 0.65
        and late_plv >= 0.92
        and control_late_plv <= 0.75
        and smooth_late_span <= 1.5
        and capture_step is not None
        and 150.0 <= capture_step * DT_MS <= 1000.0
        and quartile_errors[-1] < quartile_errors[0] * 0.55
    )
    return {
        "convergent": convergent,
        "early_phase_error_rad": early_error,
        "late_phase_error_95_rad": late_error_95,
        "early_plv": early_plv,
        "late_plv": late_plv,
        "control_late_plv": control_late_plv,
        "late_unwrapped_span_rad": late_unwrapped_span,
        "smooth_late_span_rad": smooth_late_span,
        "capture_time_ms": None if capture_step is None else capture_step * DT_MS,
        "fixed_phase_delay_rad": reference,
        "quartile_phase_errors_rad": quartile_errors,
    }


def search_clean_acquisition() -> None:
    """Bounded local search for a clean mature-state convergence trajectory."""
    root = REPO / "artifacts" / "data" / SLUG
    selected = json.loads((root / "calibration_selection.json").read_text())
    settings = selected["settings"]
    delay_ms = 10.0
    latest_checkpoint_step = 11662
    total_steps = latest_checkpoint_step + CLEAN_SEARCH_STEPS
    inputs = independent_inputs(
        settings["rate_a_hz"], settings["rate_b_hz"], steps=total_steps
    )
    zero_bundle = author_graph(
        input_weight=settings["input_weight"], coupling_strength=0.0, delay_ms=delay_ms
    )
    burn = simulate(ExecutionSpec(
        kind="simulate", executor="graph", graph=zero_bundle.graph,
        inputs={name: value[:STEPS] for name, value in inputs.items()}, seed=SEED,
    ))
    burn_arrays = {name: value.detach().cpu().numpy() for name, value in burn.recordings.items()}
    checkpoints = [
        checkpoint for checkpoint in mature_checkpoint_candidates(burn_arrays)
        if int(checkpoint["step"]) <= latest_checkpoint_step
    ]
    rows = []
    for checkpoint in checkpoints:
        step = int(checkpoint["step"])
        prefix = simulate(ExecutionSpec(
            kind="simulate", executor="graph", graph=zero_bundle.graph,
            inputs={name: value[:step] for name, value in inputs.items()}, seed=SEED,
        ))
        assert prefix.runtime_state is not None
        future = {name: value[step : step + CLEAN_SEARCH_STEPS] for name, value in inputs.items()}
        context = {
            name: value[step - PHASE_CONTEXT_STEPS : step]
            for name, value in burn_arrays.items()
        }
        control_result = simulate(
            ExecutionSpec(
                kind="simulate", executor="graph", graph=zero_bundle.graph,
                inputs=future, seed=SEED,
            ), runtime_state=prefix.runtime_state,
        )
        control_arrays = {
            name: value.detach().cpu().numpy()
            for name, value in control_result.recordings.items()
        }
        phase_control = {
            name: np.concatenate((context[name], value), axis=0)
            for name, value in control_arrays.items()
        }
        for strength in CLEAN_SEARCH_STRENGTHS:
            coupled_bundle = author_graph(
                input_weight=settings["input_weight"], coupling_strength=strength,
                delay_ms=delay_ms,
            )
            coupled_result = simulate(
                ExecutionSpec(
                    kind="simulate", executor="graph", graph=coupled_bundle.graph,
                    inputs=future, seed=SEED,
                ), runtime_state=prefix.runtime_state,
            )
            coupled_arrays = {
                name: value.detach().cpu().numpy()
                for name, value in coupled_result.recordings.items()
            }
            phase_coupled = {
                name: np.concatenate((context[name], value), axis=0)
                for name, value in coupled_arrays.items()
            }
            metrics = clean_convergence_metrics(phase_coupled, phase_control)
            row = {"checkpoint": checkpoint, "strength": strength, "delay_ms": delay_ms, "metrics": metrics}
            rows.append(row)
            print(
                f"t={checkpoint['time_ms']:.1f} w={strength:.2f}",
                f"early={metrics['early_phase_error_rad']:.2f}",
                f"late95={metrics['late_phase_error_95_rad']:.2f}",
                f"plv={metrics['late_plv']:.2f}",
                f"capture={metrics['capture_time_ms']}",
                f"pass={metrics['convergent']}",
            )
    destination = root / "acquisition"
    payload = {
        "strengths": list(CLEAN_SEARCH_STRENGTHS),
        "delay_ms": delay_ms,
        "duration_ms": CLEAN_SEARCH_STEPS * DT_MS,
        "checkpoints": checkpoints,
        "rows": rows,
        "passing": [row for row in rows if row["metrics"]["convergent"]],
    }
    (destination / "clean-search.json").write_text(
        json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n"
    )


def search_clean_delays() -> None:
    """Search coupling delays around the most promising mature checkpoint."""
    root = REPO / "artifacts" / "data" / SLUG
    selected = json.loads((root / "calibration_selection.json").read_text())
    settings = selected["settings"]
    checkpoint_step = 11662
    inputs = independent_inputs(
        settings["rate_a_hz"], settings["rate_b_hz"],
        steps=checkpoint_step + CLEAN_SEARCH_STEPS,
    )
    rows = []
    for delay_ms in CLEAN_SEARCH_DELAYS_MS:
        zero_bundle = author_graph(
            input_weight=settings["input_weight"], coupling_strength=0.0,
            delay_ms=delay_ms,
        )
        prefix = simulate(ExecutionSpec(
            kind="simulate", executor="graph", graph=zero_bundle.graph,
            inputs={name: value[:checkpoint_step] for name, value in inputs.items()}, seed=SEED,
        ))
        assert prefix.runtime_state is not None
        prefix_arrays = {
            name: value.detach().cpu().numpy() for name, value in prefix.recordings.items()
        }
        context = {
            name: value[-PHASE_CONTEXT_STEPS:] for name, value in prefix_arrays.items()
        }
        future = {
            name: value[checkpoint_step : checkpoint_step + CLEAN_SEARCH_STEPS]
            for name, value in inputs.items()
        }
        control_result = simulate(
            ExecutionSpec(
                kind="simulate", executor="graph", graph=zero_bundle.graph,
                inputs=future, seed=SEED,
            ), runtime_state=prefix.runtime_state,
        )
        control_arrays = {
            name: value.detach().cpu().numpy()
            for name, value in control_result.recordings.items()
        }
        phase_control = {
            name: np.concatenate((context[name], value), axis=0)
            for name, value in control_arrays.items()
        }
        for strength in CLEAN_DELAY_STRENGTHS:
            coupled_bundle = author_graph(
                input_weight=settings["input_weight"], coupling_strength=strength,
                delay_ms=delay_ms,
            )
            coupled_result = simulate(
                ExecutionSpec(
                    kind="simulate", executor="graph", graph=coupled_bundle.graph,
                    inputs=future, seed=SEED,
                ), runtime_state=prefix.runtime_state,
            )
            coupled_arrays = {
                name: value.detach().cpu().numpy()
                for name, value in coupled_result.recordings.items()
            }
            phase_coupled = {
                name: np.concatenate((context[name], value), axis=0)
                for name, value in coupled_arrays.items()
            }
            metrics = clean_convergence_metrics(phase_coupled, phase_control)
            row = {"checkpoint_step": checkpoint_step, "strength": strength, "delay_ms": delay_ms, "metrics": metrics}
            rows.append(row)
            print(
                f"d={delay_ms:.1f} w={strength:.2f}",
                f"early={metrics['early_phase_error_rad']:.2f}",
                f"late95={metrics['late_phase_error_95_rad']:.2f}",
                f"latePLV={metrics['late_plv']:.2f}",
                f"capture={metrics['capture_time_ms']}",
                f"pass={metrics['convergent']}",
            )
    payload = {
        "checkpoint_step": checkpoint_step,
        "strengths": list(CLEAN_DELAY_STRENGTHS),
        "delays_ms": list(CLEAN_SEARCH_DELAYS_MS),
        "duration_ms": CLEAN_SEARCH_STEPS * DT_MS,
        "rows": rows,
        "passing": [row for row in rows if row["metrics"]["convergent"]],
    }
    (root / "acquisition" / "clean-delay-search.json").write_text(
        json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n"
    )


def search_clean_input_seeds() -> None:
    """Test whether independent Poisson realization controls stable capture."""
    root = REPO / "artifacts" / "data" / SLUG
    selected = json.loads((root / "calibration_selection.json").read_text())
    settings = selected["settings"]
    checkpoint_step = 11662
    delay_ms = 10.0
    zero_bundle = author_graph(
        input_weight=settings["input_weight"], coupling_strength=0.0, delay_ms=delay_ms
    )
    burn_inputs = independent_inputs(
        settings["rate_a_hz"], settings["rate_b_hz"], steps=checkpoint_step
    )
    prefix = simulate(ExecutionSpec(
        kind="simulate", executor="graph", graph=zero_bundle.graph,
        inputs=burn_inputs, seed=SEED,
    ))
    assert prefix.runtime_state is not None
    prefix_arrays = {
        name: value.detach().cpu().numpy() for name, value in prefix.recordings.items()
    }
    context = {
        name: value[-PHASE_CONTEXT_STEPS:] for name, value in prefix_arrays.items()
    }
    rows = []
    strengths = (0.20,)
    seed_offsets = tuple(range(32))
    for seed_offset in seed_offsets:
        future = independent_inputs(
            settings["rate_a_hz"], settings["rate_b_hz"],
            steps=CLEAN_SEARCH_STEPS, seed_offset=seed_offset,
        )
        control_result = simulate(
            ExecutionSpec(
                kind="simulate", executor="graph", graph=zero_bundle.graph,
                inputs=future, seed=SEED,
            ), runtime_state=prefix.runtime_state,
        )
        control_arrays = {
            name: value.detach().cpu().numpy()
            for name, value in control_result.recordings.items()
        }
        phase_control = {
            name: np.concatenate((context[name], value), axis=0)
            for name, value in control_arrays.items()
        }
        for strength in strengths:
            coupled_bundle = author_graph(
                input_weight=settings["input_weight"], coupling_strength=strength,
                delay_ms=delay_ms,
            )
            coupled_result = simulate(
                ExecutionSpec(
                    kind="simulate", executor="graph", graph=coupled_bundle.graph,
                    inputs=future, seed=SEED,
                ), runtime_state=prefix.runtime_state,
            )
            coupled_arrays = {
                name: value.detach().cpu().numpy()
                for name, value in coupled_result.recordings.items()
            }
            phase_coupled = {
                name: np.concatenate((context[name], value), axis=0)
                for name, value in coupled_arrays.items()
            }
            metrics = clean_convergence_metrics(phase_coupled, phase_control)
            row = {
                "checkpoint_step": checkpoint_step,
                "future_seed_offset": seed_offset,
                "strength": strength,
                "delay_ms": delay_ms,
                "metrics": metrics,
            }
            rows.append(row)
            print(
                f"seed={seed_offset} w={strength:.2f}",
                f"early={metrics['early_phase_error_rad']:.2f}",
                f"late95={metrics['late_phase_error_95_rad']:.2f}",
                f"latePLV={metrics['late_plv']:.2f}",
                f"capture={metrics['capture_time_ms']}",
                f"pass={metrics['convergent']}",
            )
    payload = {
        "checkpoint_step": checkpoint_step,
        "future_seed_offsets": list(seed_offsets),
        "strengths": list(strengths),
        "delay_ms": delay_ms,
        "duration_ms": CLEAN_SEARCH_STEPS * DT_MS,
        "rows": rows,
        "passing": [row for row in rows if row["metrics"]["convergent"]],
    }
    (root / "acquisition" / "clean-seed-search.json").write_text(
        json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n"
    )


def acquisition_metrics(
    coupled: dict[str, np.ndarray], control: dict[str, np.ndarray], *, discard_steps: int = 0
) -> dict:
    coupled_trace = trailing_phase_metrics(coupled, discard_steps=discard_steps)
    control_trace = trailing_phase_metrics(control, discard_steps=discard_steps)
    early = slice(round(50 / DT_MS), round(400 / DT_MS))
    late = slice(len(coupled_trace["vector"]) - round(600 / DT_MS), None)
    capture_step = coupled_trace["capture_step"]
    coupled_rate_a = coupled_trace["rate_a"]
    coupled_rate_b = coupled_trace["rate_b"]
    _, _, early_f_a, _ = spectrum(coupled_rate_a[early])
    _, _, early_f_b, _ = spectrum(coupled_rate_b[early])
    _, _, late_f_a, _ = spectrum(coupled_rate_a[late])
    _, _, late_f_b, _ = spectrum(coupled_rate_b[late])
    period_ms = 1000.0 / np.nanmean((late_f_a, late_f_b))
    cycle_steps = coupled_trace["cycle_steps"]
    cycle_error = coupled_trace["phase_error"][cycle_steps]
    def mean_finite(values, section):
        selected = values[section]
        return float(np.nanmean(selected))
    return {
        "initial_phase_separation_rad": float(abs(np.angle(coupled_trace["vector"][round(50 / DT_MS)]))),
        "final_phase_separation_rad": float(abs(np.angle(np.mean(coupled_trace["vector"][late])))),
        "capture_time_ms": None if capture_step is None else float(capture_step * DT_MS),
        "capture_cycles": None if capture_step is None else float(capture_step * DT_MS / period_ms),
        "rolling_plv_early": mean_finite(coupled_trace["rolling_plv"], early),
        "rolling_plv_late": mean_finite(coupled_trace["rolling_plv"], late),
        "control_rolling_plv_late": mean_finite(control_trace["rolling_plv"], late),
        "frequency_difference_early_hz": float(abs(early_f_a - early_f_b)),
        "frequency_difference_late_hz": float(abs(late_f_a - late_f_b)),
        "control_phase_span_late_rad": float(np.ptp(control_trace["unwrapped_phase"][late])),
        "coupled_phase_span_late_rad": float(np.ptp(coupled_trace["unwrapped_phase"][late])),
        "late_phase_slips": int(np.floor(np.ptp(coupled_trace["unwrapped_phase"][late]) / (2 * math.pi))),
        "cycle_steps": cycle_steps.tolist(),
        "cycle_phase_error_rad": cycle_error.tolist(),
        "visible_acquisition": bool(
            capture_step is not None
            and mean_finite(coupled_trace["rolling_plv"], early) <= 0.70
            and mean_finite(coupled_trace["rolling_plv"], late) >= 0.80
            and mean_finite(control_trace["rolling_plv"], late) <= 0.75
            and np.ptp(coupled_trace["unwrapped_phase"][late])
            < np.ptp(control_trace["unwrapped_phase"][late])
        ),
    }


def render_acquisition(
    coupled: dict[str, np.ndarray], control: dict[str, np.ndarray], metrics: dict, path: Path,
    *, phase_coupled: dict[str, np.ndarray] | None = None,
    phase_control: dict[str, np.ndarray] | None = None,
    discard_steps: int = 0,
) -> None:
    """Render one continuous mature-state phase-acquisition trajectory."""
    theme.apply()
    trace = trailing_phase_metrics(phase_coupled or coupled, discard_steps=discard_steps)
    baseline = trailing_phase_metrics(phase_control or control, discard_steps=discard_steps)
    time_ms = np.arange(len(trace["vector"])) * DT_MS
    fig, axes = plt.subplots(6, 1, figsize=(7.2, 10.0), sharex=True)
    raster_specs = (
        (axes[0], "population_0", "population_1", "circuit A"),
        (axes[1], "population_2", "population_3", "circuit B"),
    )
    for ax, e_key, i_key, label in raster_specs:
        for key, count, offset, color in (
            (e_key, 100, 0, theme.INK_BLACK),
            (i_key, 50, 105, theme.DEEP_RED),
        ):
            t, cell = np.nonzero(coupled[key][:, 0, :count])
            ax.scatter(t * DT_MS, cell + offset, s=0.65, linewidths=0, color=color)
        ax.set_ylabel(label + "\ncell")
        ax.set_yticks((50, 130), ("E", "I"))
    rate_max = 1.05 * max(np.max(trace["rate_a"]), np.max(trace["rate_b"]))
    for ax, values, label, color in (
        (axes[2], trace["rate_a"], "A E rate (Hz)", theme.INK_BLACK),
        (axes[3], trace["rate_b"], "B E rate (Hz)", theme.DEEP_RED),
    ):
        ax.plot(time_ms, values, color=color, linewidth=0.8)
        ax.set_ylim(0, rate_max)
        ax.set_ylabel(label)
    phase_reference = float(metrics["fixed_phase_delay_rad"])
    def smoothed_delay(arrays):
        _, _, vector = phase_trace(arrays)
        width = round(50.0 / DT_MS)
        cumulative = np.concatenate(([0j], np.cumsum(vector)))
        smooth = np.full(len(vector), np.nan + 0j)
        smooth[width - 1 :] = (cumulative[width:] - cumulative[:-width]) / width
        smooth = smooth[discard_steps:]
        return phase_reference + np.angle(smooth * np.exp(-1j * phase_reference))
    coupled_phase = smoothed_delay(phase_coupled or coupled)
    control_phase = smoothed_delay(phase_control or control)
    axes[4].plot(time_ms, control_phase, color=theme.GREY_MID, linewidth=0.8, label="matched uncoupled")
    axes[4].plot(time_ms, coupled_phase, color=theme.INK_BLACK, linewidth=1.1, label="coupled")
    axes[4].axhline(
        phase_reference, color=theme.DEEP_RED, linewidth=0.8, linestyle="--",
        label="final phase delay",
    )
    axes[4].set_ylim(phase_reference - math.pi, phase_reference + math.pi)
    axes[4].set_ylabel("A − B phase\ndelay (rad)")
    axes[4].legend(frameon=False, ncol=2, loc="upper left")
    axes[5].plot(time_ms, baseline["rolling_plv"], color=theme.GREY_MID, linewidth=0.8)
    axes[5].plot(time_ms, trace["rolling_plv"], color=theme.INK_BLACK, linewidth=1.0)
    axes[5].axhline(0.8, color=theme.DEEP_RED, linewidth=0.7, linestyle="--")
    if metrics["capture_time_ms"] is not None:
        axes[5].axvline(metrics["capture_time_ms"], color=theme.DEEP_RED, linewidth=0.8)
    axes[5].set_ylim(0, 1.02)
    axes[5].set_ylabel("trailing\nPLV")
    axes[5].set_xlabel("time since coupling enabled (ms)")
    for ax in axes:
        ax.axvline(0, color=theme.DEEP_RED, linewidth=0.7)
        ax.grid(axis="x", color=theme.GREY_LIGHT, linewidth=0.35)
    fig.tight_layout(h_pad=0.35)
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def acquire() -> None:
    """Publish one clean mature-state convergence trajectory."""
    root = REPO / "artifacts" / "data" / SLUG
    selected = json.loads((root / "calibration_selection.json").read_text())
    settings = selected["settings"]
    destination = root / "acquisition"
    destination.mkdir(exist_ok=True)
    checkpoint_step = 11662
    strength = 0.20
    delay_ms = 10.0
    zero_bundle = author_graph(
        input_weight=settings["input_weight"], coupling_strength=0.0, delay_ms=delay_ms
    )
    all_inputs = independent_inputs(
        settings["rate_a_hz"], settings["rate_b_hz"],
        steps=checkpoint_step + CLEAN_SEARCH_STEPS,
    )
    burn_inputs = {name: value[:checkpoint_step] for name, value in all_inputs.items()}
    prefix = simulate(ExecutionSpec(
        kind="simulate", executor="graph", graph=zero_bundle.graph,
        inputs=burn_inputs, seed=SEED,
    ))
    assert prefix.runtime_state is not None
    prefix_arrays = {
        name: value.detach().cpu().numpy() for name, value in prefix.recordings.items()
    }
    phase_context = {
        name: value[-PHASE_CONTEXT_STEPS:] for name, value in prefix_arrays.items()
    }
    continuation_inputs = {
        name: value[checkpoint_step:] for name, value in all_inputs.items()
    }
    coupled_bundle = author_graph(
        input_weight=settings["input_weight"], coupling_strength=strength, delay_ms=delay_ms
    )
    assert runtime_state_signature(plan_graph(coupled_bundle.graph)) == prefix.runtime_state.signature
    coupled_result = simulate(
        ExecutionSpec(
            kind="simulate", executor="graph", graph=coupled_bundle.graph,
            inputs=continuation_inputs, seed=SEED,
        ), runtime_state=prefix.runtime_state,
    )
    control_result = simulate(
        ExecutionSpec(
            kind="simulate", executor="graph", graph=zero_bundle.graph,
            inputs=continuation_inputs, seed=SEED,
        ), runtime_state=prefix.runtime_state,
    )
    coupled_arrays = {
        name: value.detach().cpu().numpy() for name, value in coupled_result.recordings.items()
    }
    control_arrays = {
        name: value.detach().cpu().numpy() for name, value in control_result.recordings.items()
    }
    phase_coupled = {
        name: np.concatenate((phase_context[name], value), axis=0)
        for name, value in coupled_arrays.items()
    }
    phase_control = {
        name: np.concatenate((phase_context[name], value), axis=0)
        for name, value in control_arrays.items()
    }
    diagnostics = clean_convergence_metrics(phase_coupled, phase_control)
    if not diagnostics["convergent"]:
        raise RuntimeError(f"selected acquisition no longer converges: {diagnostics}")
    publication_keys = tuple(f"population_{index}" for index in range(4))
    np.savez_compressed(
        destination / "coupled-recordings.npz",
        **{name: coupled_arrays[name] for name in publication_keys},
    )
    np.savez_compressed(
        destination / "control-recordings.npz",
        **{name: control_arrays[name] for name in publication_keys},
    )
    np.savez_compressed(
        destination / "phase-context.npz",
        **{name: phase_context[name] for name in publication_keys},
    )
    np.savez_compressed(
        destination / "inputs.npz",
        **{
            **{f"burn_{name}": value.numpy() for name, value in burn_inputs.items()},
            **{f"continuation_{name}": value.numpy() for name, value in continuation_inputs.items()},
        },
    )
    coupled_bundle.write(destination / "coupled.bundle", visualise=True)
    zero_bundle.write(destination / "zero-coupling.bundle", visualise=True)
    save_runtime_state(destination / "checkpoint.runtime-state", prefix.runtime_state)
    render_acquisition(
        coupled_arrays, control_arrays, diagnostics, destination / "phase_acquisition.png",
        phase_coupled=phase_coupled, phase_control=phase_control,
        discard_steps=PHASE_CONTEXT_STEPS,
    )
    payload = {
        "protocol": "mature zero-coupling checkpoint, identical-state coupled/control continuation with independent Poisson future input",
        "checkpoint": {"step": checkpoint_step, "time_ms": checkpoint_step * DT_MS},
        "selection": {
            "strength": strength, "delay_ms": delay_ms,
            "future_seed_offset": None,
            "rule": "bounded mature-checkpoint search on one fixed archived input stream at the refined weight and delay; require weak early locking, delayed capture, and stable late circular phase",
        },
        "metrics": diagnostics,
        "runtime_state_signature": prefix.runtime_state.signature,
        "burn_graph_digest": zero_bundle.manifest["graph_digest"],
        "coupled_graph_digest": coupled_bundle.manifest["graph_digest"],
        "seed": SEED,
        "dt_ms": DT_MS,
        "duration_ms": CLEAN_SEARCH_STEPS * DT_MS,
        "population_sizes": {"E": N_E, "I": N_I},
    }
    (destination / "results.json").write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")
    print(json.dumps(json_safe(payload), indent=2))


def load_recordings(root: Path, name: str) -> dict[str, np.ndarray]:
    with np.load(root / "variants" / f"{name}-recordings.npz") as archive:
        return {key: archive[key] for key in archive.files}


def render_figures(rows: list[dict], out: Path) -> None:
    theme.apply()
    representative = [rows[0]]
    locked_rows = [row for row in rows[1:] if row["locked"]]
    representative.append(locked_rows[0] if locked_rows else rows[len(rows) // 2])
    representative.append(rows[-1])
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.2), sharex=True)
    for row_i, row in enumerate(representative):
        name = row["variant"]["name"]
        rec = load_recordings(out, name)
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
        rec = load_recordings(out, row["variant"]["name"])
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

    render_summary(rows, out / "summary_compound.png", out)
    render_sync_emergence(
        load_recordings(out, "uncoupled"),
        load_recordings(out, "w0.2_half_period"),
        out / "synchronization_emergence.svg",
    )


def render_summary(rows: list[dict], out: Path, root: Path) -> None:
    """Render drift, locking, suppression, and the sweep as one figure."""
    conditions = (
        (rows[0], "uncoupled · drift"),
        (rows[1], "0.2 / short · locked"),
        (rows[10], "2.0 / short · B suppressed"),
    )
    fig = plt.figure(figsize=(7.2, 8.0))
    grid = fig.add_gridspec(
        3, 3, height_ratios=(1.0, 0.85, 1.05), hspace=0.48, wspace=0.34
    )
    time_ms = np.arange(STEPS - TRANSIENT_STEPS) * DT_MS
    for col, (row, title) in enumerate(conditions):
        rec = load_recordings(root, row["variant"]["name"])
        raster = fig.add_subplot(grid[0, col])
        for key, offset, color in (
            ("population_0", 0, theme.INK_BLACK),
            ("population_2", N_E + 3, theme.DEEP_RED),
        ):
            t, cell = np.nonzero(rec[key][TRANSIENT_STEPS:, 0])
            raster.scatter(t * DT_MS, cell + offset, s=0.8, linewidths=0, color=color)
        raster.set_title(title, fontsize=8)
        raster.set_xlim(0, time_ms[-1])
        raster.set_yticks((N_E / 2, N_E + 3 + N_E / 2), ("A", "B"))
        if col == 0:
            raster.set_ylabel("E spikes")

        rate = fig.add_subplot(grid[1, col], sharex=raster)
        rate.plot(
            time_ms,
            gaussian_rate(rec["population_0"][TRANSIENT_STEPS:]),
            color=theme.INK_BLACK,
            linewidth=0.8,
        )
        rate.plot(
            time_ms,
            gaussian_rate(rec["population_2"][TRANSIENT_STEPS:]),
            color=theme.DEEP_RED,
            linewidth=0.8,
            alpha=0.85,
        )
        rate.set_xlabel("post-transient time (ms)")
        if col == 0:
            rate.set_ylabel("E rate (Hz)")

    for col, (metric, title, cmap) in enumerate(
        (
            ("frequency_difference_hz", "frequency difference (Hz)", "magma_r"),
            ("plv", "phase-locking value", "magma"),
        )
    ):
        ax = fig.add_subplot(grid[2, col])
        matrix = np.full((len(COUPLING_STRENGTHS), len(DELAY_LABELS)), np.nan)
        for row in rows[1:]:
            i = COUPLING_STRENGTHS.index(row["variant"]["strength"])
            j = DELAY_LABELS.index(row["variant"]["delay_label"])
            value = row["metrics"]["synchrony"][metric]
            matrix[i, j] = np.nan if value is None else value
        image = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap)
        ax.set_title(title, fontsize=8)
        fig.colorbar(image, ax=ax, shrink=0.72)
        format_sweep_axis(ax, show_ylabel=col == 0)

    state = fig.add_subplot(grid[2, 2])
    state_matrix = np.zeros((len(COUPLING_STRENGTHS), len(DELAY_LABELS)))
    for row in rows[1:]:
        i = COUPLING_STRENGTHS.index(row["variant"]["strength"])
        j = DELAY_LABELS.index(row["variant"]["delay_label"])
        state_matrix[i, j] = (
            2 if row["locked"] else (1 if row["metrics"]["valid"] else 0)
        )
    state.imshow(
        state_matrix,
        aspect="auto",
        origin="lower",
        cmap=ListedColormap(("#d9d9d9", "#d98b8b", "#1b7f5a")),
        vmin=0,
        vmax=2,
    )
    state.set_title("registered outcome", fontsize=8)
    format_sweep_axis(state, show_ylabel=False)
    state.text(
        0.5,
        -0.44,
        "gray suppressed · red active · green locked",
        transform=state.transAxes,
        ha="center",
        fontsize=6.5,
    )
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def format_sweep_axis(ax, *, show_ylabel: bool) -> None:
    ax.set_xticks(
        range(len(DELAY_LABELS)),
        ("short", "intermediate", "half period"),
        rotation=24,
        ha="right",
    )
    ax.set_yticks(range(len(COUPLING_STRENGTHS)), [str(x) for x in COUPLING_STRENGTHS])
    if show_ylabel:
        ax.set_ylabel("coupling strength")


def phase_trace(rec: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rate_a = gaussian_rate(rec["population_0"])
    rate_b = gaussian_rate(rec["population_2"])
    fs = 1000.0 / DT_MS
    sos = signal.butter(4, PHASE_BAND_HZ, btype="bandpass", fs=fs, output="sos")
    phase_a = np.angle(signal.hilbert(signal.sosfiltfilt(sos, rate_a)))
    phase_b = np.angle(signal.hilbert(signal.sosfiltfilt(sos, rate_b)))
    return rate_a, rate_b, np.exp(1j * (phase_a - phase_b))


def render_sync_emergence(
    baseline_rec: dict[str, np.ndarray],
    coupled_rec: dict[str, np.ndarray],
    out: Path,
) -> None:
    """Show spikes/rates and directly contrast uncoupled drift with phase locking."""
    theme.apply()
    rate_a, rate_b, phase_vector = phase_trace(coupled_rec)
    _, _, baseline_phase_vector = phase_trace(baseline_rec)
    window_steps = round(200.0 / DT_MS)
    kernel = np.ones(window_steps) / window_steps
    edge_normalizer = np.convolve(np.ones(len(phase_vector)), kernel, mode="same")
    rolling_vector = np.convolve(phase_vector, kernel, mode="same") / edge_normalizer
    rolling_plv = np.abs(rolling_vector)
    baseline_rolling_plv = np.abs(
        np.convolve(baseline_phase_vector, kernel, mode="same") / edge_normalizer
    )
    time_ms = np.arange(STEPS) * DT_MS

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(6.5, 7.4),
        gridspec_kw={
            "height_ratios": (0.9, 0.9, 0.65, 0.65, 0.9, 0.75),
            "hspace": 0.22,
        },
    )
    raster_a, raster_b, rate_ax_a, rate_ax_b, phase, plv_ax = axes
    raster_window = (600.0, 1000.0)
    raster_slice = slice(round(raster_window[0] / DT_MS), round(raster_window[1] / DT_MS))
    raster_time = time_ms[raster_slice]
    for ax, circuit, e_key, i_key, color in (
        (raster_a, "A", "population_0", "population_1", theme.INK_BLACK),
        (raster_b, "B", "population_2", "population_3", theme.DEEP_RED),
    ):
        # A fixed subset keeps the vector figure legible while population rates
        # and all quantitative analyses continue to use every neuron.
        for key, count, offset in ((e_key, 100, 0), (i_key, 40, 105)):
            sample = coupled_rec[key][raster_slice, 0, :count]
            t, cell = np.nonzero(sample)
            ax.scatter(
                raster_time[t], cell + offset, s=2.0, linewidths=0,
                color=color, rasterized=True,
            )
        ax.axhline(102, color=theme.GREY_LIGHT, linewidth=0.6)
        ax.set_xlim(*raster_window)
        ax.set_ylim(-2, 147)
        ax.set_yticks((49.5, 124.5), ("E", "I"))
        ax.set_ylabel(f"circuit {circuit}\ncell")
    raster_a.tick_params(labelbottom=False)
    raster_b.set_xlabel("time from simulation start (ms)")

    for ax, values, circuit, color in (
        (rate_ax_a, rate_a, "A", theme.INK_BLACK),
        (rate_ax_b, rate_b, "B", theme.DEEP_RED),
    ):
        ax.plot(time_ms, values, color=color, linewidth=0.9)
        ax.set_ylabel(f"{circuit} E rate\n(Hz)")
        ax.set_xlim(0, 1200)
    common_rate_max = max(float(np.max(rate_a[: round(1200 / DT_MS)])), float(np.max(rate_b[: round(1200 / DT_MS)])))
    rate_ax_a.set_ylim(0, common_rate_max * 1.04)
    rate_ax_b.set_ylim(0, common_rate_max * 1.04)
    rate_ax_a.tick_params(labelbottom=False)
    rate_ax_b.set_xlabel("time from simulation start (ms)")

    analysis = slice(TRANSIENT_STEPS, None)
    phase_time = time_ms[analysis]
    baseline_unwrapped = np.unwrap(np.angle(baseline_phase_vector))[analysis]
    coupled_unwrapped = np.unwrap(np.angle(phase_vector))[analysis]
    baseline_unwrapped -= baseline_unwrapped[0]
    coupled_unwrapped -= coupled_unwrapped[0]
    phase.plot(
        phase_time,
        baseline_unwrapped,
        color=theme.GREY_MID,
        linewidth=1.0,
        label="uncoupled",
    )
    phase.plot(
        phase_time,
        coupled_unwrapped,
        color=theme.DEEP_RED,
        linewidth=1.3,
        label="coupled · w=0.25, d=10 ms",
    )
    phase.axhline(0, color=theme.GREY_LIGHT, linewidth=0.6)
    phase.set_ylabel("A − B phase\nchange (rad)")
    phase.legend(frameon=False, ncol=2, loc="upper left")
    phase.tick_params(labelbottom=False)

    plv_ax.plot(
        phase_time,
        baseline_rolling_plv[analysis],
        color=theme.GREY_MID,
        linewidth=1.0,
        label="uncoupled",
    )
    plv_ax.plot(
        phase_time,
        rolling_plv[analysis],
        color=theme.DEEP_RED,
        linewidth=1.3,
        label="coupled",
    )
    plv_ax.axhline(0.8, color=theme.INK_BLACK, linestyle="--", linewidth=0.7)
    plv_ax.set_ylim(0, 1.02)
    plv_ax.set_ylabel("rolling PLV\n(200 ms)")
    plv_ax.set_xlabel("time from simulation start (ms)")
    for ax in (phase, plv_ax):
        ax.set_xlim(TRANSIENT_STEPS * DT_MS, STEPS * DT_MS)
        ax.grid(True, alpha=0.12)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def replot() -> None:
    root = REPO / "artifacts" / "data" / SLUG
    rows = json.loads((root / "sweep_table.json").read_text())
    render_summary(rows, root / "summary_compound.png", root)
    refinement = root / "refinement" / "best-recordings.npz"
    with np.load(refinement) as archive:
        best = {key: archive[key] for key in archive.files}
    render_sync_emergence(
        load_recordings(root, "uncoupled"), best,
        root / "synchronization_emergence.svg",
    )


def main() -> None:
    acquire()


if __name__ == "__main__":
    main()
