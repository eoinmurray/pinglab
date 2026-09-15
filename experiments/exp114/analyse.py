"""Measure frequency detuning, phase locking, phase lead, and E/I timing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp114 import inputs, recipe
from pingstore.contracts import PingstoreError, write_json_atomic
from pingstore.stages import stage_run
from scipy.signal import butter, correlate, hilbert, sosfiltfilt, welch

MEASUREMENT = {
    "schema": "exp114.measurement/v1",
    "population_series": "spike counts in non-overlapping 1 ms bins",
    "peak": "raw maximum bin from 20-80 Hz",
    "power_fraction": "20-80 Hz power divided by 5-150 Hz power",
    "phase": "fourth-order zero-phase Butterworth 20-80 Hz filter followed by analytic Hilbert phase",
    "phase_concentration": "magnitude of mean exp(i*(phase2-phase1))",
    "phase_offset_cycles": "circular mean phase2-phase1 divided by 2*pi",
    "ei_lag": "lag maximizing E/I population-count cross-correlation within +/-10 ms; positive means I follows E",
    "aggregation": "median across three fixed finite-network seeds",
}


def measurement_definition(cfg):
    analysed_ms = cfg["duration_ms"] - cfg["burn_in_ms"]
    return {
        **MEASUREMENT,
        "epoch": f'{cfg["burn_in_ms"]:g}-{cfg["duration_ms"]:g} ms simulation time ({analysed_ms:g} ms analysed)',
        "spectrum": f"Welch, full {analysed_ms:g} ms segment, Hann window, density scaling",
    }


def binned(spikes, cfg):
    width = round(1.0 / cfg["dt_ms"])
    start = round(cfg["burn_in_ms"] / cfg["dt_ms"])
    values = spikes[start:]
    values = values[: len(values) // width * width]
    return values.reshape(-1, width, values.shape[1]).sum((1, 2)).astype(float)


def spectrum(series, band):
    frequencies, power = welch(series - series.mean(), fs=1000.0, nperseg=len(series))
    selected = (frequencies >= band[0]) & (frequencies <= band[1])
    denominator = power[(frequencies >= 5) & (frequencies <= 150)].sum()
    return frequencies, power, float(frequencies[selected][np.argmax(power[selected])]), float(power[selected].sum() / denominator) if denominator > 0 else 0.0


def phase_metrics(series_1, series_2, band):
    sos = butter(4, band, btype="bandpass", fs=1000.0, output="sos")
    phase_1 = np.angle(hilbert(sosfiltfilt(sos, series_1)))
    phase_2 = np.angle(hilbert(sosfiltfilt(sos, series_2)))
    edge = 50
    vector = np.exp(1j * (phase_2[edge:-edge] - phase_1[edge:-edge]))
    mean = vector.mean()
    return float(abs(mean)), float(np.angle(mean) / (2 * np.pi)), phase_1, phase_2


def ei_lag_ms(e_series, i_series):
    lags = np.arange(-10, 11)
    corr = correlate(i_series - i_series.mean(), e_series - e_series.mean(), mode="full")
    full_lags = np.arange(-len(e_series) + 1, len(e_series))
    values = np.array([corr[full_lags == lag][0] for lag in lags])
    return float(lags[np.argmax(values)])


def analyse(identity, *, run_id=None):
    compute = inputs.source(REPO, identity, "compute")
    cfg = inputs.configuration(compute)
    measurement = measurement_definition(cfg)
    expected = list(recipe.conditions(cfg))
    with np.load(compute.export / "recording.npz", allow_pickle=False) as retained:
        keys = set(retained.files)
        required = {f"{cell['condition_id']}__spk_{pop}{module}" for cell in expected for pop in ("e", "i") for module in (1, 2)}
        if keys != required:
            raise PingstoreError("recording does not contain the exact fixed condition grid")
        rows, trace_arrays = [], {}
        band = cfg["gamma_band_hz"]
        for cell in expected:
            cid = cell["condition_id"]
            e = [binned(retained[f"{cid}__spk_e{m}"], cfg) for m in (1, 2)]
            i = [binned(retained[f"{cid}__spk_i{m}"], cfg) for m in (1, 2)]
            spectral = [spectrum(series, band) for series in e]
            concentration, offset, phase_1, phase_2 = phase_metrics(e[0], e[1], band)
            row = {
                **cell,
                "e_rate_1_hz": float(e[0].sum() / (cfg["n_e_per_module"] * (cfg["duration_ms"] - cfg["burn_in_ms"]) / 1000)),
                "e_rate_2_hz": float(e[1].sum() / (cfg["n_e_per_module"] * (cfg["duration_ms"] - cfg["burn_in_ms"]) / 1000)),
                "i_rate_1_hz": float(i[0].sum() / (cfg["n_i_per_module"] * (cfg["duration_ms"] - cfg["burn_in_ms"]) / 1000)),
                "i_rate_2_hz": float(i[1].sum() / (cfg["n_i_per_module"] * (cfg["duration_ms"] - cfg["burn_in_ms"]) / 1000)),
                "peak_1_hz": spectral[0][2],
                "peak_2_hz": spectral[1][2],
                "absolute_peak_difference_hz": abs(spectral[1][2] - spectral[0][2]),
                "gamma_power_fraction_1": spectral[0][3],
                "gamma_power_fraction_2": spectral[1][3],
                "phase_concentration": concentration,
                "phase_offset_cycles": offset,
                "ei_lag_1_ms": ei_lag_ms(e[0], i[0]),
                "ei_lag_2_ms": ei_lag_ms(e[1], i[1]),
            }
            rows.append(row)
            if cell["seed"] == cfg["seeds"][0] and cell["detuning_index"] in (0, len(cfg["signed_drive_differences_hz"]) - 1) and cell["coupling_index"] in (0, len(cfg["cross_e_weight_us"]) - 1):
                trace_arrays[f"{cid}__e1"] = e[0]
                trace_arrays[f"{cid}__e2"] = e[1]
                trace_arrays[f"{cid}__phase1"] = phase_1
                trace_arrays[f"{cid}__phase2"] = phase_2
        aggregate = []
        for di, difference in enumerate(cfg["signed_drive_differences_hz"]):
            for ci, coupling in enumerate(cfg["cross_e_weight_us"]):
                subset = [row for row in rows if row["detuning_index"] == di and row["coupling_index"] == ci]
                item = {"detuning_index": di, "coupling_index": ci, "drive_difference_hz": difference, "cross_weight_us": coupling}
                for key in ("e_rate_1_hz", "e_rate_2_hz", "i_rate_1_hz", "i_rate_2_hz", "peak_1_hz", "peak_2_hz", "absolute_peak_difference_hz", "gamma_power_fraction_1", "gamma_power_fraction_2", "phase_concentration", "phase_offset_cycles", "ei_lag_1_ms", "ei_lag_2_ms"):
                    item[key] = float(np.median([row[key] for row in subset]))
                threshold = cfg["locking_thresholds"]
                item["locked"] = bool(item["phase_concentration"] >= threshold["phase_concentration"] and item["absolute_peak_difference_hz"] <= threshold["absolute_peak_difference_hz"] and min(item["gamma_power_fraction_1"], item["gamma_power_fraction_2"]) >= threshold["minimum_gamma_power_fraction"])
                aggregate.append(item)
    zero_di = cfg["signed_drive_differences_hz"].index(0.0)
    uncoupled = {item["detuning_index"]: item for item in aggregate if item["coupling_index"] == 0}
    extremes = (0, len(cfg["signed_drive_differences_hz"]) - 1)
    criterion_a = all(uncoupled[index]["absolute_peak_difference_hz"] > uncoupled[zero_di]["absolute_peak_difference_hz"] for index in extremes)
    transitions = []
    for di, difference in enumerate(cfg["signed_drive_differences_hz"]):
        if difference == 0:
            continue
        cells = [item for item in aggregate if item["detuning_index"] == di]
        if not cells[0]["locked"] and any(item["locked"] for item in cells[1:]):
            transitions.append(di)
    criterion_b = bool(transitions)
    paired = [(lo, len(cfg["signed_drive_differences_hz"]) - 1 - lo) for lo in range(zero_di)]
    reversals = []
    phase_couplings = []
    for ci in range(len(cfg["cross_e_weight_us"])):
        at_coupling = {item["detuning_index"]: item for item in aggregate if item["coupling_index"] == ci}
        pair_results = [at_coupling[a]["locked"] and at_coupling[b]["locked"] and at_coupling[a]["phase_offset_cycles"] * at_coupling[b]["phase_offset_cycles"] < 0 for a, b in paired]
        reversals.extend(pair_results)
        if all(pair_results):
            seed_majorities = []
            for a, b in paired:
                for index, expected_sign in ((a, -1), (b, 1)):
                    values = [row["phase_offset_cycles"] for row in rows if row["detuning_index"] == index and row["coupling_index"] == ci]
                    seed_majorities.append(sum(np.sign(value) == expected_sign for value in values) >= 2)
            if all(seed_majorities):
                phase_couplings.append(ci)
    criterion_c = bool(phase_couplings)
    first_lock = {}
    for di in range(len(cfg["signed_drive_differences_hz"])):
        locked_indices = [item["coupling_index"] for item in aggregate if item["detuning_index"] == di and item["locked"]]
        first_lock[di] = min(locked_indices) if locked_indices else None
    criterion_e = all(first_lock[index] is not None for index in extremes) and first_lock[1] is not None and first_lock[3] is not None and (first_lock[0] > first_lock[1] or first_lock[4] > first_lock[3])
    rate_folds = []
    for di, ci in first_lock.items():
        if ci is None or di == zero_di:
            continue
        baseline = next(item for item in aggregate if item["detuning_index"] == di and item["coupling_index"] == 0)
        locked_cell = next(item for item in aggregate if item["detuning_index"] == di and item["coupling_index"] == ci)
        for key in ("e_rate_1_hz", "e_rate_2_hz", "i_rate_1_hz", "i_rate_2_hz"):
            rate_folds.append(locked_cell[key] / baseline[key])
    criterion_f = bool(rate_folds) and max(rate_folds) <= 2.0
    complete = len(rows) == len(expected)
    overall = criterion_a and criterion_b and criterion_c and complete and criterion_e and criterion_f
    success = {"criterion_a_bilateral_detuning": bool(criterion_a), "criterion_b_locking_transition": bool(criterion_b), "criterion_c_phase_sign_reversal": bool(criterion_c), "criterion_d_complete_grid": complete, "criterion_e_detuning_dependent_boundary": bool(criterion_e), "criterion_f_max_rate_fold_at_first_lock_le_2": bool(criterion_f), "overall": bool(overall), "transition_detuning_indices": transitions, "phase_reversal_pairs_found": int(sum(reversals)), "phase_reversal_coupling_indices": phase_couplings, "first_lock_coupling_index": first_lock, "max_rate_fold_at_first_lock": float(max(rate_folds)) if rate_folds else None}
    with stage_run(REPO, recipe.SLUG, "analyse", inputs={"compute": compute}, run_id=run_id, configuration=measurement) as run:
        write_json_atomic(run.export / "results.json", {"schema": "exp114.analysis/v1", "recipe": cfg, "measurement": measurement, "rows": rows, "aggregate": aggregate, "success": success})
        np.savez_compressed(run.export / "phase_traces.npz", **trace_arrays)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    analyse(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
