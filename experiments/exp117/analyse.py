"""Validate the exp117 Hopf crossing and sampled nonlinear criticality."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp117 import inputs
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def analyse(identity, *, run_id=None):
    compute = inputs.source(REPO, identity, "compute")
    cfg = inputs.configuration(compute)
    if compute.record["inputs"]:
        raise PingstoreError("exp117 compute must not have upstream inputs")
    raw = load_json(compute.export / "continuation.json")
    if raw.get("schema") != "exp117.compute/v3":
        raise PingstoreError("unsupported exp117 compute payload")
    if raw.get("configuration") != cfg:
        raise PingstoreError("exp117 compute payload and recipe disagree")
    hopf = raw["hopf"]
    critical_real, critical_imag = hopf["critical_pair_per_ms"]
    threshold = cfg["numerical_protocol"]["complex_eigenvalue_threshold_per_ms"]
    checks = {
        "critical_real_near_zero": abs(critical_real) < 1e-9,
        "nonzero_angular_frequency": abs(critical_imag) > threshold,
        "remaining_modes_stable": hopf["remaining_max_real_per_ms"] < 0,
        "transverse_crossing": hopf["crossing_slope_per_ms_per_nA"] > 0,
    }
    if not all(checks.values()):
        raise PingstoreError(f"exp117 Hopf criteria failed: {checks}")
    ramp_cfg = cfg["numerical_protocol"]["criticality_ramp"]
    with np.load(compute.export / "recording.npz", allow_pickle=False) as recording:
        drives = np.asarray(recording["drives_nA"], dtype=float)
        times = np.asarray(recording["measurement_times_ms"], dtype=float)
        states_up = np.asarray(recording["states_up"], dtype=float)
        states_down = np.asarray(recording["states_down"], dtype=float)
    expected_drives = np.linspace(
        hopf["I_ext_star_nA"] + ramp_cfg["span_relative_to_hopf_nA"][0],
        hopf["I_ext_star_nA"] + ramp_cfg["span_relative_to_hopf_nA"][1],
        ramp_cfg["points"],
    )
    expected_shape = (ramp_cfg["points"], 4, ramp_cfg["measurement_samples"])
    if (
        drives.shape != expected_drives.shape
        or not np.array_equal(drives, expected_drives)
        or times.shape != (ramp_cfg["measurement_samples"],)
        or times[0] != ramp_cfg["measurement_start_ms"]
        or times[-1] != ramp_cfg["duration_ms"]
        or not np.all(np.diff(times) > 0)
        or states_up.shape != expected_shape
        or states_down.shape != expected_shape
        or not np.isfinite(states_up).all()
        or not np.isfinite(states_down).all()
    ):
        raise PingstoreError("invalid exp117 criticality recording")
    amplitude_up = np.ptp(states_up[:, 0, :], axis=1)
    amplitude_down = np.ptp(states_down[:, 0, :], axis=1)
    branch_gap = float(np.max(np.abs(amplitude_up - amplitude_down)))
    threshold = ramp_cfg["amplitude_threshold_per_ms"]
    onset_up = next(
        (
            float(drive)
            for drive, amplitude in zip(drives, amplitude_up)
            if amplitude > threshold
        ),
        None,
    )
    onset_down = next(
        (
            float(drive)
            for drive, amplitude in zip(drives, amplitude_down)
            if amplitude > threshold
        ),
        None,
    )
    hysteresis_width = (
        None if onset_up is None or onset_down is None else onset_up - onset_down
    )
    above = drives > hopf["I_ext_star_nA"] + 1e-9
    if np.count_nonzero(above) < 2:
        raise PingstoreError("insufficient exp117 above-onset ramp points")
    relative_drive = drives[above] - hopf["I_ext_star_nA"]
    amplitude_squared = amplitude_up[above] ** 2
    slope, intercept = np.polyfit(relative_drive, amplitude_squared, 1)
    predicted = slope * relative_drive + intercept
    ss_residual = float(np.sum((amplitude_squared - predicted) ** 2))
    ss_total = float(np.sum((amplitude_squared - amplitude_squared.mean()) ** 2))
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else 0.0
    criticality_checks = {
        "branches_coincide": bool(
            branch_gap < ramp_cfg["maximum_branch_gap_per_ms"]
        ),
        "positive_amplitude_squared_slope": bool(slope > 0),
        "amplitude_squared_fit_passes": bool(
            r_squared > ramp_cfg["minimum_amplitude_squared_r2"]
        ),
    }
    criticality = {
        "classification": (
            "consistent with supercritical Hopf bifurcation"
            if all(criticality_checks.values())
            else "subcritical or inconclusive"
        ),
        "checks": criticality_checks,
        "amplitude_threshold_per_ms": threshold,
        "threshold_onset_up_nA": onset_up,
        "threshold_onset_down_nA": onset_down,
        "branch_gap_per_ms": branch_gap,
        "hysteresis_width_nA": hysteresis_width,
        "amplitude_squared_slope_per_ms2_per_nA": float(slope),
        "amplitude_squared_intercept_per_ms2": float(intercept),
        "amplitude_squared_r2": float(r_squared),
    }
    expected_tau = cfg["mean_field_choices"]["tau_GABA_sweep_ms"]
    tau_rows = raw.get("tau_GABA_sweep", [])
    if [row.get("tau_GABA_ms") for row in tau_rows] != expected_tau:
        raise PingstoreError("incomplete exp117 inhibitory-decay sweep")
    frequency_vs_tau = []
    for row in tau_rows:
        swept_hopf = row["hopf"]
        swept_real, swept_imag = swept_hopf["critical_pair_per_ms"]
        swept_checks = {
            "critical_real_near_zero": abs(swept_real) < 1e-9,
            "nonzero_angular_frequency": abs(swept_imag) > threshold,
            "remaining_modes_stable": swept_hopf["remaining_max_real_per_ms"] < 0,
            "transverse_crossing": swept_hopf["crossing_slope_per_ms_per_nA"] > 0,
        }
        if not all(swept_checks.values()):
            raise PingstoreError(
                f"exp117 tau_GABA Hopf criteria failed: {swept_checks}"
            )
        frequency_vs_tau.append(
            {
                "tau_GABA_ms": row["tau_GABA_ms"],
                "I_ext_star_nA": swept_hopf["I_ext_star_nA"],
                "f_Hopf_Hz": swept_hopf["f_Hopf_Hz"],
                "checks": swept_checks,
            }
        )
    result = {
        "schema": "exp117.analysis/v3",
        "configuration": cfg,
        "result": {
            **hopf,
            "classification": "simple Hopf bifurcation",
            "checks": checks,
            "criticality": criticality,
            "frequency_vs_tau_GABA": frequency_vs_tau,
        },
    }
    plot_rows = []
    for row in raw["continuation"]:
        state = row["equilibrium"]
        leading = row["leading_complex_per_ms"]
        values = np.asarray(
            [complex(real, imag) for real, imag in row["eigenvalues_per_ms"]]
        )
        if leading is None:
            leading_real = None
            frequency = None
            remaining_max = float(max(value.real for value in values))
        else:
            critical = complex(*leading)
            first = int(np.argmin(np.abs(values - critical)))
            second = int(np.argmin(np.abs(values - np.conjugate(critical))))
            remaining = [
                value.real
                for index, value in enumerate(values)
                if index not in {first, second}
            ]
            leading_real = float(critical.real)
            frequency = float(1000 * abs(critical.imag) / (2 * np.pi))
            remaining_max = float(max(remaining))
        plot_rows.append(
            {
                "I_ext_nA": row["I_ext_nA"],
                "rate_E_Hz": 1000 * state[0],
                "rate_I_Hz": 1000 * state[1],
                "eigenvalues_per_ms": row["eigenvalues_per_ms"],
                "critical_real_per_ms": leading_real,
                "remaining_max_real_per_ms": remaining_max,
                "complex_pair_frequency_Hz": frequency,
            }
        )
    coordinates = {
        "schema": "exp117.plot-coordinates/v4",
        "rows": plot_rows,
        "hopf": hopf,
        "criticality": {
            **criticality,
            "relative_drive_nA": (drives - hopf["I_ext_star_nA"]).tolist(),
            "amplitude_up_Hz": (1000 * amplitude_up).tolist(),
            "amplitude_down_Hz": (1000 * amplitude_down).tolist(),
        },
        "frequency_vs_tau_GABA": frequency_vs_tau,
    }
    with inputs.execution(
        REPO,
        "analyse",
        sources={"compute": compute},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        write_json_atomic(run.export / "results.json", result)
        write_json_atomic(run.export / "plot_coordinates.json", coordinates)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(analyse(args.source, run_id=args.run_id))


if __name__ == "__main__":
    main()
