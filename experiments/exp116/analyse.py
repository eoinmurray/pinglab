"""Measure the minimal exp116 evidence without executing the model."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp116 import recipe
from experiments.exp116.compute import record_environment
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def classify_onset(levels, cfg):
    if len(levels) != 2 or any(level["failed_indices"] for level in levels):
        return None
    losses = [
        [
            row
            for row in level["crossings"]
            if row.get("direction") == "loss" and "failure" not in row
        ]
        for level in levels
    ]
    if any(len(rows) != 1 for rows in losses):
        return None
    coarse, fine = losses[0][0], losses[1][0]
    criteria = cfg["criteria"]
    checks = {
        "critical_pair": coarse["critical_real"] < criteria["critical_real_per_ms"],
        "remaining_modes_damped": coarse["remaining_max_real"]
        < criteria["remaining_real_per_ms"],
        "positive_transversality": coarse["chi_per_ms_nA"] > 0
        and min(coarse["chi_finite_difference"]) > 0,
        "drive_convergence": abs(coarse["drive_nA"] - fine["drive_nA"])
        < criteria["drive_convergence_nA"],
        "frequency_convergence": abs(coarse["frequency_Hz"] - fine["frequency_Hz"])
        < criteria["frequency_convergence_Hz"],
    }
    return {
        **coarse,
        "accepted": all(checks.values()),
        "checks": checks,
        "fine_grid_drive_nA": fine["drive_nA"],
        "fine_grid_frequency_Hz": fine["frequency_Hz"],
    }


def measure_ramp(raw, arrays, onset, cfg):
    design = cfg["criticality_ramp"]
    if raw.get("ramp", {}).get("status") != "complete":
        return {"assessment": "unresolved", "reason": raw.get("ramp", {}).get("reason")}
    drives = onset["drive_nA"] + np.linspace(*design["span_nA"], design["points"])
    times = np.arange(
        design["observation_start_ms"],
        design["duration_ms"] + design["observation_step_ms"] / 2,
        design["observation_step_ms"],
    )
    if not np.array_equal(arrays["drive_nA"], drives) or not np.array_equal(
        arrays["observation_time_ms"], times
    ):
        raise PingstoreError("reference ramp grids differ from the recipe")
    amplitudes = {}
    for direction in ("up", "down"):
        values = arrays[direction]
        if (
            values.shape != (design["points"], times.size, 4)
            or not np.isfinite(values).all()
        ):
            raise PingstoreError("reference ramp trajectories are malformed")
        amplitudes[direction] = np.ptp(values[:, :, 0], axis=1)
    gap = float(np.max(np.abs(amplitudes["up"] - amplitudes["down"])))
    above = drives > onset["drive_nA"] + 1e-9
    x = drives[above] - onset["drive_nA"]
    y = amplitudes["up"][above] ** 2
    slope, intercept = np.polyfit(x, y, 1)
    residual = float(np.sum((y - (slope * x + intercept)) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - residual / total if total > 0 else 0.0
    consistent = (
        gap < design["branch_gap_threshold_per_ms"]
        and slope > 0
        and r_squared > design["minimum_amplitude_squared_r2"]
    )
    return {
        "assessment": "consistent_with_supercritical"
        if consistent
        else "subcritical_or_inconclusive",
        "branch_gap_per_ms": gap,
        "amplitude_squared_slope_per_ms2_nA": float(slope),
        "amplitude_squared_intercept_per_ms2": float(intercept),
        "amplitude_squared_r2": float(r_squared),
        "drive_nA": drives.tolist(),
        "up_amplitude_per_ms": amplitudes["up"].tolist(),
        "down_amplitude_per_ms": amplitudes["down"].tolist(),
    }


def validate_continuation(raw, arrays, condition, cfg):
    if (
        raw.get("schema") != "exp116.compute-condition/v1"
        or raw.get("condition") != condition
    ):
        raise PingstoreError("condition record differs from the exp116 recipe")
    maximum = 0.0
    for level, count in enumerate(cfg["drive_counts"]):
        prefix = f"level{level}_"
        grid = arrays[prefix + "drive_nA"]
        valid = arrays[prefix + "valid"]
        if not np.array_equal(grid, np.linspace(*cfg["drive_interval_nA"], count)):
            raise PingstoreError("continuation drive grid differs from the recipe")
        if valid.shape != (count,) or not valid.all():
            raise PingstoreError("continuation contains unresolved equilibrium points")
        if arrays[prefix + "state"].shape != (count, 4) or arrays[
            prefix + "eigenvalues"
        ].shape != (count, 4, 2):
            raise PingstoreError("continuation arrays are malformed")
        for key in ("rate_residual", "flow_residual", "scalar_discrepancy"):
            values = arrays[prefix + key]
            if values.shape != (count,) or not np.isfinite(values).all():
                raise PingstoreError("continuation residuals are malformed")
            maximum = max(maximum, float(values.max(initial=0)))
    return maximum


def analyse(identity, *, run_id=None):
    source = source_run(
        REPO / ".pingstore", identity, stage="compute", experiment=recipe.SLUG
    )
    cfg = recipe.validate(source.record["execution"]["configuration"])
    if source.record["inputs"]:
        raise PingstoreError("exp116 compute must not have upstream run inputs")
    planned = recipe.conditions(cfg)
    if {path.name for path in source.export.iterdir()} != {
        recipe.condition_id(row) for row in planned
    }:
        raise PingstoreError(
            "compute export does not contain the complete exp116 design"
        )
    with stage_run(
        REPO,
        recipe.SLUG,
        "analyse",
        run_id=run_id,
        configuration=cfg,
        inputs={"compute": source},
    ) as run:
        record_environment(run)
        results, coordinates = [], {}
        for condition in planned:
            unit = source.unit(recipe.condition_id(condition))
            raw = load_json(unit / "crossings.json")
            with np.load(unit / "continuation.npz", allow_pickle=False) as arrays:
                residual = validate_continuation(raw, arrays, condition, cfg)
                onset = classify_onset(raw["levels"], cfg)
                stable_start = float(arrays["level0_eigenvalues"][0, :, 0].max()) < 0
                if onset is not None and not stable_start:
                    onset["accepted"] = False
                    onset["checks"]["stable_lower_branch"] = False
                criticality = None
                if recipe.is_reference(condition, cfg):
                    if (
                        onset is None
                        or not onset["accepted"]
                        or not (unit / "ramps.npz").is_file()
                    ):
                        criticality = {
                            "assessment": "unresolved",
                            "reason": "accepted onset or ramp unavailable",
                        }
                    else:
                        with np.load(unit / "ramps.npz", allow_pickle=False) as ramps:
                            criticality = measure_ramp(raw, ramps, onset, cfg)
                    coordinates = {
                        "drive_nA": arrays["level0_drive_nA"],
                        "leading_real_per_ms": arrays["level0_eigenvalues"][
                            :, :, 0
                        ].max(axis=1),
                    }
                    if criticality and "drive_nA" in criticality:
                        coordinates.update(
                            ramp_drive_nA=np.asarray(criticality["drive_nA"]),
                            ramp_up_amplitude_per_ms=np.asarray(
                                criticality["up_amplitude_per_ms"]
                            ),
                            ramp_down_amplitude_per_ms=np.asarray(
                                criticality["down_amplitude_per_ms"]
                            ),
                        )
                results.append(
                    {
                        "condition": condition,
                        "purpose": "gaba_sweep"
                        if recipe.is_primary(condition, cfg)
                        else "robustness_endpoint",
                        "onset": onset if onset and onset["accepted"] else None,
                        "criticality": criticality,
                        "stable_at_zero_drive": stable_start,
                        "max_equilibrium_residual": residual,
                    }
                )
        primary = [row for row in results if row["purpose"] == "gaba_sweep"]
        primary_frequencies = [
            row["onset"]["frequency_Hz"] if row["onset"] else None for row in primary
        ]
        robust = [row for row in results if row["purpose"] == "robustness_endpoint"]
        corner_checks = []
        for sigma in cfg["robustness"]["sigma_corners_mV"]:
            for kappa in cfg["robustness"]["kappa_corners"]:
                pair = [
                    row
                    for row in robust
                    if row["condition"]["sigma_mV"] == sigma
                    and row["condition"]["kappa"] == kappa
                ]
                pair.sort(key=lambda row: row["condition"]["tau_GABA_ms"])
                corner_checks.append(
                    {
                        "sigma_mV": sigma,
                        "kappa": kappa,
                        "both_onsets_resolved": len(pair) == 2
                        and all(row["onset"] for row in pair),
                        "slower_inhibition_lowers_frequency": len(pair) == 2
                        and all(row["onset"] for row in pair)
                        and pair[1]["onset"]["frequency_Hz"]
                        < pair[0]["onset"]["frequency_Hz"],
                    }
                )
        reference = next(
            row for row in results if recipe.is_reference(row["condition"], cfg)
        )
        summary = {
            "condition_count": len(results),
            "all_onsets_resolved": all(row["onset"] for row in results),
            "reference_criticality": reference["criticality"]["assessment"],
            "reference_gaba_frequencies_strictly_decrease": all(
                left is not None and right is not None and right < left
                for left, right in zip(primary_frequencies, primary_frequencies[1:])
            ),
            "robustness_corners": corner_checks,
            "all_robustness_corners_support_direction": all(
                row["both_onsets_resolved"]
                and row["slower_inhibition_lowers_frequency"]
                for row in corner_checks
            ),
        }
        write_json_atomic(
            run.export / "results.json",
            {
                "schema": "exp116.analysis/v1",
                "reference": reference,
                "conditions": results,
                "summary": summary,
            },
        )
        np.savez_compressed(run.export / "reference-branch.npz", **coordinates)
        print(summary, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp116 compute run")
    parser.add_argument("--run-id", help="unused pre-reserved analyse identity")
    args = parser.parse_args()
    analyse(args.source, run_id=args.run_id)
