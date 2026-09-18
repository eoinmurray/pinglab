"""Measure explicit equilibrium and time-domain evidence; never run the model."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp115 import recipe
from experiments.exp115.compute import environment
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def classify_crossing(crossing, levels, index, cfg):
    criteria = cfg["criteria"]
    if "failure" in crossing:
        return {**crossing, "accepted": False, "checks": {"calculation_completed": False}}
    scale = max(1, crossing["norm_A"])
    checks = {
        "critical_pair": crossing["critical_real"] <= criteria["critical_real_scaled"] * scale
        and crossing["omega_per_ms"] > criteria["eigen_separation_scaled"] * scale,
        "other_modes_damped": crossing["remaining_max_real"]
        < -criteria["noncritical_real_scaled"] * scale,
        "simple_eigenvalue": crossing["eigen_separation"]
        > criteria["eigen_separation_scaled"] * scale,
        "quartic_agreement": crossing["quartic_root_error"]
        <= criteria["critical_real_scaled"] * scale,
        "hopf_identity": crossing["hopf_identity_error"] <= criteria["hopf_identity"],
        "frequency_identity": crossing["frequency_identity_error_Hz"]
        <= criteria["frequency_Hz"],
    }
    chi_error = max(
        abs(value - crossing["chi_per_ms_nA"])
        for value in crossing["chi_finite_difference"]
    )
    chi = abs(crossing["chi_per_ms_nA"])
    checks["transversality"] = (
        chi_error <= criteria["transversality_rel"] * chi
        and chi > criteria["sign_margin"] * chi_error
        and crossing["chi_per_ms_nA"] > 0
    )
    comparisons = [crossing]
    same_count = all(
        len(level["crossings"]) == len(levels[0]["crossings"]) for level in levels
    )
    checks["grid_crossing_count"] = same_count
    for level in levels[1:]:
        if same_count and "failure" not in level["crossings"][index]:
            comparisons.append(level["crossings"][index])
    checks["all_grids_solved"] = all(not level["failed_points"] for level in levels)
    checks["matched_crossings"] = len(comparisons) == 3 and all(
        row["direction"] == crossing["direction"] for row in comparisons
    )
    differences = {
        key: max(abs(row[key] - crossing[key]) for row in comparisons)
        for key in ("drive_nA", "frequency_Hz")
    }
    checks["drive_convergence"] = differences["drive_nA"] < criteria["drive_convergence_nA"]
    checks["frequency_convergence"] = differences["frequency_Hz"] < criteria["frequency_Hz"]
    return {
        **crossing,
        "accepted": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "convergence_discrepancies": differences,
        "transversality_discrepancy": chi_error,
    }


def validate_condition(raw, arrays, condition, cfg):
    if raw.get("schema") != "exp115.compute-condition/v2" or raw["condition"] != condition:
        raise PingstoreError("compute condition identity mismatch")
    levels = raw["levels"]
    if [row["count"] for row in levels] != cfg["drive_counts"]:
        raise PingstoreError("incomplete drive-grid refinement")
    residual_max = 0.0
    for level_index, level in enumerate(levels):
        prefix, count = f"level{level_index}_", level["count"]
        grid = arrays[prefix + "drive_nA"]
        if not np.array_equal(grid, np.linspace(*cfg["drive_interval_nA"], count)):
            raise PingstoreError("continuation drive coordinates disagree with recipe")
        valid = arrays[prefix + "valid"]
        if valid.shape != (count,) or valid.dtype != np.bool_:
            raise PingstoreError("malformed solve-availability mask")
        if [row["index"] for row in level["failed_points"]] != np.flatnonzero(~valid).tolist():
            raise PingstoreError("failed solves are not fully accounted for")
        for key, shape in (
            ("state", (4,)), ("eigenvalues", (4, 2)), ("gain_derivatives", (2, 3)),
            ("K_minus_KH", ()), ("rate_residual", ()), ("flow_residual", ()),
            ("scalar_discrepancy", ()),
        ):
            values = arrays[prefix + key]
            if values.shape != (count, *shape) or not np.isfinite(values[valid]).all():
                raise PingstoreError(f"malformed continuation measurement: {key}")
        tolerance = cfg["equilibrium_residual"] / (cfg["tightening_factor"] if level_index else 1)
        for key in ("rate_residual", "flow_residual", "scalar_discrepancy"):
            maximum = float(np.max(arrays[prefix + key][valid], initial=0))
            if maximum > tolerance:
                raise PingstoreError("stored equilibrium exceeds its residual criterion")
            residual_max = max(residual_max, maximum)
        states = arrays[prefix + "state"][valid]
        ceilings = np.array([1 / cfg["cells"][p]["tau_ref_ms"] for p in ("E", "I")])
        if (states < 0).any() or (states[:, :2] >= ceilings).any():
            raise PingstoreError("accepted equilibrium is outside its physical domain")
        for crossing in level["crossings"]:
            left_drive, right_drive = crossing["bracket_nA"]
            left, right = np.searchsorted(grid, [left_drive, right_drive])
            if (
                left >= count or right >= count or right - left != 1
                or grid[left] != left_drive or grid[right] != right_drive
                or not valid[left] or not valid[right]
            ):
                raise PingstoreError("crossing bracket lacks adjacent solved evidence")
            if arrays[prefix + "K_minus_KH"][[left, right]].prod() > 0:
                raise PingstoreError("crossing bracket has no sign change")
            if "failure" not in crossing and not left_drive <= crossing["drive_nA"] <= right_drive:
                raise PingstoreError("refined crossing is outside its bracket")
    return residual_max


def measure_ramp(raw, ramp_arrays, onset, cfg):
    design = cfg["criticality_ramp"]
    expected_drives = onset["drive_nA"] + np.linspace(*design["span_nA"], design["points"])
    expected_times = np.arange(
        design["observation_start_ms"],
        design["duration_ms"] + design["observation_step_ms"] / 2,
        design["observation_step_ms"],
    )
    if raw.get("ramp", {}).get("status") != "complete":
        return {"assessment": "unresolved", "reason": raw.get("ramp", {}).get("failure", "ramp unavailable")}
    if not np.array_equal(ramp_arrays["drive_nA"], expected_drives):
        raise PingstoreError("criticality ramp drive grid disagrees with onset")
    if not np.array_equal(ramp_arrays["observation_time_ms"], expected_times):
        raise PingstoreError("criticality ramp observation grid disagrees with recipe")
    shape, amplitudes = (design["points"], expected_times.size, 4), {}
    for direction in ("up", "down"):
        values = ramp_arrays[direction]
        endpoints = ramp_arrays[direction + "_endpoint"]
        if values.shape != shape or endpoints.shape != (design["points"], 4):
            raise PingstoreError("malformed criticality-ramp trajectories")
        if not np.isfinite(values).all() or not np.isfinite(endpoints).all():
            raise PingstoreError("nonfinite criticality-ramp trajectory")
        amplitudes[direction] = np.ptp(values[:, :, 0], axis=1)
    gap = float(np.max(np.abs(amplitudes["up"] - amplitudes["down"])))
    above = expected_drives > onset["drive_nA"] + 1e-9
    x, y = expected_drives[above] - onset["drive_nA"], amplitudes["up"][above] ** 2
    slope, intercept = np.polyfit(x, y, 1)
    residual, total = float(np.sum((y - (slope * x + intercept)) ** 2)), float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - residual / total if total > 0 else 0.0
    threshold = design["amplitude_threshold_per_ms"]
    up_on = next((float(d) for d, a in zip(expected_drives, amplitudes["up"]) if a > threshold), None)
    down_on = next((float(d) for d, a in zip(expected_drives, amplitudes["down"]) if a > threshold), None)
    consistent = (
        gap < design["branch_gap_threshold_per_ms"] and slope > 0
        and r_squared > design["minimum_amplitude_squared_r2"]
    )
    return {
        "assessment": "consistent_with_supercritical" if consistent else "subcritical_or_inconclusive",
        "branch_gap_per_ms": gap,
        "hysteresis_width_nA": float(up_on - down_on) if up_on is not None and down_on is not None else None,
        "amplitude_squared_slope_per_ms2_nA": float(slope),
        "amplitude_squared_intercept_per_ms2": float(intercept),
        "amplitude_squared_r2": float(r_squared),
        "drive_nA": expected_drives.tolist(),
        "up_amplitude_per_ms": amplitudes["up"].tolist(),
        "down_amplitude_per_ms": amplitudes["down"].tolist(),
    }


def analyse(identity, *, run_id=None):
    source = source_run(REPO / ".pingstore", identity, stage="compute", experiment=recipe.SLUG)
    cfg = recipe.validate(source.record["execution"]["configuration"])
    if source.record["inputs"]:
        raise PingstoreError("exp115 computation must be independent")
    expected = recipe.conditions(cfg)
    if {path.name for path in source.export.iterdir()} != {recipe.condition_id(c) for c in expected}:
        raise PingstoreError("compute export does not cover the frozen closure grid")
    with stage_run(REPO, recipe.SLUG, "analyse", run_id=run_id, configuration=cfg, inputs={"compute": source}) as run:
        environment(run)
        results, coordinates = [], {}
        for condition in expected:
            unit = source.unit(recipe.condition_id(condition))
            raw = load_json(unit / "crossings.json")
            with np.load(unit / "continuation.npz", allow_pickle=False) as arrays:
                residual = validate_condition(raw, arrays, condition, cfg)
                crossings = [classify_crossing(row, raw["levels"], j, cfg) for j, row in enumerate(raw["levels"][0]["crossings"])]
                valid_grid = all(not row["failed_points"] for row in raw["levels"])
                stable_start = bool(arrays["level0_valid"][0]) and float(np.max(arrays["level0_eigenvalues"][0, :, 0])) < 0
                first_loss = next((row for row in crossings if row.get("direction") == "loss"), None)
                onset = first_loss if first_loss is not None and stable_start and valid_grid and first_loss["accepted"] else None
                count_agreement = len({len(row["crossings"]) for row in raw["levels"]}) == 1
                status = "resolved" if crossings and all(row["accepted"] for row in crossings) else "no_resolved_crossing" if not crossings and valid_grid and count_agreement else "unresolved"
                criticality = None
                if recipe.requires_criticality_ramp(condition, cfg):
                    if onset is None or not (unit / "ramps.npz").is_file():
                        criticality = {"assessment": "unresolved", "reason": raw.get("ramp", {}).get("failure", "accepted onset or ramp unavailable")}
                    else:
                        with np.load(unit / "ramps.npz", allow_pickle=False) as ramps:
                            criticality = measure_ramp(raw, ramps, onset, cfg)
                result = {
                    "condition": condition, "status": status, "stable_at_lower_drive": stable_start,
                    "onset": onset, "criticality": criticality, "crossings": crossings,
                    "max_equilibrium_residual": residual,
                    "failed_points_by_grid": [len(row["failed_points"]) for row in raw["levels"]],
                    "scalar_fallbacks_by_grid": [row["scalar_fallback_count"] for row in raw["levels"]],
                }
                results.append(result)
                if condition == cfg["reference"]:
                    coordinates = {
                        "drive_nA": arrays["level0_drive_nA"],
                        "leading_real_per_ms": np.where(arrays["level0_valid"], arrays["level0_eigenvalues"][:, :, 0].max(axis=1), np.nan),
                    }
                    if criticality and "drive_nA" in criticality:
                        coordinates.update(
                            ramp_drive_nA=np.asarray(criticality["drive_nA"]),
                            ramp_up_amplitude_per_ms=np.asarray(criticality["up_amplitude_per_ms"]),
                            ramp_down_amplitude_per_ms=np.asarray(criticality["down_amplitude_per_ms"]),
                        )
        reference = next(row for row in results if row["condition"] == cfg["reference"])
        accepted = [row for result in results for row in result["crossings"] if row["accepted"]]
        ramp_results = [row["criticality"] for row in results if row["criticality"] is not None]
        numbers = {
            "schema": "exp115.analysis/v2", "conditions": results, "reference": reference,
            "summary": {
                "condition_count": len(results),
                "conditions_with_onset": sum(row["onset"] is not None for row in results),
                "resolved_conditions": sum(row["status"] == "resolved" for row in results),
                "no_resolved_crossing_conditions": sum(row["status"] == "no_resolved_crossing" for row in results),
                "unresolved_conditions": sum(row["status"] == "unresolved" for row in results),
                "accepted_crossings": len(accepted), "criticality_ramps": len(ramp_results),
                "supercritical_consistent_ramps": sum(row["assessment"] == "consistent_with_supercritical" for row in ramp_results),
                "subcritical_or_inconclusive_ramps": sum(row["assessment"] == "subcritical_or_inconclusive" for row in ramp_results),
                "unresolved_ramps": sum(row["assessment"] == "unresolved" for row in ramp_results),
                "max_equilibrium_residual": max(row["max_equilibrium_residual"] for row in results),
                "max_drive_discrepancy_nA": max((row["convergence_discrepancies"]["drive_nA"] for row in accepted), default=None),
                "max_frequency_discrepancy_Hz": max((row["convergence_discrepancies"]["frequency_Hz"] for row in accepted), default=None),
            },
        }
        write_json_atomic(run.export / "results.json", numbers)
        np.savez_compressed(run.export / "reference-branch.npz", **coordinates)
        print(numbers["summary"], flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp115 compute run")
    parser.add_argument("--run-id", help="unused pre-reserved analyse identity")
    arguments = parser.parse_args()
    analyse(arguments.source, run_id=arguments.run_id)
