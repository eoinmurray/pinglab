"""Compute the complete closure grid and independent numerical checks."""

import argparse
import hashlib
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
import scipy
from experiments.exp115 import recipe
from experiments.exp115.numerics import Model, NumericalFailure, amplitude_ramp, scan
from pingstore.contracts import write_json_atomic
from pingstore.stages import stage_run


def environment(run):
    run.record["execution"]["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    run.record["provenance"]["implementation_sha256"] = {
        str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }


def pack(scan_result, arrays, prefix):
    rows = scan_result["rows"]
    arrays[prefix + "drive_nA"] = np.array([r["drive_nA"] for r in rows])
    arrays[prefix + "valid"] = np.array(["failure" not in r for r in rows])
    for key, shape in (
        ("state", (4,)),
        ("eigenvalues", (4, 2)),
        ("gain_derivatives", (2, 3)),
        ("K_minus_KH", ()),
        ("rate_residual", ()),
        ("flow_residual", ()),
        ("scalar_discrepancy", ()),
    ):
        arrays[prefix + key] = np.array(
            [r.get(key, np.full(shape, np.nan)) for r in rows]
        )
    return {
        "count": scan_result["count"],
        "crossings": scan_result["crossings"],
        "failed_points": [
            {"index": j, **r} for j, r in enumerate(rows) if "failure" in r
        ],
        "scalar_fallback_count": sum(r.get("scalar_fallback", False) for r in rows),
    }


def compute(*, run_id=None):
    cfg = recipe.configuration()
    conditions = recipe.conditions(cfg)
    with stage_run(
        REPO, recipe.SLUG, "compute", run_id=run_id, configuration=cfg
    ) as run:
        environment(run)
        shared_scans = None
        for index, condition in enumerate(conditions, 1):
            name = recipe.condition_id(condition)
            unit = run.export / name
            unit.mkdir()
            arrays, levels = {}, []
            base_condition = condition["kappa"] == cfg["kappa_grid"][0]
            if base_condition:
                shared_scans = []
            print(f"[{index}/{len(conditions)}] {name}: continuation", flush=True)
            for level, count in enumerate(cfg["drive_counts"]):
                model = Model(cfg, condition, tight=level > 0)
                result = scan(
                    model,
                    count,
                    equilibrium_scan=None if base_condition else shared_scans[level],
                )
                if base_condition:
                    shared_scans.append(result)
                levels.append(pack(result, arrays, f"level{level}_"))
            np.savez_compressed(unit / "continuation.npz", **arrays)
            ramp_record = None
            if recipe.requires_criticality_ramp(condition, cfg):
                crossing = next(
                    (
                        row
                        for row in levels[0]["crossings"]
                        if "failure" not in row and row["direction"] == "loss"
                    ),
                    None,
                )
                if crossing is not None:
                    print("  upward/downward criticality ramps", flush=True)
                    try:
                        ramp = amplitude_ramp(Model(cfg, condition), crossing["drive_nA"])
                        np.savez_compressed(unit / "ramps.npz", **ramp)
                        ramp_record = {"status": "complete"}
                    except (NumericalFailure, ValueError, FloatingPointError) as exc:
                        failure = f"{type(exc).__name__}: {exc}"
                        ramp_record = {"status": "failed", "failure": failure}
                        print(f"  unresolved ramps: {failure}", flush=True)
                else:
                    ramp_record = {
                        "status": "failed",
                        "failure": "no refined stable-to-unstable crossing",
                    }
            write_json_atomic(
                unit / "crossings.json",
                {
                    "schema": "exp115.compute-condition/v2",
                    "condition": condition,
                    "levels": levels,
                    **({"ramp": ramp_record} if ramp_record is not None else {}),
                },
            )
            print(
                f"  saved {len(levels[0]['crossings'])} candidate crossings", flush=True
            )
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="unused pre-reserved compute identity")
    compute(run_id=parser.parse_args().run_id)
