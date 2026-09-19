"""Compute the minimal continuation, ramp and robustness evidence."""

import argparse
import hashlib
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
import scipy
from experiments.exp115.numerics import Model, NumericalFailure, amplitude_ramp, scan
from experiments.exp116 import recipe
from pingstore.contracts import write_json_atomic
from pingstore.stages import stage_run


def record_environment(run):
    run.record["execution"]["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    sources = list(Path(__file__).parent.glob("*.py")) + [
        REPO / "experiments" / "exp115" / "numerics.py"
    ]
    run.record["provenance"]["implementation_sha256"] = {
        str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(sources)
    }


def pack_scan(result, arrays, prefix):
    rows = result["rows"]
    arrays[prefix + "drive_nA"] = np.array([row["drive_nA"] for row in rows])
    arrays[prefix + "valid"] = np.array(["failure" not in row for row in rows])
    for key, shape in (
        ("state", (4,)),
        ("eigenvalues", (4, 2)),
        ("K_minus_KH", ()),
        ("rate_residual", ()),
        ("flow_residual", ()),
        ("scalar_discrepancy", ()),
    ):
        arrays[prefix + key] = np.array(
            [row.get(key, np.full(shape, np.nan)) for row in rows]
        )
    return {
        "count": result["count"],
        "crossings": result["crossings"],
        "failed_indices": [index for index, row in enumerate(rows) if "failure" in row],
    }


def compute(*, run_id=None):
    cfg = recipe.configuration()
    planned = recipe.conditions(cfg)
    with stage_run(
        REPO, recipe.SLUG, "compute", run_id=run_id, configuration=cfg
    ) as run:
        record_environment(run)
        for index, condition in enumerate(planned, 1):
            name = recipe.condition_id(condition)
            unit = run.export / name
            unit.mkdir()
            arrays, levels = {}, []
            print(f"[{index}/{len(planned)}] {name}", flush=True)
            for level, count in enumerate(cfg["drive_counts"]):
                result = scan(Model(cfg, condition, tight=level > 0), count)
                levels.append(pack_scan(result, arrays, f"level{level}_"))
            np.savez_compressed(unit / "continuation.npz", **arrays)
            ramp_status = None
            if recipe.is_reference(condition, cfg):
                crossing = next(
                    (
                        row
                        for row in levels[0]["crossings"]
                        if "failure" not in row and row["direction"] == "loss"
                    ),
                    None,
                )
                if crossing is None:
                    ramp_status = {"status": "failed", "reason": "no loss crossing"}
                else:
                    try:
                        ramp = amplitude_ramp(
                            Model(cfg, condition), crossing["drive_nA"]
                        )
                        np.savez_compressed(unit / "ramps.npz", **ramp)
                        ramp_status = {"status": "complete"}
                    except (NumericalFailure, ValueError, FloatingPointError) as exc:
                        ramp_status = {
                            "status": "failed",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
            write_json_atomic(
                unit / "crossings.json",
                {
                    "schema": "exp116.compute-condition/v1",
                    "condition": condition,
                    "levels": levels,
                    **({"ramp": ramp_status} if ramp_status is not None else {}),
                },
            )
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="unused pre-reserved compute identity")
    compute(run_id=parser.parse_args().run_id)
