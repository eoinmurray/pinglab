"""Compute exp117 stability continuation and near-onset nonlinear ramps."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
import scipy
from experiments.exp117 import inputs, numerics, recipe
from pingstore.contracts import write_json_atomic


def compute(*, run_id=None):
    with inputs.execution(REPO, "compute", sources={}, run_id=run_id) as run:
        cfg = recipe.validate(run.record["execution"]["configuration"])
        rows = numerics.continuation(cfg)
        hopf = numerics.refine_hopf(rows, cfg)
        ramps = numerics.criticality_ramps(hopf, cfg)
        tau_sweep = []
        reference_tau = cfg["imported_parameters"]["tau_GABA_reference_ms"]
        for tau_gaba_ms in cfg["mean_field_choices"]["tau_GABA_sweep_ms"]:
            if tau_gaba_ms == reference_tau:
                sweep_rows, sweep_hopf = rows, hopf
            else:
                sweep_rows = numerics.continuation(
                    cfg, tau_gaba_ms=tau_gaba_ms
                )
                sweep_hopf = numerics.refine_hopf(
                    sweep_rows, cfg, tau_gaba_ms=tau_gaba_ms
                )
            tau_sweep.append(
                {
                    "tau_GABA_ms": tau_gaba_ms,
                    "hopf": sweep_hopf,
                }
            )
        write_json_atomic(
            run.export / "continuation.json",
            {
                "schema": "exp117.compute/v3",
                "configuration": cfg,
                "continuation": rows,
                "hopf": hopf,
                "tau_GABA_sweep": tau_sweep,
            },
        )
        np.savez_compressed(run.export / "recording.npz", **ramps)
        run.record["execution"]["environment"] = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        }
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(compute(run_id=args.run_id))


if __name__ == "__main__":
    main()
