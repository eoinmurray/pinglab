"""Measure saved exp086 spikes; never simulate, draw or publish."""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.exp086 import evidence, inputs, measurements, recipe
from pingstore.contracts import PingstoreError, write_json_atomic


def analyse(identity, *, run_id=None):
    source = inputs.source(REPO, identity, "compute")
    cfg = evidence.compute_contract(source)
    steps = round((recipe.T_MS - recipe.COUPLING_ONSET_MS) / recipe.DT_MS)
    with inputs.execution(
        REPO,
        "analyse",
        sources={"compute": source},
        run_id=run_id,
        configuration=recipe.MEASUREMENT,
    ) as run:
        evidence.acquisition(source)
        trajectories = []
        for branch in recipe.branches():
            recordings = evidence.binary_arrays(
                source.file("branches", branch["label"], "spikes.npz"),
                evidence.recording_shapes(steps),
                np.uint8,
            )
            trajectory = measurements.analyse_trajectory(recordings, k=branch["k"])
            np.savez_compressed(
                run.export / f"{branch['label']}.npz",
                **{k: trajectory[k] for k in recipe.ARRAY_KEYS},
            )
            trajectories.append(measurements.public_summary(trajectory))
        selected = measurements.choose_intermediate(trajectories)
        write_json_atomic(
            run.export / "results.json",
            {
                "schema": "exp086.analysis/v1",
                "recipe": cfg,
                "measurement": recipe.MEASUREMENT,
                "status": recipe.STATUS,
                "completed_methods": [1, 2, 3],
                "simulation_run": True,
                "fixed_inputs": {
                    "seeds": list(recipe.INPUT_SEEDS),
                    "reused_suffix_for_every_k": True,
                },
                "coupling": {
                    "K_EE_equals_K_EI": True,
                    "delay_ms": recipe.COUPLING_DELAY_MS,
                    "values_us": recipe.K_VALUES.tolist(),
                },
                "trajectories": trajectories,
                "selected_intermediate": selected,
            },
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="explicit exp086 compute ID")
    parser.add_argument("--run-id", help="unused v3 identity reserved before dispatch")
    args = parser.parse_args()
    try:
        analyse(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, KeyError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"exp086 analyse: {exc}\n")


if __name__ == "__main__":
    main()
