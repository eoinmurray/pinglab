"""Retain the fixed-input coupling sweep; never analyse, draw or publish."""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools"), str(REPO / "tools/snnsim")]

from execution import ExecutionSpec, save_runtime_state, simulate
from experiments.exp086 import evidence, inputs, recipe
from pingstore.contracts import PingstoreError, write_json_atomic


def save_recordings(path, recordings, steps):
    """Keep every exposed spike sample; full-profile state traces are separate."""
    shapes = evidence.recording_shapes(steps)
    populations = {key for key in recordings if key.startswith("population_")}
    if populations != set(shapes):
        raise PingstoreError("unexpected exp086 recorded populations")
    arrays = {}
    for key in shapes:
        tensor = recordings[key]
        array = tensor.detach().cpu().numpy()
        if array.shape != shapes[key] or not np.all((array == 0) | (array == 1)):
            raise PingstoreError(f"invalid exp086 spike recording: {key}")
        arrays[key] = array.astype(np.uint8)
    np.savez_compressed(path, **arrays)


def compute(*, run_id=None):
    cfg = recipe.configuration()
    onset = round(recipe.COUPLING_ONSET_MS / recipe.DT_MS)
    steps = round(recipe.T_MS / recipe.DT_MS)
    with inputs.execution(
        REPO, "compute", sources={}, run_id=run_id, configuration=cfg
    ) as run:
        drives = recipe.make_inputs()
        np.savez_compressed(
            run.export / "inputs.npz",
            **{k: v.detach().cpu().numpy() for k, v in drives.items()},
        )
        evidence.binary_arrays(
            run.export / "inputs.npz",
            {
                f"drive_A_{recipe.INPUT_RATE_A_HZ:g}_Hz": (steps, 1, recipe.N_INPUT),
                f"drive_B_{recipe.INPUT_RATE_B_HZ:g}_Hz": (steps, 1, recipe.N_INPUT),
            },
            np.float32,
        )
        bundles = {}
        for branch in recipe.branches():
            bundle = recipe.author_network(k_ee=branch["k"], k_ei=branch["k"])
            if bundle.manifest["graph_digest"] != cfg["graphs"][branch["label"]]:
                raise PingstoreError("network changed during exp086 compute")
            bundle.write(run.export / "branches" / branch["label"] / "network.bundle")
            bundles[branch["label"]] = bundle
        prefix = simulate(
            ExecutionSpec(
                kind="simulate",
                executor="graph",
                graph=bundles[recipe.label(0.0)].graph,
                inputs={name: value[:onset] for name, value in drives.items()},
                seed=recipe.NETWORK_SEED,
            )
        )
        if prefix.runtime_state is None:
            raise RuntimeError(
                "the uncoupled prefix returned no reusable runtime state"
            )
        if prefix.runtime_state.completed_steps != onset:
            raise PingstoreError("exp086 prefix state has wrong duration")
        save_runtime_state(run.export / "prefix-state", prefix.runtime_state)
        save_recordings(run.export / "prefix-spikes.npz", prefix.recordings, onset)
        suffix = {name: value[onset:] for name, value in drives.items()}
        for branch in recipe.branches():
            result = simulate(
                ExecutionSpec(
                    kind="simulate",
                    executor="graph",
                    graph=bundles[branch["label"]].graph,
                    inputs=suffix,
                    seed=recipe.NETWORK_SEED,
                ),
                runtime_state=prefix.runtime_state.detached(),
            )
            save_recordings(
                run.export / "branches" / branch["label"] / "spikes.npz",
                result.recordings,
                steps - onset,
            )
        write_json_atomic(
            run.export / "evidence.json",
            {
                "schema": "exp086.compute/v1",
                "recipe": cfg,
                "branches": recipe.branches(),
            },
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="unused v4 identity reserved before dispatch")
    args = parser.parse_args()
    try:
        compute(run_id=args.run_id)
    except (PingstoreError, OSError, KeyError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"exp086 compute: {exc}\n")


if __name__ == "__main__":
    main()
