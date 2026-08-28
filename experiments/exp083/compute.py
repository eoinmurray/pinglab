"""Retain the full default PING drive sweep; never analyse, plot or publish."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp083 import evidence, inputs, recipe
from pingstore.contracts import PingstoreError, write_json_atomic


def simulate(graph, spikes):
    """Use the original graph-native execution seam and initialization seed."""
    sys.path.insert(0, str(REPO / "tools/snnsim"))
    import torch
    from execution import ExecutionSpec
    from execution import simulate as execute

    result = execute(
        ExecutionSpec(
            kind="simulate",
            executor="graph",
            graph=graph,
            inputs={"drive": torch.from_numpy(spikes).float()},
            seed=recipe.NETWORK_SEED,
        )
    )
    return {
        "e_spikes": result.recordings["population_0"].cpu().numpy().astype(np.uint8),
        "i_spikes": result.recordings["population_1"].cpu().numpy().astype(np.uint8),
    }


def compute(*, run_id=None):
    with inputs.execution(REPO, "compute", sources={}, run_id=run_id) as run:
        bundle = recipe.author_network()
        bundle.write(run.export / "network.bundle", visualise=False)
        (run.export / "conditions").mkdir()
        for condition in recipe.conditions():
            spikes = recipe.make_inputs(condition["input_rate_hz"])
            arrays = {"input_spikes": spikes, **simulate(bundle.graph, spikes)}
            evidence.recording_arrays(arrays)
            np.savez_compressed(
                run.export / condition["file"],
                input_spikes=arrays["input_spikes"],
                e_spikes=arrays["e_spikes"],
                i_spikes=arrays["i_spikes"],
            )
        write_json_atomic(
            run.export / "evidence.json",
            {
                "schema": "exp083.compute/v1",
                "recipe": recipe.configuration(),
                "graph": {
                    "digest": bundle.manifest["graph_digest"],
                    "name": bundle.graph["name"],
                },
                "conditions": recipe.conditions(),
            },
        )
        run.record["execution"]["simulator"] = {
            "executor": "graph",
            "network_seed": recipe.NETWORK_SEED,
            "input": "retained input_spikes",
            "recordings": ["population_0", "population_1"],
        }
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id", help="unused v3 reservation allocated before dispatch"
    )
    args = parser.parse_args()
    try:
        compute(run_id=args.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"exp083 compute: {exc}\n")


if __name__ == "__main__":
    main()
