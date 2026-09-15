"""Simulate the complete fixed two-PING coupling-by-detuning grid."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools"), str(REPO / "tools/snnsim")]

import numpy as np
import torch
from experiments.exp114 import recipe
from pingstore.stages import stage_run
from tools.snnsim.execution import GraphExecutor, plan_graph  # noqa: TID251


def simulate(cfg, condition):
    bundle = recipe.author_network(cfg, condition)
    model = GraphExecutor(plan_graph(bundle.graph), seed=condition["seed"])
    with torch.no_grad():
        for module in (1, 2):
            model.parameter_map()[f"private_{module}_to_E{module}.weight"].copy_(
                torch.eye(cfg["n_e_per_module"]) * cfg["external_weight_us"]
            )
    afferents = recipe.afferent_counts(cfg, condition)
    fields = [f"{kind}{module}.spikes" for module in (1, 2) for kind in ("E", "I")]
    inputs = {key: torch.from_numpy(value[:, None].astype(np.float32)) for key, value in afferents.items()}
    with torch.inference_mode():
        result = model(inputs, recording_fields=fields)
    arrays = {}
    for module in (1, 2):
        for kind in ("E", "I"):
            arrays[f"spk_{kind.lower()}{module}"] = result.recordings[f"{kind}{module}.spikes"][:, 0].numpy().astype(bool)
    return bundle, arrays


def compute(*, run_id=None):
    torch.set_num_threads(1)
    cfg = recipe.configuration()
    with stage_run(REPO, recipe.SLUG, "compute", run_id=run_id, configuration=cfg) as run:
        started = perf_counter()
        exported = {}
        first_bundle = None
        rows = []
        for index, condition in enumerate(recipe.conditions(cfg), start=1):
            cell_started = perf_counter()
            bundle, arrays = simulate(cfg, condition)
            if first_bundle is None:
                first_bundle = bundle
            for name, values in arrays.items():
                exported[f"{condition['condition_id']}__{name}"] = values
            rows.append({**condition, "simulation_seconds": perf_counter() - cell_started})
            print(f"[{index:02d}/60] {condition['condition_id']} {rows[-1]['simulation_seconds']:.3f}s", flush=True)
        np.savez_compressed(run.export / "recording.npz", **exported)
        first_bundle.write(run.export / "network.bundle")
        run.record["execution"].update(
            executor="snnsim.GraphExecutor",
            simulation_seconds=perf_counter() - started,
            condition_timings=rows,
            complete_grid=True,
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    compute(run_id=args.run_id)


if __name__ == "__main__":
    main()
