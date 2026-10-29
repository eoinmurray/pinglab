"""Execute the private-afferent protocol with SNNSIM's delayed graph executor.

Invoke this stage independently through its CLI; it records configuration,
provenance and scientific outputs in Pingstore. Importing the module does not
launch computation or another stage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
import torch
from experiments.exp099 import dataset, recipe
from experiments.exp099.custom.network_artifacts import model_execution
from pingstore.library import stage_run
from snnlab.sim import Performance, build  # noqa: TID251


def build_model(cfg, execution=None):
    bundle = recipe.author_network(cfg)
    built = build(
        bundle=bundle,
        seed=cfg["seed"],
        device=cfg["device"],
        performance=Performance(**cfg["performance"]),
    )
    model = built.model
    if execution is not None:
        execution["performance"] = built.metrics["performance"]
    # Each aggregate channel belongs to exactly one postsynaptic cell.
    with torch.no_grad():
        for pop in ("e", "i"):
            model.parameter_map()[f"private_{pop}_to_{pop.upper()}.weight"].copy_(
                torch.eye(cfg[f"n_{pop}"]) * cfg["external_weight_us"]
            )
    return bundle, model


def simulate(cfg, *, artifacts, chunk_steps=1000, execution=None):
    torch.set_num_threads(cfg["torch_threads"])
    bundle, model = build_model(cfg, execution)
    counts = dataset.afferent_counts(cfg)
    retained = {**counts}
    steps = len(counts["private_e"])
    for pop in ("e", "i"):
        retained[f"spk_{pop}"] = np.empty((steps, cfg[f"n_{pop}"]), dtype=bool)
        retained[f"mean_v_{pop}"] = np.empty(steps, dtype=np.float32)
    for p in model.plan.projections:
        retained[f"mean_{p.id}"] = np.empty(steps, dtype=np.float32)
    state = None
    with (
        model_execution(model, bundle, artifacts, seed=cfg["seed"], settings=cfg),
        torch.inference_mode(),
    ):
        for start in range(0, steps, chunk_steps):
            stop = min(start + chunk_steps, steps)
            result = model(
                {
                    k: torch.from_numpy(v[start:stop, None].astype(np.float32))
                    for k, v in counts.items()
                },
                runtime_state=state,
            )
            state = result.runtime_state
            data = result.diagnostics
            for pop in ("e", "i"):
                retained[f"spk_{pop}"][start:stop] = (
                    data[f"{pop.upper()}.spikes"][:, 0].numpy().astype(bool)
                )
                retained[f"mean_v_{pop}"][start:stop] = (
                    data[f"{pop.upper()}.voltage"][:, 0].mean(1).numpy()
                )
            for p in model.plan.projections:
                retained[f"mean_{p.id}"][start:stop] = (
                    data[f"{p.id}.conductance"][:, 0].mean(1).numpy()
                )
            if start % (chunk_steps * 10) == 0:
                print(
                    f"Simulated {stop * cfg['dt_ms']:.0f} / {cfg['t_ms']:.0f} ms",
                    flush=True,
                )
    weights = {k: v.detach().numpy().copy() for k, v in model.parameter_map().items()}
    return bundle, retained, weights


def compute(*, run_id=None):
    cfg = recipe.configuration()
    with stage_run(
        REPO, recipe.SLUG, "compute", run_id=run_id, configuration=cfg
    ) as run:
        run.record["scientific_artifacts"] = {
            "snapshot_stages": {"simulation": ["start"]}
        }
        bundle = recipe.author_network(cfg)
        bundle.visualise(run.export / "network.svg", view="circuit")
        simulation_start = perf_counter()
        bundle, recording, weights = simulate(
            cfg, artifacts=run.export / "simulation", execution=run.record["execution"]
        )
        simulation_seconds = perf_counter() - simulation_start
        run.record["execution"]["simulation_seconds"] = simulation_seconds
        print(f"Simulation wall time: {simulation_seconds:.3f} s", flush=True)
        bundle.write(run.export / "network.bundle", visualise=False)
        np.savez_compressed(run.export / "recording.npz", **recording)
        np.savez_compressed(run.export / "weights.npz", **weights)
        run.record["execution"]["executor"] = "snnsim.GraphExecutor"
        run.record["execution"]["parameter_binding"] = (
            "weights.npz contains all actual execution matrices; diagonal external projections replace zero placeholders in network.bundle"
        )
    return run.run_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id")
    a = p.parse_args()
    compute(run_id=a.run_id)


if __name__ == "__main__":
    main()
