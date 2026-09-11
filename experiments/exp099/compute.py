"""Execute the private-afferent protocol with SNNSIM's delayed graph executor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools"), str(REPO / "tools/snnsim")]
import numpy as np
import torch
from execution import GraphExecutor, plan_graph
from experiments.exp099 import recipe
from pingstore.stages import stage_run


def build_model(cfg):
    bundle = recipe.author_network(cfg)
    model = GraphExecutor(plan_graph(bundle.graph), seed=cfg["seed"])
    # Each aggregate channel belongs to exactly one postsynaptic cell.
    with torch.no_grad():
        for pop in ("e", "i"):
            model.parameter_map()[f"private_{pop}_to_{pop.upper()}.weight"].copy_(
                torch.eye(cfg[f"n_{pop}"]) * cfg["external_weight_us"]
            )
    return bundle, model


def simulate(cfg, *, chunk_steps=1000):
    torch.set_num_threads(1)
    bundle, model = build_model(cfg)
    counts = recipe.afferent_counts(cfg)
    fields = ["E.spikes", "I.spikes", "E.voltage", "I.voltage"] + [
        f"{p.id}.conductance" for p in model.plan.projections
    ]
    retained = {**counts}
    steps = len(counts["private_e"])
    for pop in ("e", "i"):
        retained[f"spk_{pop}"] = np.empty((steps, cfg[f"n_{pop}"]), dtype=bool)
        retained[f"mean_v_{pop}"] = np.empty(steps, dtype=np.float32)
    for p in model.plan.projections:
        retained[f"mean_{p.id}"] = np.empty(steps, dtype=np.float32)
    state = None
    with torch.inference_mode():
        for start in range(0, steps, chunk_steps):
            stop = min(start + chunk_steps, steps)
            result = model(
                {
                    k: torch.from_numpy(v[start:stop, None].astype(np.float32))
                    for k, v in counts.items()
                },
                recording_fields=fields,
                runtime_state=state,
            )
            state = result.runtime_state
            data = result.recordings
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
        simulation_start = perf_counter()
        bundle, recording, weights = simulate(cfg)
        simulation_seconds = perf_counter() - simulation_start
        run.record["execution"]["simulation_seconds"] = simulation_seconds
        print(f"Simulation wall time: {simulation_seconds:.3f} s", flush=True)
        bundle.write(run.export / "network.bundle")
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
