"""Retain untrained-network native sparse rasters; never analyse or plot."""

import argparse
import os
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
import torch
from experiments.exp054 import evidence, inputs, recipe
from pingstore.contracts import PingstoreError, write_json_atomic
from snnlab.sim.execution import GraphExecutor, plan_graph
from snnlab.sim.models import poisson_spikes
from snnlab.sim.timing import duration_steps


def simulate_probe(cfg, item, destination, device):
    bundle = recipe.author_network(cfg, item)
    model = GraphExecutor(plan_graph(bundle.graph), seed=cfg["seed"])
    parameters = model.parameter_map()
    values = recipe.initial_parameters(cfg, item)
    if set(parameters) != set(values):
        raise PingstoreError("exp054 graph parameter roles differ from recipe")
    with torch.no_grad():
        for name, value in values.items():
            if parameters[name].shape != value.shape:
                raise PingstoreError(f"exp054 graph matrix shape mismatch: {name}")
            parameters[name].copy_(value)
    model.to(device).eval()
    steps = duration_steps(cfg["sim_ms"], cfg["dt_ms"])
    channels = cfg["n_e"] if item["private"] else cfg["shared_n_in"]
    drive = poisson_spikes(
        item["rate_hz"],
        (steps, 1, channels),
        cfg["dt_ms"],
        torch.Generator().manual_seed(cfg["encoder_seed"]),
        device=device,
    )
    started = perf_counter()
    with torch.inference_mode():
        # No continuation: each probe resets voltage, conductance, refractory
        # counters and delay history, including before the unrecorded burn-in.
        result = model(
            {"drive": drive}, diagnostics=False, recording=recipe.recording(cfg)
        )
    arrays = {
        "dt": np.float32(cfg["dt_ms"]),
        "T": np.int32(steps),
        "n_trials": np.int32(1),
        "n_e": np.int32(cfg["n_e"]),
        "n_i": np.int32(cfg["n_i"]),
        "recording_start_step": np.int32(duration_steps(cfg["burn_ms"], cfg["dt_ms"])),
    }
    for prefix, label in (("e", "E"), ("i", "I")):
        coordinates = result.recorded_signals[f"{label}.spikes"].cpu().numpy()
        for field, axis in (("t", 0), ("trial", 1), ("cell", 2)):
            arrays[f"{prefix}_{field}"] = coordinates[:, axis].astype(np.int32)
    np.savez_compressed(destination, **arrays)
    evidence.raster(destination, cfg)
    return {
        "job": item,
        "graph_digest": bundle.manifest["graph_digest"],
        "duration_seconds": perf_counter() - started,
    }


def compute(*, run_id=None):
    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    device = torch.device(
        os.environ.get("PINGLAB_DEVICE")
        or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    with inputs.execution(
        REPO, "compute", sources={}, run_id=run_id, configuration=cfg
    ) as run:
        run.record["execution"].update(
            environment={
                "PINGLAB_SMOKE": "1" if cfg["profile"] == "smoke" else "0",
                **(
                    {"PINGLAB_DEVICE": os.environ["PINGLAB_DEVICE"]}
                    if os.environ.get("PINGLAB_DEVICE")
                    else {}
                ),
            },
            executor=cfg["executor"],
            device=str(device),
            snnlab_version=version("snnlab"),
            torch_version=str(torch.__version__),
            trials=[],
        )
        for item in recipe.jobs(cfg):
            destination = run.export / f"probe--{item['id']}--rasters.npz"
            run.record["execution"]["trials"].append(
                simulate_probe(cfg, item, destination, device)
            )
        write_json_atomic(
            run.export / "recordings.json",
            {
                "schema": "exp054.recordings/v1",
                "recipe": cfg,
                "jobs": recipe.jobs(cfg),
            },
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="fresh v4 identity reserved before dispatch")
    args = parser.parse_args()
    compute(run_id=args.run_id)


if __name__ == "__main__":
    main()
