"""Run the proposal's 200 ms Task 2 protocol for one HCO on CPU.

Run compute.py [--run-id RESERVED_ID] from the repository root. A new Pingstore
run retains native network evidence, complete float64 state/conductance traces,
and execution metrics. No batch benchmark or downstream stage is launched.
"""

import argparse
import sys
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

import numpy as np
import torch
from experiments.exp147 import recipe
from experiments.exp147.custom.execution import execute
from pingstore.library import stage_run, write_json_atomic


def compute(*, run_id=None):
    cfg = recipe.configuration()
    torch.set_num_threads(cfg["torch_threads"])
    bundle = recipe.author_network(cfg)
    with stage_run(REPO, "exp147", "compute", configuration=cfg, run_id=run_id) as run:
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={"simulation": ["start"]}
        )
        run.record["execution"].update(
            snnlab_version=version("snnlab"), device=cfg["device"], dtype=cfg["dtype"]
        )
        bundle.write(run.export / "simulation" / "network.bundle", visualise=False)
        bundle.visualise(run.export / "circuit.svg", view="circuit")
        result, elapsed = execute(
            bundle,
            recipe.clock(cfg),
            cfg,
            snapshot_path=run.export / "simulation" / "start",
        )
        arrays = {
            name: values[:, 0].copy()
            for name, values in result.numpy(batch=0).outputs.items()
        }
        steps = round(cfg["duration_ms"] / cfg["dt_ms"])
        arrays["input_time_ms"] = np.arange(steps) * cfg["dt_ms"]
        arrays["time_ms"] = (np.arange(steps) + 1) * cfg["dt_ms"]
        np.savez_compressed(run.export / "traces.npz", **arrays)
        result.write_metrics(run.export / "metrics.json")
        write_json_atomic(
            run.export / "timing.json",
            dict(
                execution_seconds=elapsed,
                scope="Single HCO GraphExecutor call; excludes build, snapshot and export",
            ),
        )
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    compute(**vars(parser.parse_args()))
