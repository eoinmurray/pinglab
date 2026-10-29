"""Execute the proposal's single-neuron protocol in a new Pingstore run.

Run compute.py from the repository root. Save the compiled bundle, circuit,
native simulation-start snapshot, float64 traces and native execution metrics.
Analysis and plotting are separate commands with explicit source run IDs.
"""

import argparse
import sys
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

import numpy as np
import torch
from experiments.exp146 import recipe
from experiments.exp146.custom.execution import execute
from pingstore.library import stage_run, write_json_atomic


def compute(*, run_id=None):
    cfg = recipe.configuration()
    torch.set_num_threads(cfg["torch_threads"])
    bundle = recipe.author_network(cfg)
    drive = recipe.current_waveform(cfg)
    with stage_run(REPO, "exp146", "compute", configuration=cfg, run_id=run_id) as run:
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={"simulation": ["start"]}
        )
        run.record["execution"].update(
            snnlab_version=version("snnlab"), device=cfg["device"], dtype=cfg["dtype"]
        )
        bundle.write(run.export / "simulation" / "network.bundle", visualise=False)
        bundle.visualise(run.export / "circuit.svg", view="circuit")
        result, elapsed = execute(
            bundle, drive, cfg, snapshot_path=run.export / "simulation" / "start"
        )
        arrays = {
            key: value[:, 0].copy()
            for key, value in result.numpy(batch=0).outputs.items()
        }
        steps = len(drive)
        arrays["time_ms"] = (np.arange(steps) + 1) * cfg["dt_ms"]
        arrays["input_time_ms"] = np.arange(steps) * cfg["dt_ms"]
        np.testing.assert_array_equal(arrays["current_na"], drive[:, 0, 0].numpy())
        np.savez_compressed(run.export / "traces.npz", **arrays)
        result.write_metrics(run.export / "metrics.json")
        write_json_atomic(
            run.export / "timing.json",
            dict(
                execution_seconds=elapsed,
                scope="GraphExecutor call; excludes build, snapshot and export",
            ),
        )
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    compute(**vars(parser.parse_args()))
