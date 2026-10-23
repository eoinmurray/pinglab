"""Execute the baseline/ramp/hold protocol in an independent Pingstore run.

Run compute.py with optional --seed, --device, --final-i-gain and --run-id.
The native graph, start snapshot, sparse spike events, input checksum and weight
verification are retained. Analysis and presentation run separately.
"""

import argparse
import hashlib
import os
import sys
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
import torch
from experiments.exp148 import dataset, recipe
from experiments.exp148.custom.execution import execute
from pingstore.contracts import write_json_atomic
from pingstore.stages import stage_run


def compute(*, seed=42, device="cpu", final_i_gain=0.0, run_id=None):
    cfg = recipe.configuration(
        seed=seed,
        device=os.environ.get("SNNLAB_DEVICE", device),
        final_i_gain=final_i_gain,
    )
    torch.set_num_threads(cfg["torch_threads"])
    bundle = recipe.author_network(cfg)
    external = dataset.encode_drive(cfg).to(cfg["device"])
    with stage_run(REPO, "exp148", "compute", configuration=cfg, run_id=run_id) as run:
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={"simulation": ["start"]}
        )
        run.record["execution"]["snnlab_version"] = version("snnlab")
        bundle.write(run.export / "simulation" / "network.bundle", visualise=False)
        bundle.visualise(run.export / "circuit.svg", view="circuit")
        arrays, audit, last = execute(
            bundle,
            external,
            cfg,
            snapshot_path=run.export / "simulation" / "start",
            progress=True,
        )
        np.savez_compressed(run.export / "events.npz", **arrays)
        audit["external_float32_sha256"] = hashlib.sha256(
            external.cpu().numpy().tobytes()
        ).hexdigest()
        write_json_atomic(run.export / "verification.json", audit)
        # These native metrics describe only the last call; total timing is in verification.json.
        last.write_metrics(run.export / "final-call-metrics.json")
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--final-i-gain", type=float, default=0.0)
    parser.add_argument("--run-id")
    compute(**vars(parser.parse_args()))
