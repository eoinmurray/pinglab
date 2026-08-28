from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools"), str(REPO / "tools/snnsim")]

import numpy as np
import snnlang as snn
from experiments.exp074 import recipe
from experiments.exp074.recipe import (
    DISPLAY_TRIAL,
    DT_MS,
    N_BATCH,
    N_INPUT,
    SCALE,
    T_MS,
)
from experiments.helpers import snnlang_stages as stages
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def _trial_events(
    rasters: np.lib.npyio.NpzFile, prefix: str, trial: int
) -> tuple[np.ndarray, np.ndarray]:
    mask = rasters[f"{prefix}_trial"] == trial
    return rasters[f"{prefix}_t"][mask], rasters[f"{prefix}_cell"][mask]


def analyse(identity, *, run_id=None):
    source = stages.source(REPO, recipe, identity, "compute")
    bundle = snn.load_bundle(source.export / "network.bundle")
    sim_dir = source.export / "simulation"
    with stages.execution(
        REPO, recipe, "analyse", sources={"compute": source}, run_id=run_id
    ) as run:
        with np.load(source.export / "input_spikes.npz") as arrays:
            input_spikes = arrays["input_spikes"]
        with np.load(sim_dir / "rasters.npz") as rasters:
            if not np.isclose(float(rasters["dt"]), DT_MS, rtol=1e-7, atol=0):
                raise PingstoreError("raster timestep does not match recipe")
            e_t, e_cell = _trial_events(rasters, "e", DISPLAY_TRIAL)
            i_t, i_cell = _trial_events(rasters, "i", DISPLAY_TRIAL)
        input_t, input_cell = np.nonzero(input_spikes[:, DISPLAY_TRIAL, :])
        event_counts = {
            "input": int(len(input_t)),
            "e": int(len(e_t)),
            "i": int(len(i_t)),
        }
        np.savez_compressed(
            run.export / "display.npz",
            input_t=input_t,
            input_cell=input_cell,
            e_t=e_t,
            e_cell=e_cell,
            i_t=i_t,
            i_cell=i_cell,
        )
        metrics = load_json(sim_dir / "metrics.json")
        graph = bundle.graph
        total_input = int(input_spikes.sum())
        realised_input_rate = total_input / (N_BATCH * N_INPUT * (T_MS / 1000.0))
        payload = {
            "purpose": "end-to-end integration demonstration",
            "graph": {
                "name": graph["name"],
                "digest": bundle.manifest["graph_digest"],
                "digest_short": bundle.manifest["graph_digest"][:19],
                "populations": len(graph["populations"]),
                "projections": len(graph["projections"]),
                "operations": len(graph["operations"]),
                "parameter_tensors": len(graph["parameters"]),
            },
            "config": SCALE,
            "input": {
                "shape": list(input_spikes.shape),
                "shape_text": " × ".join(str(n) for n in input_spikes.shape),
                "total_spikes": total_input,
                "realised_rate_hz": realised_input_rate,
            },
            "output": {
                "rate_e_hz": metrics["rate_e_hz"],
                "rate_i_hz": metrics["rate_i_hz"],
                "display_trial": DISPLAY_TRIAL,
                "display_trial_spikes": event_counts,
            },
        }
        write_json_atomic(
            run.export / "results.json", {"schema": "exp074.analysis/v1", **payload}
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="explicit completed input run")
    parser.add_argument("--run-id", help="unused source-neutral reservation")
    args = parser.parse_args()
    try:
        analyse(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
