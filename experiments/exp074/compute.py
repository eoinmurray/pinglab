from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools"), str(REPO / "tools/snnsim")]

import numpy as np
from experiments.exp074 import recipe
from experiments.exp074.recipe import (
    DT_MS,
    INPUT_RATE_HZ,
    N_BATCH,
    N_INPUT,
    SEED,
    T_MS,
    author_network,
)
from experiments.helpers import snnlang_stages as stages
from pingstore.contracts import PingstoreError


def run_simulator(
    bundle_dir: Path, input_path: Path, sim_dir: Path, provenance: Path
) -> dict:
    cmd = [
        sys.executable,
        str(REPO / "tools/snnsim/tool.py"),
        "sim",
        "--bundle",
        str(bundle_dir),
        "--input-file",
        str(input_path),
        "--t-ms",
        str(T_MS),
        "--n-batch",
        str(N_BATCH),
        "--input-rate",
        str(INPUT_RATE_HZ),
        "--seed",
        str(SEED),
        "--outputs",
        "rasters",
        "--out-dir",
        str(sim_dir),
        "--wipe-dir",
    ]
    return stages.command(REPO, provenance, "simulator", cmd)


def make_input(path: Path) -> np.ndarray:
    """Create the exact spike tensor consumed by the simulator."""
    rng = np.random.default_rng(SEED)
    n_steps = round(T_MS / DT_MS)
    p_step = INPUT_RATE_HZ * DT_MS / 1000.0
    spikes = (
        rng.random((n_steps, N_BATCH, N_INPUT), dtype=np.float32) < p_step
    ).astype(np.uint8)
    np.savez_compressed(path, input_spikes=spikes)
    return spikes


def compute(*, run_id=None):
    with stages.execution(REPO, recipe, "compute", run_id=run_id) as run:
        bundle = author_network()
        bundle.write(run.export / "network.bundle", visualise=False)
        make_input(run.export / "input_spikes.npz")
        run_simulator(
            run.export / "network.bundle",
            run.export / "input_spikes.npz",
            run.export / "simulation",
            run.scratch,
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="unused source-neutral reservation")
    args = parser.parse_args()
    try:
        compute(run_id=args.run_id)
    except (PingstoreError, OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
