from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools"), str(REPO / "tools/snnsim")]

from experiments.exp075 import recipe
from experiments.exp075.recipe import BATCH_SIZE, MAX_SAMPLES, SEED, T_MS, author_bundle
from experiments.helpers import snnlang_stages as stages
from pingstore.contracts import PingstoreError


def run_training(bundle_dir: Path, out_dir: Path, provenance: Path) -> dict:
    cmd = [
        sys.executable,
        str(REPO / "tools/snnsim/tool.py"),
        "train",
        "--bundle",
        str(bundle_dir),
        "--max-samples",
        str(MAX_SAMPLES),
        "--batch-size",
        str(BATCH_SIZE),
        "--t-ms",
        str(T_MS),
        "--seed",
        str(SEED),
        "--out-dir",
        str(out_dir),
        "--wipe-dir",
    ]
    return stages.command(REPO, provenance, "simulator", cmd)


def compute(*, run_id=None):
    with stages.execution(REPO, recipe, "compute", run_id=run_id) as run:
        bundle = author_bundle()
        bundle.write(run.export / "network.bundle", visualise=False)
        run_training(
            run.export / "network.bundle", run.export / "training", run.provenance
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
