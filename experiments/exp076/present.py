from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools"), str(REPO / "tools/snnsim")]

import time

from experiments.exp076 import plots, recipe
from experiments.helpers import snnlang_stages as stages
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def present(identity, *, run_id=None):
    analysis, compute, result = stages.analysis_sources(REPO, recipe, identity)
    started = time.monotonic()
    with stages.execution(
        REPO, recipe, "present", sources={"analysis": analysis}, run_id=run_id
    ) as run:
        plots.plot_training(
            load_json(compute.export / "bundle_training/metrics.json"),
            run.export / "training_curves.png",
        )
        plots.write_lifecycle_svg(run.export / "lifecycle.svg")
        write_json_atomic(
            run.export / "numbers.json",
            {
                **result,
                "run_id": run.run_id,
                "git_sha": run.record["provenance"]["git_commit"],
                "duration_s": time.monotonic() - started,
            },
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="explicit completed input run")
    parser.add_argument("--run-id", help="unused source-neutral reservation")
    args = parser.parse_args()
    try:
        present(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
