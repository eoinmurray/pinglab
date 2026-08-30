from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools"), str(REPO / "tools/snnsim")]

import time

import numpy as np
import snnlang as snn
from experiments.exp077 import plots, recipe
from experiments.helpers import snnlang_stages as stages
from pingstore.contracts import PingstoreError, write_json_atomic


def present(identity, *, run_id=None):
    analysis, compute, result = stages.analysis_sources(REPO, recipe, identity)
    started = time.monotonic()
    with stages.execution(
        REPO, recipe, "present", sources={"analysis": analysis}, run_id=run_id
    ) as run:
        recorded = {}
        for name, _ in recipe.VARIANTS:
            with np.load(
                compute.export / "variants" / f"{name}-recordings.npz"
            ) as arrays:
                recorded[name] = {
                    key: arrays[key] for key in ("population_0", "population_2")
                }
        plots.render_rasters(recorded, run.export / "matched_rasters.png")
        snn.load_bundle(
            compute.unit("variants", "reciprocal_delayed.bundle")
        ).visualise(run.export / "reciprocal_delayed.svg", view="circuit")
        write_json_atomic(run.export / "delay_timing.json", result["delay_timing"])
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
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", help="unused source-neutral reservation")
    args = parser.parse_args()
    try:
        present(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
