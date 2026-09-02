"""Run the twenty independent snnsim--Brian2 comparison protocols."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp111 import inputs, recipe, simulations
from pingstore.contracts import PingstoreError, write_json_atomic


def compute(identity, *, run_id=None):
    bank = inputs.source(REPO, identity, "compute", experiment=recipe.SOURCE_EXPERIMENT)
    with inputs.execution(
        REPO, "compute", sources={"training_bank": bank}, run_id=run_id
    ) as run:
        comparisons = simulations.run_suite(bank.export)
        write_json_atomic(
            run.export / "comparisons.json",
            {
                "schema": "exp111.compute/v1",
                "recipe": recipe.configuration(),
                "comparisons": comparisons,
            },
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp022 compute run")
    parser.add_argument("--run-id", help="unused source-neutral v4 reservation")
    arguments = parser.parse_args()
    try:
        print(compute(arguments.source, run_id=arguments.run_id))
    except (PingstoreError, OSError, KeyError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"exp111 compute: {exc}\n")


if __name__ == "__main__":
    main()
