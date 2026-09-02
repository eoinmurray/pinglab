"""Measure snnsim--Brian2 distances in retained exp111 comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp111 import inputs, measurements, recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def analyse(identity, *, run_id=None):
    source = inputs.source(REPO, identity, "compute")
    document = load_json(source.export / "comparisons.json")
    recorded_recipe = dict(document.get("recipe", {}))
    recorded_recipe.pop("acceptance", None)
    recorded_recipe["schema"] = recipe.configuration()["schema"]
    if (
        document.get("schema") != "exp111.compute/v1"
        or recorded_recipe != recipe.configuration()
        or len(document.get("comparisons", [])) != len(recipe.TESTS)
    ):
        raise PingstoreError("inconsistent exp111 compute evidence")
    analysis = measurements.evaluate(document["comparisons"])
    with inputs.execution(
        REPO, "analyse", sources={"compute": source}, run_id=run_id
    ) as run:
        write_json_atomic(run.export / "analysis.json", analysis)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp111 compute run")
    parser.add_argument("--run-id", help="unused source-neutral v4 reservation")
    arguments = parser.parse_args()
    try:
        print(analyse(arguments.source, run_id=arguments.run_id))
    except (PingstoreError, OSError, KeyError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"exp111 analyse: {exc}\n")


if __name__ == "__main__":
    main()
