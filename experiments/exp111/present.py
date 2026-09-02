"""Render the retained exp111 analysis without rerunning either simulator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp111 import inputs, plots, recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def present(identity, *, run_id=None):
    source = inputs.source(REPO, identity, "analyse")
    document = load_json(source.export / "analysis.json")
    if document.get("schema") != "exp111.analysis/v2" or len(
        document.get("tests", [])
    ) != len(recipe.TESTS):
        raise PingstoreError("incomplete exp111 analysis")
    with inputs.execution(
        REPO, "present", sources={"analysis": source}, run_id=run_id
    ) as run:
        for test in document["tests"]:
            plots.comparison_figure(test, run.export / f"{test['id']}.svg")
        write_json_atomic(run.export / "numbers.json", document)
        if any(not (run.export / name).is_file() for name in recipe.FIGURES):
            raise PingstoreError("incomplete exp111 figure export")
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp111 analyse run")
    parser.add_argument("--run-id", help="unused source-neutral v4 reservation")
    arguments = parser.parse_args()
    try:
        print(present(arguments.source, run_id=arguments.run_id))
    except (PingstoreError, OSError, KeyError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"exp111 present: {exc}\n")


if __name__ == "__main__":
    main()
