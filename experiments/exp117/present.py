"""Render exp117 analysis evidence into flat article presentation files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp117 import inputs, plots
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def present(identity, *, run_id=None):
    analysis = inputs.source(REPO, identity, "analyse")
    cfg = inputs.configuration(analysis)
    if set(analysis.record["inputs"]) != {"compute"}:
        raise PingstoreError("exp117 analysis must pin one compute run")
    results = load_json(analysis.export / "results.json")
    coordinates = load_json(analysis.export / "plot_coordinates.json")
    if results.get("schema") != "exp117.analysis/v3":
        raise PingstoreError("unsupported exp117 analysis payload")
    if coordinates.get("schema") != "exp117.plot-coordinates/v4":
        raise PingstoreError("unsupported exp117 plot coordinates")
    if results.get("configuration") != cfg:
        raise PingstoreError("exp117 analysis payload and recipe disagree")
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": analysis},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        plots.bifurcation_compound(
            coordinates, run.export / "bifurcation_compound.svg"
        )
        write_json_atomic(run.export / "numbers.json", results)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(present(args.source, run_id=args.run_id))


if __name__ == "__main__":
    main()
