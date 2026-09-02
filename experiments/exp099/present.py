"""Render validated analysis and its pinned compute source; never publish."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "tools")]

import shutil

import numpy as np
from experiments.exp099 import inputs, recipe
from experiments.exp099.render import render
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import stage_run


def present(identity: str, *, run_id: str | None = None) -> str:
    analysis = inputs.source(REPO, identity, "analyse")
    cfg = inputs.configuration(analysis)
    refs = analysis.record["inputs"]
    if set(refs) != {"compute"}:
        raise PingstoreError("exp099 analysis must pin exactly one compute input")
    compute = inputs.source(
        REPO, refs["compute"]["run_id"], "compute", reference=refs["compute"]
    )
    if inputs.configuration(compute) != cfg or compute.record["inputs"]:
        raise PingstoreError("analysis recipe or compute lineage disagrees with source")
    results = load_json(analysis.export / "results.json")
    if (
        results.get("schema") != "exp099.analysis/v1"
        or results.get("parameters") != cfg
        or results.get("measurements", {}).get("schema") != "exp099.measurements/v1"
        or results.get("measurements")
        != analysis.record["execution"].get("measurements")
    ):
        raise PingstoreError("unsupported or inconsistent exp099 analysis payload")
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs={"analysis": analysis, "compute": compute},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        with np.load(analysis.export / "measurements.npz", allow_pickle=False) as data:
            measurements = dict(data)
        with np.load(
            compute.export / "simulation/recurrent-weights.npz", allow_pickle=False
        ) as data:
            weights = dict(data)
        render(
            inputs.recording(compute),
            weights,
            measurements,
            results["measurements"],
            run.export,
            configuration=cfg,
        )
        shutil.copy2(compute.export / "network.svg", run.export / "network.svg")
        shutil.copy2(
            Path(__file__).with_name(recipe.INPUT_MAP), run.export / recipe.INPUT_MAP
        )
        write_json_atomic(run.export / "numbers.json", results)
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed exp099 v4 analyse run ID"
    )
    parser.add_argument("--run-id", help="unused v4 identity reserved before dispatch")
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
