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
from tools import snnlang as snn  # noqa: TID251
from tools import snnviz  # noqa: TID251


def _resolved(identity: str):
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
    return analysis, compute, cfg, results


def _render_condition(analysis, compute, cfg, results, output: Path) -> None:
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
        output,
        configuration=cfg,
    )


def _render_network_diagram(compute, output: Path) -> None:
    """Render the authenticated authored graph through the shared visual layer."""

    bundle = snn.load_bundle(compute.export / "network.bundle")
    visual = snn.diagram(bundle, view="expanded")
    snnviz.render_diagram(visual, output / "network.svg")


def present(identity: str, *, run_id: str | None = None) -> str:
    analysis, compute, cfg, results = _resolved(identity)
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs={"analysis": analysis, "compute": compute},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        _render_condition(analysis, compute, cfg, results, run.export)
        _render_network_diagram(compute, run.export)
        shutil.copy2(
            Path(__file__).with_name(recipe.INPUT_MAP), run.export / recipe.INPUT_MAP
        )
        write_json_atomic(run.export / "numbers.json", results)
    return run.run_id


def present_pair(
    identity: str, comparison_identity: str, *, run_id: str | None = None
) -> str:
    """Render the richer-input and isolated shared-drive media together."""
    primary = _resolved(identity)
    comparison = _resolved(comparison_identity)
    by_condition = {
        row[2].get("condition", "richer-input"): row for row in (primary, comparison)
    }
    if set(by_condition) != {"richer-input", "shared-drive-isolation"}:
        raise PingstoreError(
            "paired exp099 presentation requires richer-input and shared-drive-isolation"
        )
    richer = by_condition["richer-input"]
    shared = by_condition["shared-drive-isolation"]
    presentation_cfg = {
        "schema": "exp099.presentation/v1",
        "conditions": {
            "richer-input": richer[2],
            "shared-drive-isolation": shared[2],
        },
    }
    stage_inputs = {
        "richer_analysis": richer[0],
        "richer_compute": richer[1],
        "shared_analysis": shared[0],
        "shared_compute": shared[1],
    }
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs=stage_inputs,
        run_id=run_id,
        configuration=presentation_cfg,
    ) as run:
        for analysis, compute, cfg, results in (richer, shared):
            _render_condition(analysis, compute, cfg, results, run.export)
        _render_network_diagram(shared[1], run.export)
        shutil.copy2(
            Path(__file__).with_name(recipe.INPUT_MAP), run.export / recipe.INPUT_MAP
        )
        write_json_atomic(
            run.export / "numbers.json",
            {
                "schema": "exp099.presentation-results/v1",
                "conditions": {
                    "richer-input": richer[3],
                    "shared-drive-isolation": shared[3],
                },
            },
        )
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed exp099 v4 analyse run ID"
    )
    parser.add_argument(
        "--comparison-source",
        help="second completed analysis to package both EXP099 videos",
    )
    parser.add_argument("--run-id", help="unused v4 identity reserved before dispatch")
    args = parser.parse_args()
    if args.comparison_source:
        present_pair(args.source, args.comparison_source, run_id=args.run_id)
    else:
        present(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
