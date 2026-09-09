"""Render an explicit analysis and pinned compute source; never publish."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
from experiments.exp099 import inputs, recipe
from experiments.exp099.render import network_diagram, render
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import stage_run
from tools import snnlang as snn  # noqa: TID251
from tools.snnviz import Recording  # noqa: TID251


def resolved(identity):
    analysis = inputs.source(REPO, identity, "analyse")
    cfg = inputs.configuration(analysis)
    refs = analysis.record["inputs"]
    if set(refs) != {"compute"}:
        raise PingstoreError("exp099 analysis must pin one compute input")
    compute = inputs.source(
        REPO, refs["compute"]["run_id"], "compute", reference=refs["compute"]
    )
    results = load_json(analysis.export / "results.json")
    if (
        inputs.configuration(compute) != cfg
        or results.get("parameters") != cfg
        or results.get("schema") != "exp099.analysis/v2"
    ):
        raise PingstoreError("exp099 analysis configuration disagrees with compute")
    return analysis, compute, cfg, results


def render_outputs(
    analysis, compute, cfg, results, output, *, preview_only=False, view=None
):
    with np.load(analysis.export / "measurements.npz", allow_pickle=False) as arrays:
        measurements = dict(arrays)
    with np.load(compute.export / "weights.npz", allow_pickle=False) as arrays:
        weights = dict(arrays)
    render(
        Recording(cfg["dt_ms"], inputs.recording(compute)),
        weights,
        measurements,
        cfg,
        output,
        preview_only=preview_only,
        view=view,
    )
    network_diagram(
        cfg,
        weights,
        output / "network.svg",
        bundle=snn.load_bundle(compute.unit("network.bundle")),
    )
    write_json_atomic(output / "numbers.json", results)


def present(
    identity,
    *,
    run_id=None,
    view_start_ms=recipe.VIEW_START_MS,
    view_end_ms=recipe.VIEW_END_MS,
):
    analysis, compute, cfg, results = resolved(identity)
    view = {"start_ms": view_start_ms, "end_ms": view_end_ms, "frames": 625}
    if not 0 <= view_start_ms < view_end_ms <= cfg["t_ms"]:
        raise ValueError("presentation interval must lie within the recording")
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs={"analysis": analysis, "compute": compute},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        run.record["execution"]["presentation"] = view
        render_outputs(analysis, compute, cfg, results, run.export, view=view)
    return run.run_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    p.add_argument("--view-start-ms", type=float, default=recipe.VIEW_START_MS)
    p.add_argument("--view-end-ms", type=float, default=recipe.VIEW_END_MS)
    a = p.parse_args()
    present(
        a.source,
        run_id=a.run_id,
        view_start_ms=a.view_start_ms,
        view_end_ms=a.view_end_ms,
    )


if __name__ == "__main__":
    main()
