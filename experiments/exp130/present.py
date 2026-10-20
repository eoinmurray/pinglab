"""Render Figures 1–3 from explicit paired compute and analysis evidence.

The CLI requires explicit completed analysis evidence. Figures and browser
replays are written to a new presentation run; upstream execution is never
launched automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
from experiments.exp130 import recipe
from experiments.exp130.custom import inputs
from experiments.exp130.custom.browser import circuit_data
from experiments.exp130.custom.render import network_diagram, render_raster
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import stage_run
from snnlab import lang


def present(identity):
    analysis = inputs.source(REPO, identity, "analyse")
    cfg = inputs.configuration(analysis)
    refs = analysis.record["inputs"]
    if set(refs) != {"compute"}:
        raise PingstoreError("analysis must pin exactly one paired compute")
    compute = inputs.source(
        REPO, refs["compute"]["run_id"], "compute", reference=refs["compute"]
    )
    results = load_json(analysis.export / "results.json")
    if (
        inputs.configuration(compute) != cfg
        or results.get("parameters") != cfg
        or results.get("schema") != "exp130.analysis/v1"
    ):
        raise PingstoreError("compute and analysis disagree")
    recordings = {
        condition: inputs.recording(compute, condition)
        for condition in recipe.CONDITIONS
    }
    with np.load(analysis.export / "measurements.npz", allow_pickle=False) as raw:
        measurements = {
            condition: {
                key.split("__", 1)[1]: raw[key]
                for key in raw.files
                if key.startswith(condition + "__")
            }
            for condition in recipe.CONDITIONS
        }
    view = dict(start_ms=cfg["view_start_ms"], end_ms=cfg["view_end_ms"])
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs={"analysis": analysis, "compute": compute},
        configuration=cfg,
    ) as run:
        replays = {}
        for condition in recipe.CONDITIONS:
            with np.load(
                compute.unit(condition) / "weights.npz", allow_pickle=False
            ) as raw:
                weights = dict(raw)
            replays[condition] = circuit_data(
                recordings[condition],
                weights,
                cfg,
                view,
                measurements=measurements[condition],
            )
        shared_max = max(
            data["piston_scales"]["conductance_us"][1] for data in replays.values()
        )
        for condition, data in replays.items():
            data["piston_scales"]["conductance_us"][1] = shared_max
            data["condition"] = condition
        (run.export / "circuit.json").write_text(
            json.dumps(
                {"schema": "exp130.comparison/v1", "conditions": replays},
                separators=(",", ":"),
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        bundle_path = (
            compute.restore_bundle(
                "theta/network.bundle", run.scratch / "network.bundle"
            )
            if compute.record.get("scientific_artifacts", {}).get("native_paths")
            else compute.unit("theta", "network.bundle")
        )
        run.record["execution"]["network_diagram"] = network_diagram(
            run.export / "network.svg",
            lang.load_bundle(bundle_path),
        )
        render_raster(recordings, cfg, run.export / "spike-raster.png")
        write_json_atomic(run.export / "numbers.json", results)
        run.record["execution"]["presentation"] = dict(
            figures="fresh structural diagram, selectable recorded circuit replay, paired all-neuron raster",
            view=view,
            shared_conductance_scale_us=[0, shared_max],
            raster_interval_ms=[cfg["view_start_ms"], cfg["view_end_ms"]],
            raster_panels=["A", "B", "D", "E", "F", "H"],
            raster_counts="1 ms population counts over the complete recorded interval; no smoothing",
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    print(present(args.source))


if __name__ == "__main__":
    main()
