"""Render exactly three final figures from an explicit exp114 analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp114 import inputs, plots
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import stage_run
from tools import snnlang as snn  # noqa: TID251


def present(identity, *, run_id=None):
    analysis = inputs.source(REPO, identity, "analyse")
    refs = analysis.record["inputs"]
    if set(refs) != {"compute"}:
        raise PingstoreError("exp114 analysis must pin one compute source")
    compute = inputs.source(REPO, refs["compute"]["run_id"], "compute", reference=refs["compute"])
    result = load_json(analysis.export / "results.json")
    cfg = result["recipe"]
    with np.load(analysis.export / "phase_traces.npz", allow_pickle=False) as saved:
        traces = {name: np.array(saved[name]) for name in saved.files}
    with stage_run(REPO, "exp114", "present", inputs={"analysis": analysis, "compute": compute}, run_id=run_id, configuration={"schema": "exp114.presentation/v2", "final_figure_count": 3, "width_mm": 180, "completion_status": "complete" if result["success"]["overall"] else "incomplete"}) as run:
        bundle = snn.load_bundle(compute.unit("network.bundle"))
        plots.network(bundle, cfg, run.export / "figure1-network.svg")
        plots.locking_map(result["aggregate"], cfg, run.export / "figure2-locking.png")
        plots.phase_and_examples(result["aggregate"], traces, cfg, run.export / "figure3-phase.png")
        write_json_atomic(run.export / "numbers.json", result)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
