"""Render the retained compound figure from analysis only; never launch upstream work."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp042 import inputs, plots, recipe
from experiments.helpers import theme
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def present(identity, *, run_id=None):
    analysis = inputs.source(REPO, identity, "analyse")
    if set(analysis.record["inputs"]) != {"compute", "bank"}:
        raise PingstoreError("exp042 analysis must pin compute and bank")
    ref = analysis.record["inputs"]["compute"]
    compute = inputs.source(REPO, ref["run_id"], "compute", reference=ref)
    cfg = inputs.configuration(compute)
    data = load_json(analysis.export / "results.json")
    if (
        compute.record["inputs"]["bank"] != analysis.record["inputs"]["bank"]
        or data.get("schema") != "exp042.analysis/v4"
        or data.get("recipe") != cfg
        or data.get("measurement") != analysis.record["execution"]["configuration"]
    ):
        raise PingstoreError("inconsistent exp042 analysis lineage or payload")
    samples = {}
    with np.load(analysis.export / "rasters.npz", allow_pickle=False) as arrays:
        for name in ("cycle", "cell"):
            samples[name] = {
                **data["rasters"][name],
                "e": arrays[name + "__e"],
                "i": arrays[name + "__i"],
            }
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": analysis},
        run_id=run_id,
        configuration={"schema": "exp042.presentation/v3", "paper_mode": True},
    ) as run:
        theme.set_paper_mode(True)
        cyc, cell = (
            data["aggregate"]["jitter_sweep"],
            data["aggregate"]["cell_jitter_sweep"],
        )
        plots.fig_rhythm_compound(
            cyc, cell, samples["cycle"], samples["cell"], run.export / "rhythm_compound"
        )
        write_json_atomic(run.export / "numbers.json", {**data, "run_id": run.run_id})
        run.record["presentation_lineage"] = {
            "analysis": analysis.reference,
            "operation": "render saved summaries and illustrative spikes",
        }
        if not all((run.export / filename).is_file() for filename in recipe.FIGURES):
            raise PingstoreError("incomplete exp042 presentation")
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp042 analyse ID")
    parser.add_argument("--run-id", help="unused v4 reservation")
    args = parser.parse_args()
    try:
        present(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, ValueError) as exc:
        parser.exit(1, f"exp042 present: {exc}\n")


if __name__ == "__main__":
    main()
