"""Render saved analysis only; no simulation, measurement or publication."""

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp044 import inputs, plots
from experiments.helpers import theme
from experiments.helpers.fmt import format_duration
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def present(identity: str, *, run_id: str | None = None) -> str:
    analysis = inputs.source(REPO, identity, "analyse")
    if set(analysis.record["inputs"]) != {"compute", "bank"}:
        raise PingstoreError(
            "exp044 analysis must pin its computation and training bank"
        )
    ref = analysis.record["inputs"]["compute"]
    compute = inputs.source(REPO, ref["run_id"], "compute", reference=ref)
    cfg = inputs.configuration(compute)
    if compute.record["inputs"]["bank"] != analysis.record["inputs"]["bank"]:
        raise PingstoreError("analysis and compute disagree on the training bank")
    result = load_json(analysis.export / "results.json")
    if (
        result.get("schema") != "exp044.analysis/v1"
        or result.get("recipe") != cfg
        or result.get("measurement") != analysis.record["execution"]["configuration"]
    ):
        raise PingstoreError("unsupported or inconsistent exp044 analysis payload")
    samples = []
    with np.load(analysis.export / "rasters.npz", allow_pickle=False) as data:
        for row in result["rasters"]:
            samples.append(
                {
                    **row,
                    "e": np.array(data[row["cell_name"] + "__e"]),
                    "i": np.array(data[row["cell_name"] + "__i"]),
                }
            )
    started = time.monotonic()
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": analysis},
        run_id=run_id,
        configuration={
            "schema": "exp044.presentation/v1",
            "paper_mode": True,
            "raster_window_ms": cfg["raster"]["window_ms"],
        },
    ) as run:
        theme.set_paper_mode(True)
        plots.plot_dt_sweep(result["aggregate"], run.export / "dt_sweep")
        plots.plot_raster_strip(
            samples, run.export / "raster_strip", cfg["raster"]["window_ms"]
        )
        plots.plot_training_curves(
            result["curves"],
            cfg["dt_sweep_ms"],
            cfg["seeds"],
            run.export / "training_curves",
        )
        run.record["presentation_lineage"] = {
            "analysis": analysis.reference,
            "operation": "render saved measurements and selected raster samples",
            "historical_training_curves": analysis.record["inputs"]["bank"],
            "new_inference": analysis.record["inputs"]["compute"],
        }
        write_json_atomic(
            run.export / "numbers.json",
            {
                **result,
                "run_id": run.run_id,
                "duration_s": time.monotonic() - started,
                "duration": format_duration(time.monotonic() - started),
                "git_sha": run.record["provenance"]["git_commit"],
            },
        )
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp044 analyse ID")
    parser.add_argument("--run-id", help="unused v4 reservation")
    args = parser.parse_args()
    try:
        present(args.source, run_id=args.run_id)
    except PingstoreError as exc:
        parser.exit(1, f"exp044 present: {exc}\n")


if __name__ == "__main__":
    main()
