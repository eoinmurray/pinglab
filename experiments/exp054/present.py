"""Draw saved analysis coordinates; never simulate, remeasure or publish."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp054 import evidence, inputs, plots, recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def analysis_source(repo, identity, reference=None):
    analysis = inputs.source(repo, identity, "analyse", reference=reference)
    cfg = inputs.configuration(analysis)
    if set(analysis.record["inputs"]) != {"compute", "frequencies"}:
        raise PingstoreError("exp054 analysis must pin compute and exp041 frequencies")
    for role, experiment, stage in (
        ("compute", "exp054", "compute"),
        ("frequencies", "exp041", "analyse"),
    ):
        ref = analysis.record["inputs"][role]
        upstream = inputs.source(
            repo, ref["run_id"], stage, experiment=experiment, reference=ref
        )
        if role == "compute" and evidence.compute_contract(upstream) != cfg:
            raise PingstoreError("exp054 analysis recipe differs from computation")
    coords = evidence.read(analysis.export)
    if coords.get("schema") != "exp054.analysis/v1" or coords.get("recipe") != cfg:
        raise PingstoreError("exp054 analysis coordinates have an inconsistent recipe")
    return analysis, cfg, coords, load_json(analysis.export / "results.json")


def present(identity, *, run_id=None):
    analysis, cfg, coords, numbers = analysis_source(REPO, identity)
    with (
        inputs.execution(
            REPO,
            "present",
            sources={"analysis": analysis},
            run_id=run_id,
            configuration=cfg,
        ) as run,
        plots.configured(cfg),
    ):
        grid, private, shared = (
            coords["grid"],
            coords["private_null"],
            coords["shared_null"],
        )
        for function, name in (
            (plots.fig_turnon_maps_compound, "turnon_maps_compound.png"),
            (plots.fig_turnon_compound, "turnon_compound.png"),
            (plots.fig_grid_maps_compound, "grid_maps.png"),
            (plots.fig_grid_rasters, "grid_rasters.png"),
            (plots.fig_grid_autocorr, "grid_autocorr.png"),
        ):
            function(grid, run.export / name)
        plots.fig_rate_invariance(
            grid, private, shared, run.export / "rate_invariance.png"
        )
        plots.fig_null_autocorr(shared, private, run.export / "null_autocorr.png")
        if not all((run.export / name).is_file() for name in recipe.FIGURES):
            raise PingstoreError("incomplete exp054 presentation")
        write_json_atomic(
            run.export / "numbers.json", {**numbers, "run_id": run.run_id}
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp054 analysis run")
    parser.add_argument("--run-id", help="fresh v4 identity reserved before dispatch")
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
