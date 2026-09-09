"""Draw saved analysis coordinates; never simulate, remeasure or publish."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp054 import evidence, inputs, plots, recipe
from experiments.exp054 import theory as theory_inputs
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def analysis_source(repo, identity, reference=None):
    analysis = inputs.source(repo, identity, "analyse", reference=reference)
    cfg = inputs.configuration(analysis)
    refreshed = cfg["schema"] == "exp054.theory-refresh/v1"
    expected = {"compute", "frequencies"} | ({"theory"} if refreshed else set())
    if set(analysis.record["inputs"]) != expected:
        raise PingstoreError("exp054 analysis must pin compute and exp041 frequencies")
    upstreams = {}
    for role, experiment, stage in (
        ("compute", "exp054", "compute"),
        ("frequencies", "exp041", "analyse"),
    ):
        ref = analysis.record["inputs"][role]
        upstream = inputs.source(
            repo, ref["run_id"], stage, experiment=experiment, reference=ref
        )
        upstreams[role] = upstream
        if role == "compute" and evidence.compute_contract(
            upstream
        ) != recipe.spike_configuration(cfg):
            raise PingstoreError("exp054 analysis recipe differs from computation")
    coords = evidence.read(analysis.export)
    if coords.get("schema") != "exp054.analysis/v1" or coords.get("recipe") != cfg:
        raise PingstoreError("exp054 analysis coordinates have an inconsistent recipe")
    numbers = load_json(analysis.export / "results.json")
    if refreshed:
        ref = analysis.record["inputs"]["theory"]
        _, theory_cfg, mf = theory_inputs.source(
            repo, ref["run_id"], upstreams["frequencies"], reference=ref
        )
        if theory_cfg != cfg[
            "theory_recipe"
        ] or not evidence.mean_field_evidence.exact_values(coords["mean_field"], mf):
            raise PingstoreError(
                "exp054 analysis differs from its explicit theory source"
            )
        if numbers["mean_field"] != {k: v for k, v in mf.items() if k != "sweep"}:
            raise PingstoreError("exp054 theory numbers differ from their source")
    return analysis, cfg, coords, numbers


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
        plots.configured(recipe.spike_configuration(cfg)),
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
