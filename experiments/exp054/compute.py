"""Retain full untrained-network rasters and mean-field solutions; never analyse or plot."""

import argparse
import contextlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp033 import compute as numerical
from experiments.exp033 import measurements as numerical_validation
from experiments.exp054 import evidence, inputs, recipe
from experiments.helpers.run_cli import run_cli
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def mean_field(cfg):
    """Reuse numerical functions, never dispatch an exp033 stage."""
    mf = cfg["mean_field"]
    grid = np.linspace(*mf["drive_grid"])
    sigma = mf["sigma_V_mV"]
    reference = numerical.continuation(grid, sigma=sigma)
    if reference["hopf"] is None:
        raise PingstoreError("exp054 reference onset is unavailable")
    reference["ramp"] = numerical.ramp(reference["hopf"], sigma)
    result = {
        "schema": "exp054.mean-field/v1",
        "reference": reference,
        "frequency": [
            {"tau_gaba_ms": tau, **numerical.continuation(grid, sigma=sigma, tau=tau)}
            for tau in mf["tau_grid_ms"]
        ],
    }
    for row in (reference, *result["frequency"]):
        numerical_validation.validate_continuation(row, grid)
        if len(row["sweep"]) != len(grid):
            raise PingstoreError("incomplete exp054 numerical continuation")
    expected_drives = np.linspace(
        reference["hopf"]["I_ext_star"] - 0.1,
        reference["hopf"]["I_ext_star"] + 0.55,
        25,
    ).tolist()
    for direction in ("up", "down"):
        branch = reference["ramp"][direction]
        if [r["I_ext"] for r in branch] != expected_drives:
            raise PingstoreError("incomplete exp054 numerical ramp")
        for row in branch:
            numerical_validation.series(row, 4, end=mf["hysteresis"]["t_max_ms"])
    return result


def compute(*, run_id=None):
    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    with inputs.execution(
        REPO, "compute", sources={}, run_id=run_id, configuration=cfg
    ) as run:
        smoke = "1" if cfg["profile"] == "smoke" else "0"
        run.record["execution"]["environment"] = {"PINGLAB_SMOKE": smoke}
        write_json_atomic(run.provenance / "command.json", run.record["execution"])
        replay = run.provenance / "run.sh"
        replay.write_text(
            replay.read_text().replace(
                "\nexec ", f"\nexport PINGLAB_SMOKE={smoke}\nexec ", 1
            )
        )
        for item in recipe.jobs(cfg):
            attachments = run.provenance / "simulations" / item["id"]
            attachments.mkdir(parents=True)
            args = recipe.simulation_args(cfg, item, attachments)
            write_json_atomic(
                attachments / "command.json", {"job": item, "arguments": args}
            )
            with (
                attachments.joinpath("stdout.log").open("w") as stdout,
                attachments.joinpath("stderr.log").open("w") as stderr,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                run_cli(args, no_sync=True)
            evidence.simulation_config(
                load_json(attachments / "config.json"), cfg, item
            )
            original = attachments / "rasters.npz"
            evidence.raster(original, cfg)
            destination = run.export / "probe" / item["id"]
            destination.mkdir(parents=True)
            original.rename(destination / "rasters.npz")
        evidence.write(run.export, mean_field(cfg))
        write_json_atomic(
            run.export / "recordings.json",
            {
                "schema": "exp054.recordings/v1",
                "recipe": cfg,
                "jobs": recipe.jobs(cfg),
            },
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="fresh v3 identity reserved before dispatch")
    args = parser.parse_args()
    compute(run_id=args.run_id)


if __name__ == "__main__":
    main()
