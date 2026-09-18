"""Retain full untrained-network rasters; never analyse or plot."""

import argparse
import contextlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp054 import evidence, inputs, recipe
from experiments.helpers.run_cli import run_cli
from pingstore.contracts import load_json, write_json_atomic


def compute(*, run_id=None):
    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    with inputs.execution(
        REPO, "compute", sources={}, run_id=run_id, configuration=cfg
    ) as run:
        smoke = "1" if cfg["profile"] == "smoke" else "0"
        run.record["execution"]["environment"] = {"PINGLAB_SMOKE": smoke}
        for item in recipe.jobs(cfg):
            attachments = run.scratch / "simulations" / item["id"]
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
    parser.add_argument("--run-id", help="fresh v4 identity reserved before dispatch")
    args = parser.parse_args()
    compute(run_id=args.run_id)


if __name__ == "__main__":
    main()
