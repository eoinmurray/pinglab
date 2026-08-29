"""Retain the two raster trials and fourteen f–I trials; never plot or publish."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp023 import recipe
from experiments.helpers.run_cli import run_cli
from pingstore.contracts import PingstoreError, write_json_atomic
from pingstore.stages import stage_run


def compute(*, run_id: str | None = None) -> str:
    smoke = os.environ.get("PINGLAB_SMOKE") == "1"
    cfg = recipe.configuration(smoke=smoke)
    with stage_run(
        REPO, recipe.SLUG, "compute", run_id=run_id, configuration=cfg
    ) as run:
        environment = {"PINGLAB_SMOKE": "1" if smoke else "0"}
        run.record["execution"]["environment"] = environment
        commands = []
        for relative, scientific_args in recipe.simulations(smoke=smoke):
            destination = run.export / relative
            if destination.exists():
                raise PingstoreError(
                    f"simulation destination already exists: {destination}"
                )
            args = [*scientific_args, "--out-dir", str(destination)]
            commands.append({"output": relative, "arguments": args})
            write_json_atomic(
                run.evidence / "simulations.json", {"commands": commands}
            )
            run_cli(args, no_sync=True)
            if not (destination / "snapshot.npz").is_file():
                raise PingstoreError(
                    f"simulation did not produce {relative}/snapshot.npz"
                )
            # Retain execution attachments outside the scientific export.
            for name in ("config.json", "run.sh", "output.log", "run.jsonl"):
                attachment = destination / name
                if attachment.exists():
                    target = run.evidence / "simulations" / relative / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    attachment.rename(target)
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="unused v3 identity reserved before dispatch")
    args = parser.parse_args()
    compute(run_id=args.run_id)


if __name__ == "__main__":
    main()
