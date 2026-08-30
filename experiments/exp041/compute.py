"""Evaluate an explicit trained bank and retain raw snapshots; never train or publish."""

import argparse
import contextlib
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp041 import evidence, inputs, recipe
from experiments.helpers.run_cli import run_cli
from pingstore.contracts import PingstoreError, write_json_atomic


def compute(identity: str, *, run_id: str | None = None) -> str:
    bank = inputs.source(REPO, identity, "compute", experiment="exp022")
    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    contract = evidence.training_contract(bank.export)
    checkpoint_rows = evidence.checkpoints(bank.export, contract)
    evidence.histories(bank.export, contract)
    with inputs.execution(
        REPO, "compute", sources={"bank": bank}, run_id=run_id, configuration=cfg
    ) as run:
        environment = {"PINGLAB_SMOKE": "1" if cfg["profile"] == "smoke" else "0"}
        run.record["execution"]["environment"] = environment
        write_json_atomic(
            run.export / "evidence.json",
            {
                "schema": "exp041.compute/v1",
                "config": cfg,
                "training_contract": contract,
                "checkpoint_provenance": checkpoint_rows,
            },
        )
        commands = []
        for cell, checkpoint in zip(contract["cells"], checkpoint_rows, strict=True):
            train = bank.unit(cell["cell_name"])
            target = run.scratch / "training" / cell["cell_name"] / "config.json"
            target.parent.mkdir(parents=True)
            shutil.copyfile(train / "config.json", target)
            modes = ["infer"]
            if cell["seed"] == cfg["raster"]["seed"]:
                modes.append("snapshot")
            for mode in modes:
                destination = run.export / mode / cell["cell_name"]
                attachments = run.scratch / "simulations" / mode / cell["cell_name"]
                attachments.mkdir(parents=True)
                args = recipe.inference_args(
                    train,
                    train / checkpoint["filename"],
                    destination,
                    samples=cfg["evaluation_samples"],
                    tau_gaba_ms=cell["tau_gaba_ms"],
                    sample_index=cfg["raster"]["sample_index"]
                    if mode == "snapshot"
                    else None,
                )
                commands.append(
                    {"cell": cell["cell_name"], "mode": mode, "arguments": args}
                )
                write_json_atomic(
                    run.scratch / "simulations.json", {"commands": commands}
                )
                print(f"[{mode}] {cell['cell_name']}", flush=True)
                with (
                    (attachments / "stdout.log").open("w") as stdout,
                    (attachments / "stderr.log").open("w") as stderr,
                ):
                    with (
                        contextlib.redirect_stdout(stdout),
                        contextlib.redirect_stderr(stderr),
                    ):
                        run_cli(args, no_sync=True)
                if mode == "infer":
                    evidence.measurement(
                        destination / "metrics.json",
                        cell,
                        contract["common"],
                        cfg["evaluation_samples"],
                    )
                    evidence.population_traces(
                        destination / "pop_traces.npz",
                        contract["common"],
                        cfg["evaluation_samples"],
                    )
                else:
                    evidence.snapshot(
                        destination / "recording.npz",
                        contract["common"]["dt"],
                        contract["common"],
                    )
                for name in ("config.json", "run.sh", "output.log", "run.jsonl"):
                    attachment = destination / name
                    if attachment.exists():
                        attachment.rename(attachments / name)
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed v4 exp022 compute bank ID"
    )
    parser.add_argument("--run-id", help="unused v4 reservation")
    args = parser.parse_args()
    try:
        compute(args.source, run_id=args.run_id)
    except PingstoreError as exc:
        parser.exit(1, f"exp041 compute: {exc}\n")


if __name__ == "__main__":
    main()
