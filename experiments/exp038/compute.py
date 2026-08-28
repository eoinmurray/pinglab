"""Compute exp038 probes from an explicit bank; never analyse or publish."""

import argparse
import contextlib
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp038 import evidence, inputs, recipe
from experiments.helpers.run_cli import run_cli
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def compute(identity, *, run_id=None):
    bank = inputs.source(REPO, identity, "compute", experiment="exp022")
    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    contract = evidence.training_contract(bank.export)
    evidence.histories(bank.export, contract)
    with inputs.execution(
        REPO, "compute", sources={"bank": bank}, run_id=run_id, configuration=cfg
    ) as run:
        env = {"PINGLAB_SMOKE": "1" if cfg["profile"] == "smoke" else "0"}
        run.record["execution"]["environment"] = env
        write_json_atomic(run.provenance / "command.json", run.record["execution"])
        replay = run.provenance / "run.sh"
        replay.write_text(
            replay.read_text().replace(
                "\nexec ", f"\nexport PINGLAB_SMOKE={env['PINGLAB_SMOKE']}\nexec "
            )
        )
        write_json_atomic(
            run.export / "evidence.json",
            {
                "schema": "exp038.compute/v1",
                "recipe": cfg,
                "training_contract": contract,
                "jobs": recipe.jobs(cfg),
            },
        )
        commands = []
        for job in recipe.jobs(cfg):
            name = job["cell_name"]
            train = bank.export / name
            output = run.export / job["path"]
            attachments = run.provenance / "simulations" / job["path"]
            attachments.mkdir(parents=True)
            shutil.copyfile(train / "config.json", attachments / "training-config.json")
            args = recipe.inference_args(train, train / "weights.pth", output, job)
            commands.append({"job": job, "arguments": args})
            write_json_atomic(run.provenance / "simulations.json", commands)
            print(f"[infer] {job['path']}", flush=True)
            with (
                (attachments / "stdout.log").open("w") as stdout,
                (attachments / "stderr.log").open("w") as stderr,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                run_cli(args, no_sync=True)
            evidence.inference_config(
                load_json(output / "config.json"), contract["configs"][name], job
            )
            evidence.recordings(output, contract["configs"][name], job)
            keep = "snapshot.npz" if "sample_index" in job else "metrics.json"
            for path in output.iterdir():
                if path.name != keep:
                    path.rename(attachments / path.name)
    return run.run_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    a = p.parse_args()
    try:
        compute(a.source, run_id=a.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        p.exit(1, f"exp038 compute: {exc}\n")


if __name__ == "__main__":
    main()
