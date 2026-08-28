"""Run explicit exp025 inference from an exp022 bank; never analyse or publish."""

import argparse
import contextlib
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp025 import evidence, inputs, recipe
from experiments.exp041.import_gold2 import normalized_metrics
from experiments.helpers.run_cli import run_cli
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def payload_arrays(job):
    if job["kind"] == "snapshot":
        return {"snapshot.npz": ("dt", "spk_e", "spk_i")}
    if job["kind"] == "scale":
        return {"per_cell_rates.npz": ("rate_e_per_sample",)}
    if job["kind"] == "pfg" and job["is_ping"]:
        return {
            "pop_traces.npz": ("dt", "pop_e"),
            "rasters.npz": (
                "dt",
                "T",
                "n_trials",
                "n_e",
                "n_i",
                "e_trial",
                "e_t",
                "e_cell",
                "i_trial",
                "i_t",
                "i_cell",
            ),
        }
    return {}


def compute(identity, *, run_id=None):
    bank = inputs.source(REPO, identity, "compute", experiment="exp022")
    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    contract = evidence.training_contract(bank.export)
    evidence.histories(bank.export, contract)
    with inputs.execution(
        REPO, "compute", sources={"bank": bank}, run_id=run_id, configuration=cfg
    ) as run:
        run.record["execution"]["environment"] = {
            "PINGLAB_SMOKE": "1" if cfg["profile"] == "smoke" else "0"
        }
        write_json_atomic(run.provenance / "command.json", run.record["execution"])
        replay = run.provenance / "run.sh"
        replay.write_text(
            replay.read_text().replace(
                "\nexec ",
                f"\nexport PINGLAB_SMOKE={run.record['execution']['environment']['PINGLAB_SMOKE']}\nexec ",
            )
        )
        write_json_atomic(
            run.export / "evidence.json",
            {
                "schema": "exp025.compute/v1",
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
            args = recipe.inference_args(
                train, train / "weights_final.pth", output, job
            )
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
            simulation_config = load_json(output / "config.json")
            evidence.inference_config(simulation_config, contract["configs"][name], job)
            if job["kind"] != "snapshot":
                shutil.copyfile(
                    output / "metrics.json", attachments / "metrics.original.json"
                )
                write_json_atomic(
                    output / "metrics.json",
                    normalized_metrics(
                        load_json(output / "metrics.json"), simulation_config
                    ),
                )
            evidence.recordings(output, contract["configs"][name], job)
            for filename in ("config.json", "run.sh", "run.jsonl", "output.log"):
                if (output / filename).exists():
                    (output / filename).rename(attachments / filename)
    return run.run_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    a = p.parse_args()
    try:
        compute(a.source, run_id=a.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        p.exit(1, f"exp025 compute: {exc}\n")


if __name__ == "__main__":
    main()
