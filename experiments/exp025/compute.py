"""Run explicit exp025 inference from an exp022 bank; never analyse or publish."""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp025 import evidence, inputs, recipe
from experiments.helpers.checkpoint_inference import run_inference
from pingstore.contracts import PingstoreError


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
        for job in recipe.jobs(cfg):
            name = job["cell_name"]
            output = run.export / job["path"]
            training = contract["configs"][name]
            run_inference(
                bank.unit(name),
                output,
                run.scratch / "simulations" / job["path"],
                training,
                recipe.inference_request(training, job),
                recipe.author_network,
            )
            evidence.recordings(output, training, job)
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
