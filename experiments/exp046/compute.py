"""Record complete inference evidence from an explicit bank; never analyse or publish."""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp046 import evidence, inputs, recipe
from experiments.helpers.checkpoint_inference import run_inference
from pingstore.contracts import PingstoreError, write_json_atomic


def compute(identity, *, run_id=None):
    bank = inputs.source(REPO, identity, "compute", experiment="exp022")
    cfg = recipe.configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1")
    contract = evidence.training_contract(bank.export)
    checkpoints = evidence.checkpoints(bank.export, contract)
    with inputs.execution(
        REPO, "compute", sources={"bank": bank}, run_id=run_id, configuration=cfg
    ) as run:
        smoke = "1" if cfg["profile"] == "smoke" else "0"
        run.record["execution"]["environment"] = {"PINGLAB_SMOKE": smoke}
        write_json_atomic(
            run.export / "evidence.json",
            {
                "schema": "exp046.compute/v1",
                "config": cfg,
                "training_contract": contract,
                "checkpoint_provenance": checkpoints,
            },
        )
        for cell, checkpoint in zip(contract["cells"], checkpoints, strict=True):
            training = {
                **contract["common"],
                "seed": cell["seed"],
                "tau_gaba_ms": cell["tau_gaba_ms"],
            }
            output = run.export / "infer" / cell["cell_name"]
            run_inference(
                bank.unit(cell["cell_name"]),
                output,
                run.scratch / "simulations" / cell["cell_name"],
                training,
                recipe.inference_request(training, cfg),
                recipe.author_network,
            )
            evidence.measurement(
                output / "metrics.json",
                cell,
                contract["common"],
                cfg["evaluation_samples"],
            )
            evidence.recordings(output, contract["common"], cfg["evaluation_samples"])
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp022 compute bank")
    parser.add_argument("--run-id", help="unused source-neutral reservation")
    args = parser.parse_args()
    try:
        compute(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"exp046 compute: {exc}\n")


if __name__ == "__main__":
    main()
