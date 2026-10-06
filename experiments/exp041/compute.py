"""Evaluate an explicit trained bank and retain raw snapshots; never train or publish."""

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp041 import evidence, inputs, recipe
from experiments.helpers.checkpoint_inference import run_inference
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
        for cell, checkpoint in zip(contract["cells"], checkpoint_rows, strict=True):
            train = bank.unit(cell["cell_name"])
            training = {
                **contract["common"],
                "seed": cell["seed"],
                "tau_gaba_ms": cell["tau_gaba_ms"],
            }
            modes = ["infer"] + (
                ["snapshot"] if cell["seed"] == cfg["raster"]["seed"] else []
            )
            for mode in modes:
                destination = run.export / mode / cell["cell_name"]
                run_inference(
                    train,
                    destination,
                    run.scratch / "simulations" / mode / cell["cell_name"],
                    training,
                    recipe.inference_request(
                        training, cfg, snapshot=mode == "snapshot"
                    ),
                    recipe.author_network,
                )
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
