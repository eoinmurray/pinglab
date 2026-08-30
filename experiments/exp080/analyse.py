"""Aggregate explicit retained correctness; never simulate, train or publish."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

from experiments.exp080 import evidence, inputs, measurements, recipe
from pingstore.contracts import write_json_atomic


def analyse(identity: str, *, run_id: str | None = None) -> str:
    compute = inputs.source(REPO, identity, "compute")
    cfg = inputs.configuration(compute)
    with inputs.execution(
        REPO, "analyse", sources={"compute": compute}, run_id=run_id, configuration=cfg
    ) as run:
        historical = compute.record["execution"].get("operation") == "historical-import"
        if historical:
            evidence.require(
                bool(compute.record.get("historical_import", {}).get("producer")),
                "historical import lacks original producer identity",
            )
        document, correctness = evidence.validate(
            compute.export, cfg, historical=historical
        )
        decision = measurements.analyze(correctness, cfg)
        result = {
            "schema": "exp080.analysis/v1",
            "recipe": cfg,
            "status": "complete",
            "purpose": "empirically select the later variable-rate PING input range",
            "parameters": recipe.reported_parameters(cfg),
            "decision": decision,
            **{
                key: document[key]
                for key in (
                    "training_dataset",
                    "training",
                    "evaluation",
                    "simulator_validation",
                    "runtime_s",
                    "environment",
                    "illustration",
                )
            },
        }
        write_json_atomic(run.export / "results.json", result)
        write_json_atomic(run.export / "decision.json", decision)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed exp080 v4 compute run ID"
    )
    parser.add_argument(
        "--run-id", help="source-neutral identity reserved before dispatch"
    )
    args = parser.parse_args()
    analyse(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
