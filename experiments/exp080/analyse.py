"""Aggregate explicit retained correctness; never simulate, train or publish.

Invoke this stage explicitly from the repository root. Scientific outputs
and input provenance belong to its independently completed Pingstore run.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import argparse
from typing import Any

import numpy as np
from experiments.exp080 import recipe
from experiments.exp080.custom import evidence, inputs
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
        decision = analyze(correctness, cfg)
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


def analyze(correctness: np.ndarray, cfg: dict | None = None) -> dict[str, Any]:
    cfg = recipe.configuration() if cfg is None else cfg
    if (
        correctness.dtype != np.bool_
        or correctness.ndim != 3
        or correctness.shape[:2] != (len(cfg["rates_hz"]), len(cfg["seeds"]))
        or correctness.shape[2] == 0
    ):
        raise ValueError("expected nonempty boolean rate/seed/image correctness array")
    rows = []
    for rate_index, rate in enumerate(cfg["rates_hz"]):
        per_seed = correctness[rate_index].mean(axis=1)
        rows.append(
            {
                "rate_hz": rate,
                "accuracy": float(per_seed.mean()),
                "minimum_seed_accuracy": float(per_seed.min()),
                "maximum_seed_accuracy": float(per_seed.max()),
                "per_seed_accuracy": per_seed.tolist(),
            }
        )
    floor = next(
        (
            row["rate_hz"]
            for row in rows
            if row["minimum_seed_accuracy"] >= cfg["useful_accuracy"]
        ),
        None,
    )
    return {
        "criterion_crossed": floor is not None,
        "r_train_hz": floor,
        "recommendation": {"floor_hz": floor, "ceiling_hz": max(cfg["rates_hz"])},
        "rows": rows,
    }


if __name__ == "__main__":
    main()
