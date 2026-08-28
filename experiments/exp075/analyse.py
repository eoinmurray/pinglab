from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools"), str(REPO / "tools/snnsim")]

import snnlang as snn
from experiments.exp075 import recipe
from experiments.exp075.recipe import (
    EPOCHS,
    LEARNING_RATE,
    MAX_SAMPLES,
    SCALE,
    WEIGHT_DECAY,
)
from experiments.helpers import snnlang_stages as stages
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def analyse(identity, *, run_id=None):
    source = stages.source(REPO, recipe, identity, "compute")
    bundle = snn.load_bundle(source.export / "network.bundle")
    train_dir = source.export / "training"
    with stages.execution(
        REPO, recipe, "analyse", sources={"compute": source}, run_id=run_id
    ) as run:
        metrics = load_json(train_dir / "metrics.json")
        rows = metrics["epochs"]
        if len(rows) != EPOCHS:
            raise PingstoreError("incomplete training trajectory")
        train_losses = [float(row["loss"]) for row in rows]
        test_losses = [float(row["test_loss"]) for row in rows]
        accuracies = [float(row["acc"]) for row in rows]
        payload = {
            "purpose": "bundle-driven training integration demonstration",
            "graph": {
                "name": bundle.graph["name"],
                "digest": bundle.manifest["graph_digest"],
                "training_digest": next(
                    row["digest"]
                    for row in bundle.manifest["files"]
                    if row["path"] == "training.json"
                ),
            },
            "config": {
                **SCALE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "train_count": int(rows[0]["samples"]),
                "held_out_count": MAX_SAMPLES - int(rows[0]["samples"]),
                "dataset_split": metrics["config"]["dataset_split"],
                "validation_encoder_draws": metrics["config"][
                    "validation_encoder_draws"
                ],
                "input_rate_hz": metrics["config"]["input_rate"],
            },
            "trajectory": {
                "epochs": [int(row["ep"]) for row in rows],
                "train_loss": train_losses,
                "test_loss": test_losses,
                "accuracy_pct": accuracies,
                "train_loss_change": train_losses[-1] - train_losses[0],
                "test_loss_change": test_losses[-1] - test_losses[0],
                "accuracy_change_pct_points": accuracies[-1] - accuracies[0],
                "selected_accuracy_pct": float(metrics["best_acc"]),
                "selected_epoch": int(metrics["best_epoch"]),
            },
            "training": {
                "total_elapsed_s": float(metrics["total_elapsed_s"]),
                "weights_written": (train_dir / "weights.pth").is_file(),
                "final_weights_written": (train_dir / "weights_final.pth").is_file(),
            },
        }
        write_json_atomic(
            run.export / "results.json", {"schema": "exp075.analysis/v1", **payload}
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="explicit completed input run")
    parser.add_argument("--run-id", help="unused source-neutral reservation")
    args = parser.parse_args()
    try:
        analyse(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
