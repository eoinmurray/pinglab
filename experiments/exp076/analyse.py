from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools"), str(REPO / "tools/snnsim")]

import json

import snnlang as snn
import torch
from experiments.exp076 import recipe
from experiments.exp076.recipe import (
    LEARNING_RATE,
    MAX_SAMPLES,
    N_CLASSES,
    N_E,
    N_I,
    N_INPUT,
    READOUT_INIT,
    READOUT_TAU_MS,
    SCALE,
    TAU_GABA_MS,
    W_EI,
    W_IE,
    W_IN,
    WEIGHT_DECAY,
)
from experiments.helpers import snnlang_stages as stages
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def checkpoint_status(
    kind: str,
    weights: Path,
    *,
    artifact_path: str,
    bundle_dir: Path | None = None,
) -> dict:
    if kind not in {"bundle", "legacy"}:
        raise ValueError(kind)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    expected_shapes = {
        "W_ff.0": [N_INPUT, N_E],
        "W_ff.1": [N_E, N_CLASSES],
        "W_ee.1": [N_E, N_E],
        "W_ei.1": [N_E, N_I],
        "W_ie.1": [N_I, N_E],
        "W_ii.1": [N_I, N_I],
    }
    expected = set(expected_shapes)
    observed = set(state)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    shape_mismatch = {
        key: {
            "checkpoint": list(value.shape),
            "expected": expected_shapes[key],
        }
        for key, value in state.items()
        if key in expected_shapes and list(value.shape) != expected_shapes[key]
    }
    return {
        "kind": kind,
        "weights": artifact_path,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch": shape_mismatch,
        "ok": not missing and not unexpected and not shape_mismatch,
        "state_keys": sorted(state),
    }


def analyse(identity, *, run_id=None):
    source = stages.source(REPO, recipe, identity, "compute")
    with stages.execution(
        REPO, recipe, "analyse", sources={"compute": source}, run_id=run_id
    ) as run:
        bundle_dir = source.export / "network.bundle"
        train_dir = source.export / "bundle_training"
        legacy_train_dir = source.export / "legacy_training"
        selected_infer_dir = source.export / "bundle_replay_selected"
        final_infer_dir = source.export / "bundle_replay_final"
        legacy_load_bundle_dir = source.export / "legacy_load_bundle_checkpoint"
        bundle_load_legacy_dir = source.export / "bundle_load_legacy_checkpoint"
        bundle = snn.load_bundle(bundle_dir)
        command_records = source.record["execution"]["commands"]
        metrics = json.loads((train_dir / "metrics.json").read_text())
        selected_replay = json.loads((selected_infer_dir / "metrics.json").read_text())
        final_replay = json.loads((final_infer_dir / "metrics.json").read_text())
        legacy_load_bundle = json.loads(
            (legacy_load_bundle_dir / "metrics.json").read_text()
        )
        bundle_load_legacy = json.loads(
            (bundle_load_legacy_dir / "metrics.json").read_text()
        )
        legacy_metrics = json.loads((legacy_train_dir / "metrics.json").read_text())

        rows = metrics["epochs"]
        selected_train_acc = float(metrics["best_acc"])
        final_train_acc = float(rows[-1]["acc"])
        selected_replay_acc = float(selected_replay["best_acc"])
        final_replay_acc = float(final_replay["best_acc"])

        checkpoint_checks = {
            "bundle_selected_through_bundle": checkpoint_status(
                "bundle",
                train_dir / "weights.pth",
                artifact_path="export/bundle_training/weights.pth",
                bundle_dir=bundle_dir,
            ),
            "bundle_final_through_bundle": checkpoint_status(
                "bundle",
                train_dir / "weights_final.pth",
                artifact_path="export/bundle_training/weights_final.pth",
                bundle_dir=bundle_dir,
            ),
            "bundle_selected_through_legacy": checkpoint_status(
                "legacy",
                train_dir / "weights.pth",
                artifact_path="export/bundle_training/weights.pth",
            ),
            "legacy_selected_through_bundle": checkpoint_status(
                "bundle",
                legacy_train_dir / "weights.pth",
                artifact_path="export/legacy_training/weights.pth",
                bundle_dir=bundle_dir,
            ),
        }

        gate = load_json(source.export / "parity.json")
        if (
            gate.get("passed") is not True
            or gate.get("tests") != 1
            or any(gate.get(key) != 0 for key in ("failures", "errors", "skipped"))
        ):
            raise PingstoreError("missing successful one-step parity evidence")
        parity = {
            "automated_test": (
                "tools/snnsim/tests/test_bundle.py::"
                "test_bundle_and_legacy_one_step_training_are_exactly_equivalent"
            ),
            "initial_state_dict": "exact",
            "forward_logits": "exact",
            "cross_entropy_loss": "exact",
            "gradients": "exact",
            "adamw_step": "exact",
            "tolerance": {"rtol": 0.0, "atol": 0.0},
            "trainable_parameters": ["W_ff.0", "W_ff.1"],
            "frozen_parameters": ["W_ee.1", "W_ei.1", "W_ie.1", "W_ii.1"],
            "checkpoint_structural_ok": all(
                row["ok"] for row in checkpoint_checks.values()
            ),
        }

        payload = {
            "purpose": "checkpoint replay and bundle/legacy equivalence gate",
            "scope": (
                "validates only the current MNIST PING + MeanVoltage tools/snnsim "
                "bundle adapter subset"
            ),
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
                "w_in": list(W_IN),
                "w_ei": list(W_EI),
                "w_ie": list(W_IE),
                "tau_gaba_ms": TAU_GABA_MS,
                "readout": "MeanVoltage",
                "readout_init": list(READOUT_INIT),
                "readout_tau_ms": READOUT_TAU_MS,
                "parameter_scope": {
                    "trainable": parity["trainable_parameters"],
                    "frozen": parity["frozen_parameters"],
                },
            },
            "trajectory": {
                "epochs": [int(row["ep"]) for row in rows],
                "train_loss": [float(row["loss"]) for row in rows],
                "test_loss": [float(row["test_loss"]) for row in rows],
                "accuracy_pct": [float(row["acc"]) for row in rows],
                "selected_accuracy_pct": selected_train_acc,
                "selected_epoch": int(metrics["best_epoch"]),
                "final_accuracy_pct": final_train_acc,
            },
            "replay": {
                "selected_checkpoint_accuracy_pct": selected_replay_acc,
                "trainer_best_accuracy_pct": selected_train_acc,
                "selected_delta_pct_points": selected_replay_acc - selected_train_acc,
                "final_checkpoint_accuracy_pct": final_replay_acc,
                "trainer_final_epoch_accuracy_pct": final_train_acc,
                "final_delta_pct_points": final_replay_acc - final_train_acc,
                "evaluation_samples": int(selected_replay["n_total"]),
                "evaluation_partition": selected_replay["config"][
                    "evaluation_partition"
                ],
                "explanation": "Training validation and replay use different data partitions and encoder aggregation; their accuracies are not an exact replay test.",
            },
            "compatibility": {
                "checkpoint_checks": checkpoint_checks,
                "legacy_route_accuracy_on_bundle_checkpoint_pct": float(
                    legacy_load_bundle["best_acc"]
                ),
                "bundle_route_accuracy_on_legacy_checkpoint_pct": float(
                    bundle_load_legacy["best_acc"]
                ),
                "legacy_checkpoint_best_accuracy_pct": float(
                    legacy_metrics["best_acc"]
                ),
            },
            "parity": parity,
            "runtime": {
                "total_elapsed_s": sum(
                    row["elapsed_s"] for row in command_records.values()
                )
                + gate["elapsed_s"],
                "training_elapsed_s": float(metrics["total_elapsed_s"]),
                "command_elapsed_s": {
                    key: float(value["elapsed_s"])
                    for key, value in command_records.items()
                },
            },
            "artifacts": {
                "bundle": "export/network.bundle",
                "bundle_training": "export/bundle_training",
                "legacy_training": "export/legacy_training",
                "selected_replay": "export/bundle_replay_selected",
                "final_replay": "export/bundle_replay_final",
            },
        }
        write_json_atomic(
            run.export / "results.json", {"schema": "exp076.analysis/v1", **payload}
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
