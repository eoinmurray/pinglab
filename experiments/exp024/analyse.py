"""Audit six retained baseline histories; never train, simulate, draw or publish."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from statistics import fmean, stdev

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "tools")]

from experiments.exp024 import recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PingstoreError(f"missing or nonfinite {label}")
    return float(value)


def read_cell(directory: Path, model: str, seed: int) -> tuple[dict, dict]:
    config = load_json(directory / "config.json")
    metrics = load_json(directory / "metrics.json")
    for key, expected in (("training_cell_name", recipe.cell_name(model, seed)),
                          ("training_run_id", recipe.TRAINING_RUN), ("seed", seed),
                          ("dataset", "mnist"), ("readout_mode", "mem-mean"),
                          ("fr_reg_upper_strength", 0.0),
                          ("ei_strength", 0.0 if model == "coba" else 1.0)):
        if config.get(key) != expected:
            raise PingstoreError(f"{directory.name}: expected {key}={expected!r}")
    split = config.get("dataset_split", {})
    if (split.get("checkpoint_selection_partition") != "validation"
            or split.get("official_test_used_during_training") is not False):
        raise PingstoreError(f"{directory.name}: explicit validation split required")
    for key in ("optimizer_train_samples", "validation_samples", "official_test_samples"):
        if finite(split.get(key), key) <= 0:
            raise PingstoreError(f"{directory.name}: invalid split size {key}")
    for key in ("epochs", "t_ms", "dt", "batch_size", "max_samples", "n_in",
                "n_hidden", "n_inh", "n_out", "input_rate", "v_grad_dampen", "lr"):
        if finite(config.get(key), key) <= 0:
            raise PingstoreError(f"{directory.name}: invalid {key}")
    draws = config.get("validation_encoder_draws", {})
    if finite(draws.get("count"), "validation draw count") < 1:
        raise PingstoreError("validation draw count must be positive")
    epochs = metrics.get("epochs")
    if not isinstance(epochs, list) or len(epochs) < 2 or len(epochs) != config["epochs"]:
        raise PingstoreError(f"{directory.name}: incomplete epoch history")
    retained = []
    for index, epoch in enumerate(epochs, 1):
        if not isinstance(epoch, dict):
            raise PingstoreError(f"{directory.name}: epoch {index} must be an object")
        if epoch.get("ep") != index:
            raise PingstoreError(f"{directory.name}: epochs must be contiguous from 1")
        values = {key: finite(epoch.get(key), f"{directory.name} epoch {index} {key}")
                  for key in recipe.FIELDS}
        if not 0 <= values["acc"] <= 100 or any(
                values[key] < 0 for key in recipe.FIELDS if key != "acc"):
            raise PingstoreError(f"{directory.name}: invalid accuracy, loss or rate")
        if values["test_loss"] == 0:
            raise PingstoreError(f"{directory.name}: log CE plot requires positive loss")
        norms = epoch.get("weight_norms")
        if not isinstance(norms, dict):
            raise PingstoreError(f"{directory.name}: missing weight norms at epoch {index}")
        weight_norms = {
            key: finite(norms.get(key), f"{directory.name} epoch {index} {key}")
            for key in recipe.PARAMETERS
        }
        retained.append({"ep": index, **values, "weight_norms": weight_norms})
    return {"name": directory.name, "model": model, "seed": seed, "epochs": retained}, config


def diagnose(cell: dict) -> dict:
    epochs = cell["epochs"]
    acc_slope = recipe.slope_last_n([e["acc"] for e in epochs])
    rate_slope = recipe.slope_last_n([e["test_rate_e"] for e in epochs])
    return {
        "name": cell["name"], "model": cell["model"], "seed": cell["seed"],
        "epochs_completed": len(epochs), "window_epochs": min(recipe.WINDOW, len(epochs)),
        "final_acc": epochs[-1]["acc"], "final_e_rate_hz": epochs[-1]["test_rate_e"],
        "final_i_rate_hz": epochs[-1]["test_rate_i"], "final_loss": epochs[-1]["loss"],
        "final_validation_loss": epochs[-1]["test_loss"],
        "acc_slope_last10_pp_per_ep": acc_slope,
        "e_rate_slope_last10_hz_per_ep": rate_slope,
        "i_rate_slope_last10_hz_per_ep": recipe.slope_last_n([e["test_rate_i"] for e in epochs]),
        "loss_slope_last10_per_ep": recipe.slope_last_n([e["loss"] for e in epochs]),
        "validation_loss_slope_last10_per_ep": recipe.slope_last_n([e["test_loss"] for e in epochs]),
        "accuracy_converged": abs(acc_slope) < recipe.ACCURACY_THRESHOLD,
        "e_rate_converged": abs(rate_slope) < recipe.RATE_THRESHOLD,
        "accuracy_marker_epoch": recipe.accuracy_marker([e["acc"] for e in epochs]),
        "weight_drift": {key: {
            "first_epoch": epochs[0]["weight_norms"][key],
            "final": epochs[-1]["weight_norms"][key],
            "ratio_final_over_first": (epochs[-1]["weight_norms"][key] / epochs[0]["weight_norms"][key]
                                       if epochs[0]["weight_norms"][key] else None),
            "slope_last10": recipe.slope_last_n([e["weight_norms"][key] for e in epochs]),
        } for key in recipe.PARAMETERS},
    }


def analyse(identity: str, *, run_id: str | None = None) -> str:
    compute = source_run(REPO / ".pingstore", identity, stage="compute", experiment="exp022")
    inputs = {"compute": compute}
    bank = compute
    if "bank" in compute.record["inputs"]:
        ref = compute.record["inputs"]["bank"]
        bank = source_run(REPO / ".pingstore", ref["run_id"], stage="compute",
                          experiment="exp022", reference=ref)
        inputs["bank"] = bank
    configuration = {"training_run": recipe.TRAINING_RUN, "window_epochs": recipe.WINDOW,
                     "accuracy_threshold_pp_per_epoch": recipe.ACCURACY_THRESHOLD,
                     "rate_threshold_hz_per_epoch": recipe.RATE_THRESHOLD,
                     "accuracy_marker_fraction": recipe.ACCURACY_FRACTION,
                     "slope_estimator": "endpoint_secant", "threshold_rule": "absolute_slope_strictly_below"}
    with stage_run(REPO, recipe.SLUG, "analyse", inputs=inputs, run_id=run_id,
                   configuration=configuration) as run:
        curves, configs, rows = [], {}, []
        for model in recipe.MODELS:
            for seed in recipe.SEEDS:
                cell, config = read_cell(bank.unit(recipe.cell_name(model, seed)), model, seed)
                curves.append(cell)
                configs[cell["name"]] = config
                rows.append(diagnose(cell))
        first = next(iter(configs.values()))
        common_keys = ("epochs", "t_ms", "dt", "batch_size", "max_samples", "n_in",
                       "n_hidden", "n_inh", "n_out", "input_rate", "dataset_split",
                       "validation_encoder_draws", "lr")
        for config in configs.values():
            if any(config[key] != first[key] for key in common_keys):
                raise PingstoreError("baseline cells have mismatched training or validation scales")
        scale = {key: first[key] for key in common_keys}
        scale.update(dataset=first["dataset"], models=list(recipe.MODELS), seeds=list(recipe.SEEDS),
                     cells=len(rows), readout=first["readout_mode"],
                     window_epochs=min(recipe.WINDOW, first["epochs"]),
                     voltage_gradient_damping={model: configs[recipe.cell_name(model, recipe.SEEDS[0])]["v_grad_dampen"]
                                               for model in recipe.MODELS})
        aggregate_fields = ("final_acc", "final_e_rate_hz", "final_i_rate_hz",
                            "acc_slope_last10_pp_per_ep", "e_rate_slope_last10_hz_per_ep")
        models = {}
        for model in recipe.MODELS:
            group = [row for row in rows if row["model"] == model]
            markers = [row["accuracy_marker_epoch"] for row in group if row["accuracy_marker_epoch"] is not None]
            models[model] = {
                **{key: {"mean": fmean(row[key] for row in group),
                         "sd": stdev(row[key] for row in group)} for key in aggregate_fields},
                "accuracy_converged_count": sum(row["accuracy_converged"] for row in group),
                "e_rate_converged_count": sum(row["e_rate_converged"] for row in group),
                "accuracy_marker_epoch_mean": fmean(markers) if markers else None,
            }
        write_json_atomic(run.export / "curves.json", {"schema": "exp024.curves/v1", "cells": curves})
        write_json_atomic(run.export / "results.json", {
            "schema": "exp024.analysis/v1", "config": scale, "measurement": configuration,
            "cells": rows, "models": models,
        })
        write_json_atomic(run.scratch / "source-configurations.json", configs)
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp022 compute run ID")
    parser.add_argument("--run-id", help="unused identity reserved before dispatch")
    args = parser.parse_args()
    analyse(args.source, run_id=args.run_id)


if __name__ == "__main__":
    main()
