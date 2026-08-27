"""Validate retained cells and measurements without running a simulator."""

import math
from pathlib import Path

import numpy as np
from experiments.helpers.checkpoints import public_provenance, resolve_checkpoint
from pingstore.contracts import PingstoreError, load_json

from . import recipe


def finite(value, label: str, *, minimum=0.0, maximum=None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise PingstoreError(f"invalid or missing {label}")
    return float(value)


def _same(actual, expected) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return bool(np.isclose(actual, expected))
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same(a, b) for a, b in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same(actual[k], expected[k]) for k in actual
        )
    return actual == expected


def training_contract(bank: Path) -> dict:
    common = None
    cells = []
    for tau in recipe.TAU_GABA_SWEEP:
        for seed in recipe.SEEDS:
            name = recipe.cell_name(tau, seed)
            cfg = load_json(bank / name / "config.json")
            if not _same(cfg.get("tau_gaba_ms"), tau):
                raise PingstoreError(
                    f"{name}: config tau_gaba_ms does not match registered {tau}"
                )
            if type(cfg.get("seed")) is not int or cfg["seed"] != seed:
                raise PingstoreError(
                    f"{name}: config seed does not match registered {seed}"
                )
            try:
                selected = {k: cfg[k] for k in recipe.TRAINING_COMMON_FIELDS}
            except KeyError as exc:
                raise PingstoreError(
                    f"{name}: missing training config field {exc.args[0]}"
                ) from exc
            if common is None:
                common = selected
            else:
                for key, expected in common.items():
                    if not _same(selected[key], expected):
                        raise PingstoreError(
                            f"{name}: config {key}={selected[key]!r} disagrees with common value {expected!r}"
                        )
            cells.append({"cell_name": name, "tau_gaba_ms": tau, "seed": seed})
    for key, value in {
        "model": "ping",
        "dataset": "mnist",
        "t_ms": recipe.T_MS,
        "epochs": recipe.EPOCHS,
        "max_samples": recipe.MAX_SAMPLES,
        "dt": recipe.DT_TRAIN,
        "ei_strength": 1.0,
        "fr_reg_upper_strength": 0.0,
    }.items():
        if common[key] != value:
            raise PingstoreError(f"training recipe requires {key}={value}")
    for key in ("n_in", "n_hidden", "n_inh", "n_out", "batch_size"):
        if type(common[key]) is not int or common[key] <= 0:
            raise PingstoreError(f"invalid training {key}")
    if (
        common["n_hidden"] < recipe.RASTER_N_E_PLOT
        or common["n_inh"] < recipe.RASTER_N_I_PLOT
    ):
        raise PingstoreError("training populations are smaller than the raster sample")
    split = common["dataset_split"]
    expected_split = {
        "optimizer_train_samples": 6300,
        "validation_samples": 700,
        "official_test_samples": 10000,
        "checkpoint_selection_partition": "validation",
        "official_test_used_during_training": False,
        "source_train_partition": "official_mnist_train",
        "source_test_partition": "official_mnist_test",
    }
    if not isinstance(split, dict) or any(
        split.get(k) != v for k, v in expected_split.items()
    ):
        raise PingstoreError(
            "training requires the retained MNIST train/validation/test split"
        )
    draws = common["validation_encoder_draws"]
    if (
        not isinstance(draws, dict)
        or type(draws.get("count")) is not int
        or draws["count"] < 1
        or len(draws.get("encoder_seeds", [])) != draws["count"]
        or len(draws.get("input_rate_seeds", [])) != draws["count"]
    ):
        raise PingstoreError("missing validation encoder-draw contract")
    return {
        "common": common,
        "cells": cells,
        "registered_differences": {
            "tau_gaba_ms": list(recipe.TAU_GABA_SWEEP),
            "seeds": list(recipe.SEEDS),
        },
    }


def checkpoints(bank: Path, contract: dict) -> list[dict]:
    rows = []
    for cell in contract["cells"]:
        row = resolve_checkpoint(bank / cell["cell_name"], recipe.CHECKPOINT_ROLE)
        if (
            row["epoch"] != contract["common"]["epochs"]
            or row["training_cell"] != cell["cell_name"]
        ):
            raise PingstoreError(
                "checkpoint identity or final epoch disagrees with training contract"
            )
        rows.append(public_provenance(row))
    return rows


def histories(bank: Path, contract: dict) -> list[dict]:
    curves = []
    for cell in contract["cells"]:
        metrics = load_json(bank / cell["cell_name"] / "metrics.json")
        epochs = metrics.get("epochs")
        if not isinstance(epochs, list) or len(epochs) != contract["common"]["epochs"]:
            raise PingstoreError(f"{cell['cell_name']}: incomplete training history")
        retained = []
        for index, row in enumerate(epochs, 1):
            if not isinstance(row, dict) or row.get("ep") != index:
                raise PingstoreError("training epochs must be contiguous from 1")
            retained.append(
                {
                    "ep": index,
                    "acc": finite(row.get("acc"), "validation accuracy", maximum=100),
                    "test_rate_e": finite(row.get("test_rate_e"), "validation E rate"),
                }
            )
        curves.append({**cell, "epochs": retained})
    return curves


def measurement(path: Path, cell: dict, common: dict, samples: int) -> dict:
    m = load_json(path)
    cfg = m.get("config", {})
    for key, expected in {
        "dt": common["dt"],
        "tau_gaba_ms": cell["tau_gaba_ms"],
        "seed": cell["seed"],
        "t_ms": common["t_ms"],
        "evaluation_partition": "official_mnist_test",
        "evaluation_samples": samples,
        "dataset": "mnist",
        "n_hidden": common["n_hidden"],
        "n_inh": common["n_inh"],
    }.items():
        if not _same(cfg.get(key), expected):
            raise PingstoreError(f"inference {key} disagrees with retained recipe")
    if type(m.get("n_total")) is not int or m["n_total"] != samples:
        raise PingstoreError("incomplete inference sample count")
    acc = finite(m.get("best_acc"), "test accuracy", maximum=100)
    correct = m.get("n_correct")
    if (
        type(correct) is not int
        or not 0 <= correct <= samples
        or not np.isclose(acc, 100 * correct / samples)
    ):
        raise PingstoreError("test accuracy disagrees with retained counts")
    rates = m.get("rates_hz", {})

    def rate(names):
        keys = [key for key in names if key in rates]
        if len(keys) != 1:
            raise PingstoreError(f"missing or ambiguous population rate: {names}")
        return finite(rates[keys[0]], keys[0])

    return {
        "tau_gaba_ms": cell["tau_gaba_ms"],
        "seed": cell["seed"],
        "t_ms": common["t_ms"],
        "acc": acc,
        "e_rate_hz": rate(("hid", "hid1")),
        "n_total": samples,
    }


def snapshot(path: Path, dt: float, common: dict) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if not np.isclose(float(data["dt"]), dt):
            raise PingstoreError("snapshot timestep mismatch")
        result = {key: np.array(data[key]) for key in ("spk_e", "spk_i")}
        label = data["label"].item()
        if not isinstance(label, (int, np.integer)) or not 0 <= label <= 9:
            raise PingstoreError("invalid snapshot class label")
        result["label"] = int(label)
    steps = round(common["t_ms"] / dt)
    for key, population in (("spk_e", "n_hidden"), ("spk_i", "n_inh")):
        array = result[key]
        if array.ndim == 3 and array.shape[1] == 1:
            array = array[:, 0, :]
            result[key] = array
        if array.shape != (steps, common[population]) or not np.all(
            (array == 0) | (array == 1)
        ):
            raise PingstoreError(f"invalid snapshot shape or spikes for {key}")
    return result


def population_traces(path: Path, common: dict, samples: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        traces = np.array(data["pop_e"])
        if not np.isclose(float(data["dt"]), common["dt"]):
            raise PingstoreError("population trace timestep mismatch")
    if traces.shape != (samples, round(common["t_ms"] / common["dt"])):
        raise PingstoreError("population trace shape disagrees with evaluation recipe")
    if not np.isfinite(traces).all() or (traces < 0).any():
        raise PingstoreError("invalid population trace values")
    return traces
