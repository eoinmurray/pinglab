"""Validate selected bank cells and complete raw inference evidence."""

import math
from pathlib import PurePosixPath

import numpy as np
from experiments.exp022.checkpoints import public_provenance, resolve_checkpoint
from pingstore.contracts import PingstoreError, load_json

from . import recipe


def finite(value, label, *, minimum=0.0, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise PingstoreError(f"invalid or missing {label}")
    return float(value)


def _same(actual, expected):
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return bool(np.isclose(actual, expected))
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same(a, b) for a, b in zip(actual, expected)
        )
    return actual == expected


def checkpoint_tensors(path, train):
    """Read the final checkpoint; reject extra dynamics rather than adapting them."""
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    shapes = recipe.checkpoint_shapes(train)
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise PingstoreError("checkpoint must contain exactly the six TR-05 matrices")
    for name, shape in shapes.items():
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.float32
            or tuple(value.shape) != shape
            or not torch.isfinite(value).all()
        ):
            raise PingstoreError(f"invalid checkpoint tensor {name}")
        if name in ("W_ee.1", "W_ii.1") and torch.count_nonzero(value):
            raise PingstoreError(f"unsupported same-population recurrence {name}")
        if name in ("W_ei.1", "W_ie.1") and (value < 0).any():
            raise PingstoreError(f"negative recurrent weight {name}")
    return state


def training_contract(bank):
    cells = recipe.bank_cells()
    configs = {}
    checkpoints = []
    for cell in cells:
        name = cell["cell_name"]
        cfg = load_json(bank / name / "config.json")
        expected = recipe.training_settings(cell)
        for k, v in expected.items():
            if not _same(cfg.get(k), v):
                raise PingstoreError(f"{name}: training {k} disagrees with recipe")
        for key, value in recipe.RUNTIME_REQUIREMENTS.items():
            if not _same(cfg.get(key), value):
                raise PingstoreError(f"{name}: unsupported runtime setting {key}")
        for key in ("input_rate", "surrogate_slope", "v_grad_dampen"):
            finite(cfg.get(key), key, minimum=1e-12)
        for key in ("n_hidden", "n_inh", "n_in", "n_out"):
            if type(cfg.get(key)) is not int or cfg[key] <= 0:
                raise PingstoreError("invalid training population")
        split = cfg.get("dataset_split")
        expected_split = {
            "checkpoint_selection_partition": "validation",
            "official_test_used_during_training": False,
            "optimizer_train_samples": 6300,
            "validation_samples": 700,
            "official_test_samples": 10000,
            "source_train_partition": "official_mnist_train",
            "source_test_partition": "official_mnist_test",
        }
        if not isinstance(split, dict) or any(
            not _same(split.get(key), value) for key, value in expected_split.items()
        ):
            raise PingstoreError(
                "training requires the retained train/validation/test split"
            )
        checkpoint = public_provenance(
            resolve_checkpoint(bank / name, recipe.CHECKPOINT_ROLE)
        )
        if checkpoint["epoch"] != cfg["epochs"] or checkpoint["training_cell"] != name:
            raise PingstoreError("checkpoint identity differs")
        if cfg["n_in"] != 784 or cfg["n_out"] != 10:
            raise PingstoreError("expected MNIST input population")
        configs[name] = cfg
        checkpoints.append(checkpoint)
    return {"cells": cells, "configs": configs, "checkpoints": checkpoints}


def histories(bank, contract):
    result = {}
    for cell in contract["cells"]:
        name = cell["cell_name"]
        m = load_json(bank / name / "metrics.json")
        rows = m.get("epochs", [])
        if len(rows) != contract["configs"][name]["epochs"]:
            raise PingstoreError("incomplete training history")
        for i, row in enumerate(rows, 1):
            if row.get("ep") != i:
                raise PingstoreError("training epochs must be contiguous")
            finite(row.get("acc"), "validation accuracy", maximum=100)
            for key in ("rate_e", "rate_i"):
                finite(row.get(key), key)
            for key in ("test_rate_e", "test_rate_i", "contrast"):
                if row.get(key) is not None:
                    finite(
                        row[key], key, maximum=1 + 1e-12 if key == "contrast" else None
                    )
            for value in (row.get("weight_norms") or {}).values():
                finite(value, "weight norm")
        finite(m.get("best_acc"), "best validation accuracy", maximum=100)
        if type(m.get("best_epoch")) is not int or not 1 <= m["best_epoch"] <= len(
            rows
        ):
            raise PingstoreError("invalid selected epoch")
        result[name] = m
    return result


def metric(path, train, job):
    m = load_json(path)
    c = m.get("config", {})
    for key in (
        "dt",
        "t_ms",
        "n_in",
        "n_hidden",
        "n_inh",
        "ei_ratio",
        "ei_strength",
        "w_in",
        "w_in_initial_zero_fraction",
    ):
        if not _same(c.get(key), train[key]):
            raise PingstoreError(f"inference metric configuration differs: {key}")
    expected = {
        "dataset": "mnist",
        "evaluation_partition": "official_mnist_test",
        "evaluation_samples": job["samples"],
    }
    if any(c.get(k) != v for k, v in expected.items()):
        raise PingstoreError("evaluation split or sample count differs")
    if type(m.get("n_total")) is not int or m["n_total"] != job["samples"]:
        raise PingstoreError("incomplete sample count")
    acc = finite(m.get("best_acc"), "test accuracy", maximum=100)
    correct = m.get("n_correct")
    if (
        type(correct) is not int
        or not 0 <= correct <= job["samples"]
        or not np.isclose(acc, 100 * correct / job["samples"])
    ):
        raise PingstoreError("accuracy/count mismatch")
    for key in ("hid", "inh"):
        finite(m.get("rates_hz", {}).get(key), f"{key} population rate")
    if PurePosixPath(c.get("load_weights", "")).parts[-2:] != (
        job["cell_name"],
        "weights_final.pth",
    ):
        raise PingstoreError("metric checkpoint identity differs")
    return m


def recordings(directory, train, job):
    if job["kind"] == "infer":
        m = metric(directory / "metrics.json", train, job)
        with np.load(directory / "pop_traces.npz", allow_pickle=False) as data:
            a = data["pop_e"]
            if data["dt"].ndim != 0 or not np.isclose(float(data["dt"]), train["dt"]):
                raise PingstoreError("population trace timestep differs")
            if (
                a.shape != (job["samples"], round(train["t_ms"] / train["dt"]))
                or not np.isfinite(a).all()
                or (a < 0).any()
            ):
                raise PingstoreError("invalid population traces")
        return m
    if job["kind"] == "weights_dump":
        path = directory if directory.is_file() else directory / "weights_dump.npz"
        with np.load(path, allow_pickle=False) as data:
            for key in recipe.WEIGHT_ARRAYS:
                shape = (train["n_hidden"], train["n_inh"])
                if key.startswith("W_ie_"):
                    shape = shape[::-1]
                a = data[key]
                if a.shape != shape or not np.isfinite(a).all() or (a < 0).any():
                    raise PingstoreError("invalid recurrent weights")
        return None
    path = directory if directory.is_file() else directory / "recording.npz"
    with np.load(path, allow_pickle=False) as data:
        if not np.isclose(float(data["dt"]), train["dt"]):
            raise PingstoreError("snapshot timestep differs")
        for key, population in (("spk_e", "n_hidden"), ("spk_i", "n_inh")):
            a = data[key]
            if a.ndim == 3 and a.shape[1] == 1:
                a = a[:, 0, :]
            if a.shape != (
                round(train["t_ms"] / train["dt"]),
                train[population],
            ) or not np.all((a == 0) | (a == 1)):
                raise PingstoreError("invalid snapshot shape or spikes")
        for key, expected in (("n_e", train["n_hidden"]), ("n_i", train["n_inh"])):
            if (
                data[key].ndim != 0
                or data[key].dtype.kind not in "iu"
                or int(data[key]) != expected
            ):
                raise PingstoreError("snapshot population differs")
        label = data["label"]
        if (
            label.ndim != 0
            or label.dtype.kind not in "iu"
            or not 0 <= int(label) < train["n_out"]
        ):
            raise PingstoreError("invalid snapshot label")
