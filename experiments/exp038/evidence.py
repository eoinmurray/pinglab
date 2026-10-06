"""Validate selected bank cells and complete raw inference evidence."""


import numpy as np
from experiments.exp022.checkpoints import public_provenance, resolve_checkpoint
from experiments.exp041.evidence import _same, finite
from experiments.exp041.recipe import TRAINING_COMMON_FIELDS
from pingstore.contracts import PingstoreError, load_json

from . import recipe


def training_contract(bank):
    cells = recipe.bank_cells()
    configs = {}
    checkpoints = []
    common = None
    varying = {
        "ei_strength",
        "w_in",
        "v_grad_dampen",
        "fr_reg_upper_strength",
        "fr_reg_upper_target_hz",
    }
    for cell in cells:
        name = cell["cell_name"]
        cfg = load_json(bank / name / "config.json")
        expected = {
            "model": "ping",
            "dataset": "mnist",
            "dt": 0.1,
            "t_ms": 200.0,
            "epochs": 50,
            "max_samples": 7000,
            "seed": cell["seed"],
            "tau_gaba_ms": 6.0,
            "ei_strength": float(cell["model"] == "ping"),
            "v_grad_dampen": 1000.0,
            "w_in": [cell["w_in"], cell["w_in"] * 0.1],
            "fr_reg_upper_strength": 0.0
            if cell["rate_target_hz"] is None
            else recipe.FR_STRENGTH_UPPER,
            "fr_reg_upper_target_hz": cell["rate_target_hz"] or 0.0,
        }
        for k, v in expected.items():
            if not _same(cfg.get(k), v):
                raise PingstoreError(f"{name}: training {k} disagrees with recipe")
        selected = {k: cfg[k] for k in TRAINING_COMMON_FIELDS if k not in varying}
        if common is None:
            common = selected
        elif not _same(common, selected):
            raise PingstoreError(f"{name}: inconsistent common training recipe")
        for key in ("n_hidden", "n_inh", "n_in", "n_out"):
            if type(cfg.get(key)) is not int or cfg[key] <= 0:
                raise PingstoreError("invalid training population")
        split = cfg["dataset_split"]
        if (
            split.get("checkpoint_selection_partition") != "validation"
            or split.get("official_test_used_during_training") is not False
        ):
            raise PingstoreError("training requires held-out checkpoint selection")
        checkpoint = public_provenance(
            resolve_checkpoint(bank / name, recipe.CHECKPOINT_ROLE)
        )
        if (
            not 1 <= checkpoint["epoch"] <= cfg["epochs"]
            or checkpoint["training_cell"] != name
        ):
            raise PingstoreError("checkpoint identity differs")
        if (
            cfg["n_in"] != 784
            or cfg["n_hidden"] < recipe.EI_RASTER_N_E_PLOT
            or cfg["n_inh"] < recipe.EI_RASTER_N_I_PLOT
        ):
            raise PingstoreError(
                "training populations do not support the fixed raster sample"
            )
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
            for preferred, fallback in (
                ("test_rate_e", "rate_e"),
                ("test_rate_i", "rate_i"),
            ):
                finite(
                    row.get(preferred)
                    if row.get(preferred) is not None
                    else row.get(fallback),
                    preferred,
                )
        finite(m.get("best_acc"), "best validation accuracy", maximum=100)
        if type(m.get("best_epoch")) is not int or not 1 <= m["best_epoch"] <= len(
            rows
        ):
            raise PingstoreError("invalid selected epoch")
        result[name] = m
    return result




def population_rate(metrics, prefix):
    rates = metrics.get("rates_hz", {})
    key = next((key for key in rates if key.startswith(prefix)), None)
    return finite(rates.get(key), f"{prefix} population rate")


def metric(path, train, job):
    m = load_json(path)
    c = m.get("config", {})
    expected = {
        k: train[k] for k in ("dt", "t_ms", "n_in", "n_hidden", "n_inh", "ei_ratio")
    }
    expected["ei_strength"] = job.get("ei_strength", train["ei_strength"])
    if job["kind"] == "fi_uniform":
        expected.update(input_rate_hz=job["input_rate"], n_batch=job["trials"])
        for key in ("rate_e_hz", "rate_i_hz"):
            finite(m.get(key), key)
    else:
        expected.update(
            dataset="mnist",
            evaluation_partition="official_mnist_test",
            evaluation_samples=job["samples"],
        )
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
        population_rate(m, "hid")
        population_rate(m, "inh")
    for key, value in expected.items():
        if not _same(c.get(key), value):
            raise PingstoreError(f"inference metric configuration differs: {key}")
    return m


def recordings(directory, train, job):
    if job["kind"] not in ("rate_raster", "ei_raster"):
        path = directory if directory.is_file() else directory / "metrics.json"
        return metric(path, train, job)
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
