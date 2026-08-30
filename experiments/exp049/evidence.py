"""Validate selected bank cells and complete raw inference evidence."""

from pathlib import PurePosixPath

import numpy as np
from experiments.exp041.evidence import _same, finite
from experiments.exp041.recipe import TRAINING_COMMON_FIELDS
from experiments.helpers.checkpoints import public_provenance, resolve_checkpoint
from pingstore.contracts import PingstoreError, load_json

from . import recipe


def training_contract(bank):
    cells = recipe.bank_cells()
    configs = {}
    checkpoints = []
    common = None
    varying = {
        "ei_strength",
        "trainable_w_ei",
        "trainable_w_ie",
    }
    for cell in cells:
        name = cell["cell_name"]
        cfg = load_json(bank / name / "config.json")
        expected = {
            "hidden_sizes": [recipe.N_E],
            "n_hidden": recipe.N_E,
            "n_inh": recipe.N_I,
            "model": "ping",
            "dataset": "mnist",
            "dt": 0.1,
            "t_ms": 200.0,
            "epochs": 50,
            "max_samples": 7000,
            "seed": cell["seed"],
            "tau_gaba_ms": 6.0,
            "ei_strength": {
                "frozen_ping": 1.0,
                "trainable_ping_init": 1.0,
                "trainable_zero_init": 0.0,
                "trainable_small_init": 0.1,
            }[cell["condition"]],
            "trainable_w_ei": cell["condition"] != "frozen_ping",
            "trainable_w_ie": cell["condition"] != "frozen_ping",
            "v_grad_dampen": 1000.0,
            "w_in": [0.9, 0.09],
            "w_in_initial_zero_fraction": 0.95,
            "readout_mode": "mem-mean",
            "surrogate_slope": 1.0,
            "readout_w_init_mean": 1.12060546875,
            "readout_w_init_std": 0.8349609375,
            "lr": 0.0004,
            "batch_size": 256,
            "fr_reg_upper_strength": 0.0,
            "fr_reg_upper_target_hz": 0.0,
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
        if checkpoint["epoch"] != cfg["epochs"] or checkpoint["training_cell"] != name:
            raise PingstoreError("checkpoint identity differs")
        if cfg["n_in"] != 784:
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


def inference_config(config, train, job):
    keys = (
        "model",
        "dt",
        "dataset",
        "ei_ratio",
        "w_in",
        "readout_mode",
        "dales_law",
        "signed_readout",
        "readout_bias",
        "adaptive_threshold",
        "train_leak",
        "state_clamp",
        "trainable_w_ee",
        "trainable_w_ei",
        "trainable_w_ie",
        "trainable_w_ii",
        "n_in",
        "seed",
        "w_in_initial_zero_fraction",
        "recurrent_initial_zero_fraction",
        "tau_m_e_bounds_ms",
        "tau_m_i_bounds_ms",
        "readout_w_init_mean",
        "readout_w_init_std",
        "surrogate_slope",
    )
    expected = {
        **{k: train[k] for k in keys},
        "t_ms": train["t_ms"],
        "tau_gaba": train["tau_gaba_ms"],
        "ei_strength": train["ei_strength"],
        "infer": None if job["kind"] == "weights_dump" else True,
        "mode": "dump-weights" if job["kind"] == "weights_dump" else "sim",
        "input": "dataset",
        "spike_rate": job.get("input_rate", train["input_rate"]),
        "scale_w_in": 1.0,
        "scale_w_ei": 1.0,
        "scale_w_ie": 1.0,
        "intervention": [],
        "scale_projection": [],
        "max_samples": job.get("samples"),
    }
    if "sample_index" in job:
        expected["sample_index"] = job["sample_index"]
    elif config.get("sample_index") is not None:
        raise PingstoreError("unexpected single-image selection")
    if job["kind"] == "weights_dump":
        for key in (
            "n_in",
            "scale_w_in",
            "scale_w_ei",
            "scale_w_ie",
            "intervention",
            "scale_projection",
        ):
            expected.pop(key)
    for key, value in expected.items():
        if not _same(config.get(key), value):
            raise PingstoreError(f"inference configuration differs: {key}")
    skip = []
    if (config.get("skip_load") or []) != skip:
        raise PingstoreError("inference transfer-load policy differs")
    if config.get("n_hidden") not in (train["n_hidden"], [train["n_hidden"]]):
        raise PingstoreError("inference hidden population differs")
    for key, filename in (
        ("load_weights", "weights_final.pth"),
        ("load_config", "config.json"),
    ):
        if PurePosixPath(config.get(key, "")).parts[-2:] != (
            job["cell_name"],
            filename,
        ):
            raise PingstoreError("inference checkpoint identity differs")


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
