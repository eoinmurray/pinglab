"""Validate selected bank cells and complete raw inference evidence."""

from pathlib import PurePosixPath

import numpy as np
from experiments.exp041.evidence import _same, finite
from experiments.exp041.recipe import TRAINING_COMMON_FIELDS
from experiments.exp044.evidence import snapshot as validate_snapshot
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
            "v_grad_dampen": 1000.0 if cell["model"] == "ping" else 1.0,
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
        if checkpoint["epoch"] != cfg["epochs"] or checkpoint["training_cell"] != name:
            raise PingstoreError("checkpoint identity differs")
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


def metric(path, cfg, job):
    m = load_json(path)
    c = m.get("config", {})
    for k, v in {
        "dt": cfg["dt"],
        "t_ms": cfg["t_ms"],
        "seed": cfg["seed"],
        "tau_gaba_ms": cfg["tau_gaba_ms"],
        "dataset": "mnist",
        "n_hidden": cfg["n_hidden"],
        "n_inh": cfg["n_inh"],
        "evaluation_partition": "official_mnist_test",
        "evaluation_samples": job["samples"],
    }.items():
        if not _same(c.get(k), v):
            raise PingstoreError(f"inference {k} differs")
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
        finite(m.get("rates_hz", {}).get(key), key)
    if job["kind"] == "scale":
        finite(m.get("ce_loss"), "cross entropy")
    return m


def recordings(directory, cfg, job):
    if job["kind"] == "snapshot":
        data = validate_snapshot(
            directory / "snapshot.npz", cfg["dt"], {**cfg, "t_ms": 400.0}
        )
        return data
    metric(directory / "metrics.json", cfg, job)
    if job["kind"] == "scale":
        with np.load(directory / "per_cell_rates.npz", allow_pickle=False) as data:
            rates = data["rate_e_per_sample"]
        if (
            rates.shape != (job["samples"],)
            or not np.isfinite(rates).all()
            or np.any(rates < 0)
        ):
            raise PingstoreError("invalid sample-wise rates")
    if job["kind"] != "pfg" or not job["is_ping"]:
        return
    with np.load(directory / "pop_traces.npz", allow_pickle=False) as data:
        p = data["pop_e"]
        if (
            p.shape != (job["samples"], round(cfg["t_ms"] / cfg["dt"]))
            or not np.isfinite(p).all()
            or np.any(p < 0)
            or not np.isclose(float(data["dt"]), cfg["dt"])
        ):
            raise PingstoreError("invalid population traces")
    with np.load(directory / "rasters.npz", allow_pickle=False) as r:
        for k, v in {
            "n_trials": job["samples"],
            "T": p.shape[1],
            "n_e": cfg["n_hidden"],
            "n_i": cfg["n_inh"],
        }.items():
            if r[k].ndim != 0 or r[k].dtype.kind not in "iu" or int(r[k]) != v:
                raise PingstoreError("raster dimensions differ")
        if not np.isclose(float(r["dt"]), cfg["dt"]):
            raise PingstoreError("raster timestep differs")
        for prefix, pop in (("e", cfg["n_hidden"]), ("i", cfg["n_inh"])):
            arrays = [r[f"{prefix}_{suffix}"] for suffix in ("trial", "t", "cell")]
            if len({a.size for a in arrays}) != 1:
                raise PingstoreError("raster index lengths differ")
            for a, limit in zip(arrays, (job["samples"], p.shape[1], pop), strict=True):
                if (
                    a.ndim != 1
                    or a.dtype.kind not in "iu"
                    or np.any(a < 0)
                    or np.any(a >= limit)
                ):
                    raise PingstoreError("invalid sparse index")
            tr, ts, cell = [a.astype(np.int64) for a in arrays]
            linear = (tr * p.shape[1] + ts) * pop + cell
            if np.unique(linear).size != linear.size:
                raise PingstoreError("duplicate sparse spikes")


def inference_config(config, train, job):
    keys = (
        "model",
        "dt",
        "dataset",
        "ei_strength",
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
    )
    expected = {
        **{k: train[k] for k in keys},
        "t_ms": 400.0 if job["kind"] == "snapshot" else train["t_ms"],
        "tau_gaba": train["tau_gaba_ms"],
        "infer": True,
        "input": "dataset",
        "scale_w_in": job.get("scale", 1.0),
        "scale_w_ei": 1.0,
        "scale_w_ie": 1.0,
        "intervention": [],
        "scale_projection": [],
        "max_samples": job.get("samples"),
    }
    if job["kind"] == "snapshot":
        expected.update(digit=0, sample=0)
    for key, value in expected.items():
        if not _same(config.get(key), value):
            raise PingstoreError(f"inference configuration differs: {key}")
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
