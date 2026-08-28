"""Fail-closed scientific validation, in addition to Pingstore checksums."""

import numpy as np
from experiments.helpers.checkpoints import public_provenance, resolve_checkpoint
from pingstore.contracts import PingstoreError, load_json

from . import recipe


def training_contract(bank):
    configs, checkpoints = {}, []
    for seed in recipe.SEEDS:
        name = recipe.training_cell_name(seed)
        cfg = load_json(bank / name / "config.json")
        expected = {
            "model": "ping",
            "dataset": "mnist",
            "dt": 0.1,
            "t_ms": 200.0,
            "n_in": 784,
            "n_hidden": 1024,
            "n_inh": 256,
            "n_out": 10,
            "epochs": 50,
            "max_samples": 7000,
            "seed": seed,
            "readout_mode": "spike-count",
            "input_rates": list(recipe.TRAINING_RATES_HZ),
        }
        if any(cfg.get(k) != v for k, v in expected.items()):
            raise PingstoreError(f"{name}: training configuration differs from TR-06")
        split = cfg.get("dataset_split", {})
        if (
            split.get("checkpoint_selection_partition") != "validation"
            or split.get("official_test_used_during_training") is not False
        ):
            raise PingstoreError(
                "TR-06 requires validation selection and an untouched test partition"
            )
        metrics = load_json(bank / name / "metrics.json")
        if len(metrics.get("epochs", [])) != 50:
            raise PingstoreError("incomplete TR-06 training history")
        checkpoint = public_provenance(
            resolve_checkpoint(bank / name, recipe.CHECKPOINT_ROLE)
        )
        resolve_checkpoint(bank / name, "final_epoch")
        if checkpoint["training_cell"] != name or not 1 <= checkpoint["epoch"] <= 50:
            raise PingstoreError("wrong TR-06 checkpoint identity")
        configs[name] = cfg
        checkpoints.append(checkpoint)
    return {"configs": configs, "checkpoints": checkpoints}


def arrays(path):
    with np.load(path, allow_pickle=False) as raw:
        return {k: raw[k].copy() for k in raw.files}


def counts(path, cfg):
    data = arrays(path)
    shape = (cfg["streams_per_cell"], cfg["digits_per_stream"])
    expected = {
        "labels": shape,
        "e_counts": shape,
        "i_counts": shape,
        "out_counts": (*shape, 10),
    }
    if set(data) != set(expected):
        raise PingstoreError("missing or unexpected count arrays")
    for key, value in data.items():
        if (
            value.shape != expected[key]
            or value.dtype.kind not in "iu"
            or np.any(value < 0)
        ):
            raise PingstoreError("invalid stream-count dimensions or values")
    if np.any(data["labels"] >= 10):
        raise PingstoreError("invalid digit labels")
    return data


def stream(root, name):
    folder = root / "streams" / name
    meta = load_json(folder / "stream.json")
    raw = arrays(folder / "recordings.npz")
    conditions = (
        [[200.0, 5.0]] * 5
        if name == "matched"
        else [list(c) for c in recipe.VARIABLE_STREAM]
    )
    bounds = np.cumsum(
        [0, *[int(round(d / recipe.DT_MS)) for d, _ in conditions]]
    ).tolist()
    if meta.get("conditions") != conditions or meta.get("boundaries") != bounds:
        raise PingstoreError("stream protocol differs")
    labels = meta.get("labels", [])
    if len(labels) != 5 or any(type(v) is not int or not 0 <= v < 10 for v in labels):
        raise PingstoreError("invalid illustrative labels")
    if set(raw) != {"pixels", "spikes_e", "spikes_i", "spikes_out"}:
        raise PingstoreError(
            "stream recordings need explicit pixels and all three populations"
        )
    for key, width in (("spikes_e", 1024), ("spikes_i", 256), ("spikes_out", 10)):
        value = raw[key]
        if (
            value.shape != (bounds[-1], width)
            or value.dtype != np.int8
            or not np.all((value == 0) | (value == 1))
        ):
            raise PingstoreError("invalid binary stream recording")
    pixels = raw["pixels"]
    if (
        pixels.shape != (5, 784)
        or not np.isfinite(pixels).all()
        or np.any(pixels < 0)
        or np.any(pixels > 1)
    ):
        raise PingstoreError("invalid illustrative pixels")
    return raw, meta


def condition(root, job, cfg):
    from . import measurements

    kind = load_json(root / "evidence.json").get("condition_evidence")
    if kind == "historical-aggregate/v1":
        from .historical import aggregate

        return aggregate(root / job["path"] / "condition.json", job, cfg)
    return measurements.condition_row(
        job, counts(root / job["path"] / "counts.npz", cfg), cfg
    )


def validate_compute(root, cfg, *, historical=False):
    expected = {j["id"] for j in recipe.jobs(cfg)}
    if {p.name for p in (root / "jobs").iterdir()} != expected:
        raise PingstoreError("incomplete or extra condition jobs")
    for job in recipe.jobs(cfg):
        if historical:
            from .historical import aggregate

            aggregate(root / job["path"] / "condition.json", job, cfg)
        else:
            counts(root / job["path"] / "counts.npz", cfg)
    for name in ("matched", "variable"):
        stream(root, name)
