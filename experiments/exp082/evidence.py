"""Fail-closed validation of current and retained scientific evidence."""

import math

import numpy as np
from experiments.helpers.checkpoints import public_provenance, resolve_checkpoint
from pingstore.contracts import PingstoreError, file_sha256, load_json

from . import recipe

RETAINED_SCHEMA = "exp082.gold2-import/v1"
RETAINED_PRODUCER = "73f0883edc14aa634f5a6d55e4f4123fbfeb7508"


def aggregate(path, job, cfg):
    """Validate one retained aggregate condition row."""
    row = load_json(path)
    keys = {
        "seed",
        "duration_ms",
        "rate_hz",
        "stream_batch_size",
        "n_correct",
        "n_total",
        "accuracy",
        "output_spikes_per_presentation",
        "silent_fraction",
        "class_spike_totals",
        "rate_e_hz",
        "rate_i_hz",
    }
    if set(row) != keys or any(
        row[k] != job[k] for k in ("seed", "duration_ms", "rate_hz")
    ):
        raise PingstoreError("retained condition identity differs")
    total, correct = row["n_total"], row["n_correct"]
    if (
        type(total) is not int
        or total != cfg["digits_per_seed_cell"]
        or type(correct) is not int
        or not 0 <= correct <= total
        or row["stream_batch_size"] != cfg["stream_batch_size"]
        or row["accuracy"] != correct / total
    ):
        raise PingstoreError("invalid retained condition totals")
    spikes = row["class_spike_totals"]
    if (
        len(spikes) != 10
        or any(type(v) is not int or v < 0 for v in spikes)
        or row["output_spikes_per_presentation"] != sum(spikes) / total
    ):
        raise PingstoreError("invalid retained output counts")
    silent = row["silent_fraction"]
    if (
        not isinstance(silent, (int, float))
        or not 0 <= silent <= 1
        or not math.isclose(silent * total, round(silent * total), abs_tol=1e-9)
    ):
        raise PingstoreError("invalid retained silent fraction")
    for key in ("rate_e_hz", "rate_i_hz"):
        if (
            not isinstance(row[key], (float, int))
            or not math.isfinite(row[key])
            or row[key] < 0
        ):
            raise PingstoreError("invalid retained population rate")
    return row


def validate_import(run, cfg):
    """Validate the provenance retained by an existing immutable import run."""
    record = run.record.get("historical_import", {})
    if (
        run.record["execution"].get("operation") != "historical-import"
        or record.get("schema") != RETAINED_SCHEMA
        or record.get("producer_commit") != RETAINED_PRODUCER
        or cfg != recipe.configuration()
    ):
        raise PingstoreError("aggregate evidence lacks its retained import contract")
    proof = load_json(run.directory / "provenance/import.json")
    if (
        proof.get("schema") != RETAINED_SCHEMA
        or proof.get("source_files") != 199
        or proof.get("source_bytes") != 6079619
    ):
        raise PingstoreError("retained import selection differs")
    mappings = proof.get("files", [])
    if len(mappings) != 199 or len({m["source"] for m in mappings}) != 199:
        raise PingstoreError("incomplete retained source mapping")
    for item in mappings:
        target = run.directory / item["target"]
        if not target.is_relative_to(run.directory) or ".." in target.parts:
            raise PingstoreError("unsafe retained source mapping")
        if (
            target.stat().st_size != item["size_bytes"]
            or file_sha256(target) != item["sha256"]
        ):
            raise PingstoreError("retained source mapping checksum differs")
    old = load_json(
        run.directory / "provenance/archive/derived/artifacts/data/exp082/numbers.json"
    )
    rows = [
        aggregate(run.export / j["path"] / "condition.json", j, cfg)
        for j in recipe.jobs(cfg)
    ]

    def key(row):
        return row["duration_ms"], row["rate_hz"], row["seed"]

    if sorted(rows, key=key) != sorted(old["grid_per_seed"], key=key):
        raise PingstoreError("retained aggregate rows differ from source summary")


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
            aggregate(root / job["path"] / "condition.json", job, cfg)
        else:
            counts(root / job["path"] / "counts.npz", cfg)
    for name in ("matched", "variable"):
        stream(root, name)
