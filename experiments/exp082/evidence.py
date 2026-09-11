"""Fail-closed validation of current and retained scientific evidence."""

import math
from typing import Any, cast

import numpy as np
from experiments.exp022.checkpoints import public_provenance, resolve_checkpoint
from pingstore.contracts import PingstoreError, load_json
from pingstore.layout import canonical_export_file, canonical_export_unit

from . import recipe

RETAINED_SCHEMA = "exp082.gold2-import/v1"
RETAINED_PRODUCER = "73f0883edc14aa634f5a6d55e4f4123fbfeb7508"


def _file(root, *parts):
    return (
        root.file(*parts)
        if hasattr(root, "file")
        else canonical_export_file(root, *parts)
    )


def _unit(root, *parts):
    return (
        root.unit(*parts)
        if hasattr(root, "unit")
        else canonical_export_unit(root, *parts)
    )


def _root(root):
    return root.outputs if hasattr(root, "outputs") else root


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
        or cfg != recipe.configuration(version=1)
    ):
        raise PingstoreError("aggregate evidence lacks its retained import contract")
    if record.get("source_files") != 199 or record.get("source_bytes") != 6079619:
        raise PingstoreError("retained import selection differs")
    if (
        record.get("scientific_files") != 135
        or record.get("archived_metadata_files") != 64
    ):
        raise PingstoreError("retained source mapping partition differs")
    numbers_path = run.export / "historical-summary.json"
    old = load_json(numbers_path)
    rows = [
        aggregate(run.file(j["path"], "condition.json"), j, cfg)
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


def validate_image_stream_bank(value, cfg):
    """Validate the single retained image plan for a v3 evaluation run."""
    if not isinstance(value, dict) or set(value) != {
        "policy",
        "sampling_seed",
        "indices",
        "labels",
        "dataset",
    }:
        raise PingstoreError("image-stream bank record differs")
    if (
        value["policy"] != cfg.get("image_stream_policy")
        or value["sampling_seed"] != cfg.get("image_sampling_seed")
    ):
        raise PingstoreError("image-stream bank policy differs")
    dataset = value["dataset"]
    if not isinstance(dataset, dict) or set(dataset) != {
        "partition",
        "images",
        "labels",
    }:
        raise PingstoreError("image-stream dataset record differs")
    images, dataset_labels = dataset["images"], dataset["labels"]
    if (
        dataset["partition"] != "official_mnist_test"
        or not isinstance(images, dict)
        or not isinstance(dataset_labels, dict)
        or set(images) != {"shape", "dtype", "sha256"}
        or set(dataset_labels) != {"shape", "dtype", "sha256"}
        or images.get("shape") != [10_000, recipe.N_INPUT]
        or dataset_labels.get("shape") != [10_000]
        or images.get("dtype") != "float32"
        or dataset_labels.get("dtype") != "int64"
        or any(
            not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
            for record in (images, dataset_labels)
        )
    ):
        raise PingstoreError("image-stream dataset identity differs")
    indices, labels = value["indices"], value["labels"]
    shape = (cfg["streams_per_cell"], cfg["digits_per_stream"])
    if (
        not isinstance(indices, list)
        or not isinstance(labels, list)
        or len(indices) != shape[0]
        or len(labels) != shape[0]
    ):
        raise PingstoreError("image-stream bank dimensions differ")
    for index_row, label_row in zip(indices, labels, strict=True):
        if (
            not isinstance(index_row, list)
            or not isinstance(label_row, list)
            or len(index_row) != shape[1]
            or len(label_row) != shape[1]
            or len(set(index_row)) != len(index_row)
            or any(
                type(index) is not int or not 0 <= index < 10_000
                for index in index_row
            )
            or any(type(label) is not int or not 0 <= label < 10 for label in label_row)
        ):
            raise PingstoreError("invalid image-stream bank values")
    return value


def stream(root, name, *, conditions=None):
    folder = _unit(root, "streams", name)
    meta = load_json(folder / "stream.json")
    raw = arrays(folder / "recording.npz")
    if conditions is None:
        conditions = (
            [[200.0, 5.0]] * 5
            if name == "matched"
            else [list(c) for c in recipe.VARIABLE_STREAM]
        )
    else:
        conditions = [list(value) for value in conditions]
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


def showcase_configuration(*, conditions=None, version=2):
    if version not in (1, 2):
        raise ValueError("unsupported exp082 showcase recipe version")
    return {
        "schema": f"exp082.showcase-selection/v{version}",
        **(
            {
                "dt_ms": recipe.DT_MS,
                **recipe.refractory_execution_configuration(recipe.DT_MS),
            }
            if version >= 2 else {}
        ),
        "conditions": [
            list(value)
            for value in (
                recipe.SHOWCASE_CONDITIONS if conditions is None else conditions
            )
        ],
        "candidate_order": "ascending integer index",
        "digit_seed_base": recipe.SHOWCASE_DIGIT_SEED_BASE,
        "encoding_seed_base": recipe.SHOWCASE_ENCODING_SEED_BASE,
        "candidate_limit": recipe.SHOWCASE_CANDIDATE_LIMIT,
        "targets": recipe.SHOWCASE_TARGETS,
        "training_seed": recipe.SEEDS[0],
    }


def validate_showcase(root):
    saved = load_json(_root(root) / "evidence.json")
    selected = saved.get("selected")
    if (
        saved.get("schema") not in (
            "exp082.showcase-selection/v1", "exp082.showcase-selection/v2"
        )
        or saved.get("configuration")
        not in tuple(
            showcase_configuration(conditions=conditions, version=version)
            for version in (1, 2)
            for conditions in (
                recipe.SHOWCASE_CONDITIONS, recipe.FIXED_DURATION_SHOWCASE_CONDITIONS
            )
        )
        or saved["configuration"]["schema"] != saved["schema"]
        or not isinstance(selected, dict)
        or selected.keys() != recipe.SHOWCASE_TARGETS.keys()
    ):
        raise PingstoreError("showcase selection contract differs")
    candidate_values = saved.get("candidates")
    if not isinstance(candidate_values, list) or not candidate_values:
        raise PingstoreError("showcase candidate history is missing")
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(candidate_values):
        if not isinstance(row, dict):
            raise PingstoreError("invalid showcase candidate record")
        expected = {
            "candidate_index": index,
            "digit_seed": recipe.SHOWCASE_DIGIT_SEED_BASE + index,
            "encoding_seed": recipe.SHOWCASE_ENCODING_SEED_BASE + index,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise PingstoreError("showcase candidate order differs")
        labels, predictions, correct = (
            row.get("labels"),
            row.get("predictions"),
            row.get("correct"),
        )
        if (
            not isinstance(labels, list)
            or not isinstance(predictions, list)
            or not isinstance(correct, list)
            or not all(len(value) == 5 for value in (labels, predictions, correct))
            or any(
                type(value) is not int or not 0 <= value < 10
                for value in labels + predictions
            )
            or any(type(value) is not int or value not in (0, 1) for value in correct)
            or correct
            != [int(a == b) for a, b in zip(labels, predictions, strict=True)]
            or row.get("n_correct") != sum(correct)
        ):
            raise PingstoreError("invalid showcase candidate outcome")
        candidates.append(cast(dict[str, Any], row))
    for name, target in recipe.SHOWCASE_TARGETS.items():
        first = None
        for candidate in candidates:
            if candidate["n_correct"] == target:
                first = candidate["candidate_index"]
                break
        if type(first) is not int:
            raise PingstoreError("showcase target has no qualifying candidate")
        if selected.get(name) != first or first is None:
            raise PingstoreError(
                "showcase did not retain the first qualifying candidate"
            )
        raw, meta = stream(root, name, conditions=saved["configuration"]["conditions"])
        row = candidates[first]
        predictions = [
            int(raw["spikes_out"][start:stop].sum(axis=0).argmax())
            for start, stop in zip(
                meta["boundaries"][:-1], meta["boundaries"][1:], strict=True
            )
        ]
        if meta["labels"] != row["labels"] or predictions != row["predictions"]:
            raise PingstoreError("selected showcase recording differs from its outcome")
    return saved


def showcase_evidence(repo, run):
    if (
        run.record["stage"] != "compute"
        or run.record["experiment"] != recipe.SLUG
        or run.record["execution"].get("operation") != "showcase-selection"
        or set(run.record["inputs"]) != {"bank"}
    ):
        raise PingstoreError("invalid exp082 showcase compute run")
    pin = run.record["inputs"]["bank"]
    from . import inputs

    bank = inputs.source(
        repo, pin["run_id"], "compute", experiment="exp022", reference=pin
    )
    saved = validate_showcase(run.export)
    if run.record["execution"].get("configuration") != saved["configuration"]:
        raise PingstoreError("showcase execution configuration differs")
    if saved.get("training_contract") != training_contract(bank.export):
        raise PingstoreError("showcase training contract differs")
    return bank, saved


def condition(root, job, cfg):
    from . import measurements

    kind = load_json(_root(root) / "evidence.json").get("condition_evidence")
    if kind == "historical-aggregate/v1":
        return aggregate(_file(root, job["path"], "condition.json"), job, cfg)
    return measurements.condition_row(
        job, counts(_file(root, job["path"], "counts.npz"), cfg), cfg
    )


def validate_compute(root, cfg, *, historical=False):
    expected = {j["id"] for j in recipe.jobs(cfg)}
    base = _root(root)
    role = "condition.json" if historical else "counts.npz"
    suffix = f"--{role}"
    if (base / "jobs").is_dir():
        units = {path.name for path in (base / "jobs").iterdir() if path.is_dir()}
    else:
        units = {
            path.name.removeprefix("jobs--").removesuffix(suffix)
            for path in base.glob(f"jobs--*{suffix}")
            if path.is_file()
        }
    if units != expected:
        raise PingstoreError("incomplete or extra condition jobs")
    expected_labels = None
    if not historical and cfg["schema"] == "exp082.recipe/v3":
        saved = load_json(base / "evidence.json")
        expected_labels = np.asarray(
            validate_image_stream_bank(saved.get("image_stream_bank"), cfg)["labels"]
        )
    for job in recipe.jobs(cfg):
        if historical:
            aggregate(_file(root, job["path"], "condition.json"), job, cfg)
        else:
            retained = counts(_file(root, job["path"], "counts.npz"), cfg)
            if expected_labels is not None and not np.array_equal(
                retained["labels"], expected_labels
            ):
                raise PingstoreError("condition labels differ from shared image bank")
    for name in ("matched", "variable"):
        stream(root, name)
