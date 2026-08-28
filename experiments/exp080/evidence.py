"""Validate retained decoder evidence without importing torch or simulating."""

import math
import re

import numpy as np
from pingstore.contracts import PingstoreError, file_sha256, load_json


def require(condition, message):
    if not condition:
        raise PingstoreError(f"exp080 evidence: {message}")


def dataset(record, count, names):
    require(isinstance(record, dict), "missing dataset provenance")
    require(
        record.get("image_shape") == [count, 28, 28]
        and record.get("label_shape") == [count],
        "dataset dimensions disagree",
    )
    hashes = record.get("raw_sha256", {})
    require(
        set(hashes) == set(names)
        and all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in hashes.values()),
        "missing raw MNIST source hashes",
    )


def illustration(output, record, cfg, *, historical=False):
    if record == {"kind": "historical-image", "path": "feature_images.png"}:
        from PIL import Image

        require(
            historical, "historical illustration needs an explicit import operation"
        )
        require(
            (output / "feature_images.png")
            .read_bytes()
            .startswith(b"\x89PNG\r\n\x1a\n"),
            "historical illustration is not a PNG",
        )
        with Image.open(output / "feature_images.png") as image:
            image.verify()
        return
    require(
        record == {"kind": "samples", "path": "feature_samples.npz"},
        "unsupported illustration evidence",
    )
    with np.load(output / "feature_samples.npz", allow_pickle=False) as data:
        require(
            set(data.files) == {"image", "features_mV", "rates_hz"},
            "illustration array keys differ",
        )
        image, features = data["image"], data["features_mV"]
        require(
            image.shape == (28, 28) and image.dtype == np.uint8,
            "illustration image must retain its original uint8 pixels",
        )
        require(
            features.shape == (3, 784)
            and features.dtype == np.float32
            and np.isfinite(features).all()
            and (features >= 0).all()
            and (features <= 65).all(),
            "invalid illustrative feature samples",
        )
        require(
            np.array_equal(data["rates_hz"], cfg["illustration_rates_hz"]),
            "illustration rates disagree",
        )


def validate(output, cfg, *, historical=False, document=None):
    document = load_json(output / "evidence.json") if document is None else document
    require(
        document.get("schema") == "exp080.compute/v1" and document.get("recipe") == cfg,
        "compute schema or recipe disagrees",
    )
    records = document["training"]
    require([r["seed"] for r in records] == cfg["seeds"], "decoder seeds disagree")
    for record in records:
        if record.get("checkpoint_retention") == "memory_only":
            require(
                not historical,
                "historical imports must retain their checkpoint evidence",
            )
            require(
                "checkpoint" not in record and "checkpoint_sha256" not in record,
                "memory-only training must not declare a checkpoint file",
            )
            require(
                not (output / f"models/seed-{record['seed']}/decoder.pt").exists(),
                "memory-only training contains an unexpected checkpoint",
            )
        else:
            checkpoint = f"models/seed-{record['seed']}/decoder.pt"
            require(
                record.get("checkpoint") == checkpoint, "unexpected checkpoint path"
            )
            require(
                file_sha256(output / checkpoint) == record.get("checkpoint_sha256"),
                "checkpoint hash differs from selected training record",
            )
        training_path = output / f"models/seed-{record['seed']}/training.json"
        require(load_json(training_path) == record, "training record copies disagree")
        history = record["history"]
        require(
            [row["epoch"] for row in history] == list(range(1, cfg["epochs"] + 1)),
            "incomplete epoch history",
        )
        require(
            all(
                isinstance(row[key], (int, float))
                and math.isfinite(row[key])
                and 0 <= row[key] <= 1
                for row in history
                for key in ("train_accuracy", "validation_accuracy")
            ),
            "invalid training accuracy",
        )
        best = max(history, key=lambda row: row["validation_accuracy"])
        require(
            record["selected_epoch"] == best["epoch"]
            and record["selected_validation_accuracy"] == best["validation_accuracy"],
            "checkpoint is not the first validation maximum",
        )
    dataset(
        document["training_dataset"],
        60000,
        ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
    )
    dataset(
        document["evaluation"]["dataset"],
        cfg["test_count"],
        ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
    )
    arrays = output / "held_out_correctness.npz"
    require(
        file_sha256(arrays) == document["evaluation"]["arrays_sha256"],
        "held-out array checksum disagrees",
    )
    with np.load(arrays, allow_pickle=False) as data:
        require(
            set(data.files) == {"correctness", "rates_hz", "seeds", "labels"},
            "held-out array keys disagree",
        )
        correctness = data["correctness"]
        require(
            correctness.dtype == np.bool_
            and correctness.shape
            == (len(cfg["rates_hz"]), len(cfg["seeds"]), cfg["test_count"]),
            "wrong held-out correctness dtype or dimensions",
        )
        require(
            np.array_equal(data["rates_hz"], cfg["rates_hz"])
            and np.array_equal(data["seeds"], cfg["seeds"]),
            "held-out rate or seed ordering disagrees",
        )
        labels = data["labels"]
        require(
            labels.shape == (cfg["test_count"],)
            and labels.dtype == np.int64
            and (labels >= 0).all()
            and (labels <= 9).all(),
            "invalid held-out labels",
        )
    checks = document["simulator_validation"]
    require(
        checks.get("checks")
        == {
            "ampa_decay_in_unit_interval": True,
            "early_spike_exceeds_late_spike": True,
            "zero_input_feature_is_zero_by_construction": True,
            "timestep_count_exact": True,
        }
        and 65 >= checks["early_spike_mV"] > checks["late_spike_mV"] > 0,
        "simulator validation record disagrees",
    )
    illustration(output, document["illustration"], cfg, historical=historical)
    return document, correctness
