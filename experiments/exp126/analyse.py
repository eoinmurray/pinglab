"""Validate one explicit compute source and summarize accuracy and timing."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp126.recipe import CONFIG, SLUG
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def analyse(identity):
    source = source_run(REPO / ".pingstore", identity, stage="compute", experiment=SLUG)
    cfg = source.record["execution"]["configuration"]
    if cfg != CONFIG:
        raise ValueError("compute source differs from the committed scientific recipe")
    learning = load_json(source.file("learning.json"))
    rows = learning["epochs"]
    if [row["epoch"] for row in rows] != list(range(cfg["epochs"] + 1)):
        raise ValueError("incomplete epoch history")
    with np.load(source.file("dataset.npz"), allow_pickle=False) as data:
        train, validation = data["train_indices"], data["validation_indices"]
        labels = data["official_train_labels"]
        if labels.shape != (60000,) or data["official_test_labels"].shape != (10000,):
            raise ValueError("unexpected MNIST split sizes")
        combined = np.concatenate((train, validation))
        if (
            len(np.unique(combined)) != len(combined)
            or combined.min() < 0
            or combined.max() >= len(labels)
        ):
            raise ValueError("invalid or overlapping training/validation indices")
        for name, indices, count in (
            ("train", train, cfg["train_per_class"]),
            ("validation", validation, cfg["validation_per_class"]),
        ):
            if not np.array_equal(
                np.bincount(labels[indices], minlength=10), np.full(10, count)
            ):
                raise ValueError("partition is not class balanced")
            if data[f"{name}_pixels"].shape != (len(indices), 28, 28):
                raise ValueError("partition images do not match selected indices")
        if data["test_pixels"].shape != (10000, 28, 28):
            raise ValueError("unexpected test images")
        test_labels = data["official_test_labels"].copy()
    counts = dict(train=len(train), validation=len(validation), test=len(test_labels))
    for row in rows:
        for split in ("train", "validation"):
            metric = row[split]
            if (
                metric["samples"] != counts[split]
                or not np.isfinite(metric["loss"])
                or metric["loss"] < 0
                or not 0 <= metric["accuracy"] <= 1
            ):
                raise ValueError("invalid learning metrics")
    selected = min(rows[1:], key=lambda row: row["validation"]["loss"])["epoch"]
    if learning["selected_epoch"] != selected:
        raise ValueError("checkpoint selection disagrees with validation loss")
    with np.load(source.file("test_predictions.npz"), allow_pickle=False) as test:
        predicted = test["predictions"]
        if (
            not np.array_equal(test["indices"], np.arange(10000))
            or not np.array_equal(test["labels"], test_labels)
            or predicted.shape != (10000,)
            or not np.issubdtype(predicted.dtype, np.integer)
            or (predicted < 0).any()
            or (predicted >= 10).any()
        ):
            raise ValueError("invalid official-test predictions")
        accuracy = float(np.mean(predicted == test_labels))
        if (
            abs(accuracy - learning["test"]["accuracy"]) > 1e-12
            or learning["test"]["samples"] != 10000
            or not np.isfinite(learning["test"]["loss"])
        ):
            raise ValueError("test metrics disagree with predictions")
        confusion = np.zeros((10, 10), dtype=np.int64)
        np.add.at(confusion, (test_labels, predicted), 1)
    with (
        np.load(source.file("initial_weights.npz"), allow_pickle=False) as initial,
        np.load(source.file("selected_weights.npz"), allow_pickle=False) as saved,
    ):
        if set(initial.files) != set(saved.files):
            raise ValueError("initial and selected model parameters differ")
        if any(not np.isfinite(saved[key]).all() for key in saved.files):
            raise ValueError("nonfinite selected parameters")
        changes = {
            key: float(np.linalg.norm(saved[key] - initial[key]))
            for key in ("input.weight", "output.weight")
        }
    execution = source.record["execution"]
    timing = execution["timing"]
    epochs = timing["epoch_timings"]
    seconds = np.asarray(timing["inference_seconds"], dtype=float)
    if (
        [row["epoch"] for row in epochs] != list(range(cfg["epochs"] + 1))
        or seconds.shape != (cfg["inference_repeats"],)
        or not np.isfinite(seconds).all()
        or (seconds <= 0).any()
    ):
        raise ValueError("invalid timing protocol")
    for row in epochs:
        for key in ("optimization_seconds", "evaluation_seconds", "checkpoint_seconds"):
            if not np.isfinite(row[key]) or row[key] < 0:
                raise ValueError("invalid epoch timing")
    median = float(np.median(seconds))
    summary = dict(
        test_inference_median_seconds=median,
        test_inference_min_seconds=float(seconds.min()),
        test_inference_max_seconds=float(seconds.max()),
        test_images_per_second=counts["test"] / median,
        test_amortized_ms_per_image=1000 * median / counts["test"],
    )
    result = dict(
        **learning,
        split_counts=counts,
        confusion_matrix=confusion.tolist(),
        weight_change_l2=changes,
        configuration=cfg,
    )
    with stage_run(
        REPO, SLUG, "analyse", inputs={"compute": source}, configuration=cfg
    ) as run:
        write_json_atomic(run.export / "results.json", result)
        # Benchmark summaries and device metadata are projections of compute provenance.
        run.record["execution"]["benchmark"] = {
            key: execution[key]
            for key in (
                "device",
                "device_name",
                "cpu_threads",
                "machine",
                "versions",
                "timing",
            )
        }
        run.record["execution"]["benchmark"].update(summary)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="one explicit full compute run ID; assemble comparison phases first",
    )
    identity = parser.parse_args().source
    first = source_run(REPO / ".pingstore", identity, stage="compute", experiment=SLUG)
    if (
        first.record["execution"]["configuration"].get("schema")
        == "exp126.current-profile/v1"
    ):
        from experiments.exp126.compute_parts.current_profiling import (
            analyse as current_analyse,
        )

        current_analyse(identity)
    elif (
        first.record["execution"]["configuration"].get("schema") == "exp126.profile/v1"
    ):
        from experiments.exp126.compute_parts.profiling import (
            analyse as profile_analyse,
        )

        profile_analyse(identity)
    elif (
        first.record["execution"]["configuration"].get("schema")
        == "exp126.assembled/v1"
    ):
        from experiments.exp126.compute_parts.comparison_analysis import compare

        compare(identity)
    else:
        analyse(identity)
