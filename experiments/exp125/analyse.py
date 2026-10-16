"""Validate and measure one explicitly named Scikit Digits compute run."""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
from experiments.exp125.recipe import SLUG
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def analyse(identity):
    source = source_run(REPO / ".pingstore", identity, stage="compute", experiment=SLUG)
    cfg = source.record["execution"]["configuration"]
    if cfg["schema"] != "exp125.recipe/v1":
        raise ValueError("unsupported scientific recipe")
    learning = load_json(source.file("learning.json"))
    rows = learning["epochs"]
    if [row["epoch"] for row in rows] != list(range(cfg["epochs"] + 1)):
        raise ValueError("incomplete epoch history")
    with np.load(source.file("dataset.npz")) as data:
        partitions = [data[f"{name}_indices"] for name in ("train", "validation", "test")]
        joined = np.concatenate(partitions)
        if not np.array_equal(np.sort(joined), np.arange(len(data["labels"]))):
            raise ValueError("partitions overlap or omit examples")
        split_counts = {name: len(part) for name, part in zip(("train", "validation", "test"), partitions, strict=True)}
        for name, part in zip(("train", "validation", "test"), partitions, strict=True):
            if len(np.unique(data["labels"][part])) != cfg["outputs"]:
                raise ValueError("a split omits classes")
    for row in rows:
        for split in ("train", "validation"):
            metrics = row[split]
            if metrics["samples"] != split_counts[split] or not np.isfinite(metrics["loss"]) or not 0 <= metrics["accuracy"] <= 1:
                raise ValueError("invalid learning metrics")
    expected_epoch = min(rows, key=lambda row: row["validation"]["loss"])["epoch"]
    if expected_epoch != learning["selected_epoch"]:
        raise ValueError("checkpoint selection does not match validation loss")
    with np.load(source.file("test_predictions.npz")) as test:
        if not np.array_equal(test["indices"], partitions[2]):
            raise ValueError("test predictions do not match held-out partition")
        accuracy = float(np.mean(test["predictions"] == test["labels"]))
        if accuracy != learning["test"]["accuracy"]:
            raise ValueError("test accuracy disagrees with predictions")
        confusion = np.zeros((cfg["outputs"], cfg["outputs"]), dtype=np.int64)
        np.add.at(confusion, (test["labels"], test["predictions"]), 1)
    initial_path = source.file("training", "start", "parameters.npz")
    if not initial_path.exists():
        initial_path = source.file("initial_weights.npz")
    with np.load(initial_path) as initial, np.load(source.file("selected_weights.npz")) as selected:
        changes = {key: float(np.linalg.norm(selected[key] - initial[key])) for key in ("input.weight", "output.weight")}
        if not all(np.isfinite(selected[key]).all() for key in selected.files):
            raise ValueError("nonfinite selected weights")
    results = {**learning, "split_counts": split_counts, "weight_change_l2": changes,
               "confusion_matrix": confusion.tolist(), "configuration": cfg}
    with stage_run(REPO, SLUG, "analyse", inputs={"compute": source}, configuration=cfg) as run:
        write_json_atomic(run.export / "results.json", results)
        run.record["execution"]["source_measurements"] = {key: source.record["execution"][key] for key in
            ("device", "cpu_threads", "machine", "training_seconds", "test_inference_seconds", "training_and_test_seconds", "versions", "dataset")}
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    analyse(parser.parse_args().source)
