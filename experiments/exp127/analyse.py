"""Validate recordings and measure the before/after classifier comparison.

The CLI requires an explicit completed compute run. It validates the retained
recordings and writes measurements to a new analysis run, without simulating
or launching presentation.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp127.recipe import SLUG
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def measurements(spikes, labels, cfg):
    counts = spikes.sum(axis=0).astype(int)
    predictions = counts.argmax(axis=1)
    cumulative = spikes.cumsum(axis=0)
    eligible = (cumulative.max(axis=2) >= cfg["decision_min_spikes"]) & (
        np.abs(cumulative[:, :, 0] - cumulative[:, :, 1]) >= cfg["decision_min_lead"]
    )
    latency, decision_class = [], []
    for sample in range(len(labels)):
        steps = np.flatnonzero(eligible[:, sample])
        latency.append(float(steps[0] * cfg["dt_ms"]) if len(steps) else None)
        decision_class.append(
            int(cumulative[steps[0], sample].argmax()) if len(steps) else None
        )
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "samples": len(labels),
        "predictions": predictions.tolist(),
        "counts": counts.tolist(),
        "output_spikes": counts.sum(axis=1).tolist(),
        "latency_ms": latency,
        "decision_class": decision_class,
    }


def analyse(identity):
    source = source_run(REPO / ".pingstore", identity, stage="compute", experiment=SLUG)
    cfg = source.record["execution"]["configuration"]
    if cfg.get("schema") != "exp127.learning/v1":
        raise ValueError("unsupported recorded learning protocol")
    learning = load_json(source.file("learning.json"))
    epochs = learning["epochs"]
    if [row["epoch"] for row in epochs] != list(range(cfg["epochs"] + 1)):
        raise ValueError("incomplete learning history")
    expected = min(epochs, key=lambda row: row["validation_loss"])["epoch"]
    if learning["selection"]["epoch"] != expected:
        raise ValueError("checkpoint is not the earliest minimum-validation-loss epoch")
    with (
        np.load(source.file("dataset.npz")) as data,
        np.load(source.file("recording.npz")) as recording,
    ):
        labels = data["test_labels"]
        for name in ("train", "validation", "test"):
            inputs = data[f"{name}_spikes"]
            if (
                inputs.shape != (200, cfg[f"{name}_samples"], 3)
                or not np.isin(inputs, [0, 1]).all()
            ):
                raise ValueError("input dimensions or binary occupancy are invalid")
            if not np.array_equal(inputs, data[f"{name}_poisson_counts"] > 0):
                raise ValueError("Poisson encoding differs from recorded counts")
        results = {}
        for mode in ("before", "after"):
            spikes = recording[f"{mode}_output_spikes"]
            if (
                spikes.shape != (200, cfg["test_samples"], 2)
                or not np.isin(spikes, [0, 1]).all()
            ):
                raise ValueError("invalid output spike recording")
            if not np.array_equal(spikes.sum(axis=0), recording[f"{mode}_scores"]):
                raise ValueError(
                    "classification scores disagree with the spike readout"
                )
            weights = recording[f"{mode}_weights"]
            if (
                weights.shape != (3, 2)
                or not np.isfinite(weights).all()
                or (weights < 0).any()
            ):
                raise ValueError("invalid weights or source/target orientation")
            results[mode] = measurements(spikes, labels, cfg)
        change = float(
            np.linalg.norm(recording["after_weights"] - recording["before_weights"])
        )
        if change == 0:
            raise ValueError("no synaptic weights learned")
        examples = [int(np.flatnonzero(labels == cls)[0]) for cls in (0, 1)]
    with np.load(source.file("training_trace.npz")) as trace:
        losses, weights, gradients = (
            trace["losses"],
            trace["weights"],
            trace["gradients"],
        )
        updates = cfg["trace_updates"]
        if (
            losses.shape != (updates,)
            or weights.shape != (updates + 1, 3, 2)
            or gradients.shape != (updates, 3, 2)
        ):
            raise ValueError("invalid training trace axes")
        if not all(
            np.isfinite(value).all()
            for value in (losses, weights, gradients, trace["voltages"])
        ):
            raise ValueError("nonfinite training trace")
        if (
            (weights < 0).any()
            or not np.count_nonzero(gradients)
            or losses[-1] >= losses[0]
        ):
            raise ValueError("teaching loop did not learn")
        if (
            trace["outputs"].shape != (updates, 200, 2, 2)
            or not np.isin(trace["outputs"], [0, 1]).all()
        ):
            raise ValueError("invalid training output spikes")
        with np.load(source.file("dataset.npz")) as dataset:
            targets = dataset["train_labels"][trace["indices"]]
            if not np.array_equal(targets, [0, 1]):
                raise ValueError("training examples must cover both classes")
        window = cfg["trace_window"]
        start = int(np.argmax(losses[: -window + 1] - losses[window - 1 :]))
        training_summary = {
            "losses": losses.tolist(),
            "window_start": start,
            "window_length": window,
            "samples": cfg["train_samples"],
            "protocol": "separate fixed-batch SNNLab teaching loop",
        }
    numbers = {
        "schema": "exp127.learning-metrics/v1",
        "configuration": cfg,
        "epochs": epochs,
        "selected_epoch": expected,
        "test": results,
        "example_indices": examples,
        "weight_change_l2": change,
        "training_trace": training_summary,
    }
    with stage_run(
        REPO, SLUG, "analyse", inputs={"compute": source}, configuration=cfg
    ) as run:
        write_json_atomic(run.export / "numbers.json", numbers)
        run.record["execution"]["source_measurements"] = source.record["execution"][
            "versions"
        ]
    print(run.run_id)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    analyse(parser.parse_args().source)
