"""Validate paired SHD measurements and extract the executed model settings.

Run `uv run python experiments/exp124/analyse.py --source <compute-run-id>`
against an explicit, validated Pingstore compute run. Split inventories, matched
initialization, finite learned weights and unchanged feedback are checked before
interpreting native epoch measurements and one held-out diagnostic response.

Native snapshots identify the learned and fixed parameters; historical v3/v4
evidence retains its original bundle/NPZ readers. Analysis saves measurements
and raster autocorrelograms in a new immutable run. The report reads model
settings directly from the compute snapshots; this stage does not format tables,
acquire data, train models or invoke presentation.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
from experiments.exp124 import recipe
from pingstore.library import load_json, source_run, stage_run, write_json_atomic
from snnlab.analysis import spike_autocorrelogram
from snnlab.sim import NetworkSnapshot, TrainingHistory


def parameters(source, model, stage):
    snapshot_path = source.file(model, "train", stage, "snapshot.json")
    if snapshot_path.exists():
        snapshot = NetworkSnapshot.read(snapshot_path)
        return {
            name: value.numpy() for name, value in snapshot.load_parameters().items()
        }
    # The report's immutable r149 compute predates native snapshot retention.
    filename = "initial_weights.npz" if stage == "start" else "final_weights.npz"
    with np.load(source.file(model, filename), allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def network_definition(source, model, role):
    snapshot_path = source.file(model, "train", "end", "snapshot.json")
    if snapshot_path.exists():
        snapshot = NetworkSnapshot.read(snapshot_path)
        if snapshot.stage != "end" or snapshot.execution.kind != "train":
            raise ValueError("expected a native final-training snapshot")
        metadata = load_json(snapshot_path)
        return metadata["network" if role == "graph.json" else "training"]
    native = source.file(model, "train", "network.bundle", role)
    return load_json(native if native.exists() else source.file(model, role))


def recorded_populations(source):
    """Use saved recording choices, or the executed spiking graph for older runs."""
    cfg = source.record["execution"]["configuration"]
    if "recording" in cfg:
        return {model: cfg["recording"][model] for model in cfg["models"]}
    return {
        model: [
            population["id"].lower()
            for population in network_definition(source, model, "graph.json")[
                "populations"
            ]
            if population["spiking"]
        ]
        for model in cfg["models"]
    }


def trainable_parameters(source, model):
    """Read native resolved membership instead of maintaining a second inventory."""
    path = source.file(model, "train", "end", "snapshot.json")
    if path.exists():
        info = NetworkSnapshot.read(path).parameter_info
        if any(parameter.trainable is None for parameter in info.values()):
            raise ValueError("snapshot lacks resolved trainable parameter membership")
        return {name for name, parameter in info.items() if parameter.trainable}
    # Pre-snapshot evidence retains the original compiled training specification.
    training = network_definition(source, model, "training.json")
    if "resolved_parameters" in training:
        return set(training["resolved_parameters"]["trainable"])
    return {
        name
        for group in training["parameter_groups"]
        if not group["frozen"]
        for name in group["parameters"]
    }


def spike_autocorrelation(spikes, dt_ms, *, max_lag_ms=200, bin_ms=1.0):
    """Serialize SNNLab's population autocorrelogram; undefined bins are null."""
    lags, values = spike_autocorrelogram(
        spikes, dt_ms, max_lag_ms=max_lag_ms, bin_ms=bin_ms
    )
    return {
        "lag_ms": lags.tolist(),
        "correlation": [
            float(value) if np.isfinite(value) else None for value in values
        ],
    }


def raster_measurements(source):
    cfg = source.record["execution"]["sample_raster_configuration"]
    settings = source.record["execution"]["configuration"]
    measured = {
        "selection": cfg["sample_selection"],
        "class": cfg["sample_class"],
        "official_train_index": cfg["official_train_index"],
        "noise_seed": cfg["inference_noise_seed"],
        "duration_ms": cfg["duration_ms"],
        "dt_ms": cfg["dt_ms"],
        "models": {},
    }
    filename = (
        "recording.npz" if source.file("recording.npz").exists() else "rasters.npz"
    )
    with np.load(source.file(filename), allow_pickle=False) as arrays:
        steps = round(cfg["duration_ms"] / cfg["dt_ms"])
        for model, populations in recorded_populations(source).items():
            graph = network_definition(source, model, "graph.json")
            sizes = {p["id"].lower(): p["size"] for p in graph["populations"]}
            scores = arrays[f"{model}_scores"]
            if scores.shape != (sizes["readout"],) or not np.isfinite(scores).all():
                raise ValueError("invalid sample scores")
            autocorrelations = {}
            for pop in populations:
                spikes = arrays[f"{model}_{pop}"]
                if (
                    spikes.shape != (steps, sizes[pop])
                    or not np.isin(spikes, (0, 1)).all()
                ):
                    raise ValueError("invalid sample spikes")
                autocorrelations[pop] = spike_autocorrelation(
                    spikes,
                    cfg["dt_ms"],
                    bin_ms=settings.get("autocorrelation_bin_ms", 1.0),
                    max_lag_ms=settings.get("autocorrelation_max_lag_ms", 200.0),
                )
            prediction = int(scores.argmax())
            measured["models"][model] = {
                "prediction": prediction,
                "correct": prediction == cfg["sample_class"],
                "autocorrelations": autocorrelations,
            }
    return measured


def _validate_dataset_selection(*, source, cfg):
    """Validate the retained dataset partitions and raster selection."""
    selection = (
        "selection.npz" if source.file("selection.npz").exists() else "spikes.npz"
    )
    with np.load(source.file(selection), allow_pickle=False) as data:
        labels, split, indices = (
            data[key] for key in ("labels", "split", "official_train_indices")
        )
        if len(np.unique(indices)) != len(labels) or not np.isin(split, (0, 1)).all():
            raise ValueError("invalid or overlapping dataset selection")
        for code, name in enumerate(("train", "validation")):
            if not np.array_equal(
                np.bincount(labels[split == code], minlength=20),
                cfg[f"{name}_class_counts"],
            ):
                raise ValueError(
                    "dataset class counts differ from executed configuration"
                )
        if (
            cfg["full_dataset"]
            and len(labels)
            != source.record["execution"]["dataset"]["official_train_count"]
        ):
            raise ValueError("incomplete full dataset")
        sample = source.record["execution"]["sample_raster_configuration"]
        first = np.flatnonzero(split == 1)[0]
        if (sample["official_train_index"], sample["sample_class"]) != (
            indices[first],
            labels[first],
        ):
            raise ValueError("raster is not the first validation utterance")


def measurements(source):
    cfg = source.record["execution"]["configuration"]
    if (
        cfg.get("schema")
        not in (
            "exp124.recipe/v3",
            "exp124.recipe/v4",
            "exp124.recipe/v5",
        )
        or len(cfg["models"]) != 2
        or len(set(cfg["models"])) != 2
    ):
        raise ValueError("expected complete paired calibrated SHD training")
    _validate_dataset_selection(source=source, cfg=cfg)
    conditions, initial, native_metrics, trainable = {}, {}, {}, {}
    for model in cfg["models"]:
        start, end = (
            parameters(source, model, "start"),
            parameters(source, model, "end"),
        )
        if set(start) != set(end) or any(
            not np.isfinite(v).all() for v in end.values()
        ):
            raise ValueError("invalid final parameter inventory")
        trainable[model] = trainable_parameters(source, model)
        if not trainable[model] or not trainable[model] <= start.keys():
            raise ValueError("trainable parameter membership differs from saved arrays")
        changes = {
            key: float(np.linalg.norm(end[key] - start[key]))
            for key in sorted(trainable[model])
        }
        if not all(changes.values()) or any(
            not np.array_equal(start[key], end[key])
            for key in start
            if key not in trainable[model]
        ):
            raise ValueError(
                "learned weights did not change or fixed parameters changed"
            )
        initial[model] = start
        filename = (
            "metrics.json"
            if source.file(model, "metrics.json").exists()
            else "native_metrics.json"
        )
        metrics = load_json(source.file(model, filename))
        history = TrainingHistory.from_metrics(metrics)
        history.validate(expected_epochs=cfg["epochs"], require_complete=True)
        native_metrics[model] = metrics
        if [epoch.epoch for epoch in history.epochs] != list(range(cfg["epochs"] + 1)):
            raise ValueError("incomplete epoch history")
        curves = history.curves(metrics=("cross_entropy", "accuracy"))
        rows = [
            {
                "epoch": epoch.epoch,
                **{
                    split: {
                        name: float(curves[split, metric].values[index])
                        if np.isfinite(curves[split, metric].values[index])
                        else None
                        for name, metric in (
                            ("loss", "cross_entropy"),
                            ("accuracy", "accuracy"),
                        )
                    }
                    for split in ("train", "validation")
                },
            }
            for index, epoch in enumerate(history.epochs)
        ]
        for row in rows:
            for split in ("train", "validation"):
                loss, accuracy = row[split].values()
                if split == "train" and row["epoch"] == 0:
                    continue
                if (
                    loss is None
                    or accuracy is None
                    or not np.isfinite(loss)
                    or not 0 <= accuracy <= 1
                ):
                    raise ValueError("invalid epoch metrics")
        conditions[model] = {
            "epochs": rows,
            "weight_change_l2": changes,
        }
    baseline = cfg["models"][0]
    for model in cfg["models"][1:]:
        if trainable[model] != trainable[baseline]:
            raise ValueError("paired trainable parameter memberships differ")
        if any(
            not np.array_equal(initial[baseline][key], initial[model][key])
            for key in trainable[baseline]
        ):
            raise ValueError("paired initial learned weights differ")
    return {
        "conditions": conditions,
        "training_metrics": native_metrics,
    }


def analyse(identity):
    source = source_run(
        REPO / ".pingstore",
        identity,
        stage="compute",
        experiment=recipe.configuration()["experiment"],
    )
    results, sample = measurements(source), raster_measurements(source)
    execution = source.record["execution"]
    with stage_run(
        REPO,
        source.record["experiment"],
        "analyse",
        inputs={"compute": source},
        configuration=execution["configuration"],
    ) as run:
        write_json_atomic(run.export / "results.json", results)
        write_json_atomic(run.export / "raster_measurements.json", sample)
        run.record["execution"]["source_measurements"] = {"models": execution["models"]}
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    analyse(parser.parse_args().source)
