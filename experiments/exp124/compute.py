"""Train COBA/PING and record one matched held-out response with native SNNLab."""

import argparse
import gzip
import hashlib
import os
import shutil
import sys
import urllib.request
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import h5py
import numpy as np
import torch
from experiments.exp124 import recipe
from experiments.helpers.network_artifacts import snapshot_request
from pingstore.contracts import write_json_atomic
from pingstore.stages import stage_run
from snnlab import extensions
from snnlab.sim import (
    Batching,
    CheckpointSelection,
    DatasetEncoder,
    DatasetSnapshotBinding,
    DenseArrayBinding,
    ExecutionSpec,
    RecordingSpec,
    SignalRecording,
    TrainingMetrics,
    ValidationSpec,
    infer,
    resolve_device,
    train,
)

BASE = "https://zenkelab.org/datasets/"


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def select_indices(all_labels, cfg):
    """Stratified held-out split; full mode retains every official train item."""
    rng = np.random.default_rng(cfg["seed"])
    train, validation = [], []
    for label in range(20):
        available = np.flatnonzero(all_labels == label)
        selected = rng.permutation(available)
        held_out = max(1, round(len(available) * cfg["validation_fraction"]))
        n_train = len(available) - held_out
        if n_train < 1:
            raise ValueError("each class needs training and validation examples")
        if not cfg["full_dataset"]:
            selected = np.concatenate((selected[:1], selected[n_train : n_train + 1]))
            n_train = 1
        train.extend(selected[:n_train])
        validation.extend(selected[n_train:])
    return train, validation


def load_inputs(cfg, cache):
    cache.mkdir(parents=True, exist_ok=True)
    filename = "shd_train.h5.gz"
    expected = "d47c9825dee33347913e8ce0f2be08b0"  # Official archive, pinned in the retained evidence.
    archive = cache / filename
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        urllib.request.urlretrieve(BASE + filename, temporary)
        if digest(temporary, "md5") != expected:
            raise ValueError("SHD download checksum mismatch")
        temporary.rename(archive)
    if digest(archive, "md5") != expected:
        raise ValueError("cached SHD checksum mismatch")
    path = archive.with_suffix("")
    temporary = path.with_suffix(".unpack")
    with gzip.open(archive, "rb") as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target)
    temporary.replace(path)
    with h5py.File(path, "r") as handle:
        all_labels = np.asarray(handle["labels"], dtype=np.int64)
        train, validation = select_indices(all_labels, cfg)
        indices = np.asarray(train + validation, dtype=np.int64)
        offsets, channels, times = [0], [], []
        for index in indices:
            t = np.asarray(handle["spikes/times"][index], dtype=np.float64) * 1000
            ch = np.asarray(handle["spikes/units"][index], dtype=np.int64)
            if (
                not np.isfinite(t).all()
                or (t < 0).any()
                or (ch < 0).any()
                or (ch >= 700).any()
            ):
                raise ValueError("invalid SHD events")
            keep = t < cfg["duration_ms"]
            offsets.append(offsets[-1] + int(keep.sum()))
            channels.append(ch[keep])
            times.append(t[keep])
        labels = all_labels[indices]
        speakers = np.asarray(handle["extra/speaker"])[indices]
    cfg.update(
        train_examples=len(train),
        validation_examples=len(validation),
        train_class_counts=np.bincount(labels[: len(train)], minlength=20).tolist(),
        validation_class_counts=np.bincount(
            labels[len(train) :], minlength=20
        ).tolist(),
    )
    if cfg.get("full_dataset", False) and len(indices) != len(all_labels):
        raise ValueError(
            "full-data selection did not retain every official training sample"
        )
    arrays = dict(
        event_offsets=np.asarray(offsets),
        event_time_ms=np.concatenate(times),
        event_channel=np.concatenate(channels),
        labels=labels,
        official_train_indices=indices,
        speakers=speakers,
        split=np.asarray([0] * len(train) + [1] * len(validation)),
    )
    metadata = dict(
        url=BASE + filename,
        archive_md5=expected,
        archive_sha256=digest(archive),
        hdf5_sha256=digest(path),
        selection_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        train_count=len(train),
        validation_count=len(validation),
        official_train_count=len(all_labels),
        noise_seeds={
            "training": cfg["seed"] + 2,
            "validation_evaluation": cfg["seed"] + 4,
        },
        evaluation_noise="same corruption each epoch; matched across models",
        retained_events=int(sum(len(t) for t in times)),
        official_test_accessed=False,
    )
    return labels, arrays, metadata


def encode_shd(arrays, selected, *, dt_ms, channels, config, seed):
    """Bin ragged SHD events and apply the experiment's insertion/deletion noise."""
    spikes = np.zeros(
        (round(config["duration_ms"] / dt_ms), len(selected), channels),
        dtype=np.float32,
    )
    for column, local_index in enumerate(selected):
        index = int(local_index) + config["sample_offset"]
        start, end = arrays["event_offsets"][index : index + 2]
        steps = np.floor(arrays["event_time_ms"][start:end] / dt_ms).astype(np.int64)
        keep = steps < len(spikes)
        spikes[steps[keep], column, arrays["event_channel"][start:end][keep]] = 1
    noise = np.random.default_rng(seed).random(spikes.shape, dtype=np.float32)
    occupied = spikes != 0
    spikes[occupied & (noise < config["spike_deletion_probability"])] = 0
    spikes[~occupied & (noise < config["input_noise_hz"] * dt_ms / 1000)] = 1
    return torch.from_numpy(spikes)


extensions.register_encoder("exp124.shd_events/v1", encode_shd)


def train_condition(condition, run, cfg, dataset, device):
    bundle = recipe.author_network(cfg, condition)
    directory = run.export / condition
    directory.mkdir()
    checkpoint = run.scratch / f"{condition}.checkpoint"

    def binding(split):
        return DatasetSnapshotBinding(
            path=dataset,
            input_id="drive",
            dataset_id="shd",
            split=split,
            target_id="class",
            label_key=f"{split}_labels",
            encoder=DatasetEncoder(
                kind="custom",
                definition="exp124.shd_events/v1",
                seed=cfg["seed"] + (2 if split == "train" else 4),
                config={
                    "duration_ms": cfg["duration_ms"],
                    "sample_offset": 0 if split == "train" else cfg["train_examples"],
                    "input_noise_hz": cfg["input_noise_hz"],
                    "spike_deletion_probability": cfg["spike_deletion_probability"],
                },
            ),
        )

    def progress(report):
        print(
            f"{condition} epoch {report.epoch}: train={report.train.accuracy:.1%}; "
            f"validation={report.validation.accuracy:.1%}",
            flush=True,
        )

    result = train(
        snapshot_request(
            ExecutionSpec(
                kind="train",
                graph=bundle.graph,
                training=bundle.training,
                input_bindings=(binding("train"),),
                validation=ValidationSpec(input_bindings=(binding("validation"),)),
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                seed=cfg["seed"],
                device=device,
                metrics=TrainingMetrics(
                    train="online",
                    validation="epoch",
                    initial=("validation",),
                    update_history=False,
                ),
                batching=Batching(
                    order="stratified",
                    stratify_by="class",
                    remainder="merge_small",
                    min_batch_size=min(20, cfg["batch_size"]),
                ),
                checkpoint_selection=CheckpointSelection.disabled(),
                save_final_checkpoint=checkpoint,
                diagnostics=False,
                compile_mode="default" if device.startswith("cuda") else None,
                dataset_prefetch=1,
                training_timing=True,
                on_epoch_end=progress,
            ),
            bundle,
            directory / "train",
        )
    )
    # Retain native epoch measurements, without repeated input/optimizer diagnostics.
    fields = (
        "epoch",
        "train_cross_entropies",
        "train_accuracy",
        "train_measurement",
        "validation_cross_entropies",
        "validation_accuracy",
        "validation_measurement",
    )
    write_json_atomic(
        directory / "metrics.json",
        {
            "epochs": [
                {key: row.get(key) for key in fields}
                for row in result.metrics["epochs"]
            ]
        },
    )
    timing = {
        key: value
        for key, value in result.metrics["training_timing"].items()
        if key.endswith("_s")
    }
    run.record["execution"].setdefault("models", {})[condition] = {
        "duration_seconds": timing["total_s"] - timing["setup_s"],
        "native_timing": timing,
        "device": device,
    }
    return bundle, checkpoint


def record_sample(condition, bundle, checkpoint, drive, run, cfg):
    result = infer(
        snapshot_request(
            ExecutionSpec(
                kind="infer",
                graph=bundle.graph,
                checkpoint=checkpoint,
                input_bindings=(DenseArrayBinding("drive", drive),),
                recording=RecordingSpec(
                    signals=tuple(
                        SignalRecording(f"{pop.upper()}.spikes")
                        for pop in recipe.POPULATIONS[condition]
                    )
                ),
                seed=cfg["seed"],
                device="cpu",
                diagnostics=False,
                retain_parameters=False,
                retain_final_state=False,
                retain_runtime_state=False,
            ),
            bundle,
            run.export / f"{condition}--inference",
        )
    )
    return {
        f"{condition}_scores": result.outputs["class_scores"][0].cpu().numpy(),
        **{
            f"{condition}_{pop}": result.recorded_signals[f"{pop.upper()}.spikes"][:, 0]
            .cpu()
            .numpy()
            .astype(np.uint8)
            for pop in recipe.POPULATIONS[condition]
        },
    }


def compute(
    *,
    run_id=None,
    device=None,
    cache=None,
    cpu_threads=4,
    epochs=50,
    seed=42,
    batch_size=256,
    smoke=False,
):
    cfg = recipe.configuration(
        epochs=epochs, seed=seed, batch_size=batch_size, smoke=smoke
    )
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(cpu_threads)
    device = resolve_device(device or os.environ.get("PINGLAB_DEVICE", "cpu"))
    if str(device) == "cuda":
        device = f"cuda:{torch.cuda.current_device()}"
    device = str(device)
    labels, arrays, metadata = load_inputs(
        cfg, Path(cache or os.environ.get("SHD_CACHE", "/tmp/shd"))
    )
    arrays.update(
        train_labels=labels[: cfg["train_examples"]],
        validation_labels=labels[cfg["train_examples"] :],
    )
    selected = cfg["train_examples"]
    sample_cfg = dict(cfg, sample_offset=0)
    drive = encode_shd(
        arrays,
        [selected],
        dt_ms=cfg["dt_ms"],
        channels=700,
        config=sample_cfg,
        seed=seed + 4,
    )
    with stage_run(
        REPO, recipe.SLUG, "compute", run_id=run_id, configuration=cfg
    ) as run:
        dataset = run.scratch / "dataset.npz"
        np.savez_compressed(dataset, **arrays)
        np.savez_compressed(
            run.export / "selection.npz",
            **{
                key: arrays[key]
                for key in ("labels", "split", "official_train_indices", "speakers")
            },
        )
        run.record["execution"].update(
            dataset=metadata,
            snnlab_version=version("snnlab"),
            sample_raster_configuration={
                "sample_selection": "first validation utterance in source order; not selected by prediction or activity",
                "sample_class": int(labels[selected]),
                "official_train_index": int(arrays["official_train_indices"][selected]),
                "inference_noise_seed": seed + 4,
                "duration_ms": cfg["duration_ms"],
                "dt_ms": cfg["dt_ms"],
            },
        )
        recordings = {"input_spikes": drive[:, 0].numpy().astype(np.uint8)}
        for condition in recipe.MODELS:
            bundle, checkpoint = train_condition(condition, run, cfg, dataset, device)
            recordings.update(
                record_sample(condition, bundle, checkpoint, drive, run, cfg)
            )
        np.savez_compressed(run.export / "recording.npz", **recordings)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--device")
    parser.add_argument("--cache")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=int(os.environ.get("PINGLAB_CPU_THREADS", "4")),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One training and validation utterance per class; use --epochs 1 for a quick check",
    )
    compute(**vars(parser.parse_args()))
