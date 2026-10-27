"""Train the paired SHD classifiers and retain native execution evidence.

Run `uv run python experiments/exp124/compute.py` for the full recipe or pass
`--protocol local-check` for one epoch with one training and validation utterance
per class. SNNLAB_DEVICE controls native device selection; SHD_CACHE selects the
verified archive cache. Pass --parallel to run one model per CUDA GPU.
Acquisition, training and replay happen only on invocation.

SNNLab owns batching, optimization, epoch measurements, recording and snapshots.
Each topology gets one complete bundle and circuit; training captures start/end
parameters and best-validation weights; CPU replay captures its start parameters.
Native metrics, selection provenance and one matched diagnostic response survive in Pingstore.
The temporary dataset and transfer checkpoint are discarded on successful completion.
Analysis and presentation require separate, explicitly selected runs.
"""

import argparse
import os
import sys
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
import torch
from experiments.exp124 import dataset, recipe
from hpc.concurrent_compute import run_active_workers
from pingstore.library import file_sha256, stage_run, write_json_atomic
from snnlab.sim import (
    Batching,
    CheckpointSelection,
    DatasetEncoder,
    DatasetSnapshotBinding,
    DenseArrayBinding,
    Performance,
    RecordingSpec,
    SignalRecording,
    SnapshotSpec,
    TrainingMetrics,
    ValidationSpec,
    resolve_device,
    simulate,
    train,
)


def train_condition(condition, run, cfg, dataset_path, drive, device):
    bundle = recipe.author_network(cfg, condition)
    directory = run.export / condition
    directory.mkdir()
    checkpoint = run.scratch / f"{condition}.checkpoint"
    selected_checkpoint = run.scratch / f"{condition}.best-checkpoint"

    def binding(split):
        return DatasetSnapshotBinding(
            path=dataset_path,
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

    # One authored bundle and one circuit per topology; no per-snapshot diagrams.
    bundle.write(directory / "train" / "network.bundle", visualise=False)
    bundle.visualise(directory / "circuit.svg", view="circuit")
    result = train(
        bundle=bundle,
        input_bindings=(binding("train"),),
        validation=ValidationSpec(input_bindings=(binding("validation"),)),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
        device=device,
        performance=Performance(**cfg["performance"]),
        metrics=TrainingMetrics(**cfg["training_metrics"]),
        batching=Batching(**cfg["batching"]),
        checkpoint_selection=CheckpointSelection.from_dict(
            cfg["best_checkpoint_selection"]
        ),
        save_selected_checkpoint=selected_checkpoint,
        save_final_checkpoint=checkpoint,
        diagnostics=False,
        training_timing=True,
        on_epoch_end=progress,
        # Start weights support the report's matched-initialization and weight-change checks.
        snapshot=SnapshotSpec(path=directory / "train", stage="both", parameters=True),
    )
    # Retain native resolved policies as well as the full epoch measurements.
    write_json_atomic(directory / "metrics.json", result.metrics)
    timing = {
        key: value
        for key, value in result.metrics["training_timing"].items()
        if key.endswith("_s")
    }
    run.record["execution"].setdefault("models", {})[condition] = {
        "duration_seconds": timing["total_s"] - timing["setup_s"],
        "native_timing": timing,
        "resolved_performance": result.metrics.get("performance"),
        "device": device,
    }
    # Preserve the validation-selected parameters without changing the report's
    # final-epoch diagnostic. SNNLab captures the restored checkpoint natively.
    simulate(
        bundle=bundle,
        checkpoint=selected_checkpoint,
        input_bindings=(DenseArrayBinding("drive", drive),),
        seed=cfg["seed"],
        device=cfg["sample_device"],
        performance=Performance(**cfg["performance"]),
        diagnostics=False,
        retain_parameters=False,
        retain_final_state=False,
        retain_runtime_state=False,
        snapshot=SnapshotSpec(
            path=directory / "best-validation" / "start",
            stage="start",
            parameters=True,
        ),
    )
    return bundle, checkpoint


def record_sample(condition, bundle, checkpoint, drive, run, cfg):
    result = simulate(
        bundle=bundle,
        checkpoint=checkpoint,
        input_bindings=(DenseArrayBinding("drive", drive),),
        recording=RecordingSpec(
            signals=tuple(
                SignalRecording(f"{pop.upper()}.spikes")
                for pop in cfg["recording"][condition]
            )
        ),
        seed=cfg["seed"],
        device=cfg["sample_device"],
        performance=Performance(**cfg["performance"]),
        diagnostics=False,
        retain_parameters=False,
        retain_final_state=False,
        retain_runtime_state=False,
        snapshot=SnapshotSpec(
            path=run.export / f"{condition}--inference" / "start",
            stage="start",
            parameters=True,
        ),
    )
    write_json_atomic(run.export / condition / "sample_metrics.json", result.metrics)
    sample = result.numpy(batch=0)
    # Retain native axes, cell IDs and sampling coordinates with binary exports.
    run.record["execution"].setdefault("recording_info", {})[condition] = {
        name: asdict(
            replace(info, array_dtype="uint8", conversion="binary_spikes_to_uint8")
        )
        for name, info in sample.recording.info.items()
    }
    return {
        f"{condition}_scores": sample.outputs["class_scores"],
        **{
            f"{condition}_{pop}": sample.recorded_signals[
                f"{pop.upper()}.spikes"
            ].astype(np.uint8)
            for pop in cfg["recording"][condition]
        },
    }


def compute(*, run_id=None, protocol="full"):
    """Run both models sequentially through the shared compute pipeline."""
    return _compute(run_id=run_id, protocol=protocol, parallel=False)


def compute_parallel(*, run_id=None, protocol="full"):
    """Run one spawned process per model/GPU into one coordinator-owned run."""
    return _compute(run_id=run_id, protocol=protocol, parallel=True)


def condition_worker(directory, item, index, cfg, dataset_path, drive, threads, digest):
    """Write one model's evidence and return its isolated execution metadata."""
    if file_sha256(dataset_path) != digest:
        raise ValueError("prepared dataset changed before model execution")
    device = item["device"]
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    torch.set_num_threads(threads)
    scratch = directory / ".scratch/shards" / str(index)
    scratch.mkdir(parents=True, exist_ok=True)
    child = SimpleNamespace(
        export=directory / "export", scratch=scratch, record={"execution": {}}
    )
    model = item["id"]
    bundle, checkpoint = train_condition(model, child, cfg, dataset_path, drive, device)
    recordings = record_sample(model, bundle, checkpoint, drive, child, cfg)
    np.savez_compressed(scratch / "recording.npz", **recordings)
    if file_sha256(dataset_path) != digest:
        raise ValueError("prepared dataset changed during model execution")
    return {"model": model, "execution": child.record["execution"]}


def worker_inventory(directory, item, index):
    """Hash this worker's disjoint native exports and temporary recordings."""
    paths = []
    for prefix in (item["id"], f"{item['id']}--inference"):
        root = directory / "export" / prefix
        if not root.is_dir():
            raise ValueError(f"missing model output: {prefix}")
        paths.extend(path for path in root.rglob("*") if path.is_file())
    paths.append(directory / ".scratch/shards" / str(index) / "recording.npz")
    return {
        str(path.relative_to(directory)): file_sha256(path) for path in sorted(paths)
    }


def run_conditions(run, cfg, dataset_path, drive, items, threads, parallel):
    """Schedule the same model worker sequentially or concurrently, then collect."""
    digest = file_sha256(dataset_path)
    arguments = (cfg, dataset_path, drive, threads, digest)
    if parallel:
        records = run_active_workers(
            run=run,
            items=items,
            worker=condition_worker,
            arguments=arguments,
            inventory=worker_inventory,
        )
        results = [record["result"] for record in records]
    else:
        results = [
            condition_worker(run.directory, item, index, *arguments)
            for index, item in enumerate(items)
        ]
    if file_sha256(dataset_path) != digest:
        raise ValueError("prepared dataset changed before collection")
    recordings = {}
    for index, (item, result) in enumerate(zip(items, results, strict=True)):
        model = item["id"]
        if result["model"] != model:
            raise ValueError("worker returned another model's metadata")
        for section in ("models", "recording_info"):
            metadata = result["execution"][section]
            if set(metadata) != {model}:
                raise ValueError("worker returned incomplete or overlapping metadata")
            run.record["execution"].setdefault(section, {}).update(metadata)
        with np.load(
            run.scratch / "shards" / str(index) / "recording.npz", allow_pickle=False
        ) as arrays:
            expected = {f"{model}_scores"} | {
                f"{model}_{pop}" for pop in cfg["recording"][model]
            }
            if set(arrays.files) != expected or recordings.keys() & expected:
                raise ValueError("worker recordings are incomplete or overlap")
            recordings.update({key: arrays[key] for key in arrays.files})
    run.record["execution"]["concurrency"] = {
        "mode": "spawned_processes" if parallel else "sequential",
        "work_items": items,
        "cpu_threads_per_worker": threads,
        "single_run_coordinator": True,
    }
    return recordings


def execution_items(cfg, device, cpu_threads, parallel):
    """Resolve execution allocation before loading data or opening a run."""
    if parallel:
        count = len(cfg["models"])
        if not device.startswith("cuda") or torch.cuda.device_count() < count:
            raise ValueError("parallel compute requires one visible CUDA GPU per model")
        if cpu_threads < count:
            raise ValueError(
                "parallel compute requires at least one CPU thread per model"
            )
        return [
            {"id": model, "device": f"cuda:{index}"}
            for index, model in enumerate(cfg["models"])
        ], cpu_threads // count
    return [{"id": model, "device": device} for model in cfg["models"]], cpu_threads


def _compute(*, run_id, protocol, parallel):
    cfg = recipe.configuration(protocol)
    cpu_threads = int(os.environ.get("PINGLAB_CPU_THREADS", "4"))
    if cpu_threads < 1:
        raise ValueError("PINGLAB_CPU_THREADS must be positive")
    torch.set_num_threads(cpu_threads)
    device = str(resolve_device())
    items, worker_threads = execution_items(cfg, device, cpu_threads, parallel)
    labels, arrays, metadata, counts = dataset.load_inputs(
        cfg, Path(os.environ.get("SHD_CACHE", "/tmp/shd"))
    )
    # Dataset sizes are resolved from verified data, without mutating the recipe.
    cfg = recipe.configuration(
        protocol,
        train_class_counts=counts["train_class_counts"],
        validation_class_counts=counts["validation_class_counts"],
    )
    arrays.update(
        train_labels=labels[: cfg["train_examples"]],
        validation_labels=labels[cfg["train_examples"] :],
    )
    selected = cfg["train_examples"]
    sample_cfg = dict(cfg, sample_offset=0)
    drive = dataset.encode_shd(
        arrays,
        [selected],
        dt_ms=cfg["dt_ms"],
        channels=cfg["n_input"],
        config=sample_cfg,
        seed=cfg["seed"] + 4,
    )
    with stage_run(
        REPO, cfg["experiment"], "compute", run_id=run_id, configuration=cfg
    ) as run:
        dataset_path = run.scratch / "dataset.npz"
        np.savez_compressed(dataset_path, **arrays)
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={
                path: ["start"]
                for model in cfg["models"]
                for path in (f"{model}--inference", f"{model}/best-validation")
            }
        )
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
                "sample_selection": "first validation utterance in split order; not selected by prediction or activity",
                "sample_class": int(labels[selected]),
                "official_train_index": int(arrays["official_train_indices"][selected]),
                "inference_noise_seed": cfg["seed"] + 4,
                "duration_ms": cfg["duration_ms"],
                "dt_ms": cfg["dt_ms"],
            },
        )
        recordings = run_conditions(
            run, cfg, dataset_path, drive, items, worker_threads, parallel
        )
        np.savez_compressed(run.export / "recording.npz", **recordings)
    return run.run_id


def main(argv=None):
    """Resolve the requested run identity and scientific protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--protocol", choices=("full", "local-check"), default="full")
    parser.add_argument(
        "--parallel", action="store_true", help="one model per CUDA GPU"
    )
    args = vars(parser.parse_args(argv))
    entrypoint = compute_parallel if args.pop("parallel") else compute
    return entrypoint(**args)


if __name__ == "__main__":
    main()
