"""Paired COBA/PING compute, with one GPU per worker and one atomic run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
import torch
from experiments.exp135 import dataset, recipe
from experiments.helpers.hpc import concurrent_compute as concurrent
from experiments.helpers.network_artifacts import snapshot_request
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)
from pingstore.stages import (
    _capture_code,
    operation_lock,
    reserve_stage,
    stage_run,
    utc_now,
)
from snnlab import sim

PLAN = Path(".scratch/paired-plan.json")


def report_epoch(model, report):
    print(
        f"{model}: epoch {report.epoch}/{report.total_epochs}: "
        f"train loss={report.train.loss:.5f}, accuracy={report.train.accuracy:.1%}; "
        f"validation loss={report.validation.loss:.5f}, accuracy={report.validation.accuracy:.1%}",
        flush=True,
    )


def training_request(output, data_root, cfg, model, *, device):
    duration = cfg.get("crop_ms", cfg["resolved_dataset"]["presentation_ms"])
    bundle = recipe.author_network(model, presentation_ms=duration, seed=cfg["seed"])
    encoder = sim.DatasetEncoder("event_bin", duration_ms=duration)
    bindings = {
        split: sim.DatasetSnapshotBinding(
            path=data_root / f"{split}-events.npz",
            input_id="events",
            dataset_id="tonic-dvs-gesture-training",
            split=split,
            encoder=sim.DatasetEncoder(
                "custom",
                definition="exp135.random_crop/v1",
                seed=cfg["crop_seed"],
                config={"duration_ms": duration, "random": split == "train"},
            )
            if cfg.get("crop_ms")
            else encoder,
            target_id="gesture",
        )
        for split in ("train", "validation")
    }
    spec = sim.ExecutionSpec(
        kind="train",
        graph=bundle.graph,
        training=bundle.training,
        input_bindings=(bindings["train"],),
        validation=sim.ValidationSpec(input_bindings=(bindings["validation"],)),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
        device=device,
        compile_mode=cfg.get("compile_mode"),
        batching=sim.Batching(order="shuffle"),
        metrics=sim.TrainingMetrics(train="online", initial=("validation",)),
        checkpoint_selection=sim.CheckpointSelection(),
        save_final_checkpoint=output / "final.checkpoint",
        save_selected_checkpoint=output / "selected.checkpoint",
        training_timing=True,
        diagnostics=False,
        on_epoch_end=lambda report: report_epoch(model, report),
        retain_runtime_state=False,
        retain_final_state=False,
        protocol={
            "development_subset": True,
            "official_test_used": False,
            "model": model,
        },
    )
    spec = snapshot_request(spec, bundle, output / "training")
    bundle.write(output / "training/network.bundle", visualise=True)
    return bundle, spec


def record_sample(output, data_root, cfg, model, bundle, *, device):
    rows = cfg["resolved_dataset"]["splits"]["validation"]
    index = min(range(len(rows)), key=lambda i: rows[i]["sample_id"])
    with np.load(data_root / "validation-events.npz", allow_pickle=False) as arrays:
        selected = arrays["event_sample"] == index
        steps = np.rint(arrays["event_time_ms"][selected] / cfg["dt_ms"]).astype(
            np.int64
        )
        channels = arrays["event_channel"][selected].copy()
        crop_start_ms = 0
        if cfg.get("crop_ms"):
            crop_start_ms = dataset.crop_start(
                rows[index]["duration_ms"],
                cfg["crop_ms"],
                seed=cfg["crop_seed"],
                sample_index=index,
                random=False,
            )
            keep = (steps >= crop_start_ms) & (steps < crop_start_ms + cfg["crop_ms"])
            steps, channels = steps[keep] - crop_start_ms, channels[keep]
    batches = np.zeros(len(steps), dtype=np.int64)
    input_events = np.column_stack((steps, batches, channels))
    duration = cfg.get("crop_ms", cfg["resolved_dataset"]["presentation_ms"])
    spec = sim.ExecutionSpec(
        kind="infer",
        graph=bundle.graph,
        checkpoint=output / "final.checkpoint",
        input_bindings=(
            sim.EventStreamBinding(
                "events",
                torch.from_numpy(steps),
                torch.from_numpy(batches),
                torch.from_numpy(channels),
                round(duration / cfg["dt_ms"]),
                1,
            ),
        ),
        recording=sim.RecordingSpec(
            signals=tuple(
                sim.SignalRecording(f"{pop}.spikes", kind="spike_events")
                for pop in recipe.POPULATIONS[model]
            ),
            block_steps=128,
        ),
        seed=cfg["seed"],
        device=device,
        diagnostics=False,
        retain_parameters=False,
        retain_final_state=False,
        retain_runtime_state=False,
        protocol={"sample_id": rows[index]["sample_id"], "checkpoint_role": "final"},
    )
    request = snapshot_request(spec, bundle, output / "inference")
    bundle.write(output / "inference/network.bundle", visualise=True)
    result = sim.infer(request)
    np.savez_compressed(
        output / "spikes.npz",
        input_events=input_events,
        scores=result.outputs["class_scores"][0].cpu().numpy(),
        **{
            pop: result.recorded_signals[f"{pop}.spikes"].cpu().numpy()
            for pop in recipe.POPULATIONS[model]
        },
    )
    write_json_atomic(
        output / "sample.json",
        dict(
            **rows[index],
            sample_index=index,
            crop_start_ms=crop_start_ms,
            model=model,
            checkpoint_role="final",
            presentation_ms=duration,
            dt_ms=cfg["dt_ms"],
            populations={
                p["id"]: p["size"]
                for p in bundle.graph["populations"]
                if p["id"] in recipe.POPULATIONS[model]
            },
        ),
    )


def train_case(export, cfg, model, *, device):
    """A worker owns exactly one condition directory, never sibling output."""
    torch.set_num_threads(int(os.environ.get("PINGLAB_CPU_THREADS", "1")))
    output = export / model
    output.mkdir(exist_ok=False)
    bundle, request = training_request(output, export, cfg, model, device=device)
    result = sim.train(request)
    write_json_atomic(
        output / "training-history.json",
        dict(
            epochs=result.metrics["epochs"],
            timing=result.metrics["training_timing"],
            selection=result.selected_checkpoint.selection_record,
        ),
    )
    del result
    record_sample(output, export, cfg, model, bundle, device=device)


def prepare_dataset(export, archive, download, *, local_check=None):
    cache = Path(os.environ.get("PINGLAB_DATASET_CACHE", "/tmp/pinglab-datasets"))
    path = dataset.verified_archive(
        archive or cache / "ibmGestureTrain.tar.gz", download=download
    )
    return dataset.prepare(path, export, local_check=local_check)


def clean_source(repo):
    source = _capture_code(repo, repo)
    if source["dirty"] or source["code_dirty"]:
        raise PingstoreError(
            "concurrent compute requires committed, clean execution source"
        )
    return source


def working(run_id):
    return concurrent.working_directory(
        REPO,
        run_id,
        0,
        recipe.SHARDS,
        experiment=recipe.SLUG,
        expected_count=recipe.SHARDS,
    )


def prepare_parallel(*, run_id=None, archive=None, download=False, cfg=None):
    source = clean_source(REPO)
    with operation_lock(REPO / ".pingstore", exclusive=False):
        run_id = run_id or reserve_stage(REPO / ".pingstore", recipe.SLUG, "compute")
        directory = working(run_id)
        with concurrent.compute_lock(directory, exclusive=True):
            if (directory / PLAN).exists() or any((directory / "export").iterdir()):
                raise PingstoreError(
                    "reservation already contains work; inspect it or reserve a new run"
                )
            started = utc_now()
            cfg = dict(cfg or recipe.configuration())
            dispatch_plan = (
                load_json(directory / ".scratch/dispatch-plan.json")
                if (directory / ".scratch/dispatch-plan.json").exists()
                else None
            )
            if dispatch_plan is not None and (
                dispatch_plan.get("schema") != "exp135.modal-plan/v1"
                or dispatch_plan["source"] != source
                or dispatch_plan["configuration"] != cfg
            ):
                raise PingstoreError(
                    "submitted recipe or source differs from the frozen Modal plan"
                )
            cfg["resolved_dataset"] = prepare_dataset(
                directory / "export", archive, download
            )
            cfg["dataset_files"] = {
                f"{split}-events.npz": file_sha256(
                    directory / f"export/{split}-events.npz"
                )
                for split in ("train", "validation")
            }
            plan = dict(
                schema="exp135.paired-plan/v1",
                run_id=run_id,
                source=source,
                configuration=cfg,
                preparation_started_at=started,
                prepared_at=utc_now(),
                dataset_files={
                    f"{split}-events.npz": file_sha256(
                        directory / f"export/{split}-events.npz"
                    )
                    for split in ("train", "validation")
                },
                dispatch_plan=dispatch_plan,
            )
            if clean_source(REPO) != source:
                raise PingstoreError("source changed during dataset preparation")
            write_json_atomic(directory / PLAN, plan)
    return run_id


def load_plan(directory, run_id):
    plan = load_json(directory / PLAN)
    if (
        plan.get("schema") != "exp135.paired-plan/v1"
        or plan["run_id"] != run_id
        or plan["configuration"]["work_items"] != recipe.work_items()
        or plan["source"] != clean_source(REPO)
    ):
        raise PingstoreError("frozen plan identity, allocation or source changed")
    for name, digest in plan["dataset_files"].items():
        if (
            name not in {"train-events.npz", "validation-events.npz"}
            or file_sha256(directory / "export" / name) != digest
        ):
            raise PingstoreError("shared dataset snapshot changed")
    if set(plan["dataset_files"]) != {"train-events.npz", "validation-events.npz"}:
        raise PingstoreError("incomplete dataset pins")
    return plan


def case_inventory(directory, model):
    root = directory / "export" / model
    required = (
        "training-history.json",
        "sample.json",
        "spikes.npz",
        "training/start/snapshot.json",
        "training/end/snapshot.json",
        "inference/start/snapshot.json",
        "inference/end/snapshot.json",
        "final.checkpoint/manifest.json",
        "selected.checkpoint/manifest.json",
    )
    required += tuple(
        f"{execution}/network.bundle/reports/{view}.{extension}"
        for execution in ("training", "inference")
        for view in ("circuit", "training", "expanded")
        for extension in ("svg", "png")
    )
    if any(not (root / name).is_file() for name in required):
        raise PingstoreError(f"incomplete {model} scientific output")
    return {
        p.relative_to(directory / "export").as_posix(): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def run_shard(run_id, index, *, device):
    if not 0 <= index < recipe.SHARDS:
        raise PingstoreError("invalid condition shard")
    with operation_lock(REPO / ".pingstore", exclusive=False):
        directory = working(run_id)
        plan = load_plan(directory, run_id)
        model = recipe.MODELS[index]
        return concurrent.execute_shard(
            repo=REPO,
            experiment=recipe.SLUG,
            run_id=run_id,
            index=index,
            count=recipe.SHARDS,
            expected_count=recipe.SHARDS,
            inputs={},
            configuration=plan["configuration"],
            items=[recipe.work_items()[index]],
            run_items=lambda: train_case(
                directory / "export", plan["configuration"], model, device=device
            ),
            inventory=lambda: case_inventory(directory, model),
            check_inputs=lambda: load_plan(directory, run_id),
        )


def collect(run_id):
    with operation_lock(REPO / ".pingstore", exclusive=False):
        directory = working(run_id)
        with concurrent.collect_shards(
            repo=REPO,
            experiment=recipe.SLUG,
            run_id=run_id,
            count=recipe.SHARDS,
            inputs={},
            configuration=lambda: load_plan(directory, run_id)["configuration"],
            items_for=lambda i: [recipe.work_items()[i]],
            inventory_for=lambda i: case_inventory(directory, recipe.MODELS[i]),
            collect=True,
        ) as (_, records):
            plan = load_plan(directory, run_id)
            if {p.name for p in (directory / "export").iterdir()} != set(
                recipe.MODELS
            ) | set(plan["dataset_files"]):
                raise PingstoreError("unexpected or missing shared scientific outputs")
            with stage_run(
                REPO,
                recipe.SLUG,
                "compute",
                run_id=run_id,
                configuration=plan["configuration"],
            ) as run:
                concurrent.retain_worker_provenance(run, records)
                run.record["execution"].update(
                    backend="snnlab.sim.train+infer",
                    official_test_used=False,
                    dataset_preparation=dict(
                        started_at=plan["preparation_started_at"],
                        completed_at=plan["prepared_at"],
                    ),
                )
                run.record["execution"]["started_at"] = plan["preparation_started_at"]
                if plan.get("dispatch_plan"):
                    run.record["execution"]["resources"] = plan["dispatch_plan"][
                        "resources"
                    ]
    return run_id


def launch_workers(run_id, *, popen=None):
    """Start both isolated GPU processes before waiting; stop siblings on failure."""
    popen = popen or subprocess.Popen
    if torch.cuda.device_count() != recipe.SHARDS:
        raise RuntimeError("paired Modal compute requires exactly two visible GPUs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    if len(visible) != recipe.SHARDS or len(set(visible)) != recipe.SHARDS:
        raise RuntimeError("two distinct CUDA device assignments are required")
    processes = []
    pool = None
    try:
        for index in range(recipe.SHARDS):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--run-id",
                run_id,
                "--shard-index",
                str(index),
                "--device",
                "cuda",
            ]
            environment = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=visible[index],
                PINGLAB_CPU_THREADS=str(
                    max(1, int(os.environ.get("PINGLAB_CPU_THREADS", "4")) // 2)
                ),
            )
            processes.append(popen(command, cwd=REPO, env=environment))
        pool = ThreadPoolExecutor(max_workers=recipe.SHARDS)
        futures = {pool.submit(p.wait): p for p in processes}
        for future in as_completed(futures):
            if future.result() != 0:
                raise RuntimeError(
                    "condition worker failed; shared compute remains incomplete"
                )
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


def compute(
    *,
    archive=None,
    download=False,
    run_id=None,
    device="cpu",
    epochs=recipe.EPOCHS,
    batch_size=recipe.BATCH_SIZE,
    seed=recipe.TRAIN_SEED,
    local_check=False,
    full_gestures=False,
):
    cfg = (
        recipe.local_check_configuration(full_gestures=full_gestures)
        if local_check
        else recipe.configuration(epochs=epochs, batch_size=batch_size, seed=seed)
    )
    if local_check and device != "cpu":
        raise ValueError("the local-check phase uses CPU")
    if device == "cuda":
        if torch.cuda.device_count() != recipe.SHARDS:
            raise RuntimeError("paired compute requires two GPUs; request A100-80GB:2")
        run_id = prepare_parallel(
            run_id=run_id, archive=archive, download=download, cfg=cfg
        )
        launch_workers(run_id)
        return collect(run_id)
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        run_id=run_id,
        configuration=cfg,
        phase=("local-check-full" if full_gestures else "local-check")
        if local_check
        else None,
    ) as run:
        cfg["resolved_dataset"] = prepare_dataset(
            run.export, archive, download, local_check=cfg.get("local_check")
        )
        cfg["dataset_files"] = {
            f"{split}-events.npz": file_sha256(run.export / f"{split}-events.npz")
            for split in ("train", "validation")
        }
        write_json_atomic(run.directory / "run.json", run.record)
        for model in recipe.MODELS:
            train_case(run.export, cfg, model, device="cpu")
        run.record["execution"].update(
            backend="snnlab.sim.train+infer", device="cpu", official_test_used=False
        )
    return run.run_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--phase",
        choices=(
            "local-check",
            "local-check-full",
            "gpu-benchmark-coba",
            "gpu-benchmark-ping",
        ),
    )
    parser.add_argument("--dataset-archive", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=os.environ.get("PINGLAB_DEVICE", "cpu"),
    )
    parser.add_argument("--shard-index", type=int, choices=range(recipe.SHARDS))
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Validate both completed workers and finalize their shared reservation.",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print settings without download, reservation or execution.",
    )
    args = parser.parse_args(argv)
    if args.phase and args.phase.startswith("gpu-benchmark-"):
        if args.plan:
            print(json.dumps(recipe.benchmark_configuration(), indent=2))
            return
        from experiments.exp135.benchmark import run_benchmark

        run_benchmark(args.phase.removeprefix("gpu-benchmark-"), args.run_id)
        return
    if args.plan:
        print(
            json.dumps(
                recipe.local_check_configuration(
                    full_gestures=args.phase == "local-check-full"
                )
                if args.phase
                else recipe.configuration(),
                indent=2,
            )
        )
        return
    if args.collect or args.shard_index is not None:
        if (
            not args.run_id
            or args.download
            or args.dataset_archive
            or args.phase
            or (args.collect and args.shard_index is not None)
        ):
            parser.error(
                "worker/collector requires an existing --run-id and no dataset overrides"
            )
        if args.collect:
            collect(args.run_id)
        else:
            run_shard(args.run_id, args.shard_index, device=args.device)
        return
    compute(
        archive=args.dataset_archive,
        download=args.download,
        run_id=args.run_id,
        device=args.device,
        local_check=args.phase is not None,
        full_gestures=args.phase == "local-check-full",
    )


if __name__ == "__main__":
    main()
