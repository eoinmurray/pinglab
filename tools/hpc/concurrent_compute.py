"""Shared resumable HPC concurrency for one hidden Pingstore compute run.

Experiments own scientific work items and partitioning in ``recipe.py``.  This
module owns only the concurrency lifecycle: reservation validation, worker and
collector exclusion, frozen shard records, payload verification, and projection
of worker provenance into the completed run record.
"""

from __future__ import annotations

import contextlib
import fcntl
import multiprocessing
import os
import socket
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any

from pingstore.contracts import PingstoreError, load_json, run_root, write_json_atomic
from pingstore.stages import _capture_code, stage_reservation, utc_now

SHARD_SCHEMA = "pinglab.concurrent-shard/v1"
SCHEDULER_KEYS = ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID")

WorkItem = Mapping[str, Any]
Inventory = Mapping[str, str]


def work_item_ids(items: Sequence[WorkItem]) -> list[str]:
    """Return stable work identities and reject ambiguous plans."""
    identities = [item.get("id") for item in items]
    if any(not isinstance(identity, str) or not identity for identity in identities):
        raise PingstoreError("concurrent work items require nonempty string ids")
    if len(set(identities)) != len(identities):
        raise PingstoreError("concurrent work item ids must be unique within a shard")
    return identities


def working_directory(
    repo: Path,
    run_id: str,
    index: int,
    count: int,
    *,
    experiment: str,
    expected_count: int,
) -> Path:
    """Resolve and validate an unused compute reservation for one shard."""
    if count != expected_count or not 0 <= index < count:
        raise PingstoreError(
            f"{experiment} requires {expected_count} shards and an index in "
            f"[0, {expected_count})"
        )
    destination = run_root(repo / ".pingstore", run_id)
    directory = destination.with_name(f".{run_id}.tmp")
    record = stage_reservation(directory)
    if (
        record["experiment"] != experiment
        or record["stage"] != "compute"
        or record["run_id"] != run_id
        or destination.exists()
        or (directory / "run.json").exists()
    ):
        raise PingstoreError(
            f"shards require an unused {experiment} v4 compute reservation"
        )
    return directory


@contextlib.contextmanager
def compute_lock(directory: Path, *, exclusive: bool) -> Iterator[None]:
    """Exclude collectors from workers without serialising sibling workers."""
    path = directory / ".scratch/compute.lock"
    if any(p.is_symlink() for p in (directory, *directory.parents, path.parent, path)):
        raise PingstoreError("compute working paths must not use symlinks")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as handle:
        try:
            fcntl.flock(
                handle,
                (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB,
            )
        except BlockingIOError as exc:
            raise PingstoreError("compute reservation is busy") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _worker_metadata() -> dict[str, Any]:
    import torch

    device = (
        {
            "type": "cuda",
            "name": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        if torch.cuda.is_available()
        else {"type": "cpu", "torch": torch.__version__}
    )
    return {
        "host": socket.gethostname(),
        "device": device,
        "command": [sys.executable, *sys.argv],
        "scheduler": {
            key: os.environ[key] for key in SCHEDULER_KEYS if key in os.environ
        },
    }


def execute_shard(
    *,
    repo: Path,
    experiment: str,
    run_id: str,
    index: int,
    count: int,
    expected_count: int,
    inputs: Mapping[str, Mapping[str, str]],
    configuration: Mapping[str, Any],
    items: Sequence[WorkItem],
    run_items: Callable[[], None],
    inventory: Callable[[], Inventory],
    check_inputs: Callable[[], None],
    capture_code: Callable[[Path, Path], Mapping[str, Any]] = _capture_code,
) -> dict[str, Any]:
    """Run or verify one frozen shard of an incomplete compute reservation."""
    directory = working_directory(
        repo,
        run_id,
        index,
        count,
        experiment=experiment,
        expected_count=expected_count,
    )
    with compute_lock(directory, exclusive=False):
        folder = directory / ".scratch/shards" / str(index)
        folder.mkdir(parents=True, exist_ok=True)
        lock = folder / "writer.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise PingstoreError(
                "shard is busy; interrupted locks need explicit recovery"
            ) from exc
        os.close(descriptor)
        try:
            working_directory(
                repo,
                run_id,
                index,
                count,
                experiment=experiment,
                expected_count=expected_count,
            )
            code = dict(capture_code(repo, directory))
            if code.get("code_dirty") or code.get("dirty"):
                raise PingstoreError(
                    f"distributed {experiment} compute requires committed execution code"
                )
            expected = {
                "schema": SHARD_SCHEMA,
                "run_id": run_id,
                "experiment": experiment,
                "inputs": dict(inputs),
                "configuration": dict(configuration),
                "index": index,
                "count": count,
                "source": code,
                "work_items": work_item_ids(items),
            }
            marker = folder / "completed.json"
            if marker.exists():
                previous = load_json(marker)
                if any(previous.get(key) != value for key, value in expected.items()):
                    raise PingstoreError(
                        "shard source, inputs, configuration or allocation changed; "
                        "reserve a fresh compute run"
                    )
                if previous.get("files") != dict(inventory()):
                    raise PingstoreError("shard payload changed or is incomplete")
                return previous
            started = utc_now()
            run_items()
            check_inputs()
            record = {
                **expected,
                "started_at": started,
                "completed_at": utc_now(),
                **_worker_metadata(),
                "files": dict(inventory()),
            }
            write_json_atomic(marker, record)
            return record
        finally:
            lock.unlink()


@contextlib.contextmanager
def collect_shards(
    *,
    repo: Path,
    experiment: str,
    run_id: str,
    count: int,
    inputs: Mapping[str, Mapping[str, str]],
    configuration: Mapping[str, Any] | Callable[[], Mapping[str, Any]],
    items_for: Callable[[int], Sequence[WorkItem]],
    inventory_for: Callable[[int], Inventory],
    collect: bool,
) -> Iterator[tuple[Path, list[dict[str, Any]]]]:
    """Lock a compute reservation and verify every shard before final assembly."""
    directory = working_directory(
        repo,
        run_id,
        0,
        count,
        experiment=experiment,
        expected_count=count,
    )
    with compute_lock(directory, exclusive=True):
        working_directory(
            repo,
            run_id,
            0,
            count,
            experiment=experiment,
            expected_count=count,
        )
        shard_root = directory / ".scratch/shards"
        if not collect and shard_root.exists():
            raise PingstoreError("sharded work requires explicit --collect")
        records: list[dict[str, Any]] = []
        if collect:
            if list(shard_root.glob("*/writer.lock")):
                raise PingstoreError("compute shards are still running")
            # Resolve frozen worker configuration only while collectors exclude workers.
            configuration = configuration() if callable(configuration) else configuration
            allocated: set[str] = set()
            for index in range(count):
                record = load_json(shard_root / str(index) / "completed.json")
                work_items = work_item_ids(items_for(index))
                overlap = allocated.intersection(work_items)
                if overlap:
                    raise PingstoreError(
                        "work items are allocated to more than one shard: "
                        + ", ".join(sorted(overlap))
                    )
                allocated.update(work_items)
                expected = {
                    "schema": SHARD_SCHEMA,
                    "run_id": run_id,
                    "experiment": experiment,
                    "inputs": dict(inputs),
                    "configuration": dict(configuration),
                    "index": index,
                    "count": count,
                    "work_items": work_items,
                }
                if any(record.get(key) != value for key, value in expected.items()):
                    raise PingstoreError(
                        "shard inputs, configuration, allocation or identity mismatch"
                    )
                if record.get("files") != dict(inventory_for(index)):
                    raise PingstoreError("shard payload changed or is incomplete")
                records.append(record)
        yield directory, records


def retain_worker_provenance(run: Any, records: Sequence[Mapping[str, Any]]) -> None:
    """Project discarded shard bookkeeping into authoritative run metadata."""
    for record in records:
        if record["source"] != run.record["provenance"]:
            raise PingstoreError("worker and collector execution code differ")
    workers = [
        {
            key: record[key]
            for key in (
                "index",
                "started_at",
                "completed_at",
                "host",
                "device",
                "scheduler",
                "command",
                "work_items",
            )
        }
        for record in records
    ]
    run.record["execution"]["shards"] = workers
    run.record["execution"]["collector_started_at"] = run.record["execution"][
        "started_at"
    ]
    run.record["execution"]["started_at"] = min(
        record["started_at"] for record in records
    )
    run.record["execution"]["workers_completed_at"] = max(
        record["completed_at"] for record in records
    )


def _active_worker(directory, item, index, count, worker, arguments, inventory, frozen):
    """One non-resumable child of an already active, coordinator-owned stage."""
    with compute_lock(directory, exclusive=False):
        folder = directory / ".scratch/shards" / str(index)
        folder.mkdir(parents=True, exist_ok=True)
        lock = folder / "writer.lock"
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        try:
            started = utc_now()
            result = worker(directory, item, index, *arguments)
            record = {
                **frozen,
                "schema": SHARD_SCHEMA,
                "index": index,
                "count": count,
                "work_items": work_item_ids((item,)),
                "allocation": dict(item),
                "worker_pid": os.getpid(),
                "started_at": started,
                "completed_at": utc_now(),
                **_worker_metadata(),
                "files": dict(inventory(directory, item, index)),
                "result": result,
            }
            write_json_atomic(folder / "completed.json", record)
        finally:
            lock.unlink()


def run_active_workers(*, run, items, worker, arguments=(), inventory):
    """Spawn isolated same-host workers inside one active StageRun.

    Unlike execute_shard/collect_shards, the coordinator already owns the stage
    writer lock and its run.json. Children never reserve, complete or edit that
    manifest. They write disjoint outputs and scratch completion records. This
    path is not resumable: any failure cancels and joins siblings, leaving the
    coordinator's stage hidden. Spawn avoids inherited CUDA and RNG state.
    """
    identities = work_item_ids(items)
    if not identities:
        raise PingstoreError("concurrent execution requires work items")
    directory = run.directory
    active = load_json(directory / "run.json")
    if (
        active["run_id"] != run.run_id
        or active["stage"] != "compute"
        or not (directory / ".writer.lock").is_file()
        or (directory / ".scratch/shards").exists()
    ):
        raise PingstoreError("workers require a fresh active coordinator-owned compute stage")
    frozen = {
        "run_id": run.run_id,
        "experiment": active["experiment"],
        "inputs": active["inputs"],
        "configuration": active["execution"]["configuration"],
        "source": active["provenance"],
    }
    context = multiprocessing.get_context("spawn")
    processes = []
    try:
        for index, item in enumerate(items):
            process = context.Process(
                name=f"{run.run_id}-{item['id']}",
                target=_active_worker,
                args=(directory, item, index, len(items), worker, arguments, inventory, frozen),
            )
            process.start()
            processes.append(process)
        pending = {process.sentinel: process for process in processes}
        while pending:
            for sentinel in wait(list(pending)):
                process = pending.pop(sentinel)
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(f"concurrent worker {process.name} failed with exit code {process.exitcode}")
    finally:
        # This also handles spawn failures, KeyboardInterrupt and failed children.
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
    records, occupied = [], set()
    with compute_lock(directory, exclusive=True):
        for index, item in enumerate(items):
            folder = directory / ".scratch/shards" / str(index)
            if (folder / "writer.lock").exists():
                raise PingstoreError("worker exited without releasing its writer lock")
            record = load_json(folder / "completed.json")
            expected = {
                **frozen, "schema": SHARD_SCHEMA, "index": index,
                "count": len(items), "work_items": [item["id"]],
                "allocation": dict(item),
            }
            if any(record.get(key) != value for key, value in expected.items()):
                raise PingstoreError("worker completion does not match the active compute allocation")
            files = dict(inventory(directory, item, index))
            if not files or record.get("files") != files or occupied.intersection(files):
                raise PingstoreError("worker payload is missing, changed or overlaps another worker")
            occupied.update(files)
            records.append(record)
    # Only the coordinator updates authoritative provenance; no finalization here.
    run.record["execution"]["workers"] = [
        {key: record[key] for key in (
            "index", "work_items", "started_at", "completed_at", "host",
            "device", "scheduler", "command", "allocation", "worker_pid",
        )}
        for record in records
    ]
    return records
