"""Durable Modal submission receipts and independent result collection.

This module does not import the training app or build/upload an image when polled.
"""

import fcntl
import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

from pingstore.contracts import (
    STAGE_ID_RE,
    PingstoreError,
    load_json,
    validate_operational_run_directory,
    write_json_atomic,
)
from pingstore.stages import (
    operation_lock,
    reserve_stage,
    source_run,
    stage_reservation,
    utc_now,
)

VOLUME = "pinglab-compute-results"
RECEIPT = Path(".scratch/modal-dispatch.json")


def submit(
    root, experiment, worker, *, options, arguments, app_id, inputs=None, plan=None
):
    """Reserve once, spawn once, persist the call ID, and return without waiting."""
    dispatch = uuid.uuid4().hex
    inputs = inputs or {}
    with operation_lock(root, exclusive=False):
        for pin in inputs.values():
            source_run(root, pin["run_id"], reference=pin)
        identity = reserve_stage(
            root, experiment, "compute", origin=f"modal-{dispatch}"
        )
        temporary = root / "runs" / f".{identity}.tmp"
        reservation = stage_reservation(temporary)
        receipt = {
            "schema": "pinglab.modal-dispatch/v1",
            "run_id": identity,
            "experiment": experiment,
            "dispatch": dispatch,
            "app_id": app_id,
            "function_call_id": None,
            "volume": VOLUME,
            "remote_path": f"{dispatch}/.pingstore/runs/{identity}",
            "prepared_at": utc_now(),
            "resources": options,
            "inputs": inputs,
        }
        if plan is not None:
            receipt["plan"] = plan
        if inputs:
            # Standard incomplete-run input pins protect remote jobs from pruning.
            write_json_atomic(
                temporary / "run.json",
                {
                    **reservation,
                    "inputs": inputs,
                    "execution": {"operation": "modal-dispatch"},
                },
            )
        write_json_atomic(temporary / RECEIPT, receipt)
        # Keep the prepared receipt if submission raises: its outcome may be unknown.
        call = worker.with_options(**options).spawn(
            experiment, reservation, dispatch, **arguments, submission=receipt
        )
        receipt.update(function_call_id=call.object_id, submitted_at=utc_now())
        write_json_atomic(temporary / RECEIPT, receipt)
        return receipt


def pending(root, identity):
    match = STAGE_ID_RE.fullmatch(identity)
    if match is None or match.group(3) != "compute":
        raise PingstoreError("expected an explicit compute run ID")
    directory = root / "runs" / f".{identity}.tmp"
    reservation = stage_reservation(directory)
    receipt = load_json(directory / RECEIPT)
    dispatch = receipt.get("dispatch", "")
    if (
        receipt.get("schema") != "pinglab.modal-dispatch/v1"
        or receipt.get("run_id") != identity
        or receipt.get("experiment") != reservation["experiment"]
        or reservation["origin"] != f"modal-{dispatch}"
        or len(dispatch) != 32
        or any(c not in "0123456789abcdef" for c in dispatch)
        or receipt.get("volume") != VOLUME
        or receipt.get("remote_path") != f"{dispatch}/.pingstore/runs/{identity}"
    ):
        raise PingstoreError("dispatch receipt does not match reservation")
    return directory, reservation, receipt


def completed_files(receipt, volume, call_factory):
    """Prefer durable volume evidence; call results can expire after seven days."""
    try:
        marker = json.loads(
            b"".join(volume.read_file(f"{receipt['dispatch']}/completion.json"))
        )
    except FileNotFoundError:
        if not receipt["function_call_id"]:
            return None, "submission_unknown"
        try:
            files = call_factory(receipt["function_call_id"]).get(timeout=0)
        except TimeoutError:
            return None, "pending"
        return files, "completed"
    if (
        marker.get("run_id") != receipt["run_id"]
        or marker.get("dispatch") != receipt["dispatch"]
    ):
        raise PingstoreError("remote completion marker identifies a different job")
    return marker["files"], "completed"


def collect_files(root, identity, receipt, files, volume):
    """Retry interrupted transfers under the same reservation, then validate/rename."""
    if (
        not isinstance(files, list)
        or not files
        or any(not isinstance(f, str) for f in files)
    ):
        raise PingstoreError("invalid remote file inventory")
    paths = [Path(f) for f in files]
    if len(set(paths)) != len(paths) or any(
        p.is_absolute() or ".." in p.parts or p == Path(".") for p in paths
    ):
        raise PingstoreError("unsafe remote result path")
    temporary, reservation, current = pending(root, identity)
    if current != receipt:
        raise PingstoreError("dispatch receipt changed during collection")
    for pin in receipt.get("inputs", {}).values():
        source_run(root, pin["run_id"], reference=pin)
    download = temporary / ".scratch" / identity
    download.mkdir(parents=True, exist_ok=True)
    for relative in paths:
        target = download / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        with partial.open("wb") as handle:
            for chunk in volume.read_file(
                f"{receipt['remote_path']}/{relative.as_posix()}"
            ):
                handle.write(chunk)
        partial.replace(target)
    record = validate_operational_run_directory(download)
    if (
        record["run_id"] != identity
        or record["experiment"] != reservation["experiment"]
        or record["stage"] != "compute"
        or record["origin"] != reservation["origin"]
        or record["inputs"] != receipt.get("inputs", {})
        or record.get("scope") != ("phase" if reservation.get("phase") else "full")
        or record.get("phase") != reservation.get("phase")
    ):
        raise PingstoreError("downloaded result does not match submitted compute")
    record["execution"]["modal"] = receipt
    write_json_atomic(download / "run.json", record)
    validate_operational_run_directory(download)
    if stage_reservation(temporary) != reservation:
        raise PingstoreError("reservation changed during collection")
    destination = root / "runs" / identity
    if destination.exists():
        raise FileExistsError(destination)
    download.rename(destination)
    shutil.rmtree(temporary)
    return destination


@contextmanager
def collection_lock(directory):
    path = directory / ".scratch/modal-collect.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PingstoreError("another collector owns this run") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def check(root, identity, *, collect=False, volume=None, call_factory=None):
    match = STAGE_ID_RE.fullmatch(identity)
    if match is None or match.group(3) != "compute":
        raise PingstoreError("expected an explicit compute run ID")
    with operation_lock(root, exclusive=False):
        destination = root / "runs" / identity
        if destination.exists():
            validate_operational_run_directory(destination)
            return {"run_id": identity, "status": "collected"}
        directory, _, receipt = pending(root, identity)
        if volume is None or call_factory is None:
            import modal

            volume = volume if volume is not None else modal.Volume.from_name(VOLUME)
            call_factory = call_factory or modal.FunctionCall.from_id
        with collection_lock(directory):
            files, status = completed_files(receipt, volume, call_factory)
            if status == "completed" and collect:
                collect_files(root, identity, receipt, files, volume)
                status = "collected"
        return {
            "run_id": identity,
            "status": status,
            "app_id": receipt["app_id"],
            "function_call_id": receipt["function_call_id"],
        }
