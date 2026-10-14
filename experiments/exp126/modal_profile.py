"""Explicit L4 diagnosis of retained comparison; no training or library edits."""

import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import modal
from experiments.helpers.modal_compute import (
    REMOTE,
    REPO,
    SOURCE,
    SOURCE_DIRS,
    app,
    image,
    results,
)

sys.path[:0] = [str(REPO), str(REPO / "tools")]
from pingstore.contracts import validate_operational_run_directory
from pingstore.stages import operation_lock, reserve_stage, stage_reservation

SOURCE_ID = "exp126-r018-compute"
INPUTS = [
    SOURCE_ID,
    "exp126-r016-compute-phase-snntorch",
    "exp126-r017-compute-phase-snnlab",
]
if modal.is_local():
    for identity in INPUTS:
        validate_operational_run_directory(REPO / ".pingstore/runs" / identity)
        image = image.add_local_dir(
            REPO / ".pingstore/runs" / identity,
            str(SOURCE / ".pingstore/runs" / identity),
        )


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=16384,
    timeout=1800,
    serialized=True,
    include_source=False,
    volumes={"/modal-output": results},
    retries=0,
)
def profile_worker(reservation, dispatch):
    for directory in SOURCE_DIRS:
        shutil.rmtree(REMOTE / directory)
        shutil.copytree(SOURCE / directory, REMOTE / directory)
    for filename in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(SOURCE / filename, REMOTE / filename)
    sys.path[:0] = [str(REMOTE), str(REMOTE / "tools")]
    from pingstore.contracts import (
        validate_operational_run_directory,
    )
    from pingstore.contracts import (
        write_json_atomic as remote_write_json,
    )
    from pingstore.layout import initialize_layout

    for identity in INPUTS:
        shutil.copytree(
            SOURCE / ".pingstore/runs" / identity, REMOTE / ".pingstore/runs" / identity
        )
    identity = reservation["run_id"]
    temporary = REMOTE / ".pingstore/runs" / f".{identity}.tmp"
    temporary.mkdir(parents=True)
    initialize_layout(temporary, "exp126")
    remote_write_json(temporary / ".reservation.json", reservation)
    subprocess.run(
        [
            sys.executable,
            "experiments/exp126/compute.py",
            "--profile",
            "--source",
            SOURCE_ID,
            "--device",
            "cuda",
            "--cpu-threads",
            "4",
            "--run-id",
            identity,
        ],
        cwd=REMOTE,
        check=True,
    )
    completed = temporary.with_name(identity)
    validate_operational_run_directory(completed)
    destination = (
        Path("/modal-output").resolve() / dispatch / ".pingstore/runs" / identity
    )
    staging = destination.with_name(f".{identity}.tmp")
    shutil.copytree(completed, staging)
    validate_operational_run_directory(staging)
    staging.rename(destination)
    results.commit()
    return [
        str(path.relative_to(destination))
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    ]


@app.local_entrypoint()
def profile():
    with operation_lock(REPO / ".pingstore", exclusive=False):
        dispatch = uuid.uuid4().hex
        identity = reserve_stage(
            REPO / ".pingstore", "exp126", "compute", origin=f"modal-{dispatch}"
        )
        temporary = REPO / ".pingstore/runs" / f".{identity}.tmp"
        reservation = stage_reservation(temporary)
        print(f"Profiling run: {identity}; dispatch={dispatch}", flush=True)
        files = profile_worker.remote(reservation, dispatch)
        download = temporary / ".scratch" / identity
        download.mkdir(parents=True)
        for filename in files:
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe remote path")
            target = download / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                for block in results.read_file(
                    f"{dispatch}/.pingstore/runs/{identity}/{filename}"
                ):
                    handle.write(block)
        record = validate_operational_run_directory(download)
        source = validate_operational_run_directory(
            REPO / ".pingstore/runs" / SOURCE_ID
        )
        if (
            record["run_id"] != identity
            or record["origin"] != reservation["origin"]
            or record["inputs"]
            != {
                "comparison": {
                    "run_id": SOURCE_ID,
                    "payload_digest": source["payload_digest"],
                }
            }
        ):
            raise ValueError("profile download differs from requested work")
        download.rename(temporary.with_name(identity))
        shutil.rmtree(temporary)
        print(identity, flush=True)
