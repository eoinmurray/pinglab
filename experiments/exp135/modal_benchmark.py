"""Detached, phase-scoped Modal benchmark; never launches production training."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal
from experiments.helpers.modal_compute import REMOTE, SOURCE, SOURCE_DIRS, image

app = modal.App("exp135-gpu-benchmark")
volume = modal.Volume.from_name("pinglab-compute-results")


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=65536,
    timeout=14400,
    serialized=True,
    include_source=False,
    volumes={"/modal-output": volume},
    retries=0,
)
def worker(experiment, reservation, dispatch, *, submission):
    for directory in SOURCE_DIRS:
        shutil.rmtree(REMOTE / directory)
        shutil.copytree(SOURCE / directory, REMOTE / directory)
    for filename in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(SOURCE / filename, REMOTE / filename)
    sys.path[:0] = [str(REMOTE), str(REMOTE / "tools")]
    from pingstore.contracts import (
        validate_operational_run_directory,
        write_json_atomic,
    )
    from pingstore.layout import initialize_layout

    temporary = REMOTE / ".pingstore/runs" / f".{reservation['run_id']}.tmp"
    temporary.mkdir(parents=True)
    initialize_layout(temporary, experiment)
    write_json_atomic(temporary / ".reservation.json", reservation)
    env = dict(os.environ, PINGLAB_CPU_THREADS="4")
    subprocess.run(
        [
            sys.executable,
            "experiments/exp135/compute.py",
            "--run-id",
            reservation["run_id"],
            "--phase",
            reservation["phase"],
        ],
        cwd=REMOTE,
        env=env,
        check=True,
    )
    completed = temporary.with_name(reservation["run_id"])
    record = validate_operational_run_directory(completed)
    record["execution"]["modal"] = {
        **submission,
        "function_call_id": modal.current_function_call_id(),
    }
    write_json_atomic(completed / "run.json", record)
    destination = Path("/modal-output").resolve() / submission["remote_path"]
    shutil.copytree(completed, destination)
    validate_operational_run_directory(destination)
    files = [
        str(p.relative_to(destination))
        for p in sorted(destination.rglob("*"))
        if p.is_file()
    ]
    write_json_atomic(
        destination.parents[2] / "completion.json",
        dict(run_id=reservation["run_id"], dispatch=dispatch, files=files),
    )
    volume.commit()
    return files


@app.function(
    image=image,
    cpu=0.25,
    memory=512,
    timeout=14400,
    serialized=True,
    include_source=False,
)
def coordinate(jobs):
    calls = [
        worker.spawn("exp135", reservation, receipt["dispatch"], submission=receipt)
        for reservation, receipt in jobs
    ]
    outcomes, errors = [], []
    for call in calls:
        try:
            outcomes.append(call.get())
        except Exception as error:
            errors.append(str(error))
    if errors:
        raise RuntimeError("Benchmark worker errors: " + "; ".join(errors))
    return outcomes


@app.local_entrypoint()
def main():
    import uuid

    from experiments.helpers.modal_jobs import RECEIPT
    from pingstore.contracts import write_json_atomic
    from pingstore.stages import reserve_stage, stage_reservation, utc_now

    benchmark_plan = json.loads(
        subprocess.check_output(
            [
                str(Path.cwd() / ".venv/bin/python"),
                "experiments/exp135/compute.py",
                "--phase",
                "gpu-benchmark-coba",
                "--plan",
            ],
            text=True,
        )
    )
    root = Path("/Users/eoin/pinglab/.pingstore")
    jobs = []
    for model in ("coba", "ping"):
        dispatch = uuid.uuid4().hex
        identity = reserve_stage(
            root,
            "exp135",
            "compute",
            origin=f"modal-{dispatch}",
            phase=f"gpu-benchmark-{model}",
        )
        temporary = root / "runs" / f".{identity}.tmp"
        receipt = dict(
            schema="pinglab.modal-dispatch/v1",
            run_id=identity,
            experiment="exp135",
            dispatch=dispatch,
            app_id=app.app_id,
            function_call_id=None,
            volume="pinglab-compute-results",
            remote_path=f"{dispatch}/.pingstore/runs/{identity}",
            prepared_at=utc_now(),
            resources=dict(gpu="A100-80GB", cpu=4, memory=65536, timeout=14400),
            inputs={},
            plan=benchmark_plan,
        )
        write_json_atomic(temporary / RECEIPT, receipt)
        jobs.append((stage_reservation(temporary), receipt))
    call = coordinate.spawn(jobs)
    for reservation, receipt in jobs:
        receipt.update(function_call_id=call.object_id, submitted_at=utc_now())
        write_json_atomic(
            root / "runs" / f".{reservation['run_id']}.tmp" / RECEIPT, receipt
        )
        print(json.dumps(receipt), flush=True)
