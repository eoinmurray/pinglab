"""Dispatch durable compute on Modal; collect later with modal_jobs.py."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal

REMOTE = Path("/root/pinglab")
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE
SOURCE = Path("/root/pinglab-source")
SOURCE_DIRS = ("experiments", "tools")
SUPPORTED = ("exp047", "exp080", "exp081", "exp124", "exp134")
sys.path[:0] = [str(REPO), str(REPO / "tools")]

app = modal.App("pinglab-compute")
results = modal.Volume.from_name("pinglab-compute-results", create_if_missing=True)

# A real checkout lets the existing stage helper capture commit and dirty state.
# Current local sources are mounted over that baseline, including uncommitted edits.
if modal.is_local():
    cached_image = os.environ.get("PINGLAB_MODAL_IMAGE_ID")
    if cached_image:
        if not cached_image.startswith("im-"):
            raise ValueError("PINGLAB_MODAL_IMAGE_ID must identify a built Modal image")
        image = modal.Image.from_id(cached_image)
    else:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
        image = (
            modal.Image.debian_slim(python_version="3.12")
            .apt_install(
                "git", "pkg-config", "libcairo2-dev", "libpango1.0-dev", "ffmpeg"
            )
            .run_commands(
                f"git init {REMOTE}",
                f"git -C {REMOTE} remote add origin https://github.com/eoinmurray/pinglab",
                f"git -C {REMOTE} fetch --depth 1 origin {commit}",
                f"git -C {REMOTE} reset --hard FETCH_HEAD",
            )
            .uv_sync(str(REPO), extra_options="--no-dev")
            .workdir(str(REMOTE))
            .env({"PYTHONPATH": str(SOURCE)})
        )
    for directory in SOURCE_DIRS:
        image = image.add_local_dir(
            REPO / directory,
            str(SOURCE / directory),
            ignore=["**/__pycache__/**", "**/*.pyc", "**/.env", "**/.env.*"],
        )
    for filename in ("pyproject.toml", "uv.lock"):
        image = image.add_local_file(REPO / filename, str(SOURCE / filename))
else:
    image = None


@app.function(
    image=image,
    gpu=None,
    cpu=4,
    memory=16384,
    timeout=86400,
    serialized=True,
    include_source=False,
    volumes={"/modal-output": results},
    retries=0,
)
def compute(
    experiment: str,
    reservation: dict,
    dispatch: str,
    smoke: bool,
    device: str = "cuda",
    cpu: int = 4,
    epochs: int = 0,
    batch_size: int = 0,
    seed: int = 42,
    input_refs: dict | None = None,
    input_runs: list | None = None,
    submission: dict | None = None,
) -> list[str]:
    # Replace whole directories so local deletions are reflected too.
    for directory in SOURCE_DIRS:
        shutil.rmtree(REMOTE / directory)
        shutil.copytree(SOURCE / directory, REMOTE / directory)
    for filename in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(SOURCE / filename, REMOTE / filename)
    # An explicitly reused image must still satisfy the submitted dependency lock.
    from importlib.metadata import version

    import tomllib

    locked = tomllib.loads((REMOTE / "uv.lock").read_text())
    expected_snnlab = {
        row["version"] for row in locked["package"] if row["name"] == "snnlab"
    }
    if version("snnlab") not in expected_snnlab:
        raise RuntimeError(
            f"worker SNNLab {version('snnlab')} does not match submitted lock {expected_snnlab}"
        )
    sys.path[:0] = [str(REMOTE), str(REMOTE / "tools")]
    from pingstore.contracts import (
        validate_operational_run_directory,
        write_json_atomic,
    )
    from pingstore.layout import initialize_layout

    if experiment not in SUPPORTED or reservation["experiment"] != experiment:
        raise ValueError("unsupported experiment or mismatched reservation")
    input_refs = input_refs or {}
    if input_refs != (submission or {}).get("inputs", {}):
        raise ValueError("worker inputs differ from dispatch receipt")
    from experiments.helpers.modal_inputs import restore_inputs
    from pingstore.stages import source_run

    results.reload()
    restore_inputs(REMOTE / ".pingstore", Path("/modal-output"), input_runs or [])
    for pin in input_refs.values():
        source_run(REMOTE / ".pingstore", pin["run_id"], reference=pin)
    run_id = reservation["run_id"]
    temporary = REMOTE / ".pingstore" / "runs" / f".{run_id}.tmp"
    temporary.mkdir(parents=True)
    initialize_layout(temporary, experiment)
    write_json_atomic(temporary / ".reservation.json", reservation)
    environment = dict(
        os.environ,
        PINGLAB_SMOKE="1" if smoke else "0",
        PINGLAB_DEVICE=device,
        EXP080_DEVICE=device,
        EXP081_DEVICE=device,
        PINGLAB_CPU_THREADS=str(cpu),
        PINGLAB_MODAL_CPU_CORES=str(cpu),
    )
    command = [
        sys.executable,
        f"experiments/{experiment}/compute.py",
        "--run-id",
        run_id,
    ]
    if experiment == "exp134":
        if device != "cpu" or smoke or set(input_refs) != {"trained_network", "images"}:
            raise ValueError(
                "exp134 requires full CPU compute and explicit checkpoint/image inputs"
            )
        command.extend(
            (
                "--source",
                input_refs["trained_network"]["run_id"],
                "--image-source",
                input_refs["images"]["run_id"],
            )
        )
    if experiment == "exp124":
        command.extend(("--seed", str(seed)))
        if smoke:
            command.extend(("--smoke", "--epochs", str(epochs or 1)))
        elif epochs:
            command.extend(("--epochs", str(epochs)))
        if batch_size:
            command.extend(("--batch-size", str(batch_size)))
    subprocess.run(
        command,
        cwd=REMOTE,
        env=environment,
        check=True,
    )
    completed = temporary.with_name(run_id)
    record = validate_operational_run_directory(completed)
    if record["inputs"] != input_refs:
        raise ValueError("completed Modal compute used different inputs")
    record["execution"]["modal"] = {
        **(submission or {}),
        "function_call_id": modal.current_function_call_id(),
        "dispatch": dispatch,
    }
    write_json_atomic(completed / "run.json", record)
    validate_operational_run_directory(completed)
    destination = (
        Path("/modal-output").resolve() / dispatch / ".pingstore" / "runs" / run_id
    )
    staging = destination.with_name(f".{run_id}.tmp")
    shutil.copytree(completed, staging)
    validate_operational_run_directory(staging)
    staging.rename(destination)
    files = [
        str(path.relative_to(destination))
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    ]
    write_json_atomic(
        destination.parents[2] / "completion.json",
        {"run_id": run_id, "dispatch": dispatch, "files": files},
    )
    results.commit()
    return files


@app.local_entrypoint()
def main(
    experiment: str,
    smoke: bool = False,
    gpu: str = "A10G",
    cpu: int = 4,
    epochs: int = 0,
    timeout: int = 86400,
    batch_size: int = 0,
    seed: int = 42,
    source: str = "",
    image_source: str = "",
):
    from experiments.helpers.modal_inputs import prepare_inputs, upload_inputs
    from experiments.helpers.modal_jobs import submit
    from pingstore.stages import operation_lock

    if "--detach" not in sys.argv:
        raise ValueError("durable dispatch requires: modal run --detach ...")

    if experiment not in SUPPORTED:
        raise ValueError(f"experiment must be one of {', '.join(SUPPORTED)}")
    if cpu < 1:
        raise ValueError("cpu must be positive")
    if not 1 <= timeout <= 86400:
        raise ValueError("timeout must be between one second and 24 hours")
    if epochs < 0 or batch_size < 0:
        raise ValueError(
            "epochs and batch size must be positive, or zero to use defaults"
        )
    if (epochs or batch_size or seed != 42) and experiment != "exp124":
        raise ValueError("training overrides require exp124")
    gpu_resource = None if gpu.lower() == "none" else gpu
    device = "cpu" if gpu_resource is None else "cuda"
    root = REPO / ".pingstore"
    if experiment == "exp134":
        if device != "cpu" or smoke or not source or not image_source:
            raise ValueError(
                "exp134 requires --gpu none, --source and --image-source, without --smoke"
            )
    elif source or image_source:
        raise ValueError("explicit source roles are currently defined for exp134")
    # Keep source evidence protected throughout transfer and dispatch.
    with operation_lock(root, exclusive=False):
        input_refs, sources = prepare_inputs(
            root, {"trained_network": source, "images": image_source} if source else {}
        )
        upload_inputs(results, sources)
        receipt = submit(
            root,
            experiment,
            compute,
            app_id=app.app_id,
            options={"gpu": gpu_resource, "cpu": cpu, "timeout": timeout},
            arguments={
                "smoke": smoke,
                "device": device,
                "cpu": cpu,
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed,
                "input_refs": input_refs,
                "input_runs": [s.reference for s in sources.values()],
            },
            inputs=input_refs,
        )
    print(
        f"Dispatched {receipt['run_id']}; call {receipt['function_call_id']}",
        flush=True,
    )
    print(
        f"The job is independent of this process. Check/collect later with:\n"
        f"  uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML "
        f"python experiments/helpers/modal_jobs.py --run-id {receipt['run_id']} --collect",
        flush=True,
    )
