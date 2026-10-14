"""Run a supported compute stage on Modal and fetch its completed v4 run."""

import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import modal

REMOTE = Path("/root/pinglab")
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE
SOURCE = Path("/root/pinglab-source")
SOURCE_DIRS = ("experiments", "tools")
SUPPORTED = ("exp047", "exp080", "exp081", "exp124", "exp126")
sys.path[:0] = [str(REPO), str(REPO / "tools")]

app = modal.App("pinglab-compute")
results = modal.Volume.from_name("pinglab-compute-results", create_if_missing=True)

# A real checkout lets the existing stage helper capture commit and dirty state.
# Current local sources are mounted over that baseline, including uncommitted edits.
if modal.is_local():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "pkg-config", "libcairo2-dev", "libpango1.0-dev", "ffmpeg")
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
            REPO / directory, str(SOURCE / directory),
            ignore=["**/__pycache__/**", "**/*.pyc", "**/.env", "**/.env.*"],
        )
    for filename in ("pyproject.toml", "uv.lock"):
        image = image.add_local_file(REPO / filename, str(SOURCE / filename))
else:
    image = None


@app.function(image=image, gpu="A10G", cpu=4, memory=16384, timeout=86400,
              serialized=True, include_source=False,
              volumes={"/modal-output": results}, retries=0)
def compute(experiment: str, reservation: dict, dispatch: str, smoke: bool,
            device: str = "cuda", cpu: int = 4, train_examples: int = 0,
            validation_examples: int = 0, epochs: int = 0,
            implementation: str = "", seed: int = 42) -> list[str]:
    # Replace whole directories so local deletions are reflected too.
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

    if experiment not in SUPPORTED or reservation["experiment"] != experiment:
        raise ValueError("unsupported experiment or mismatched reservation")
    run_id = reservation["run_id"]
    temporary = REMOTE / ".pingstore" / "runs" / f".{run_id}.tmp"
    temporary.mkdir(parents=True)
    initialize_layout(temporary, experiment)
    write_json_atomic(temporary / ".reservation.json", reservation)
    environment = dict(os.environ, PINGLAB_SMOKE="1" if smoke else "0",
                       PINGLAB_DEVICE=device, EXP080_DEVICE=device, EXP081_DEVICE=device,
                       PINGLAB_CPU_THREADS=str(cpu), PINGLAB_MODAL_CPU_CORES=str(cpu))
    command = [sys.executable, f"experiments/{experiment}/compute.py", "--run-id", run_id]
    if implementation:
        if experiment != "exp126" or implementation not in ("snntorch", "snnlab"):
            raise ValueError("invalid equation-matched comparison condition")
        command.extend(("--phase", implementation, "--seed", str(seed)))
    for option, value in (("--train-examples", train_examples),
                          ("--validation-examples", validation_examples), ("--epochs", epochs)):
        if value:
            command.extend((option, str(value)))
    subprocess.run(
        command,
        cwd=REMOTE, env=environment, check=True,
    )
    completed = temporary.with_name(run_id)
    validate_operational_run_directory(completed)
    destination = Path("/modal-output").resolve() / dispatch / ".pingstore" / "runs" / run_id
    staging = destination.with_name(f".{run_id}.tmp")
    shutil.copytree(completed, staging)
    validate_operational_run_directory(staging)
    staging.rename(destination)
    results.commit()
    return [str(path.relative_to(destination)) for path in sorted(destination.rglob("*"))
            if path.is_file()]


@app.function(image=image, gpu="A10:2", cpu=8, memory=32768, timeout=86400,
              serialized=True, include_source=False,
              volumes={"/modal-output": results}, retries=0)
def compute_pair(phases: list[dict], cpu_per_phase: int = 4) -> list[dict]:
    """Two independently completed phases, parallel processes on distinct GPUs."""
    import json
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

    if len(phases) != 2 or {row["implementation"] for row in phases} != {"snntorch", "snnlab"}:
        raise ValueError("paired allocation requires exactly both implementations")
    names = json.loads(subprocess.check_output([sys.executable, "-c",
        "import json,torch; print(json.dumps([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))"], text=True))
    if len(names) != 2 or names[0] != names[1]:
        raise ValueError(f"paired GPU hardware differs: {names}")
    print(f"Matched allocation: {names}; {cpu_per_phase * 2} shared CPUs; one GPU per phase", flush=True)
    # Download once before workers read the cache; prevent concurrent extraction.
    subprocess.run([sys.executable, "-c",
        "from torchvision.datasets import MNIST; MNIST('/tmp/pinglab-mnist',train=True,download=True); MNIST('/tmp/pinglab-mnist',train=False,download=True)"], cwd=REMOTE, check=True)
    processes = []
    for gpu_index, phase in enumerate(phases):
        reservation = phase["reservation"]
        identity = reservation["run_id"]
        if reservation.get("phase") != phase["implementation"] or reservation["experiment"] != "exp126":
            raise ValueError("phase reservation differs from work assignment")
        temporary = REMOTE / ".pingstore/runs" / f".{identity}.tmp"
        temporary.mkdir(parents=True)
        initialize_layout(temporary, "exp126")
        write_json_atomic(temporary / ".reservation.json", reservation)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_index),
            PINGLAB_DEVICE="cuda", PINGLAB_CPU_THREADS=str(cpu_per_phase),
            PINGLAB_MODAL_CPU_CORES=str(cpu_per_phase * 2), PINGLAB_MODAL_SHARED_HOST="1")
        command = [sys.executable, "experiments/exp126/compute.py", "--phase", phase["implementation"],
                   "--seed", str(phase["seed"]), "--run-id", identity]
        processes.append((phase, subprocess.Popen(command, cwd=REMOTE, env=environment)))
    # Both Popen calls precede any wait; children never share a CUDA device.
    completed, errors = [], []
    for phase, process in processes:
        code = process.wait()
        identity = phase["reservation"]["run_id"]
        if code:
            errors.append(f"{identity} exited with status {code}")
            continue
        source = REMOTE / ".pingstore/runs" / identity
        validate_operational_run_directory(source)
        destination = Path("/modal-output").resolve() / phase["dispatch"] / ".pingstore/runs" / identity
        staging = destination.with_name(f".{identity}.tmp")
        shutil.copytree(source, staging)
        validate_operational_run_directory(staging)
        staging.rename(destination)
        completed.append({**phase, "files": [str(path.relative_to(destination)) for path in sorted(destination.rglob("*")) if path.is_file()]})
    results.commit()
    if errors:
        raise RuntimeError("; ".join(errors))
    return completed


@app.function(image=image, cpu=4, memory=16384, timeout=3600,
              serialized=True, include_source=False,
              volumes={"/modal-output": results}, retries=0)
def assemble_exp126(reservation: dict, dispatch: str, phases: list[dict]) -> list[str]:
    """Explicit CPU-only assembly of completed phase evidence on the Volume."""
    for directory in SOURCE_DIRS:
        shutil.rmtree(REMOTE / directory)
        shutil.copytree(SOURCE / directory, REMOTE / directory)
    for filename in ("pyproject.toml", "uv.lock"):
        shutil.copyfile(SOURCE / filename, REMOTE / filename)
    sys.path[:0] = [str(REMOTE), str(REMOTE / "tools")]
    from experiments.exp126.compute_parts.assembly import assemble
    from pingstore.contracts import (
        run_root,
        validate_operational_run_directory,
        write_json_atomic,
    )
    from pingstore.layout import initialize_layout

    results.reload()
    identities = []
    for phase in phases:
        identity = phase["run_id"]
        source = Path("/modal-output").resolve() / phase["dispatch"] / ".pingstore/runs" / identity
        record = validate_operational_run_directory(source)
        if record["experiment"] != "exp126" or record.get("scope") != "phase":
            raise ValueError("assembly source is not an exp126 phase")
        destination = run_root(REMOTE / ".pingstore", identity)
        shutil.copytree(source, destination)
        validate_operational_run_directory(destination)
        identities.append(identity)
    identity = reservation["run_id"]
    temporary = run_root(REMOTE / ".pingstore", identity).with_name(f".{identity}.tmp")
    temporary.mkdir(parents=True)
    initialize_layout(temporary, "exp126")
    write_json_atomic(temporary / ".reservation.json", reservation)
    assemble(REMOTE, identities, identity)
    completed = temporary.with_name(identity)
    destination = Path("/modal-output").resolve() / dispatch / ".pingstore/runs" / identity
    staging = destination.with_name(f".{identity}.tmp")
    shutil.copytree(completed, staging)
    validate_operational_run_directory(staging)
    staging.rename(destination)
    results.commit()
    return [str(path.relative_to(destination)) for path in sorted(destination.rglob("*")) if path.is_file()]


@app.local_entrypoint()
def main(experiment: str, smoke: bool = False, gpu: str = "A10G", cpu: int = 4,
         train_examples: int = 0, validation_examples: int = 0, epochs: int = 0):
    from pingstore.contracts import load_json, validate_operational_run_directory
    from pingstore.stages import operation_lock, reserve_stage, stage_reservation

    if experiment not in SUPPORTED:
        raise ValueError(f"experiment must be one of {', '.join(SUPPORTED)}")
    if cpu < 1:
        raise ValueError("cpu must be positive")
    if any(value < 0 for value in (train_examples, validation_examples, epochs)):
        raise ValueError("training overrides must be positive, or zero to use defaults")
    if any((train_examples, validation_examples, epochs)):
        if experiment != "exp124":
            raise ValueError("training count overrides are supported only for exp124")
        if smoke:
            raise ValueError("use either smoke or explicit training overrides")
        if train_examples % 20 or validation_examples % 20:
            raise ValueError("SHD example counts must be multiples of 20")
    gpu_resource = None if gpu.lower() == "none" else gpu
    device = "cpu" if gpu_resource is None else "cuda"
    root = REPO / ".pingstore"
    dispatch = uuid.uuid4().hex
    with operation_lock(root, exclusive=False):
        run_id = reserve_stage(root, experiment, "compute", origin=f"modal-{dispatch}")
        temporary = root / "runs" / f".{run_id}.tmp"
        reservation = stage_reservation(temporary)
        remote_path = f"{dispatch}/.pingstore/runs/{run_id}"
        print(f"Reserved {run_id}; retained result: pinglab-compute-results/{remote_path}",
              flush=True)
        files = compute.with_options(gpu=gpu_resource, cpu=cpu).remote(
            experiment, reservation, dispatch, smoke, device, cpu,
            train_examples, validation_examples, epochs)
        # Download under the reservation; partial transfers stay hidden.
        download = temporary / ".scratch" / run_id
        download.mkdir(parents=True)
        for filename in files:
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe result path: {filename}")
            target = download / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                for chunk in results.read_file(f"{remote_path}/{filename}"):
                    handle.write(chunk)
        record = validate_operational_run_directory(download)
        if (record["run_id"] != run_id or record["experiment"] != experiment
                or record["stage"] != "compute" or record["origin"] != reservation["origin"]
                or record["inputs"]):
            raise ValueError("downloaded run does not match the dispatched compute")
        if load_json(temporary / ".reservation.json") != reservation:
            raise ValueError("local reservation changed during dispatch")
        destination = root / "runs" / run_id
        if destination.exists():
            raise FileExistsError(destination)
        download.rename(destination)
        shutil.rmtree(temporary)
    print(run_id)
