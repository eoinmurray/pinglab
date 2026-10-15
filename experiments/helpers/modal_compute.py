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
SUPPORTED = ("exp047", "exp080", "exp081", "exp124")
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
            full_dataset: bool = False,
            variant: str = "", validation_fraction: float = 0.1, models: str = "", batch_size: int = 0,
            readout_gain: float = 0.0, seed: int = 42, submission: dict | None = None) -> list[str]:
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
    expected_snnlab = {row["version"] for row in locked["package"] if row["name"] == "snnlab"}
    if version("snnlab") not in expected_snnlab:
        raise RuntimeError(f"worker SNNLab {version('snnlab')} does not match submitted lock {expected_snnlab}")
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
    if experiment == "exp124":
        command.extend(("--seed", str(seed)))
    if readout_gain:
        if experiment != "exp124" or readout_gain <= 0:
            raise ValueError("readout gain override requires exp124 and a positive value")
        command.extend(("--readout-gain", str(readout_gain)))
    if batch_size:
        if experiment != "exp124" or batch_size < 1:
            raise ValueError("batch size override requires exp124 and a positive size")
        command.extend(("--batch-size", str(batch_size)))
    if models:
        if experiment != "exp124" or any(name not in ("coba", "ping") for name in models.split(",")):
            raise ValueError("models must select SHD conditions")
        command.extend(("--models", *models.split(",")))
    if full_dataset or variant:
        if experiment != "exp124":
            raise ValueError("SHD recipe options require exp124")
        if full_dataset:
            command.extend(("--full-dataset", "--validation-fraction", str(validation_fraction)))
        if variant:
            command.extend(("--variant", variant))
    for option, value in (("--train-examples", train_examples),
                          ("--validation-examples", validation_examples), ("--epochs", epochs)):
        if value:
            command.extend((option, str(value)))
    subprocess.run(
        command,
        cwd=REMOTE, env=environment, check=True,
    )
    completed = temporary.with_name(run_id)
    record = validate_operational_run_directory(completed)
    record["execution"]["modal"] = {
        **(submission or {}), "function_call_id": modal.current_function_call_id(),
        "dispatch": dispatch,
    }
    write_json_atomic(completed / "run.json", record)
    validate_operational_run_directory(completed)
    destination = Path("/modal-output").resolve() / dispatch / ".pingstore" / "runs" / run_id
    staging = destination.with_name(f".{run_id}.tmp")
    shutil.copytree(completed, staging)
    validate_operational_run_directory(staging)
    staging.rename(destination)
    files = [str(path.relative_to(destination)) for path in sorted(destination.rglob("*")) if path.is_file()]
    write_json_atomic(destination.parents[2] / "completion.json",
                      {"run_id": run_id, "dispatch": dispatch, "files": files})
    results.commit()
    return files


@app.local_entrypoint()
def main(experiment: str, smoke: bool = False, gpu: str = "A10G", cpu: int = 4,
         train_examples: int = 0, validation_examples: int = 0, epochs: int = 0,
         full_dataset: bool = False, variant: str = "", validation_fraction: float = 0.1,
         models: str = "", timeout: int = 86400, batch_size: int = 0, readout_gain: float = 0.0, seed: int = 42):
    from experiments.helpers.modal_jobs import submit

    if "--detach" not in sys.argv:
        raise ValueError("durable dispatch requires: modal run --detach ...")

    if experiment not in SUPPORTED:
        raise ValueError(f"experiment must be one of {', '.join(SUPPORTED)}")
    if cpu < 1:
        raise ValueError("cpu must be positive")
    if readout_gain < 0 or (readout_gain and experiment != "exp124"):
        raise ValueError("readout gain override requires exp124 and a positive value")
    if batch_size < 0 or (batch_size and experiment != "exp124"):
        raise ValueError("batch size override requires exp124 and a positive size")
    if not 1 <= timeout <= 86400:
        raise ValueError("timeout must be between one second and 24 hours")
    if models and (experiment != "exp124" or any(name not in ("coba", "ping") for name in models.split(","))
                   or len(set(models.split(","))) != len(models.split(","))):
        raise ValueError("models must select unique SHD conditions")
    if (full_dataset or variant) and experiment != "exp124":
        raise ValueError("SHD recipe options require exp124")
    if full_dataset and (smoke or train_examples or validation_examples):
        raise ValueError("full dataset cannot be combined with smoke or example-count overrides")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between zero and one")
    if variant and variant not in ("baseline", "diverse", "damping", "sum", "gain", "combined", "maximum", "tuned", "paper", "calibrated"):
        raise ValueError("unsupported SHD training variant")
    if full_dataset and variant == "tuned":
        raise ValueError("full-data Modal execution requires an unbalanced-batch profile such as paper")
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
    receipt = submit(
        root, experiment, compute, app_id=app.app_id,
        options={"gpu": gpu_resource, "cpu": cpu, "timeout": timeout},
        arguments={"smoke": smoke, "device": device, "cpu": cpu,
                   "train_examples": train_examples, "validation_examples": validation_examples,
                   "epochs": epochs, "full_dataset": full_dataset, "variant": variant,
                   "validation_fraction": validation_fraction, "models": models,
                   "batch_size": batch_size, "readout_gain": readout_gain, "seed": seed})
    print(f"Dispatched {receipt['run_id']}; call {receipt['function_call_id']}", flush=True)
    print(f"The job is independent of this process. Check/collect later with:\n"
          f"  uv run --no-project --python 3.12 --with 'modal>=1.1,<2' --with PyYAML "
          f"python experiments/helpers/modal_jobs.py --run-id {receipt['run_id']} --collect", flush=True)
