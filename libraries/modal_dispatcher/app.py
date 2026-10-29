"""Modal image and durable worker for the experiment dispatcher.

Use the package CLI to dispatch; importing this module configures the Modal app.
Status checks import only jobs.py and never prepare this image.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal

REMOTE = Path("/root/pinglab")
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE
SOURCE = Path("/root/pinglab-source")
OUTPUT = Path("/modal-output")
SOURCE_DIRS = ("experiments", "libraries")
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

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
        source_bundle = os.environ.get("PINGLAB_MODAL_GIT_BUNDLE")
        baseline = os.environ.get("PINGLAB_MODAL_BASE_COMMIT", commit)
        if len(baseline) != 40 or any(c not in "0123456789abcdef" for c in baseline):
            raise ValueError("Modal baseline must be a full Git commit hash")
        if source_bundle:
            bundle_heads = subprocess.check_output(
                ["git", "bundle", "list-heads", source_bundle, "HEAD"],
                cwd=REPO,
                text=True,
            ).split()
            if bundle_heads != [commit, "HEAD"]:
                raise ValueError(
                    "source bundle must contain the exact clean checkout HEAD"
                )
        elif baseline != commit:
            raise ValueError("an alternate baseline requires an explicit source bundle")
        image = (
            modal.Image.debian_slim(python_version="3.12")
            .apt_install(
                "git",
                "pkg-config",
                "libcairo2-dev",
                "libpango1.0-dev",
                "ffmpeg",
                "graphviz",
            )
            .run_commands(
                f"git init {REMOTE}",
                f"git -C {REMOTE} remote add origin https://github.com/eoinmurray/pinglab",
                f"git -C {REMOTE} fetch --depth 1 origin {baseline}",
                f"git -C {REMOTE} reset --hard FETCH_HEAD",
            )
        )
        if source_bundle:
            image = image.add_local_file(
                source_bundle, "/tmp/pinglab-source.bundle", copy=True
            ).run_commands(
                f"git -C {REMOTE} fetch /tmp/pinglab-source.bundle HEAD",
                f"git -C {REMOTE} reset --hard FETCH_HEAD",
                "rm /tmp/pinglab-source.bundle",
            )
        image = (
            image.uv_sync(str(REPO), extra_options="--no-dev")
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
    volumes={str(OUTPUT): results},
    retries=0,
)
def compute(
    experiment: str,
    reservation: dict,
    dispatch: str,
    arguments: list[str],
    environment: dict[str, str],
    input_refs: dict | None = None,
    input_runs: list | None = None,
    submission: dict | None = None,
) -> list[str]:
    # Older image baselines contain tools/ instead of libraries/. Remove both
    # layouts before copying current sources, including local deletions.
    for directory in (*SOURCE_DIRS, "tools"):
        if (REMOTE / directory).exists():
            shutil.rmtree(REMOTE / directory)
    for directory in SOURCE_DIRS:
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
    sys.path[:0] = [str(REMOTE), str(REMOTE / "libraries")]
    from pingstore.library import (
        initialize_layout,
        validate_operational_run_directory,
        write_json_atomic,
    )

    if reservation["experiment"] != experiment:
        raise ValueError("unsupported experiment or mismatched reservation")
    input_refs = input_refs or {}
    if input_refs != (submission or {}).get("inputs", {}):
        raise ValueError("worker inputs differ from dispatch receipt")
    from libraries.modal_dispatcher.inputs import restore_inputs
    from libraries.modal_dispatcher.request import validate_request

    validate_request(REMOTE, experiment, arguments, environment)
    plan = (submission or {}).get("plan", {})
    if plan.get("arguments") != arguments or plan.get("environment") != environment:
        raise ValueError("worker command differs from dispatch receipt")
    from pingstore.library import source_run

    results.reload()
    restore_inputs(REMOTE / ".pingstore", OUTPUT, input_runs or [])
    for pin in input_refs.values():
        source_run(REMOTE / ".pingstore", pin["run_id"], reference=pin)
    run_id = reservation["run_id"]
    temporary = REMOTE / ".pingstore" / "runs" / f".{run_id}.tmp"
    temporary.mkdir(parents=True)
    initialize_layout(temporary, experiment)
    write_json_atomic(temporary / ".reservation.json", reservation)
    worker_environment = dict(os.environ, **environment)
    command = [
        sys.executable,
        f"experiments/{experiment}/compute.py",
        "--run-id",
        run_id,
        *arguments,
    ]

    subprocess.run(
        command,
        cwd=REMOTE,
        env=worker_environment,
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
    destination = OUTPUT.resolve() / dispatch / ".pingstore" / "runs" / run_id
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
    arguments: str = "[]",
    inputs: str = "{}",
    environment: str = "{}",
    gpu: str = "A10G",
    cpu: int = 4,
    memory: int = 16384,
    timeout: int = 86400,
    store: str = "",
):
    import json

    from libraries.modal_dispatcher.request import dispatch

    if "--detach" not in sys.argv:
        raise ValueError("durable dispatch requires: modal run --detach ...")
    receipt = dispatch(
        REPO,
        experiment,
        compute,
        results,
        app_id=app.app_id,
        arguments=json.loads(arguments),
        inputs=json.loads(inputs),
        environment=json.loads(environment),
        gpu=gpu,
        cpu=cpu,
        memory=memory,
        timeout=timeout,
        store=Path(store) if store else None,
    )
    print(json.dumps(receipt), flush=True)
    print(
        "Check/collect later with: uv run --no-project --python 3.12 "
        "--with 'modal>=1.1,<2' --with PyYAML "
        "python -m libraries.modal_dispatcher check "
        f"--run-id {receipt['run_id']} --collect"
        + (f" --root {str(store)!r}" if store else ""),
        flush=True,
    )
