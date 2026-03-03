"""Shared Modal GPU runner for any pinglab study.

Usage:
    uv run modal run src/scripts/modal_run.py --study study.15-modal-test
    uv run modal run --detach src/scripts/modal_run.py --study study.14-ping-snn-poisson-input
"""
from __future__ import annotations

import modal

app = modal.App("pinglab-study")

volume = modal.Volume.from_name("pinglab-artifacts", create_if_missing=True)
VOLUME_PATH = "/root/artifacts"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("graphviz")
    .pip_install(
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy>=2.0",
        "scipy>=1.12",
        "matplotlib>=3.8",
        "pyyaml>=6.0",
        "graphviz>=0.20",
        "pydantic>=2.12",
        "rich>=13.9",
        "tqdm>=4.67",
    )
    .add_local_python_source("pinglab")
)


def _resolve_study_dir(study_name: str) -> str:
    """Return the local experiment directory path for a study name."""
    import os
    base = os.path.join(os.path.dirname(__file__), "..", "experiments")
    # Try exact match first
    candidate = os.path.join(base, study_name)
    if os.path.isdir(candidate):
        return os.path.abspath(candidate)
    # Try prefix match (e.g. "study.15" -> "study.15-modal-test")
    import glob
    matches = glob.glob(os.path.join(base, f"{study_name}-*"))
    dirs = [m for m in matches if os.path.isdir(m)]
    if len(dirs) == 1:
        return os.path.abspath(dirs[0])
    if len(dirs) > 1:
        raise ValueError(f"Ambiguous study selector '{study_name}': {dirs}")
    raise FileNotFoundError(f"No study matching '{study_name}' in {base}")


def _read_modal_config(study_dir: str) -> dict:
    """Read config.json \"modal\" section for resource settings.

    Expected format in config.json:
        "modal": {
            "gpu": "T4",        # GPU type (default: "T4")
            "cpu": 2,           # CPU count (default: 2)
            "memory": 4096,     # Memory in MB (default: not set / Modal default)
            "timeout": 14400    # Timeout in seconds (default: 14400 = 4h)
        }
    """
    import json, os
    config_path = os.path.join(study_dir, "config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        return json.load(f).get("modal", {})


# We need the study name at import time so Modal can build the image with
# the correct experiment files. Use an env var set by the local entrypoint.
import os as _os
_STUDY_NAME = _os.environ.get("PINGLAB_STUDY", "")
_STUDY_DIR = ""
_MODAL_CFG: dict = {}

if _STUDY_NAME:
    try:
        _STUDY_DIR = _resolve_study_dir(_STUDY_NAME)
        _MODAL_CFG = _read_modal_config(_STUDY_DIR)
        # Dynamically add the study's experiment directory to the image
        image = image.add_local_dir(
            _STUDY_DIR,
            remote_path="/root/experiment",
        )
    except (FileNotFoundError, ValueError):
        pass

_GPU = _MODAL_CFG.get("gpu", "T4")
_CPU = float(_MODAL_CFG.get("cpu", 2))
_MEMORY = int(_MODAL_CFG["memory"]) if "memory" in _MODAL_CFG else None
_TIMEOUT = int(_MODAL_CFG.get("timeout", 14400))

_func_kwargs: dict = dict(
    image=image,
    gpu=_GPU,
    cpu=_CPU,
    volumes={VOLUME_PATH: volume},
    timeout=_TIMEOUT,
)
if _MEMORY is not None:
    _func_kwargs["memory"] = _MEMORY


@app.function(**_func_kwargs)
def run_study(study_name: str):
    import inspect
    import json
    import shutil
    import sys
    from pathlib import Path

    # Write a stub settings module so `from settings import ARTIFACTS_ROOT` resolves.
    stub = Path("/root/settings.py")
    stub.write_text(
        "from pathlib import Path\n"
        "ARTIFACTS_ROOT = Path('/tmp/artifacts')\n"
    )
    sys.path.insert(0, "/root")

    # Copy experiment files to a writable working directory
    work_dir = Path("/root/work")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    shutil.copytree("/root/experiment", work_dir)

    # Patch sys.path so run.py's local imports (e.g. `from plots import ...`) work
    sys.path.insert(0, str(work_dir))

    # Import the study's run module
    import importlib.util
    spec = importlib.util.spec_from_file_location("run", work_dir / "run.py")
    run_module = importlib.util.module_from_spec(spec)
    run_module.__file__ = str(work_dir / "run.py")
    spec.loader.exec_module(run_module)

    # Detect which parameters main() accepts
    sig = inspect.signature(run_module.main)
    params = set(sig.parameters.keys())

    artifacts_dir = Path(VOLUME_PATH) / study_name
    data_dir = Path("/tmp/data")

    kwargs: dict = {}
    if "artifacts_dir" in params:
        kwargs["artifacts_dir"] = artifacts_dir
    if "data_dir" in params:
        kwargs["data_dir"] = data_dir
    if "checkpoint_dir" in params:
        kwargs["checkpoint_dir"] = artifacts_dir
    if "on_epoch_end" in params:
        kwargs["on_epoch_end"] = volume.commit

    results = run_module.main(**kwargs)

    volume.commit()

    if results is not None:
        print("\n=== Results ===")
        print(json.dumps(results, indent=2))
    return results


@app.local_entrypoint()
def main(study: str = ""):
    import subprocess
    import sys
    import shutil
    import tempfile
    import json
    from pathlib import Path

    if not study:
        print("Usage: modal run src/scripts/modal_run.py --study <study-name>")
        sys.exit(1)

    # Resolve full study name for artifact paths
    study_dir = _resolve_study_dir(study)
    study_name = Path(study_dir).name

    # Wipe local artifacts dir upfront so stale results aren't visible
    local_dest = Path(__file__).resolve().parent.parent / "posts" / "_artifacts" / study_name
    if local_dest.exists():
        shutil.rmtree(local_dest)

    results = run_study.remote(study_name)
    if results is not None:
        print("\n=== Remote results ===")
        print(json.dumps(results, indent=2))

    # Download artifacts from Volume to local _artifacts dir
    local_dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable, "-m", "modal", "volume", "get",
            "pinglab-artifacts",
            study_name,
            tmp,
            "--force",
        ]
        print(f"\nDownloading artifacts to {local_dest} ...")
        subprocess.run(cmd, check=True)

        downloaded = Path(tmp) / study_name
        if downloaded.is_dir():
            for item in downloaded.iterdir():
                shutil.move(str(item), str(local_dest / item.name))
        else:
            for item in Path(tmp).iterdir():
                shutil.move(str(item), str(local_dest / item.name))

    print("Artifacts downloaded.")
