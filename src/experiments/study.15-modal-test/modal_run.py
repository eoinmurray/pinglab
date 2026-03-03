"""Run study.15 on Modal with a T4 GPU.

Usage:
    uv run modal run src/experiments/study.15-modal-test/modal_run.py
"""

import modal

app = modal.App("pinglab-study15")

volume = modal.Volume.from_name("pinglab-artifacts", create_if_missing=True)
VOLUME_PATH = "/root/artifacts"
STUDY_NAME = "study.15-modal-test"

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
    .add_local_dir(
        "src/experiments/study.15-modal-test",
        remote_path="/root/experiment",
    )
)


@app.function(
    image=image,
    gpu="T4",
    volumes={VOLUME_PATH: volume},
    timeout=14400,
)
def train():
    import json
    import shutil
    import sys
    from pathlib import Path

    # Write a stub settings module so `from settings import ARTIFACTS_ROOT` resolves.
    # The actual artifacts_dir is overridden via the parameter, but the import must succeed.
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

    # Patch sys.path so run.py's `from plots import ...` works
    sys.path.insert(0, str(work_dir))

    # Import and run
    # We need to adjust __file__ context — easiest is to exec or importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location("run", work_dir / "run.py")
    run_module = importlib.util.module_from_spec(spec)
    # Patch the module's __file__ so Path(__file__).parent works
    run_module.__file__ = str(work_dir / "run.py")
    spec.loader.exec_module(run_module)

    artifacts_dir = Path(VOLUME_PATH) / STUDY_NAME
    data_dir = Path("/tmp/mnist")

    results = run_module.main(
        artifacts_dir=artifacts_dir,
        data_dir=data_dir,
        checkpoint_dir=artifacts_dir,
        on_epoch_end=volume.commit,
    )

    volume.commit()

    print("\n=== Results ===")
    print(json.dumps(results, indent=2))
    return results


@app.local_entrypoint()
def main():
    import subprocess, sys, shutil, tempfile, json
    from pathlib import Path

    # Wipe local artifacts dir upfront so stale results aren't visible during the run
    local_dest = Path(__file__).resolve().parent.parent.parent / "posts" / "_artifacts" / STUDY_NAME
    if local_dest.exists():
        shutil.rmtree(local_dest)

    results = train.remote()
    print("\n=== Remote results ===")
    print(json.dumps(results, indent=2))

    # Download artifacts from Volume to local _artifacts dir
    local_dest.mkdir(parents=True, exist_ok=True)

    # Download volume subdir into a temp location, then move contents up
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable, "-m", "modal", "volume", "get",
            "pinglab-artifacts",
            STUDY_NAME,
            tmp,
            "--force",
        ]
        print(f"\nDownloading artifacts to {local_dest} ...")
        subprocess.run(cmd, check=True)

        # modal volume get creates <tmp>/study.15-modal-test/*, move contents up
        downloaded = Path(tmp) / STUDY_NAME
        if downloaded.is_dir():
            for item in downloaded.iterdir():
                shutil.move(str(item), str(local_dest / item.name))
        else:
            # Files landed directly in tmp
            for item in Path(tmp).iterdir():
                shutil.move(str(item), str(local_dest / item.name))

    print("Artifacts downloaded.")
