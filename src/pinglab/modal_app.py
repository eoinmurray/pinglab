"""Modal remote execution for oscilloscope train/infer.

Wraps the oscilloscope CLI so the same args run identically on Modal's
infrastructure. Artifacts are written to a Modal Volume and synced back
to the local out-dir after completion.

Usage (via oscilloscope --modal flag):
    uv run python src/pinglab/oscilloscope.py train --modal --model cuba ...
    (oscilloscope intercepts --modal and calls this module)
"""
from __future__ import annotations

import modal

VOLUME_NAME = "pinglab-artifacts"
REMOTE_ARTIFACTS = "/artifacts"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.2",
        "torchvision",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "snntorch>=0.9.4",
        "imageio",
        "h5py",
    )
    .apt_install("ffmpeg", "curl")
    .run_commands("python -c \"from torchvision import datasets; datasets.MNIST('/tmp/mnist', train=True, download=True); datasets.MNIST('/tmp/mnist', train=False, download=True)\"")
    .run_commands(
        "mkdir -p /tmp/shd/SHD && "
        "curl -fL https://zenkelab.org/datasets/shd_train.h5.gz -o /tmp/shd/SHD/shd_train.h5.gz && "
        "curl -fL https://zenkelab.org/datasets/shd_test.h5.gz -o /tmp/shd/SHD/shd_test.h5.gz && "
        "gunzip /tmp/shd/SHD/shd_train.h5.gz && "
        "gunzip /tmp/shd/SHD/shd_test.h5.gz"
    )
    .add_local_dir("src/pinglab", remote_path="/root/pinglab")
)

app = modal.App("pinglab")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _run_oscilloscope_impl(cli_args: list[str]) -> str:
    import subprocess
    import sys

    args = list(cli_args)

    def _remap_to_volume(path: str) -> str:
        """Translate a local src/artifacts/... path to the Modal volume mount.
        Handles both relative (src/artifacts/...) and absolute
        (/Users/.../src/artifacts/...) paths."""
        marker = "src/artifacts/"
        idx = path.find(marker)
        if idx != -1:
            return f"{REMOTE_ARTIFACTS}/{path[idx + len(marker):]}"
        for prefix in ("artifacts/", "artifacts"):
            if path.startswith(prefix):
                return f"{REMOTE_ARTIFACTS}/{path[len(prefix):]}"
        return path

    if "--out-dir" in args:
        idx = args.index("--out-dir")
        remote_out = _remap_to_volume(args[idx + 1])
        args[idx + 1] = remote_out
    else:
        remote_out = f"{REMOTE_ARTIFACTS}/default"
        args.extend(["--out-dir", remote_out])

    # Remap --from-dir (training-run dir used by infer --dt-sweep etc.) and
    # --load-weights so they point at the volume inside the container.
    for flag in ("--from-dir", "--load-weights"):
        if flag in args:
            idx = args.index(flag)
            args[idx + 1] = _remap_to_volume(args[idx + 1])

    # Use CUDA if available
    if "--device" not in args:
        import torch
        if torch.cuda.is_available():
            args.extend(["--device", "cuda"])

    cmd = [sys.executable, "/root/pinglab/oscilloscope.py", *args]
    print(f"modal> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/root/pinglab", capture_output=False)
    volume.commit()

    if result.returncode != 0:
        raise RuntimeError(f"oscilloscope exited with code {result.returncode}")

    return remote_out


@app.function(
    image=image,
    volumes={REMOTE_ARTIFACTS: volume},
    timeout=43200,
    cpu=4,
)
def run_cpu(cli_args: list[str]) -> str:
    return _run_oscilloscope_impl(cli_args)


@app.function(
    image=image,
    volumes={REMOTE_ARTIFACTS: volume},
    timeout=43200,
    gpu="T4",
)
def run_gpu_t4(cli_args: list[str]) -> str:
    return _run_oscilloscope_impl(cli_args)


@app.function(
    image=image,
    volumes={REMOTE_ARTIFACTS: volume},
    timeout=43200,
    gpu="A10G",
)
def run_gpu_a10g(cli_args: list[str]) -> str:
    return _run_oscilloscope_impl(cli_args)


@app.function(
    image=image,
    volumes={REMOTE_ARTIFACTS: volume},
    timeout=43200,
    gpu="L4",
)
def run_gpu_l4(cli_args: list[str]) -> str:
    return _run_oscilloscope_impl(cli_args)


@app.function(
    image=image,
    volumes={REMOTE_ARTIFACTS: volume},
    timeout=43200,
    gpu="A100-80GB",
)
def run_gpu_a100(cli_args: list[str]) -> str:
    return _run_oscilloscope_impl(cli_args)


@app.function(
    image=image,
    volumes={REMOTE_ARTIFACTS: volume},
    timeout=43200,
    gpu="H100",
)
def run_gpu_h100(cli_args: list[str]) -> str:
    return _run_oscilloscope_impl(cli_args)


_GPU_DISPATCH = {
    "none": run_cpu,
    "T4": run_gpu_t4,
    "L4": run_gpu_l4,
    "A10G": run_gpu_a10g,
    "A100": run_gpu_a100,
    "H100": run_gpu_h100,
}


def dispatch_to_modal(cli_args: list[str], local_out_dir: str, gpu: str = "T4"):
    """Called from oscilloscope.py when --modal is set.

    Submits the job to Modal, waits for completion, then syncs artifacts
    back to the local out-dir.
    """
    import subprocess
    from pathlib import Path

    fn = _GPU_DISPATCH.get(gpu)
    if fn is None:
        raise ValueError(f"Unknown --modal-gpu {gpu!r}, choose from {list(_GPU_DISPATCH)}")

    print(f"Dispatching to Modal (gpu={gpu})...")
    print(f"  args: {' '.join(cli_args)}")
    print(f"  local out-dir: {local_out_dir}")

    with modal.enable_output():
        # detach=True so the remote app keeps running even if the local
        # client disconnects (e.g. Taskfile parallel dispatcher exits after
        # all 8 deps are kicked off).
        with app.run(detach=True):
            remote_out = fn.remote(cli_args)

    volume_path = remote_out.removeprefix(REMOTE_ARTIFACTS + "/")
    local_path = Path(local_out_dir)
    local_parent = local_path.parent
    local_parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSyncing artifacts from Modal Volume...")
    sync_cmd = [
        "uv", "run", "modal", "volume", "get", VOLUME_NAME,
        volume_path + "/", str(local_parent) + "/",
        "--force",
    ]
    subprocess.run(sync_cmd, check=True)
    print(f"  → {local_path}/")
