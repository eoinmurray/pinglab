"""Short non-interactive Wilkes3 environment/data/write diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torchvision.datasets import MNIST

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments import exp022  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip())
    cli_root = Path("/tmp/mnist")
    if cli_root.resolve() != args.data_root.resolve():
        raise SystemExit("/tmp/mnist does not resolve to the reviewed persistent cache")
    dataset = MNIST(root=cli_root, train=True, download=False)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repository_commit": commit,
        "repository_dirty": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "mnist_train_samples": len(dataset),
        "exp022_registered_cells": len(exp022.CANONICAL_CELLS),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
    }
    payload["checks_passed"] = bool(
        not dirty
        and payload["cuda_available"]
        and payload["gpu_compute_capability"]
        and payload["gpu_compute_capability"][0] == 8
        and payload["mnist_train_samples"] == 60000
        and payload["exp022_registered_cells"] == 102
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["checks_passed"]:
        raise SystemExit("Wilkes3 diagnostic checks failed; inspect the JSON record")


if __name__ == "__main__":
    main()
