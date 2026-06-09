"""Provenance metadata (git SHA, env hash, run ID) and the .running marker."""

from __future__ import annotations

import datetime
import hashlib
import os
import subprocess
import sys
from pathlib import Path


def _git_sha() -> str:
    """Return current git SHA with '(dirty)' suffix if uncommitted changes.

    Honors PINGLAB_GIT_SHA env var as a fallback so Modal containers (which
    don't have .git mounted) still record the host's SHA.
    """
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            .decode()
            .strip()
        )
        dirty = subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD"],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            timeout=2,
        )
        return f"{sha} (dirty)" if dirty else sha
    except Exception:
        return os.environ.get("PINGLAB_GIT_SHA", "unknown")


def _env_hash() -> str:
    """Hash of uv.lock (or pyproject.toml fallback) for env reproducibility."""
    for name in ("uv.lock", "pyproject.toml"):
        p = Path(name)
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
            return f"{name}:{h}"
    return "unknown"


def run_id() -> str:
    """Compact run ID: r-YYYYMMDD-HHMMSS."""
    now = datetime.datetime.now()
    return f"r-{now.strftime('%Y%m%d-%H%M%S')}"


def provenance() -> dict:
    """Return a provenance dict to embed in config.json."""
    import torch

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return {
        "git_sha": _git_sha(),
        "torch_version": torch.__version__,
        "device": device,
        "python_env_hash": _env_hash(),
        "run_id": run_id(),
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def write_running_marker(out_dir: Path, run_id_str: str) -> Path:
    """Write enriched .running file with PID, start time, run_id, cmd.

    Deleted by atexit hook in caller.
    """
    marker = Path(out_dir) / ".running"
    try:
        marker.write_text(
            f"pid={os.getpid()}\n"
            f"started={datetime.datetime.now().isoformat(timespec='seconds')}\n"
            f"run_id={run_id_str}\n"
            f"cmd={' '.join(sys.argv)}\n"
        )
    except Exception:
        pass
    return marker
