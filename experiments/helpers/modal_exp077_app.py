"""Modal application for one exp077 Step 5 training cell."""

from __future__ import annotations

import io
import os
import tarfile
import time
import traceback
from pathlib import Path
from typing import Any

import modal

from . import modal_backend

app = modal.App("pinglab-exp077")
image = modal_backend._source_image_exp077(modal)
gpu = os.environ.get("PINGLAB_MODAL_GPU", "L40S")


def _archive(root: Path) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        if root.exists():
            for path in sorted(root.rglob("*")):
                archive.add(path, arcname=path.relative_to(root))
    return payload.getvalue()


@app.function(image=image, gpu=gpu, timeout=43_200)
def train_cell(stage: str, probe_uS: float, seed: int) -> dict[str, Any]:
    """Train one conductance/seed cell and return its complete Step 5 subtree."""
    import sys

    repo = Path("/workspace/pinglab")
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "experiments"))
    started = time.monotonic()
    error = None
    try:
        os.environ.update(
            {
                "EXP077_STAGE": stage,
                "EXP077_PROBE_US": str(probe_uS),
                "EXP077_SEED": str(seed),
            }
        )
        from experiments.exp077 import step_5

        step_5()
    except BaseException:  # noqa: BLE001 - return remote traceback and artifacts
        error = traceback.format_exc()
    root = repo / "artifacts" / "data" / "exp077" / "step5"
    payload = _archive(root)
    return {
        "stage": stage,
        "probe_uS": probe_uS,
        "seed": seed,
        "elapsed_s": time.monotonic() - started,
        "success": error is None,
        "error": error,
        "artifact_tar_gz": payload,
        "artifact_sha256": modal_backend.sha256_bytes(payload),
    }
