"""Modal application for exp080's frozen held-out evaluation."""

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

repo = modal_backend.REMOTE_REPO
image = modal_backend._source_image_exp080(modal).add_local_dir(
    str(modal_backend.REPO / "artifacts" / "data" / "exp080"),
    str(repo / "artifacts" / "data" / "exp080"),
    ignore=["step2_response_library.float32.npy"],
)
app = modal.App("pinglab-exp080-evaluation")
gpu = os.environ.get("PINGLAB_MODAL_GPU", "L40S")


def _archive(root: Path) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for path in sorted(root.rglob("*")):
            archive.add(path, arcname=path.relative_to(root))
    return payload.getvalue()


@app.function(image=image, gpu=gpu, timeout=43_200)
def evaluate() -> dict[str, Any]:
    """Run the frozen protocol and return only newly generated Step 6 files."""
    import sys

    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "experiments"))
    output = Path("/tmp/exp080-step6")
    started = time.monotonic()
    error = None
    try:
        from experiments import exp080

        exp080.evaluate_frozen_decoders(
            repo / "artifacts" / "data" / "exp080" / "frozen_evaluation_protocol.json",
            output,
        )
    except BaseException:  # noqa: BLE001 - return traceback for the ledger
        error = traceback.format_exc()
    payload = _archive(output)
    return {
        "elapsed_s": time.monotonic() - started,
        "success": error is None,
        "error": error,
        "artifact_tar_gz": payload,
        "artifact_sha256": modal_backend.sha256_bytes(payload),
    }
