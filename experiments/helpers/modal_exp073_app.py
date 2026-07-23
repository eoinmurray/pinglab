"""Top-level Modal app/function definitions for exp073.

Modal requires decorated functions to live at module scope.  The orchestration
and artifact collection stay in ``modal_backend``; this module is intentionally
thin so the experiment recipe remains the source of scientific parameters.
"""

from __future__ import annotations

import os
from typing import Any

import modal

from . import modal_backend

app = modal.App("pinglab-exp073")
image = modal_backend._source_image(modal)
gpu = os.environ.get("PINGLAB_MODAL_GPU", "L40S")


@app.function(image=image, gpu=gpu, timeout=14400)
def train_one(cell: str, attempt: str, stage: str, ping_only: bool) -> dict[str, Any]:
    return modal_backend._remote_train_exp073_cell(
        cell=cell,
        attempt=attempt,
        stage=stage,
        ping_only=ping_only,
    )
