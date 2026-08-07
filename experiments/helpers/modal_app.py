"""Top-level Modal app/function definitions for generic experiment jobs.

Modal requires decorated functions to live at module scope.  Runner selection,
job identity, recipe environment, and artifact paths are supplied by the local
dispatcher; scientific parameters remain owned by the experiment runner.
"""

from __future__ import annotations

import os
from typing import Any

import modal

from . import modal_backend

app = modal.App("pinglab-experiments")
image = modal_backend._source_image(modal)
gpu = os.environ.get("PINGLAB_MODAL_GPU", "L40S")


@app.function(image=image, gpu=gpu, timeout=modal_backend.MAX_RUNTIME_S)
def run_job(
    slug: str,
    runner: str,
    job_id: str,
    env: dict[str, str],
    is_done_name: str,
    run_job_name: str,
) -> dict[str, Any]:
    return modal_backend._remote_run_job(
        slug=slug,
        runner=runner,
        job_id=job_id,
        env=env,
        is_done_name=is_done_name,
        run_job_name=run_job_name,
    )
