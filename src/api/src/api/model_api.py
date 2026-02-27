from __future__ import annotations

import modal
from fastapi import FastAPI

from api.web_app import create_app


app = modal.App("pinglab-api")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "fastapi[standard]>=0.115.0",
        "orjson>=3.10.0",
        "numpy>=2.2.6",
        "scipy>=1.15.3",
        "torch>=2.2.0",
        "matplotlib>=3.10.0",
        "joblib>=1.5.2",
        "tqdm-joblib>=0.0.5",
        "pydantic>=2.12.4",
        "pyarrow>=23.0.0",
        "pyyaml>=6.0.2",
        "tqdm>=4.67.1",
    )
    .add_local_python_source("api", "pinglab")
)

web_app = create_app()


@app.function(
    image=image,
    timeout=600,
    cpu=2.0,
    memory=4096,
    min_containers=0,
    scaledown_window=900,
)
@modal.asgi_app()
def modal_api() -> FastAPI:
    return web_app
