from __future__ import annotations

import time
from typing import Any

from fastapi import Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
import pyarrow as pa

try:
    from fastapi.responses import ORJSONResponse as FastResponse
except ImportError:  # pragma: no cover - fallback when orjson is unavailable
    FastResponse = JSONResponse

from pinglab.service.contracts import (
    RunRequest,
    RunResponse,
    WeightsRequest,
    WeightsResponse,
    build_weights_preview,
    run_simulation,
    run_timing_headers,
)

ARROW_MEDIA_TYPE = "application/vnd.apache.arrow.stream"


def _wants_arrow(request: Request) -> bool:
    accept = request.headers.get("accept", "")
    return ARROW_MEDIA_TYPE in accept


def _to_arrow_stream_bytes(payload: dict[str, Any]) -> bytes:
    table = pa.Table.from_pylist([payload])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _run_response_from_result(result: RunResponse, request: Request) -> Response:
    payload = result.model_dump(mode="json")
    serialize_start = time.perf_counter()
    if _wants_arrow(request):
        body = _to_arrow_stream_bytes(payload)
        response = Response(content=body, media_type=ARROW_MEDIA_TYPE)
    else:
        response = FastResponse(content=payload)
    serialize_ms = (time.perf_counter() - serialize_start) * 1000.0
    for key, value in run_timing_headers(
        result,
        serialize_ms=serialize_ms,
        response_bytes=len(response.body),
    ).items():
        response.headers[key] = value
    return response


def weights_from_payload(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    request = WeightsRequest.model_validate(payload or {})
    response = build_weights_preview(request)
    return response.model_dump(mode="json")


def create_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Pinglab-Core-Sim-Ms",
            "X-Pinglab-Input-Prep-Ms",
            "X-Pinglab-Weights-Build-Ms",
            "X-Pinglab-Analysis-Ms",
            "X-Pinglab-Response-Build-Ms",
            "X-Pinglab-Server-Compute-Ms",
            "X-Pinglab-Serialize-Ms",
            "X-Pinglab-Response-Bytes",
        ],
    )

    @app.get("/run", response_model=RunResponse)
    def run_get(request: Request) -> Response:
        return _run_response_from_result(run_simulation(RunRequest()), request)

    @app.post("/run", response_model=RunResponse)
    def run_post(
        request: Request, payload: dict[str, Any] | None = Body(default=None)
    ) -> Response:
        run_request = RunRequest.model_validate(payload or {})
        return _run_response_from_result(run_simulation(run_request), request)

    @app.get("/weights", response_model=WeightsResponse)
    def weights_get(request: Request) -> Response | dict[str, Any]:
        payload = weights_from_payload(None)
        if _wants_arrow(request):
            return Response(
                content=_to_arrow_stream_bytes(payload),
                media_type=ARROW_MEDIA_TYPE,
            )
        return payload

    @app.post("/weights", response_model=WeightsResponse)
    def weights_post(
        request: Request, payload: dict[str, Any] | None = Body(default=None)
    ) -> Response | dict[str, Any]:
        resolved_payload = weights_from_payload(payload)
        if _wants_arrow(request):
            return Response(
                content=_to_arrow_stream_bytes(resolved_payload),
                media_type=ARROW_MEDIA_TYPE,
            )
        return resolved_payload

    return app
