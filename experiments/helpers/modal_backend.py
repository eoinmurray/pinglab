"""Minimal Modal backend for experiment cell jobs.

This is deliberately narrower than helpers/runpod.py.  RunPod remains the
historical fleet backend; Modal is introduced first as a synchronous, one-runner
escape hatch for exp073 after RunPod image/data transport became the bottleneck.

Design constraints:
  * no scientific parameters are accepted here;
  * the experiment runner remains the recipe;
  * source is shipped as local files, not through a prebuilt GHCR image;
  * each Modal function returns a compressed artifact subtree for local publish;
  * provider billing is recorded as a timestamp estimate until reconciled.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import REPO

REMOTE_REPO = Path("/workspace/pinglab")
REMOTE_ARTIFACTS = Path("/tmp/pinglab-artifacts/exp073")

# Modal's public pricing is per second for GPU time; CPU/memory are billed
# separately.  These are enough for a conservative experiment ledger, but not
# exact provider billing.
GPU_USD_PER_SECOND = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10G": 0.000306,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100": 0.000583,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_modal():
    try:
        import modal
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Modal backend requires the `modal` package. Run `uv sync` after this "
            "branch's pyproject update, then authenticate with `uv run modal setup`."
        ) from exc
    return modal


def _source_image(modal: Any):
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .uv_pip_install(
            "torch",
            "numpy",
            "scipy",
            "scikit-learn",
            "h5py",
            "matplotlib",
            "snntorch",
        )
        .add_local_dir(
            str(REPO / "experiments"),
            str(REMOTE_REPO / "experiments"),
            ignore=[
                "__pycache__",
                ".pytest_cache",
                "*.staging",
                "*.old-*",
            ],
        )
        .add_local_dir(str(REPO / "tools"), str(REMOTE_REPO / "tools"), ignore=["__pycache__"])
        .add_local_file(str(REPO / "README.md"), str(REMOTE_REPO / "README.md"))
        .add_local_file(str(REPO / "pyproject.toml"), str(REMOTE_REPO / "pyproject.toml"))
    )
    return image


def _tar_tree(root: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        if root.exists():
            for path in sorted(root.rglob("*")):
                archive.add(path, arcname=path.relative_to(root))
    return buffer.getvalue()


def _extract_tree(payload: bytes, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        destination_root = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if destination_root != target and destination_root not in target.parents:
                raise RuntimeError(f"refusing unsafe Modal artifact path: {member.name!r}")
        archive.extractall(destination)


def _remote_train_exp073_cell(
    *,
    cell: str,
    attempt: str,
    stage: str,
    ping_only: bool,
) -> dict[str, Any]:
    """Executed inside Modal. Kept top-level so Modal can serialize it cleanly."""
    import sys
    import traceback

    os.chdir(REMOTE_REPO)
    sys.path.insert(0, str(REMOTE_REPO / "experiments"))
    sys.path.insert(0, str(REMOTE_REPO / "tools" / "snn"))
    os.environ.update(
        {
            "CELLS": cell,
            "EXP073_ATTEMPT": attempt,
            "EXP073_STAGE": stage,
            "EXP073_PING_ONLY": "1" if ping_only else "0",
            "PINGLAB_ARTIFACTS_ROOT": str(REMOTE_ARTIFACTS),
            "PYTHONUNBUFFERED": "1",
        }
    )
    started_wall = time.monotonic()
    started_at = utc_now()
    error = None
    try:
        import exp073

        exp073.pod_run()
    except BaseException:  # noqa: BLE001 — serialize failure into the ledger
        error = traceback.format_exc()
    artifact_payload = _tar_tree(REMOTE_ARTIFACTS)
    elapsed_s = time.monotonic() - started_wall
    cell_root = REMOTE_ARTIFACTS / "cells" / stage / attempt / cell
    success = error is None and (cell_root / "checkpoint_selection.json").exists()
    return {
        "cell": cell,
        "attempt": attempt,
        "stage": stage,
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_s": elapsed_s,
        "success": success,
        "error": error,
        "artifact_tar_gz": artifact_payload,
        "artifact_tar_gz_sha256": sha256_bytes(artifact_payload),
    }


def dispatch_exp073(
    *,
    cells: list[str],
    attempt: str,
    stage: str,
    ping_only: bool,
    live: bool,
    local_collect_dir: Path,
    ledger_path: Path,
    timeout_s: int,
) -> None:
    """Run exp073 cells on Modal and collect artifacts synchronously."""
    gpu = os.environ.get("PINGLAB_MODAL_GPU", "L40S")
    print(f"{'LIVE' if live else 'DRY-RUN'}  runner=exp073  backend=modal  gpu={gpu}")
    print(f"jobs: {' '.join(cells)}")
    print("set PINGLAB_MODAL_GPU to choose a different Modal GPU SKU")
    if not live:
        print("\n(dry-run — nothing created. Re-run with --live to spend.)")
        return

    modal = _require_modal()
    app = modal.App("pinglab-exp073")
    image = _source_image(modal)

    @app.function(image=image, gpu=gpu, timeout=timeout_s, serialized=True)
    def train_one(cell: str, attempt: str, stage: str, ping_only: bool) -> dict[str, Any]:
        return _remote_train_exp073_cell(
            cell=cell,
            attempt=attempt,
            stage=stage,
            ping_only=ping_only,
        )

    output_context = getattr(modal, "enable_output", lambda: contextlib.nullcontext())()
    events: list[dict[str, Any]] = []
    started = utc_now()
    started_clock = time.monotonic()
    with output_context:
        with app.run():
            for cell in cells:
                result = train_one.remote(cell, attempt, stage, ping_only)
                payload = bytes(result.pop("artifact_tar_gz"))
                expected = result["artifact_tar_gz_sha256"]
                actual = sha256_bytes(payload)
                if actual != expected:
                    raise RuntimeError(f"Modal artifact hash mismatch for {cell}: {actual} != {expected}")
                if local_collect_dir.exists():
                    # Modal returns the whole exp073 scratch subtree.  Keep
                    # previous cells, but replace this cell's destination before
                    # extracting to avoid mixing stale and fresh files.
                    stale = local_collect_dir / "cells" / stage / attempt / cell
                    if stale.exists():
                        shutil.rmtree(stale)
                _extract_tree(payload, local_collect_dir)
                events.append({**result, "artifact_tar_gz_sha256": actual})

    elapsed = time.monotonic() - started_clock
    gpu_rate = GPU_USD_PER_SECOND.get(gpu)
    billable_gpu_s = sum(float(event.get("elapsed_s", 0.0)) for event in events)
    estimated_gpu_spend = None if gpu_rate is None else billable_gpu_s * gpu_rate
    ledger = {
        "provider": "modal",
        "backend": "experiments.helpers.modal_backend",
        "started_at": started,
        "finished_at": utc_now(),
        "elapsed_s": elapsed,
        "cells": events,
        "gpu": gpu,
        "gpu_usd_per_second": gpu_rate,
        "billable_gpu_seconds_estimate": billable_gpu_s,
        "total_spend_usd": round(float(estimated_gpu_spend or 0.0), 6),
        "exact_provider_billing": False,
        "billing_status": "timestamp_estimate_pending_provider_reconciliation",
        "active_pods_after_collection": 0,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    failed = [event for event in events if not event.get("success")]
    if failed:
        names = ", ".join(event["cell"] for event in failed)
        raise SystemExit(f"Modal exp073 job(s) failed: {names}; artifacts were collected for post-mortem")
    print(f"collected Modal artifacts into {local_collect_dir}")
    print(f"wrote Modal compute ledger {ledger_path}")
