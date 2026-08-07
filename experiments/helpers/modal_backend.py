"""Generic Modal backend for experiment jobs.

This mirrors the runner-owned contract in helpers/runpod.py: the backend owns
cloud execution and artifact transport, while each runner owns job ids, its
completion predicate, its one-job action, and all scientific parameters.

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
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import REPO

REMOTE_REPO = Path("/workspace/pinglab")
REMOTE_ARTIFACTS_ROOT = Path("/tmp/pinglab-artifacts")
MAX_RUNTIME_S = 54000

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


def _is_modal_auth_error(exc: BaseException) -> bool:
    cls = exc.__class__
    return cls.__name__ == "AuthError" and cls.__module__.startswith("modal")


def _modal_auth_help() -> str:
    return (
        "Modal authentication is missing. Run `uv run modal setup` on this host, "
        "or provide approved Modal token environment variables, then re-run with "
        "`--modal --live`."
    )


def _source_image(modal: Any):
    image = (
        modal.Image.debian_slim(python_version="3.10")
        .uv_pip_install(
            "torch",
            "numpy",
            "scipy",
            "scikit-learn",
            "h5py",
            "matplotlib",
            "snntorch",
        )
        .env(
            {
                "PYTHONPATH": (
                    f"{REMOTE_REPO / 'experiments'}:"
                    f"{REMOTE_REPO / 'tools' / 'snn'}"
                )
            }
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


def _load_runner_hooks(
    *,
    runner: str,
    is_done_name: str,
    run_job_name: str,
) -> tuple[Any, Any]:
    """Import a runner and resolve its declared generic job hooks."""
    import importlib
    import sys

    if not runner.isidentifier() or not runner.startswith("exp"):
        raise ValueError(f"invalid experiment runner: {runner!r}")
    # Modal may reuse a warm container. Runner recipes commonly resolve their
    # registered stage from the environment at import time, so never retain a
    # previous job's module globals across calls.
    sys.modules.pop(runner, None)
    importlib.invalidate_caches()
    module = importlib.import_module(runner)
    is_done = getattr(module, is_done_name)
    run_job = getattr(module, run_job_name)
    if not callable(is_done) or not callable(run_job):
        raise TypeError(f"runner hooks must be callable: {is_done_name}, {run_job_name}")
    return is_done, run_job


def _remote_run_job(
    *,
    slug: str,
    runner: str,
    job_id: str,
    env: dict[str, str],
    is_done_name: str,
    run_job_name: str,
) -> dict[str, Any]:
    """Execute one runner-owned job inside Modal and return its artifact tree."""
    import sys
    import traceback

    os.chdir(REMOTE_REPO)
    sys.path.insert(0, str(REMOTE_REPO / "experiments"))
    sys.path.insert(0, str(REMOTE_REPO / "tools" / "snn"))
    artifacts_root = REMOTE_ARTIFACTS_ROOT / slug
    os.environ.update(
        {
            **env,
            "PINGLAB_ARTIFACTS_ROOT": str(artifacts_root),
            "PYTHONUNBUFFERED": "1",
        }
    )
    started_wall = time.monotonic()
    started_at = utc_now()
    error = None
    skipped = False
    try:
        is_done, run_job = _load_runner_hooks(
            runner=runner,
            is_done_name=is_done_name,
            run_job_name=run_job_name,
        )
        if is_done(job_id):
            skipped = True
        else:
            run_job(job_id)
            if not is_done(job_id):
                raise RuntimeError(
                    f"runner {runner} job {job_id!r} returned without satisfying "
                    f"{is_done_name}"
                )
    except BaseException:  # noqa: BLE001 — serialize failure into the ledger
        error = traceback.format_exc()
    artifact_payload = _tar_tree(artifacts_root)
    elapsed_s = time.monotonic() - started_wall
    success = error is None
    return {
        "runner": runner,
        "job_id": job_id,
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_s": elapsed_s,
        "success": success,
        "skipped": skipped,
        "error": error,
        "artifact_tar_gz": artifact_payload,
        "artifact_tar_gz_sha256": sha256_bytes(artifact_payload),
    }


def dispatch(
    *,
    slug: str,
    runner: str,
    job_ids: list[str],
    live: bool,
    local_collect_dir: Path,
    ledger_path: Path,
    timeout_s: int,
    extra_env: dict[str, str] | None = None,
    is_done_name: str = "cell_done",
    run_job_name: str = "run_full_cell",
) -> None:
    """Fan out arbitrary runner-owned jobs on Modal and collect artifacts."""
    if not slug or Path(slug).name != slug:
        raise ValueError(f"invalid experiment slug: {slug!r}")
    if not runner.isidentifier() or not runner.startswith("exp"):
        raise ValueError(f"invalid experiment runner: {runner!r}")
    if not job_ids or len(set(job_ids)) != len(job_ids):
        raise ValueError("job_ids must be non-empty and unique")
    if timeout_s <= 0 or timeout_s > MAX_RUNTIME_S:
        raise ValueError(f"timeout_s must be in 1..{MAX_RUNTIME_S}")
    gpu = os.environ.get("PINGLAB_MODAL_GPU", "L40S")
    print(f"{'LIVE' if live else 'DRY-RUN'}  runner={runner}  backend=modal  gpu={gpu}")
    print(f"jobs: {' '.join(job_ids)}")
    print("set PINGLAB_MODAL_GPU to choose a different Modal GPU SKU")
    if not live:
        print("\n(dry-run — nothing created. Re-run with --live to spend.)")
        return

    modal = _require_modal()
    from . import modal_app

    output_context = getattr(modal, "enable_output", lambda: contextlib.nullcontext())()
    events: list[dict[str, Any]] = []
    started = utc_now()
    started_clock = time.monotonic()
    try:
        with output_context:
            with modal_app.app.run():
                remote = modal_app.run_job.with_options(timeout=timeout_s)
                calls = [
                    (
                        job_id,
                        remote.spawn(
                            slug,
                            runner,
                            job_id,
                            dict(extra_env or {}),
                            is_done_name,
                            run_job_name,
                        ),
                    )
                    for job_id in job_ids
                ]
                for job_id, call in calls:
                    result = call.get()
                    payload = bytes(result.pop("artifact_tar_gz"))
                    expected = result["artifact_tar_gz_sha256"]
                    actual = sha256_bytes(payload)
                    if actual != expected:
                        raise RuntimeError(f"Modal artifact hash mismatch for {job_id}: {actual} != {expected}")
                    _extract_tree(payload, local_collect_dir)
                    events.append({**result, "artifact_tar_gz_sha256": actual})
    except BaseException as exc:
        if _is_modal_auth_error(exc):
            raise SystemExit(_modal_auth_help()) from exc
        raise

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
        "slug": slug,
        "runner": runner,
        "jobs": events,
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
        names = ", ".join(event["job_id"] for event in failed)
        raise SystemExit(f"Modal {runner} job(s) failed: {names}; artifacts were collected for post-mortem")
    print(f"collected Modal artifacts into {local_collect_dir}")
    print(f"wrote Modal compute ledger {ledger_path}")
