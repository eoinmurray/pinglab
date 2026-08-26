"""Managed R2 transport for immutable native Pingstore dataset archives."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .archive import archive_dataset, restore_dataset
from .catalogue import Catalogue
from .contracts import PingstoreError

DEFAULT_DATASET_STORE = "r2:pinglab/datasets"
DEFAULT_DATASET_URI = "r2://pinglab/datasets"


def _run(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["rclone", *args], stderr=subprocess.STDOUT, text=True
        )
    except FileNotFoundError as exc:
        raise PingstoreError("rclone is required for native R2 operations") from exc
    except subprocess.CalledProcessError as exc:
        raise PingstoreError(exc.output.strip() or "rclone operation failed") from exc


def _key(dataset_id: str) -> str:
    parts = dataset_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise PingstoreError("native dataset ID must be collection/snapshot")
    return dataset_id


def archive_dataset_r2(
    catalogue: Catalogue,
    dataset_id: str,
    *,
    store: str = DEFAULT_DATASET_STORE,
    logical_uri: str = DEFAULT_DATASET_URI,
) -> dict[str, Any]:
    """Build, upload, and checksum-verify one frozen native dataset."""
    key = _key(dataset_id)
    remote = f"{store.rstrip('/')}/{key}"
    if _run("lsf", remote).strip():
        raise PingstoreError(f"native R2 dataset already exists: {logical_uri}/{key}")
    with tempfile.TemporaryDirectory(prefix="pingstore-r2-archive-") as temporary:
        bundle = Path(temporary) / "bundle"
        manifest = archive_dataset(catalogue, dataset_id, bundle)
        _run("copy", str(bundle), remote, "--immutable")
        _run("check", str(bundle), remote, "--one-way")
    return {
        **manifest,
        "archive": {"uri": f"{logical_uri.rstrip('/')}/{key}", "store_key": key},
        "verified": True,
    }


def restore_dataset_r2(
    key: str,
    destination_root: Path,
    *,
    store: str = DEFAULT_DATASET_STORE,
) -> dict[str, Any]:
    """Download and fully verify a native dataset into a clean Pingstore root."""
    remote = f"{store.rstrip('/')}/{_key(key)}"
    with tempfile.TemporaryDirectory(prefix="pingstore-r2-restore-") as temporary:
        bundle = Path(temporary) / "bundle"
        _run("copy", remote, str(bundle))
        return restore_dataset(bundle, destination_root)


def inspect_dataset_r2(
    key: str, *, store: str = DEFAULT_DATASET_STORE
) -> dict[str, Any]:
    """Read the small remote archive identity without downloading payloads."""
    remote = f"{store.rstrip('/')}/{_key(key)}/archive.json"
    try:
        value = json.loads(_run("cat", remote))
    except json.JSONDecodeError as exc:
        raise PingstoreError(
            f"invalid remote native archive manifest: {remote}"
        ) from exc
    value["remote"] = remote
    return value
