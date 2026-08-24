"""Read-only compatibility with immutable Runstore archives."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from runstore.archive import restore_archive, verify_archive
from runstore.campaigns import catalogue
from runstore.storage import build_store


def legacy_catalogue(
    *,
    local_roots: Iterable[Path],
    store_spec: str | None = None,
    logical_uri: str = "r2://pinglab/campaigns",
) -> list[dict[str, Any]]:
    store = build_store(store_spec, logical_base_uri=logical_uri) if store_spec else None
    return catalogue(local_roots, store)


def verify_legacy_archive(
    archive_id: str,
    *,
    store_spec: str,
    logical_uri: str = "r2://pinglab/campaigns",
) -> dict[str, Any]:
    store = build_store(store_spec, logical_base_uri=logical_uri)
    return verify_archive(store, archive_id)


def restore_legacy_archive(
    archive_id: str,
    destination: Path,
    *,
    store_spec: str,
    logical_uri: str = "r2://pinglab/campaigns",
) -> dict[str, Any]:
    store = build_store(store_spec, logical_base_uri=logical_uri)
    return restore_archive(store, archive_id, destination)
