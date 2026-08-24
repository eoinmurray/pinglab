"""Read-only inventory of legacy local Pinglab scientific data."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from runstore.contract import (
    inventory_payload,
    validate_inventory,
    verify_payload,
)
from runstore.contract import (
    load_json as load_legacy_json,
)

from .contracts import canonical_digest

COLLECTION_RE = re.compile(r'collection:\s*"([a-z0-9-]+)"')


def writing_collections(writings_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(writings_root.glob("exp*.typ")):
        match = COLLECTION_RE.search(path.read_text(errors="replace")[:2000])
        if match:
            result[path.stem] = match.group(1)
    return result


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_local(
    repo: Path,
    *,
    artifacts_root: Path | None = None,
    scratch_root: Path | None = None,
    restored_root: Path | None = None,
) -> dict[str, Any]:
    repo = repo.resolve()
    artifacts = (artifacts_root or repo / "artifacts/data").resolve()
    scratch = (scratch_root or repo / "temp/experiments").resolve()
    restored = (restored_root or repo / "runs/restored").resolve()
    memberships = writing_collections(repo / "writings")
    payloads: list[dict[str, Any]] = []

    if artifacts.is_dir():
        for directory in sorted(path for path in artifacts.iterdir() if path.is_dir()):
            manifest = directory / "_manifest.json"
            provenance = directory / "_provenance.json"
            experiment = directory.name if directory.name.startswith("exp") else None
            payloads.append(
                {
                    "physical_id": f"local-artifact:{directory}",
                    "kind": "artifact-directory",
                    "path": str(directory),
                    "experiment": experiment,
                    "collection": memberships.get(experiment or ""),
                    "manifest": str(manifest) if manifest.is_file() else None,
                    "provenance": str(provenance) if provenance.is_file() else None,
                    "file_count": sum(1 for path in directory.rglob("*") if path.is_file()),
                }
            )

    if scratch.is_dir():
        for directory in sorted(path for path in scratch.iterdir() if path.is_dir()):
            payloads.append(
                {
                    "physical_id": f"local-scratch:{directory}",
                    "kind": "scratch-directory",
                    "path": str(directory),
                    "experiment": directory.name if directory.name.startswith("exp") else None,
                    "collection": memberships.get(directory.name),
                    "file_count": sum(1 for path in directory.rglob("*") if path.is_file()),
                }
            )

    if restored.is_dir():
        for run_json in sorted(restored.glob("*/run.json")):
            root = run_json.parent
            run = json.loads(run_json.read_text())
            inventory_file = root / "inventory.json"
            derived = root / "derived/artifacts/data"
            payloads.append(
                {
                    "physical_id": f"restored:{run.get('run_id', root.name)}",
                    "kind": "restored-archive",
                    "path": str(root),
                    "collection": run.get("execution", {}).get("collection"),
                    "legacy_run_id": run.get("run_id"),
                    "archive": run.get("archive"),
                    "inventory": str(inventory_file) if inventory_file.is_file() else None,
                    "experiments": sorted(
                        path.name for path in derived.iterdir() if path.is_dir()
                    )
                    if derived.is_dir()
                    else [],
                }
            )

    return {
        "schema": "pingstore.migration-inventory/v1",
        "repo": str(repo),
        "payloads": payloads,
        "payload_count": len(payloads),
        "digest": canonical_digest(payloads),
    }


def add_remote_catalogue(
    inventory: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    payloads = list(inventory["payloads"])
    existing_archives = {
        payload.get("archive", {}).get("archive_id")
        for payload in payloads
        if isinstance(payload.get("archive"), dict)
    }
    for row in rows:
        store_key = row.get("store_key")
        if not store_key or row.get("archive_id") in existing_archives:
            continue
        payloads.append(
            {
                "physical_id": f"r2-archive:{store_key}",
                "kind": "r2-archive",
                "path": f"r2://pinglab/campaigns/{store_key}",
                "collection": row.get("collection"),
                "legacy_run_id": row.get("campaign_id"),
                "archive": {
                    "archive_id": row.get("archive_id"),
                    "store_key": store_key,
                },
                "experiments": [],
            }
        )
    inventory = dict(inventory)
    inventory["payloads"] = payloads
    inventory["payload_count"] = len(payloads)
    inventory["digest"] = canonical_digest(payloads)
    return inventory


def verify_local_inventory(inventory: dict[str, Any], *, deep: bool = False) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for payload in inventory["payloads"]:
        if payload["kind"] == "r2-archive":
            results.append(
                {
                    "physical_id": payload["physical_id"],
                    "state": "remote-deferred",
                }
            )
            continue
        path = Path(payload["path"])
        state = "present" if path.exists() else "missing"
        detail: dict[str, Any] = {
            "physical_id": payload["physical_id"],
            "state": state,
        }
        if deep and path.is_dir() and payload["kind"] != "restored-archive":
            actual = inventory_payload(path, run_id=payload["physical_id"])
            detail["payload_digest"] = actual["payload_digest"]
        if deep and payload.get("inventory"):
            inventory_path = Path(payload["inventory"])
            detail["inventory_sha256"] = _file_sha(inventory_path)
            legacy_inventory = validate_inventory(load_legacy_json(inventory_path))
            verify_payload(path, legacy_inventory)
            detail["payload_digest"] = legacy_inventory["payload_digest"]
        results.append(detail)
    return {
        "schema": "pingstore.verification/v1",
        "deep": deep,
        "results": results,
        "passed": all(
            item["state"] in {"present", "remote-deferred"} for item in results
        ),
    }
