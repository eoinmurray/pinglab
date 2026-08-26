"""Read-only inventory of legacy local Pinglab scientific data."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .contracts import canonical_digest
from .payload import (
    inventory_payload,
    validate_inventory,
    verify_payload,
)
from .payload import (
    load_json as load_legacy_json,
)
from .registry import load_registry, memberships, registry_path

COLLECTION_RE = re.compile(r'collection:\s*"([a-z0-9-]+)"')


def writing_collections(writings_root: Path) -> dict[str, str]:
    registry = writings_root.parent / "experiments/collections/registry.json"
    if registry.is_file():
        return memberships(writings_root.parent)
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
    artifacts = (artifacts_root or repo / ".artifacts").resolve()
    scratch = (scratch_root or repo / "temp/experiments").resolve()
    restored = (restored_root or repo / "runs/restored").resolve()
    registry = (
        load_registry(repo) if registry_path(repo).is_file() else {"historical": {}}
    )
    memberships = (
        dict(registry["experiments"])
        if "experiments" in registry
        else writing_collections(repo / "writings")
    )
    historical = registry.get("historical", {})
    payloads: list[dict[str, Any]] = []

    if artifacts.is_dir():
        for directory in sorted(path for path in artifacts.iterdir() if path.is_dir()):
            manifest = directory / "_manifest.json"
            provenance = directory / "_provenance.json"
            experiment = directory.name if directory.name.startswith("exp") else None
            historical_row = historical.get(experiment or "")
            payloads.append(
                {
                    "physical_id": f"local-artifact:{directory}",
                    "kind": "artifact-directory",
                    "path": str(directory),
                    "experiment": experiment,
                    "collection": memberships.get(experiment or "")
                    or (
                        historical_row.get("collection")
                        if isinstance(historical_row, dict)
                        else None
                    ),
                    "historical": historical_row,
                    "manifest": str(manifest) if manifest.is_file() else None,
                    "provenance": str(provenance) if provenance.is_file() else None,
                    "file_count": sum(
                        1 for path in directory.rglob("*") if path.is_file()
                    ),
                }
            )

    if scratch.is_dir():
        for directory in sorted(path for path in scratch.iterdir() if path.is_dir()):
            payloads.append(
                {
                    "physical_id": f"local-scratch:{directory}",
                    "kind": "scratch-directory",
                    "path": str(directory),
                    "experiment": directory.name
                    if directory.name.startswith("exp")
                    else None,
                    "collection": memberships.get(directory.name),
                    "file_count": sum(
                        1 for path in directory.rglob("*") if path.is_file()
                    ),
                }
            )

    if restored.is_dir():
        for run_json in sorted(restored.glob("*/run.json")):
            root = run_json.parent
            run = json.loads(run_json.read_text())
            inventory_file = root / "inventory.json"
            derived = root / "derived/.artifacts"
            payloads.append(
                {
                    "physical_id": f"restored:{run.get('run_id', root.name)}",
                    "kind": "restored-archive",
                    "path": str(root),
                    "collection": run.get("execution", {}).get("collection"),
                    "legacy_run_id": run.get("run_id"),
                    "archive": run.get("archive"),
                    "inventory": str(inventory_file)
                    if inventory_file.is_file()
                    else None,
                    "experiments": sorted(
                        path.name for path in derived.iterdir() if path.is_dir()
                    )
                    if derived.is_dir()
                    else [],
                }
            )

    registry_state = {
        "memberships": memberships,
        "historical_dispositions": historical,
    }
    return {
        "schema": "pingstore.migration-inventory/v1",
        "repo": str(repo),
        "payloads": payloads,
        "payload_count": len(payloads),
        "memberships": memberships,
        "historical_dispositions": historical,
        "registry_digest": canonical_digest(registry_state),
        "digest": canonical_digest(payloads),
    }


def verify_local_inventory(
    inventory: dict[str, Any], *, deep: bool = False
) -> dict[str, Any]:
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
