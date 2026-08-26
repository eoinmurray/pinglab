"""Finalize successful direct local runners as immutable ExperimentRuns."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .catalogue import Catalogue
from .contracts import EXPERIMENT_RUN_SCHEMA, PingstoreError, write_json_atomic
from .inventory import writing_collections
from .payload import inventory_payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _snapshot_authored_sources(repo: Path, slug: str, root: Path) -> dict:
    """Retain the exact mutable writing and implementation used by a run."""
    source_root = root / "authored-sources"
    source_root.mkdir(parents=True)
    result: dict[str, dict[str, str]] = {}
    for label, source in (
        ("writing", repo / "writings" / f"{slug}.typ"),
        ("implementation", repo / "experiments" / f"{slug}.py"),
    ):
        if not source.is_file():
            continue
        target = source_root / source.name
        shutil.copy2(source, target)
        result[label] = {
            "path": target.relative_to(root).as_posix(),
            "sha256": _sha256(target),
        }
    return result


def capture_local_run(
    repo: Path,
    slug: str,
    staging: Path,
    *,
    root: Path | None = None,
) -> dict:
    """Copy one successful staged result into immutable native storage.

    This runs before the historical artifact-view swap. Failure therefore leaves
    the previously visible result untouched.
    """
    repo = repo.resolve()
    collection = writing_collections(repo / "writings").get(slug)
    if collection is None:
        raise PingstoreError(f"cannot finalize {slug}: collection membership missing")
    manifest_file = staging / "_manifest.json"
    if not manifest_file.is_file():
        raise PingstoreError(f"cannot finalize {slug}: _manifest.json missing")
    manifest = json.loads(manifest_file.read_text())
    local_id = manifest.get("run_id")
    if not isinstance(local_id, str) or not local_id:
        raise PingstoreError(f"cannot finalize {slug}: run ID missing")
    run_id = f"{slug}/{local_id}"
    catalogue = Catalogue(
        root or Path(os.environ.get("PINGSTORE_ROOT", repo / ".pingstore"))
    )
    dataset_file = catalogue.dataset_path(collection)
    if not dataset_file.exists():
        members = sorted(
            experiment
            for experiment, member_collection in writing_collections(
                repo / "writings"
            ).items()
            if member_collection == collection
        )
        catalogue.create_dataset(collection, members)
    else:
        catalogue.register_experiment(collection, slug)

    run_root = catalogue.run_path(collection, slug, run_id)
    if run_root.exists():
        raise PingstoreError(f"immutable run already exists: {run_id}")
    temporary = run_root.with_name(run_root.name + ".staging")
    if temporary.exists():
        shutil.rmtree(temporary)
    payload = temporary / "derived/.artifacts" / slug
    payload.parent.mkdir(parents=True)
    shutil.copytree(staging, payload)
    inventory = inventory_payload(payload, run_id=run_id)
    authored = _snapshot_authored_sources(repo, slug, temporary)
    completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run = {
        "schema": EXPERIMENT_RUN_SCHEMA,
        "run_id": run_id,
        "collection": collection,
        "experiment": slug,
        "status": "finalized",
        "disposition": "candidate",
        "source": {
            "git_commit": manifest.get("git_sha"),
            "dirty": manifest.get("dirty"),
            "code_dirty": manifest.get("code_dirty"),
            "dirty_patch": manifest.get("patch"),
            "authored": authored,
        },
        "execution": {
            "host": manifest.get("host", "local"),
            "command": ["uv", "run", "python", f"experiments/{slug}.py"],
            "started_at": manifest.get("run_at"),
            "completed_at": completed_at,
            "configuration": manifest.get("scale"),
        },
        "upstream_runs": [],
        "upstream_datasets": [],
        "payload": {
            "location": str(run_root / "derived/.artifacts" / slug),
            "inventory_digest": "sha256:" + inventory["payload_digest"],
        },
        "archive": None,
        "legacy_identity": None,
    }
    write_json_atomic(temporary / "run.json", run)
    write_json_atomic(temporary / "inventory.json", inventory)
    run_root.parent.mkdir(parents=True, exist_ok=True)
    os.rename(temporary, run_root)
    catalogue.register_run(run)
    return run


def capture_failed_local_run(
    repo: Path,
    slug: str,
    staging: Path,
    *,
    root: Path | None = None,
) -> dict:
    """Retain an incomplete local execution without advancing official evidence."""
    repo = repo.resolve()
    collection = writing_collections(repo / "writings").get(slug)
    if collection is None:
        raise PingstoreError(
            f"cannot record failed {slug}: collection membership missing"
        )
    manifest_file = staging / "_manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.is_file() else {}
    local_id = str(manifest.get("run_id") or "unknown")
    failed_at = datetime.now(timezone.utc)
    suffix = failed_at.strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{slug}/{local_id}-failed-{suffix}"
    catalogue = Catalogue(
        root or Path(os.environ.get("PINGSTORE_ROOT", repo / ".pingstore"))
    )
    dataset_file = catalogue.dataset_path(collection)
    if not dataset_file.exists():
        members = sorted(
            experiment
            for experiment, member_collection in writing_collections(
                repo / "writings"
            ).items()
            if member_collection == collection
        )
        catalogue.create_dataset(collection, members)
    else:
        catalogue.register_experiment(collection, slug)
    run_root = catalogue.run_path(collection, slug, run_id)
    if run_root.exists():
        raise PingstoreError(f"immutable failed run already exists: {run_id}")
    payload = run_root / "incomplete-payload"
    payload.parent.mkdir(parents=True)
    shutil.copytree(staging, payload)
    authored = _snapshot_authored_sources(repo, slug, run_root)
    run = {
        "schema": EXPERIMENT_RUN_SCHEMA,
        "run_id": run_id,
        "collection": collection,
        "experiment": slug,
        "status": "failed",
        "disposition": "temporary",
        "source": {
            "git_commit": manifest.get("git_sha"),
            "dirty": manifest.get("dirty"),
            "code_dirty": manifest.get("code_dirty"),
            "dirty_patch": manifest.get("patch"),
            "authored": authored,
        },
        "execution": {
            "host": manifest.get("host", "local"),
            "command": ["uv", "run", "python", f"experiments/{slug}.py"],
            "started_at": manifest.get("run_at"),
            "completed_at": failed_at.isoformat(timespec="seconds"),
            "configuration": manifest.get("scale"),
        },
        "upstream_runs": [],
        "upstream_datasets": [],
        "payload": {"location": str(payload), "inventory_digest": None},
        "archive": None,
        "legacy_identity": None,
    }
    write_json_atomic(run_root / "run.json", run)
    catalogue.register_run(run)
    return run


def capture_campaign_metadata(root: Path, plan: dict) -> dict:
    """Write candidate ExperimentRun records before campaign inventory freezes."""
    collection = plan["collection"]
    campaign_id = plan["campaign_id"]
    records_root = root / "pingstore"
    rows = [
        experiment for stage in plan["stages"] for experiment in stage["experiments"]
    ]
    experiments = sorted(row["slug"] for row in rows)
    runs: dict[str, list[str]] = {}
    proposal: dict[str, str] = {}
    for row in rows:
        slug = row["slug"]
        run_id = f"{campaign_id}/{slug}"
        source = Path(row["paths"]["derived"])
        payload_inventory = inventory_payload(source, run_id=run_id)
        run = {
            "schema": EXPERIMENT_RUN_SCHEMA,
            "run_id": run_id,
            "collection": collection,
            "experiment": slug,
            "status": "finalized",
            "disposition": "candidate",
            "source": plan.get("source", {}),
            "execution": {
                "host": "campaign",
                "command": row["command"],
                "campaign_id": campaign_id,
            },
            "upstream_runs": [
                f"{campaign_id}/{dependency}" for dependency in row["dependencies"]
            ],
            "upstream_datasets": [],
            "payload": {
                "location": str(source),
                "inventory_digest": "sha256:" + payload_inventory["payload_digest"],
            },
            "archive": None,
            "legacy_identity": {"campaign_id": campaign_id},
        }
        write_json_atomic(records_root / "experiment-runs" / slug / "run.json", run)
        runs[slug] = [run_id]
        proposal[slug] = run_id
    dataset = {
        "schema": "pingstore.collection-dataset/v1",
        "dataset_id": f"{collection}/{campaign_id}-candidate",
        "collection": collection,
        "status": "working",
        "experiments": experiments,
        "runs": runs,
        "official_runs": dict(proposal),
        "preview_overrides": {},
        "collection_assets": [],
        "upstream_datasets": [],
        "migration": {"campaign_id": campaign_id},
        "selection_proposal": proposal,
        "digest": None,
    }
    write_json_atomic(records_root / "collection-dataset.json", dataset)
    return dataset
