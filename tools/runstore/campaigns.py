"""Small campaign catalogue and atomic local publication-view activation."""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .contract import (
    CONTRACT_VERSION,
    ContractError,
    load_json,
    validate_inventory,
    validate_run_manifest,
    verify_payload,
    write_json_atomic,
)
from .promotion import FIGURE_SUFFIXES, _generating_git_commit, _sha256, _source_rows
from .storage import Store

VIEW_NAME = ".runstore-view.json"
VIEW_SCHEMA = "runstore/view-v1"


def _profile(root: Path) -> str | None:
    plan = root / "collection-plan.json"
    if not plan.is_file():
        return None
    value = load_json(plan).get("profile")
    return value if isinstance(value, str) else None


def _campaign_row(
    run: dict[str, Any],
    *,
    location: str,
    identity: str,
    root: str | None = None,
    profile: str | None = None,
    store_key: str | None = None,
) -> dict[str, Any]:
    return {
        "campaign_id": run["run_id"],
        "archive_id": (run.get("archive") or {}).get("archive_id") or identity,
        "collection": run["execution"].get("collection"),
        "kind": run["kind"],
        "status": run["status"],
        "created_at_utc": run["created_at_utc"],
        "git_commit": run["source"].get("git_commit"),
        "profile": profile,
        "locations": [location],
        "local_root": root,
        "store_key": store_key,
    }


def discover_local_campaigns(roots: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    seen: set[Path] = set()
    for search_root in roots:
        search_root = search_root.expanduser().resolve()
        if not search_root.is_dir():
            continue
        candidates = [search_root / "run.json", *search_root.glob("*/run.json")]
        for manifest in candidates:
            if not manifest.is_file() or manifest.parent in seen:
                continue
            seen.add(manifest.parent)
            try:
                run = validate_run_manifest(load_json(manifest))
            except ContractError:
                continue
            if run["kind"] not in {"campaign", "legacy"}:
                continue
            rows.append(
                _campaign_row(
                    run,
                    location="local",
                    identity=run["run_id"],
                    root=str(manifest.parent),
                    profile=_profile(manifest.parent),
                )
            )
    return rows


def discover_remote_campaigns(store: Store) -> list[dict[str, Any]]:
    rows = []
    for archive_id in store.archive_ids():
        try:
            run = validate_run_manifest(
                json.loads(store.read_bytes(archive_id, "run.json"))
            )
        except (ContractError, json.JSONDecodeError):
            continue
        if run["kind"] not in {"campaign", "legacy"}:
            continue
        profile = None
        try:
            plan = json.loads(store.read_bytes(archive_id, "collection-plan.json"))
            if isinstance(plan, dict) and isinstance(plan.get("profile"), str):
                profile = plan["profile"]
        except (ContractError, json.JSONDecodeError):
            pass
        rows.append(
            _campaign_row(
                run,
                location="r2",
                identity=archive_id,
                profile=profile,
                store_key=archive_id,
            )
        )
    return rows


def catalogue(
    local_roots: Iterable[Path],
    store: Store | None,
    *,
    active_campaign_id: str | None = None,
) -> list[dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    for row in [
        *discover_local_campaigns(local_roots),
        *(discover_remote_campaigns(store) if store else []),
    ]:
        key = row["campaign_id"]
        if key not in combined:
            combined[key] = row
            continue
        existing = combined[key]
        existing["locations"] = sorted(set(existing["locations"] + row["locations"]))
        for field in ("archive_id", "local_root", "profile", "store_key"):
            if existing.get(field) is None and row.get(field) is not None:
                existing[field] = row[field]
    for row in combined.values():
        row["active"] = row["campaign_id"] == active_campaign_id
    return sorted(
        combined.values(), key=lambda row: (row["created_at_utc"], row["campaign_id"])
    )


def resolve_local_campaign(value: str, roots: Iterable[Path]) -> Path:
    direct = Path(value).expanduser()
    if direct.is_dir():
        return direct.resolve()
    matches = [
        Path(row["local_root"])
        for row in discover_local_campaigns(roots)
        if value in {row["campaign_id"], row["archive_id"]}
    ]
    if not matches:
        raise ContractError(f"no local campaign matches {value!r}")
    if len(matches) > 1:
        raise ContractError(f"multiple local campaigns match {value!r}; pass a path")
    return matches[0]


def _stage_experiment(
    run_root: Path,
    experiment: str,
    staging_root: Path,
    run: dict[str, Any],
    inventory: dict[str, Any],
    timestamp: str,
) -> int:
    source_relative = Path("derived/artifacts/data") / experiment
    source = run_root / source_relative
    if not (source / "numbers.json").is_file():
        raise ContractError(f"{experiment} promotion source requires numbers.json")
    if not any(
        item.is_file() and item.suffix.lower() in FIGURE_SUFFIXES
        for item in source.rglob("*")
    ):
        raise ContractError(f"{experiment} promotion source requires a figure")
    rows = _source_rows(run_root, source, inventory)
    destination = staging_root / experiment
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    for row in rows:
        target = destination / row["path"]
        if (
            target.stat().st_size != row["size_bytes"]
            or _sha256(target) != row["sha256"]
        ):
            raise ContractError(
                f"staged file differs from source: {experiment}/{row['path']}"
            )
    write_json_atomic(
        destination / "_provenance.json",
        {
            "contract_version": CONTRACT_VERSION,
            "run_id": run["run_id"],
            "campaign_id": run["run_id"],
            "generating_git_commit": _generating_git_commit(run, source),
            "campaign_source_git_commit": run["source"]["git_commit"],
            "source_directory": source_relative.as_posix(),
            "source_inventory_payload_digest": inventory["payload_digest"],
            "archive": run["archive"],
            "promoted_at_utc": timestamp,
            "files": rows,
        },
    )
    return len(rows)


def _planned_experiments(run_root: Path) -> list[str]:
    plan = load_json(run_root / "collection-plan.json")
    stages = plan.get("stages")
    if not isinstance(stages, list):
        raise ContractError("collection plan does not contain experiment stages")
    experiments: list[str] = []
    for stage in stages:
        if not isinstance(stage, dict) or not isinstance(
            stage.get("experiments"), list
        ):
            raise ContractError("collection plan has malformed experiment stages")
        for row in stage["experiments"]:
            slug = row.get("slug") if isinstance(row, dict) else None
            if not isinstance(slug, str) or not slug:
                raise ContractError("collection plan has an invalid experiment slug")
            experiments.append(slug)
    if not experiments or len(experiments) != len(set(experiments)):
        raise ContractError("collection plan must name unique experiments")
    return sorted(experiments)


def activate_campaign(
    run_root: Path, *, artifacts_root: Path, activated_at_utc: str | None = None
) -> dict[str, Any]:
    run_root = run_root.resolve()
    run = validate_run_manifest(load_json(run_root / "run.json"))
    inventory = validate_inventory(load_json(run_root / "inventory.json"))
    if run["kind"] not in {"campaign", "legacy"} or not run["execution"].get(
        "collection"
    ):
        raise ContractError("activation requires a collection campaign")
    if run["status"] not in {"complete", "legacy"}:
        raise ContractError("activation requires a complete or legacy campaign")
    if inventory["run_id"] != run["run_id"]:
        raise ContractError("run.json and inventory.json use different run IDs")
    verify_payload(run_root, inventory)
    source_root = run_root / "derived/artifacts/data"
    experiments = sorted(path.name for path in source_root.iterdir() if path.is_dir())
    planned = _planned_experiments(run_root)
    if experiments != planned:
        missing = sorted(set(planned) - set(experiments))
        unexpected = sorted(set(experiments) - set(planned))
        raise ContractError(
            "campaign derived experiments do not match its collection plan: "
            f"missing={missing}, unexpected={unexpected}"
        )

    artifacts_root = artifacts_root.resolve()
    artifacts_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            dir=artifacts_root.parent,
            prefix=f".{artifacts_root.name}.campaign-staging-",
        )
    )
    backup = (
        artifacts_root.parent / f".{artifacts_root.name}.campaign-backup-{uuid4().hex}"
    )
    timestamp = activated_at_utc or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    moved_existing = False
    try:
        if artifacts_root.is_dir():
            shutil.copytree(artifacts_root, staging, dirs_exist_ok=True)
        file_count = sum(
            _stage_experiment(run_root, experiment, staging, run, inventory, timestamp)
            for experiment in experiments
        )
        view = {
            "schema": VIEW_SCHEMA,
            "campaign_id": run["run_id"],
            "collection": run["execution"]["collection"],
            "generating_git_commit": run["source"]["git_commit"],
            "source_inventory_payload_digest": inventory["payload_digest"],
            "activated_at_utc": timestamp,
            "experiments": experiments,
        }
        write_json_atomic(staging / VIEW_NAME, view)
        if artifacts_root.exists():
            artifacts_root.rename(backup)
            moved_existing = True
        staging.rename(artifacts_root)
        if moved_existing:
            shutil.rmtree(backup)
        return {**view, "artifacts_root": str(artifacts_root), "file_count": file_count}
    except Exception:
        if moved_existing and not artifacts_root.exists() and backup.exists():
            backup.rename(artifacts_root)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def current_view(artifacts_root: Path, *, verify_files: bool = True) -> dict[str, Any]:
    artifacts_root = artifacts_root.resolve()
    view = load_json(artifacts_root / VIEW_NAME)
    if view.get("schema") != VIEW_SCHEMA:
        raise ContractError(f"{VIEW_NAME} has an unsupported schema")
    errors = []
    for experiment in view.get("experiments", []):
        root = artifacts_root / experiment
        try:
            provenance = load_json(root / "_provenance.json")
        except ContractError as exc:
            errors.append(str(exc))
            continue
        for field in ("campaign_id", "source_inventory_payload_digest"):
            expected = view[field]
            if provenance.get(field) != expected:
                errors.append(f"{experiment} {field} does not match active view")
        campaign_source = provenance.get(
            "campaign_source_git_commit", provenance.get("generating_git_commit")
        )
        if campaign_source != view["generating_git_commit"]:
            errors.append(
                f"{experiment} campaign source commit does not match active view"
            )
        if verify_files:
            for row in provenance.get("files", []):
                target = root / row["path"]
                if (
                    not target.is_file()
                    or target.stat().st_size != row["size_bytes"]
                    or _sha256(target) != row["sha256"]
                ):
                    errors.append(f"{experiment}/{row['path']} differs from provenance")
    return {
        **view,
        "artifacts_root": str(artifacts_root),
        "valid": not errors,
        "errors": errors,
    }
