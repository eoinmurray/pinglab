"""Resumable local execution for one isolated collection campaign."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pingstore.campaign_promotion import promote_experiment
from pingstore.campaign_runtime import initialize_run
from pingstore.native import capture_campaign_metadata
from pingstore.payload import (
    ContractError,
    inventory_payload,
    validate_inventory,
    validate_run_manifest,
    verify_payload,
)

from .graph import COLLECTION
from .plan import REPO, build_plan, validate_campaign_root

PLAN_NAME = "collection-plan.json"
STATUS_DIR = "collection-status"


class CollectionError(ValueError):
    """Raised when a campaign violates the collection execution contract."""


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise CollectionError(f"invalid or missing campaign file: {path}") from exc
    if not isinstance(value, dict):
        raise CollectionError(f"campaign file must contain an object: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _directory_snapshot(root: Path) -> dict[str, Any]:
    rows = []
    for item in sorted(
        root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()
    ):
        if item.is_symlink():
            raise CollectionError(f"composition source contains a symlink: {item}")
        if item.is_file():
            rows.append(
                {
                    "path": item.relative_to(root).as_posix(),
                    "size_bytes": item.stat().st_size,
                    "sha256": _sha256(item),
                }
            )
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"file_count": len(rows), "payload_digest": digest}


def _replace_plan_root(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_plan_root(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_plan_root(item, old, new) for key, item in value.items()}
    return value


def _inspect_campaign(root: Path) -> dict[str, Any]:
    try:
        run = validate_run_manifest(load_json(root / "run.json"))
        inventory = validate_inventory(load_json(root / "inventory.json"))
        verify_payload(root, inventory)
    except (ContractError, OSError) as exc:
        raise CollectionError(f"invalid campaign evidence: {exc}") from exc
    return {"run": run, "inventory": "valid"}


def compose_campaign(
    root: Path,
    campaign_id: str,
    *,
    base_root: Path,
    overlay_root: Path,
    replacements: list[str],
) -> dict[str, Any]:
    """Compose a complete campaign from a frozen base and selected repair outputs."""
    root = validate_campaign_root(root)
    base_root = base_root.resolve()
    overlay_root = overlay_root.resolve()
    _require_clean_source()

    base_run = load_json(base_root / "run.json")
    base_inventory = load_json(base_root / "inventory.json")
    if base_run["status"] != "complete":
        raise CollectionError("composition base campaign must be complete")
    inspected_base = _inspect_campaign(base_root)
    if inspected_base.get("inventory") != "valid":
        raise CollectionError("composition base inventory is not valid")

    overlay_run = load_json(overlay_root / "run.json")
    base_plan = load_json(base_root / PLAN_NAME)
    rows = [row for stage in base_plan["stages"] for row in stage["experiments"]]
    planned = {row["slug"] for row in rows}
    selected = set(replacements)
    if not selected or selected - planned:
        raise CollectionError(
            f"invalid composition replacements: {sorted(selected - planned)}"
        )

    base_derived = base_root / "derived/.artifacts"
    if {path.name for path in base_derived.iterdir() if path.is_dir()} != planned:
        raise CollectionError(
            "composition base does not contain every planned experiment"
        )

    source_rows: dict[str, dict[str, Any]] = {}
    for slug in sorted(planned):
        source_run = overlay_run if slug in selected else base_run
        source_root = overlay_root if slug in selected else base_root
        source = source_root / "derived/.artifacts" / slug
        if not (source / "numbers.json").is_file():
            raise CollectionError(f"composition source is missing {slug}/numbers.json")
        numbers = load_json(source / "numbers.json")
        provenance = numbers.get("collection_provenance")
        if (
            not isinstance(provenance, dict)
            or provenance.get("campaign_id") != source_run["run_id"]
        ):
            raise CollectionError(
                f"{slug} provenance does not match its source campaign"
            )
        snapshot = _directory_snapshot(source)
        source_rows[slug] = {
            "run_id": source_run["run_id"],
            "source_git_commit": provenance.get("source_git_commit")
            or source_run["source"]["git_commit"],
            "source_directory": str(source),
            **snapshot,
        }

    command = [
        sys.executable,
        "-m",
        "experiments.collections.gamma_gated_sparsity",
        "compose",
        "--campaign-root",
        str(root),
        "--campaign-id",
        campaign_id,
        "--base-root",
        str(base_root),
        "--overlay-root",
        str(overlay_root),
        *[part for slug in sorted(selected) for part in ("--replace", slug)],
    ]
    try:
        initialize_run(
            root,
            run_id=campaign_id,
            kind="campaign",
            experiment=None,
            collection=COLLECTION,
            upstream=[base_run["run_id"], overlay_run["run_id"]],
            provenance_notes="composite publication campaign",
            command=command,
            repository=REPO,
        )
        run = load_json(root / "run.json")
        destination = root / "derived/.artifacts"
        destination.mkdir(parents=True)
        for slug in sorted(planned):
            source = Path(source_rows[slug]["source_directory"])
            shutil.copytree(source, destination / slug)

        plan = _replace_plan_root(base_plan, str(base_root), str(root))
        plan["campaign_id"] = campaign_id
        plan["campaign_root"] = str(root)
        plan["profile"] = "production-composite"
        plan["source"] = run["source"]
        plan["composition"] = {
            "schema": "pinglab.campaign-composition/v1",
            "base_run_id": base_run["run_id"],
            "overlay_run_id": overlay_run["run_id"],
            "replacements": sorted(selected),
        }
        write_json_atomic(root / PLAN_NAME, plan)
        composition = {
            "schema": "pinglab.campaign-composition/v1",
            "run_id": campaign_id,
            "base": {
                "run_id": base_run["run_id"],
                "inventory_payload_digest": base_inventory["payload_digest"],
            },
            "overlay": {"run_id": overlay_run["run_id"]},
            "experiments": source_rows,
        }
        write_json_atomic(root / "composition.json", composition)
        return {
            "campaign_root": str(root),
            "campaign_id": campaign_id,
            "experiments": len(planned),
            "replacements": sorted(selected),
            "payload": _inspect_campaign(root),
        }
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    clean = not subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lock = REPO / "uv.lock"
    return {
        "git_commit": commit,
        "git_clean": clean,
        "lockfile": {"path": "uv.lock", "sha256": _sha256(lock)}
        if lock.is_file()
        else None,
    }


def utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _require_clean_source() -> None:
    source = source_provenance()
    if source["git_commit"] is None:
        raise CollectionError("collection execution requires a Git checkout")
    if source["git_clean"] is not True:
        raise CollectionError("collection execution requires a clean Git checkout")
    if source["lockfile"] is None:
        raise CollectionError("collection execution requires uv.lock")


def initialize_campaign(root: Path, campaign_id: str, *, smoke: bool) -> dict[str, Any]:
    root = validate_campaign_root(root)
    _require_clean_source()
    plan = build_plan(root, campaign_id, smoke=smoke)
    if not plan["executable"]:
        raise CollectionError("collection plan contains non-integrated runners")
    command = [
        sys.executable,
        "-m",
        "experiments.collections.gamma_gated_sparsity",
        "run",
        "--campaign-root",
        str(root),
    ]
    initialize_run(
        root,
        run_id=campaign_id,
        kind="campaign",
        experiment=None,
        collection=COLLECTION,
        provenance_notes="isolated smoke campaign" if smoke else "publication campaign",
        command=command,
        repository=REPO,
    )
    exp022_root = root / "exp022"
    exp022_root.rmdir()
    manifest_command = [
        sys.executable,
        "-m",
        "experiments.exp022",
        "--campaign-manifest",
        str(exp022_root),
        "--campaign-id",
        campaign_id,
        "--tier",
        "all",
    ]
    if smoke:
        manifest_command.append("--plumbing")
    subprocess.run(manifest_command, cwd=REPO, check=True)
    plan["profile"] = "smoke" if smoke else "production"
    plan["source"] = load_json(root / "run.json")["source"]
    plan["exp022_manifest"] = str(exp022_root / "campaign.json")
    write_json_atomic(root / PLAN_NAME, plan)
    (root / STATUS_DIR).mkdir()
    return plan


def load_plan(root: Path) -> dict[str, Any]:
    root = validate_campaign_root(root)
    plan = load_json(root / PLAN_NAME)
    if plan.get("collection") != COLLECTION:
        raise CollectionError("campaign plan belongs to another collection")
    if plan.get("campaign_root") != str(root):
        raise CollectionError("campaign plan root does not match its location")
    source = source_provenance()
    repair_sources = [
        source
        for repair in (plan.get("repairs") or {}).values()
        if isinstance(repair, dict)
        for source in (repair.get("source"), repair.get("integration_source"))
        if isinstance(source, dict)
    ]
    if source not in [plan.get("source"), *repair_sources]:
        raise CollectionError(
            "campaign source commit or lockfile differs from checkout"
        )
    if source.get("git_clean") is not True:
        raise CollectionError("campaign execution requires a clean Git checkout")
    return plan


def rows_in_order(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for stage in plan["stages"] for row in stage["experiments"]]


def _runner_environment(plan: dict[str, Any], row: dict[str, Any]) -> dict[str, str]:
    paths = row["paths"]
    environment = {
        **os.environ,
        "PINGLAB_REQUIRE_ISOLATED": "1",
        "PINGLAB_RUN_STATE_DIR": paths["state"],
        "PINGLAB_RUN_DERIVED_DIR": paths["derived"],
        "PINGLAB_RUN_LOG_DIR": paths["logs"],
        "PINGLAB_COLLECTION_DERIVED_ROOT": str(
            Path(plan["campaign_root"]) / "derived/.artifacts"
        ),
        "PINGLAB_TRAINING_ROOT": str(Path(plan["exp022_manifest"]).parent / "cells"),
        "PINGLAB_CAMPAIGN_ID": plan["campaign_id"],
    }
    if plan["profile"] == "smoke":
        environment["PINGLAB_SMOKE"] = "1"
    return environment


def _status_path(root: Path, slug: str) -> Path:
    return root / STATUS_DIR / f"{slug}.json"


def _outputs_valid(row: dict[str, Any]) -> bool:
    return all(Path(path).is_file() for path in row["required_outputs"])


def _collection_provenance(plan: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    repair = (plan.get("repairs") or {}).get(row["slug"])
    source = repair["source"] if repair else plan["source"]
    exp022 = load_json(Path(plan["exp022_manifest"]))
    provenance = {
        "campaign_id": plan["campaign_id"],
        "collection": plan["collection"],
        "experiment": row["slug"],
        "source_git_commit": source["git_commit"],
        "lockfile_sha256": (source.get("lockfile") or {}).get("sha256"),
        "exp022_manifest_sha256": exp022["manifest_sha256"],
        "dependencies": list(row["dependencies"]),
        "training_run": row.get("training_run"),
    }
    if repair:
        provenance["repair"] = {
            "base_source_git_commit": plan["source"]["git_commit"],
            "repair_run_root": repair["repair_run_root"],
            "integrated_at_utc": repair["integrated_at_utc"],
        }
    return provenance


def _outputs_valid_for_plan(plan: dict[str, Any], row: dict[str, Any]) -> bool:
    if not _outputs_valid(row):
        return False
    try:
        document = load_json(Path(row["required_outputs"][0]))
        if plan.get("composition"):
            composition = load_json(Path(plan["campaign_root"]) / "composition.json")
            source = composition.get("experiments", {}).get(row["slug"])
            provenance = document.get("collection_provenance")
            output_root = (
                Path(plan["campaign_root"]) / "derived/.artifacts" / row["slug"]
            )
            if not isinstance(source, dict) or not isinstance(provenance, dict):
                return False
            snapshot = _directory_snapshot(output_root)
            return (
                provenance.get("campaign_id") == source.get("run_id")
                and provenance.get("source_git_commit")
                == source.get("source_git_commit")
                and snapshot["file_count"] == source.get("file_count")
                and snapshot["payload_digest"] == source.get("payload_digest")
            )
        return document.get("collection_provenance") == _collection_provenance(
            plan, row
        )
    except CollectionError:
        return False


def _stamp_collection_provenance(plan: dict[str, Any], row: dict[str, Any]) -> None:
    """Bind one experiment's scientific payload to the immutable campaign."""
    output = Path(row["required_outputs"][0])
    document = load_json(output)
    provenance = _collection_provenance(plan, row)
    existing = document.get("collection_provenance")
    if existing is not None and existing != provenance:
        raise CollectionError(
            f"{row['slug']} numbers.json belongs to a different campaign"
        )
    write_json_atomic(output, {**document, "collection_provenance": provenance})


def _write_status(root: Path, slug: str, **fields: object) -> None:
    current = {}
    path = _status_path(root, slug)
    if path.exists():
        current = load_json(path)
    write_json_atomic(path, {**current, "experiment": slug, **fields})


def integrate_repair(root: Path, repair_root: Path, slug: str) -> dict[str, Any]:
    """Atomically register one repaired downstream result in a frozen campaign."""
    root = validate_campaign_root(root)
    repair_root = validate_campaign_root(repair_root)
    plan = load_json(root / PLAN_NAME)
    if plan.get("collection") != COLLECTION or plan.get("campaign_root") != str(root):
        raise CollectionError("invalid campaign plan for repair integration")
    if (root / "inventory.json").exists():
        raise CollectionError("cannot repair a finalized campaign")

    rows = {row["slug"]: row for row in rows_in_order(plan)}
    if slug not in rows or slug == "exp022":
        raise CollectionError(f"unknown repairable downstream experiment: {slug}")
    if slug in (plan.get("repairs") or {}):
        raise CollectionError(f"campaign already registers a repair for {slug}")

    repair_manifest = load_json(repair_root / "repair-run.json")
    integration_source = source_provenance()
    if integration_source.get("git_clean") is not True:
        raise CollectionError("repair integration requires a clean Git checkout")
    if repair_manifest.get("experiment") != slug:
        raise CollectionError("repair run belongs to another experiment")
    if Path(str(repair_manifest.get("base_campaign_root"))).resolve() != root:
        raise CollectionError("repair run belongs to another base campaign")
    if repair_manifest.get("base_campaign_source_git_commit") != plan["source"].get(
        "git_commit"
    ):
        raise CollectionError("repair run base commit differs from campaign")
    repair_commit = repair_manifest.get("source_git_commit")
    if not isinstance(repair_commit, str):
        raise CollectionError("repair run does not record its source commit")
    ancestry = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            repair_commit,
            integration_source["git_commit"],
        ],
        cwd=REPO,
    )
    if ancestry.returncode != 0:
        raise CollectionError(
            "repair run source is not an ancestor of the integration checkout"
        )
    lockfile = subprocess.run(
        ["git", "show", f"{repair_commit}:uv.lock"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    repair_source = {
        "git_commit": repair_commit,
        "git_clean": True,
        "lockfile": {
            "path": "uv.lock",
            "sha256": hashlib.sha256(lockfile).hexdigest(),
        },
    }
    manifest_sha = _sha256(Path(plan["exp022_manifest"]))
    if repair_manifest.get("exp022_manifest_file_sha256") != manifest_sha:
        raise CollectionError("repair run used a different exp022 manifest")

    row = rows[slug]
    source_dir = repair_root / "derived/.artifacts" / slug
    destination = Path(row["paths"]["derived"])
    required_names = [Path(path).name for path in row["required_outputs"]]
    missing = [name for name in required_names if not (source_dir / name).is_file()]
    if missing:
        raise CollectionError("repair run is missing outputs: " + ", ".join(missing))

    integrated_at = utc_now()
    repair = {
        "source": repair_source,
        "integration_source": integration_source,
        "repair_run_root": str(repair_root),
        "repair_run_manifest_sha256": _sha256(repair_root / "repair-run.json"),
        "integrated_at_utc": integrated_at,
    }
    updated_plan = {**plan, "repairs": {**(plan.get("repairs") or {}), slug: repair}}
    run_path = root / "run.json"
    run = load_json(run_path)
    status_path = _status_path(root, slug)
    previous_status = load_json(status_path) if status_path.exists() else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{slug}.pre-repair")
    if backup.exists():
        raise CollectionError(f"repair backup already exists: {backup}")
    staging = Path(tempfile.mkdtemp(dir=destination.parent, prefix=f".{slug}.repair-"))
    moved_existing = False
    installed_repair = False
    try:
        shutil.copytree(source_dir, staging, dirs_exist_ok=True)
        numbers = staging / "numbers.json"
        document = load_json(numbers)
        provenance = _collection_provenance(updated_plan, row)
        existing = document.get("collection_provenance")
        if existing is not None and existing != provenance:
            raise CollectionError("repair output belongs to a different campaign")
        write_json_atomic(numbers, {**document, "collection_provenance": provenance})

        if destination.exists():
            destination.rename(backup)
            moved_existing = True
        staging.rename(destination)
        installed_repair = True
        write_json_atomic(root / PLAN_NAME, updated_plan)

        upstream = list(run.get("upstream") or [])
        repair_ref = f"{slug}-repair:{repair_source['git_commit']}:{repair_root}"
        if repair_ref not in upstream:
            upstream.append(repair_ref)
        notes = run.get("provenance_notes", "")
        note = (
            f"{slug} repaired by {repair_source['git_commit']} from {repair_root} "
            f"and integrated by {integration_source['git_commit']}; "
            f"all other outputs retain campaign source {plan['source']['git_commit']}"
        )
        write_json_atomic(
            run_path,
            {
                **run,
                "upstream": upstream,
                "provenance_notes": f"{notes}; {note}".strip("; "),
            },
        )
        _write_status(
            root,
            slug,
            state="complete",
            repair_source_git_commit=repair_source["git_commit"],
            integration_source_git_commit=integration_source["git_commit"],
            repair_run_root=str(repair_root),
            ended_at_utc=integrated_at,
        )
        if moved_existing:
            shutil.rmtree(backup)
    except BaseException:
        if installed_repair and destination.exists():
            shutil.rmtree(destination)
        if moved_existing and backup.exists() and not destination.exists():
            backup.rename(destination)
        write_json_atomic(root / PLAN_NAME, plan)
        write_json_atomic(run_path, run)
        if previous_status is None:
            status_path.unlink(missing_ok=True)
        else:
            write_json_atomic(status_path, previous_status)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return {
        "campaign_id": plan["campaign_id"],
        "experiment": slug,
        "source_git_commit": repair_source["git_commit"],
        "integration_source_git_commit": integration_source["git_commit"],
        "repair_run_root": str(repair_root),
        "outputs": required_names,
    }


def _run_exp022(plan: dict[str, Any], row: dict[str, Any]) -> None:
    root = Path(plan["campaign_root"])
    manifest = Path(plan["exp022_manifest"])
    environment = _runner_environment(plan, row)
    list_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.exp022",
            "--campaign-list",
            str(manifest),
            "--retry-only",
        ],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
    )
    if list_result.returncode != 0:
        detail = list_result.stderr.strip() or list_result.stdout.strip()
        raise CollectionError(f"exp022 campaign preflight failed: {detail}")
    cells = [line for line in list_result.stdout.splitlines() if line]
    _write_status(root, "exp022", state="running", started_at_utc=utc_now())
    for cell in cells:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.exp022",
                "--campaign-train-cell",
                cell,
                "--campaign",
                str(manifest),
            ],
            cwd=REPO,
            env=environment,
            check=True,
        )
    _aggregate_exp022(plan, row, environment=environment)


def _aggregate_exp022(
    plan: dict[str, Any],
    row: dict[str, Any],
    *,
    environment: dict[str, str] | None = None,
) -> None:
    root = Path(plan["campaign_root"])
    manifest = Path(plan["exp022_manifest"])
    environment = environment or _runner_environment(plan, row)
    _write_status(root, "exp022", state="aggregating", started_at_utc=utc_now())
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.exp022",
            "--campaign-aggregate",
            str(manifest),
        ],
        cwd=REPO,
        env=environment,
        check=True,
    )
    if not _outputs_valid(row):
        raise CollectionError("exp022 aggregation did not produce numbers.json")
    _stamp_collection_provenance(plan, row)
    _write_status(root, "exp022", state="complete", ended_at_utc=utc_now())


def _run_downstream(plan: dict[str, Any], row: dict[str, Any]) -> None:
    root = Path(plan["campaign_root"])
    slug = row["slug"]
    if _outputs_valid_for_plan(plan, row):
        _write_status(root, slug, state="complete", resumed=True)
        return
    _write_status(root, slug, state="running", started_at_utc=utc_now())
    try:
        subprocess.run(
            row["command"],
            cwd=REPO,
            env=_runner_environment(plan, row),
            check=True,
        )
    except BaseException:
        _write_status(root, slug, state="failed", ended_at_utc=utc_now())
        raise
    if not _outputs_valid(row):
        _write_status(root, slug, state="failed", ended_at_utc=utc_now())
        raise CollectionError(f"{slug} completed without required outputs")
    _stamp_collection_provenance(plan, row)
    _write_status(root, slug, state="complete", ended_at_utc=utc_now())


def run_local(root: Path) -> None:
    plan = load_plan(root)
    for row in rows_in_order(plan):
        if row["slug"] == "exp022":
            if not _outputs_valid_for_plan(plan, row):
                _run_exp022(plan, row)
        else:
            for dependency in row["dependencies"]:
                dependency_row = next(
                    item for item in rows_in_order(plan) if item["slug"] == dependency
                )
                if not _outputs_valid_for_plan(plan, dependency_row):
                    raise CollectionError(
                        f"{row['slug']} dependency {dependency} is incomplete"
                    )
            _run_downstream(plan, row)


def aggregate_exp022(root: Path) -> None:
    """Aggregate an already-complete exp022 bank without training cells."""
    plan = load_plan(root)
    row = next(item for item in rows_in_order(plan) if item["slug"] == "exp022")
    _aggregate_exp022(plan, row)


def run_experiment(root: Path, slug: str) -> None:
    """Run one downstream node after validating its declared dependencies."""
    plan = load_plan(root)
    rows = {row["slug"]: row for row in rows_in_order(plan)}
    if slug not in rows or slug == "exp022":
        raise CollectionError(f"unknown downstream experiment: {slug}")
    row = rows[slug]
    for dependency in row["dependencies"]:
        if not _outputs_valid_for_plan(plan, rows[dependency]):
            raise CollectionError(f"{slug} dependency {dependency} is incomplete")
    _run_downstream(plan, row)


def run_experiment_shard(root: Path, slug: str, index: int, count: int) -> None:
    """Run one deterministic shard of an experiment's resumable inference jobs."""
    plan = load_plan(root)
    rows = {row["slug"]: row for row in rows_in_order(plan)}
    if slug not in rows or slug == "exp022":
        raise CollectionError(f"unknown downstream experiment: {slug}")
    row = rows[slug]
    for dependency in row["dependencies"]:
        if not _outputs_valid_for_plan(plan, rows[dependency]):
            raise CollectionError(f"{slug} dependency {dependency} is incomplete")

    environment = _runner_environment(plan, row)
    os.environ.update(environment)
    from .workloads import execute_shard

    status_path = root / "collection-shards" / slug / f"{index}.json"
    write_json_atomic(
        status_path,
        {
            "experiment": slug,
            "shard_index": index,
            "shard_count": count,
            "state": "running",
            "started_at_utc": utc_now(),
        },
    )
    try:
        result = execute_shard(slug, index, count, smoke=plan.get("profile") == "smoke")
    except BaseException:
        write_json_atomic(
            status_path,
            {
                "experiment": slug,
                "shard_index": index,
                "shard_count": count,
                "state": "failed",
                "ended_at_utc": utc_now(),
            },
        )
        raise
    write_json_atomic(
        status_path,
        {
            **result,
            "experiment": slug,
            "state": "complete",
            "ended_at_utc": utc_now(),
        },
    )


def campaign_status(root: Path) -> dict[str, Any]:
    plan = load_plan(root)
    rows = []
    for row in rows_in_order(plan):
        status_path = _status_path(Path(plan["campaign_root"]), row["slug"])
        status = (
            load_json(status_path) if status_path.exists() else {"state": "pending"}
        )
        rows.append(
            {
                "experiment": row["slug"],
                "state": status.get("state", "pending"),
                "outputs_valid": _outputs_valid_for_plan(plan, row),
            }
        )
    return {"campaign_id": plan["campaign_id"], "experiments": rows}


def validate_campaign(root: Path) -> dict[str, Any]:
    status = campaign_status(root)
    invalid = [row for row in status["experiments"] if not row["outputs_valid"]]
    if invalid:
        raise CollectionError(
            "campaign outputs incomplete: "
            + ", ".join(row["experiment"] for row in invalid)
        )
    return status


def finalize_campaign(root: Path) -> dict[str, Any]:
    """Validate every planned output, then freeze its Pingstore inventory."""
    root = validate_campaign_root(root)
    validate_campaign(root)
    run = load_json(root / "run.json")
    inventory_path = root / "inventory.json"
    if run.get("status") != "complete" or not inventory_path.is_file():
        capture_campaign_metadata(root, load_plan(root))
        inventory = inventory_payload(root, run_id=run["run_id"])
        write_json_atomic(inventory_path, inventory)
        run["status"] = "complete"
        write_json_atomic(root / "run.json", run)
        run = load_json(root / "run.json")
    if run.get("status") != "complete" or not inventory_path.is_file():
        raise CollectionError("Pingstore did not finalize the campaign inventory")
    inventory = load_json(inventory_path)
    return {
        "campaign_id": run["run_id"],
        "status": run["status"],
        "file_count": inventory["file_count"],
        "total_size_bytes": inventory["total_size_bytes"],
        "payload_digest": inventory["payload_digest"],
    }


def _checkout_source(checkout: Path) -> dict[str, Any]:
    checkout = checkout.resolve()
    if checkout == REPO.resolve():
        raise CollectionError(
            "publication build requires a separate disposable checkout"
        )
    if not (checkout / ".git").exists():
        raise CollectionError(f"publication checkout is not a Git worktree: {checkout}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    clean = not subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lock = checkout / "uv.lock"
    return {
        "git_commit": commit,
        "git_clean": clean,
        "lockfile": {"path": "uv.lock", "sha256": _sha256(lock)}
        if lock.is_file()
        else None,
    }


def build_publication(root: Path, checkout: Path) -> dict[str, Any]:
    """Promote a finalized campaign into a separate checkout and build it."""
    root = validate_campaign_root(root)
    plan = load_plan(root)
    validate_campaign(root)
    run = load_json(root / "run.json")
    if run.get("status") != "complete" or not (root / "inventory.json").is_file():
        raise CollectionError("campaign must be finalized before publication build")
    checkout = checkout.resolve()
    target_source = _checkout_source(checkout)
    allowed_sources = [
        plan["source"],
        *[
            source
            for repair in (plan.get("repairs") or {}).values()
            if isinstance(repair, dict)
            for source in (repair.get("source"), repair.get("integration_source"))
            if isinstance(source, dict)
        ],
    ]
    if target_source not in allowed_sources:
        raise CollectionError(
            "publication checkout must be clean and match a campaign source commit and lockfile"
        )
    uv = shutil.which("uv")
    if uv is None:
        raise CollectionError("uv is required for publication build")
    promoted = []
    for row in rows_in_order(plan):
        promote_experiment(
            root,
            row["slug"],
            artifacts_root=checkout / ".artifacts",
        )
        promoted.append(row["slug"])
    built = subprocess.run(
        [uv, "run", "--frozen", "--project", str(checkout), "demolab", "build"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    output = built.stdout + built.stderr
    if "stubbed:" in output or "failed to build" in output:
        raise CollectionError("Demolab build produced stubbed entries:\n" + output)
    return {
        "campaign_id": plan["campaign_id"],
        "checkout": str(checkout),
        "promoted": promoted,
        "site": str(checkout / "artifacts" / "site"),
    }
