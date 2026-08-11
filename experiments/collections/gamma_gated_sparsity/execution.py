"""Resumable local execution for one isolated collection campaign."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    clean = not subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    lock = REPO / "uv.lock"
    return {
        "git_commit": commit,
        "git_clean": clean,
        "lockfile": {"path": "uv.lock", "sha256": _sha256(lock)}
        if lock.is_file() else None,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
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
    plan = build_plan(root, campaign_id)
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
    subprocess.run(
        [
            sys.executable, "-m", "tools.runstore", "init", str(root),
            "--run-id", campaign_id, "--kind", "campaign",
            "--collection", COLLECTION,
            "--provenance-notes",
            "isolated smoke campaign" if smoke else "publication campaign",
            "--command", *command,
        ],
        cwd=REPO,
        check=True,
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
    if source != plan.get("source"):
        raise CollectionError("campaign source commit or lockfile differs from checkout")
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


def _write_status(root: Path, slug: str, **fields: object) -> None:
    current = {}
    path = _status_path(root, slug)
    if path.exists():
        current = load_json(path)
    write_json_atomic(path, {**current, "experiment": slug, **fields})


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
        check=True,
        capture_output=True,
        text=True,
    )
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
    _write_status(root, "exp022", state="complete", ended_at_utc=utc_now())


def _run_downstream(plan: dict[str, Any], row: dict[str, Any]) -> None:
    root = Path(plan["campaign_root"])
    slug = row["slug"]
    if _outputs_valid(row):
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
    _write_status(root, slug, state="complete", ended_at_utc=utc_now())


def run_local(root: Path) -> None:
    plan = load_plan(root)
    for row in rows_in_order(plan):
        if row["slug"] == "exp022":
            if not _outputs_valid(row):
                _run_exp022(plan, row)
        else:
            for dependency in row["dependencies"]:
                dependency_row = next(
                    item for item in rows_in_order(plan) if item["slug"] == dependency
                )
                if not _outputs_valid(dependency_row):
                    raise CollectionError(
                        f"{row['slug']} dependency {dependency} is incomplete"
                    )
            _run_downstream(plan, row)


def campaign_status(root: Path) -> dict[str, Any]:
    plan = load_plan(root)
    rows = []
    for row in rows_in_order(plan):
        status_path = _status_path(Path(plan["campaign_root"]), row["slug"])
        status = load_json(status_path) if status_path.exists() else {"state": "pending"}
        rows.append({
            "experiment": row["slug"],
            "state": status.get("state", "pending"),
            "outputs_valid": _outputs_valid(row),
        })
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
    """Validate every planned output, then freeze the runstore inventory."""
    root = validate_campaign_root(root)
    validate_campaign(root)
    run = load_json(root / "run.json")
    inventory_path = root / "inventory.json"
    if run.get("status") != "complete" or not inventory_path.is_file():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.runstore",
                "inspect",
                str(root),
                "--finalize",
            ],
            cwd=REPO,
            check=True,
        )
        run = load_json(root / "run.json")
    if run.get("status") != "complete" or not inventory_path.is_file():
        raise CollectionError("runstore did not finalize the campaign inventory")
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
        raise CollectionError("publication build requires a separate disposable checkout")
    if not (checkout / ".git").exists():
        raise CollectionError(f"publication checkout is not a Git worktree: {checkout}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=checkout, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    clean = not subprocess.run(
        ["git", "status", "--porcelain"], cwd=checkout, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    lock = checkout / "uv.lock"
    return {
        "git_commit": commit,
        "git_clean": clean,
        "lockfile": {"path": "uv.lock", "sha256": _sha256(lock)}
        if lock.is_file() else None,
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
    if target_source != plan["source"]:
        raise CollectionError(
            "publication checkout must be clean and match the campaign commit and lockfile"
        )
    uv = shutil.which("uv")
    if uv is None:
        raise CollectionError("uv is required for publication build")
    promoted = []
    for row in rows_in_order(plan):
        subprocess.run(
            [
                uv,
                "run",
                "--frozen",
                "--project",
                str(checkout),
                "python",
                "-m",
                "tools.runstore",
                "promote",
                str(root),
                row["slug"],
            ],
            cwd=checkout,
            check=True,
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
