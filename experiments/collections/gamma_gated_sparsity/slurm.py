"""Wilkes3 submission and status support for the collection campaign."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Any

from .execution import (
    CollectionError,
    _outputs_valid_for_plan,
    campaign_status,
    load_json,
    load_plan,
    rows_in_order,
    utc_now,
    write_json_atomic,
)
from .plan import REPO, validate_campaign_root
from .workloads import shard_count

TIERS = ("standard", "fine_dt", "canonical_coba", "canonical_ping", "variable_rate")
SUBMISSION_NAME = "collection-submission.json"
CANARY_SUBMISSION_PREFIX = "canary-submission"
SLURM_TIME = re.compile(r"^[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]$")


def _valid_time(value: object) -> bool:
    return (
        isinstance(value, str)
        and SLURM_TIME.fullmatch(value) is not None
        and value != "00:00:00"
    )


def _positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise CollectionError(f"Slurm field {field} must be a positive integer")
    return value


def load_resources(path: Path) -> dict[str, Any]:
    """Load site/account details kept outside Git and measured before production."""
    config = load_json(path.resolve())
    required_strings = ("account", "partition", "mnist_cache", "uv")
    for field in required_strings:
        if not isinstance(config.get(field), str) or not config[field].strip():
            raise CollectionError(f"Slurm resources require non-empty {field}")
    if not Path(config["mnist_cache"]).is_absolute():
        raise CollectionError("Slurm mnist_cache must be an absolute path")
    if not Path(config["uv"]).is_absolute():
        raise CollectionError("Slurm uv must be an absolute path")
    tiers = config.get("exp022")
    if not isinstance(tiers, dict) or set(tiers) != set(TIERS):
        raise CollectionError(f"Slurm exp022 resources must define: {', '.join(TIERS)}")
    for tier in TIERS:
        row = tiers[tier]
        if not isinstance(row, dict) or not _valid_time(row.get("time")):
            raise CollectionError(f"Slurm exp022.{tier} requires time")
        _positive_int(row.get("concurrency"), f"exp022.{tier}.concurrency")
        _positive_int(row.get("cpus"), f"exp022.{tier}.cpus")
        _positive_int(row.get("memory_gb"), f"exp022.{tier}.memory_gb")
        _positive_int(row.get("gpus"), f"exp022.{tier}.gpus")
    jobs = config.get("jobs")
    if not isinstance(jobs, dict):
        raise CollectionError("Slurm resources require jobs")
    for name in ("aggregate", "downstream", "heavy_downstream", "finalize"):
        row = jobs.get(name)
        if not isinstance(row, dict):
            raise CollectionError(f"Slurm resources require jobs.{name}")
        if not _valid_time(row.get("time")):
            raise CollectionError(f"Slurm jobs.{name} requires time")
        _positive_int(row.get("cpus"), f"jobs.{name}.cpus")
        _positive_int(row.get("memory_gb"), f"jobs.{name}.memory_gb")
        gpus = row.get("gpus", 0)
        if not isinstance(gpus, int) or isinstance(gpus, bool) or gpus < 0:
            raise CollectionError(
                f"Slurm jobs.{name}.gpus must be a non-negative integer"
            )
    return config


def _run(
    command: list[str], *, submit: bool, test_only: bool, dry_id: str
) -> str:
    if not submit and not test_only:
        return dry_id
    actual = [*command]
    if test_only:
        actual.insert(1, "--test-only")
    result = subprocess.run(
        actual, cwd=REPO, check=True, capture_output=True, text=True
    )
    if test_only:
        return "<test-only>"
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise CollectionError(
            f"sbatch returned an invalid job ID: {result.stdout.strip()}"
        )
    return job_id


def _dependency(job_ids: list[str]) -> list[str]:
    return [f"--dependency=afterok:{':'.join(job_ids)}"] if job_ids else []


def _job_args(resources: dict[str, Any], kind: str) -> list[str]:
    row = resources["jobs"][kind]
    result = [
        f"--account={resources['account']}",
        f"--partition={resources['partition']}",
        f"--time={row['time']}",
        f"--cpus-per-task={row['cpus']}",
        f"--mem={row['memory_gb']}G",
    ]
    if row.get("gpus", 0):
        result.append(f"--gres=gpu:{row['gpus']}")
    return result


def _exp022_cells(manifest: Path, tier: str, uv: str) -> list[str]:
    result = subprocess.run(
        [
            uv,
            "run",
            "--frozen",
            "python",
            "-m",
            "experiments.exp022.compute",
            "--campaign-list",
            str(manifest),
            "--tier",
            tier,
            "--retry-only",
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _submit_exp022_tier(
    plan: dict[str, Any],
    resources: dict[str, Any],
    tier: str,
    *,
    attempt: str,
    submit: bool,
    test_only: bool,
    cells_override: list[str] | None = None,
    job_prefix: str = "exp022",
) -> dict[str, Any] | None:
    manifest = Path(plan["exp022_manifest"])
    cells = (
        cells_override
        if cells_override is not None
        else _exp022_cells(manifest, tier, resources["uv"])
    )
    if not cells:
        return None
    selection = (
        Path(plan["campaign_root"]) / "submissions" / f"exp022-{tier}-{attempt}.cells"
    )
    if submit:
        selection.parent.mkdir(parents=True, exist_ok=True)
        selection.write_text("\n".join(cells) + "\n")
        selection.chmod(0o444)
    tier_resources = resources["exp022"][tier]
    logs = Path(plan["campaign_root"]) / "logs" / "exp022"
    if submit:
        logs.mkdir(parents=True, exist_ok=True)
    array = f"0-{len(cells) - 1}%{tier_resources['concurrency']}"
    exports = ",".join(
        (
            f"PINGLAB_ROOT={REPO}",
            f"EXP022_MANIFEST={manifest}",
            f"EXP022_TIER={tier}",
            f"EXP022_SELECTION={selection}",
            f"EXP022_UV={resources['uv']}",
            f"PINGLAB_DATA_ROOT={resources['mnist_cache']}",
        )
    )
    command = [
        "sbatch",
        "--parsable",
        f"--account={resources['account']}",
        f"--partition={resources['partition']}",
        f"--time={tier_resources['time']}",
        f"--cpus-per-task={tier_resources['cpus']}",
        f"--mem={tier_resources['memory_gb']}G",
        f"--gres=gpu:{tier_resources['gpus']}",
        f"--array={array}",
        f"--output={logs}/%A_%a.out",
        f"--error={logs}/%A_%a.err",
        f"--export={exports}",
        str(REPO / "experiments" / "exp022" / "slurm" / "train-array.sbatch"),
    ]
    return {
        "name": f"{job_prefix}-{tier}",
        "job_id": _run(
            command,
            submit=submit,
            test_only=test_only,
            dry_id=f"<{tier}-job-id>",
        ),
        "cells": cells,
        "command": command,
    }


def submit_canaries(
    root: Path,
    resources_path: Path,
    *,
    submit: bool = False,
    test_only: bool = False,
) -> dict[str, Any]:
    """Plan or submit one still-missing production cell from every exp022 tier."""
    root = validate_campaign_root(root)
    plan = load_plan(root)
    resources = load_resources(resources_path)
    if submit and test_only:
        raise CollectionError("live submission and test-only are mutually exclusive")
    if plan.get("profile") != "production":
        raise CollectionError("resource canaries require a production campaign")

    manifest = Path(plan["exp022_manifest"])
    attempt = utc_now().replace(":", "").replace("-", "")
    jobs = []
    payload = {
        "campaign_id": plan["campaign_id"],
        "created_at_utc": utc_now(),
        "mode": "submitting" if submit else "test-only" if test_only else "dry-run",
        "purpose": "production-resource-canaries",
        "source": plan["source"],
        "exp022_manifest_sha256": load_json(manifest)["manifest_sha256"],
        "resource_file_sha256": hashlib.sha256(
            resources_path.resolve().read_bytes()
        ).hexdigest(),
        "resources_path": str(resources_path.resolve()),
        "resources": resources,
        "jobs": jobs,
    }
    record_path = (
        root / "submissions" / f"{CANARY_SUBMISSION_PREFIX}-{attempt}.json"
    )
    for tier in TIERS:
        missing = _exp022_cells(manifest, tier, resources["uv"])
        if not missing:
            continue
        job = _submit_exp022_tier(
            plan,
            resources,
            tier,
            attempt=f"canary-{attempt}",
            submit=submit,
            test_only=test_only,
            cells_override=[missing[0]],
            job_prefix="exp022-canary",
        )
        assert job is not None
        jobs.append(job)
        if submit:
            write_json_atomic(record_path, payload)
    if submit:
        payload["mode"] = "submitted"
        write_json_atomic(record_path, payload)
    return payload


def _submit_job(
    plan: dict[str, Any],
    resources: dict[str, Any],
    *,
    name: str,
    action: str,
    kind: str,
    dependencies: list[str],
    submit: bool,
    test_only: bool,
    slug: str | None = None,
) -> dict[str, Any]:
    root = Path(plan["campaign_root"])
    logs = (
        root.parent / ".scheduler-logs" / plan["campaign_id"]
        if kind == "finalize"
        else root / "logs" / "collection"
    )
    if submit:
        logs.mkdir(parents=True, exist_ok=True)
    command = [
        "sbatch",
        "--parsable",
        *_job_args(resources, kind),
        *(_dependency(dependencies) if not test_only else []),
        f"--job-name={name}",
        f"--output={logs}/%x_%j.out",
        f"--error={logs}/%x_%j.err",
        f"--export=ALL,PINGLAB_ROOT={REPO}",
        str(
            REPO
            / "experiments"
            / "collections"
            / "gamma_gated_sparsity"
            / "collection-job.sbatch"
        ),
        action,
        str(root),
        resources["uv"],
        resources["mnist_cache"],
    ]
    if slug is not None:
        command.append(slug)
    return {
        "name": name,
        "job_id": _run(
            command,
            submit=submit,
            test_only=test_only,
            dry_id=f"<{name}-job-id>",
        ),
        "command": command,
    }


def _submit_experiment_shards(
    plan: dict[str, Any],
    resources: dict[str, Any],
    *,
    slug: str,
    dependencies: list[str],
    submit: bool,
    test_only: bool,
) -> dict[str, Any]:
    root = Path(plan["campaign_root"])
    count = shard_count(slug)
    logs = root / "logs" / "collection" / f"{slug}-shards"
    if submit:
        logs.mkdir(parents=True, exist_ok=True)
    name = f"ggs-{slug}-inference"
    command = [
        "sbatch",
        "--parsable",
        *_job_args(resources, "heavy_downstream"),
        *(_dependency(dependencies) if not test_only else []),
        f"--array=0-{count - 1}%{count}",
        f"--job-name={name}",
        f"--output={logs}/%A_%a.out",
        f"--error={logs}/%A_%a.err",
        f"--export=ALL,PINGLAB_ROOT={REPO}",
        str(
            REPO
            / "experiments"
            / "collections"
            / "gamma_gated_sparsity"
            / "collection-job.sbatch"
        ),
        "run-experiment-shard",
        str(root),
        resources["uv"],
        resources["mnist_cache"],
        slug,
        str(count),
    ]
    return {
        "name": name,
        "job_id": _run(
            command,
            submit=submit,
            test_only=test_only,
            dry_id=f"<{name}-job-id>",
        ),
        "experiment": slug,
        "shard_count": count,
        "partition": "ordered-round-robin",
        "command": command,
    }


def submit_campaign(
    root: Path,
    resources_path: Path,
    *,
    submit: bool = False,
    test_only: bool = False,
) -> dict[str, Any]:
    """Submit missing work with afterok dependencies, or print an exact dry run."""
    root = validate_campaign_root(root)
    plan = load_plan(root)
    resources = load_resources(resources_path)
    if submit and test_only:
        raise CollectionError("live submission and test-only are mutually exclusive")
    if plan.get("profile") not in {"smoke", "production"}:
        raise CollectionError("Slurm submission requires a smoke or production campaign")
    existing_path = root / "submissions" / SUBMISSION_NAME
    if submit and existing_path.exists():
        raise CollectionError("a Slurm submission record already exists; use resume")

    attempt = utc_now().replace(":", "").replace("-", "")
    jobs: list[dict[str, Any]] = []
    payload = {
        "campaign_id": plan["campaign_id"],
        "created_at_utc": utc_now(),
        "mode": "submitting" if submit else "test-only" if test_only else "dry-run",
        "resources_path": str(resources_path.resolve()),
        "resources": resources,
        "source": plan["source"],
        "exp022_manifest_sha256": load_json(Path(plan["exp022_manifest"]))[
            "manifest_sha256"
        ],
        "resource_file_sha256": hashlib.sha256(
            resources_path.resolve().read_bytes()
        ).hexdigest(),
        "expected_outputs": [
            output for row in rows_in_order(plan) for output in row["required_outputs"]
        ],
        "jobs": jobs,
    }

    def record(job: dict[str, Any]) -> None:
        jobs.append(job)
        if submit:
            write_json_atomic(existing_path, payload)

    by_name: dict[str, str] = {}
    rows = {row["slug"]: row for row in rows_in_order(plan)}
    if not _outputs_valid_for_plan(plan, rows["exp022"]):
        for tier in TIERS:
            job = _submit_exp022_tier(
                plan,
                resources,
                tier,
                attempt=attempt,
                submit=submit,
                test_only=test_only,
            )
            if job is not None:
                record(job)
                by_name[job["name"]] = job["job_id"]
        aggregate = _submit_job(
            plan,
            resources,
            name="ggs-exp022-aggregate",
            action="aggregate-exp022",
            kind="aggregate",
            dependencies=list(by_name.values()),
            submit=submit,
            test_only=test_only,
        )
        record(aggregate)
        by_name["exp022"] = aggregate["job_id"]

    for row in rows_in_order(plan):
        slug = row["slug"]
        if slug == "exp022" or _outputs_valid_for_plan(plan, row):
            continue
        if slug in {"exp023", "exp024", "exp025", "exp041", "exp042", "exp044", "exp046", "exp081"}:
            from .execution import _stage_adapter
            adapter = _stage_adapter(slug)
            require_staged, reserve = adapter.require_staged, adapter.reserve
            require_staged(row)
            if submit:
                reserve(REPO, row, origin="slurm-wilkes")
        dependency_ids = [by_name[d] for d in row["dependencies"] if d in by_name]
        if shard_count(slug) > 1:
            shards = _submit_experiment_shards(
                plan,
                resources,
                slug=slug,
                dependencies=dependency_ids,
                submit=submit,
                test_only=test_only,
            )
            record(shards)
            dependency_ids = [shards["job_id"]]
        job = _submit_job(
            plan,
            resources,
            name=f"ggs-{slug}",
            action="run-experiment",
            kind="downstream",
            dependencies=dependency_ids,
            submit=submit,
            test_only=test_only,
            slug=slug,
        )
        record(job)
        by_name[slug] = job["job_id"]

    leaf_ids = [
        by_name[row["slug"]]
        for row in rows.values()
        if row["slug"] in by_name
        and not any(row["slug"] in other["dependencies"] for other in rows.values())
    ]
    if jobs:
        final = _submit_job(
            plan,
            resources,
            name="ggs-finalize",
            action="finalize",
            kind="finalize",
            dependencies=leaf_ids or list(by_name.values()),
            submit=submit,
            test_only=test_only,
        )
        record(final)
    if submit:
        payload["mode"] = "submitted"
        write_json_atomic(existing_path, payload)
    return payload


def resume_campaign(
    root: Path,
    resources_path: Path,
    *,
    submit: bool = False,
    test_only: bool = False,
) -> dict[str, Any]:
    """Submit only work whose required outputs still fail validation."""
    root = validate_campaign_root(root)
    previous = root / "submissions" / SUBMISSION_NAME
    if not previous.is_file():
        raise CollectionError("cannot resume before the initial Slurm submission")
    archived = previous.with_name(
        f"collection-submission-{utc_now().replace(':', '')}.json"
    )
    if submit:
        previous.replace(archived)
    try:
        return submit_campaign(
            root, resources_path, submit=submit, test_only=test_only
        )
    except BaseException:
        if submit and archived.exists() and not previous.exists():
            archived.replace(previous)
        raise


def slurm_status(root: Path) -> dict[str, Any]:
    root = validate_campaign_root(root)
    submission = load_json(root / "submissions" / SUBMISSION_NAME)
    ids = [row["job_id"] for row in submission["jobs"] if str(row["job_id"]).isdigit()]
    scheduler: dict[str, str] = {}
    if ids:
        result = subprocess.run(
            ["squeue", "--noheader", "--jobs", ",".join(ids), "--format=%i|%T"],
            check=True,
            capture_output=True,
            text=True,
        )
        scheduler = dict(
            line.split("|", 1) for line in result.stdout.splitlines() if "|" in line
        )
        missing = [job_id for job_id in ids if job_id not in scheduler]
        if missing:
            accounting = subprocess.run(
                [
                    "sacct",
                    "--noheader",
                    "--allocations",
                    "--jobs",
                    ",".join(missing),
                    "--format=JobIDRaw,State",
                    "--parsable2",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            for line in accounting.stdout.splitlines():
                if "|" not in line:
                    continue
                job_id, state, *_rest = line.split("|")
                if job_id in missing:
                    scheduler[job_id] = state
    return {
        "campaign": campaign_status(root),
        "submission": submission,
        "scheduler": scheduler,
    }
