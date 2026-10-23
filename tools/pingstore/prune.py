"""Hash-bound pruning of unused, unprotected operational runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
from collections import defaultdict
from pathlib import Path

from .allocation import allocation_watermarks, persist_watermarks
from .contracts import (
    EXPERIMENT_RE,
    PingstoreError,
    file_sha256,
    load_json,
    validate_operational_run_directory,
)
from .discovery import discover_store, validated_run_graph
from .stages import operation_lock
from .usage import ancestry, report_usage

PLAN_SCHEMA = "pingstore.prune-plan/v5"


def _directory_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _hidden_inputs(
    runs: Path, records: dict[str, dict], reasons: dict[str, set[str]]
) -> list[dict]:
    hidden = []
    for directory in sorted(
        path for path in runs.iterdir() if path.name.startswith(".")
    ):
        if not directory.is_dir() or directory.is_symlink():
            continue
        state = {"name": directory.name}
        manifest = directory / "run.json"
        reservation = directory / ".reservation.json"
        if manifest.is_file() and not manifest.is_symlink():
            state["run_json_sha256"] = file_sha256(manifest)
            record = load_json(manifest)
            inputs = record.get("inputs", {})
            if not isinstance(inputs, dict):
                raise PingstoreError(
                    f"{directory.name}: incomplete run has invalid inputs"
                )
            for reference in inputs.values():
                if not isinstance(reference, dict):
                    raise PingstoreError(
                        f"{directory.name}: incomplete run has invalid input"
                    )
                run_id = reference.get("run_id")
                parent = records.get(run_id)
                if parent is None or parent["payload_digest"] != reference.get(
                    "payload_digest"
                ):
                    raise PingstoreError(
                        f"{directory.name}: missing or changed input {run_id}"
                    )
                reasons[run_id].add(f"incomplete-input:{directory.name}")
        if reservation.is_file() and not reservation.is_symlink():
            state["reservation_sha256"] = file_sha256(reservation)
        hidden.append(state)
    return hidden


def _plan_hash(plan: dict) -> str:
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _experiment_scope(
    experiments: list[str] | tuple[str, ...] | set[str] | None,
    records: dict[str, dict],
) -> set[str] | None:
    if experiments is None:
        return None
    scope = set(experiments)
    if not scope:
        raise PingstoreError("experiment-scoped prune requires at least one experiment")
    invalid = sorted(value for value in scope if not EXPERIMENT_RE.fullmatch(value))
    if invalid:
        raise PingstoreError(f"invalid experiment filter: {', '.join(invalid)}")
    available = {record["experiment"] for record in records.values()}
    missing = sorted(scope - available)
    if missing:
        raise PingstoreError(f"experiment filter has no runs: {', '.join(missing)}")
    return scope


def build_plan(
    repo: Path,
    experiments: list[str] | tuple[str, ...] | set[str] | None = None,
    stages: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict:
    repo = repo.resolve()
    runs = repo / ".pingstore/runs"
    if not runs.is_dir() or runs.is_symlink():
        raise PingstoreError(f"prune requires a real runs directory: {runs}")
    for path in runs.iterdir():
        if path.is_symlink() or not path.is_dir():
            raise PingstoreError(
                f"prune does not accept unsupported runs entry: {path}"
            )
    # Unused runs can have missing historical inputs. Validate every payload,
    # then require complete, digest-matched ancestry for everything we retain.
    records = {
        path.name: validate_operational_run_directory(path)
        for path in sorted(runs.iterdir())
        if not path.name.startswith(".")
    }
    scope = _experiment_scope(experiments, records)
    stage_scope = set(stages) if stages is not None else None
    if stage_scope is not None and (
        not stage_scope or stage_scope - {"compute", "analyse", "present"}
    ):
        raise PingstoreError("invalid or empty stage filter")
    reasons: dict[str, set[str]] = defaultdict(set)
    if stage_scope is not None:
        for run_id, record in records.items():
            if record["stage"] not in stage_scope:
                reasons[run_id].add("out-of-stage-scope")

    if scope is not None:
        for run_id, record in records.items():
            if record["experiment"] not in scope:
                reasons[run_id].add("out-of-scope")

    selections, used_by = report_usage(repo, records, experiments=scope)
    used = set(used_by)
    for run_id, reports in used_by.items():
        reasons[run_id].update(f"used-by:{article}" for article in reports)
    markers = {}
    for run_id in records:
        marker = runs / run_id / ".protected"
        if marker.is_symlink() or (marker.exists() and not marker.is_file()):
            raise PingstoreError(f"protection marker must be a regular file: {marker}")
        if marker.exists():
            markers[run_id] = file_sha256(marker)
            reasons[run_id].add("protected-marker")
    protected = ancestry(records, set(markers))
    for run_id in protected - set(markers):
        reasons[run_id].add("protected-ancestor")
    hidden = _hidden_inputs(runs, records, reasons)
    keep = ancestry(records, set(reasons))
    validated_run_graph(runs, keep)
    for run_id in keep - set(reasons):
        reasons[run_id].add("required-ancestor")
    watermarks = allocation_watermarks(repo / ".pingstore")

    rows = []
    for run_id, record in sorted(records.items()):
        directory = runs / run_id
        rows.append(
            {
                "run_id": run_id,
                "stage": record["stage"],
                "experiment": record["experiment"],
                "used": run_id in used,
                "used_by": sorted(used_by.get(run_id, ())),
                "protected": run_id in protected,
                "protected_marker_sha256": markers.get(run_id),
                "payload_digest": record["payload_digest"],
                "run_json_sha256": file_sha256(directory / "run.json"),
                "bytes": _directory_bytes(directory),
                "reasons": sorted(reasons[run_id]) if run_id in keep else ["unused"],
            }
        )
    plan = {
        "schema": PLAN_SCHEMA,
        "policy": "keep-report-usage-manual-protection-incomplete-inputs-and-ancestry",
        "experiments": sorted(scope) if scope is not None else None,
        "stages": sorted(stage_scope) if stage_scope is not None else None,
        "allocation_watermarks": watermarks,
        "report_selections": selections,
        "hidden": hidden,
        "keep": [row for row in rows if row["run_id"] in keep],
        "prune": [row for row in rows if row["run_id"] not in keep],
    }
    return {**plan, "plan_hash": _plan_hash(plan)}


def _live_writer(directory: Path) -> bool:
    lock = directory / ".writer.lock"
    if not lock.is_file():
        return False
    try:
        pid = int(lock.read_text().strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _validate_survivors(runs: Path, expected: set[str]) -> None:
    records, _ = discover_store(runs)
    if set(records) != expected:
        raise PingstoreError("pruned store contains an unexpected completed-run set")


def _check_controls(repo: Path, runs: Path, plan: dict) -> None:
    """Detect manual protection or report-selection edits during confirmation."""
    from .contracts import validate_run

    records = {
        row["run_id"]: validate_run(load_json(runs / row["run_id"] / "run.json"))
        for row in plan["keep"] + plan["prune"]
    }
    scope = set(plan["experiments"]) if plan["experiments"] is not None else None
    selections, _ = report_usage(repo, records, experiments=scope)
    if selections != plan["report_selections"]:
        raise PingstoreError("report selections changed during pruning")
    for row in plan["keep"] + plan["prune"]:
        marker = runs / row["run_id"] / ".protected"
        if marker.is_symlink() or (marker.exists() and not marker.is_file()):
            raise PingstoreError("protection marker changed during pruning")
        digest = file_sha256(marker) if marker.exists() else None
        if digest != row["protected_marker_sha256"]:
            raise PingstoreError("protection marker changed during pruning")


def apply_plan(
    repo: Path,
    expected_hash: str,
    experiments: list[str] | tuple[str, ...] | set[str] | None = None,
    stages: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_hash):
        raise PingstoreError(
            "--confirm requires the complete sha256 plan hash from --dry-run"
        )
    repo = repo.resolve()
    store = repo / ".pingstore"
    runs = store / "runs"
    staged = store / f".prune-{expected_hash[7:19]}-runs.tmp"
    previous = store / f".prune-{expected_hash[7:19]}-runs.old"
    with operation_lock(store, exclusive=True):
        try:
            plan = build_plan(
                repo,
                experiments,
                stages,
            )
            if plan["plan_hash"] != expected_hash:
                raise PingstoreError(
                    f"prune plan changed: expected {expected_hash}, now {plan['plan_hash']}"
                )
            for state in plan["hidden"]:
                directory = runs / state["name"]
                if _live_writer(directory):
                    raise PingstoreError(
                        f"active writer prevents pruning: {directory.name}"
                    )
            if not plan["prune"]:
                return plan
            if staged.exists() or previous.exists():
                raise PingstoreError(
                    "unfinished prune staging directory requires recovery"
                )

            keep_ids = {row["run_id"] for row in plan["keep"]}
            # Persist counters before any run can disappear; dry-runs never write them.
            persist_watermarks(store, plan["allocation_watermarks"])
            staged.mkdir()
            for path in sorted(runs.iterdir()):
                if path.name.startswith(".") or path.name in keep_ids:
                    if path.is_dir() and not path.is_symlink():
                        shutil.copytree(
                            path,
                            staged / path.name,
                            copy_function=os.link,
                            symlinks=True,
                        )
                    else:
                        raise PingstoreError(
                            f"unsupported entry in runs directory: {path}"
                        )
            _validate_survivors(staged, keep_ids)
            _check_controls(repo, runs, plan)
            os.replace(runs, previous)
            try:
                os.replace(staged, runs)
                _validate_survivors(runs, keep_ids)
                _check_controls(repo, previous, plan)
            except BaseException:
                if runs.exists():
                    os.replace(runs, staged)
                os.replace(previous, runs)
                raise
            shutil.rmtree(previous)
            return plan
        finally:
            if staged.exists() and runs.exists():
                shutil.rmtree(staged)


def render_plan(plan: dict) -> str:
    keep_bytes = sum(row["bytes"] for row in plan["keep"])
    prune_bytes = sum(row["bytes"] for row in plan["prune"])
    lines = [
        f"Plan: {plan['plan_hash']}",
        "Scope: "
        + (
            ", ".join(plan["experiments"]) if plan["experiments"] else "all experiments"
        ),
        "Stages: " + (", ".join(plan.get("stages") or []) or "all stages"),
        "Policy: used report ancestry + manual .protected ancestry + incomplete inputs",
        f"Used: {sum(row['used'] for row in plan['keep'])} runs",
        f"Protected: {sum(row['protected'] for row in plan['keep'])} runs",
        f"Keep: {len(plan['keep'])} runs ({keep_bytes / 2**30:.2f} GiB)",
        f"Prune: {len(plan['prune'])} runs ({prune_bytes / 2**30:.2f} GiB)",
        "",
        "KEEP",
    ]
    lines.extend(
        f"{row['run_id']}\t{row['bytes']}\t{'used' if row['used'] else 'unused'}\t{','.join(row['reasons'])}"
        for row in plan["keep"]
    )
    lines.extend(["", "PRUNE"])
    lines.extend(
        f"{row['run_id']}\t{row['bytes']}\t{'used' if row['used'] else 'unused'}\t{','.join(row['reasons'])}"
        for row in plan["prune"]
    )
    scope_args = " ".join(
        f"--experiment {shlex.quote(experiment)}"
        for experiment in (plan["experiments"] or [])
    )
    command = "uv run pingstore prune"
    if scope_args:
        command += f" {scope_args}"
    for stage in plan.get("stages") or []:
        command += f" --stage {shlex.quote(stage)}"
    command += f" --confirm {plan['plan_hash']}"
    lines.extend(["", f"Confirm with: {command}"])
    return "\n".join(lines)
