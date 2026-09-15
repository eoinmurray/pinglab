"""Explicit, hash-bound pruning of superseded operational runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
from collections import defaultdict
from pathlib import Path

from .contracts import (
    EXPERIMENT_RE,
    PingstoreError,
    file_sha256,
    load_json,
    validate_collections,
)
from .discovery import discover_store
from .stages import operation_lock

PLAN_SCHEMA = "pingstore.prune-plan/v3"
HPC_MARKER = re.compile(r"(?:^|[-_.])(?:slurm|hpc|wilkes|csd3|gpu-q)(?:$|[-_.0-9])")
PROVENANCE_KEYS = {
    "host",
    "host_record",
    "origin",
    "producer_host",
    "producer_origin",
    "scheduler",
}


def _authoritative_provenance_values(value, key: str = ""):
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _authoritative_provenance_values(child, child_key)
    elif isinstance(value, list):
        for child in value:
            yield from _authoritative_provenance_values(child, key)
    elif key in PROVENANCE_KEYS and isinstance(value, str):
        yield value


def is_hpc_run(record: dict) -> bool:
    """Recognize recorded HPC execution without treating paths or run names as evidence."""
    provenance = {
        "origin": record.get("origin"),
        "execution": record.get("execution"),
        "scientific_execution": record.get("scientific_execution"),
        "historical_import": record.get("historical_import"),
    }
    return any(
        HPC_MARKER.search(value.lower())
        for value in _authoritative_provenance_values(provenance)
    )


def _directory_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _add_declared_roots(
    repo: Path, records: dict[str, dict], reasons: dict[str, set[str]]
) -> None:
    collections = repo / ".pingstore/collections.json"
    if collections.exists():
        for view, run_ids in validate_collections(load_json(collections)).items():
            for run_id in run_ids:
                if run_id not in records:
                    raise PingstoreError(
                        f"collection {view!r} names missing run {run_id}"
                    )
                reasons[run_id].add(f"named-view:{view}")
    defaults = repo / "writings/run-defaults.json"
    if defaults.exists():
        value = load_json(defaults)
        for article, pins in value.items():
            if not isinstance(pins, dict):
                raise PingstoreError(f"invalid writing defaults for {article}")
            for run_id in pins.values():
                if run_id not in records:
                    raise PingstoreError(
                        f"writing default {article!r} names missing run {run_id}"
                    )
                reasons[run_id].add(f"writing-default:{article}")


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


def _counter(run_id: str) -> int:
    match = re.search(r"-r([0-9]+)-(?:compute|analyse|present)(?:-|$)", run_id)
    if match is None:
        raise PingstoreError(f"cannot read run counter: {run_id}")
    return int(match.group(1))


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


def _retirement_scope(
    repo: Path,
    experiments: list[str] | tuple[str, ...] | set[str] | None,
    records: dict[str, dict],
) -> set[str]:
    retired = set(experiments or ())
    invalid = sorted(value for value in retired if not EXPERIMENT_RE.fullmatch(value))
    if invalid:
        raise PingstoreError(f"invalid retirement experiment: {', '.join(invalid)}")
    available = {record["experiment"] for record in records.values()}
    missing = sorted(retired - available)
    if missing:
        raise PingstoreError(f"retirement experiment has no runs: {', '.join(missing)}")
    for experiment in sorted(retired):
        writing = repo / "writings" / f"{experiment}.typ"
        legacy = repo / "experiments" / f"{experiment}.py"
        staged = repo / "experiments" / experiment
        if writing.is_file():
            raise PingstoreError(
                f"cannot retire {experiment} while its writing exists: {writing}"
            )
        if legacy.is_file() or (
            staged.is_dir()
            and any(
                (staged / f"{stage}.py").is_file()
                for stage in ("compute", "analyse", "present")
            )
        ):
            raise PingstoreError(
                f"cannot retire {experiment} while runnable experiment code exists"
            )
    return retired


def build_plan(
    repo: Path,
    experiments: list[str] | tuple[str, ...] | set[str] | None = None,
    retire_experiments: list[str] | tuple[str, ...] | set[str] | None = None,
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
    records, discovered = discover_store(runs)
    retired = _retirement_scope(repo, retire_experiments, records)
    scope = _experiment_scope(experiments, records)
    if retired and scope is None:
        scope = retired
    elif not retired.issubset(scope or set()):
        raise PingstoreError("retirement experiments must be inside the prune scope")
    reasons: dict[str, set[str]] = defaultdict(set)

    if scope is not None:
        for run_id, record in records.items():
            if record["experiment"] not in scope:
                reasons[run_id].add("out-of-scope")

    latest: dict[str, dict] = {}
    for row in discovered:
        if scope is not None and row["experiment"] not in scope:
            continue
        if row["experiment"] in retired:
            continue
        current = latest.get(row["experiment"])
        if current is None or (row["created_at"], row["id"]) > (
            current["created_at"],
            current["id"],
        ):
            latest[row["experiment"]] = row
    for row in latest.values():
        reasons[row["id"]].add("latest-visible")
    for run_id, record in records.items():
        if (scope is None or record["experiment"] in scope) and is_hpc_run(record):
            reasons[run_id].add("hpc")
    _add_declared_roots(repo, records, reasons)
    hidden = _hidden_inputs(runs, records, reasons)

    keep = set(reasons)
    todo = list(keep)
    while todo:
        child = todo.pop()
        for reference in records[child]["inputs"].values():
            parent = reference["run_id"]
            if parent not in keep:
                keep.add(parent)
                reasons[parent].add("required-ancestor")
                todo.append(parent)

    # The allocator derives its next counter from directories. Never permit a
    # deleted identity to become reusable, even for an experiment with no UI run.
    by_experiment: dict[str, list[str]] = defaultdict(list)
    for run_id, record in records.items():
        by_experiment[record["experiment"]].append(run_id)
    for experiment, run_ids in by_experiment.items():
        if experiment in retired:
            continue
        high = max(run_ids, key=lambda run_id: (_counter(run_id), run_id))
        if high not in keep:
            keep.add(high)
            reasons[high].add("identity-high-watermark")

    retirement_high_watermarks = {
        experiment: max(
            _counter(run_id)
            for run_id, record in records.items()
            if record["experiment"] == experiment
        )
        for experiment in sorted(retired)
    }

    rows = []
    for run_id, record in sorted(records.items()):
        directory = runs / run_id
        rows.append(
            {
                "run_id": run_id,
                "stage": record["stage"],
                "experiment": record["experiment"],
                "payload_digest": record["payload_digest"],
                "run_json_sha256": file_sha256(directory / "run.json"),
                "bytes": _directory_bytes(directory),
                "reasons": (
                    sorted(reasons[run_id])
                    if run_id in keep
                    else [
                        "retired-experiment"
                        if record["experiment"] in retired
                        else "superseded"
                    ]
                ),
            }
        )
    plan = {
        "schema": PLAN_SCHEMA,
        "policy": "keep-hpc-latest-pins-high-watermarks-and-ancestry-with-explicit-retirement",
        "experiments": sorted(scope) if scope is not None else None,
        "retire_experiments": sorted(retired),
        "retirement_high_watermarks": retirement_high_watermarks,
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


def apply_plan(
    repo: Path,
    expected_hash: str,
    experiments: list[str] | tuple[str, ...] | set[str] | None = None,
    retire_experiments: list[str] | tuple[str, ...] | set[str] | None = None,
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
            plan = build_plan(repo, experiments, retire_experiments)
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
            os.replace(runs, previous)
            try:
                os.replace(staged, runs)
                _validate_survivors(runs, keep_ids)
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
        "Retire: "
        + (
            ", ".join(plan["retire_experiments"])
            if plan["retire_experiments"]
            else "none"
        ),
        "Retirement high-watermarks: "
        + (
            ", ".join(
                f"{experiment}=r{counter:03d}"
                for experiment, counter in plan["retirement_high_watermarks"].items()
            )
            if plan["retirement_high_watermarks"]
            else "none"
        ),
        f"Keep: {len(plan['keep'])} runs ({keep_bytes / 2**30:.2f} GiB)",
        f"Prune: {len(plan['prune'])} runs ({prune_bytes / 2**30:.2f} GiB)",
        "",
        "KEEP",
    ]
    lines.extend(
        f"{row['run_id']}\t{row['bytes']}\t{','.join(row['reasons'])}"
        for row in plan["keep"]
    )
    lines.extend(["", "PRUNE"])
    lines.extend(
        f"{row['run_id']}\t{row['bytes']}\t{','.join(row['reasons'])}"
        for row in plan["prune"]
    )
    scope_args = " ".join(
        f"--experiment {shlex.quote(experiment)}"
        for experiment in (plan["experiments"] or [])
    )
    retirement_args = " ".join(
        f"--retire-experiment {shlex.quote(experiment)}"
        for experiment in plan["retire_experiments"]
    )
    command = "uv run pingstore prune"
    if scope_args:
        command += f" {scope_args}"
    if retirement_args:
        command += f" {retirement_args}"
    command += f" --confirm {plan['plan_hash']}"
    lines.extend(["", f"Confirm with: {command}"])
    return "\n".join(lines)
