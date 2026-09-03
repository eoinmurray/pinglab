"""Explicit, hash-bound pruning of superseded operational runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

from .contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    validate_collections,
    validate_operational_run_directory,
)
from .discovery import discover_runs
from .locking import operation_lock

PLAN_SCHEMA = "pingstore.prune-plan/v1"
HPC_MARKER = re.compile(r"(?:^|[-_.])(?:slurm|hpc|wilkes|csd3|gpu-q)(?:$|[-_.0-9])")
PROVENANCE_KEYS = {
    "host",
    "host_record",
    "origin",
    "producer_host",
    "producer_origin",
    "scheduler",
}


def _visible_runs(runs: Path) -> list[Path]:
    return [
        path
        for path in sorted(runs.iterdir())
        if path.is_dir() and not path.name.startswith(".") and not path.is_symlink()
    ]


def _validated_graph(runs: Path) -> dict[str, dict]:
    records = {
        directory.name: validate_operational_run_directory(directory)
        for directory in _visible_runs(runs)
    }
    for child, record in records.items():
        for reference in record["inputs"].values():
            parent = records.get(reference["run_id"])
            if (
                parent is None
                or parent["payload_digest"] != reference["payload_digest"]
            ):
                raise PingstoreError(
                    f"{child}: missing or changed input {reference['run_id']}"
                )
    return records


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


def build_plan(repo: Path) -> dict:
    repo = repo.resolve()
    runs = repo / ".pingstore/runs"
    if not runs.is_dir() or runs.is_symlink():
        raise PingstoreError(f"prune requires a real runs directory: {runs}")
    for path in runs.iterdir():
        if path.is_symlink() or not path.is_dir():
            raise PingstoreError(
                f"prune does not accept unsupported runs entry: {path}"
            )
    records = _validated_graph(runs)
    discovered = discover_runs(runs)
    reasons: dict[str, set[str]] = defaultdict(set)

    latest: dict[str, dict] = {}
    for row in discovered:
        current = latest.get(row["experiment"])
        if current is None or (row["created_at"], row["id"]) > (
            current["created_at"],
            current["id"],
        ):
            latest[row["experiment"]] = row
    for row in latest.values():
        reasons[row["id"]].add("latest-visible")
    for run_id, record in records.items():
        if is_hpc_run(record):
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
    for run_ids in by_experiment.values():
        high = max(run_ids, key=lambda run_id: (_counter(run_id), run_id))
        if high not in keep:
            keep.add(high)
            reasons[high].add("identity-high-watermark")

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
                "reasons": sorted(reasons[run_id])
                if run_id in keep
                else ["superseded"],
            }
        )
    plan = {
        "schema": PLAN_SCHEMA,
        "policy": "keep-hpc-and-latest-visible-with-ancestry",
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
    records = _validated_graph(runs)
    if set(records) != expected:
        raise PingstoreError("pruned store contains an unexpected completed-run set")
    discover_runs(runs)


def apply_plan(repo: Path, expected_hash: str) -> dict:
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
            plan = build_plan(repo)
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
    lines.extend(
        ["", f"Confirm with: uv run pingstore prune --confirm {plan['plan_hash']}"]
    )
    return "\n".join(lines)
