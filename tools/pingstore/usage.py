"""Current report selections and their recorded transitive input ancestry."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .articles import articles
from .contracts import PingstoreError


def ancestry(records: dict[str, dict], roots: set[str]) -> set[str]:
    seen: set[str] = set()
    pending = list(roots)
    while pending:
        identity = pending.pop()
        if identity in seen:
            continue
        if identity not in records:
            raise PingstoreError(f"missing report or protected input: {identity}")
        seen.add(identity)
        pending.extend(ref["run_id"] for ref in records[identity]["inputs"].values())
    return seen


def report_usage(
    repo: Path, records: dict[str, dict], *, experiments: set[str] | None = None,
) -> tuple[dict, dict[str, set[str]]]:
    """Resolve all reports, allowing missing out-of-scope pins for scoped pruning.

    Keep unavailable pins in the selections so confirmation still detects edits.
    Existing selections and their ancestry remain validated across experiments.
    """
    latest = {}
    for identity, record in records.items():
        completed = record["execution"].get("completed_at")
        if record["stage"] != "present" or not completed:
            continue
        key = (datetime.fromisoformat(completed.replace("Z", "+00:00")).timestamp(), identity)
        experiment = record["experiment"]
        if experiment not in latest or key > latest[experiment]:
            latest[experiment] = key
    selections = {}
    used_by = defaultdict(set)
    for article, metadata in articles(repo).items():
        selected = dict(metadata["runs"])
        for experiment in metadata["inputs"]:
            if experiment not in selected and experiment in latest:
                selected[experiment] = latest[experiment][1]
        roots = set()
        for experiment, identity in selected.items():
            record = records.get(identity)
            if record is None:
                # Check both the declared input and the run ID: a mismatched
                # pin must not hide a missing reference to the pruning scope.
                if (experiments is not None and experiment not in experiments
                        and identity.split("-", 1)[0] not in experiments):
                    continue
                raise PingstoreError(f"report {article!r} names missing run {identity}")
            if record["experiment"] != experiment or record["stage"] != "present":
                raise PingstoreError(f"report {article!r} pin is not a presentation for {experiment}")
            roots.add(identity)
        selections[article] = {"inputs": metadata["inputs"], "runs": selected}
        for identity in ancestry(records, roots):
            used_by[identity].add(article)
    return selections, dict(used_by)
