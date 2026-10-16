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


def report_usage(repo: Path, records: dict[str, dict]) -> tuple[dict, dict[str, set[str]]]:
    """Use explicit pins, otherwise Astro's latest completed presentation per input."""
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
        for experiment, identity in selected.items():
            record = records.get(identity)
            if record is None:
                raise PingstoreError(f"report {article!r} names missing run {identity}")
            if record["experiment"] != experiment or record["stage"] != "present":
                raise PingstoreError(f"report {article!r} pin is not a presentation for {experiment}")
        selections[article] = {"inputs": metadata["inputs"], "runs": selected}
        for identity in ancestry(records, set(selected.values())):
            used_by[identity].add(article)
    return selections, dict(used_by)
