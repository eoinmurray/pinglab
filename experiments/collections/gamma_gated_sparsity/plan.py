"""Resolve collection execution paths into a cold-readable plan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .graph import COLLECTION, PENDING_ROOT_DECISIONS, ordered_experiments

REPO = Path(__file__).resolve().parents[3]


def validate_campaign_root(root: Path) -> Path:
    if not root.is_absolute():
        raise ValueError("campaign root must be an explicit absolute path")
    resolved = root.resolve()
    forbidden = (REPO, REPO / "artifacts", REPO / "temp")
    for base in forbidden:
        base = base.resolve()
        if resolved == base or base in resolved.parents:
            raise ValueError("campaign root must be external to the repository")
    return resolved


def build_plan(root: Path, campaign_id: str) -> dict[str, Any]:
    resolved = validate_campaign_root(root)
    stages: dict[int, list[dict[str, object]]] = {}
    depths: dict[str, int] = {}
    for experiment in ordered_experiments():
        depth = 0
        if experiment.dependencies:
            depth = max(depths[dependency] for dependency in experiment.dependencies) + 1
        depths[experiment.slug] = depth
        state = (
            resolved / "exp022"
            if experiment.slug == "exp022"
            else resolved / "downstream" / experiment.slug
        )
        stages.setdefault(depth, []).append(
            {
                **experiment.as_dict(),
                "paths": {
                    "state": str(state),
                    "derived": str(
                        resolved / "derived" / "artifacts" / "data" / experiment.slug
                    ),
                    "logs": str(resolved / "logs" / experiment.slug),
                },
            }
        )
    return {
        "collection": COLLECTION,
        "campaign_id": campaign_id,
        "campaign_root": str(resolved),
        "executable": all(
            row["integrated"] or row["slug"] == "exp022"
            for stage in stages.values()
            for row in stage
        ),
        "stages": [
            {"index": index, "experiments": stages[index]} for index in sorted(stages)
        ],
        "pending_root_decisions": list(PENDING_ROOT_DECISIONS),
        "excluded": ["exp048"],
        "blocking_issues": [69, 47],
    }
