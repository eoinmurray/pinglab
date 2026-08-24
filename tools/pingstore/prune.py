"""Non-destructive pruning plans."""

from __future__ import annotations

from typing import Any

from .catalogue import Catalogue


def pruning_plan(catalogue: Catalogue, collection: str) -> dict[str, Any]:
    dataset = catalogue.load_dataset(collection)
    protected = set(dataset["official_runs"].values())
    protected.update(dataset["preview_overrides"].values())
    rows: list[dict[str, str]] = []
    for experiment, run_ids in sorted(dataset["runs"].items()):
        for run_id in run_ids:
            run_file = catalogue.run_path(collection, experiment, run_id) / "run.json"
            disposition = "unknown"
            if run_file.is_file():
                import json

                disposition = json.loads(run_file.read_text()).get(
                    "disposition", "unknown"
                )
            action = "retain"
            reason = "selected or retained"
            if run_id not in protected and disposition == "temporary":
                action = "eligible"
                reason = "unselected temporary run"
            rows.append(
                {
                    "experiment": experiment,
                    "run_id": run_id,
                    "action": action,
                    "reason": reason,
                }
            )
    return {
        "schema": "pingstore.pruning-plan/v1",
        "collection": collection,
        "destructive": False,
        "rows": rows,
    }
