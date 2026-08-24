"""Idempotent shadow migration from legacy artifacts and Runstore archives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from runstore.contract import inventory_payload
from runstore.contract import load_json as load_legacy_json

from .catalogue import Catalogue
from .contracts import (
    EXPERIMENT_RUN_SCHEMA,
    PingstoreError,
    canonical_digest,
    write_json_atomic,
)


def classify(inventory: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for payload in inventory["payloads"]:
        classification = "legacy-unverified"
        blocker = None
        if payload["kind"] == "r2-archive":
            classification = "verified-archived"
        elif payload["kind"] == "restored-archive" and payload.get("inventory"):
            classification = "verified-archived"
        elif payload["kind"] == "artifact-directory" and payload.get("provenance"):
            classification = "verified-local"
        elif payload["kind"] == "artifact-directory" and payload.get("manifest"):
            classification = "candidate-local"
        elif payload["kind"] == "scratch-directory":
            classification = "temporary-scratch"
        if payload.get("experiment") and not payload.get("collection"):
            blocker = "unresolved collection membership"
        rows.append(
            {
                "physical_id": payload["physical_id"],
                "classification": classification,
                "blocker": blocker,
            }
        )
    return {
        "schema": "pingstore.migration-classification/v1",
        "inventory_digest": inventory["digest"],
        "rows": rows,
        "blocked": sum(row["blocker"] is not None for row in rows),
        "digest": canonical_digest(rows),
    }


def build_plan(inventory: dict[str, Any], classifications: dict[str, Any]) -> dict[str, Any]:
    by_id = {row["physical_id"]: row for row in classifications["rows"]}
    operations: list[dict[str, Any]] = []
    for payload in inventory["payloads"]:
        row = by_id[payload["physical_id"]]
        operations.append(
            {
                "physical_id": payload["physical_id"],
                "action": "block" if row["blocker"] else "import-reference",
                "classification": row["classification"],
                "collection": payload.get("collection"),
                "experiment": payload.get("experiment"),
                "path": payload["path"],
                "blocker": row["blocker"],
            }
        )
    return {
        "schema": "pingstore.migration-plan/v1",
        "inventory_digest": inventory["digest"],
        "classification_digest": classifications["digest"],
        "operations": operations,
        "blocked": sum(op["action"] == "block" for op in operations),
        "digest": canonical_digest(operations),
    }


def _manifest_run(operation: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(Path(payload["manifest"]).read_text())
    experiment = operation["experiment"]
    local_id = manifest.get("run_id", "legacy")
    return {
        "schema": EXPERIMENT_RUN_SCHEMA,
        "run_id": f"{experiment}/{local_id}",
        "collection": operation["collection"],
        "experiment": experiment,
        "status": "finalized",
        "disposition": "candidate",
        "source": {
            "git_commit": manifest.get("git_sha"),
            "dirty": manifest.get("dirty"),
            "code_dirty": manifest.get("code_dirty"),
            "dirty_patch": manifest.get("patch"),
        },
        "execution": {
            "host": manifest.get("host", "local"),
            "command": [],
            "started_at": manifest.get("run_at"),
            "completed_at": None,
        },
        "upstream_runs": [],
        "upstream_datasets": [],
        "payload": {
            "location": operation["path"],
            "inventory_digest": "sha256:" + "0" * 64,
        },
        "archive": None,
        "legacy_identity": {"manifest": payload["manifest"]},
    }


def _referenced_run(
    operation: dict[str, Any],
    payload: dict[str, Any],
    *,
    run_id: str,
    experiment: str,
    source: Path,
    disposition: str,
) -> dict[str, Any]:
    actual = inventory_payload(source, run_id=run_id)
    return {
        "schema": EXPERIMENT_RUN_SCHEMA,
        "run_id": run_id,
        "collection": operation["collection"],
        "experiment": experiment,
        "status": "finalized",
        "disposition": disposition,
        "source": {},
        "execution": {
            "host": "legacy",
            "command": [],
            "campaign_id": payload.get("legacy_run_id"),
        },
        "upstream_runs": [],
        "upstream_datasets": [],
        "payload": {
            "location": str(source),
            "inventory_digest": "sha256:" + actual["payload_digest"],
        },
        "archive": payload.get("archive"),
        "legacy_identity": {
            "physical_id": payload["physical_id"],
            "legacy_run_id": payload.get("legacy_run_id"),
        },
    }


def import_shadow(
    inventory: dict[str, Any],
    plan: dict[str, Any],
    *,
    catalogue: Catalogue,
    migration_root: Path,
) -> dict[str, Any]:
    payloads = {payload["physical_id"]: payload for payload in inventory["payloads"]}
    experiments: dict[str, set[str]] = {}
    for operation in plan["operations"]:
        if operation["experiment"] and operation["collection"]:
            experiments.setdefault(operation["collection"], set()).add(
                operation["experiment"]
            )
        payload = payloads[operation["physical_id"]]
        if operation["collection"]:
            experiments.setdefault(operation["collection"], set()).update(
                payload.get("experiments", [])
            )
    for collection, members in experiments.items():
        path = catalogue.dataset_path(collection)
        if not path.exists():
            catalogue.create_dataset(
                collection,
                sorted(members),
                migration={"plan_digest": plan["digest"]},
            )

    imported: list[str] = []
    proposals: dict[str, dict[str, str]] = {}
    for operation in plan["operations"]:
        if operation["action"] == "block":
            continue
        payload = payloads[operation["physical_id"]]
        runs_to_import: list[dict[str, Any]] = []
        if operation["classification"] == "candidate-local" and payload.get(
            "manifest"
        ):
            runs_to_import.append(_manifest_run(operation, payload))
        elif operation["classification"] == "verified-local" and payload.get(
            "provenance"
        ):
            provenance = json.loads(Path(payload["provenance"]).read_text())
            experiment = operation["experiment"]
            legacy_id = provenance.get("run_id", "legacy")
            runs_to_import.append(
                _referenced_run(
                    operation,
                    payload,
                    run_id=f"{legacy_id}/{experiment}",
                    experiment=experiment,
                    source=Path(operation["path"]),
                    disposition="retained",
                )
            )
        elif operation["classification"] == "verified-archived" and payload.get(
            "experiments"
        ):
            for experiment in payload["experiments"]:
                run_id = f"{payload['legacy_run_id']}/{experiment}"
                runs_to_import.append(
                    _referenced_run(
                        operation,
                        payload,
                        run_id=run_id,
                        experiment=experiment,
                        source=Path(operation["path"])
                        / "derived/artifacts/data"
                        / experiment,
                        disposition="retained",
                    )
                )
                if payload.get("legacy_run_id") == "gold-2":
                    proposals.setdefault(operation["collection"], {})[
                        experiment
                    ] = run_id
        for run in runs_to_import:
            root = catalogue.run_path(
                run["collection"], run["experiment"], run["run_id"]
            )
            run_file = root / "run.json"
            if run_file.exists():
                existing = json.loads(run_file.read_text())
                if existing != run:
                    raise PingstoreError(f"imported run drift: {run['run_id']}")
            else:
                write_json_atomic(run_file, run)
            catalogue.register_run(run)
            imported.append(run["run_id"])

    for collection, proposal in proposals.items():
        dataset = catalogue.load_dataset(collection)
        dataset["selection_proposal"] = proposal
        catalogue.save_dataset(dataset)

    result = {
        "schema": "pingstore.migration-import/v1",
        "plan_digest": plan["digest"],
        "imported_runs": sorted(imported),
        "selection_proposals": proposals,
    }
    write_json_atomic(migration_root / "import.json", result)
    return result


def load_migration_documents(root: Path) -> tuple[dict, dict, dict]:
    return (
        load_legacy_json(root / "inventory.json"),
        load_legacy_json(root / "classifications.json"),
        load_legacy_json(root / "import-plan.json"),
    )
