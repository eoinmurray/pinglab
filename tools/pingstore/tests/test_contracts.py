from __future__ import annotations

import copy

import pytest
from pingstore.contracts import (
    COLLECTION_DATASET_SCHEMA,
    EXPERIMENT_RUN_SCHEMA,
    PingstoreError,
    validate_collection_dataset,
    validate_experiment_run,
)


def dataset() -> dict:
    return {
        "schema": COLLECTION_DATASET_SCHEMA,
        "dataset_id": "demo/working",
        "collection": "demo",
        "status": "working",
        "experiments": ["exp001"],
        "runs": {"exp001": ["exp001/r001"]},
        "official_runs": {},
        "preview_overrides": {"exp001": "exp001/r001"},
        "collection_assets": [],
        "upstream_datasets": [],
        "migration": None,
        "digest": None,
    }


def run() -> dict:
    return {
        "schema": EXPERIMENT_RUN_SCHEMA,
        "run_id": "exp001/r001",
        "collection": "demo",
        "experiment": "exp001",
        "status": "finalized",
        "disposition": "candidate",
        "source": {},
        "execution": {"command": [], "host": "local"},
        "upstream_runs": [],
        "upstream_datasets": [],
        "payload": {
            "location": "/tmp/result",
            "inventory_digest": "sha256:" + "a" * 64,
        },
        "archive": None,
        "legacy_identity": None,
    }


def test_valid_contracts() -> None:
    assert validate_collection_dataset(dataset())
    assert validate_experiment_run(run())


def test_official_pointer_must_select_retained_run() -> None:
    value = dataset()
    value["official_runs"]["exp001"] = "exp001/missing"
    with pytest.raises(PingstoreError, match="must select a retained run"):
        validate_collection_dataset(value)


def test_official_is_not_a_run_disposition() -> None:
    value = run()
    value["disposition"] = "official"
    with pytest.raises(PingstoreError, match="disposition"):
        validate_experiment_run(value)


def test_finalized_run_requires_digest() -> None:
    value = copy.deepcopy(run())
    value["payload"]["inventory_digest"] = None
    with pytest.raises(PingstoreError, match="inventory digest"):
        validate_experiment_run(value)
