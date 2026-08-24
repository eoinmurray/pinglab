from __future__ import annotations

from pathlib import Path

import pytest
from pingstore.catalogue import Catalogue
from pingstore.contracts import EXPERIMENT_RUN_SCHEMA, PingstoreError


def make_run(location: Path, run_id: str = "exp001/r001") -> dict:
    return {
        "schema": EXPERIMENT_RUN_SCHEMA,
        "run_id": run_id,
        "collection": "demo",
        "experiment": "exp001",
        "status": "finalized",
        "disposition": "candidate",
        "source": {},
        "execution": {"command": [], "host": "local"},
        "upstream_runs": [],
        "upstream_datasets": [],
        "payload": {
            "location": str(location),
            "inventory_digest": "sha256:" + "b" * 64,
        },
        "archive": None,
        "legacy_identity": None,
    }


def test_register_select_and_freeze(tmp_path: Path) -> None:
    catalogue = Catalogue(tmp_path / "store")
    catalogue.create_dataset("demo", ["exp001"])
    run = make_run(tmp_path / "payload")
    run_root = catalogue.run_path("demo", "exp001", run["run_id"])
    run_root.mkdir(parents=True)
    from pingstore.contracts import write_json_atomic

    write_json_atomic(run_root / "run.json", run)
    catalogue.register_run(run)
    catalogue.select("exp001", run["run_id"])
    frozen = catalogue.freeze("demo", "release-1")
    assert frozen["status"] == "frozen"
    assert frozen["digest"].startswith("sha256:")
    assert frozen["official_runs"] == {"exp001": "exp001/r001"}


def test_freeze_requires_complete_official_mapping(tmp_path: Path) -> None:
    catalogue = Catalogue(tmp_path / "store")
    catalogue.create_dataset("demo", ["exp001"])
    with pytest.raises(PingstoreError, match="official runs"):
        catalogue.freeze("demo", "release-1")
