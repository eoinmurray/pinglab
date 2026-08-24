from __future__ import annotations

import json
from pathlib import Path

from pingstore.catalogue import Catalogue
from pingstore.inventory import inventory_local, verify_local_inventory
from pingstore.migration import build_plan, classify, import_shadow


def make_repo(root: Path) -> None:
    (root / "writings").mkdir(parents=True)
    (root / "writings/exp001.typ").write_text('collection: "demo",\n')
    artifact = root / "artifacts/data/exp001"
    artifact.mkdir(parents=True)
    (artifact / "numbers.json").write_text("{}\n")
    (artifact / "_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "r001",
                "run_at": "2026-08-24T00:00:00Z",
                "git_sha": "abc123",
                "dirty": False,
                "code_dirty": False,
                "patch": None,
                "host": "local",
            }
        )
    )


def test_inventory_plan_and_shadow_import_are_idempotent(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    make_repo(repo)
    inventory = inventory_local(repo)
    classifications = classify(inventory)
    plan = build_plan(inventory, classifications)
    assert plan["blocked"] == 0
    catalogue = Catalogue(tmp_path / "pingstore")
    migration_root = catalogue.root / "migrations/m1"
    first = import_shadow(
        inventory, plan, catalogue=catalogue, migration_root=migration_root
    )
    second = import_shadow(
        inventory, plan, catalogue=catalogue, migration_root=migration_root
    )
    assert first == second
    assert catalogue.load_dataset("demo")["runs"] == {"exp001": ["exp001/r001"]}


def test_unresolved_membership_is_blocked_without_preventing_other_imports(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    make_repo(repo)
    unknown = repo / "artifacts/data/exp999"
    unknown.mkdir()
    (unknown / "numbers.json").write_text("{}")
    inventory = inventory_local(repo)
    plan = build_plan(inventory, classify(inventory))
    assert plan["blocked"] == 1
    result = import_shadow(
        inventory,
        plan,
        catalogue=Catalogue(tmp_path / "pingstore"),
        migration_root=tmp_path / "migration",
    )
    assert result["imported_runs"] == ["exp001/r001"]


def test_local_verification_defers_remote_payloads() -> None:
    inventory = {
        "payloads": [
            {
                "physical_id": "r2-archive:gold-1",
                "kind": "r2-archive",
                "path": "r2://pinglab/campaigns/gold-1",
            }
        ]
    }
    result = verify_local_inventory(inventory)
    assert result["passed"] is True
    assert result["results"][0]["state"] == "remote-deferred"
