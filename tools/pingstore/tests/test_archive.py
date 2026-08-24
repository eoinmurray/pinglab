from __future__ import annotations

from pathlib import Path

from pingstore.archive import archive_dataset, restore_dataset
from pingstore.catalogue import Catalogue
from pingstore.contracts import EXPERIMENT_RUN_SCHEMA, write_json_atomic
from pingstore.materialize import materialize_shadow


def test_frozen_dataset_round_trips_and_materializes(tmp_path: Path) -> None:
    source = Catalogue(tmp_path / "source")
    source.create_dataset("demo", ["exp001"])
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "result.txt").write_text("evidence")
    run_id = "exp001/r001"
    run = {
        "schema": EXPERIMENT_RUN_SCHEMA,
        "run_id": run_id,
        "collection": "demo",
        "experiment": "exp001",
        "status": "finalized",
        "disposition": "retained",
        "source": {},
        "execution": {"command": [], "host": "local"},
        "upstream_runs": [],
        "upstream_datasets": [],
        "payload": {"location": str(payload), "inventory_digest": "sha256:" + "a" * 64},
        "archive": None,
        "legacy_identity": None,
    }
    run_root = source.run_path("demo", "exp001", run_id)
    run_root.mkdir(parents=True)
    write_json_atomic(run_root / "run.json", run)
    source.register_run(run)
    source.select("exp001", run_id)
    source.freeze("demo", "gold")

    bundle = tmp_path / "bundle"
    archive_dataset(source, "demo/gold", bundle)
    restored_root = tmp_path / "restored"
    restore_dataset(bundle, restored_root)

    restored = Catalogue(restored_root)
    working = restored.create_dataset("demo", ["exp001"])
    working["runs"]["exp001"] = [run_id]
    working["official_runs"]["exp001"] = run_id
    restored.save_dataset(working)
    view = tmp_path / "view"
    materialize_shadow(restored, "demo", view)
    assert (view / "exp001/result.txt").read_text() == "evidence"
