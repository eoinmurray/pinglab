from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tools.runstore.archive import archive_run, restore_archive, verify_archive
from tools.runstore.contract import ContractError, inventory_payload, write_json_atomic
from tools.runstore.lifecycle import initialize_run
from tools.runstore.storage import LocalStore

EXAMPLE = Path(__file__).parents[1] / "examples" / "minimal-run"


def copy_example(destination: Path) -> Path:
    shutil.copytree(EXAMPLE, destination)
    return destination


def test_local_archive_verify_and_restore_round_trip(tmp_path: Path) -> None:
    source = copy_example(tmp_path / "source")
    store = LocalStore(tmp_path / "store")

    archived = archive_run(source, "fixture-v1", store)
    assert archived["archive"]["archive_id"] == "fixture-v1"
    assert json.loads((source / "run.json").read_text()) == archived

    verified = verify_archive(store, "fixture-v1")
    assert verified["file_count"] == 1
    assert (
        verified["payload_digest"]
        == json.loads((source / "inventory.json").read_text())["payload_digest"]
    )

    destination = tmp_path / "restored"
    restored = restore_archive(store, "fixture-v1", destination)
    assert restored["run_id"] == archived["run_id"]
    assert (destination / "run.json").exists()
    assert (
        destination / "derived/artifacts/data/exp000/numbers.json"
    ).read_bytes() == (
        source / "derived/artifacts/data/exp000/numbers.json"
    ).read_bytes()


def test_graph_execution_identity_survives_archive_and_restore(tmp_path: Path) -> None:
    source = tmp_path / "source"
    graph_digest = "sha256:" + "a" * 64
    training_digest = "sha256:" + "b" * 64
    run = initialize_run(
        source,
        run_id="graph-fixture",
        kind="adhoc",
        experiment="exp123",
        collection=None,
        command=["tool.py", "train", "--executor", "graph"],
        executor="graph",
        graph_digest=graph_digest,
        training_digest=training_digest,
        repository=tmp_path,
    )
    run["status"] = "complete"
    write_json_atomic(source / "run.json", run)
    write_json_atomic(
        source / "inventory.json",
        inventory_payload(source, run_id=run["run_id"]),
    )
    store = LocalStore(tmp_path / "store")
    archived = archive_run(source, "graph-fixture", store)
    restored_root = tmp_path / "restored"
    restore_archive(store, "graph-fixture", restored_root)
    restored = json.loads((restored_root / "run.json").read_text())
    assert (
        archived["execution"]
        == restored["execution"]
        == {
            "experiment": "exp123",
            "collection": None,
            "command": ["tool.py", "train", "--executor", "graph"],
            "executor": "graph",
            "graph_digest": graph_digest,
            "training_digest": training_digest,
        }
    )


def test_archive_refuses_existing_identity(tmp_path: Path) -> None:
    store = LocalStore(tmp_path / "store")
    archive_run(copy_example(tmp_path / "first"), "same-id", store)
    with pytest.raises(ContractError, match="already exists"):
        archive_run(copy_example(tmp_path / "second"), "same-id", store)


def test_archive_refuses_source_with_recorded_archive(tmp_path: Path) -> None:
    source = copy_example(tmp_path / "source")
    store = LocalStore(tmp_path / "store")
    archive_run(source, "first-id", store)
    with pytest.raises(ContractError, match="already records"):
        archive_run(source, "second-id", store)


def test_verify_detects_remote_corruption(tmp_path: Path) -> None:
    source = copy_example(tmp_path / "source")
    store_root = tmp_path / "store"
    store = LocalStore(store_root)
    archive_run(source, "corrupt-me", store)
    payload = store_root / "corrupt-me" / "derived/artifacts/data/exp000/numbers.json"
    payload.write_bytes(b"corrupt but deliberately different")
    with pytest.raises(ContractError, match="mismatch"):
        verify_archive(store, "corrupt-me")


def test_verify_detects_unexpected_remote_object(tmp_path: Path) -> None:
    source = copy_example(tmp_path / "source")
    store_root = tmp_path / "store"
    store = LocalStore(store_root)
    archive_run(source, "extra-object", store)
    (store_root / "extra-object" / "unexpected.txt").write_text("unexpected")
    with pytest.raises(ContractError, match="object set differs"):
        verify_archive(store, "extra-object")


def test_restore_refuses_existing_destination(tmp_path: Path) -> None:
    source = copy_example(tmp_path / "source")
    store = LocalStore(tmp_path / "store")
    archive_run(source, "fixture-v1", store)
    destination = tmp_path / "already-here"
    destination.mkdir()
    with pytest.raises(ContractError, match="already exists"):
        restore_archive(store, "fixture-v1", destination)
