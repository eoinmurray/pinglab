from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.runstore.cli import inspect
from tools.runstore.contract import (
    ContractError,
    inventory_payload,
    load_json,
    validate_inventory,
    validate_run_manifest,
    verify_payload,
)

EXAMPLE = Path(__file__).parents[1] / "examples" / "minimal-run"


def test_committed_example_is_valid() -> None:
    run = validate_run_manifest(load_json(EXAMPLE / "run.json"))
    inventory = validate_inventory(load_json(EXAMPLE / "inventory.json"))
    assert run["run_id"] == inventory["run_id"]
    verify_payload(EXAMPLE, inventory)


def test_inventory_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "state").mkdir()
    (tmp_path / "derived").mkdir()
    (tmp_path / "derived" / "b.json").write_text("{}\n")
    (tmp_path / "state" / "a.bin").write_bytes(b"abc")
    first = inventory_payload(
        tmp_path, run_id="fixture", generated_at_utc="2026-08-11T00:00:00Z"
    )
    second = inventory_payload(
        tmp_path, run_id="fixture", generated_at_utc="2026-08-11T00:00:00Z"
    )
    assert first == second
    assert [row["path"] for row in first["files"]] == [
        "derived/b.json",
        "state/a.bin",
    ]


def test_verify_detects_modified_payload(tmp_path: Path) -> None:
    payload = tmp_path / "state.bin"
    payload.write_bytes(b"before")
    inventory = inventory_payload(tmp_path, run_id="fixture")
    payload.write_bytes(b"after")
    with pytest.raises(ContractError, match="payload does not match"):
        verify_payload(tmp_path, inventory)


@pytest.mark.parametrize("unsafe", ["/absolute", "../escape", "a/../escape", r"a\\b"])
def test_inventory_rejects_unsafe_paths(unsafe: str) -> None:
    inventory = {
        "contract_version": "runstore/v1",
        "run_id": "fixture",
        "generated_at_utc": "2026-08-11T00:00:00Z",
        "file_count": 1,
        "total_size_bytes": 0,
        "payload_digest": "0" * 64,
        "files": [
            {"path": unsafe, "size_bytes": 0, "sha256": "0" * 64, "role": "state"}
        ],
    }
    with pytest.raises(ContractError):
        validate_inventory(inventory)


def test_inspect_is_read_only_for_unmanaged_legacy(tmp_path: Path, capsys) -> None:
    (tmp_path / "legacy.bin").write_bytes(b"legacy")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert inspect(tmp_path) == 0
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert before == after
    output = capsys.readouterr().out
    assert "unmanaged-legacy" in output
    assert "missing run.json" in output


def test_inspect_rejects_stale_inventory(tmp_path: Path) -> None:
    run = json.loads((EXAMPLE / "run.json").read_text())
    (tmp_path / "run.json").write_text(json.dumps(run))
    (tmp_path / "payload.bin").write_bytes(b"current")
    inventory = inventory_payload(tmp_path, run_id=run["run_id"])
    (tmp_path / "inventory.json").write_text(json.dumps(inventory))
    (tmp_path / "payload.bin").write_bytes(b"changed")
    with pytest.raises(ContractError, match="payload does not match"):
        inspect(tmp_path)
