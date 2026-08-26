from __future__ import annotations

from pathlib import Path

import pytest
from pingstore.catalogue import Catalogue
from pingstore.contracts import EXPERIMENT_RUN_SCHEMA, PingstoreError, write_json_atomic
from pingstore.materialize import (
    cutover,
    materialize_publication_view,
    materialize_shadow,
)


def test_shadow_materialization_uses_preview_override(tmp_path: Path) -> None:
    catalogue = Catalogue(tmp_path / "store")
    catalogue.create_dataset("demo", ["exp001"])
    for suffix in ("official", "preview"):
        payload = tmp_path / suffix
        payload.mkdir()
        (payload / "numbers.json").write_text(suffix)
        run_id = f"exp001/{suffix}"
        run = {
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
                "location": str(payload),
                "inventory_digest": "sha256:" + "c" * 64,
            },
            "archive": None,
            "legacy_identity": None,
        }
        root = catalogue.run_path("demo", "exp001", run_id)
        root.mkdir(parents=True)
        write_json_atomic(root / "run.json", run)
        catalogue.register_run(run)
    catalogue.select("exp001", "exp001/official")
    catalogue.select("exp001", "exp001/preview", preview=True)
    destination = tmp_path / "shadow"
    view = materialize_shadow(catalogue, "demo", destination)
    assert (destination / "exp001/numbers.json").read_text() == "preview"
    assert view["selections"] == {"exp001": "exp001/preview"}


def test_cutover_is_authority_gated() -> None:
    with pytest.raises(PingstoreError, match="separately reviewed"):
        cutover()


def test_publication_view_excludes_raw_run_state(tmp_path: Path) -> None:
    catalogue = Catalogue(tmp_path / "store")
    catalogue.create_dataset("demo", ["exp001"])
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "numbers.json").write_text("{}")
    (payload / "measurements.npz").write_bytes(b"raw")
    run = {
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
            "location": str(payload),
            "inventory_digest": "sha256:" + "e" * 64,
        },
        "archive": None,
        "legacy_identity": None,
    }
    root = catalogue.run_path("demo", "exp001", run["run_id"])
    root.mkdir(parents=True)
    write_json_atomic(root / "run.json", run)
    catalogue.register_run(run)
    destination = tmp_path / "publication"
    materialize_publication_view(catalogue, destination, activate=True)
    assert (destination / "exp001/numbers.json").is_file()
    assert not (destination / "exp001/measurements.npz").exists()
    assert (payload / "measurements.npz").is_file()


def test_shadow_can_render_selection_proposal_without_accepting_it(
    tmp_path: Path,
) -> None:
    catalogue = Catalogue(tmp_path / "store")
    catalogue.create_dataset("demo", ["exp001"])
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "numbers.json").write_text("proposed")
    run_id = "exp001/proposed"
    run = {
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
            "location": str(payload),
            "inventory_digest": "sha256:" + "d" * 64,
        },
        "archive": None,
        "legacy_identity": None,
    }
    root = catalogue.run_path("demo", "exp001", run_id)
    root.mkdir(parents=True)
    write_json_atomic(root / "run.json", run)
    catalogue.register_run(run)
    dataset = catalogue.load_dataset("demo")
    dataset["selection_proposal"] = {"exp001": run_id}
    catalogue.save_dataset(dataset)
    destination = tmp_path / "shadow"
    view = materialize_shadow(
        catalogue, "demo", destination, use_preview=False, use_proposal=True
    )
    assert view["proposal"] is True
    assert catalogue.load_dataset("demo")["official_runs"] == {"exp001": run_id}
