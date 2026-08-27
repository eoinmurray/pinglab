"""v3 enforcement and mixed-store compatibility; no scientific execution."""

import pytest

from pingstore import stages
from pingstore.contracts import (
    LEGACY_RUN_SCHEMA, RUN_SCHEMA, PingstoreError, load_json, payload_digest,
    validate_run_directory, write_json_atomic,
)
from pingstore.discovery import discover_runs
from pingstore.layout import export_directory, initialize_layout, presentation_directory
from pingstore.materialize import materialize_run, materialize_view


def make_run(store, stage="present", *, number=1, schema=RUN_SCHEMA):
    suffix = f"-{stage}" if stage else ""
    identity = f"exp001-r{number:03d}{suffix}-local"
    directory = store / "runs" / identity
    initialize_layout(directory, "exp001", schema=schema)
    record = {
        "schema": schema, "run_id": identity, "experiment": "exp001",
        "collection": "demo", "origin": "local",
        "created_at": "2026-08-27T12:00:00+00:00",
        "execution": {}, "provenance": {},
    }
    if stage:
        record.update(stage=stage, inputs={})
    output = directory / ("export" if schema == RUN_SCHEMA else "presentation")
    (output / "numbers.json").write_text('{"value": 1}\n')
    record["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", record)
    return directory


def resign(directory):
    record = load_json(directory / "run.json")
    record["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", record)


@pytest.mark.parametrize("stage", ["compute", "analyse", "present"])
def test_v3_minimal_root_and_default_export(tmp_path, stage):
    directory = make_run(tmp_path, stage)
    record = validate_run_directory(directory)
    assert {p.name for p in directory.iterdir()} == {"run.json", "export"}
    assert export_directory(directory, record) == directory / "export"
    expected = directory / "export" if stage == "present" else None
    assert presentation_directory(directory, record) == expected


@pytest.mark.parametrize("stage", ["compute", "analyse", "present"])
def test_only_present_exports_must_be_flat(tmp_path, stage):
    directory = make_run(tmp_path, stage)
    (directory / "export/nested").mkdir()
    resign(directory)
    if stage == "present":
        with pytest.raises(PingstoreError, match="flat regular files"):
            validate_run_directory(directory)
    else:
        validate_run_directory(directory)


@pytest.mark.parametrize("entry", ["presentation", "unexpected.json"])
def test_v3_rejects_extra_root_entries(tmp_path, entry):
    directory = make_run(tmp_path)
    (directory / entry).write_text("unexpected")
    resign(directory)
    with pytest.raises(PingstoreError, match="v3 run requires"):
        validate_run_directory(directory)


def test_v3_requires_stage_and_counter_first_id(tmp_path):
    directory = make_run(tmp_path)
    record = load_json(directory / "run.json")
    del record["stage"]
    write_json_atomic(directory / "run.json", record)
    with pytest.raises(PingstoreError, match="explicit stage"):
        validate_run_directory(directory)
    record.update(stage="present", run_id="exp001-present-r001-local")
    write_json_atomic(directory / "run.json", record)
    with pytest.raises(PingstoreError, match="staged run ID"):
        validate_run_directory(directory)


@pytest.mark.parametrize("relative", ["README.md", "provenance/nested/source.patch"])
def test_optional_evidence_is_checksummed(tmp_path, relative):
    directory = make_run(tmp_path)
    evidence = directory / relative
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("original")
    resign(directory)
    validate_run_directory(directory)
    evidence.write_text("changed")
    with pytest.raises(PingstoreError, match="checksum mismatch"):
        validate_run_directory(directory)


@pytest.mark.parametrize("relative", ["export/link", "provenance/link", "run.json"])
def test_v3_rejects_symlinks(tmp_path, relative):
    directory = make_run(tmp_path)
    target = tmp_path / "outside"
    target.write_text("outside")
    link = directory / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists():
        link.unlink()
    link.symlink_to(target)
    with pytest.raises(PingstoreError):
        validate_run_directory(directory)


def test_discovery_resolves_mixed_schemas_and_validates_excluded_runs(tmp_path):
    compute = make_run(tmp_path, "compute")
    make_run(tmp_path, "analyse", number=2)
    present = make_run(tmp_path, number=3)
    legacy = make_run(tmp_path, None, number=4, schema=LEGACY_RUN_SCHEMA)
    make_run(tmp_path, "compute", number=5, schema=LEGACY_RUN_SCHEMA)
    rows = discover_runs(tmp_path / "runs")
    assert [row["presentation"] for row in rows] == [
        f"{present.name}/export", f"{legacy.name}/presentation",
    ]
    (compute / "export/numbers.json").write_text("corrupted")
    with pytest.raises(PingstoreError, match="checksum mismatch"):
        discover_runs(tmp_path / "runs")


def test_discovery_omits_empty_and_bookkeeping_only_exports(tmp_path):
    directory = make_run(tmp_path)
    (directory / "export/numbers.json").write_text("")
    write_json_atomic(directory / "export/_manifest.json", {"stage": "present"})
    resign(directory)
    assert discover_runs(tmp_path / "runs") == []


def test_materialization_copies_only_whole_present_export(tmp_path):
    directory = make_run(tmp_path)
    (directory / "export/download.unusual").write_bytes(b"arbitrary suffix")
    write_json_atomic(directory / "provenance/command.json", {"command": ["example"]})
    resign(directory)
    materialize_run(tmp_path, directory.name, tmp_path / "artifacts")
    copied = tmp_path / "artifacts/exp001"
    assert {p.name: p.read_bytes() for p in copied.iterdir()} == {
        p.name: p.read_bytes() for p in (directory / "export").iterdir()
    }


@pytest.mark.parametrize("stage", ["compute", "analyse"])
@pytest.mark.parametrize("schema", [RUN_SCHEMA, LEGACY_RUN_SCHEMA])
def test_both_materializers_reject_scientific_stages(tmp_path, stage, schema):
    directory = make_run(tmp_path, stage, schema=schema)
    write_json_atomic(tmp_path / "collections.json", {"demo": [directory.name]})
    with pytest.raises(PingstoreError, match="cannot be published"):
        materialize_run(tmp_path, directory.name, tmp_path / "artifacts")
    with pytest.raises(PingstoreError, match="cannot be published"):
        materialize_view(tmp_path, "demo", tmp_path / "view")
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "view").exists()


def test_stage_writer_finishes_v3_with_separate_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "memberships", lambda repo: {"exp001": "demo"})
    monkeypatch.setattr(stages, "_capture_code", lambda repo, directory: {
        "git_commit": "fixture", "dirty": False,
    })
    identity = stages.reserve_stage(tmp_path / ".pingstore", "exp001", "present", origin="local")
    temporary = tmp_path / ".pingstore/runs" / f".{identity}.tmp"
    assert stages.stage_reservation(temporary)["schema"] == RUN_SCHEMA
    with stages.stage_run(tmp_path, "exp001", "present", run_id=identity) as run:
        (run.export / "numbers.json").write_text("{}")
    assert not temporary.exists()
    assert validate_run_directory(run.directory)["schema"] == RUN_SCHEMA
    assert (run.export / "_manifest.json").is_file()
    assert (run.provenance / "run.sh").is_file()
    assert not (run.export / "provenance").exists()
    assert not (run.directory / "presentation").exists()


def test_legacy_reservation_is_not_rewritten(tmp_path):
    old = tmp_path / "export/provenance/reservation.json"
    write_json_atomic(old, {"run_id": "exp001-r001-compute-local"})
    before = old.read_bytes()
    with pytest.raises(PingstoreError, match="legacy v2 reservation"):
        stages.stage_reservation(tmp_path)
    assert old.read_bytes() == before
    assert not (tmp_path / "provenance").exists()
