"""V4 storage, validation and stage lifecycle checks; no scientific execution."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pingstore import stages
from pingstore.contracts import (
    RUN_SCHEMA,
    PingstoreError,
    load_json,
    payload_digest,
    validate_run_directory,
    write_json_atomic,
)
from pingstore.discovery import discover_runs
from pingstore.layout import (
    canonical_export_mapping,
    canonical_export_relative,
    export_directory,
    initialize_layout,
    presentation_directory,
)


def make_run(store, stage="present", *, number=1):
    identity = f"exp001-r{number:03d}-{stage}"
    directory = store / "runs" / identity
    initialize_layout(directory, "exp001")
    record = {
        "schema": RUN_SCHEMA, "run_id": identity, "experiment": "exp001",
        "collection": "demo", "origin": "local",
        "created_at": "2026-08-27T12:00:00+00:00",
        "execution": {}, "provenance": {},
        "stage": stage, "inputs": {},
    }
    output = directory / "export"
    (output / "numbers.json").write_text('{"value": 1}\n')
    write_json_atomic(directory / "run.json", {**record, "payload_digest": "sha256:" + "0" * 64})
    record["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", record)
    return directory


def resign(directory):
    record = load_json(directory / "run.json")
    record["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", record)


@pytest.mark.parametrize("stage", ["compute", "analyse", "present"])
def test_v4_minimal_root_and_default_export(tmp_path, stage):
    directory = make_run(tmp_path, stage)
    record = validate_run_directory(directory)
    assert {p.name for p in directory.iterdir()} == {"run.json", "README.md", "export"}
    assert export_directory(directory, record) == directory / "export"
    expected = directory / "export" if stage == "present" else None
    assert presentation_directory(directory, record) == expected


@pytest.mark.parametrize("stage", ["compute", "analyse", "present"])
def test_only_present_exports_must_be_flat(tmp_path, stage):
    directory = make_run(tmp_path, stage)
    (directory / "export/nested").mkdir()
    if stage != "present":
        (directory / "export/nested/metrics.json").write_text("{}")
        (directory / "export/nested/recording.npz").write_bytes(b"fixture")
    resign(directory)
    if stage == "present":
        with pytest.raises(PingstoreError, match="flat regular files"):
            validate_run_directory(directory)
    else:
        validate_run_directory(directory)


def test_compute_and_analyse_reject_deeper_than_one_unit_directory(tmp_path):
    directory = make_run(tmp_path, "compute")
    deep = directory / "export/unit/inner/value.json"
    deep.parent.mkdir(parents=True)
    deep.write_text("{}")
    resign(directory)
    with pytest.raises(PingstoreError, match="scientific unit directory"):
        validate_run_directory(directory)


def test_stage_completion_normalizes_tool_native_paths_and_references(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(stages, "memberships", lambda repo: {"exp001": "demo"})
    monkeypatch.setattr(
        stages,
        "_capture_code",
        lambda repo, directory: {"git_commit": "fixture", "dirty": False},
    )
    with stages.stage_run(tmp_path, "exp001", "compute") as run:
        artifact = run.export / "jobs/condition-a/result.json"
        artifact.parent.mkdir(parents=True)
        write_json_atomic(artifact, {"path": "jobs/condition-a/result.json"})
        run.record["artifact"] = "export/jobs/condition-a/result.json"
    record = validate_run_directory(run.directory)
    target = run.export / "jobs--condition-a--result.json"
    assert load_json(target) == {"path": "jobs--condition-a--result.json"}
    assert record["artifact"] == "export/jobs--condition-a--result.json"
    assert "export_root" not in record


def test_stage_completion_rejects_canonical_path_collisions(tmp_path, monkeypatch):
    monkeypatch.setattr(stages, "memberships", lambda repo: {"exp001": "demo"})
    monkeypatch.setattr(
        stages,
        "_capture_code",
        lambda repo, directory: {"git_commit": "fixture", "dirty": False},
    )
    with pytest.raises(PingstoreError, match="canonical export paths collide"):
        with stages.stage_run(tmp_path, "exp001", "compute") as run:
            for relative in (
                "jobs/condition-a/result.json",
                "jobs--condition-a/result.json",
            ):
                target = run.export / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("{}")


def test_bundle_internal_paths_become_role_names():
    assert canonical_export_relative(
        Path("branches/k/network.bundle/reports/summary.md")
    ) == Path("branches--k--network.bundle/reports--summary.md")


def test_singleton_units_flatten_and_standardize_recording_name():
    mapping = canonical_export_mapping(
        [Path("fi/ping-r2/recording.npz"), Path("bundle/metrics.json"), Path("bundle/config.json")]
    )
    assert mapping == {
        "fi/ping-r2/recording.npz": "fi--ping-r2--recording.npz",
        "bundle/metrics.json": "bundle/metrics.json",
        "bundle/config.json": "bundle/config.json",
    }


@pytest.mark.parametrize(
    "name", ["snapshot.npz", "recordings.npz", "unit--snapshot.npz"]
)
def test_v4_rejects_legacy_recording_aliases(tmp_path, name):
    directory = make_run(tmp_path, "compute")
    (directory / "export" / name).write_bytes(b"fixture")
    resign(directory)
    with pytest.raises(PingstoreError, match="noncanonical scientific role filename"):
        validate_run_directory(directory)


def test_v4_rejects_singleton_unit_directory(tmp_path):
    directory = make_run(tmp_path, "compute")
    unit = directory / "export/unit"
    unit.mkdir()
    (unit / "metrics.json").write_text("{}")
    resign(directory)
    with pytest.raises(PingstoreError, match="at least two files"):
        validate_run_directory(directory)


@pytest.mark.parametrize("entry", ["presentation", "unexpected.json"])
def test_v4_rejects_extra_root_entries(tmp_path, entry):
    directory = make_run(tmp_path)
    (directory / entry).write_text("unexpected")
    resign(directory)
    with pytest.raises(PingstoreError, match="v4 run must contain exactly"):
        validate_run_directory(directory)


def test_v4_requires_stage_and_counter_first_id(tmp_path):
    directory = make_run(tmp_path)
    record = load_json(directory / "run.json")
    del record["stage"]
    write_json_atomic(directory / "run.json", record)
    with pytest.raises(PingstoreError, match="stage must"):
        validate_run_directory(directory)
    record.update(stage="present", run_id="exp001-present-r001-local")
    write_json_atomic(directory / "run.json", record)
    with pytest.raises(PingstoreError, match="run_id must encode"):
        validate_run_directory(directory)


@pytest.mark.parametrize("relative", ["export/data--nested--value.json", "export/value.bin"])
def test_export_data_is_checksummed(tmp_path, relative):
    directory = make_run(tmp_path, "compute")
    evidence = directory / relative
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("original")
    resign(directory)
    validate_run_directory(directory)
    evidence.write_text("changed")
    with pytest.raises(PingstoreError, match="checksum mismatch"):
        validate_run_directory(directory)


@pytest.mark.parametrize("relative", ["export/link", "README.md", "run.json"])
def test_v4_rejects_symlinks(tmp_path, relative):
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


def test_discovery_requires_v4_and_validates_excluded_runs(tmp_path):
    compute = make_run(tmp_path, "compute")
    make_run(tmp_path, "analyse", number=2)
    present = make_run(tmp_path, number=3)
    rows = discover_runs(tmp_path / "runs")
    assert [row["presentation"] for row in rows] == [f"{present.name}/export"]
    (compute / "export/numbers.json").write_text("corrupted")
    with pytest.raises(PingstoreError, match="checksum mismatch"):
        discover_runs(tmp_path / "runs")


def test_discovery_omits_empty_and_bookkeeping_only_exports(tmp_path):
    directory = make_run(tmp_path)
    (directory / "export/numbers.json").write_text("")
    write_json_atomic(directory / "export/_manifest.json", {"stage": "present"})
    resign(directory)
    assert discover_runs(tmp_path / "runs") == []


def test_non_v4_evidence_is_rejected_by_operational_readers(tmp_path):
    directory = make_run(tmp_path, "compute")
    record = load_json(directory / "run.json")
    record["schema"] = "pingstore.run/v3"
    write_json_atomic(directory / "run.json", record)
    before = (directory / "run.json").read_bytes(), payload_digest(directory)
    with pytest.raises(PingstoreError, match="operational run schema"):
        stages.source_run(tmp_path, directory.name)
    with pytest.raises(PingstoreError, match="operational run schema"):
        discover_runs(tmp_path / "runs")
    assert before == ((directory / "run.json").read_bytes(), payload_digest(directory))


@pytest.mark.parametrize("origin", ["local", "slurm-wilkes", "slurm-zenith"])
def test_stage_writer_finishes_v4_with_readme_and_no_provenance(tmp_path, monkeypatch, origin):
    monkeypatch.setattr(stages, "memberships", lambda repo: {"exp001": "demo"})
    monkeypatch.setattr(stages, "_capture_code", lambda repo, directory: {
        "git_commit": "fixture", "dirty": False,
    })
    identity = stages.reserve_stage(tmp_path / ".pingstore", "exp001", "present", origin=origin)
    assert identity == "exp001-r001-present"
    temporary = tmp_path / ".pingstore/runs" / f".{identity}.tmp"
    assert stages.stage_reservation(temporary)["schema"] == RUN_SCHEMA
    with stages.stage_run(tmp_path, "exp001", "present", run_id=identity) as run:
        (run.export / "numbers.json").write_text("{}")
    assert not temporary.exists()
    record = validate_run_directory(run.directory)
    assert record["schema"] == RUN_SCHEMA
    assert record["origin"] == origin
    assert {path.name for path in run.export.iterdir()} == {"numbers.json"}
    assert (run.directory / "README.md").is_file()
    assert not (run.directory / "provenance").exists()
    assert {path.name for path in run.directory.iterdir()} == {"run.json", "README.md", "export"}
    assert not (run.directory / "presentation").exists()


def test_legacy_reservation_is_not_rewritten(tmp_path):
    old = tmp_path / "export/provenance/reservation.json"
    write_json_atomic(old, {"run_id": "exp001-r001-compute-local"})
    before = old.read_bytes()
    with pytest.raises(PingstoreError, match="legacy v2/v3 reservation"):
        stages.stage_reservation(tmp_path)
    assert old.read_bytes() == before
    assert not (tmp_path / "provenance").exists()


def test_source_neutral_reservations_keep_origin_and_avoid_cross_origin_collisions(tmp_path):
    runs = tmp_path / "runs"
    # All earlier formats occupy their counters, including incomplete evidence.
    (runs / "exp001-r007-compute-local").mkdir(parents=True)
    (runs / ".exp001-analyse-r008-slurm.tmp").mkdir()
    (runs / "exp001-r009-present").mkdir()
    origins = ["local", "slurm-wilkes", "slurm-zenith"] * 4
    with ThreadPoolExecutor(max_workers=6) as pool:
        identities = list(pool.map(lambda origin: stages.reserve_stage(
            tmp_path, "exp001", "compute", origin=origin), origins))
    assert len(set(identities)) == len(origins)
    for origin, identity in zip(origins, identities):
        assert identity.endswith("-compute")
        assert int(identity.split("-")[1][1:]) >= 10
        reservation = stages.stage_reservation(runs / f".{identity}.tmp")
        assert reservation["origin"] == origin


def test_suffixed_reservation_cannot_be_completed(tmp_path):
    path = tmp_path / "runs/.exp001-r001-compute-local.tmp"
    reservation = path / ".reservation.json"
    write_json_atomic(reservation, {"schema": RUN_SCHEMA,
        "run_id": "exp001-r001-compute-local", "experiment": "exp001",
        "stage": "compute", "origin": "local"})
    before = reservation.read_bytes()
    with pytest.raises(PingstoreError, match="source-neutral reservation"):
        stages.stage_reservation(path)
    assert reservation.read_bytes() == before
