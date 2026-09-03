"""User-owned run projection and URL inputs preserve authoritative provenance."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from pingstore.cli import main
from pingstore.contracts import PingstoreError, payload_digest
from pingstore.presentation_inputs import display_origin, display_timing, projection
from pingstore.tests.test_discovery import make_run


@pytest.fixture
def source(tmp_path):
    (tmp_path / "writings").mkdir()
    (tmp_path / "writings/exp001.typ").write_text('#let inputs = ("exp001",)\n')
    (tmp_path / "writings/exp092.typ").write_text(
        '#let inputs = ("exp001", "exp002")\n'
    )
    return tmp_path


def test_projection_is_read_only_and_distinguishes_sizes(source):
    store = source / ".pingstore/runs"
    parent = make_run(store, "exp001-r001-compute", stage="compute")
    parent_meta = json.loads((parent / "run.json").read_text())
    reference = {
        "run_id": parent.name,
        "payload_digest": parent_meta["payload_digest"],
    }
    child = make_run(
        store, "exp001-r002-present", inputs={"first": reference, "also": reference}
    )
    (source / ".pingstore/collections.json").write_text(
        json.dumps({"chosen": [child.name]})
    )
    before = {p: p.read_bytes() for p in store.rglob("*") if p.is_file()}
    data = projection(source)
    assert len(data["runs"]) == 1
    run = data["runs"][0]
    assert run["collection"] == "demo" and run["views"] == ["chosen"]
    assert run["basepath"] == f"/.pingstore/runs/{child.name}/export"
    assert run["upstream_runs"] == [parent.name]
    assert run["export_bytes"] == sum(
        p.stat().st_size for p in (child / "export").iterdir()
    )
    assert run["payload_bytes"] > run["export_bytes"]
    assert run["upstream_payload_bytes"] == sum(
        len(v)
        for p, v in before.items()
        if p.is_relative_to(parent) and p != parent / "run.json"
    )
    assert before == {p: p.read_bytes() for p in store.rglob("*") if p.is_file()}
    assert not (source / ".demolab").exists()


@pytest.mark.parametrize("stage", ["compute", "analyse", "present"])
@pytest.mark.parametrize("operation", ["execute", "import"])
def test_duration_projects_recorded_operation_for_every_stage(source, stage, operation):
    directory = make_run(source / ".pingstore/runs", f"exp001-r001-{stage}", stage=stage,
                         execution={
                             "operation": operation,
                             "started_at": "2026-08-27T23:59:00.250+02:00",
                             "completed_at": "2026-08-27T22:01:37.750Z",
                         })
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    data = projection(source)
    row = data["display_runs"][0]
    assert row["duration_seconds"] == 157.5
    assert row["execution_operation"] == operation
    if stage == "present":
        assert data["runs"][0]["duration_seconds"] == 157.5
    else:
        assert data["runs"] == []
    assert before == {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}


@pytest.mark.parametrize("execution,expected", [
    ({}, None),
    ({"completed_at": "2026-08-27T10:00:00Z"}, None),
    ({"started_at": "2026-08-27T10:00:00Z"}, None),
    ({"started_at": "2026-08-27T10:00:00Z", "completed_at": "2026-08-27T10:00:00Z"}, 0),
])
def test_duration_missing_is_not_inferred_from_creation_or_file_times(source, execution, expected):
    directory = make_run(source / ".pingstore/runs", execution=execution)
    os.utime(directory / "run.json", (100, 200))
    assert projection(source)["runs"][0]["duration_seconds"] == expected


@pytest.mark.parametrize("started,completed", [
    ("2026-08-27T10:00:00Z", "2026-08-27T09:59:59Z"),
    ("2026-08-27T10:00:00", "2026-08-27T10:01:00Z"),
    ("2026-08-27T10:00:00Z", "2026-08-27T10:01:00"),
    ("invalid", "2026-08-27T10:01:00Z"),
    ("2026-08-27T10:00:00Z", 100),
])
def test_invalid_duration_fails_projection(source, started, completed):
    make_run(source / ".pingstore/runs", execution={
        "started_at": started, "completed_at": completed,
    })
    with pytest.raises(PingstoreError, match="invalid execution duration"):
        projection(source)


def make_scientific_run(source, **changes):
    timing = {
        "duration_seconds": 198256,
        "started_at": "2026-08-18T15:42:06Z",
        "completed_at": "2026-08-20T22:46:22Z",
        "origin": "slurm",
        "jobs": 2,
        "job_seconds": 10800.75,
    }
    timing.update(changes)
    directory = make_run(source / ".pingstore/runs", "exp001-r001-compute", stage="compute",
                         execution={"operation": "import", "started_at": "2026-08-27T10:00:00Z",
                                    "completed_at": "2026-08-27T10:00:03Z"},
                         scientific_execution=timing)
    return directory


def test_scientific_span_and_job_total_are_separate_from_import(source):
    directory = make_scientific_run(source)
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    row = projection(source)["display_runs"][0]
    assert row["duration_seconds"] == 3
    assert row["origin"] == "local"
    assert row["display_origin"] == {
        "value": "slurm", "basis": "scientific-execution",
    }
    assert row["display_timing"]["duration_seconds"] == 198256
    assert row["display_timing"]["basis"] == "scientific-execution"
    assert row["display_timing"]["import_seconds"] == 3
    assert row["scientific_timing"] == {
        "duration_seconds": 198256, "started_at": "2026-08-18T15:42:06Z",
        "completed_at": "2026-08-20T22:46:22Z", "origin": "slurm",
        "jobs": 2, "job_seconds": 10800.75,
    }
    assert before == {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "record, expected",
    [
        (
            {
                "origin": "local",
                "execution": {"operation": "historical-import"},
                "historical_import": {"producer": {"origin": "slurm"}},
            },
            {"value": "slurm", "basis": "historical-producer"},
        ),
        (
            {"origin": "local", "execution": {"operation": "historical-import"}},
            {"value": "unknown", "basis": "unrecorded-import-source"},
        ),
        (
            {"origin": "slurm-wilkes", "execution": {"operation": "execute"}},
            {"value": "slurm-wilkes", "basis": "recorded-operation"},
        ),
        (
            {
                "origin": "local",
                "execution": {"operation": "historical-import"},
                "historical_import": {"producer_origin": "hpc"},
            },
            {"value": "hpc", "basis": "historical-producer"},
        ),
        (
            {
                "origin": "local",
                "execution": {"operation": "historical-import"},
                "historical_import": {"producer": {"slurm_job_id": "123"}},
            },
            {"value": "slurm", "basis": "historical-producer"},
        ),
    ],
)
def test_display_origin_distinguishes_scientific_work_from_import(record, expected):
    assert display_origin(record) == expected


def test_display_timing_uses_historical_hpc_wall_clock_instead_of_import():
    record = {
        "run_id": "exp001-r001-compute",
        "origin": "local",
        "execution": {
            "operation": "historical-import",
            "started_at": "2026-08-28T10:00:00Z",
            "completed_at": "2026-08-28T10:00:03Z",
        },
        "historical_import": {"producer": {"status": {
            "started_at_utc": "2026-08-20T02:42:34Z",
            "ended_at_utc": "2026-08-20T03:38:41Z",
        }}},
    }
    assert display_timing(record, None) == {
        "duration_seconds": 3367,
        "started_at": "2026-08-20T02:42:34Z",
        "completed_at": "2026-08-20T03:38:41Z",
        "basis": "historical-producer",
        "import_seconds": 3,
    }


def test_display_timing_omits_import_duration_when_hpc_wall_clock_is_unknown():
    record = {
        "run_id": "exp001-r001-compute",
        "origin": "local",
        "execution": {
            "operation": "historical-import",
            "started_at": "2026-08-28T10:00:00Z",
            "completed_at": "2026-08-28T10:00:03Z",
        },
    }
    assert display_timing(record, None) == {
        "duration_seconds": None,
        "basis": "unrecorded-import-source",
        "import_seconds": 3,
    }


@pytest.mark.parametrize("value", [-1, None, True, float("nan")])
def test_scientific_timing_rejects_invalid_duration(source, value):
    make_scientific_run(source, duration_seconds=value)
    with pytest.raises(PingstoreError, match="invalid scientific duration"):
        projection(source)


def test_url_selection_requires_declared_key_and_valid_present_run(source):
    run = make_run(source / ".pingstore/runs", "exp001-r001-present")
    value = "/" + (run / "export").relative_to(source).as_posix()
    assert projection(source, article="exp001", overrides={"source.exp001": value})
    with pytest.raises(PingstoreError, match="not declared"):
        projection(source, article="exp001", overrides={"source.exp002": value})
    with pytest.raises(PingstoreError, match="validated presentation"):
        projection(
            source, article="exp001", overrides={"source.exp001": value + "/other"}
        )


def test_experiment_dependencies_do_not_depend_on_runs_or_selection(source):
    (source / "writings/exp092.typ").write_text(
        '#let inputs = ("exp092", "baseline.exp001", "candidate.exp001", "exp002")\n'
    )
    declared = {"exp002": ("exp001", "exp001"), "exp003": ("exp002",)}
    expected = {
        "exp001": {"upstream": [], "downstream": ["exp002", "exp092"]},
        "exp002": {"upstream": ["exp001"], "downstream": ["exp003", "exp092"]},
        "exp003": {"upstream": ["exp002"], "downstream": []},
        "exp092": {"upstream": ["exp001", "exp002"], "downstream": []},
    }
    assert projection(source, declared_dependencies=declared)["experiment_dependencies"] == expected
    store = source / ".pingstore/runs"
    parent = make_run(store, "exp001-r001-present")
    # A recorded run edge is deliberately different from the declared experiment graph.
    make_run(store, "exp009-r001-compute", stage="compute", inputs={
        "source": {
            "run_id": parent.name, "payload_digest": payload_digest(parent),
        },
    })
    data = projection(source, declared_dependencies=declared, article="exp001", overrides={
        "source.exp001": f"/.pingstore/runs/{parent.name}/export",
    })
    assert data["experiment_dependencies"] == expected
    assert all("upstream_experiments" not in row and "downstream_experiments" not in row
               for row in data["display_runs"])


def test_defaults_and_corrupt_payload_fail_closed(source):
    run = make_run(source / ".pingstore/runs", "exp001-r001-present")
    (source / "writings/run-defaults.json").write_text(
        json.dumps({"exp001": {"exp001": run.name}})
    )
    assert projection(source)["defaults"]["exp001"]["exp001"] == run.name
    (run / "export/numbers.json").write_text('{"value": 999}')
    with pytest.raises(PingstoreError, match="checksum"):
        projection(source)


def test_wrong_upstream_payload_pin_fails(source):
    store = source / ".pingstore/runs"
    parent = make_run(store, "exp001-r001-compute", stage="compute")
    make_run(
        store,
        "exp001-r002-present",
        inputs={
            "analysis": {
                "run_id": parent.name,
                "payload_digest": "sha256:" + "0" * 64,
            }
        },
    )
    with pytest.raises(PingstoreError, match="changed upstream"):
        projection(source)


def test_empty_store_and_unavailable_default(source):
    assert projection(source)["runs"] == []
    (source / "writings/run-defaults.json").write_text(
        '{"exp001":{"exp001":"exp001-r001-present"}}'
    )
    with pytest.raises(PingstoreError, match="unavailable default"):
        projection(source)


def test_cli_uses_working_directory_and_preserves_unchanged_output(source, monkeypatch):
    make_run(source / ".pingstore/runs", "exp001-r001-present")
    monkeypatch.chdir(source)
    monkeypatch.delenv("DEMOLAB_INPUTS", raising=False)
    monkeypatch.delenv("DEMOLAB_ARTICLE", raising=False)
    assert main(["presentation-inputs"]) == 0
    target = source / ".demolab/pinglab-inputs.json"
    assert json.loads(target.read_text()) == projection(source)
    before = target.stat().st_mtime_ns
    assert main(["presentation-inputs"]) == 0
    assert target.stat().st_mtime_ns == before


def test_cli_explicit_root_and_failure_preserve_output(source, monkeypatch, capsys):
    run = make_run(source / ".pingstore/runs", "exp001-r001-present")
    monkeypatch.setenv("DEMOLAB_ARTICLE", "exp001")
    monkeypatch.setenv(
        "DEMOLAB_INPUTS",
        json.dumps({"source.exp001": f"/.pingstore/runs/{run.name}/export"}),
    )
    command = ["presentation-inputs", "--root", str(source)]
    assert main(command) == 0
    target = source / ".demolab/pinglab-inputs.json"
    before = target.read_bytes(), target.stat().st_mtime_ns
    (run / "export/numbers.json").write_text("corrupted")
    assert main(command) == 1
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before
    output = capsys.readouterr()
    assert output.out == ""
    assert "checksum" in output.err
    assert "Traceback" not in output.err


@pytest.mark.parametrize("value", ["{", "[]", "null", '{"source.exp001": 1}'])
def test_cli_rejects_invalid_input_environment(source, monkeypatch, capsys, value):
    monkeypatch.setenv("DEMOLAB_INPUTS", value)
    assert main(["presentation-inputs", "--root", str(source)]) == 1
    assert not (source / ".demolab").exists()
    output = capsys.readouterr()
    assert output.out == ""
    assert "pingstore presentation-inputs:" in output.err
    assert "Traceback" not in output.err


def test_module_entry_point(source):
    make_run(source / ".pingstore/runs", "exp001-r001-present")
    result = subprocess.run(
        [sys.executable, "-m", "pingstore", "presentation-inputs"],
        cwd=source,
        env={**os.environ, "DEMOLAB_INPUTS": "{}", "DEMOLAB_ARTICLE": ""},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ""
    assert "1 present runs" in result.stderr
    assert json.loads(
        (source / ".demolab/pinglab-inputs.json").read_text()
    ) == projection(source)


@pytest.mark.parametrize("stage", ["compute", "analyse"])
def test_nonpresent_rows_are_display_only_with_recursive_export_sizes(source, stage):
    store = source / ".pingstore/runs"
    other = make_run(store, f"exp001-r001-{stage}", stage=stage)
    nested = other / "export/cells--seed1"
    nested.mkdir(parents=True)
    (nested / "weights.bin").write_bytes(b"nested scientific data")
    (nested / "metrics.json").write_text("{}")
    metadata = json.loads((other / "run.json").read_text())
    metadata["payload_digest"] = payload_digest(other)
    (other / "run.json").write_text(json.dumps(metadata))
    present = make_run(store, "exp001-r002-present")
    data = projection(source)
    assert [row["id"] for row in data["runs"]] == [present.name]
    row = next(row for row in data["display_runs"] if row["id"] == other.name)
    assert row["stage"] == stage
    assert row["created_at"] == "2026-08-27T10:00:00+00:00"
    assert "basepath" not in row and "files" not in row
    assert row["export_bytes"] == sum(
        p.stat().st_size for p in (other / "export").rglob("*") if p.is_file()
    )
    with pytest.raises(PingstoreError, match="validated presentation"):
        projection(source, article="exp001", overrides={
            "source.exp001": f"/.pingstore/runs/{other.name}/export",
        })
    (source / "writings/run-defaults.json").write_text(json.dumps({
        "exp001": {"exp001": other.name},
    }))
    with pytest.raises(PingstoreError, match="unavailable default"):
        projection(source)


def test_corrupt_display_only_run_fails_validation(source):
    other = make_run(source / ".pingstore/runs", "exp001-r001-compute", stage="compute")
    (other / "export/numbers.json").write_text("corrupt")
    with pytest.raises(PingstoreError, match="checksum"):
        projection(source)
