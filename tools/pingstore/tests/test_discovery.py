from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pingstore.cli import main
from pingstore.contracts import PingstoreError, payload_digest, payload_inventory
from pingstore.discovery import discover_runs


def make_run(source: Path, run_id: str = "exp001-r001-present", **overrides) -> Path:
    directory = source / run_id
    (directory / "export").mkdir(parents=True)
    (directory / "README.md").write_text("Run notes\n")
    (directory / "export/numbers.json").write_text('{"value": 1}\n')
    (directory / "export/_manifest.json").write_text(
        '{"run_id": "wrong", "run_at": "wrong"}'
    )
    record = {
        "schema": "pingstore.run/v4",
        "run_id": run_id,
        "experiment": run_id.split("-")[0],
        "collection": "demo",
        "origin": "local",
        "created_at": "2026-08-27T12:00:00+02:00",
        "stage": "present",
        "inputs": {},
        "execution": {},
        "provenance": {},
        "payload_digest": payload_digest(directory),
        **overrides,
    }
    (directory / "run.json").write_text(json.dumps({**record, "payload_digest": "sha256:" + "0" * 64}))
    record["payload_digest"] = payload_digest(directory)
    (directory / "run.json").write_text(json.dumps(record))
    return directory


def test_projection_uses_authoritative_metadata_without_mutation(tmp_path):
    directory = make_run(tmp_path)
    os.utime(directory / "run.json", (100, 100))
    before = payload_inventory(directory), (directory / "run.json").read_bytes()
    mtimes = {p: p.stat().st_mtime_ns for p in directory.rglob("*")}
    assert discover_runs(tmp_path) == [
        {
            "id": directory.name,
            "experiment": "exp001",
            "label": directory.name,
            "created_at": "2026-08-27T10:00:00+00:00",
            "presentation": f"{directory.name}/export",
        }
    ]
    assert before == (
        payload_inventory(directory),
        (directory / "run.json").read_bytes(),
    )
    assert mtimes == {p: p.stat().st_mtime_ns for p in directory.rglob("*")}


def test_all_runs_sorted_without_selection_or_filtering(tmp_path):
    names = ["exp002-r001-present", "exp001-r002-present", "exp001-r001-present"]
    for name in names:
        make_run(tmp_path, name)
    assert [row["id"] for row in discover_runs(tmp_path)] == sorted(names)


def test_hidden_incomplete_entries_and_files_are_ignored(tmp_path):
    (tmp_path / ".exp001-r001-present.tmp").mkdir()
    (tmp_path / ".metadata").write_text("not a run")
    (tmp_path / "notes.txt").write_text("not a run")
    assert discover_runs(tmp_path) == []


def test_symlink_candidates_are_not_followed(tmp_path):
    source = tmp_path / "runs"
    source.mkdir()
    target = make_run(tmp_path / "outside")
    (source / target.name).symlink_to(target, target_is_directory=True)
    (source / "broken").symlink_to(tmp_path / "absent")
    assert discover_runs(source) == []


@pytest.mark.parametrize("nested", [False, True])
def test_symlink_source_or_ancestor_is_rejected(tmp_path, nested):
    actual = tmp_path / "actual"
    (actual / "runs").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(PingstoreError, match="symlinks"):
        discover_runs(link / "runs" if nested else link)


@pytest.mark.parametrize(
    "path",
    [
        "export/numbers.json",
        "export/_manifest.json",
    ],
)
def test_corrupt_payload_fails_including_export_and_nested_metadata(tmp_path, path):
    directory = make_run(tmp_path)
    (directory / path).write_bytes(b"corrupted")
    with pytest.raises(PingstoreError, match="payload checksum mismatch"):
        discover_runs(tmp_path)


def test_readme_history_can_be_amended_without_changing_payload(tmp_path):
    directory = make_run(tmp_path)
    digest = json.loads((directory / "run.json").read_text())["payload_digest"]
    (directory / "README.md").write_text("Corrected history\n")
    assert discover_runs(tmp_path)
    assert json.loads((directory / "run.json").read_text())["payload_digest"] == digest


@pytest.mark.parametrize(
    "damage",
    [
        "missing-manifest",
        "extra-root",
        "nested-presentation",
        "symlink-payload",
        "invalid-json",
    ],
)
def test_invalid_visible_run_is_not_silently_omitted(tmp_path, damage):
    directory = make_run(tmp_path)
    if damage == "missing-manifest":
        (directory / "run.json").unlink()
    elif damage == "extra-root":
        (directory / "extra").write_text("unexpected")
    elif damage == "nested-presentation":
        (directory / "export/nested").mkdir()
    elif damage == "symlink-payload":
        (directory / "export/link").symlink_to(directory / "README.md")
    else:
        (directory / "run.json").write_text("{")
    with pytest.raises(PingstoreError, match=directory.name):
        discover_runs(tmp_path)


@pytest.mark.parametrize(
    "fields",
    [
        {"schema": "pingstore.run/v1"},
        {"experiment": "exp002"},
        {"run_id": "exp001-other-local"},
        {"export_root": "../outside"},
    ],
)
def test_invalid_identity_schema_or_export_root(tmp_path, fields):
    manifest = make_run(tmp_path) / "run.json"
    record = json.loads(manifest.read_text())
    record.update(fields)
    manifest.write_text(json.dumps(record))
    with pytest.raises(PingstoreError):
        discover_runs(tmp_path)


@pytest.mark.parametrize(
    "timestamp",
    ["yesterday", "2026-08-27", "2026-08-27T12:00:00", "2026-99-27T12:00:00Z"],
)
def test_invalid_or_naive_timestamp_is_not_replaced_by_file_time(tmp_path, timestamp):
    make_run(tmp_path, created_at=timestamp)
    with pytest.raises(PingstoreError, match="cannot discover"):
        discover_runs(tmp_path)


def test_utc_z_timestamp(tmp_path):
    make_run(tmp_path, created_at="2026-08-27T12:00:00Z")
    assert discover_runs(tmp_path)[0]["created_at"] == "2026-08-27T12:00:00+00:00"


def test_cli_source_precedence_and_default(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEMOLAB_PREVIEW_SOURCE", raising=False)
    make_run(tmp_path / ".pingstore/runs")
    assert main(["discover"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1
    env_source = tmp_path / "env-runs"
    env_source.mkdir()
    monkeypatch.setenv("DEMOLAB_PREVIEW_SOURCE", str(env_source))
    assert main(["discover"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert main(["discover", "--source", ".pingstore/runs"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1


def test_cli_fails_without_partial_json_or_traceback(tmp_path, capsys):
    make_run(tmp_path)
    (tmp_path / "zzz-invalid").mkdir()
    assert main(["discover", "--source", str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "zzz-invalid" in output.err
    assert "Traceback" not in output.err


@pytest.mark.parametrize("file_source", [False, True])
def test_missing_or_file_source_fails_without_creating_it(
    tmp_path, capsys, file_source
):
    source = tmp_path / "missing"
    if file_source:
        source.write_text("not a directory")
    assert main(["discover", "--source", str(source)]) == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "existing runs directory" in output.err
    assert source.is_file() == file_source


def test_module_entry_point_and_environment_protocol(tmp_path):
    make_run(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "pingstore", "discover"],
        env={**os.environ, "DEMOLAB_PREVIEW_SOURCE": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert len(json.loads(result.stdout)) == 1
    assert result.stderr == ""


def test_unknown_subcommands_are_rejected(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["delete", "exp001-r001-present-local"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
