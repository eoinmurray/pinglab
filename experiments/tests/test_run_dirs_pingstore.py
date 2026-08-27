from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.helpers import run_dirs
from experiments.helpers.paths import runner_paths


def test_default_runner_paths_live_only_in_hidden_pingstore_run() -> None:
    paths = runner_paths("exp001")
    assert ".pingstore/runs/.exp001-" in paths.state.as_posix()
    assert paths.state.parts[-2:] == ("export", "state")
    assert paths.derived == paths.state.parent.parent / "presentation"
    assert "temp/experiments" not in paths.state.as_posix()


def test_legacy_success_finalizer_captures_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    figures = tmp_path / ".pingstore/runs/.exp001-r001-local.tmp/presentation"
    figures.mkdir(parents=True)
    (figures / "numbers.json").write_text("{}\n")
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        run_dirs, "artifacts_and_figures", lambda _slug: (tmp_path / "scratch", figures)
    )
    monkeypatch.setattr(
        run_dirs,
        "runner_paths",
        lambda _slug: SimpleNamespace(isolated=False, state=tmp_path / "scratch"),
    )

    def fake_manifest(path: Path, **kwargs: object) -> None:
        (path / "_manifest.json").write_text(json.dumps(kwargs))

    monkeypatch.setattr(run_dirs, "write_manifest", fake_manifest)
    monkeypatch.setattr(
        run_dirs,
        "finalize_local_run",
        lambda _repo, slug, temporary: (
            calls.append(("finalize", temporary))
            or {"run_id": f"{slug}-r001-local", "experiment": slug}
        ),
    )
    monkeypatch.setattr(
        run_dirs,
        "materialize_run",
        lambda _root, run_id, _artifacts: calls.append(("materialize", run_id)),
    )

    result = run_dirs.finalize_prepared_run("exp001", "r001")
    assert result == run_dirs.FIGURES_ROOT / "exp001"
    assert (figures / "_manifest.json").is_file()
    assert len(calls) == 2
    assert calls[0] == ("finalize", figures.parent)
    assert calls[1] == ("materialize", "exp001-r001-local")


def test_legacy_success_finalizer_skips_isolated_campaign(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        run_dirs, "runner_paths", lambda _slug: SimpleNamespace(isolated=True)
    )
    monkeypatch.setattr(
        run_dirs,
        "finalize_local_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    assert run_dirs.finalize_prepared_run("exp001", "r001") is None


def test_legacy_runner_failure_keeps_hidden_run_for_postmortem(
    tmp_path: Path,
    monkeypatch,
) -> None:
    figures = tmp_path / ".artifacts/exp001"
    figures.mkdir(parents=True)
    (figures / "result.txt").write_text("accepted")
    monkeypatch.setattr(
        run_dirs, "artifacts_and_figures", lambda _slug: (tmp_path / "scratch", figures)
    )

    @run_dirs.preserve_active_view("exp001")
    def failing_run() -> None:
        (figures / "result.txt").write_text("partial")
        (figures / "new.txt").write_text("partial")
        raise RuntimeError("failed")

    with pytest.raises(RuntimeError, match="failed"):
        failing_run()
    assert (figures / "result.txt").read_text() == "partial"
    assert (figures / "new.txt").read_text() == "partial"
    assert not figures.with_name("exp001.pre-run").exists()


def test_local_prepare_reuses_log_created_inside_exact_hidden_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    temporary = tmp_path / ".pingstore/runs/.exp001-r001-local.tmp"
    log = temporary / "export/state/logs/exp001.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text('{"event":"started"}\n')
    monkeypatch.setattr(run_dirs, "REPO", tmp_path)
    monkeypatch.setattr(run_dirs, "FIGURES_ROOT", tmp_path / ".artifacts")
    monkeypatch.setattr(
        run_dirs,
        "runner_paths",
        lambda _slug: SimpleNamespace(isolated=False),
    )

    def fake_manifest(path: Path, **kwargs: object) -> None:
        (path / "_manifest.json").write_text(json.dumps(kwargs))

    monkeypatch.setattr(run_dirs, "write_manifest", fake_manifest)
    state, files = run_dirs.prepare_staged("exp001", "r001")

    assert files == temporary / "presentation"
    assert state == temporary / "export/state"
    assert log.read_text() == '{"event":"started"}\n'


def test_direct_v2_run_reuse_and_selection(tmp_path, monkeypatch):
    from experiments.helpers import paths, provenance
    from pingstore.contracts import validate_run_directory

    registry = tmp_path / "experiments/collections/registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({
        "schema": "pingstore.experiment-registry/v1",
        "experiments": {"exp001": "demo"}, "historical": {},
    }))
    for module in (paths, run_dirs):
        monkeypatch.setattr(module, "REPO", tmp_path)
        monkeypatch.setattr(module, "FIGURES_ROOT", tmp_path / ".artifacts")
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / ".pingstore/runs")
    monkeypatch.setattr(provenance, "git_state", lambda: ("abc123", False))
    monkeypatch.setattr(provenance, "_code_dirty", lambda: False)
    with run_dirs.published_run("exp001", "r001") as (state, presentation):
        (state / "weights.pt").write_bytes(b"trained")
        paths.log_runner_event("exp001", "completed", run_id="r001")
        (presentation / "plot.svg").write_text("<svg/>")
    first = tmp_path / ".pingstore/runs/exp001-r001-local"
    original = validate_run_directory(first)
    assert (first / "export/state/logs/exp001.jsonl").is_file()
    with pytest.raises(RuntimeError, match="before finalizing"):
        paths.log_runner_event("exp001", "completed", run_id="r001")
    assert paths.active_run_state("exp001") == first / "export/state"
    assert not (tmp_path / ".artifacts/exp001/run.sh").exists()
    assert paths.current_run_number("exp001") == 1
    with run_dirs.published_run("exp001", "r002", plot_only=True) as (state, presentation):
        assert (state / "weights.pt").read_bytes() == b"trained"
        assert (presentation / "plot.svg").read_text() == "<svg/>"
        (presentation / "plot.svg").write_text("<svg>new</svg>")
    assert validate_run_directory(first) == original
    assert paths.active_run_state("exp001").parent.parent.name == "exp001-r002-local"


def test_direct_v2_nested_presentation_never_publishes(tmp_path, monkeypatch):
    from experiments.helpers import paths, provenance
    from pingstore.contracts import PingstoreError

    registry = tmp_path / "experiments/collections/registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({
        "schema": "pingstore.experiment-registry/v1",
        "experiments": {"exp001": "demo"}, "historical": {},
    }))
    for module in (paths, run_dirs):
        monkeypatch.setattr(module, "REPO", tmp_path)
        monkeypatch.setattr(module, "FIGURES_ROOT", tmp_path / ".artifacts")
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / ".pingstore/runs")
    monkeypatch.setattr(provenance, "git_state", lambda: ("abc123", False))
    monkeypatch.setattr(provenance, "_code_dirty", lambda: False)
    with pytest.raises(PingstoreError, match="flat"):
        with run_dirs.published_run("exp001", "r001") as (_state, presentation):
            (presentation / "nested").mkdir()
    assert (tmp_path / ".pingstore/runs/.exp001-r001-local.tmp").is_dir()
    assert not (tmp_path / ".pingstore/runs/exp001-r001-local").exists()
    assert not (tmp_path / ".artifacts/exp001").exists()


def test_explicitly_selected_compact_bank_resolves_declared_export_root(tmp_path, monkeypatch):
    from experiments.helpers import paths
    from pingstore.contracts import payload_digest, write_json_atomic
    from pingstore.native import capture_local_run

    registry = tmp_path / "experiments/collections/registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({
        "schema": "pingstore.experiment-registry/v1",
        "experiments": {"exp001": "demo"}, "historical": {},
    }))
    staging = tmp_path / "input"
    staging.mkdir()
    (staging / "_manifest.json").write_text(json.dumps({"run_id": "r001", "host": "local"}))
    run = capture_local_run(tmp_path, "exp001", staging)
    directory = tmp_path / ".pingstore/runs" / run["run_id"]
    # Complete the synthetic compact-export fixture before using it as evidence.
    (directory / "export/cells/cell-a").mkdir(parents=True)
    (directory / "export/cells/cell-a/config.json").write_text("{}")
    run["export_root"] = "export/cells"
    run["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", run)
    active = tmp_path / ".artifacts/exp001"
    active.mkdir(parents=True)
    (active / "_manifest.json").write_text(json.dumps({"pingstore_run_id": run["run_id"]}))
    monkeypatch.setattr(paths, "REPO", tmp_path)
    monkeypatch.setattr(paths, "FIGURES_ROOT", tmp_path / ".artifacts")
    monkeypatch.setattr(paths, "RUNS_ROOT", tmp_path / ".pingstore/runs")
    assert paths.active_run_state("exp001") == directory / "export/cells"
