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
    assert paths.state.parts[-2:] == ("files", "state")
    assert paths.derived == paths.state.parent
    assert "temp/experiments" not in paths.state.as_posix()


def test_legacy_success_finalizer_captures_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    figures = tmp_path / ".pingstore/runs/.exp001-r001-local.tmp/files"
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
    log = temporary / "files/state/logs/exp001.jsonl"
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

    assert files == temporary / "files"
    assert state == files / "state"
    assert log.read_text() == '{"event":"started"}\n'
