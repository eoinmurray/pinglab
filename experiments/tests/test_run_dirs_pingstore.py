from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.helpers import run_dirs


def test_legacy_success_finalizer_captures_once(
    tmp_path: Path, monkeypatch,
) -> None:
    figures = tmp_path / "artifacts/data/exp001"
    figures.mkdir(parents=True)
    (figures / "numbers.json").write_text("{}\n")
    calls: list[list[str]] = []

    monkeypatch.setattr(
        run_dirs, "artifacts_and_figures", lambda _slug: (tmp_path / "scratch", figures)
    )
    monkeypatch.setattr(
        run_dirs, "runner_paths", lambda _slug: SimpleNamespace(isolated=False)
    )

    def fake_manifest(path: Path, **kwargs: object) -> None:
        (path / "_manifest.json").write_text(json.dumps(kwargs))

    monkeypatch.setattr(run_dirs, "write_manifest", fake_manifest)
    monkeypatch.setattr(
        run_dirs.subprocess,
        "run",
        lambda command, **_kwargs: calls.append(command),
    )

    result = run_dirs.finalize_prepared_run("exp001", "r001")
    assert result == figures
    assert (figures / "_manifest.json").is_file()
    assert len(calls) == 1
    assert calls[0][-2:] == ["--staging", str(figures)]


def test_legacy_success_finalizer_skips_isolated_campaign(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        run_dirs, "runner_paths", lambda _slug: SimpleNamespace(isolated=True)
    )
    monkeypatch.setattr(
        run_dirs.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    assert run_dirs.finalize_prepared_run("exp001", "r001") is None


def test_legacy_runner_failure_restores_active_view(
    tmp_path: Path, monkeypatch,
) -> None:
    figures = tmp_path / "artifacts/data/exp001"
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
    assert (figures / "result.txt").read_text() == "accepted"
    assert not (figures / "new.txt").exists()
    assert not figures.with_name("exp001.pre-run").exists()
