from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from experiments.helpers.paths import (
    DERIVED_ENV,
    LOG_ENV,
    REQUIRE_ISOLATED_ENV,
    STATE_ENV,
    runner_paths,
)

REPO = Path(__file__).resolve().parents[2]


def _runstore(*args: str | Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-m", "tools.runstore", *(str(arg) for arg in args)],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for item in sorted(root.rglob("*")):
        if item.is_file():
            digest.update(item.relative_to(root).as_posix().encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()


def _write_exp024_upstream(root: Path) -> None:
    epochs = [
        {
            "ep": epoch,
            "acc": 70.0 + epoch,
            "loss": 1.0 / epoch,
            "test_loss": 1.2 / epoch,
            "test_rate_e": 5.0 + epoch,
            "test_rate_i": 8.0 + epoch,
            "act": 0.2,
            "grad_norm": 0.1,
        }
        for epoch in range(1, 4)
    ]
    for model in ("coba", "ping"):
        for seed in (42, 43, 44):
            cell = root / f"{model}__off__seed{seed}"
            cell.mkdir(parents=True)
            (cell / "metrics.json").write_text(json.dumps({"epochs": epochs}))
            (cell / "config.json").write_text(json.dumps({"git_sha": "test"}))


def test_runner_paths_require_complete_absolute_triplet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(REQUIRE_ISOLATED_ENV, "1")
    monkeypatch.setenv(STATE_ENV, str(tmp_path / "state"))
    with pytest.raises(RuntimeError, match="all-or-none"):
        runner_paths("exp024")

    monkeypatch.setenv(DERIVED_ENV, str(tmp_path / "derived"))
    monkeypatch.setenv(LOG_ENV, str(tmp_path / "logs"))
    paths = runner_paths("exp024")
    assert paths.isolated
    assert paths.state == (tmp_path / "state").resolve()
    assert paths.derived == (tmp_path / "derived").resolve()
    assert paths.logs == (tmp_path / "logs").resolve()


def test_runner_paths_reject_repository_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(STATE_ENV, str(tmp_path / "state"))
    monkeypatch.setenv(DERIVED_ENV, str(REPO / "artifacts/data/exp024"))
    monkeypatch.setenv(LOG_ENV, str(tmp_path / "logs"))
    with pytest.raises(RuntimeError, match="repository artifacts"):
        runner_paths("exp024")


def test_exp024_isolated_run_cannot_modify_active_artifacts(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    _runstore(
        "init",
        campaign,
        "--run-id",
        "representative-campaign",
        "--kind",
        "campaign",
        "--collection",
        "gamma-gated-sparsity",
        "--command",
        sys.executable,
        "experiments/exp024.py",
    )
    state = campaign / "downstream" / "exp024"
    derived = campaign / "derived" / "artifacts" / "data" / "exp024"
    logs = campaign / "logs" / "exp024"
    upstream = campaign / "exp022" / "cells"
    state.mkdir(parents=True)
    _write_exp024_upstream(upstream)

    active = REPO / "artifacts" / "data" / "exp024"
    before = _tree_digest(active)
    environment = os.environ.copy()
    environment.update(
        {
            REQUIRE_ISOLATED_ENV: "1",
            STATE_ENV: str(state),
            DERIVED_ENV: str(derived),
            LOG_ENV: str(logs),
            "PINGLAB_TRAINING_ROOT": str(upstream),
            "MPLBACKEND": "Agg",
        }
    )

    result = subprocess.run(
        [sys.executable, str(REPO / "experiments" / "exp024.py")],
        cwd=REPO,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (derived / "numbers.json").is_file()
    assert (derived / "coba_curves.svg").is_file()
    assert (derived / "ping_curves.svg").is_file()
    events = [
        json.loads(line) for line in (logs / "exp024.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["started", "completed"]
    assert all(event.get("isolated", True) for event in events)
    assert _tree_digest(active) == before

    store = tmp_path / "store"
    _runstore("inspect", campaign, "--finalize")
    _runstore(
        "archive",
        campaign,
        "--archive-id",
        "representative-campaign",
        "--store",
        store,
    )
    restored = tmp_path / "restored"
    _runstore("restore", "representative-campaign", restored, "--store", store)
    promoted = tmp_path / "promoted"
    _runstore("promote", campaign, "exp024", "--artifacts-root", promoted)

    reverse_link = json.loads((promoted / "exp024" / "_provenance.json").read_text())
    assert reverse_link["campaign_id"] == "representative-campaign"
    assert reverse_link["archive"]["archive_id"] == "representative-campaign"
    assert (restored / "derived/artifacts/data/exp024/numbers.json").is_file()
    assert _tree_digest(active) == before
