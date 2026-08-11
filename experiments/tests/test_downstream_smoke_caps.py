from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from experiments import exp041, exp042, exp044, exp049


@pytest.mark.parametrize("module", [exp041, exp044, exp049])
def test_quantitative_inference_is_capped_in_smoke(
    module, monkeypatch, tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "config.json").write_text(json.dumps({"tau_gaba_ms": 6.0}))
    (train_dir / "weights.pth").touch()
    observed: list[str] = []

    monkeypatch.setattr(module, "ARTIFACTS", tmp_path / "derived")
    monkeypatch.setattr(module, "SMOKE", True)
    monkeypatch.setattr(module, "run_cli", lambda cmd: observed.extend(cmd))
    module._infer_cell(train_dir, ["--outputs", "pop_traces"], "infer")

    assert observed[observed.index("--max-samples") + 1] == "100"


@pytest.mark.parametrize("module", [exp041, exp044, exp049])
def test_single_sample_inference_does_not_restrict_sample_index(
    module, monkeypatch, tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "config.json").write_text(json.dumps({"tau_gaba_ms": 6.0}))
    (train_dir / "weights.pth").touch()
    observed: list[str] = []

    monkeypatch.setattr(module, "ARTIFACTS", tmp_path / "derived")
    monkeypatch.setattr(module, "SMOKE", True)
    monkeypatch.setattr(module, "run_cli", lambda cmd: observed.extend(cmd))
    module._infer_cell(train_dir, ["--sample-index", "50"], "snapshot")

    assert "--max-samples" not in observed


def test_exp042_baseline_inference_is_capped_in_smoke(
    monkeypatch, tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "config.json").write_text("{}")
    (train_dir / "weights.pth").touch()
    observed: list[str] = []

    def fake_run(cmd: list[str]) -> None:
        observed.extend(cmd)
        out_dir = Path(cmd[cmd.index("--out-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text("{}")
        np.savez(out_dir / "rasters.npz", n_trials=np.int32(0))

    exp042._BASE_CACHE.clear()
    monkeypatch.setattr(exp042, "ARTIFACTS", tmp_path / "derived")
    monkeypatch.setattr(exp042, "SMOKE", True)
    monkeypatch.setattr(exp042, "run_cli", fake_run)
    exp042._run_baseline(train_dir)

    assert observed[observed.index("--max-samples") + 1] == "100"
