from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from experiments import exp025, exp037, exp041, exp042, exp044, exp049
from experiments.helpers.checkpoints import sha256_file


def _write_final_checkpoint(train_dir: Path, config: dict) -> None:
    (train_dir / "config.json").write_text(json.dumps(config))
    checkpoint = train_dir / "weights_final.pth"
    checkpoint.write_bytes(b"final")
    (train_dir / "metrics.json").write_text(json.dumps({
        "config": {"epochs": 50},
        "checkpoints": {
            "final_epoch": {
                "filename": checkpoint.name,
                "epoch": 50,
                "sha256": sha256_file(checkpoint),
            }
        },
    }))


@pytest.mark.parametrize("module", [exp041, exp044, exp049])
def test_quantitative_inference_is_capped_in_smoke(
    module, monkeypatch, tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    _write_final_checkpoint(train_dir, {"tau_gaba_ms": 6.0})
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
    _write_final_checkpoint(train_dir, {"tau_gaba_ms": 6.0})
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
    _write_final_checkpoint(train_dir, {})
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


def test_smoke_grids_retain_every_writeup_anchor() -> None:
    assert {1.0, 3.0} <= set(exp025.W_IN_SCALE_VALUES)
    assert {0.0, 0.8, 1.0} <= set(exp037.PERTURB_DROP_LEVELS)
    assert {0.0, 14.0, 100.0} <= set(exp042.JITTER_SIGMAS_MS)
    assert {0.0, 0.5, 1.0, 2.0, 5.0, 9.0, 14.0} <= set(
        exp042.CELL_JITTER_SIGMAS_MS
    )


def test_exp042_json_output_replaces_nonfinite_fit_values() -> None:
    value = exp042._json_safe({"r2": float("nan"), "nested": [float("inf"), 1.0]})
    assert value == {"r2": None, "nested": [None, 1.0]}
