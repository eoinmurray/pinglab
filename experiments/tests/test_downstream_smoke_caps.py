from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from experiments import (
    exp025,
    exp037,
    exp038,
    exp041,
    exp042,
    exp044,
    exp046,
    exp049,
)
from experiments.exp042 import recipe as exp042_recipe
from experiments.exp042 import simulation as exp042_simulation
from experiments.helpers.checkpoints import sha256_file
from experiments.helpers.datasets import MNIST_REDUCED_EVAL_SAMPLES


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


@pytest.mark.parametrize("module", [exp041, exp049])
def test_quantitative_inference_is_capped_in_smoke(
    module, monkeypatch, tmp_path: Path,
) -> None:
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    _write_final_checkpoint(train_dir, {"tau_gaba_ms": 6.0})
    observed: list[str] = []

    monkeypatch.setattr(module, "ARTIFACTS", tmp_path / "derived")
    monkeypatch.setattr(module, "SMOKE", True)
    monkeypatch.setattr(module, "EVAL_MAX_SAMPLES", 100)
    monkeypatch.setattr(module, "run_cli", lambda cmd: observed.extend(cmd))
    module._infer_cell(train_dir, ["--outputs", "pop_traces"], "infer")

    assert observed[observed.index("--max-samples") + 1] == "100"


@pytest.mark.parametrize("module", [exp041, exp049])
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

    monkeypatch.setattr(exp042_simulation, "run_cli", fake_run)
    simulator = exp042_simulation.Simulator(tmp_path / "scratch", tmp_path / "commands",
                                          exp042_recipe.configuration(smoke=True))
    simulator._run_baseline(train_dir)

    assert observed[observed.index("--max-samples") + 1] == "100"


def test_exp042_production_inference_uses_publication_subset(
    monkeypatch, tmp_path
) -> None:
    observed: list[str] = []

    def fake_run(command):
        observed.extend(command)
        out = Path(command[command.index("--out-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text("{}")

    train_dir = tmp_path / "cell"
    train_dir.mkdir()
    _write_final_checkpoint(train_dir, {})
    monkeypatch.setattr(exp042_simulation, "run_cli", fake_run)
    simulator = exp042_simulation.Simulator(tmp_path / "scratch", tmp_path / "commands",
                                          exp042_recipe.configuration())
    simulator._run_with_override(train_dir, tmp_path / "override.npz")
    assert observed[observed.index("--max-samples") + 1] == "1000"


def test_exp042_override_file_is_deleted_after_inference(
    monkeypatch, tmp_path: Path,
) -> None:
    train_dir = tmp_path / "cell"
    train_dir.mkdir()
    (train_dir / "config.json").write_text('{"dt": 0.1}')
    simulator = exp042_simulation.Simulator(tmp_path, tmp_path / "commands",
                                          exp042_recipe.configuration())
    monkeypatch.setattr(simulator, "_run_baseline", lambda _path: ({}, {}))
    seen: list[Path] = []

    def fake_build(_rasters, _condition, _generator, _dt, path) -> None:
        path.write_bytes(b"override")

    def fake_run(_train_dir, path) -> dict:
        assert path.exists()
        seen.append(path)
        return {"best_acc": 90.0, "rates_hz": {}, "n_total": 1000}

    monkeypatch.setattr(simulator, "_build_override_file", fake_build)
    monkeypatch.setattr(simulator, "_run_with_override", fake_run)
    simulator.evaluate(train_dir, {"id": "fixture", "condition": "phase_shuffled_i", "seed_offset": 42})

    assert len(seen) == 1
    assert not seen[0].exists()


@pytest.mark.parametrize(
    "module",
    [exp025, exp037, exp038, exp041, exp042, exp044, exp046, exp049],
)
def test_reduced_pool_downstream_evaluation_contract(module) -> None:
    assert module.EVAL_MAX_SAMPLES == MNIST_REDUCED_EVAL_SAMPLES == 1000


def test_smoke_grids_retain_every_writeup_anchor() -> None:
    assert {1.0, 3.0} <= set(exp025.W_IN_SCALE_VALUES)
    assert {0.0, 0.8, 1.0} <= set(exp037.PERTURB_DROP_LEVELS)
    assert {0.0, 14.0, 100.0} <= set(exp042.JITTER_SIGMAS_MS)
    assert {0.0, 0.5, 1.0, 2.0, 5.0, 9.0, 14.0} <= set(
        exp042.CELL_JITTER_SIGMAS_MS
    )
