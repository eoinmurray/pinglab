from __future__ import annotations

import json
from pathlib import Path

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
    exp082,
)
from experiments.helpers.checkpoints import (
    cache_tag,
    public_provenance,
    resolve_checkpoint,
    sha256_file,
    training_horizon,
)


def _checkpoint_bank(root: Path) -> Path:
    root.mkdir()
    selected = root / "weights.pth"
    final = root / "weights_final.pth"
    selected.write_bytes(b"selected")
    final.write_bytes(b"final")
    (root / "metrics.json").write_text(json.dumps({
        "training_cell_name": "cell-a",
        "best_epoch": 3,
        "config": {"epochs": 5},
        "checkpoints": {
            "best_validation": {
                "filename": selected.name,
                "epoch": 3,
                "sha256": sha256_file(selected),
            },
            "final_epoch": {
                "filename": final.name,
                "epoch": 5,
                "sha256": sha256_file(final),
            },
        },
    }))
    return root


def test_resolver_distinguishes_selected_and_final_checkpoints(tmp_path: Path) -> None:
    bank = _checkpoint_bank(tmp_path / "cell")
    selected = resolve_checkpoint(bank, "best_validation")
    final = resolve_checkpoint(bank, "final_epoch")
    assert selected["path"].name == "weights.pth"
    assert selected["epoch"] == 3
    assert final["path"].name == "weights_final.pth"
    assert final["epoch"] == 5
    assert public_provenance(final)["training_cell"] == "cell-a"
    assert cache_tag(selected) != cache_tag(final)


def test_resolver_rejects_checkpoint_hash_drift(tmp_path: Path) -> None:
    bank = _checkpoint_bank(tmp_path / "cell")
    (bank / "weights_final.pth").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        resolve_checkpoint(bank, "final_epoch")


def test_training_horizon_is_read_from_upstream_configs(tmp_path: Path) -> None:
    banks = [_checkpoint_bank(tmp_path / f"cell-{index}") for index in range(2)]
    assert training_horizon(banks) == 5

    metrics_path = banks[1] / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["config"]["epochs"] = 50
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(RuntimeError, match="mixed upstream training horizons"):
        training_horizon(banks)


def test_collection_checkpoint_roles_are_explicit() -> None:
    assert {
        exp025.CHECKPOINT_ROLE,
        exp041.CHECKPOINT_ROLE,
        exp042.CHECKPOINT_ROLE,
        exp044.CHECKPOINT_ROLE,
        exp046.CHECKPOINT_ROLE,
        exp049.CHECKPOINT_ROLE,
    } == {"final_epoch"}
    assert {
        exp037.CHECKPOINT_ROLE,
        exp038.CHECKPOINT_ROLE,
        exp082.CHECKPOINT_ROLE,
    } == {"best_validation"}


def test_exp044_command_loads_final_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank = _checkpoint_bank(tmp_path / "cell")
    (bank / "config.json").write_text(json.dumps({"dt": 0.1, "t_ms": 200.0}))
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        commands.append(command)
        out = Path(command[command.index("--out-dir") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps({
            "best_acc": 10.0,
            "n_total": 1,
            "rates_hz": {},
        }))

    monkeypatch.setattr(exp044, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(exp044, "run_cli", fake_run)
    exp044.measure_rate_acc(bank)
    loaded = Path(commands[0][commands[0].index("--load-weights") + 1])
    assert loaded == (bank / "weights_final.pth").resolve()
