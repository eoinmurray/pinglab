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
    checkpoint_policy,
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


def test_checkpoint_policy_is_purpose_based_and_fail_closed() -> None:
    assert checkpoint_policy("endpoint_dynamics") == {
        "purpose": "endpoint_dynamics",
        "role": "final_epoch",
    }
    assert checkpoint_policy("deployment_performance") == {
        "purpose": "deployment_performance",
        "role": "best_validation",
    }
    with pytest.raises(ValueError, match="unknown checkpoint purpose"):
        checkpoint_policy("exp042")


def test_collection_runners_derive_one_role_from_their_analysis_purpose() -> None:
    endpoint = (exp025, exp041, exp042, exp044, exp046, exp049)
    deployment = (exp037, exp038, exp082)
    for module in endpoint:
        assert module.ANALYSIS_PURPOSE == "endpoint_dynamics"
        assert module.CHECKPOINT_POLICY == checkpoint_policy(module.ANALYSIS_PURPOSE)
        assert module.CHECKPOINT_ROLE == module.CHECKPOINT_POLICY["role"]
    for module in deployment:
        assert module.ANALYSIS_PURPOSE == "deployment_performance"
        assert module.CHECKPOINT_POLICY == checkpoint_policy(module.ANALYSIS_PURPOSE)
        assert module.CHECKPOINT_ROLE == module.CHECKPOINT_POLICY["role"]


def test_exp044_command_loads_final_checkpoint(tmp_path: Path) -> None:
    from experiments.exp044.recipe import inference_args

    bank = _checkpoint_bank(tmp_path / "cell")
    checkpoint = resolve_checkpoint(bank, exp044.CHECKPOINT_ROLE)
    command = inference_args(bank, checkpoint["path"], tmp_path / "infer", samples=1000)
    assert Path(command[command.index("--load-weights") + 1]) == (bank / "weights_final.pth").resolve()
    assert command[command.index("--max-samples") + 1] == "1000"


def test_epoch_metrics_prefers_complete_record_and_supports_legacy_jsonl(tmp_path):
    from experiments.helpers.checkpoints import epoch_metrics

    jsonl = tmp_path / "metrics.jsonl"
    jsonl.write_text(json.dumps({"ep": 1, "acc": 10, "timestamp": "then"}) + "\n")
    assert epoch_metrics(tmp_path)[0]["acc"] == 10
    complete = [{"ep": 1, "acc": 20, "grad_norms": {"W": 2.0}}]
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({"epochs": complete}))
    assert epoch_metrics(tmp_path) == complete
    jsonl.unlink()
    assert epoch_metrics(tmp_path) == complete
    metrics.write_text(json.dumps({"epochs": "invalid"}))
    with pytest.raises(RuntimeError, match="invalid epoch"):
        epoch_metrics(tmp_path)


def test_exp022_and_exp049_read_compact_epoch_records(tmp_path, monkeypatch):
    from experiments import exp022

    rows = [{"ep": 1, "acc": 80, "rate_e": 5, "rate_i": 20, "contrast": 0.4},
            {"ep": 2, "acc": 90, "test_rate_e": 6, "test_rate_i": 21, "contrast": 0.6}]
    (tmp_path / "metrics.json").write_text(json.dumps({"epochs": rows}))
    assert exp022.training_curve(tmp_path) == ([1, 2], [80.0, 90.0])
    assert exp022.final_rates(tmp_path) == (6.0, 21.0)
    monkeypatch.setattr(exp049, "COND_ORDER", ["frozen_ping"])
    monkeypatch.setattr(exp049, "SEEDS", [42])
    monkeypatch.setattr(exp049, "cell_dir", lambda *_: tmp_path)
    curves = exp049._load_epoch_curves()[tmp_path.name]
    assert curves["ep"] == [1, 2]
    assert curves["rate_e"] == [5, 6]
    assert curves["contrast"] == [0.4, 0.6]


def test_exp025_raster_replay_never_modifies_upstream_bank(tmp_path, monkeypatch):
    bank = _checkpoint_bank(tmp_path / "bank")
    (bank / "config.json").write_text("{}")
    legacy = bank / "infer/snapshot.npz"
    legacy.parent.mkdir()
    legacy.write_bytes(b"original snapshot")
    original = {p.relative_to(bank): p.read_bytes() for p in bank.rglob("*") if p.is_file()}
    state = tmp_path / "exp025-state"
    monkeypatch.setattr(exp025, "ARTIFACTS", state)
    monkeypatch.setattr(exp025, "baseline_dir", lambda *_: bank)

    def infer(command):
        out = Path(command[command.index("--out-dir") + 1])
        assert state in out.parents
        (out / "snapshot.npz").write_bytes(b"new analysis snapshot")

    monkeypatch.setattr(exp025, "run_cli", infer)
    monkeypatch.setattr(exp025, "render_raster", lambda *_: None)
    exp025.generate_raster("coba", tmp_path / "raster.png")
    assert exp025.raster_snapshot("coba").read_bytes() == b"new analysis snapshot"
    assert {p.relative_to(bank): p.read_bytes() for p in bank.rglob("*") if p.is_file()} == original
