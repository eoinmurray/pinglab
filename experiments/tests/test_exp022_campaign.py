from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments import exp022
from experiments import exp022_campaign as campaign


CONCRETE_TIERS = ("standard", "fine_dt", "canonical_coba", "canonical_ping", "variable_rate")


def test_registry_has_90_unique_cells_partitioned_once() -> None:
    names = [cell["name"] for cell in exp022.CANONICAL_CELLS]
    assert len(names) == len(set(names)) == 90
    tiered = [cell["name"] for tier in CONCRETE_TIERS for cell in exp022.cells_in_resource_tier(tier)]
    assert sorted(tiered) == sorted(names)


@pytest.mark.parametrize("family,run_id", exp022.TRAINING_RUN_IDS.items())
def test_registry_training_run_identity(family: str, run_id: str) -> None:
    cells = [cell for cell in exp022.CANONICAL_CELLS if cell["family"] == family]
    assert cells
    assert {cell["training_run_id"] for cell in cells} == {run_id}


def test_all_resolved_commands_keep_family_contract(tmp_path: Path) -> None:
    for cell in exp022.CANONICAL_CELLS:
        samples, epochs = exp022.cell_samples_epochs(cell)
        args = exp022.build_train_args(cell, tmp_path / cell["name"], samples, epochs)
        assert args[args.index("--epochs") + 1] == "50"
        assert args[args.index("--seed") + 1] == str(cell["seed"])
        assert args[args.index("--dt") + 1] == str(cell["dt_ms"])
        assert args[args.index("--tau-gaba") + 1] == str(cell["tau_gaba"])
        if cell["family"] == "canonical":
            assert args[args.index("--max-samples") + 1] == "70000"
        else:
            assert args[args.index("--max-samples") + 1] == "7000"
        if cell["family"] == "variable_rate":
            assert args[args.index("--readout") + 1] == "spike-rate"
            assert args[args.index("--input-rates") + 1:] == ["0.5", "1.0", "2.0", "5.0", "10.0", "25.0"]


def _manifest_cell(tmp_path: Path, *, epochs: int = 2, samples: int = 100) -> dict:
    directory = tmp_path / "cells" / "ping__variable_rate__seed42"
    return {
        "name": "ping__variable_rate__seed42",
        "training_run_id": "TR-06",
        "resource_tier": "variable_rate",
        "output_directory": str(directory),
        "parameters": {
            "epochs": epochs,
            "max_samples": samples,
            "arguments": {
                "--model": "ping", "--dataset": "mnist", "--epochs": str(epochs),
                "--max-samples": str(samples), "--dt": "0.1", "--t-ms": "200.0",
                "--tau-gaba": "6.0", "--seed": "42", "--readout": "spike-rate",
                "--input-rates": ["0.5", "1.0", "2.0", "5.0", "10.0", "25.0"],
            },
        },
    }


def _write_valid_cell(row: dict) -> Path:
    directory = Path(row["output_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    expected = campaign._expected_config(row)
    identity = {
        "training_cell_name": row["name"],
        "training_run_id": row["training_run_id"],
        "campaign_resolved_parameters": row["parameters"],
    }
    (directory / "config.json").write_text(json.dumps({**expected, **identity}))
    (directory / "metrics.json").write_text(json.dumps({**identity, "config": expected}))
    epochs = row["parameters"]["epochs"]
    samples = row["parameters"]["max_samples"]
    (directory / "metrics.jsonl").write_text("\n".join(
        json.dumps({"ep": epoch, "samples": samples, "acc": 10.0})
        for epoch in range(1, epochs + 1)
    ) + "\n")
    torch.save({
        "b_out": torch.ones(10),
        "W_ff.0": torch.ones(784, 1024),
        "W_ff.1": torch.ones(1024, 10),
        "W_ei.1": torch.ones(1024, 256),
        "W_ie.1": torch.ones(256, 1024),
    }, directory / "weights.pth")
    return directory


def test_validator_states_and_valid_checkpoint(tmp_path: Path) -> None:
    row = _manifest_cell(tmp_path)
    assert campaign.validate_cell(row)["state"] == "missing"
    directory = Path(row["output_directory"])
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    assert campaign.validate_cell(row)["state"] == "partial"
    directory.rename(tmp_path / "discarded")
    _write_valid_cell(row)
    assert campaign.validate_cell(row) == {"valid": True, "state": "complete", "reasons": []}


def test_validator_rejects_corrupt_mismatched_and_short_history(tmp_path: Path) -> None:
    row = _manifest_cell(tmp_path)
    directory = _write_valid_cell(row)
    (directory / "weights.pth").write_bytes(b"not a checkpoint")
    assert any("checkpoint load failed" in reason for reason in campaign.validate_cell(row)["reasons"])
    _write_valid_cell(row)
    config = json.loads((directory / "config.json").read_text())
    config["seed"] = 44
    (directory / "config.json").write_text(json.dumps(config))
    assert any("seed mismatch" in reason for reason in campaign.validate_cell(row)["reasons"])
    config["seed"] = 42
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "metrics.jsonl").write_text(json.dumps({"ep": 1, "samples": 100}) + "\n")
    assert any("epoch 2" in reason for reason in campaign.validate_cell(row)["reasons"])


def test_preserve_partial_never_overwrites(tmp_path: Path) -> None:
    row = _manifest_cell(tmp_path)
    directory = Path(row["output_directory"])
    directory.mkdir(parents=True)
    (directory / "broken.txt").write_text("evidence")
    preserved = campaign.preserve_partial(directory)
    assert preserved is not None and (preserved / "broken.txt").read_text() == "evidence"
    assert not directory.exists()


def test_status_identifies_retry_cells(tmp_path: Path) -> None:
    complete = _manifest_cell(tmp_path)
    missing = {**_manifest_cell(tmp_path), "name": "missing", "output_directory": str(tmp_path / "cells" / "missing")}
    _write_valid_cell(complete)
    status = campaign.summarize_status({"campaign_id": "test", "cells": [complete, missing]})
    assert status["counts"] == {"complete": 1, "missing": 1}
    assert status["retry_cells"] == ["missing"]


def test_campaign_train_does_not_touch_valid_cell(tmp_path: Path, monkeypatch) -> None:
    row = _manifest_cell(tmp_path)
    directory = _write_valid_cell(row)
    before = campaign.sha256_file(directory / "weights.pth")
    manifest = {
        "campaign_id": "test", "manifest_sha256": "abc",
        "repository": {"commit": "deadbeef", "dirty": False},
        "campaign_root": str(tmp_path), "cells": [row],
    }
    monkeypatch.setattr(exp022, "_checked_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        exp022.subprocess, "run",
        lambda *_args, **_kwargs: pytest.fail("valid cell must not launch training"),
    )
    assert exp022._campaign_train(tmp_path / "campaign.json", row["name"]) == 0
    assert campaign.sha256_file(directory / "weights.pth") == before
