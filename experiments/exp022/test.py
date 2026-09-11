from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from experiments.exp022 import compute as exp022
from experiments.exp022 import recipe
from experiments.exp022.analyse import _gamma_psd, bank_composition, measure_snapshot
from experiments.helpers import rhythmicity as figure_metrics
from pingstore.contracts import PingstoreError

CONCRETE_TIERS = (
    "standard",
    "fine_dt",
    "canonical_coba",
    "canonical_ping",
    "variable_rate",
)


def test_registry_has_102_unique_cells_partitioned_once() -> None:
    names = [cell["name"] for cell in recipe.CANONICAL_CELLS]
    assert len(names) == len(set(names)) == 102
    tiered = [
        cell["name"]
        for tier in CONCRETE_TIERS
        for cell in recipe.cells_in_resource_tier(tier)
    ]
    assert sorted(tiered) == sorted(names)


def test_tr02_registry_uses_explicit_hz_targets() -> None:
    cells = [
        cell for cell in recipe.CANONICAL_CELLS if cell["training_run_id"] == "TR-02"
    ]
    assert {cell["rate_target_hz"] for cell in cells} == {
        None,
        25.0,
        10.0,
        5.0,
        2.5,
        1.0,
    }
    for cell in cells:
        args = cell["extra"]
        if cell["rate_target_hz"] is None:
            assert "--fr-reg-upper-target-hz" not in args
        else:
            assert "--fr-reg-upper-target-hz" in args
            assert "--fr-reg-upper-strength" in args
            strength = args[args.index("--fr-reg-upper-strength") + 1]
            assert strength == "0.041"


def test_downstream_contract_interface_is_isolated_and_fail_closed() -> None:
    cells = recipe.training_run_cells("TR-06")
    assert len(cells) == 3
    cells[0]["input_rates_hz"].append(999.0)
    assert 999.0 not in recipe.training_run_cell("TR-06", seed=42)["input_rates_hz"]
    assert recipe.training_run_values("TR-06", "seed") == (42, 43, 44)
    with pytest.raises(ValueError, match="unknown exp022 training-run ID"):
        recipe.training_run_cells("TR-99")
    with pytest.raises(ValueError, match="expected one TR-02 cell"):
        recipe.training_run_cell("TR-02", seed=42)
    with pytest.raises(ValueError, match="cell contract mismatch"):
        recipe.require_training_run_cells("TR-06", {"invented-cell"})


def test_bank_python_identity_stays_inside_environment(
    monkeypatch, tmp_path: Path
) -> None:
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text("")
    monkeypatch.setattr(exp022.sys, "executable", str(bin_dir / "python3"))
    assert exp022.python_executable() == str(python)


def test_bank_python_identity_normalizes_parent_alias(
    monkeypatch, tmp_path: Path
) -> None:
    real_bin = tmp_path / "real" / "venv" / "bin"
    real_bin.mkdir(parents=True)
    python = real_bin / "python"
    python.write_text("")
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "real", target_is_directory=True)
    monkeypatch.setattr(
        exp022.sys, "executable", str(alias / "venv" / "bin" / "python3")
    )
    assert exp022.python_executable() == str(python)


def test_exp022_display_path_accepts_external_bank_root(tmp_path: Path) -> None:
    external = tmp_path / "bank" / "derived"
    assert recipe._display_path(external) == external


@pytest.mark.parametrize("family,run_id", recipe.TRAINING_RUN_IDS.items())
def test_registry_training_run_identity(family: str, run_id: str) -> None:
    cells = [cell for cell in recipe.CANONICAL_CELLS if cell["family"] == family]
    assert cells
    assert {cell["training_run_id"] for cell in cells} == {run_id}


def test_every_registered_training_run_has_guide_and_results_sections() -> None:
    writing = (exp022.REPO / "writings" / "exp022.typ").read_text()
    run_ids = tuple(recipe.TRAINING_RUN_IDS.values())
    assert len(run_ids) == len(set(run_ids))
    results = re.findall(r"^\s*=== (TR-\d+) —", writing, re.MULTILINE)
    specifications = re.findall(
        r"^\s*=== Specification: (TR-\d+) —", writing, re.MULTILINE
    )
    for run_id in run_ids:
        assert results.count(run_id) == 1
        assert specifications.count(run_id) == 1


def test_tr07_low_input_controls_use_production_contract(tmp_path: Path) -> None:
    cells = [
        cell for cell in recipe.CANONICAL_CELLS if cell["training_run_id"] == "TR-07"
    ]
    assert len(cells) == 12
    assert {cell["seed"] for cell in cells} == {42, 43, 44}
    assert {cell["w_in"] for cell in cells} == {0.05, 0.1, 0.3, 0.9}
    for cell in cells:
        args = recipe.build_train_args(
            cell,
            tmp_path / cell["name"],
            recipe.SUBSET_MAX_SAMPLES,
            recipe.EPOCHS_STANDARD,
        )
        assert args[args.index("--max-samples") + 1] == "7000"
        assert args[args.index("--epochs") + 1] == "50"
        assert args[args.index("--seed") + 1] == str(cell["seed"])
        assert args[args.index("--w-in") + 1] == str(cell["w_in"])
        assert args[args.index("--fr-reg-upper-target-hz") + 1] == "1.0"
        assert args[args.index("--fr-reg-upper-strength") + 1] == "0.041"


def test_all_resolved_commands_keep_family_contract(tmp_path: Path) -> None:
    for cell in recipe.CANONICAL_CELLS:
        samples, epochs = recipe.cell_samples_epochs(cell)
        args = recipe.build_train_args(cell, tmp_path / cell["name"], samples, epochs)
        assert args[args.index("--epochs") + 1] == "50"
        assert args[args.index("--seed") + 1] == str(cell["seed"])
        assert args[args.index("--dt") + 1] == str(cell["dt_ms"])
        assert args[args.index("--tau-gaba") + 1] == str(cell["tau_gaba"])
        assert args[args.index("--n-hidden") + 1] == "1024"
        assert args[args.index("--input-rate") + 1] == "25.0"
        assert args[args.index("--weight-decay") + 1] == "0.0"
        assert "--dales-law" in args
        expected_w_in = str(cell["w_in"]) if cell["family"] == "low_w_in" else "0.9"
        assert args[args.index("--w-in") + 1] == expected_w_in
        expected_readout_mean = (
            recipe.TR06_READOUT_W_INIT_MEAN
            if cell["family"] == "variable_rate"
            else recipe.SHARED_READOUT_W_INIT_MEAN
        )
        expected_readout_std = (
            recipe.TR06_READOUT_W_INIT_STD
            if cell["family"] == "variable_rate"
            else recipe.SHARED_READOUT_W_INIT_STD
        )
        assert args[args.index("--readout-w-init-mean") + 1] == expected_readout_mean
        assert args[args.index("--readout-w-init-std") + 1] == expected_readout_std
        assert "--readout-w-out-scale" not in args
        if cell["model"] == "coba":
            assert args[args.index("--ei-strength") + 1] == "0"
            assert args[args.index("--v-grad-dampen") + 1] == "1"
        else:
            assert args[args.index("--v-grad-dampen") + 1] == "1000"
            if cell["model"] == "ping":
                assert args[args.index("--ei-strength") + 1] == "1"
        if cell["family"] == "canonical":
            assert args[args.index("--max-samples") + 1] == "60000"
        else:
            assert args[args.index("--max-samples") + 1] == "7000"
        if cell["family"] == "variable_rate":
            assert args[args.index("--readout") + 1] == "spike-count"
            assert tuple(map(float, args[args.index("--input-rates") + 1 :])) == (
                recipe.VARIABLE_RATE_TRAINING_RATES_HZ
            )


def test_all_resolved_cells_have_complete_scientific_contract(tmp_path: Path) -> None:
    contracts = []
    for cell in recipe.CANONICAL_CELLS:
        samples, epochs = recipe.cell_samples_epochs(cell)
        args = recipe.build_train_args(cell, tmp_path / cell["name"], samples, epochs)
        resolved = exp022.resolved_parameters(
            cell,
            args,
            samples,
            epochs,
            scientific_contract=recipe.scientific_contract(cell, samples, epochs),
        )
        contract = resolved["scientific_contract"]
        contracts.append(contract)
        assert contract["input"]["channels"] == 784
        assert contract["topology"] == {
            "excitatory_neurons": 1024,
            "inhibitory_neurons": 256,
            "output_neurons": 10,
            "output_population": "spiking_lif",
            "ei_loop_enabled": cell["model"] != "coba",
        }
        assert contract["dynamics"]["tau_ampa_ms"] == 2.0
        assert contract["constraints"]["dales_law"] is True
        assert contract["optimizer"]["weight_decay"] == 0.0
        assert contract["optimizer"]["gradient_clip_norm"] == 1.0
        assert contract["dataset"]["optimizer_train_samples"] == round(samples * 0.9)
        assert contract["dataset"]["validation_samples"] == samples - round(
            samples * 0.9
        )
        assert contract["dataset"]["official_test_samples"] == 10000
        assert contract["dataset"]["official_test_used_during_training"] is False
        if cell["family"] == "variable_rate":
            assert contract["input"]["rate_hz"] is None
            assert contract["input"]["rate_distribution_hz"] == list(
                recipe.VARIABLE_RATE_TRAINING_RATES_HZ
            )
        else:
            assert contract["input"]["rate_hz"] == 25.0
            assert contract["input"]["rate_distribution_hz"] is None

    assert len(contracts) == 102


def test_every_production_argument_is_mapped_or_operational(tmp_path: Path) -> None:
    for cell in recipe.CANONICAL_CELLS:
        samples, epochs = recipe.cell_samples_epochs(cell)
        args = recipe.build_train_args(cell, tmp_path / cell["name"], samples, epochs)
        parameters = exp022.resolved_parameters(
            cell,
            args,
            samples,
            epochs,
            scientific_contract=recipe.scientific_contract(cell, samples, epochs),
        )
        row = {"parameters": parameters}
        expected = exp022._expected_config(row)
        assert expected


def test_unmapped_manifest_argument_fails_closed(tmp_path: Path) -> None:
    row = _manifest_cell(tmp_path)
    row["parameters"]["arguments"]["--future-scientific-knob"] = "1"
    result = exp022.validate_cell(row, load_checkpoint=False)
    assert result["state"] == "missing" or not result["valid"]
    row["output_directory"] = str(_write_valid_cell(_manifest_cell(tmp_path)))
    result = exp022.validate_cell(row, load_checkpoint=False)
    assert not result["valid"]
    assert "no saved-config mapping" in result["reasons"][0]


@pytest.mark.parametrize(
    ("flag", "raw", "key", "expected"),
    [
        ("--w-in", "0.9", "w_in", [0.9, 0.09]),
        ("--trainable-w-ei", True, "trainable_w_ei", True),
        ("--trainable-w-ie", True, "trainable_w_ie", True),
        ("--n-hidden", "1024", "hidden_sizes", [1024]),
        ("--dales-law", True, "dales_law", True),
    ],
)
def test_scientific_argument_saved_config_transform(
    flag: str,
    raw: object,
    key: str,
    expected: object,
) -> None:
    row = {"parameters": {"arguments": {flag: raw}}}
    assert exp022._same(exp022._expected_config(row)[key], expected)


def test_validator_rejects_each_resolved_scientific_config_mismatch(
    tmp_path: Path,
) -> None:
    cell = recipe.PLANNED_VARIABLE_RATE_CELLS[0]
    samples, epochs = 100, 2
    args = recipe.build_train_args(
        cell, tmp_path / "cells" / cell["name"], samples, epochs
    )
    row = {
        "name": cell["name"],
        "training_run_id": cell["training_run_id"],
        "resource_tier": "variable_rate",
        "output_directory": str(tmp_path / "cells" / cell["name"]),
        "parameters": exp022.resolved_parameters(
            cell,
            args,
            samples,
            epochs,
            scientific_contract=recipe.scientific_contract(cell, samples, epochs),
        ),
    }
    directory = _write_valid_cell(row)
    expected = exp022._expected_config(row)
    original = json.loads((directory / "config.json").read_text())
    for key, value in expected.items():
        mutated = dict(original)
        if isinstance(value, bool):
            mutated[key] = not value
        elif isinstance(value, (int, float)):
            mutated[key] = value + 1
        elif isinstance(value, list):
            mutated[key] = [*value, "mismatch"]
        else:
            mutated[key] = f"{value}-mismatch"
        (directory / "config.json").write_text(json.dumps(mutated))
        result = exp022.validate_cell(row, load_checkpoint=False)
        assert not result["valid"], key
        assert any(f"config {key} mismatch" in reason for reason in result["reasons"])
        (directory / "config.json").write_text(json.dumps(original))


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
                "--model": "ping",
                "--dataset": "mnist",
                "--epochs": str(epochs),
                "--max-samples": str(samples),
                "--dt": "0.1",
                "--t-ms": "200.0",
                "--tau-gaba": "6.0",
                "--seed": "42",
                "--readout": "spike-count",
                "--input-rates": ["0.5", "1.0", "2.0", "5.0", "10.0", "25.0"],
            },
        },
    }


def _write_valid_cell(row: dict) -> Path:
    directory = Path(row["output_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    expected = exp022._expected_config(row)
    identity = {
        "training_cell_name": row["name"],
        "training_run_id": row["training_run_id"],
        "bank_resolved_parameters": row["parameters"],
    }
    roles = ("W_in", "W_out", "W_EE_1", "W_EI_1", "W_IE_1", "W_II_1")
    initialization = {
        role: {
            "distribution": "lower_clamped_normal",
            "zeros_remain_trainable": True,
            "requested_initial_zero_fraction": (
                expected.get("w_in_initial_zero_fraction", 0.0)
                if role == "W_in"
                else 0.0
            ),
            "statistics": {"n_parameters": 1},
        }
        for role in roles
    }
    config = {**expected, **identity, "weight_initialization": initialization}
    (directory / "config.json").write_text(json.dumps(config))
    epochs = row["parameters"]["epochs"]
    samples = round(row["parameters"]["max_samples"] * 0.9)
    (directory / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps({"ep": epoch, "samples": samples, "acc": 10.0})
            for epoch in range(1, epochs + 1)
        )
        + "\n"
    )
    state = {
        "b_out": torch.ones(10),
        "W_ff.0": torch.ones(784, 1024),
        "W_ff.1": torch.ones(1024, 10),
        "W_ei.1": torch.ones(1024, 256),
        "W_ie.1": torch.ones(256, 1024),
    }
    torch.save(state, directory / "weights.pth")
    torch.save(
        {**state, "b_out": torch.full((10,), 2.0)}, directory / "weights_final.pth"
    )
    checkpoints = {
        "best_validation": {
            "filename": "weights.pth",
            "epoch": 1,
            "sha256": exp022.sha256_file(directory / "weights.pth"),
        },
        "final_epoch": {
            "filename": "weights_final.pth",
            "epoch": epochs,
            "sha256": exp022.sha256_file(directory / "weights_final.pth"),
        },
    }
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                **identity,
                "config": {**expected, "weight_initialization": initialization},
                "best_epoch": 1,
                "checkpoints": checkpoints,
                "weight_final": {role: {"zero_fraction": 0.0} for role in roles},
            }
        )
    )
    return directory


def test_validator_states_and_valid_checkpoint(tmp_path: Path) -> None:
    row = _manifest_cell(tmp_path)
    assert exp022.validate_cell(row)["state"] == "missing"
    directory = Path(row["output_directory"])
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    assert exp022.validate_cell(row)["state"] == "partial"
    directory.rename(tmp_path / "discarded")
    _write_valid_cell(row)
    assert exp022.validate_cell(row) == {
        "valid": True,
        "state": "complete",
        "reasons": [],
    }


def test_validator_recognizes_w_ff_readout_without_named_output_key(
    tmp_path: Path,
) -> None:
    row = _manifest_cell(tmp_path)
    directory = _write_valid_cell(row)
    checkpoint = torch.load(
        directory / "weights.pth", map_location="cpu", weights_only=True
    )
    checkpoint.pop("b_out")
    torch.save(checkpoint, directory / "weights.pth")
    metrics = json.loads((directory / "metrics.json").read_text())
    metrics["checkpoints"]["best_validation"]["sha256"] = exp022.sha256_file(
        directory / "weights.pth"
    )
    (directory / "metrics.json").write_text(json.dumps(metrics))
    assert exp022.validate_cell(row) == {
        "valid": True,
        "state": "complete",
        "reasons": [],
    }


def test_validator_rejects_corrupt_mismatched_and_short_history(tmp_path: Path) -> None:
    row = _manifest_cell(tmp_path)
    directory = _write_valid_cell(row)
    (directory / "weights.pth").write_bytes(b"not a checkpoint")
    assert any(
        "checkpoint load failed" in reason
        for reason in exp022.validate_cell(row)["reasons"]
    )
    _write_valid_cell(row)
    config = json.loads((directory / "config.json").read_text())
    config["seed"] = 44
    (directory / "config.json").write_text(json.dumps(config))
    assert any(
        "seed mismatch" in reason for reason in exp022.validate_cell(row)["reasons"]
    )
    config["seed"] = 42
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "metrics.jsonl").write_text(
        json.dumps({"ep": 1, "samples": 100}) + "\n"
    )
    assert any("epoch 2" in reason for reason in exp022.validate_cell(row)["reasons"])


def test_preserve_partial_never_overwrites(tmp_path: Path) -> None:
    row = _manifest_cell(tmp_path)
    directory = Path(row["output_directory"])
    directory.mkdir(parents=True)
    (directory / "broken.txt").write_text("evidence")
    preserved = exp022.preserve_partial(directory)
    assert (
        preserved is not None and (preserved / "broken.txt").read_text() == "evidence"
    )
    assert not directory.exists()


def test_status_identifies_retry_cells(tmp_path: Path) -> None:
    complete = _manifest_cell(tmp_path)
    missing = {
        **_manifest_cell(tmp_path),
        "name": "missing",
        "output_directory": str(tmp_path / "cells" / "missing"),
    }
    _write_valid_cell(complete)
    status = exp022.summarize_status(
        {
            "bank_id": "test",
            "bank_root": str(tmp_path),
            "cells": [complete, missing],
        }
    )
    assert status["counts"] == {"complete": 1, "missing": 1}
    assert status["retry_cells"] == ["missing"]


def test_bank_worker_does_not_touch_valid_cell(tmp_path: Path, monkeypatch) -> None:
    row = _manifest_cell(tmp_path)
    directory = _write_valid_cell(row)
    before = exp022.sha256_file(directory / "weights.pth")
    manifest = {
        "bank_id": "test",
        "manifest_sha256": "abc",
        "repository": {"commit": "deadbeef", "dirty": False},
        "bank_root": str(tmp_path),
        "cells": [row],
    }
    monkeypatch.setattr(exp022, "_checked_bank_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        exp022.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("valid cell must not launch training"),
    )
    assert exp022._train_bank_cell(tmp_path / "bank.json", row["name"]) == 0
    assert exp022.sha256_file(directory / "weights.pth") == before


def _attempt_manifest(tmp_path: Path, row: dict) -> dict:
    return {
        "bank_id": "test",
        "manifest_sha256": "abc",
        "repository": {"commit": "deadbeef", "dirty": False},
        "bank_root": str(tmp_path),
        "cells": [row],
        "_runtime_commands": {row["name"]: ["tool", "train"]},
    }


def test_running_cell_is_reported_and_excluded_from_retry(tmp_path: Path) -> None:
    row = _manifest_cell(tmp_path)
    manifest = _attempt_manifest(tmp_path, row)
    record, lock = exp022.acquire_attempt(manifest, row)
    status = exp022.summarize_status(manifest, load_checkpoint=False)
    assert status["cells"][0]["state"] == "running"
    assert status["retry_cells"] == []
    exp022.status_path(manifest, row["name"]).unlink()
    lock_only_status = exp022.summarize_status(manifest, load_checkpoint=False)
    assert lock_only_status["cells"][0]["state"] == "running"
    assert lock_only_status["retry_cells"] == []
    exp022.release_attempt(lock, record["attempt_id"])


def test_duplicate_attempt_cannot_move_live_output(tmp_path: Path, monkeypatch) -> None:
    row = _manifest_cell(tmp_path)
    manifest = _attempt_manifest(tmp_path, row)
    record, lock = exp022.acquire_attempt(manifest, row)
    directory = Path(row["output_directory"])
    directory.mkdir(parents=True)
    evidence = directory / "live.txt"
    evidence.write_text("still writing")
    monkeypatch.setattr(exp022, "_checked_bank_manifest", lambda _path: manifest)
    with pytest.raises(RuntimeError, match="active attempt"):
        exp022._train_bank_cell(tmp_path / "bank.json", row["name"])
    assert evidence.read_text() == "still writing"
    assert not (tmp_path / "failed").exists()
    exp022.release_attempt(lock, record["attempt_id"])


def test_stale_attempt_requires_explicit_recovery(tmp_path: Path) -> None:
    row = _manifest_cell(tmp_path)
    manifest = _attempt_manifest(tmp_path, row)
    record, lock = exp022.acquire_attempt(manifest, row)
    status_file = exp022.status_path(manifest, row["name"])
    stale = json.loads(status_file.read_text())
    stale["pid"] = 999_999_999
    exp022.atomic_json(status_file, stale)
    status = exp022.summarize_status(manifest, load_checkpoint=False)
    assert status["cells"][0]["state"] == "stale"
    assert status["retry_cells"] == []
    assert status["recoverable_cells"] == [row["name"]]
    with pytest.raises(RuntimeError, match="use --recover-stale"):
        exp022.acquire_attempt(manifest, row)
    recovered, recovered_lock = exp022.acquire_attempt(
        manifest, row, recover_stale=True
    )
    assert recovered["attempt_id"] != record["attempt_id"]
    exp022.release_attempt(recovered_lock, recovered["attempt_id"])
    lock.unlink(missing_ok=True)


def test_failed_subprocess_without_metrics_records_failure(
    tmp_path: Path, monkeypatch
) -> None:
    row = _manifest_cell(tmp_path)
    manifest = _attempt_manifest(tmp_path, row)
    monkeypatch.setattr(exp022, "_checked_bank_manifest", lambda _path: manifest)
    monkeypatch.setattr(exp022, "_gpu_metadata", lambda: {"available": False})
    monkeypatch.setattr(
        exp022.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 7),
    )
    assert exp022._train_bank_cell(tmp_path / "bank.json", row["name"]) == 1
    attempt = json.loads((Path(row["output_directory"]) / "attempt.json").read_text())
    assert attempt["state"] == "failed"
    assert attempt["exit_code"] == 7


def test_preserve_partial_avoids_timestamp_collision(
    tmp_path: Path, monkeypatch
) -> None:
    row = _manifest_cell(tmp_path)
    directory = Path(row["output_directory"])
    directory.mkdir(parents=True)
    (directory / "evidence").write_text("new")
    monkeypatch.setattr(
        exp022, "utc_now", lambda: "2026-08-11T00:00:00+00:00"
    )
    occupied = tmp_path / "failed" / row["name"] / "2026-08-11T00-00-00+00-00"
    occupied.mkdir(parents=True)
    preserved = exp022.preserve_partial(directory)
    assert preserved is not None
    assert preserved.name.endswith("-1")
    assert (preserved / "evidence").read_text() == "new"


def _write_checked_manifest(
    tmp_path: Path, monkeypatch, tier: str = "variable_rate"
) -> Path:
    monkeypatch.setattr(
        exp022, "git_identity", lambda _repo: ("deadbeef", False)
    )
    monkeypatch.setattr(
        exp022,
        "lock_identity",
        lambda _repo: {"path": "uv.lock", "sha256": "lock"},
    )
    cells = recipe.cells_in_resource_tier(tier)
    payload = exp022.create_manifest(
        repo=exp022.REPO,
        bank_root=tmp_path,
        bank_id="checked",
        cells=cells,
        tier_for=recipe.cell_resource_tier,
        samples_epochs=recipe.cell_samples_epochs,
        build_args=recipe.build_train_args,
        scientific_contract_for=recipe.scientific_contract,
        selection_tier=tier,
    )
    path = tmp_path / "bank.json"
    exp022.write_manifest(path, payload)
    return path


@pytest.mark.parametrize(
    "mutation",
    [
        "command",
        "command_shell",
        "output_directory",
        "required_outputs",
        "duplicate",
        "missing",
    ],
)
def test_checked_manifest_rejects_rehashed_executable_mutations(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
) -> None:
    path = _write_checked_manifest(tmp_path, monkeypatch)
    payload = exp022.load_manifest(path)
    payload.pop("manifest_sha256")
    if mutation == "command":
        payload["cells"][0]["command"][-1] = "tampered"
    elif mutation == "command_shell":
        payload["cells"][0]["command_shell"] += " --tampered"
    elif mutation == "output_directory":
        payload["cells"][0]["output_directory"] = str(tmp_path.parent / "escape")
    elif mutation == "required_outputs":
        payload["cells"][0]["required_outputs"] = ["weights.pth"]
    elif mutation == "duplicate":
        payload["cells"].append(dict(payload["cells"][0]))
    else:
        payload["cells"].pop()
    exp022.write_manifest(path, payload)
    with pytest.raises(SystemExit):
        exp022._checked_bank_manifest(path)


def test_bank_creation_refuses_existing_destination(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "bank"
    root.mkdir()
    evidence = root / "keep.txt"
    evidence.write_text("existing work")
    monkeypatch.setattr(
        exp022,
        "reserve_stage",
        lambda *_args, **_kwargs: pytest.fail("must not reserve over existing work"),
    )
    with pytest.raises(SystemExit, match="already exists"):
        exp022._handle_bank_cli(["--bank-create", str(root)])
    assert evidence.read_text() == "existing work"


def test_hpc_array_uses_frozen_cells_and_shared_wrapper(tmp_path: Path) -> None:
    from experiments.exp022 import hpc

    plan = {
        "account": "gpu-account",
        "partition": "ampere",
        "walltime": "01:00:00",
        "collector_walltime": "00:30:00",
        "cpus": 4,
        "memory_gb": 32,
        "partitions": [["a"], ["b"], ["c"]],
        "concurrency": 2,
    }
    command = hpc.command(plan, tmp_path / "plan.json", "compute")
    assert "--array=0-2%2" in command
    assert str(exp022.REPO / "experiments/helpers/slurm-stage.sbatch") in command
    assert command[-1] == "exp022"


def test_mnist_link_helper_accepts_existing_and_concurrent_creation(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    (cache / "MNIST").mkdir(parents=True)
    link = tmp_path / "mnist"
    helper = exp022.REPO / "experiments" / "exp022" / "slurm" / "ensure-mnist-link.sh"
    commands = [[str(helper), str(cache), str(link)] for _ in range(2)]
    processes = [subprocess.Popen(command) for command in commands]
    assert [process.wait() for process in processes] == [0, 0]
    subprocess.run(commands[0], check=True)
    assert link.resolve() == cache.resolve()


def test_wilkes_modules_load_in_sanitized_environment(tmp_path: Path) -> None:
    calls = tmp_path / "module-calls.txt"
    initializer = tmp_path / "modules.sh"
    initializer.write_text(
        'module() { printf "%s\\n" "$*" >> "$EXP022_MODULE_CALLS"; }\n'
    )
    helper = exp022.REPO / "experiments" / "exp022" / "slurm" / "load-wilkes-modules.sh"
    subprocess.run(
        [
            "env",
            "-i",
            f"PATH={Path('/usr/bin')}:/bin",
            f"EXP022_MODULES_INIT={initializer}",
            f"EXP022_MODULE_CALLS={calls}",
            "/bin/bash",
            "-c",
            f"source {helper}",
        ],
        check=True,
    )
    assert calls.read_text().splitlines() == ["purge", "load rhel8/default-amp"]


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = REPO / "experiments" / "exp022"
SLURM = EXPERIMENT / "slurm"


@pytest.mark.parametrize(
    "entrypoint",
    [
        "compute.py",
        "analyse.py",
        "present.py",
        "slurm/wilkes_diagnostic.py",
    ],
)
def test_file_entrypoints_resolve_from_an_external_directory(entrypoint, tmp_path):
    completed = subprocess.run(
        [sys.executable, str(EXPERIMENT / entrypoint), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "usage:" in completed.stdout


"""Physical-time checks for the collection's nonintegral analysis bins."""


def periodic_raster(dt, period_ms=30.0, duration_ms=1800.0):
    raster = np.zeros((round(duration_ms / dt), 10), dtype=bool)
    events = np.rint(np.arange(0, duration_ms, period_ms) / dt).astype(int)
    raster[events] = True
    return raster


@pytest.mark.parametrize("dt", [0.05, 0.1, 0.2, 0.3, 0.6])
def test_spectrum_reports_physical_frequency_across_timestep_grid(dt):
    frequencies, power, peak = _gamma_psd(periodic_raster(dt), dt)
    assert frequencies.shape == power.shape
    assert peak == pytest.approx(1000 / 30, abs=0.7)


@pytest.mark.parametrize("dt", [0.1, 0.3, 0.6])
def test_autocorrelation_lags_and_scalar_lookup_use_physical_time(dt):
    raster = periodic_raster(dt)
    lags, ac = figure_metrics.spike_autocorrelogram(raster, dt, max_lag_ms=60)
    window = (lags >= 20) & (lags <= 40)
    peak_lag = lags[window][np.argmax(ac[window])]
    assert peak_lag == pytest.approx(30, abs=1.2)
    # IEI bins remain physical milliseconds and can differ from the AC bins.
    result = figure_metrics.rhythmicity_scalars(
        lags, ac, np.array([30.0]), np.array([1]), bio_lag_ms=30.0
    )
    expected = ac[round(30 / (lags[1] - lags[0]))]
    assert result["biophysical"] == expected
    assert result["iei_anchored"] == expected


@pytest.mark.parametrize("dt,steps", [(0.1, 2000), (0.3, 666), (0.6, 333)])
def test_snapshot_rate_integral_preserves_spike_counts_in_partial_bin(tmp_path, dt, steps):
    spikes = np.zeros((steps, 2), dtype=bool)
    spikes[-1] = True
    source, destination = tmp_path / "recording.npz", tmp_path / "rasters.npz"
    np.savez(source, spk_e=spikes, spk_i=spikes, dt=np.float32(dt))
    measure_snapshot(source, destination)
    with np.load(destination) as measured:
        assert measured["bin_widths_ms"].sum() == pytest.approx(steps * dt)
        assert len(measured["bin_widths_ms"]) == 200
        assert measured["bin_widths_ms"][-1] == pytest.approx(1.0 if dt == 0.1 else 0.8)
        for population in ("e", "i"):
            integrated = np.sum(measured[f"{population}_rate"] * measured["bin_widths_ms"]) / 1000
            assert integrated == pytest.approx(1.0)
            assert measured[f"{population}_hz"] == pytest.approx(1000 / (steps * dt))


def test_bank_composition_rejects_overlap_or_missing_cells():
    record = {
        "inputs": {"retained_bank": {"run_id": "source", "payload_digest": "digest"}},
        "bank_reuse": {"plan": {
            "reused_cells": ["old"], "new_cells": ["new"],
            "diagnostic_policy": "regenerate_all",
        }},
    }
    result = bank_composition(record, {"old", "new"})
    assert result["reused_cells"] == result["new_cells"] == 1
    with pytest.raises(PingstoreError, match="partition"):
        bank_composition(record, {"old", "new", "missing"})
    record["bank_reuse"]["plan"]["new_cells"].append("old")
    with pytest.raises(PingstoreError, match="partition"):
        bank_composition(record, {"old", "new"})
