from __future__ import annotations

import copy
import json
import re
import subprocess
from pathlib import Path

import pytest
import torch
from experiments import exp022 as exp022_package
from experiments.exp022 import campaign, recipe, tr06_diagnostic
from experiments.exp022 import compute as exp022
from experiments.helpers import archive

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


def test_campaign_python_identity_stays_inside_environment(
    monkeypatch, tmp_path: Path
) -> None:
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text("")
    monkeypatch.setattr(campaign.sys, "executable", str(bin_dir / "python3"))
    assert campaign.python_executable() == str(python)


def test_campaign_python_identity_normalizes_parent_alias(
    monkeypatch, tmp_path: Path
) -> None:
    real_bin = tmp_path / "real" / "venv" / "bin"
    real_bin.mkdir(parents=True)
    python = real_bin / "python"
    python.write_text("")
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "real", target_is_directory=True)
    monkeypatch.setattr(
        campaign.sys, "executable", str(alias / "venv" / "bin" / "python3")
    )
    assert campaign.python_executable() == str(python)


def test_exp022_display_path_accepts_external_campaign_root(tmp_path: Path) -> None:
    external = tmp_path / "campaign" / "derived"
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
        resolved = campaign.resolved_parameters(
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
        parameters = campaign.resolved_parameters(
            cell,
            args,
            samples,
            epochs,
            scientific_contract=recipe.scientific_contract(cell, samples, epochs),
        )
        row = {"parameters": parameters}
        expected = campaign._expected_config(row)
        assert expected


def test_unmapped_manifest_argument_fails_closed(tmp_path: Path) -> None:
    row = _manifest_cell(tmp_path)
    row["parameters"]["arguments"]["--future-scientific-knob"] = "1"
    result = campaign.validate_cell(row, load_checkpoint=False)
    assert result["state"] == "missing" or not result["valid"]
    row["output_directory"] = str(_write_valid_cell(_manifest_cell(tmp_path)))
    result = campaign.validate_cell(row, load_checkpoint=False)
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
    assert campaign._same(campaign._expected_config(row)[key], expected)


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
        "parameters": campaign.resolved_parameters(
            cell,
            args,
            samples,
            epochs,
            scientific_contract=recipe.scientific_contract(cell, samples, epochs),
        ),
    }
    directory = _write_valid_cell(row)
    expected = campaign._expected_config(row)
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
        result = campaign.validate_cell(row, load_checkpoint=False)
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
    expected = campaign._expected_config(row)
    identity = {
        "training_cell_name": row["name"],
        "training_run_id": row["training_run_id"],
        "campaign_resolved_parameters": row["parameters"],
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
            "sha256": campaign.sha256_file(directory / "weights.pth"),
        },
        "final_epoch": {
            "filename": "weights_final.pth",
            "epoch": epochs,
            "sha256": campaign.sha256_file(directory / "weights_final.pth"),
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
    assert campaign.validate_cell(row)["state"] == "missing"
    directory = Path(row["output_directory"])
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}")
    assert campaign.validate_cell(row)["state"] == "partial"
    directory.rename(tmp_path / "discarded")
    _write_valid_cell(row)
    assert campaign.validate_cell(row) == {
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
    metrics["checkpoints"]["best_validation"]["sha256"] = campaign.sha256_file(
        directory / "weights.pth"
    )
    (directory / "metrics.json").write_text(json.dumps(metrics))
    assert campaign.validate_cell(row) == {
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
        for reason in campaign.validate_cell(row)["reasons"]
    )
    _write_valid_cell(row)
    config = json.loads((directory / "config.json").read_text())
    config["seed"] = 44
    (directory / "config.json").write_text(json.dumps(config))
    assert any(
        "seed mismatch" in reason for reason in campaign.validate_cell(row)["reasons"]
    )
    config["seed"] = 42
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "metrics.jsonl").write_text(
        json.dumps({"ep": 1, "samples": 100}) + "\n"
    )
    assert any("epoch 2" in reason for reason in campaign.validate_cell(row)["reasons"])


def test_preserve_partial_never_overwrites(tmp_path: Path) -> None:
    row = _manifest_cell(tmp_path)
    directory = Path(row["output_directory"])
    directory.mkdir(parents=True)
    (directory / "broken.txt").write_text("evidence")
    preserved = campaign.preserve_partial(directory)
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
    status = campaign.summarize_status(
        {
            "campaign_id": "test",
            "campaign_root": str(tmp_path),
            "cells": [complete, missing],
        }
    )
    assert status["counts"] == {"complete": 1, "missing": 1}
    assert status["retry_cells"] == ["missing"]


def test_campaign_train_does_not_touch_valid_cell(tmp_path: Path, monkeypatch) -> None:
    row = _manifest_cell(tmp_path)
    directory = _write_valid_cell(row)
    before = campaign.sha256_file(directory / "weights.pth")
    manifest = {
        "campaign_id": "test",
        "manifest_sha256": "abc",
        "repository": {"commit": "deadbeef", "dirty": False},
        "campaign_root": str(tmp_path),
        "cells": [row],
    }
    monkeypatch.setattr(exp022, "_checked_manifest", lambda _path: manifest)
    monkeypatch.setattr(
        exp022.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("valid cell must not launch training"),
    )
    assert exp022._campaign_train(tmp_path / "campaign.json", row["name"]) == 0
    assert campaign.sha256_file(directory / "weights.pth") == before


def _attempt_manifest(tmp_path: Path, row: dict) -> dict:
    return {
        "campaign_id": "test",
        "manifest_sha256": "abc",
        "repository": {"commit": "deadbeef", "dirty": False},
        "campaign_root": str(tmp_path),
        "cells": [row],
        "_runtime_commands": {row["name"]: ["tool", "train"]},
    }


def test_running_cell_is_reported_and_excluded_from_retry(tmp_path: Path) -> None:
    row = _manifest_cell(tmp_path)
    manifest = _attempt_manifest(tmp_path, row)
    record, lock = campaign.acquire_attempt(manifest, row)
    status = campaign.summarize_status(manifest, load_checkpoint=False)
    assert status["cells"][0]["state"] == "running"
    assert status["retry_cells"] == []
    campaign.status_path(manifest, row["name"]).unlink()
    lock_only_status = campaign.summarize_status(manifest, load_checkpoint=False)
    assert lock_only_status["cells"][0]["state"] == "running"
    assert lock_only_status["retry_cells"] == []
    campaign.release_attempt(lock, record["attempt_id"])


def test_duplicate_attempt_cannot_move_live_output(tmp_path: Path, monkeypatch) -> None:
    row = _manifest_cell(tmp_path)
    manifest = _attempt_manifest(tmp_path, row)
    record, lock = campaign.acquire_attempt(manifest, row)
    directory = Path(row["output_directory"])
    directory.mkdir(parents=True)
    evidence = directory / "live.txt"
    evidence.write_text("still writing")
    monkeypatch.setattr(exp022, "_checked_manifest", lambda _path: manifest)
    with pytest.raises(RuntimeError, match="active attempt"):
        exp022._campaign_train(tmp_path / "campaign.json", row["name"])
    assert evidence.read_text() == "still writing"
    assert not (tmp_path / "failed").exists()
    campaign.release_attempt(lock, record["attempt_id"])


def test_stale_attempt_requires_explicit_recovery(tmp_path: Path) -> None:
    row = _manifest_cell(tmp_path)
    manifest = _attempt_manifest(tmp_path, row)
    record, lock = campaign.acquire_attempt(manifest, row)
    status_file = campaign.status_path(manifest, row["name"])
    stale = json.loads(status_file.read_text())
    stale["pid"] = 999_999_999
    campaign.atomic_json(status_file, stale)
    status = campaign.summarize_status(manifest, load_checkpoint=False)
    assert status["cells"][0]["state"] == "stale"
    assert status["retry_cells"] == []
    assert status["recoverable_cells"] == [row["name"]]
    with pytest.raises(RuntimeError, match="use --recover-stale"):
        campaign.acquire_attempt(manifest, row)
    recovered, recovered_lock = campaign.acquire_attempt(
        manifest, row, recover_stale=True
    )
    assert recovered["attempt_id"] != record["attempt_id"]
    campaign.release_attempt(recovered_lock, recovered["attempt_id"])
    lock.unlink(missing_ok=True)


def test_failed_subprocess_without_metrics_records_failure(
    tmp_path: Path, monkeypatch
) -> None:
    row = _manifest_cell(tmp_path)
    manifest = _attempt_manifest(tmp_path, row)
    monkeypatch.setattr(exp022, "_checked_manifest", lambda _path: manifest)
    monkeypatch.setattr(exp022, "_gpu_metadata", lambda: {"available": False})
    monkeypatch.setattr(
        exp022.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 7),
    )
    assert exp022._campaign_train(tmp_path / "campaign.json", row["name"]) == 1
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
    monkeypatch.setattr(campaign, "utc_now", lambda: "2026-08-11T00:00:00+00:00")
    occupied = tmp_path / "failed" / row["name"] / "2026-08-11T00-00-00+00-00"
    occupied.mkdir(parents=True)
    preserved = campaign.preserve_partial(directory)
    assert preserved is not None
    assert preserved.name.endswith("-1")
    assert (preserved / "evidence").read_text() == "new"


def _write_checked_manifest(
    tmp_path: Path, monkeypatch, tier: str = "variable_rate"
) -> Path:
    monkeypatch.setattr(campaign, "git_identity", lambda _repo: ("deadbeef", False))
    monkeypatch.setattr(
        campaign, "lock_identity", lambda _repo: {"path": "uv.lock", "sha256": "lock"}
    )
    cells = recipe.cells_in_resource_tier(tier)
    payload = campaign.create_manifest(
        repo=exp022.REPO,
        campaign_root=tmp_path,
        campaign_id="checked",
        cells=cells,
        tier_for=recipe.cell_resource_tier,
        samples_epochs=recipe.cell_samples_epochs,
        build_args=recipe.build_train_args,
        scientific_contract_for=recipe.scientific_contract,
        selection_tier=tier,
    )
    path = tmp_path / "campaign.json"
    campaign.write_manifest(path, payload)
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
    payload = campaign.load_manifest(path)
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
    campaign.write_manifest(path, payload)
    with pytest.raises(SystemExit):
        exp022._checked_manifest(path)


def test_external_aggregation_completes_compute_without_downstream_dispatch(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = {
        "campaign_id": "external",
        "campaign_root": str(tmp_path),
        "cells": [{} for _ in recipe.CANONICAL_CELLS],
    }
    monkeypatch.setattr(exp022, "_checked_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(
        campaign,
        "summarize_status",
        lambda _manifest: {
            "retry_cells": [],
            "cells": [{"name": "cell", "valid": True}],
        },
    )
    observed = []
    monkeypatch.setattr(
        exp022, "capture_campaign", lambda path, value: observed.append((path, value))
    )
    monkeypatch.setattr(
        exp022.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "aggregation must not dispatch downstream stages"
        ),
    )
    assert exp022._handle_campaign_cli(
        [
            "compute.py",
            "--campaign-aggregate",
            str(tmp_path / "campaign.json"),
        ]
    )
    assert observed == [(tmp_path / "campaign.json", manifest)]
    assert recipe.training_root_provenance(tmp_path)["location"] == "external"


def test_post_aggregation_check_allows_only_generated_exp022_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_checked_manifest(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign, "git_identity", lambda _repo: ("deadbeef", True))
    monkeypatch.setattr(
        campaign,
        "git_dirty_paths",
        lambda _repo: [".artifacts/exp022/numbers.json", ".demolab/pdfs/exp022.pdf"],
    )
    assert (
        exp022._checked_manifest(
            manifest_path,
            allow_generated_dirty=True,
        )["campaign_id"]
        == "checked"
    )
    monkeypatch.setattr(
        campaign,
        "git_dirty_paths",
        lambda _repo: ["experiments/exp022/compute.py"],
    )
    with pytest.raises(SystemExit, match="clean source worktree"):
        exp022._checked_manifest(manifest_path, allow_generated_dirty=True)


@pytest.mark.parametrize(
    "occupied", ["empty", "manifest", "cell", "status", "arbitrary"]
)
def test_campaign_creation_refuses_existing_destination(
    tmp_path: Path,
    monkeypatch,
    occupied: str,
) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    if occupied == "manifest":
        (root / "campaign.json").write_text("original manifest")
    elif occupied == "cell":
        cell = root / "cells" / "existing"
        cell.mkdir(parents=True)
        (cell / "weights.pth").write_text("expensive checkpoint")
    elif occupied == "status":
        status = root / "status"
        status.mkdir()
        (status / "cell.json").write_text("running")
    elif occupied == "arbitrary":
        (root / "notes.txt").write_text("keep me")
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        campaign,
        "create_manifest",
        lambda **_kwargs: {"campaign_id": "must-not-overwrite"},
    )
    with pytest.raises(SystemExit, match="already exists and will not be modified"):
        exp022._handle_campaign_cli(
            [
                "compute.py",
                "--campaign-manifest",
                str(root),
                "--campaign-id",
                "must-not-overwrite",
                "--tier",
                "variable_rate",
            ]
        )
    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_verified_archive_source_is_manifest_cells_not_legacy(
    tmp_path: Path, monkeypatch
) -> None:
    row = _manifest_cell(tmp_path)
    manifest = _attempt_manifest(tmp_path, row)
    manifest["cells"] = [{} for _ in recipe.CANONICAL_CELLS]
    checked = {}

    def fake_checked_manifest(_path, *, allow_generated_dirty=False):
        checked["allow_generated_dirty"] = allow_generated_dirty
        return manifest

    monkeypatch.setattr(exp022, "_checked_manifest", fake_checked_manifest)
    monkeypatch.setattr(
        campaign,
        "summarize_status",
        lambda _manifest: {
            "retry_cells": [],
            "recoverable_cells": [],
            "cells": [{"state": "complete"} for _ in recipe.CANONICAL_CELLS],
        },
    )
    selected, source = archive.verified_campaign_source(tmp_path / "campaign.json")
    assert selected is manifest
    assert source == (tmp_path / "cells").resolve()
    assert source != (archive.ARTIFACTS_ROOT / "exp022").resolve()
    assert checked["allow_generated_dirty"] is True


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


def test_submission_selection_is_frozen_read_only() -> None:
    submit = (
        exp022.REPO / "experiments" / "exp022" / "slurm" / "submit-tier.sh"
    ).read_text()
    array = (
        exp022.REPO / "experiments" / "exp022" / "slurm" / "train-array.sbatch"
    ).read_text()
    assert 'chmod 0444 "$selection"' in submit
    assert 'mapfile -t cells < "$EXP022_SELECTION"' in array
    assert "--campaign-list" not in array


def test_portable_cell_contract_ignores_only_output_path() -> None:
    source = _manifest_cell(Path("/source"))
    destination = _manifest_cell(Path("/destination"))
    source["family"] = destination["family"] = "variable_rate"
    source["parameters"]["arguments"]["--out-dir"] = "/source/cell"
    destination["parameters"]["arguments"]["--out-dir"] = "/destination/cell"

    assert exp022._portable_cell_contract(source) == exp022._portable_cell_contract(
        destination
    )

    destination["parameters"]["arguments"]["--fr-reg-upper-strength"] = "0.041"
    assert exp022._portable_cell_contract(source) != exp022._portable_cell_contract(
        destination
    )


def test_import_compatible_cell_restamps_destination_and_keeps_origin(
    tmp_path: Path,
) -> None:
    source_root = (tmp_path / "source").resolve()
    destination_root = (tmp_path / "destination").resolve()
    source_row = _manifest_cell(source_root)
    source_row["family"] = "variable_rate"
    source_row["parameters"]["arguments"]["--out-dir"] = source_row["output_directory"]
    _write_valid_cell(source_row)
    source_manifest = {
        "schema": campaign.SCHEMA,
        "schema_version": campaign.SCHEMA_VERSION,
        "campaign_id": "base",
        "campaign_root": str(source_root),
        "repository": {"commit": "a" * 40, "dirty": False},
        "cells": [source_row],
    }
    source_root.mkdir(exist_ok=True)
    campaign.write_manifest(source_root / "campaign.json", source_manifest)

    destination_row = copy.deepcopy(source_row)
    destination_row["output_directory"] = str(
        destination_root / "cells" / destination_row["name"]
    )
    destination_row["parameters"]["arguments"]["--out-dir"] = destination_row[
        "output_directory"
    ]
    destination_manifest = {
        **source_manifest,
        "campaign_id": "repair",
        "campaign_root": str(destination_root),
        "repository": {"commit": "b" * 40, "dirty": False},
        "cells": [destination_row],
        "manifest_sha256": "c" * 64,
    }

    result = exp022._import_compatible_cells(
        destination_manifest, source_root / "campaign.json"
    )

    assert result["imported"] == [destination_row["name"]]
    assert result["pending_incompatible"] == []
    assert campaign.validate_cell(destination_row)["valid"]
    imported = json.loads(
        (Path(destination_row["output_directory"]) / "metrics.json").read_text()
    )
    assert imported["campaign_id"] == "repair"
    assert imported["imported_cell_provenance"]["campaign_id"] == "base"


"""Relocation checks that never train, submit jobs or touch retained evidence."""

import os
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = REPO / "experiments" / "exp022"
SLURM = EXPERIMENT / "slurm"


@pytest.mark.parametrize(
    "imports",
    [
        "from experiments.exp022 import campaign, compute",
        "from experiments.exp022 import compute, campaign",
        "from experiments.exp022 import tr06_diagnostic; "
        "from experiments.exp022 import campaign, compute",
        "import sys; sys.path.insert(0, 'experiments'); import exp022; "
        "from experiments.exp022 import campaign, compute; "
        "assert exp022.campaign is campaign; "
        "assert exp022.run_tr06_diagnostic is compute.run_tr06_diagnostic",
    ],
)
def test_relocated_imports_preserve_campaign_and_scheduler_hooks(imports):
    subprocess.run(
        [sys.executable, "-c", imports + "; assert compute.campaign is campaign"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "entrypoint",
    [
        "compute.py",
        "analyse.py",
        "present.py",
        "tr06_diagnostic.py",
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


def test_slurm_scripts_and_collection_references_resolve():
    scripts = sorted(SLURM.glob("*.sh")) + sorted(SLURM.glob("*.sbatch"))
    scripts.append(
        REPO / "experiments/collections/gamma_gated_sparsity/collection-job.sbatch"
    )
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)
        for reference in re.findall(r"experiments/exp022/[\w./-]+", script.read_text()):
            target = REPO / reference
            assert target.is_file(), (script, reference)
    # This helper is executed directly; module initialization is only sourced.
    assert os.access(SLURM / "ensure-mnist-link.sh", os.X_OK)


def test_submit_wrapper_finds_repository_before_validation(tmp_path):
    cache = tmp_path / "cache"
    (cache / "MNIST").mkdir(parents=True)
    manifest = tmp_path / "campaign.json"
    manifest.write_text("{}")
    uv = tmp_path / "uv"
    uv.write_text(
        '#!/bin/bash\nprintf "cwd=%s\\n" "$PWD"\nprintf "arg=%s\\n" "$@"\nexit 23\n'
    )
    uv.chmod(0o755)
    completed = subprocess.run(
        ["bash", str(SLURM / "submit-tier.sh"), str(manifest), "standard", "--dry-run"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "EXP022_SLURM_ACCOUNT": "test-account",
            "EXP022_WALLTIME": "00:01:00",
            "EXP022_CONCURRENCY": "1",
            "EXP022_MNIST_CACHE": str(cache),
            "EXP022_UV": str(uv),
        },
    )
    # Stop at mocked validation: no scheduler or campaign mutation is involved.
    assert completed.returncode == 23, completed.stderr
    assert f"cwd={REPO}" in completed.stdout
    assert "arg=experiments/exp022/compute.py" in completed.stdout
    assert "arg=--campaign-validate" in completed.stdout


def test_tr06_diagnostic_variants_change_only_the_readout_contract(tmp_path) -> None:
    commands = {
        variant: tr06_diagnostic.diagnostic_args(
            variant,
            output=tmp_path / variant,
            max_samples=700,
            epochs=10,
            seed=42,
            device="cuda",
            n_hidden=64,
            t_ms=50.0,
            dt_ms=1.0,
        )
        for variant in tr06_diagnostic.VARIANTS
    }
    registered = commands["registered-spike-count"]
    fanin = commands["fanin-spike-count"]
    mean_005 = commands["mean-005-spike-count"]
    mean_010 = commands["mean-010-spike-count"]
    control = commands["mem-mean-control"]

    assert registered[registered.index("--readout") + 1] == "spike-count"
    assert (
        registered[registered.index("--readout-w-init-mean") + 1]
        == exp022_package.SHARED_READOUT_W_INIT_MEAN
    )
    assert (
        registered[registered.index("--readout-w-init-std") + 1]
        == exp022_package.SHARED_READOUT_W_INIT_STD
    )
    assert fanin[fanin.index("--readout") + 1] == "spike-count"
    assert "--readout-w-init-mean" not in fanin
    assert "--readout-w-init-std" not in fanin
    assert mean_005[mean_005.index("--readout-w-init-mean") + 1] == "0.05"
    assert mean_010[mean_010.index("--readout-w-init-mean") + 1] == "0.1"
    assert mean_005[mean_005.index("--readout-w-init-std") + 1] == "0.04"
    assert mean_010[mean_010.index("--readout-w-init-std") + 1] == "0.08"
    assert control[control.index("--readout") + 1] == "mem-mean"
    assert (
        control[control.index("--readout-w-init-mean") + 1]
        == exp022_package.SHARED_READOUT_W_INIT_MEAN
    )
    for command in commands.values():
        assert command[command.index("--max-samples") + 1] == "700"
        assert command[command.index("--epochs") + 1] == "10"
        assert command[command.index("--seed") + 1] == "42"
        assert command[command.index("--device") + 1] == "cuda"
        assert command[command.index("--n-hidden") + 1] == "64"
        assert command[command.index("--t-ms") + 1] == "50.0"
        assert command[command.index("--dt") + 1] == "1.0"
        start = command.index("--input-rates") + 1
        assert tuple(map(float, command[start : start + 11])) == (
            0.5,
            0.75,
            1.0,
            1.5,
            2.0,
            3.0,
            5.0,
            7.5,
            10.0,
            15.0,
            25.0,
        )
