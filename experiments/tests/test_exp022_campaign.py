from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch
from experiments import exp022
from experiments.exp022_support import campaign
from experiments.helpers import archive

CONCRETE_TIERS = ("standard", "fine_dt", "canonical_coba", "canonical_ping", "variable_rate")


def test_registry_has_90_unique_cells_partitioned_once() -> None:
    names = [cell["name"] for cell in exp022.CANONICAL_CELLS]
    assert len(names) == len(set(names)) == 90
    tiered = [cell["name"] for tier in CONCRETE_TIERS for cell in exp022.cells_in_resource_tier(tier)]
    assert sorted(tiered) == sorted(names)


def test_campaign_python_identity_stays_inside_environment(monkeypatch, tmp_path: Path) -> None:
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text("")
    monkeypatch.setattr(campaign.sys, "executable", str(bin_dir / "python3"))
    assert campaign.python_executable() == str(python)


def test_campaign_python_identity_normalizes_parent_alias(monkeypatch, tmp_path: Path) -> None:
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
    assert exp022._display_path(external) == external


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
        assert args[args.index("--w-in") + 1] == "0.9"
        assert args[args.index("--readout-w-init-mean") + 1] == exp022.SHARED_READOUT_W_INIT_MEAN
        assert args[args.index("--readout-w-init-std") + 1] == exp022.SHARED_READOUT_W_INIT_STD
        assert "--readout-w-out-scale" not in args
        if cell["model"] == "coba":
            assert args[args.index("--ei-strength") + 1] == "0"
            assert args[args.index("--v-grad-dampen") + 1] == "1"
        else:
            assert args[args.index("--v-grad-dampen") + 1] == "1000"
            if cell["model"] == "ping":
                assert args[args.index("--ei-strength") + 1] == "1"
        if cell["family"] == "canonical":
            assert args[args.index("--max-samples") + 1] == "70000"
        else:
            assert args[args.index("--max-samples") + 1] == "7000"
        if cell["family"] == "variable_rate":
            assert args[args.index("--readout") + 1] == "spike-count"
            assert tuple(map(float, args[args.index("--input-rates") + 1:])) == (
                exp022.VARIABLE_RATE_TRAINING_RATES_HZ
            )


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
                "--tau-gaba": "6.0", "--seed": "42", "--readout": "spike-count",
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
    samples = round(row["parameters"]["max_samples"] * 0.8)
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


def test_validator_recognizes_w_ff_readout_without_named_output_key(tmp_path: Path) -> None:
    row = _manifest_cell(tmp_path)
    directory = _write_valid_cell(row)
    checkpoint = torch.load(directory / "weights.pth", map_location="cpu", weights_only=True)
    checkpoint.pop("b_out")
    torch.save(checkpoint, directory / "weights.pth")
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
    status = campaign.summarize_status({
        "campaign_id": "test", "campaign_root": str(tmp_path),
        "cells": [complete, missing],
    })
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


def _attempt_manifest(tmp_path: Path, row: dict) -> dict:
    return {
        "campaign_id": "test", "manifest_sha256": "abc",
        "repository": {"commit": "deadbeef", "dirty": False},
        "campaign_root": str(tmp_path), "cells": [row],
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
    recovered, recovered_lock = campaign.acquire_attempt(manifest, row, recover_stale=True)
    assert recovered["attempt_id"] != record["attempt_id"]
    campaign.release_attempt(recovered_lock, recovered["attempt_id"])
    lock.unlink(missing_ok=True)


def test_failed_subprocess_without_metrics_records_failure(tmp_path: Path, monkeypatch) -> None:
    row = _manifest_cell(tmp_path)
    manifest = _attempt_manifest(tmp_path, row)
    monkeypatch.setattr(exp022, "_checked_manifest", lambda _path: manifest)
    monkeypatch.setattr(exp022, "_gpu_metadata", lambda: {"available": False})
    monkeypatch.setattr(
        exp022.subprocess, "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 7),
    )
    assert exp022._campaign_train(tmp_path / "campaign.json", row["name"]) == 1
    attempt = json.loads((Path(row["output_directory"]) / "attempt.json").read_text())
    assert attempt["state"] == "failed"
    assert attempt["exit_code"] == 7


def test_preserve_partial_avoids_timestamp_collision(tmp_path: Path, monkeypatch) -> None:
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


def _write_checked_manifest(tmp_path: Path, monkeypatch, tier: str = "variable_rate") -> Path:
    monkeypatch.setattr(campaign, "git_identity", lambda _repo: ("deadbeef", False))
    monkeypatch.setattr(campaign, "lock_identity", lambda _repo: {"path": "uv.lock", "sha256": "lock"})
    cells = exp022.cells_in_resource_tier(tier)
    payload = campaign.create_manifest(
        repo=exp022.REPO, campaign_root=tmp_path, campaign_id="checked",
        cells=cells, tier_for=exp022.cell_resource_tier,
        samples_epochs=exp022.cell_samples_epochs, build_args=exp022.build_train_args,
        selection_tier=tier,
    )
    path = tmp_path / "campaign.json"
    campaign.write_manifest(path, payload)
    return path


@pytest.mark.parametrize("mutation", [
    "command", "command_shell", "output_directory", "required_outputs",
    "duplicate", "missing",
])
def test_checked_manifest_rejects_rehashed_executable_mutations(
    tmp_path: Path, monkeypatch, mutation: str,
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


def test_external_aggregation_dispatches_external_training_root(tmp_path: Path, monkeypatch) -> None:
    manifest = {
        "campaign_id": "external", "campaign_root": str(tmp_path),
        "cells": [{} for _ in exp022.CANONICAL_CELLS],
    }
    monkeypatch.setattr(exp022, "_checked_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(campaign, "summarize_status", lambda _manifest: {
        "retry_cells": [],
        "cells": [{"name": "cell", "valid": True}],
    })
    observed = {}
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        observed.update(kwargs["env"])
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(exp022.subprocess, "run", fake_run)
    assert exp022._handle_campaign_cli([
        "exp022.py", "--campaign-aggregate", str(tmp_path / "campaign.json"),
    ])
    assert observed["PINGLAB_TRAINING_ROOT"] == str(tmp_path / "cells")
    assert exp022.training_root_provenance(tmp_path)["location"] == "external"
    assert [command[-2:] for command in commands[1:]] == [
        ["--plot-only", "appendix-rasters"],
        ["--plot-only", "comparison-rasters"],
    ]


def test_post_aggregation_check_allows_only_generated_exp022_artifacts(
    tmp_path: Path, monkeypatch,
) -> None:
    manifest_path = _write_checked_manifest(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign, "git_identity", lambda _repo: ("deadbeef", True))
    monkeypatch.setattr(
        campaign, "git_dirty_paths",
        lambda _repo: ["artifacts/data/exp022/numbers.json", "artifacts/pdfs/exp022.pdf"],
    )
    assert exp022._checked_manifest(
        manifest_path, allow_generated_dirty=True,
    )["campaign_id"] == "checked"
    monkeypatch.setattr(
        campaign, "git_dirty_paths", lambda _repo: ["experiments/exp022.py"],
    )
    with pytest.raises(SystemExit, match="clean source worktree"):
        exp022._checked_manifest(manifest_path, allow_generated_dirty=True)


@pytest.mark.parametrize("occupied", ["empty", "manifest", "cell", "status", "arbitrary"])
def test_campaign_creation_refuses_existing_destination(
    tmp_path: Path, monkeypatch, occupied: str,
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
        for path in root.rglob("*") if path.is_file()
    }
    monkeypatch.setattr(
        campaign, "create_manifest",
        lambda **_kwargs: {"campaign_id": "must-not-overwrite"},
    )
    with pytest.raises(SystemExit, match="already exists and will not be modified"):
        exp022._handle_campaign_cli([
            "exp022.py", "--campaign-manifest", str(root),
            "--campaign-id", "must-not-overwrite", "--tier", "variable_rate",
        ])
    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }
    assert after == before


def test_verified_archive_source_is_manifest_cells_not_legacy(tmp_path: Path, monkeypatch) -> None:
    row = _manifest_cell(tmp_path)
    manifest = _attempt_manifest(tmp_path, row)
    manifest["cells"] = [{} for _ in exp022.CANONICAL_CELLS]
    checked = {}

    def fake_checked_manifest(_path, *, allow_generated_dirty=False):
        checked["allow_generated_dirty"] = allow_generated_dirty
        return manifest

    monkeypatch.setattr(exp022, "_checked_manifest", fake_checked_manifest)
    monkeypatch.setattr(campaign, "summarize_status", lambda _manifest: {
        "retry_cells": [], "recoverable_cells": [],
        "cells": [{"state": "complete"} for _ in exp022.CANONICAL_CELLS],
    })
    selected, source = archive.verified_campaign_source(tmp_path / "campaign.json")
    assert selected is manifest
    assert source == (tmp_path / "cells").resolve()
    assert source != (archive.ARTIFACTS_ROOT / "exp022").resolve()
    assert checked["allow_generated_dirty"] is True


def test_mnist_link_helper_accepts_existing_and_concurrent_creation(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    (cache / "MNIST").mkdir(parents=True)
    link = tmp_path / "mnist"
    helper = exp022.REPO / "experiments" / "exp022_support" / "ensure-mnist-link.sh"
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
    helper = exp022.REPO / "experiments" / "exp022_support" / "load-wilkes-modules.sh"
    subprocess.run(
        [
            "env", "-i", f"PATH={Path('/usr/bin')}:/bin",
            f"EXP022_MODULES_INIT={initializer}",
            f"EXP022_MODULE_CALLS={calls}",
            "/bin/bash", "-c", f"source {helper}",
        ],
        check=True,
    )
    assert calls.read_text().splitlines() == ["purge", "load rhel8/default-amp"]


def test_submission_selection_is_frozen_read_only() -> None:
    submit = (
        exp022.REPO / "experiments" / "exp022_support" / "submit-tier.sh"
    ).read_text()
    array = (
        exp022.REPO / "experiments" / "exp022_support" / "train-array.sbatch"
    ).read_text()
    assert 'chmod 0444 "$selection"' in submit
    assert 'mapfile -t cells < "$EXP022_SELECTION"' in array
    assert "--campaign-list" not in array
