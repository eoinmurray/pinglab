from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
from experiments.exp022 import campaign, recipe, reuse_contract
from pingstore.contracts import PingstoreError, file_sha256, payload_digest
from pingstore.stages import SourceRun


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True))


def _directory(repo: Path) -> Path:
    return repo / ".pingstore/runs/exp022-r001-compute"


def _repin(repo: Path, monkeypatch) -> None:
    directory = _directory(repo)
    record = json.loads((directory / "run.json").read_text())
    record["payload_digest"] = payload_digest(directory)
    _json(directory / "run.json", record)
    monkeypatch.setattr(reuse_contract, "SOURCE_REFERENCE", {
        "run_id": record["run_id"], "payload_digest": record["payload_digest"],
    })


@pytest.fixture
def bank(tmp_path: Path, monkeypatch) -> Path:
    directory = _directory(tmp_path)
    export = directory / "export"
    export.mkdir(parents=True)
    for cell in recipe.LEGACY_CANONICAL_CELLS:
        unit = export / cell["name"]
        unit.mkdir()
        samples = cell.get("max_samples", 7000)
        arguments = recipe.build_train_args(cell, unit, samples, 50)
        scientific = recipe.scientific_contract(dict(cell, dt_ms=0.1), samples, 50)
        scientific["dynamics"] = {
            "dt_ms": cell["dt_ms"], "tau_ampa_ms": 2.0,
            "tau_gaba_ms": cell["tau_gaba"], "presentation_duration_ms": 200.0,
        }
        parameters = campaign.resolved_parameters(cell, arguments, samples, 50, scientific)
        for flag in ("--refractory-e-ms", "--refractory-i-ms", "--refractory-policy"):
            parameters["arguments"].pop(flag)
        config = campaign._expected_config({"parameters": parameters})
        config.update({
            "training_cell_name": cell["name"], "training_run_id": cell["training_run_id"],
            "campaign_resolved_parameters": parameters,
            "campaign_id": "historical-repair", "campaign_manifest_sha256": "b" * 64,
            "campaign_repository_commit": reuse_contract.REPAIRED_COMMIT,
            "git_sha": reuse_contract.REPAIRED_COMMIT[:8],
        })
        repaired = cell["family"] == "low_w_in" or (
            cell["family"] == "activity_frontier" and cell.get("rate_target_hz") is not None
        )
        if not repaired:
            config["git_sha"] = reuse_contract.INHERITED_COMMIT[:8]
            config["imported_cell_provenance"] = {
                "repository_commit": reuse_contract.INHERITED_COMMIT,
                "campaign_id": "original-gold2", "campaign_manifest_sha256": "a" * 64,
                "source_directory": f"/historical/{cell['name']}",
            }
        metrics = copy.deepcopy(config)
        metrics["config"] = copy.deepcopy(config)
        for key in ("model", "hidden_sizes", "surrogate_slope"):
            metrics["config"].pop(key)
        metrics["best_epoch"] = 25
        metrics["epochs"] = [{"ep": ep, "samples": round(samples * 0.9)} for ep in range(1, 51)]
        metrics["checkpoints"] = {}
        for role, filename, epoch in (
            ("best_validation", "weights.pth", 25), ("final_epoch", "weights_final.pth", 50),
        ):
            (unit / filename).write_bytes(f"{cell['name']}:{role}".encode())
            metrics["checkpoints"][role] = {
                "filename": filename, "epoch": epoch, "sha256": file_sha256(unit / filename),
            }
        _json(unit / "config.json", config)
        _json(unit / "metrics.json", metrics)
    (directory / "README.md").write_text("# Historical scientific training bank\n")
    _json(directory / "run.json", {
        "schema": "pingstore.run/v4", "run_id": "exp022-r001-compute", "experiment": "exp022",
        "stage": "compute", "collection": "gamma-gated-sparsity", "origin": "local",
        "created_at": "2026-08-27T09:29:36+00:00", "execution": {"operation": "import"},
        "provenance": {}, "inputs": {}, "payload_digest": "sha256:" + "0" * 64,
        "scientific_execution": {
            "cells": 102, "inherited_cells": 60, "retrained_cells": 42, "origin": "slurm",
        },
    })
    _repin(tmp_path, monkeypatch)
    return tmp_path


def test_inspection_preserves_source_and_separates_roles_and_training_lineage(bank: Path) -> None:
    directory = _directory(bank)
    before = {str(path.relative_to(directory)): path.read_bytes()
              for path in directory.rglob("*") if path.is_file()}
    source, plan = reuse_contract.inspect_source(bank)
    assert plan["source"] == source.reference
    assert [len(plan[key]) for key in (
        "reused_cells", "new_cells", "replaced_cells", "diagnostics", "per_cell",
    )] == [90, 12, 12, 34, 90]
    assert set(plan["reused_cells"]).isdisjoint(plan["new_cells"])
    assert set(plan["new_cells"]) & set(plan["replaced_cells"]) == {
        f"ping__dt0p05__seed{seed}" for seed in (42, 43, 44)
    }
    assert "ping__dt0p1__seed42" in plan["reused_cells"]
    assert "ping__dt0p25__seed42" not in plan["diagnostics"]
    old = plan["per_cell"]["ping__canonical__seed42"]
    assert old["original_training"]["repository_commit"] == reuse_contract.INHERITED_COMMIT
    assert old["retained_provenance"]["campaign_repository_commit"] == reuse_contract.REPAIRED_COMMIT
    repaired = plan["per_cell"]["ping__rt5hz__seed42"]
    assert repaired["original_training"]["repository_commit"] == reuse_contract.REPAIRED_COMMIT
    assert old["checkpoint_roles"]["best_validation"]["epoch"] == 25
    assert old["checkpoint_roles"]["final_epoch"]["epoch"] == 50
    assert old["files"]["weights.pth"]["sha256"] != old["files"]["weights_final.pth"]["sha256"]
    assert plan["source_operation"] == {
        "origin": "local", "operation": "import", "scientific_origin": "slurm",
    }
    assert old["refractory_interpretation"]["e_ms"] == pytest.approx(1.2)
    assert old["refractory_interpretation"]["i_ms"] == pytest.approx(0.6)
    assert len(json.dumps(plan)) > 0
    assert before == {str(path.relative_to(directory)): path.read_bytes()
                      for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("damage", ["missing_cell", "unexpected_cell", "extra_role", "missing_final"])
def test_rejects_incomplete_or_extra_scientific_payload(bank: Path, monkeypatch, damage: str) -> None:
    export = _directory(bank) / "export"
    unit = export / "ping__dt0p1__seed42"
    if damage == "missing_cell":
        shutil.rmtree(unit)
    elif damage == "unexpected_cell":
        shutil.copytree(unit, export / "duplicate-ping")
    elif damage == "extra_role":
        (unit / "unexpected.txt").write_text("unexpected scientific content")
    else:
        (unit / "weights_final.pth").unlink()
    _repin(bank, monkeypatch)
    with pytest.raises(PingstoreError, match="102-cell registry|four retained"):
        reuse_contract.inspect_source(bank)


@pytest.mark.parametrize("damage", [
    "seed", "dt", "refractory", "parameters", "nested_config", "identity", "original_commit",
    "inherited_as_repaired", "epoch_history", "samples", "checkpoint_hash", "checkpoint_epoch",
    "checkpoint_roles",
])
def test_rejects_incompatible_cells_even_with_a_recomputed_fixture_pin(
    bank: Path, monkeypatch, damage: str,
) -> None:
    unit = _directory(bank) / "export/ping__dt0p1__seed42"
    config = json.loads((unit / "config.json").read_text())
    metrics = json.loads((unit / "metrics.json").read_text())
    if damage in {"seed", "dt"}:
        config[damage] = 999
    elif damage == "refractory":
        config["refractory_e_ms"] = 3.0
    elif damage == "parameters":
        for payload in (config, metrics):
            payload["campaign_resolved_parameters"]["arguments"]["--lr"] = "0.1"
    elif damage == "nested_config":
        metrics["config"]["n_hidden"] = 256
    elif damage == "identity":
        for payload in (config, metrics):
            payload["training_cell_name"] = "ping__dt0p1__seed43"
    elif damage == "original_commit":
        config["git_sha"] = reuse_contract.REPAIRED_COMMIT[:8]
    elif damage == "inherited_as_repaired":
        for payload in (config, metrics):
            payload.pop("imported_cell_provenance")
    elif damage == "epoch_history":
        metrics["epochs"][-1]["ep"] = 49
    elif damage == "samples":
        metrics["epochs"][-1]["samples"] = 100
    elif damage == "checkpoint_hash":
        (unit / "weights_final.pth").write_bytes(b"different checkpoint")
    elif damage == "checkpoint_epoch":
        metrics["checkpoints"]["final_epoch"]["epoch"] = 25
    else:
        metrics["checkpoints"]["final_epoch"]["filename"] = "weights.pth"
    _json(unit / "config.json", config)
    _json(unit / "metrics.json", metrics)
    _repin(bank, monkeypatch)
    with pytest.raises(PingstoreError, match="ping__dt0p1__seed42"):
        reuse_contract.inspect_source(bank)


def test_exact_source_pin_rejects_an_otherwise_valid_replacement(bank: Path) -> None:
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(reuse_contract, "SOURCE_REFERENCE", {
            "run_id": "exp022-r001-compute", "payload_digest": "sha256:" + "f" * 64,
        })
        with pytest.raises(PingstoreError, match="identity or checksum changed"):
            reuse_contract.inspect_source(bank)
    finally:
        monkeypatch.undo()


def test_payload_drift_fails_before_inspection(bank: Path) -> None:
    (_directory(bank) / "export/ping__dt0p1__seed42/weights.pth").write_bytes(b"drift")
    with pytest.raises(PingstoreError, match="payload checksum mismatch"):
        reuse_contract.inspect_source(bank)


def test_source_checked_again_after_inspection(bank: Path, monkeypatch) -> None:
    def drift(source):
        (source.unit("ping__dt0p1__seed42") / "weights.pth").write_bytes(b"late drift")
        check(source)

    check = SourceRun.check_unchanged
    monkeypatch.setattr(SourceRun, "check_unchanged", drift)
    with pytest.raises(PingstoreError, match="payload checksum mismatch"):
        reuse_contract.inspect_source(bank)


@pytest.mark.parametrize("damage", ["duplicate", "new_name", "new_dt", "new_seed", "reused_definition"])
def test_registry_drift_cannot_change_the_approved_partition(bank: Path, monkeypatch, damage: str) -> None:
    cells = copy.deepcopy(recipe.CANONICAL_CELLS)
    if damage == "duplicate":
        cells.append(copy.deepcopy(cells[0]))
    elif damage == "new_name":
        next(cell for cell in cells if cell["name"] == "ping__dt0p2__seed42")["name"] = "unexpected"
    elif damage in {"new_dt", "new_seed"}:
        cell = next(cell for cell in cells if cell["name"] == "ping__dt0p2__seed42")
        cell["dt_ms" if damage == "new_dt" else "seed"] = 0.25 if damage == "new_dt" else 43
    else:
        cells[0]["seed"] = 999
    monkeypatch.setattr(recipe, "CANONICAL_CELLS", cells)
    with pytest.raises(PingstoreError, match="unique cells|partition|definition changed"):
        reuse_contract.inspect_source(bank)


def test_run_lineage_counts_cannot_relabel_the_source(bank: Path) -> None:
    path = _directory(bank) / "run.json"
    record = json.loads(path.read_text())
    record["scientific_execution"]["retrained_cells"] = 102
    _json(path, record)
    with pytest.raises(PingstoreError, match="original training lineage"):
        reuse_contract.inspect_source(bank)
