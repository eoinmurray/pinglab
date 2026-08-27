"""Exp024 contract and measurement checks using tiny retained histories, no simulation."""

import copy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.exp024 import analyse, collection, present, recipe
from pingstore import stages
from pingstore.contracts import (
    LEGACY_RUN_SCHEMA,
    RUN_SCHEMA,
    PingstoreError,
    load_json,
    payload_digest,
    validate_run_directory,
    write_json_atomic,
)
from pingstore.discovery import discover_runs
from pingstore.layout import initialize_layout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(analyse, "REPO", tmp_path)
    monkeypatch.setattr(present, "REPO", tmp_path)
    monkeypatch.setattr(stages, "memberships", lambda repo: {"exp022": "demo", "exp024": "demo"})
    monkeypatch.setattr(stages, "_capture_code", lambda *args: {"git_commit": "fixture", "dirty": False})
    return tmp_path


def config(model, seed):
    return {
        "training_cell_name": recipe.cell_name(model, seed), "training_run_id": "TR-02",
        "seed": seed, "dataset": "mnist", "readout_mode": "mem-mean",
        "fr_reg_upper_strength": 0, "ei_strength": 0 if model == "coba" else 1,
        "epochs": 12, "t_ms": 200, "dt": .1, "batch_size": 256,
        "max_samples": 7000, "n_in": 784, "n_hidden": 1024, "n_inh": 256,
        "n_out": 10, "input_rate": 25, "v_grad_dampen": 1 if model == "coba" else 1000,
        "lr": .0004, "validation_encoder_draws": {"count": 3},
        "dataset_split": {"checkpoint_selection_partition": "validation",
                          "official_test_used_during_training": False,
                          "optimizer_train_samples": 6300, "validation_samples": 700,
                          "official_test_samples": 10000},
    }


def make_compute(repo, *, schema=RUN_SCHEMA, mutation=None):
    identity = "exp022-r001-compute-local"
    directory = repo / ".pingstore/runs" / identity
    initialize_layout(directory, "exp022", schema=schema)
    for model in recipe.MODELS:
        for seed in recipe.SEEDS:
            cfg = config(model, seed)
            metrics = {"epochs": [{
                "ep": epoch, "acc": 90 + epoch * .02, "loss": 1 / epoch,
                "test_loss": .4 + .01 / epoch, "test_rate_e": (100 + epoch) if model == "coba" else (10 + epoch * .08),
                "test_rate_i": 0 if model == "coba" else 60 + epoch,
                "weight_norms": {"W_ff.0": 2 + epoch, "W_ff.1": 50 + epoch * .1},
            } for epoch in range(1, 13)]}
            if mutation:
                mutation(cfg, metrics)
            cell = directory / "export/cells" / recipe.cell_name(model, seed)
            write_json_atomic(cell / "config.json", cfg)
            write_json_atomic(cell / "metrics.json", metrics)
    write_json_atomic(directory / "run.json", {
        "schema": schema, "run_id": identity, "experiment": "exp022", "collection": "demo",
        "origin": "local", "stage": "compute", "inputs": {}, "export_root": "export/cells",
        "created_at": "2026-08-27T12:00:00+00:00", "execution": {}, "provenance": {},
        "payload_digest": payload_digest(directory),
    })
    return identity, directory


def resign(directory):
    record = load_json(directory / "run.json")
    record["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", record)


@pytest.mark.parametrize("schema", [RUN_SCHEMA, LEGACY_RUN_SCHEMA])
def test_stages_pin_sources_preserve_measurements_and_never_publish(repo, schema):
    identity, bank = make_compute(repo, schema=schema)
    before = payload_digest(bank)
    analysis_id = analyse.analyse(identity)
    analysis_root = repo / ".pingstore/runs" / analysis_id
    results = load_json(analysis_root / "export/results.json")
    assert results["models"]["coba"]["final_e_rate_hz"]["mean"] == 112
    assert results["models"]["ping"]["e_rate_slope_last10_hz_per_ep"]["mean"] == pytest.approx(.08)
    assert results["models"]["ping"]["e_rate_converged_count"] == 0
    assert results["models"]["ping"]["accuracy_converged_count"] == 3
    assert results["config"]["epochs"] == 12
    assert results["cells"][0]["weight_drift"]["W_ff.0"]["first_epoch"] == 3
    assert discover_runs(repo / ".pingstore/runs") == []
    assert not (analysis_root / "export/numbers.json").exists()
    presentation_id = present.present(analysis_id)
    output = repo / ".pingstore/runs" / presentation_id
    record = validate_run_directory(output)
    numbers = load_json(output / "export/numbers.json")
    assert numbers["models"] == results["models"]
    assert numbers["cells"] == results["cells"]
    assert record["inputs"]["analysis"]["run_id"] == analysis_id
    assert record["inputs"]["compute"]["payload_digest"] == before
    assert all((output / "export" / name).is_file() for name in recipe.FIGURES)
    assert "validation accuracy" in (output / "export/coba_curves.svg").read_text()
    assert presentation_id not in (output / "export/coba_curves.svg").read_text()
    assert not (output / "presentation").exists()
    assert not (repo / ".artifacts").exists()
    assert payload_digest(bank) == before
    assert discover_runs(repo / ".pingstore/runs")[0]["id"] == presentation_id


@pytest.mark.parametrize("mutation,match", [
    (lambda c, m: m["epochs"][0].pop("test_rate_e"), "nonfinite"),
    (lambda c, m: m["epochs"][0].update(test_loss=float("nan")), "nonfinite"),
    (lambda c, m: m["epochs"].pop(), "incomplete"),
    (lambda c, m: m["epochs"][2].update(ep=2), "contiguous"),
    (lambda c, m: c.update(fr_reg_upper_strength=.041), "fr_reg_upper_strength"),
    (lambda c, m: c.pop("dataset_split"), "validation split"),
    (lambda c, m: c.update(seed=999), "seed"),
])
def test_invalid_scientific_evidence_leaves_only_hidden_failure(repo, mutation, match):
    identity, _ = make_compute(repo, mutation=mutation)
    with pytest.raises(PingstoreError, match=match):
        analyse.analyse(identity)
    assert not list((repo / ".pingstore/runs").glob("exp024-*"))
    assert list((repo / ".pingstore/runs").glob(".exp024-*.tmp"))
    assert not (repo / ".artifacts").exists()


def test_missing_cell_and_tampered_sources_are_rejected(repo):
    identity, bank = make_compute(repo)
    target = bank / "export/cells/coba__off__seed42/metrics.json"
    target.unlink()
    with pytest.raises(PingstoreError, match="checksum"):
        analyse.analyse(identity)
    resign(bank)
    with pytest.raises((PingstoreError, FileNotFoundError)):
        analyse.analyse(identity)


def test_present_rejects_wrong_stage_and_rechecks_compute_pin(repo):
    identity, bank = make_compute(repo)
    with pytest.raises(PingstoreError, match="not a analyse"):
        present.present(identity)
    analysis_id = analyse.analyse(identity)
    cfg = bank / "export/cells/coba__off__seed42/config.json"
    cfg.write_text(cfg.read_text() + "\n")
    resign(bank)
    with pytest.raises(PingstoreError, match="checksum changed"):
        present.present(analysis_id)


def test_diagnostics_use_endpoint_secant_absolute_threshold_and_first_crossing():
    values = [90, 100, 0, 0, 0, 0, 0, 0, 0, 90.45]
    assert recipe.slope_last_n(values) == pytest.approx(.05)
    assert recipe.accuracy_marker([90, 20, 90]) == 1
    assert recipe.accuracy_marker([0, 0]) is None
    with pytest.raises(ValueError):
        recipe.slope_last_n([1])
    cell = {"name": "fixture", "model": "ping", "seed": 42, "epochs": [
        {"acc": 90, "test_rate_e": 20 - i, "test_rate_i": 1, "loss": 1,
         "test_loss": 1, "weight_norms": {key: 1 for key in recipe.PARAMETERS}}
        for i in range(10)]}
    assert analyse.diagnose(cell)["e_rate_converged"] is False


def test_presentation_does_not_invoke_analysis(repo, monkeypatch):
    identity, _ = make_compute(repo)
    analysis_id = analyse.analyse(identity)
    monkeypatch.setattr(analyse, "analyse", lambda *a, **k: pytest.fail("upstream execution"))
    monkeypatch.setattr(analyse, "read_cell", lambda *a, **k: pytest.fail("training cell read"))
    present.present(analysis_id)


def test_collection_dispatch_tracks_runs_without_materialization(repo, monkeypatch):
    identity, _ = make_compute(repo)
    manifest = repo / "campaign/campaign.json"
    write_json_atomic(manifest, {"pingstore_run_id": identity})
    row = {"execution": {"mode": "exp024-staged"},
           "paths": {"state": str(repo / "campaign/downstream/exp024")},
           "required_outputs": [str(repo / "campaign/downstream/exp024/stage-refs.json")]}
    plan = {"exp022_manifest": str(manifest)}
    reserved = collection.reserve(repo, row, origin="slurm-wilkes")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        source = command[command.index("--source") + 1]
        run_id = command[command.index("--run-id") + 1]
        function = analyse.analyse if command[2].endswith("analyse") else present.present
        function(source, run_id=run_id)
        return SimpleNamespace(stdout=run_id + "\n")

    monkeypatch.setattr(collection.subprocess, "run", run)
    refs = collection.execute(repo, plan, row)
    assert refs["analyse"]["run_id"] == reserved["analyse"]
    assert refs["present"]["run_id"] == reserved["present"]
    assert len(calls) == 2
    assert collection.completed(repo, plan, row).record["stage"] == "present"
    assert not (repo / ".artifacts").exists()
    assert not (repo / "campaign/derived").exists()
    old = copy.deepcopy(row)
    old["execution"] = {"mode": "monolithic"}
    with pytest.raises(PingstoreError, match="legacy"):
        collection.execute(repo, plan, old)


def test_collection_retries_presentation_with_fresh_id_and_reuses_analysis(repo, monkeypatch):
    identity, _ = make_compute(repo)
    manifest = repo / "campaign/campaign.json"
    write_json_atomic(manifest, {"pingstore_run_id": identity})
    row = {"execution": {"mode": "exp024-staged"},
           "paths": {"state": str(repo / "campaign/exp024")},
           "required_outputs": [str(repo / "campaign/exp024/stage-refs.json")]}
    plan = {"exp022_manifest": str(manifest)}
    calls = []
    original_plot = present.plot_model_curves

    def run(command, **kwargs):
        module = command[2]
        calls.append(module)
        source = command[command.index("--source") + 1]
        run_id = command[command.index("--run-id") + 1]
        function = analyse.analyse if module.endswith("analyse") else present.present
        function(source, run_id=run_id)
        return SimpleNamespace(stdout=run_id + "\n")

    def fail(*args):
        raise RuntimeError("plot failure")

    monkeypatch.setattr(collection.subprocess, "run", run)
    monkeypatch.setattr(present, "plot_model_curves", fail)
    with pytest.raises(RuntimeError, match="plot failure"):
        collection.execute(repo, plan, row)
    refs_before = load_json(Path(row["required_outputs"][0]))
    failed = next((repo / ".pingstore/runs").glob(".exp024-*-present-*.tmp"))
    failed_digest = payload_digest(failed)
    monkeypatch.setattr(present, "plot_model_curves", original_plot)
    refs = collection.execute(repo, plan, row)
    assert refs["analyse"] == refs_before["analyse"]
    assert calls.count("experiments.exp024.analyse") == 1
    assert refs["present"]["run_id"] not in failed.name
    assert payload_digest(failed) == failed_digest
    assert collection.completed(repo, plan, row).record["run_id"] == refs["present"]["run_id"]


@pytest.mark.parametrize("entrypoint", ["analyse.py", "present.py"])
def test_cli_requires_explicit_source_and_resolves_outside_repo(tmp_path, entrypoint):
    script = Path(__file__).parents[1] / "exp024" / entrypoint
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "--source" in result.stdout
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert "--source" in result.stderr
    assert not (tmp_path / ".pingstore").exists()
