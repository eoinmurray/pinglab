"""Focused unit tests for the standalone EXP081 analytical runner."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.exp081 import analyse, collection, compute, present
from experiments.exp081 import recipe as exp081
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


def test_zero_drive_has_zero_operating_point_and_variance() -> None:
    _, voltage = exp081.linear_operating_point(0.0, 1.2)
    variance = exp081.predicted_variance(np.asarray([0.0]), np.asarray([1.2]))
    assert float(voltage) == exp081.PARAMETERS["E_L_mV"]
    assert variance[0] == 0.0


def test_finite_window_has_first_zero_at_five_hz() -> None:
    transfer = exp081.complete_transfer(np.asarray([5.0]), 3.0, 1.2)
    assert abs(transfer[0]) < 1e-10


def test_empirical_simulator_replays_and_zero_drive_rests() -> None:
    rates_hz = np.asarray([0.0, 2.5])
    probes = np.asarray([1.2, 1.2])
    first = compute.simulate_features(rates_hz, probes, 8, 123)
    replay = compute.simulate_features(rates_hz, probes, 8, 123)
    assert np.array_equal(first, replay)
    assert np.all(first[0] == 0.0)
    assert np.all(first[1] >= 0.0)


@pytest.fixture
def stage_repo(tmp_path, monkeypatch):
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setenv("EXP081_DEVICE", "cpu")
    monkeypatch.setattr(stages, "memberships", lambda repo: {"exp081": "demo"})
    monkeypatch.setattr(
        stages, "_capture_code", lambda *args: {"git_commit": "fixture", "dirty": False}
    )
    cfg = exp081.configuration(smoke=True)
    cfg.update(
        input_rate_grid_hz=[0.0, 3.0, 25.0],
        moment_draws=8,
        distribution_draws=16,
        frequency_plot_points=40,
    )
    monkeypatch.setattr(exp081, "configuration", lambda **kwargs: cfg)

    def fixture_samples(rates, probes, draws, seed, **kwargs):
        rates, probes = np.broadcast_arrays(rates, probes)
        return (rates[..., None] * probes[..., None] * np.linspace(0, 1, draws)).astype(
            np.float32
        )

    monkeypatch.setattr(compute, "simulate_features", fixture_samples)
    return tmp_path, cfg


def run_directory(repo, identity):
    return repo / ".pingstore/runs" / identity


def test_independent_stages_pin_v3_and_preserve_measurements(stage_repo, monkeypatch):
    repo, cfg = stage_repo
    compute_id = compute.compute()
    upstream = run_directory(repo, compute_id)
    compute_record = validate_run_directory(upstream)
    assert compute_record["schema"] == RUN_SCHEMA
    assert compute_record["inputs"] == {}
    before = payload_digest(upstream)
    assert sorted(p.name for p in (upstream / "export").iterdir()) == [
        "distribution_samples.npz",
        "feature_samples.npz",
    ]
    monkeypatch.setattr(
        compute, "simulate_features", lambda *a, **k: pytest.fail("upstream simulation")
    )
    analysis_id = analyse.analyse(compute_id)
    analysis_root = run_directory(repo, analysis_id)
    with (
        np.load(upstream / "export/feature_samples.npz") as raw,
        np.load(analysis_root / "export/moments.npz") as moments,
    ):
        np.testing.assert_array_equal(
            moments["empirical_mean_mV"], raw["samples_mV"].mean(axis=-1)
        )
        np.testing.assert_array_equal(
            moments["empirical_sd_mV"], raw["samples_mV"].std(axis=-1, ddof=1)
        )
    with np.load(analysis_root / "export/histograms.npz") as histogram:
        np.testing.assert_allclose(histogram["probability"].sum(axis=1), 1)
    assert discover_runs(repo / ".pingstore/runs") == []
    monkeypatch.setattr(
        analyse, "analyse", lambda *a, **k: pytest.fail("upstream analysis")
    )
    monkeypatch.setattr(
        exp081,
        "complete_transfer",
        lambda *a, **k: pytest.fail("new analytical calculation"),
    )
    present_id = present.present(analysis_id)
    output = run_directory(repo, present_id)
    record = validate_run_directory(output)
    assert record["schema"] == RUN_SCHEMA
    assert record["stage"] == "present"
    assert record["inputs"]["analysis"]["run_id"] == analysis_id
    assert record["inputs"]["compute"]["payload_digest"] == before
    assert all(p.is_file() for p in (output / "export").iterdir())
    assert len(list((output / "export").glob("*.svg"))) == 4
    assert load_json(output / "export/numbers.json") == load_json(
        analysis_root / "export/results.json"
    )
    assert record["execution"]["configuration"] == cfg
    assert payload_digest(upstream) == before
    assert not (repo / ".artifacts").exists()
    assert [r["id"] for r in discover_runs(repo / ".pingstore/runs")] == [present_id]


@pytest.mark.parametrize("stage", ["compute", "analyse"])
def test_downstream_rejects_v2_even_when_typed(stage_repo, stage):
    repo, cfg = stage_repo
    identity = f"exp081-r001-{stage}-local"
    directory = run_directory(repo, identity)
    initialize_layout(directory, "exp081", schema=LEGACY_RUN_SCHEMA)
    write_json_atomic(
        directory / "run.json",
        {
            "schema": LEGACY_RUN_SCHEMA,
            "run_id": identity,
            "experiment": "exp081",
            "collection": "demo",
            "origin": "local",
            "stage": stage,
            "inputs": {},
            "created_at": "2026-08-27T12:00:00+00:00",
            "provenance": {},
            "execution": {"configuration": cfg},
            "payload_digest": payload_digest(directory),
        },
    )
    validate_run_directory(directory)
    with pytest.raises(PingstoreError, match="requires v4"):
        (analyse.analyse if stage == "compute" else present.present)(identity)
    assert len(list((repo / ".pingstore/runs").iterdir())) == 1


def test_wrong_stage_and_tampered_evidence_are_rejected(stage_repo):
    repo, _ = stage_repo
    identity = compute.compute()
    with pytest.raises(PingstoreError, match="not a analyse"):
        present.present(identity)
    analysis_id = analyse.analyse(identity)
    root = run_directory(repo, identity)
    manifest = load_json(root / "run.json")
    manifest["execution"]["configuration"]["seed"] = 999
    write_json_atomic(root / "run.json", manifest)
    with pytest.raises(PingstoreError, match="recipe or compute lineage"):
        present.present(analysis_id)
    with (root / "export/feature_samples.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(PingstoreError, match="checksum"):
        analyse.analyse(identity)


def test_failed_analysis_remains_hidden(stage_repo):
    repo, _ = stage_repo
    identity = compute.compute()
    root = run_directory(repo, identity)
    path = root / "export/feature_samples.npz"
    with np.load(path) as data:
        arrays = dict(data)
    arrays["samples_mV"][0, 0, 0] = np.nan
    np.savez_compressed(path, **arrays)
    record = load_json(root / "run.json")
    record["payload_digest"] = payload_digest(root)
    write_json_atomic(root / "run.json", record)
    with pytest.raises(PingstoreError, match="sample shape"):
        analyse.analyse(identity)
    assert not list((repo / ".pingstore/runs").glob("exp081-*-analyse"))
    assert list((repo / ".pingstore/runs").glob(".exp081-*-analyse.tmp"))


def test_zero_sample_summary_has_no_nonfinite_json():
    summary = analyse.summarize(np.zeros(3), np.zeros(3))
    assert summary == {
        "pearson_r": None,
        "mean_absolute_error_mV": 0.0,
        "median_predicted_empirical_ratio": None,
    }


@pytest.mark.parametrize("name", ["compute", "analyse", "present"])
def test_stage_cli_help_and_explicit_sources(tmp_path, name):
    script = Path(__file__).parents[1] / "exp081" / f"{name}.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--run-id" in result.stdout
    if name != "compute":
        result = subprocess.run(
            [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True
        )
        assert result.returncode != 0 and "--source" in result.stderr
    assert not (tmp_path / ".pingstore").exists()


def test_retired_entrypoints_fail_without_execution(tmp_path):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "experiments.exp081"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requires independent stages" in result.stderr


def test_collection_reserves_dispatches_and_resumes_without_v2_capture(
    stage_repo, monkeypatch
):
    from experiments.collections.gamma_gated_sparsity import execution
    from experiments.collections.gamma_gated_sparsity.plan import build_plan

    repo, _ = stage_repo
    plan = build_plan(repo / "campaign", "fixture", smoke=True)
    plan["profile"] = "smoke"
    row = next(
        row
        for stage in plan["stages"]
        for row in stage["experiments"]
        if row["slug"] == "exp081"
    )
    assert row["execution"]["mode"] == "exp081-staged"
    identities = collection.reserve(repo, row, origin="slurm-wilkes")
    for stage, identity in identities.items():
        assert identity.endswith("-" + stage)
        reservation = load_json(
            repo
            / ".pingstore/runs"
            / f".{identity}.tmp"
            / ".reservation.json"
        )
        assert reservation["origin"] == "slurm-wilkes"
    calls = []

    def dispatch(command, **kwargs):
        calls.append(command)
        stage = command[2].rsplit(".", 1)[1]
        run_id = command[command.index("--run-id") + 1]
        assert kwargs["env"]["PINGLAB_SMOKE"] == "1"
        if stage == "compute":
            identity = compute.compute(run_id=run_id)
        else:
            source = command[command.index("--source") + 1]
            identity = (analyse.analyse if stage == "analyse" else present.present)(
                source, run_id=run_id
            )
        return SimpleNamespace(stdout=identity + "\n")

    monkeypatch.setattr(collection.subprocess, "run", dispatch)
    refs = collection.execute(repo, plan, row)
    assert len(calls) == 3
    assert refs["compute"]["run_id"] == identities["compute"]
    assert (
        collection.completed(repo, plan, row).record["run_id"] == identities["present"]
    )
    assert collection.execute(repo, plan, row) == refs
    assert len(calls) == 3
    assert not (repo / ".artifacts").exists()
    monkeypatch.setattr(execution, "REPO", repo)
    assert execution._outputs_valid_for_plan(plan, row)
    legacy = {**row, "execution": {"mode": "monolithic"}}
    assert not execution._outputs_valid_for_plan(plan, legacy)
    with pytest.raises(PingstoreError, match="not conformant"):
        collection.execute(repo, plan, legacy)
