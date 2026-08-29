"""EXP099 conformance using synthetic recordings, never production simulation."""

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from experiments.exp099 import analyse, compute, present, recipe
from pingstore import stages
from pingstore.contracts import (
    LEGACY_RUN_SCHEMA,
    PingstoreError,
    load_json,
    payload_digest,
    validate_operational_run_directory,
    write_json_atomic,
)
from pingstore.discovery import discover_runs
from pingstore.layout import initialize_layout


@pytest.fixture
def lab(tmp_path, monkeypatch):
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(stages, "memberships", lambda _: {"exp099": "demo"})
    monkeypatch.setattr(
        stages, "_capture_code", lambda *a: {"git_commit": "fixture", "dirty": False}
    )
    cfg = recipe.configuration()
    cfg.update(dt_ms=1.0, n_e=20, n_i=5)
    monkeypatch.setattr(recipe, "configuration", lambda *a: cfg)

    def synthetic_recording(output, bundle):
        root = output / "simulation"
        root.mkdir()
        t = np.arange(2000)[:, None]
        e = (t + np.arange(20)) % 25 == 0
        i = (t + np.arange(5)) % 25 == 5
        data = {
            "dt": 1.0,
            "spk_e": e,
            "spk_i": i,
            "v_e_1": -60 + np.broadcast_to(np.sin(t / 25), e.shape),
            "v_i_1": -60 + np.broadcast_to(np.cos(t / 25), i.shape),
            "ge_e_1": np.broadcast_to(1 + np.sin(t / 25) / 2, e.shape),
            "gi_e_1": np.broadcast_to(1 + np.cos(t / 25) / 2, e.shape),
            "input_afferent_shared": e,
            "input_afferent_e_private": e,
            "input_afferent_i_private": e,
            "input_structured_spikes_e": e,
            "input_structured_spikes_i": e,
            "input_weather_scale": np.ones(2000),
            "input_afferent_shared_scale": np.ones(2000),
        }
        for population, spikes in (("e", e), ("i", i)):
            for channel in ("excitatory", "inhibitory"):
                for kind in ("private", "shared", "executed"):
                    data[f"input_{channel}_{population}_{kind}"] = spikes * 0.01
        np.savez_compressed(root / "snapshot.npz", **data)
        write_json_atomic(
            root / "config.json", {"_simulation_recipe": cfg["simulation"]}
        )
        np.savez_compressed(
            root / "recurrent-weights.npz",
            w_ee=np.eye(20) * 0.85,
            w_ei=np.full((20, 5), 0.6),
            w_ie=np.full((5, 20), 3.0),
            w_ii=np.eye(5) * 0.4,
            w_in_e=np.eye(20) * 0.08,
            w_in_i=np.full((20, 5), 0.02),
        )
        (output / "network.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20"/>'
        )
        return ["synthetic-fixture"]

    monkeypatch.setattr(compute, "simulate", synthetic_recording)
    return tmp_path


def directory(repo, identity):
    return repo / ".pingstore/runs" / identity


def forbid(*args, **kwargs):
    pytest.fail("downstream stage launched upstream work")


def test_stages_pin_v3_keep_raw_evidence_and_render_without_analysis(lab, monkeypatch):
    renderer = importlib.import_module("experiments.exp099.render")
    compute_id = compute.compute()
    upstream = directory(lab, compute_id)
    before = payload_digest(upstream)
    assert compute_id == "exp099-r001-compute"
    monkeypatch.setattr(compute, "simulate", forbid)
    analysis_id = analyse.analyse(compute_id)
    analysis_root = directory(lab, analysis_id)
    with np.load(analysis_root / "export/measurements.npz") as metrics:
        assert metrics["rhythm_centres"][[0, -1]].tolist() == [200, 1800]
        with np.load(upstream / "export/simulation/snapshot.npz") as raw:
            np.testing.assert_array_equal(metrics["mean_g_e"], raw["ge_e_1"].mean(1))
        # The scalar summary intentionally excludes the renderer's last window.
        summary = load_json(analysis_root / "export/results.json")["results"][
            "richer-input"
        ]
        contrasts = metrics["rhythm_contrast"][:-1]
        assert summary["peak_rhythmicity"] == contrasts.max()
        assert (
            summary["peak_rhythmicity_time_ms"]
            == metrics["rhythm_centres"][np.argmax(contrasts)]
        )
    assert discover_runs(lab / ".pingstore/runs") == []
    monkeypatch.setattr(analyse, "measure", forbid)
    monkeypatch.setattr(analyse, "rhythmicity_metrics", forbid)
    monkeypatch.setattr(analyse, "rolling_conductance_loop_score", forbid)

    def sample_frames(fig, update, output, **kwargs):
        assert kwargs == {"frames": 600, "fps": 25, "bitrate": 3800}
        for frame in (0, 139, 140, 309, 310, 479, 480, 599):
            update(frame)
            fig.canvas.draw()
        assert len(fig.axes) == 7
        # Encoding is outside this fixture test; the real poster is rendered.
        output.write_bytes(b"fixture-video")

    monkeypatch.setattr(renderer, "save_animation", sample_frames)
    present_id = present.present(analysis_id)
    output = directory(lab, present_id)
    record = validate_operational_run_directory(output)
    assert record["stage"] == "present"
    assert record["inputs"]["compute"]["payload_digest"] == before
    assert record["inputs"]["analysis"]["run_id"] == analysis_id
    assert all(path.is_file() for path in (output / "export").iterdir())
    assert (output / "export" / recipe.POSTER).stat().st_size > 1000
    assert load_json(output / "export/numbers.json") == load_json(
        analysis_root / "export/results.json"
    )
    assert payload_digest(upstream) == before
    assert [row["id"] for row in discover_runs(lab / ".pingstore/runs")] == [present_id]
    assert not (lab / ".artifacts").exists() and not (lab / "assets").exists()


@pytest.mark.parametrize("stage", ["compute", "analyse"])
def test_v2_inputs_are_rejected_before_reservation(lab, stage):
    identity = f"exp099-r001-{stage}-local"
    root = directory(lab, identity)
    initialize_layout(root, "exp099", schema=LEGACY_RUN_SCHEMA)
    write_json_atomic(
        root / "run.json",
        {
            "schema": LEGACY_RUN_SCHEMA,
            "run_id": identity,
            "experiment": "exp099",
            "collection": "demo",
            "origin": "local",
            "stage": stage,
            "inputs": {},
            "created_at": "2026-08-28T12:00:00+00:00",
            "provenance": {},
            "execution": {},
            "payload_digest": payload_digest(root),
        },
    )
    with pytest.raises(PingstoreError, match="requires v4"):
        (analyse.analyse if stage == "compute" else present.present)(identity)
    assert list(root.parent.iterdir()) == [root]


def test_wrong_stage_payload_and_manifest_tampering_are_rejected(lab):
    identity = compute.compute()
    with pytest.raises(PingstoreError, match="not a analyse"):
        present.present(identity)
    analysis_id = analyse.analyse(identity)
    root = directory(lab, identity)
    record = load_json(root / "run.json")
    record["execution"]["configuration"]["seed"] = 999
    write_json_atomic(root / "run.json", record)
    with pytest.raises(PingstoreError, match="recipe or compute lineage"):
        present.present(analysis_id)
    with (root / "export/simulation/snapshot.npz").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(PingstoreError, match="checksum"):
        analyse.analyse(identity)


def test_failed_presentation_stays_hidden_and_source_unchanged(lab, monkeypatch):
    identity = compute.compute()
    analysis_id = analyse.analyse(identity)
    root = directory(lab, identity)
    before = payload_digest(root)

    def fail_render(*args):
        raise RuntimeError("render failed")

    monkeypatch.setattr(present, "render", fail_render)
    with pytest.raises(RuntimeError, match="render failed"):
        present.present(analysis_id)
    assert payload_digest(root) == before
    assert not list(root.parent.glob("exp099-*-present"))
    assert list(root.parent.glob(".exp099-*-present.tmp"))


def test_reserved_identity_and_missing_recorded_inputs(lab):
    reserved = stages.reserve_stage(
        lab / ".pingstore", "exp099", "compute", origin="slurm-test"
    )
    assert compute.compute(run_id=reserved) == reserved
    root = directory(lab, reserved)
    record = validate_operational_run_directory(root)
    assert record["origin"] == "slurm-test"
    # A correctly checksummed but scientifically incomplete fixture must fail,
    # rather than reconstructing inputs in the presentation stage.
    snapshot = root / "export/simulation/snapshot.npz"
    with np.load(snapshot) as data:
        arrays = dict(data)
    del arrays["input_afferent_shared"]
    np.savez_compressed(snapshot, **arrays)
    record["payload_digest"] = payload_digest(root)
    write_json_atomic(root / "run.json", record)
    with pytest.raises(ValueError, match="input_afferent_shared"):
        analyse.analyse(reserved)
    assert not list(root.parent.glob("exp099-*-analyse"))


def test_analysis_payload_cannot_disagree_with_authoritative_settings(lab):
    identity = compute.compute()
    analysis_id = analyse.analyse(identity)
    root = directory(lab, analysis_id)
    path = root / "export/results.json"
    result = load_json(path)
    result["measurements"]["loop_window_ms"] = 999
    write_json_atomic(path, result)
    record = load_json(root / "run.json")
    record["payload_digest"] = payload_digest(root)
    write_json_atomic(root / "run.json", record)
    with pytest.raises(PingstoreError, match="inconsistent exp099 analysis"):
        present.present(analysis_id)
    assert not list(root.parent.glob(".exp099-*-present.tmp"))


@pytest.mark.parametrize("name", ["compute", "analyse", "present"])
def test_cli_help_and_explicit_sources(tmp_path, name):
    script = Path(__file__).parents[1] / "exp099" / f"{name}.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--run-id" in result.stdout
    if name != "compute":
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0 and "--source" in result.stderr
    assert not (tmp_path / ".pingstore").exists()


def test_import_and_retired_entrypoints_never_dispatch(tmp_path):
    root = Path(__file__).resolve().parents[2]
    expression = (
        "import sys; from unittest.mock import patch; "
        f"sys.path[:0] = [{str(root)!r}, {str(root / 'tools')!r}]; "
        "guard = patch('subprocess.run', side_effect=AssertionError('dispatch')); "
        "guard.start(); "
        "from experiments.exp099 import recipe, compute, analyse, present"
    )
    result = subprocess.run(
        [sys.executable, "-c", expression],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [sys.executable, "-m", "experiments.exp099"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0 and "requires independent stages" in result.stderr
