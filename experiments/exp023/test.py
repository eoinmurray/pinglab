from __future__ import annotations

from pathlib import Path

"""Pipeline fixtures and bounded graph/kernel numerical conformance checks."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from experiments.exp023 import analyse, compute, inputs, present, recipe
from pingstore import stages
from pingstore.contracts import (
    PingstoreError,
    load_json,
    payload_digest,
    write_json_atomic,
)
from pingstore.discovery import discover_runs
from pingstore.layout import initialize_layout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for module in (compute, analyse, present):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(stages, "memberships", lambda _: {"exp023": "demo"})
    monkeypatch.setattr(
        stages, "_capture_code", lambda *args: {"git_commit": "fixture", "dirty": False}
    )
    monkeypatch.setattr(recipe, "N_E", 4)
    monkeypatch.setattr(recipe, "N_I", 2)
    monkeypatch.setenv("PINGLAB_SMOKE", "1")
    calls = []

    def simulate(spec):
        calls.append(spec)
        point = spec.input_bindings[0]
        steps = point.steps_count
        e = torch.zeros(steps, 1, recipe.N_E)
        i = torch.zeros(steps, 1, recipe.N_I)
        e[::100, 0, 1] = 1
        parameters = {row["id"]: row for row in spec.graph["parameters"]}
        if parameters["E_to_I.weight"]["initializer"]["mean"]:
            i[5::100, 0, 0] = 1
        diagnostics = {
            "spk_e": e,
            "spk_i": i,
            "v_e": torch.full_like(e, -60),
            "v_i": torch.full_like(i, -55),
            "ge_e": torch.full_like(e, 0.01),
            "ge_i": torch.full_like(i, 0.03),
            "gi_e": torch.full_like(e, 0.02 if i.any() else 0),
        }
        return SimpleNamespace(
            diagnostics=diagnostics if spec.diagnostics else {},
            outputs={"spk_e_count": e.sum(0), "spk_i_count": i.sum(0)},
            parameters={"input_to_E.weight": torch.zeros(recipe.N_IN, recipe.N_E)},
            metrics={"execution_protocol": {"fixture": True}},
        )

    monkeypatch.setattr(compute, "simulate", simulate)
    return tmp_path, calls


def resign(directory):
    record = load_json(directory / "run.json")
    record["payload_digest"] = payload_digest(directory)
    write_json_atomic(directory / "run.json", record)


def test_independent_v4_stages_preserve_measurements_and_never_publish(
    repo, monkeypatch
):
    root, calls = repo
    compute_id = compute.compute()
    assert len(calls) == 16
    source = inputs.source(root, compute_id, "compute")
    assert source.record["inputs"] == {}
    assert not list(source.export.rglob("run.sh"))
    before = source.reference
    monkeypatch.setenv("PINGLAB_SMOKE", "0")
    monkeypatch.setattr(
        compute, "simulate", lambda *a, **k: pytest.fail("downstream simulation")
    )
    analysis_id = analyse.analyse(compute_id)
    analysis = inputs.source(root, analysis_id, "analyse")
    results = load_json(analysis.export / "results.json")
    assert results["config"]["profile"] == "smoke"
    assert results["config"]["drive"]["fi_sweep"]["t_ms"] == 200
    assert results["raster"]["coba"]["e_rate_hz"] == pytest.approx(25)
    assert results["raster"]["ping"]["i_rate_hz"] == pytest.approx(50)
    assert results["raster"]["ping"]["e_index"] == 1
    assert results["f_gamma_hz"]["coba"] is None
    with np.load(analysis.export / "traces.npz") as data:
        np.testing.assert_allclose(data["ping__ii_e"], -0.4)
        np.testing.assert_allclose(data["ping__ie_e"], 0.6)
    assert discover_runs(root / ".pingstore/runs") == []
    monkeypatch.setattr(
        analyse, "population_psd", lambda *a: pytest.fail("presentation remeasurement")
    )
    monkeypatch.setattr(
        analyse, "select_traces", lambda *a: pytest.fail("presentation selection")
    )
    presentation_id = present.present(analysis_id)
    output = inputs.source(root, presentation_id, "present")
    assert output.record["inputs"] == {
        "analysis": analysis.reference,
        "compute": before,
    }
    assert (
        load_json(output.export / "numbers.json")["fi_curves"] == results["fi_curves"]
    )
    assert all(p.is_file() for p in output.export.iterdir())
    assert (output.export / "overview_compound.png").is_file()
    assert (output.export / "traces__ping__i_i.svg").is_file()
    assert not (output.directory / "presentation").exists()
    assert not (root / ".artifacts").exists()
    assert inputs.source(root, compute_id, "compute").reference == before
    assert [row["id"] for row in discover_runs(root / ".pingstore/runs")] == [
        presentation_id
    ]


def test_failure_leaves_hidden_run_and_never_runs_downstream(repo, monkeypatch):
    root, _ = repo

    def fail(*a, **k):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(compute, "simulate", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        compute.compute()
    assert not list((root / ".pingstore/runs").glob("exp023-*"))
    assert len(list((root / ".pingstore/runs").glob(".exp023-*.tmp"))) == 1
    with pytest.raises(PingstoreError):
        analyse.analyse(".exp023-r001-compute-local.tmp")


def test_sources_and_authoritative_pins_are_validated(repo):
    root, _ = repo
    identity = compute.compute()
    with pytest.raises(PingstoreError, match="not a analyse"):
        present.present(identity)
    analysis_id = analyse.analyse(identity)
    source = inputs.source(root, identity, "compute")
    record_path = source.directory / "run.json"
    record = load_json(record_path)
    record["execution"]["note"] = "changed manifest, unchanged payload"
    write_json_atomic(record_path, record)
    assert present.present(analysis_id).startswith("exp023-")


def test_missing_or_corrupt_snapshot_is_not_silently_recomputed(repo):
    root, calls = repo
    identity = compute.compute()
    source = inputs.source(root, identity, "compute")
    source.file("scope", "ping", "recording.npz").write_bytes(b"bad fixture")
    with pytest.raises(PingstoreError, match="checksum"):
        analyse.analyse(identity)
    resign(source.directory)
    with pytest.raises(ValueError):
        analyse.analyse(identity)
    assert len(calls) == 16


def test_unsupported_schema_is_rejected_before_scientific_consumption(repo):
    root, _ = repo
    identity = "exp023-r001-compute"
    directory = root / ".pingstore/runs" / identity
    initialize_layout(directory, "exp023")
    write_json_atomic(
        directory / "run.json",
        {
            "schema": "pingstore.run/v3",
            "run_id": identity,
            "experiment": "exp023",
            "stage": "compute",
            "inputs": {},
            "origin": "local",
            "collection": "demo",
            "created_at": "2026-08-27T12:00:00+00:00",
            "execution": {},
            "provenance": {},
            "payload_digest": payload_digest(directory),
        },
    )
    with pytest.raises(PingstoreError, match="operational run schema"):
        analyse.analyse(identity)


@pytest.mark.parametrize("flag", [[], ["--plot-only"], ["--skip-training"]])
def test_retired_launcher_rejects_all_combined_modes(flag):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "experiments.exp023", *flag],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "independent stages" in result.stderr


def test_spectrum_and_selection_preserve_original_rules():
    spikes = np.zeros((4000, 4), dtype=bool)
    spikes[::250, :] = True
    _, _, peak = analyse.population_psd(spikes, 0.1, (5, 150))
    assert peak == pytest.approx(40, abs=0.01)
    assert analyse.pick_active(spikes) == 0
    assert analyse.pick_active(np.zeros_like(spikes)) is None
    assert analyse.population_psd(np.zeros_like(spikes), 0.1, (5, 150))[2] is None


@pytest.mark.parametrize("silent", [False, True])
def test_native_compact_recordings_preserve_trace_selection_and_rates(tmp_path, silent):
    cfg = recipe.configuration(smoke=True)
    cfg.update(n_e=4, n_i=2)
    point = {"t_ms": 10, "dt_ms": 0.1, "n_in": 3}
    rng = np.random.default_rng(42)
    e = (rng.random((100, 4)) < 0.1) & (not silent)
    i = (rng.random((100, 2)) < 0.1) & (not silent)
    diagnostics = {"spk_e": torch.tensor(e[:, None]), "spk_i": torch.tensor(i[:, None])}
    for population, size in (("e", 4), ("i", 2)):
        for signal in ("v", "ge", "gi") if population == "e" else ("v", "ge"):
            diagnostics[f"{signal}_{population}"] = torch.tensor(
                rng.uniform(-70, -50, (100, 1, size))
                if signal == "v"
                else rng.uniform(0, 0.2, (100, 1, size))
            )
    result = SimpleNamespace(
        diagnostics=diagnostics,
        outputs={
            "spk_e_count": torch.tensor(e.sum(0)[None]),
            "spk_i_count": torch.tensor(i.sum(0)[None]),
        },
    )
    np.savez_compressed(
        tmp_path / "scope.npz", **compute.recording(result, cfg, point, traces=True)
    )
    scope = analyse.snapshot(tmp_path / "scope.npz", cfg, point, traces=True)
    traces, selected = analyse.select_traces(scope, cfg["biophysics"])
    assert selected["e_index"] == int(e.sum(0).argmax())
    assert selected["i_index"] == (None if silent else int(i.sum(0).argmax()))
    np.testing.assert_array_equal(
        traces["v_e"], diagnostics["v_e"][:, 0, selected["e_index"]].numpy()
    )
    np.savez_compressed(
        tmp_path / "counts.npz", **compute.recording(result, cfg, point, traces=False)
    )
    counts = analyse.snapshot(tmp_path / "counts.npz", cfg, point)
    for population, spikes in (("e", e), ("i", i)):
        assert counts[f"spk_{population}_count"] == spikes.sum()
    # A dense sweep is rejected rather than silently accepted through an older interface.
    np.savez_compressed(
        tmp_path / "dense.npz", dt=0.1, T=100, n_e=4, n_i=2, spk_e=e, spk_i=i
    )
    with pytest.raises(PingstoreError, match="unsupported native recording fields"):
        analyse.snapshot(tmp_path / "dense.npz", cfg, point)
    scope.pop("v_e_selected")
    np.savez_compressed(tmp_path / "missing.npz", **scope)
    with pytest.raises(PingstoreError, match="unsupported native recording fields"):
        analyse.snapshot(tmp_path / "missing.npz", cfg, point, traces=True)


def test_geometry_and_duration_are_explicit_in_graph_requests():
    for smoke, duration in ((False, 400), (True, 200)):
        cfg = recipe.configuration(smoke=smoke)
        trials = list(recipe.trials(smoke=smoke))
        assert len(trials) == 16
        assert {point["n_in"] for _, _, point, traces in trials if traces} == {1024}
        assert {point["n_in"] for _, _, point, traces in trials if not traces} == {784}
        for _, _, point, traces in trials:
            bundle = recipe.author_network(cfg, point, traces=traces)
            request = recipe.execution_request(bundle, point, traces=traces)
            assert request.executor == "graph"
            assert request.input_bindings[0].steps_count == duration * 10
            assert request.input_bindings[0].rates_hz == (point["input_rate_hz"],)
        assert {
            point["input_rate_hz"] for _, _, point, traces in trials if not traces
        } == set(recipe.FI_RATES_HZ)


@pytest.mark.parametrize("schema", ["exp023.recipe/v1", "exp023.recipe/v2"])
def test_old_recipes_are_rejected_before_downstream_run_creation(repo, schema):
    root, _ = repo
    identity = compute.compute()
    analysis_id = analyse.analyse(identity)
    source = inputs.source(root, identity, "compute")
    record = load_json(source.directory / "run.json")
    record["execution"]["configuration"]["schema"] = schema
    write_json_atomic(source.directory / "run.json", record)
    before = set((root / ".pingstore/runs").iterdir())
    for action, argument in (
        (analyse.analyse, identity),
        (present.present, analysis_id),
    ):
        with pytest.raises(PingstoreError, match="requires native graph recipe v3"):
            action(argument)
    assert set((root / ".pingstore/runs").iterdir()) == before


@pytest.mark.parametrize("cell", recipe.CELLS)
def test_graph_matches_independent_kernel_schedule(monkeypatch, cell):
    from snnlab.sim.execution import GraphExecutor, plan_graph
    from snnlab.sim.models import fast_sigmoid_spike, lif_step_expeuler

    cfg = recipe.configuration()
    cfg.update(n_e=16, n_i=4)
    point = recipe.operating_point(cell, 100, 24)
    point["t_ms"] = 60
    bundle = recipe.author_network(cfg, point, traces=True)
    model = GraphExecutor(plan_graph(bundle.graph), seed=42)
    parameters = model.parameter_map()
    drive = torch.zeros(600, 1, 24)
    drive[::2] = 1
    voltage_e, voltage_i = torch.full((1, 16), -65.0), torch.full((1, 4), -65.0)
    ref_e, ref_i = torch.zeros((1, 16), dtype=torch.long), torch.zeros((1, 4), dtype=torch.long)
    spikes_e, spikes_i = torch.zeros(1, 16), torch.zeros(1, 4)
    ge_e, ge_i, gi_e = torch.zeros(1, 16), torch.zeros(1, 4), torch.zeros(1, 16)
    histories = {name: [] for name in ("spk_e", "spk_i", "v_e", "v_i", "ge_e", "ge_i", "gi_e")}
    ampa, gaba = np.exp(-0.1 / 2.0), np.exp(-0.1 / 6.0)

    def spike(voltage):
        return fast_sigmoid_spike(voltage + 50.0, 5.0)

    # This explicit physical schedule checks graph lowering, recurrent delay,
    # synaptic decay and diagnostic routing independently of GraphExecutor.
    with torch.inference_mode():
        native = model({"drive": drive})
        for step in range(600):
            ge_e = ge_e * ampa + drive[step] @ parameters["input_to_E.weight"]
            ge_i = ge_i * ampa + spikes_e @ parameters["E_to_I.weight"]
            gi_e = gi_e * gaba + spikes_i @ parameters["I_to_E.weight"]
            voltage_e, spikes_e, ref_e = lif_step_expeuler(
                voltage_e, ref_e, ge_e, gi_e, 1.0, 0.05, 12, spike,
                v_grad_dampen=cfg.get("v_grad_dampen", 80.0), dt_override=0.1,
            )
            voltage_i, spikes_i, ref_i = lif_step_expeuler(
                voltage_i, ref_i, ge_i, None, 0.5, 0.1, 6, spike,
                v_grad_dampen=cfg.get("v_grad_dampen", 80.0), dt_override=0.1,
            )
            for name, value in {
                "spk_e": spikes_e, "spk_i": spikes_i,
                "v_e": voltage_e, "v_i": voltage_i,
                "ge_e": ge_e, "ge_i": ge_i, "gi_e": gi_e,
            }.items():
                histories[name].append(value.clone())
    for name, values in histories.items():
        torch.testing.assert_close(native.diagnostics[name], torch.stack(values),
            rtol=0, atol=0 if name.startswith("spk") else 2e-5)
    assert native.diagnostics["spk_e"].any()
    assert bool(native.diagnostics["spk_i"].any()) == (cell == "ping")
    if cell == "ping":
        conductance = native.diagnostics["gi_e"][:, 0]
        silent_previous_i = ~native.diagnostics["spk_i"][:-1, 0].any(1).bool()
        torch.testing.assert_close(conductance[1:][silent_previous_i],
            conductance[:-1][silent_previous_i] * gaba, rtol=0, atol=1e-7)


def test_native_sweep_reductions_match_full_raster_counts():
    from snnlab.sim.execution import simulate

    cfg = recipe.configuration()
    cfg.update(n_e=16, n_i=4)
    point = recipe.operating_point("ping", 100, 24)
    point["t_ms"] = 20
    with torch.inference_mode():
        scope = simulate(
            recipe.execution_request(
                recipe.author_network(cfg, point, traces=True), point, traces=True
            )
        )
        sweep = simulate(
            recipe.execution_request(
                recipe.author_network(cfg, point, traces=False), point, traces=False
            )
        )
    full = compute.recording(scope, cfg, point, traces=True)
    reduced = compute.recording(sweep, cfg, point, traces=False)
    for population in ("e", "i"):
        assert reduced[f"spk_{population}_count"] == full[f"spk_{population}"].sum()
    assert sweep.diagnostics == {}
    assert len(sweep.metrics["online_reductions"]) == 2
    assert all(
        count == 0 for count in sweep.metrics["retained_signal_samples"].values()
    )


def test_recipe_v3_records_graph_protocol_without_cli_arguments():
    cfg = recipe.configuration()
    assert cfg["schema"] == "exp023.recipe/v3"
    assert len(list(recipe.trials())) == 16
    assert all(
        "scientific_args" not in point
        for point in cfg["drive"]["raster_operating_points"].values()
    )
    point = cfg["drive"]["raster_operating_points"]["ping"]
    bundle = recipe.author_network(cfg, point, traces=True)
    populations = {row["id"]: row for row in bundle.graph["populations"]}
    assert populations["E"]["neuron"]["refractory_steps"] == 12
    assert populations["I"]["neuron"]["refractory_steps"] == 6
    projections = {row["id"]: row for row in bundle.graph["projections"]}
    assert projections["I_to_E"]["synapse"]["tau"] == {"value": 6.0, "unit": "ms"}
    assert all(
        row["delay"]["value"] == 0.1
        for row in projections.values()
        if row["connection"] == "recurrent"
    )


def test_new_recipe_drift_is_rejected(repo):
    root, _ = repo
    identity = compute.compute()
    run = inputs.source(root, identity, "compute")
    record = load_json(run.directory / "run.json")
    record["execution"]["configuration"]["biophysics"]["tau_gaba_ms"] = 9.0
    write_json_atomic(run.directory / "run.json", record)
    with pytest.raises(PingstoreError, match="recipe differs"):
        analyse.analyse(identity)
