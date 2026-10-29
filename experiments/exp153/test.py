"""Phase classification and source-to-closure regression checks.

Synthetic rates isolate amplitude decay, phase drift and invalid-rhythm handling.
A small native graph checks four delayed populations and effective-weight scaling
without consuming historical runs or running a parameter sweep.
"""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from experiments.exp153 import analyse, compute, present, recipe
from snnlab import lang
from snnlab import meanfield as mf


@pytest.fixture
def fixture():
    cfg = recipe.configuration()
    bundle = recipe.author_network(cfg)
    weights = {
        projection["id"]: 0.12 if "_K_E" in projection["id"] else 0.4
        for projection in bundle.graph["projections"]
    }
    return bundle, recipe.configuration(incoming_weights_us=weights)


def waveform(frequency=50, amplitude=4, phase=0, decay=None):
    time = np.arange(0, 5000.5, 0.5)
    envelope = amplitude if decay is None else amplitude * np.exp(-time / decay)
    rate = 20 + envelope * np.sin(2 * np.pi * frequency * time / 1000 + phase)
    return time, rate


def test_sustained_gamma_passes_both_windows(fixture):
    _, cfg = fixture
    time, rate = waveform()
    result = analyse.gate_metrics(
        time, {f"PING_{c}_E.rate": rate for c in ("A", "B")}, cfg
    )
    assert result["gate_passed"]
    assert result["circuits"]["A"]["late"]["frequency_hz"] == pytest.approx(50)


@pytest.mark.parametrize("amplitude,decay", [(0, None), (1e-5, None), (10, 600)])
def test_tonic_numerical_ripples_and_damped_transients_fail(fixture, amplitude, decay):
    _, cfg = fixture
    time, rate = waveform(amplitude=amplitude, decay=decay)
    result = analyse.gate_metrics(
        time, {f"PING_{c}_E.rate": rate for c in ("A", "B")}, cfg
    )
    assert not result["gate_passed"]


def test_one_failed_circuit_stops_gate(fixture):
    _, cfg = fixture
    time, rate = waveform()
    result = analyse.gate_metrics(
        time, {"PING_A_E.rate": rate, "PING_B_E.rate": rate * 0 + 20}, cfg
    )
    assert result["circuits"]["A"]["passed"]
    assert not result["gate_passed"]


@pytest.mark.parametrize("frequency,locked", [(50, True), (51, False)])
def test_phase_checks_fixed_offset_and_frequency_drift(fixture, frequency, locked):
    _, cfg = fixture
    time, a = waveform()
    _, b = waveform(frequency=frequency, phase=0.6)
    result = analyse.pair_metrics(time, {"PING_A_E.rate": a, "PING_B_E.rate": b}, cfg)
    assert result["valid"]
    assert result["locked"] is locked
    assert result["drift_hz"] == pytest.approx(50 - frequency, abs=0.01)


@pytest.mark.parametrize("offset,expected", [(np.pi / 2, -0.25), (3 * np.pi / 2, 0.25)])
def test_circular_mean_phase_sign_and_wrapping(fixture, offset, expected):
    _, cfg = fixture
    time, a = waveform()
    _, b = waveform(phase=offset)
    result = analyse.pair_metrics(time, {"PING_A_E.rate": a, "PING_B_E.rate": b}, cfg)
    assert result["locked"]
    assert result["mean_phase_cycles"] == pytest.approx(expected, abs=1e-10)


def test_phase_trace_preserves_accumulated_cycles_and_observed_support(fixture):
    _, cfg = fixture
    time, a = waveform(frequency=50)
    _, b = waveform(frequency=51)
    trace, details = analyse.relative_phase_trace(
        time, {"PING_A_E.rate": a, "PING_B_E.rate": b}, cfg
    )
    t, phase = np.asarray(trace["time_ms"]), np.asarray(trace["relative_phase_cycles"])
    assert t[0] > 0 and cfg["duration_ms"] - 25 < t[-1] <= cfg["duration_ms"]
    expected = (
        analyse.rhythm(time, a, [0, cfg["duration_ms"]], cfg["phase"])[1]
        - analyse.rhythm(time, b, [0, cfg["duration_ms"]], cfg["phase"])[1]
    ) / (2 * np.pi)
    np.testing.assert_allclose(phase, expected[np.isfinite(expected)])
    assert phase[-1] < -1
    assert np.polyfit(t / 1000, phase, 1)[0] == pytest.approx(-1, abs=0.01)
    assert details["onset_ms"] == 500


def test_tonic_diagnostic_keeps_rates_without_fabricating_phase(fixture):
    _, cfg = fixture
    time, rate = waveform(amplitude=0)
    state = {f"PING_{c}_{p}.rate": rate for c in ("A", "B") for p in ("E", "I")}
    onset, trace, rates = analyse.diagnostic(time, state, cfg)
    assert not onset["coupled"]["valid"] and not onset["coupled"]["locked"]
    assert trace == dict(time_ms=[], relative_phase_cycles=[])
    assert rates["time_ms"][-1] == 1500
    np.testing.assert_array_equal(rates["A_E"], np.full(3001, 20))


def test_initialized_sums_scaled_once_and_delays_and_onset_retained(fixture):
    bundle, cfg = fixture
    original = copy.deepcopy(bundle.graph)
    model = recipe.build_model(bundle, cfg, difference_hz=-4, coupling_us=0.06)
    assert len(model.states) == 18
    assert model.defaults["drive_A.rate"] == 23
    assert model.defaults["drive_B.rate"] == 27
    assert not any("attention" in name for name in model.defaults)
    for circuit in ("A", "B"):
        for population in ("E", "I"):
            name = f"PING_{circuit}_{population}_to_{population}"
            assert model.defaults[name + ".coupling"] == 0.4
            assert model.projection_delays[name] == 0.1
    assert model.defaults["PING_A_input_to_E.coupling"] == pytest.approx(0.4 * 9.2)
    assert model.defaults["PING_A_E_to_PING_B_I_K_EI.coupling"] == 0.06
    assert model.projection_delays["PING_A_E_to_I"] == 0.1
    assert model.projection_delays["PING_A_E_to_PING_B_I_K_EI"] == 2
    assert len(model.controls) == 4
    for control in model.controls.values():
        assert control(499.99) == 0 and control(500) == 1
    assert bundle.graph == original
    with pytest.raises(ValueError, match="characteristic-root"):
        mf.jacobian(model)


def test_native_matrix_orientation_and_zero_edges(fixture):
    bundle, _ = fixture
    graph = copy.deepcopy(bundle.graph)
    for population in graph["populations"]:
        population["size"] = 2
    arrays = {
        p["id"] + ".weight": torch.tensor([[0.0, 1.0], [2.0, 0.0], [3.0, 4.0]])
        for p in graph["projections"]
    }
    sums = compute.mean_incoming_weights(SimpleNamespace(graph=graph), arrays)
    assert set(sums.values()) == {5.0}


def test_short_native_dde_trajectory_and_tamper_rejection(fixture, tmp_path):
    bundle, cfg = fixture
    cfg["duration_ms"] = 3
    model = recipe.build_model(bundle, cfg)
    solution = compute.integrate(model, cfg)
    solution.write(tmp_path / "condition-000-trajectory.json")
    model.write(tmp_path / "condition-000-model.json")
    source = SimpleNamespace(file=lambda name: tmp_path / name)
    time, state = analyse.read_trajectory(
        source, {"label": "condition-000", "difference_hz": 0, "coupling_us": 0}, cfg
    )
    assert time[-1] == 3 and len(state) == 18
    path = tmp_path / "condition-000-trajectory.json"
    data = json.loads(path.read_text())
    data["integration"]["delay_mode"] = "omit"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="explicit-delay"):
        analyse.read_trajectory(
            source,
            {"label": "condition-000", "difference_hz": 0, "coupling_us": 0},
            cfg,
        )


def test_standalone_graph_contains_all_local_and_reciprocal_paths(fixture, tmp_path):
    from snnlab import viz

    bundle, cfg = fixture
    assert cfg["schema"] == recipe.SCHEMA
    assert {row["id"] for row in bundle.graph["inputs"]} == {
        "drive_A",
        "drive_B",
    }
    assert {row["id"]: row["size"] for row in bundle.graph["populations"]} == {
        f"PING_{c}_{p}": size for c in ("A", "B") for p, size in (("E", 80), ("I", 20))
    }
    assert len(bundle.graph["projections"]) == 14
    assert not any("attention" in row["id"] for row in bundle.graph["projections"])
    bundle.write(tmp_path / "network.bundle", visualise=False)
    assert lang.load_bundle(tmp_path / "network.bundle").graph == bundle.graph
    viz.render_diagram(present.network_diagram(bundle), tmp_path / "circuit.svg")
    assert (tmp_path / "circuit.svg").is_file()


def test_initialization_only_capture_and_weight_reproducibility(fixture, tmp_path):
    from experiments.exp153.custom.initialization import capture_start
    from snnlab.sim import NetworkSnapshot, Performance, build

    bundle, cfg = fixture
    built = build(
        graph=bundle.graph,
        seed=cfg["network"]["network_seed"],
        device="cpu",
        performance=Performance(compile_mode=None),
    )
    repeated = build(
        graph=bundle.graph,
        seed=cfg["network"]["network_seed"],
        device="cpu",
        performance=Performance(compile_mode=None),
    )
    for key, value in built.parameters.items():
        torch.testing.assert_close(value, repeated.parameters[key], rtol=0, atol=0)
    capture_start(bundle, built, cfg, tmp_path / "start")
    snapshot = NetworkSnapshot.read(tmp_path / "start", parameters=True)
    assert snapshot.graph_identity == bundle.manifest["graph_digest"]
    assert snapshot.execution["initialization_only"]
    assert snapshot.execution["completed_steps"] == 0
    sums = compute.mean_incoming_weights(bundle, snapshot.load_parameters())
    for circuit in ("A", "B"):
        assert sums[f"PING_{circuit}_E_to_E"] == pytest.approx(0.4 * 79 / 80)
        assert sums[f"PING_{circuit}_I_to_I"] == pytest.approx(0.8 * 19 / 20)
        assert sums[f"PING_{circuit}_input_to_E"] == pytest.approx(0.2, rel=0.01)
    assert (
        len(
            recipe.build_model(
                bundle, recipe.configuration(incoming_weights_us=sums)
            ).states
        )
        == 18
    )


def test_compute_preparation_needs_no_upstream_runs(monkeypatch, tmp_path):
    from contextlib import contextmanager

    export = tmp_path / "export"
    export.mkdir()
    seen = {}

    @contextmanager
    def stage(*args, **kwargs):
        seen.update(kwargs)
        yield SimpleNamespace(
            export=export,
            directory=tmp_path,
            run_id="exp153-r999-compute",
            record={"execution": {}},
        )

    def no_source(*args, **kwargs):
        raise AssertionError("standalone compute read an upstream run")

    monkeypatch.setattr(compute, "source_run", no_source)
    monkeypatch.setattr(compute, "stage_run", stage)
    monkeypatch.setattr(compute, "run_conditions", lambda *args: 143)
    assert compute.compute() == "exp153-r999-compute"
    assert seen["inputs"] == {}
    assert seen["configuration"]["schema"] == recipe.SCHEMA
    assert len(seen["configuration"]["incoming_weights_us"]) == 14
    assert (export / recipe.SNAPSHOT / "snapshot.json").is_file()


def test_no_cross_experiment_or_scratch_imports():
    import ast
    from pathlib import Path

    for path in Path(__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else []
                )
            )
            for module in modules:
                if module.startswith("experiments.exp"):
                    assert module.split(".")[1] == "exp153", (path, module)
        assert ".scratch/" not in path.read_text() if path.name != "test.py" else True


def test_stage_ancestry_rejects_foreign_runs_before_reading(monkeypatch, tmp_path):
    seen = []

    def source(store, identity, **kwargs):
        seen.append(identity)
        assert kwargs["experiment"] == recipe.SLUG
        return SimpleNamespace(
            record={"inputs": {"foreign": {"run_id": "exp000-r001-compute"}}}
        )

    monkeypatch.setattr(compute, "source_run", source)
    with pytest.raises(ValueError, match="within this experiment"):
        compute.ancestry(tmp_path, "exp153-r999-compute")
    assert seen == ["exp153-r999-compute"]


def test_qualitative_rhythms_need_not_be_gamma(fixture):
    _, cfg = fixture
    time, rate = waveform(frequency=25.3)
    state = {f"PING_{c}_E.rate": rate for c in ("A", "B")}
    assert analyse.gate_metrics(time, state, cfg)["gate_passed"]
    assert analyse.pair_metrics(time, state, cfg)["locked"]


def test_parallel_conditions_match_serial_native_integration(fixture, tmp_path):
    from pingstore.library import write_json_atomic

    bundle, cfg = fixture
    cfg["duration_ms"] = 3
    export = tmp_path / "export"
    export.mkdir()
    bundle.write(export / "network.bundle", visualise=False)
    record = dict(
        run_id="exp153-r999-compute",
        experiment="exp153",
        stage="compute",
        inputs={},
        execution=dict(configuration=cfg),
        provenance={},
    )
    write_json_atomic(tmp_path / "run.json", record)
    (tmp_path / ".writer.lock").touch()
    run = SimpleNamespace(directory=tmp_path, run_id=record["run_id"], record=record)
    items = [
        dict(
            id=f"cpu-{i}",
            conditions=[
                dict(label=f"condition-{i:03d}", difference_hz=0, coupling_us=0)
            ],
        )
        for i in range(2)
    ]
    workers = compute.run_active_workers(
        run=run,
        items=items,
        worker=compute.run_conditions,
        arguments=(cfg,),
        inventory=compute.condition_inventory,
    )
    assert len(workers) == len(run.record["execution"]["workers"]) == 2
    serial = compute.integrate(recipe.build_model(bundle, cfg), cfg)
    for item in items:
        _, state = analyse.read_trajectory(
            SimpleNamespace(file=lambda name: export / name), item["conditions"][0], cfg
        )
        for name in state:
            np.testing.assert_allclose(
                state[name], serial.state[name], rtol=1e-12, atol=1e-12
            )


def test_matched_grid_and_symmetric_relaxation(fixture):
    bundle, cfg = fixture
    assert cfg["protocol"] == "matched-grid"
    assert cfg["duration_ms"] == 5000
    assert cfg["closure"]["tau_rate_ms"]["PING_A_E"] == 5
    assert cfg["closure"]["tau_rate_ms"]["PING_B_E"] == 5
    assert cfg["example"]["difference_hz"] == -4
    assert cfg["example"]["coupling_us"] == 0.09
    rows = recipe.conditions(cfg)
    assert len(rows) == 143
    assert len({row["label"] for row in rows}) == 143
    assert sum(row["coupling_us"] == 0 for row in rows) == 11
    assert (
        sum(row["difference_hz"] == -4 and row["coupling_us"] == 0.09 for row in rows)
        == 1
    )
    model = recipe.build_model(bundle, cfg, difference_hz=-4, coupling_us=0.09)
    assert model.defaults["drive_A.rate"] == 23
    assert model.defaults["drive_B.rate"] == 27
    for kwargs in (dict(difference_hz=50), dict(difference_hz=1), dict(coupling_us=0)):
        with pytest.raises(ValueError):
            recipe.configuration(
                incoming_weights_us=cfg["incoming_weights_us"], **kwargs
            )


def grid_metrics(cfg):
    return [
        dict(
            **condition,
            valid=True,
            locked=True,
            concentration=1.0,
            rhythms=dict(
                A=dict(frequency_hz=50 + condition["difference_hz"] / 10),
                B=dict(frequency_hz=50 - condition["difference_hz"] / 10),
            ),
        )
        for condition in recipe.conditions(cfg)
    ]


def test_invalid_uncoupled_calibration_masks_entire_column(fixture):
    _, cfg = fixture
    metrics = grid_metrics(cfg)
    metrics[0]["valid"] = False
    arrays, cells = analyse.assemble_map(metrics, cfg)
    assert arrays["valid"].shape == (13, 11)
    assert not arrays["valid"][:, 0].any()
    assert not arrays["locked"][:, 0].any()
    assert np.isnan(arrays["detuning_hz"][0])
    assert np.isnan(arrays["concentration"][:, 0]).all()
    assert all(
        not row["valid"] and row["uncoupled_detuning_hz"] is None for row in cells[:13]
    )
    assert arrays["valid"][:, 1:].all()
    with pytest.raises(ValueError, match="grid coverage"):
        analyse.assemble_map(metrics[::-1], cfg)
    with pytest.raises(ValueError, match="grid coverage"):
        analyse.assemble_map(metrics[:-1], cfg)


def test_invalid_coupled_cell_does_not_mask_other_conditions(fixture):
    _, cfg = fixture
    metrics = grid_metrics(cfg)
    metrics[1]["valid"] = False
    arrays, _ = analyse.assemble_map(metrics, cfg)
    assert arrays["valid"].sum() == 142
    assert arrays["valid"][0, 0] and not arrays["valid"][1, 0]
    assert arrays["detuning_hz"][0] == pytest.approx(-2)


def test_onset_uses_exact_display_and_frequency_windows(fixture):
    _, cfg = fixture
    time, rate = waveform()
    state = {f"PING_{c}_{p}.rate": rate for c in ("A", "B") for p in ("E", "I")}
    onset, trace, rates = analyse.diagnostic(time, state, cfg)
    assert onset["coupled"]["locked"]
    assert onset["coupled"]["rhythms"]["A"]["window_ms"] == [800, 1500]
    for circuit in ("A", "B"):
        assert onset["frequencies"][circuit]["first"]["window_ms"] == [0, 500]
        assert onset["frequencies"][circuit]["last"]["window_ms"] == [1000, 1500]
    assert rates["time_ms"][0] == 0 and rates["time_ms"][-1] == 1500
    assert trace["time_ms"][-1] < 1500
    np.testing.assert_array_equal(rates["A_E"], rate[:3001])


def test_figures_match_layout_and_mask_invalid_cells(fixture, tmp_path):
    import matplotlib.pyplot as plt
    from experiments import theme

    _, cfg = fixture
    theme.apply()
    arrays, _ = analyse.assemble_map(grid_metrics(cfg), cfg)
    for invalid in (False, True):
        if invalid:
            arrays["detuning_hz"][0] = np.nan
            arrays["valid"][:, 0] = False
        fig = present.map_figure(arrays)
        fig.savefig(tmp_path / f"map-{invalid}.svg")
        assert np.ma.getmaskarray(fig.axes[0].collections[0].get_array()).sum() == (
            13 if invalid else 0
        )
        assert fig.axes[0].get_ylim() == (0, 0.12)
        plt.close(fig)
    time, rate = waveform()
    state = {f"PING_{c}_{p}.rate": rate for c in ("A", "B") for p in ("E", "I")}
    onset, trace, rates = analyse.diagnostic(time, state, cfg)
    fig = present.onset_figure(rates, trace, onset, cfg)
    assert len(fig.axes) == 3
    for axis in fig.axes:
        assert axis.get_xlim() == (0, 1500)
        assert np.array_equal(axis.lines[-1].get_xdata(), [500, 500])
    assert len(fig.axes[0].lines) == len(fig.axes[1].lines) == 3
    fig.savefig(tmp_path / "onset.svg")
    plt.close(fig)


def test_onset_only_limits_execution_to_selected_figure_two_condition():
    cfg = recipe.configuration(onset_only=True)
    assert cfg["duration_ms"] == cfg["example"]["display_window_ms"][1] == 1500
    assert cfg["protocol"] == "onset-only"
    assert recipe.conditions(cfg) == [
        dict(label="onset", difference_hz=-4.0, coupling_us=0.09)
    ]
    with pytest.raises(ValueError, match="explicit map source"):
        present.presentation_numbers({}, cfg, None)


def test_cross_delay_tuning_retains_onset_and_local_dynamics(fixture):
    _, base = fixture
    cfg = recipe.configuration(
        cross_delay_ms=5,
        cross_tau_ms=0.8,
        incoming_weights_us=base["incoming_weights_us"],
    )
    assert cfg["network"]["local"] == base["network"]["local"]
    model = recipe.build_model(recipe.author_network(cfg), cfg, coupling_us=0.09)
    for path, delay in model.projection_delays.items():
        if "_K_E" in path:
            assert delay == 5
            assert model.defaults[path + ".tau"] == 0.8
    for control in model.controls.values():
        assert control(499.99) == 0 and control(500) == 1
    with pytest.raises(ValueError, match="finite and positive"):
        recipe.configuration(cross_delay_ms=0)


def test_rate_tuning_reaches_native_model_without_changing_reference(fixture):
    bundle, base = fixture
    tuned = recipe.configuration(
        onset_only=True,
        tau_rate_e_ms=10,
        tau_rate_i_ms=4,
        incoming_weights_us=base["incoming_weights_us"],
    )
    assert tuned["network"] == base["network"]
    model = recipe.build_model(bundle, tuned, difference_hz=-4, coupling_us=0.09)
    for circuit in ("A", "B"):
        assert model.defaults[f"PING_{circuit}_E.tau_rate"] == 10
        assert model.defaults[f"PING_{circuit}_I.tau_rate"] == 4
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="relaxation times"):
            recipe.configuration(tau_rate_e_ms=value)


def test_feedback_tuning_is_authored_before_initialization():
    cfg = recipe.configuration(
        recurrent_ee_us=0.4,
        recurrent_ii_us=0.8,
        local_feedback_scale=1.5,
    )
    bundle = recipe.author_network(cfg)
    local = cfg["network"]["local"]
    assert local["E_to_I"]["mean_us"] == 0.75
    assert local["I_to_E"]["mean_us"] == 1.5
    for path in ("E_to_I", "I_to_E"):
        assert local[path]["std_us"] / local[path]["mean_us"] == pytest.approx(0.1)
    assert len(bundle.graph["projections"]) == 14
    for name in (
        "recurrent_ee_us",
        "recurrent_ii_us",
        "local_feedback_scale",
        "sensory_gain",
    ):
        with pytest.raises(ValueError, match="finite and positive"):
            recipe.configuration(**{name: 0})


def test_figure_two_preview_uses_pinned_rates_without_a_map(monkeypatch, tmp_path):
    from pingstore.library import write_json_atomic

    cfg = recipe.configuration(onset_only=True, tau_rate_e_ms=10, sensory_gain=4.6)
    time = np.arange(0, 1500.5, 0.5)
    state = {
        f"PING_{c}_{p}.rate": 20 + 4 * np.sin(2 * np.pi * 50 * time / 1000)
        for c in ("A", "B")
        for p in ("E", "I")
    }
    onset, trace, rates = analyse.diagnostic(time, state, cfg)
    analysis_cfg = recipe.analysis_configuration(cfg)
    write_json_atomic(
        tmp_path / "results.json", dict(configuration=analysis_cfg, onset=onset)
    )
    write_json_atomic(tmp_path / "onset-phase.json", trace)
    np.savez_compressed(tmp_path / "onset-rates.npz", **rates)
    verified = []
    computed = SimpleNamespace(
        record=dict(execution=dict(configuration=cfg)),
        reference=dict(run_id="exp153-r900-compute", payload_digest="compute-pin"),
        check_unchanged=lambda: verified.append("compute"),
    )
    analysed = SimpleNamespace(
        record=dict(
            stage="analyse",
            execution=dict(configuration=analysis_cfg),
            inputs=dict(compute=dict(run_id="exp153-r900-compute")),
        ),
        reference=dict(run_id="exp153-r901-analyse", payload_digest="analysis-pin"),
        file=lambda name: tmp_path / name,
        check_unchanged=lambda: verified.append("analyse"),
    )
    monkeypatch.setattr(
        present,
        "ancestry",
        lambda *args: {
            "exp153-r900-compute": computed,
            "exp153-r901-analyse": analysed,
        },
    )
    output = present.preview_onset("exp153-r901-analyse", tmp_path / "preview")
    assert {p.name for p in output.iterdir()} == {
        "coupling-onset.png",
        "coupling-onset.svg",
        "coupling-onset.pdf",
        "source.json",
    }
    assert json.loads((output / "source.json").read_text()) == dict(
        compute=computed.reference,
        analyse=analysed.reference,
    )
    assert sorted(verified) == ["analyse", "compute"]
