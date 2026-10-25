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
from experiments.exp153 import analyse, compute, recipe
from snnlab import lang
from snnlab import meanfield as mf


@pytest.fixture
def fixture():
    net = lang.Network("exp153_test", dt=0.1 * lang.ms)
    populations, drives = {}, {}
    for circuit in ("A", "B"):
        drives[circuit] = net.input(
            f"drive_{circuit}",
            shape=("time", "batch", 3),
            signal_type="spikes",
            unit="spike",
        )
        for population in ("E", "I"):
            name = f"PING_{circuit}_{population}"
            populations[name] = net.population(
                name, size=2, neuron=lang.COBA_LIF(refractory=2 * lang.ms)
            )
    weights = {}

    def connect(source, target, name, inhibitory=False, delay=0, cross=False):
        weights[name] = 0.12 if cross else 0.4
        net.connect(
            source,
            target,
            name=name,
            synapse=(lang.GABA if inhibitory else lang.AMPA)(tau=2 * lang.ms),
            weight=lang.Constant(0.1),
            delay=delay * lang.ms,
            connection="feedback" if delay else "feedforward",
            modulation=lang.StepGain(at=500 * lang.ms, before=0, after=1)
            if cross
            else None,
        )

    for c in ("A", "B"):
        e, i = populations[f"PING_{c}_E"], populations[f"PING_{c}_I"]
        connect(drives[c], e.excitatory, f"PING_{c}_input_to_E")
        connect(e.spikes, i.excitatory, f"PING_{c}_E_to_I", delay=0.1)
        connect(i.spikes, e.inhibitory, f"PING_{c}_I_to_E", inhibitory=True, delay=0.1)
        other = "B" if c == "A" else "A"
        for pop in ("E", "I"):
            connect(
                e.spikes,
                populations[f"PING_{other}_{pop}"].excitatory,
                f"PING_{c}_E_to_PING_{other}_{pop}_K_E{pop}",
                delay=2,
                cross=True,
            )
    bundle = lang.compile(net)
    source = dict(
        schema="exp133.recipe/v1",
        base_coupling_us=0.12,
        duration_ms=5000,
        mean_input_hz=25,
        onset_ms=500,
        settling_ms=300,
        input_differences_hz=list(range(-10, 11, 2)),
        coupling_us=np.arange(13).tolist(),
        volley_distance_ms=15,
        volley_prominence_fraction=0.1,
        min_volleys=20,
        max_iei_cv=0.2,
        gamma_band_hz=[30, 80],
        classification=dict(
            max_abs_drift_hz=0.25, min_concentration=0.95, max_phase_span_cycles=1
        ),
    )
    source["coupling_us"] = (np.arange(13) / 100).tolist()
    return bundle, recipe.configuration(source, weights)


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
    assert phase[0] == 0 and phase[-1] < -1
    assert np.polyfit(t / 1000, phase, 1)[0] == pytest.approx(-1, abs=0.01)
    assert details["reference_time_ms"] == t[0]
    assert details["onset_ms"] == 500


def test_phase_trace_rejects_a_tonic_signal(fixture):
    _, cfg = fixture
    time, rate = waveform(amplitude=0)
    with pytest.raises(ValueError, match="common observed peaks"):
        analyse.relative_phase_trace(
            time, {"PING_A_E.rate": rate, "PING_B_E.rate": rate}, cfg
        )


def test_imported_sums_scaled_once_and_delays_and_onset_retained(fixture):
    bundle, cfg = fixture
    original = copy.deepcopy(bundle.graph)
    model = recipe.build_model(bundle, cfg, difference_hz=-4, coupling_us=0.06)
    assert len(model.states) == 14
    assert model.defaults["drive_A.rate"] == 23
    assert model.defaults["drive_B.rate"] == 27
    assert model.defaults["PING_A_input_to_E.coupling"] == 0.4
    assert model.defaults["PING_A_E_to_PING_B_I_K_EI.coupling"] == 0.06
    assert model.projection_delays["PING_A_E_to_I"] == 0.1
    assert model.projection_delays["PING_A_E_to_PING_B_I_K_EI"] == 2
    assert len(model.controls) == 4
    for control in model.controls.values():
        assert control(499.99) == 0 and control(500) == 1
    assert bundle.graph == original
    with pytest.raises(ValueError, match="characteristic-root"):
        mf.jacobian(model)


def test_native_matrix_orientation_and_zero_edges(fixture, monkeypatch):
    bundle, _ = fixture
    arrays = {
        p["id"] + ".weight": torch.tensor([[0.0, 1.0], [2.0, 0.0], [3.0, 4.0]])
        for p in bundle.graph["projections"]
    }
    monkeypatch.setattr(
        compute.NetworkSnapshot,
        "read",
        lambda path: SimpleNamespace(
            load_parameters=lambda: arrays,
            graph_identity=bundle.manifest["graph_digest"],
            stage="start",
        ),
    )
    source = SimpleNamespace(file=lambda *args: "unused")
    sums = compute.imported_weights(source, bundle)
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
    assert time[-1] == 3 and len(state) == 14
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


def test_no_cross_experiment_code_or_scratch_imports():
    from pathlib import Path

    for path in Path(__file__).parent.glob("*.py"):
        if path.name == "test.py":
            continue
        text = path.read_text()
        assert "from experiments.exp133" not in text
        assert "from experiments.exp148" not in text
        assert ".scratch/" not in text


def test_qualitative_rhythms_need_not_match_exp133_gamma(fixture):
    _, cfg = fixture
    time, rate = waveform(frequency=25.3)
    state = {f"PING_{c}_E.rate": rate for c in ("A", "B")}
    assert analyse.gate_metrics(time, state, cfg)["gate_passed"]
    assert analyse.pair_metrics(time, state, cfg)["locked"]


def test_parallel_conditions_match_serial_native_integration(fixture, tmp_path):
    from pingstore.contracts import write_json_atomic

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


def test_example_has_one_control_and_explicit_relaxation_asymmetry(fixture):
    bundle, cfg = fixture
    assert cfg["protocol"] == "phase-trace" and "sweep" not in cfg
    assert cfg["closure"]["tau_rate_ms"]["PING_A_E"] == 20
    assert cfg["closure"]["tau_rate_ms"]["PING_B_E"] == 16
    model = recipe.build_model(bundle, cfg, difference_hz=40, coupling_us=0.08)
    assert model.defaults["drive_A.rate"] == 45
    assert model.defaults["drive_B.rate"] == 5
    for kwargs in (dict(difference_hz=50), dict(rate_scale_b=0), dict(coupling_us=0)):
        with pytest.raises(ValueError):
            recipe.configuration(
                cfg["imported_protocol"], cfg["incoming_weights_us"], **kwargs
            )
