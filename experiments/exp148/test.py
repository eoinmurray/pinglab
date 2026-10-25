"""Validate the three-window protocol, native recording and scientific measurements.

Small circuits check fixed recurrent/input pathways and immutable weights. Synthetic
spikes verify window isolation, spectrum scaling and local pulse detection.
"""

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from experiments.exp148 import dataset, recipe
from experiments.exp148.analyse import (
    measure,
    measure_hold_dynamics,
    validate_events,
    window_spectra,
)
from experiments.exp148.compute import execute
from experiments.exp148.present import transition_figure
from snnlab.sim import (
    DenseArrayBinding,
    Performance,
    RecordingSpec,
    SignalRecording,
    SnapshotSpec,
    simulate,
)


def small_configuration(**overrides):
    return recipe.configuration(
        n_e=8, n_i=4, n_external=8, baseline_ms=1, ramp_ms=1, hold_ms=1, **overrides
    )


def test_three_windows_and_only_i_modulation():
    cfg = recipe.configuration()
    assert cfg["duration_ms"] == 2300
    assert recipe.windows(cfg) == (
        ("baseline", 50, 800),
        ("ramp", 800, 1550),
        ("hold", 1550, 2300),
    )
    np.testing.assert_array_equal(
        recipe.drive_gain(np.array([0, 50, 800, 1175, 1550, 2300]), cfg),
        [1, 1, 1, 0.5, 0, 0],
    )
    graph = recipe.author_network(small_configuration()).graph
    projections = {p["id"]: p for p in graph["projections"]}
    assert len(projections) == 6
    assert projections["X_to_I"]["modulation"] == dict(
        kind="ramp_gain", before=1, after=0, start_ms=51, end_ms=52
    )
    assert all(
        p.get("modulation") is None
        for name, p in projections.items()
        if name != "X_to_I"
    )
    assert {
        p["source"] for name, p in projections.items() if name.startswith("X_")
    } == {"X.value"}
    for kwargs in (
        dict(baseline_ms=0),
        dict(ramp_ms=0.5),
        dict(hold_ms=float("nan")),
        dict(final_i_gain=-0.1),
        dict(final_i_gain=1.1),
        dict(input_connection_probability=0),
        dict(n_e=0),
        dict(seed=-1),
        dict(compile_mode="unknown"),
    ):
        with pytest.raises(ValueError):
            recipe.configuration(**kwargs)


def test_shared_constant_input_and_physical_weights():
    cfg = small_configuration()
    assert cfg["jump_means_us"]["X_to_I"] == pytest.approx(0.00432)
    assert cfg["jump_means_us"]["I_to_E"] == pytest.approx(0.008)
    dense = recipe.configuration(input_connection_probability=0.2)
    for name in ("X_to_E", "X_to_I"):
        assert cfg["jump_means_us"][name] * cfg[
            "input_connection_probability"
        ] == pytest.approx(
            dense["jump_means_us"][name] * dense["input_connection_probability"]
        )
    encoded = dataset.encode_drive(cfg).numpy()
    assert encoded.shape == (1060, 1, 8)
    rng = np.random.default_rng(cfg["seed"] + cfg["external_seed_offset"])
    expected = (
        rng.random(encoded.shape, dtype=np.float32)
        < cfg["external_rate_hz"] * cfg["dt_ms"] / 1000
    )
    np.testing.assert_array_equal(encoded, expected)
    np.testing.assert_array_equal(dataset.encode_drive(cfg).numpy(), encoded)


def test_native_recording_and_immutable_weights(tmp_path):
    torch.set_num_threads(1)
    cfg = small_configuration(input_connection_probability=1)
    bundle = recipe.author_network(cfg)
    drive = dataset.encode_drive(cfg)
    drive[:] = 1
    blocks = {p: [] for p in ("E", "I")}

    def collect(block):
        if block.kind == "spike_events":
            blocks[block.signal.split(".")[0]].append(block.values.numpy())

    result = simulate(
        bundle=bundle,
        input_bindings=(DenseArrayBinding("X", drive),),
        seed=cfg["seed"],
        device=cfg["device"],
        performance=Performance(**cfg["performance"]),
        recording=RecordingSpec(
            signals=tuple(
                SignalRecording(f"{p}.spikes", kind="spike_events") for p in blocks
            ),
            sink=collect,
            sink_id="test.exp148.spikes",
            block_steps=137,
        ),
        snapshot=SnapshotSpec(path=tmp_path / "start", stage="start", parameters=True),
    )
    for p, pieces in blocks.items():
        rows = np.concatenate(pieces)
        times, cells = np.nonzero(result.outputs[p + "_spikes"][:, 0].numpy())
        assert len(times) > 0
        np.testing.assert_array_equal(rows[:, 0], times)
        np.testing.assert_array_equal(rows[:, 2], cells)
    for name, value in result.parameters.items():
        torch.testing.assert_close(
            value, result.snapshots["start"].parameters[name], rtol=0, atol=0
        )
    assert (
        result.snapshots["start"].metadata["network"]["projections"]
        == bundle.graph["projections"]
    )


def test_state_recording_preserves_spikes(tmp_path):
    torch.set_num_threads(1)
    recorded = small_configuration(record_state=True)
    execute(recorded, tmp_path / "recorded")
    execute(small_configuration(), tmp_path / "unrecorded")
    with (
        np.load(tmp_path / "recorded/events.npz") as a,
        np.load(tmp_path / "unrecorded/events.npz") as b,
    ):
        assert set(a.files) == set(b.files)
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    with np.load(tmp_path / "recorded/state-traces.npz") as saved:
        assert len(saved.files) == 10
        assert all(saved[k].shape == (1060,) for k in saved.files)
        assert all(np.isfinite(saved[k]).all() for k in saved.files)


def synthetic_events():
    times = np.arange(1560, 2300, 25, dtype=float)
    return {
        p + "_" + q: arr.copy()
        for p in ("E", "I")
        for q, arr in (("time_ms", times), ("cell", np.zeros(len(times), dtype=int)))
    }


def test_measurements_and_bad_events():
    cfg = recipe.configuration()
    events = synthetic_events()
    result = measure(events, cfg)
    assert len(result["windows"]) == 3
    assert result["hold_cycles"]["E"]["frequency_hz"] == pytest.approx(40)
    assert result["windows"][0]["E_spikes"] == 0
    events["E_cell"][0] = cfg["n_e"]
    with pytest.raises(ValueError, match="outside its population"):
        validate_events(events, cfg)


def test_local_peaks_keep_small_early_volleys():
    cfg = recipe.configuration()
    early = np.repeat(np.arange(1555, 1800, 10), 2)
    late = np.repeat(np.arange(1805, 2300, 25), 100)
    events = {p + "_time_ms": np.concatenate([early, late]) for p in ("E", "I")}
    result = measure_hold_dynamics(events, cfg)
    for p in ("E", "I"):
        assert result["intervals"][0][p]["median_interval_ms"] == 10
        assert result["intervals"][-1][p]["median_interval_ms"] == 25


def test_spectra_isolate_all_three_windows_and_scale_rates():
    cfg = recipe.configuration(n_e=4, n_i=1)
    events = dict(E_time_ms=np.arange(1550, 2300, 25.0), I_time_ms=np.array([]))
    spectra = window_spectra(events, cfg)
    assert set(spectra) == {"baseline", "ramp", "hold"}
    assert np.all(spectra["baseline"]["E"]["power"] == 0)
    assert np.all(spectra["ramp"]["E"]["power"] == 0)
    assert np.all(spectra["hold"]["I"]["power"] == 0)
    hold = spectra["hold"]["E"]
    assert hold["metadata"]["nfft"] == 1500
    f = hold["frequencies_hz"]
    band = (f >= 30) & (f <= 50)
    assert f[band][np.argmax(hold["power"][band])] == 40
    larger = window_spectra(events, dict(cfg, n_e=8))
    np.testing.assert_allclose(larger["hold"]["E"]["power"], hold["power"] / 4)
    # A spike at the start of transition must not leak into the AI spectrum.
    boundary = window_spectra(dict(events, E_time_ms=np.array([800.0])), cfg)
    assert np.all(boundary["baseline"]["E"]["power"] == 0)
    assert np.any(boundary["ramp"]["E"]["power"] > 0)


def test_figure_has_three_windows_and_full_width_control():
    cfg = recipe.configuration()
    saved = synthetic_events()
    events = {p: (saved[p + "_time_ms"], saved[p + "_cell"]) for p in ("E", "I")}
    fig = transition_figure(events, cfg, window_spectra(saved, cfg))
    assert len(fig.axes) == 5
    assert fig.axes[1].get_position().x0 == fig.axes[0].get_position().x0
    assert fig.axes[1].get_position().width == fig.axes[0].get_position().width
    assert fig.axes[0].get_xlim() == (0, 2250)
    assert [t.get_text() for t in fig.axes[0].texts][:3] == [
        "AI TARGET",
        "AI-TO-PING TRANSITION",
        "PING TARGET",
    ]
    assert [a.get_title(loc="left") for a in fig.axes[-3:]] == [
        "Window 1: AI TARGET",
        "Window 2: AI-TO-PING TRANSITION",
        "Window 3: PING TARGET",
    ]
    plt.close(fig)


def test_no_cross_experiment_or_scratch_dependency():
    for path in Path(__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not any(
                name.startswith("experiments.exp")
                and not name.startswith("experiments.exp148")
                for name in names
            )
        if path.name != "test.py":
            assert not any(
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and ".scratch" in node.value
                for node in ast.walk(ast.parse(path.read_text()))
            )


def mean_field_test_configuration():
    return recipe.mean_field_configuration(
        recipe.configuration(),
        dict(E_to_E=0.4, E_to_I=0.8, I_to_E=1.6, I_to_I=1.6, X_to_E=0.86, X_to_I=0.86),
    )


def test_native_mean_field_protocol_and_state_basis():
    from snnlab import meanfield as mf

    cfg = mean_field_test_configuration()
    net = cfg["spiking_configuration"]
    bundle = recipe.author_network(net)
    model = recipe.mean_field_model(cfg, bundle)
    assert list(model.states) == [
        "E.rate",
        "I.rate",
        "E_to_I.conductance",
        "I_to_E.conductance",
    ]
    assert set(model.reconstructions) == {"E_to_E.conductance", "I_to_I.conductance"}
    assert bundle.graph["projections"][-1]["modulation"] is not None
    assert any(
        "quasistatic" in value and "omit" in value for value in model.assumptions
    )
    control = recipe.mean_field_control(cfg)
    for time in (0, 800, 1175, 1550, 2300):
        assert control(time) == recipe.drive_gain(time, net)
    branch = mf.continuation(
        model, "X_to_I.gain", np.linspace(1, 0, 21), guess=[2, 6, 0.01, 0.02]
    )
    assert branch[0][2].max_real < 0 < branch[-1][2].max_real
    crossings = mf.find_stability_crossings(model, "X_to_I.gain", branch)
    assert len(crossings) == 1 and crossings[0].oscillatory
    assert crossings[0].stable_before and crossings[0].stable_after is False
    # Reconstruction ratios cannot vary during a four-state protocol.
    with pytest.raises(ValueError, match="reconstruction"):
        mf.integrate(
            model,
            duration=1,
            initial=branch[0][1].state,
            controls={"E_to_E.gain": lambda time: 1},
        )


def test_mean_field_rejects_unhandled_graph_modulation():
    cfg = mean_field_test_configuration()
    bundle = recipe.author_network(cfg["spiking_configuration"])
    bundle.graph["projections"][0]["modulation"] = dict(kind="unexpected")
    with pytest.raises(ValueError, match="only Input"):
        recipe.mean_field_model(cfg, bundle)
    bundle = recipe.author_network(cfg["spiking_configuration"])
    bundle.graph["projections"][-1]["modulation"]["after"] = 0.5
    with pytest.raises(ValueError, match="disagrees"):
        recipe.mean_field_model(cfg, bundle)


def test_mean_field_stage_uses_native_source_and_preserves_outputs(
    tmp_path, monkeypatch
):
    from experiments.exp148 import compute
    from pingstore import stages
    from pingstore.contracts import load_json
    from pingstore.stages import source_run, stage_run

    monkeypatch.setattr(stages, "memberships", lambda repo: {"exp148": "experiments"})
    monkeypatch.setattr(
        stages,
        "_capture_code",
        lambda repo, directory: {
            "git_commit": "0" * 40,
            "dirty": False,
            "code_dirty": False,
        },
    )

    monkeypatch.setattr(compute, "REPO", tmp_path)
    cfg = small_configuration()
    with stage_run(tmp_path, "exp148", "compute", configuration=cfg) as run:
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={"simulation": ["start"]}
        )
        execute(cfg, run.export / "simulation")
    identity = compute.compute_mean_field(run.run_id)
    model_run = source_run(tmp_path / ".pingstore", identity)
    assert model_run.record["inputs"]["spiking"]["run_id"] == run.run_id
    assert model_run.record["execution"]["snnlab_version"]
    saved = load_json(model_run.file("mean-field-model.json"))
    assert saved["state_order"] == [
        "E.rate",
        "I.rate",
        "E_to_I.conductance",
        "I_to_E.conductance",
    ]
    results = load_json(model_run.file("mean-field.json"))
    assert len(results["continuation"]) == 201
    assert results["continuation"][0]["alpha"] == 1
    assert results["continuation"][-1]["alpha"] == 0
    with np.load(model_run.file("mean-field.npz")) as data:
        assert data["state"].shape == (len(data["time_ms"]), 4)
        assert np.isfinite(data["state"]).all()
        assert data["time_ms"][0] == 0
        assert data["time_ms"][-1] == cfg["duration_ms"]
        np.testing.assert_allclose(
            data["state"][0, :2],
            np.array(results["continuation"][0]["state"][:2]) + [0.001, 0],
        )
