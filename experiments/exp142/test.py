"""Verify physical input timing, frozen weights and numerical comparisons.

Synthetic networks check causality, refractory periods, native snapshots and
continuous refinement. Presentation checks verify the experiment's figure layout.
"""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pytest
from experiments.exp142 import present as figures
from experiments.exp142 import recipe
from experiments.exp142.analyse import (
    add_model,
    check_anchor,
    compare,
    decision,
    figure_summary,
    output_clock,
    refinement_check,
    validate,
)
from experiments.exp142.custom.continuous import simulate_continuous
from experiments.exp142.custom.inputs import encode_sample, event_digest
from scipy.integrate import quad


def simulate(*, out=0.5, dt=0.1, max_step=0.025, ni=0):
    weights = {
        "input_to_E.weight": np.full((1, 1), 0.5, np.float32),
        "E_to_I.weight": np.zeros((1, ni), np.float32),
        "I_to_E.weight": np.zeros((ni, 1), np.float32),
        "readout_projection.weight": np.full((1, 1), out, np.float32),
    }
    events = dict(
        time_ns=np.array([0], np.int64),
        batches=np.array([0], np.int64),
        channels=np.array([0], np.int64),
    )
    return simulate_continuous(
        events,
        weights,
        defaults=recipe.configuration()["physics"],
        duration_ms=8,
        sample_dt_ms=dt,
        max_step_ms=max_step,
        rtol=1e-10,
        atol=1e-11,
    )[0]


def test_output_is_causal_coba_and_does_not_feed_back():
    active, silent = simulate(), simulate(out=0)
    hidden = active["spike_neurons"] == 0
    assert hidden.any()
    np.testing.assert_allclose(
        active["spike_time_ns"][hidden],
        silent["spike_time_ns"][silent["spike_neurons"] == 0],
        rtol=0,
        atol=1e-4,
    )
    arrival = active["spike_time_ns"][hidden][0] / 1e6
    assert active["output_spike_time_ns"].size
    assert active["output_spike_time_ns"][0] / 1e6 > arrival
    assert np.all(active["voltage_mv"][active["v_time_ns"] < arrival * 1e6, 1] == -65)
    # Independent scalar integrating-factor quadrature before the first output root.
    stop = active["output_spike_time_ns"][0] / 1e6
    sample = np.flatnonzero(
        (active["v_time_ns"] / 1e6 > arrival) & (active["v_time_ns"] / 1e6 < stop)
    )[0]
    t = active["v_time_ns"][sample] / 1e6 - arrival

    def integral_rate(s):
        return 0.05 * s + 1.0 * (-np.expm1(-s / 2))

    expected = (
        -65 * np.exp(-integral_rate(t))
        + quad(
            lambda s: -3.25 * np.exp(integral_rate(s) - integral_rate(t)),
            0,
            t,
            epsabs=1e-11,
        )[0]
    )
    assert active["voltage_mv"][sample, 1] == pytest.approx(expected, abs=1e-7)
    clock = active["g_time_ns"] / 1e6
    before_second = (clock > arrival) & (
        clock < active["spike_time_ns"][hidden][1] / 1e6
    )
    np.testing.assert_allclose(
        active["g_exc_ns"][before_second, 1],
        500 * np.exp(-(clock[before_second] - arrival) / 2),
        atol=1e-8,
    )


def test_output_refractory_and_counts_and_recording_clock_independence():
    a, b = simulate(), simulate(dt=0.2, max_step=0.0125)
    np.testing.assert_array_equal(a["class_scores"], b["class_scores"])
    np.testing.assert_allclose(a["spike_time_ns"], b["spike_time_ns"], atol=1e-2)
    times = a["output_spike_time_ns"] / 1e6
    assert np.all(
        np.diff(times) >= recipe.configuration()["physics"]["refractory_e_ms"]
    )
    for spike in times:
        refractory = (a["v_time_ns"] / 1e6 >= spike) & (
            a["v_time_ns"] / 1e6 < spike + 1.2
        )
        assert np.all(a["voltage_mv"][refractory, 1] == -65)
    np.testing.assert_array_equal(
        a["class_scores"], np.bincount(a["output_spike_neurons"], minlength=1)
    )


def test_refinement_rejects_changed_output_counts():
    a = simulate()
    cfg = recipe.configuration()
    cfg.update(n_e=1, n_output=1, duration_ns=8_000_000, sample_dt_ns=100_000)
    bad = {key: value.copy() for key, value in a.items()}
    bad["class_scores"][0] += 1
    with pytest.raises(AssertionError):
        refinement_check(a, bad, cfg, 0)


def test_configuration_is_independent():
    a, b = recipe.configuration(), recipe.configuration()
    a["physics"]["recurrent_delay_ms"] = 99
    assert b["physics"]["recurrent_delay_ms"] == 0.2


@pytest.mark.parametrize("mode", ["coba", "ping"])
def test_native_delays_weights_and_snapshots(mode, tmp_path):
    import json

    from experiments.exp142.analyse import validate
    from experiments.exp142.custom.discrete import execute

    cfg = recipe.configuration()
    cfg.update(n_input=1, n_e=1, n_i=1, n_output=1, duration_ns=8_000_000)
    weights = {
        "input_to_E.weight": np.full((1, 1), 0.5, np.float32),
        "readout_projection.weight": np.full((1, 1), 0.5, np.float32),
    }
    if mode == "ping":
        weights.update(
            {
                "E_to_I.weight": np.full((1, 1), 0.1, np.float32),
                "I_to_E.weight": np.full((1, 1), 0.1, np.float32),
            }
        )
    events = dict(
        time_ns=np.array([0], np.int64),
        batches=np.array([0], np.int64),
        channels=np.array([0], np.int64),
    )
    r, d = execute(events, weights, mode, cfg, 100_000, tmp_path / mode, matched=True)
    validate(r, cfg, 1 if mode == "ping" else 0)
    assert d["projection_delay_steps"]["readout_projection"] == 1
    if mode == "ping":
        assert d["projection_delay_steps"]["E_to_I"] == 3
        assert d["projection_delay_steps"]["I_to_E"] == 3
    assert d["refractory_timing"] == "following_steps"
    assert d["build_metrics"]["performance"]["resolved"]["device"] == "cpu"
    assert "execution_path" in d["execution_metrics"]
    for stage in ("start",):
        metadata = json.loads((tmp_path / mode / stage / "snapshot.json").read_text())
        with np.load(tmp_path / mode / stage / "parameters.npz") as saved:
            for name, value in weights.items():
                np.testing.assert_array_equal(
                    saved[metadata["parameters"][name]["key"]], value
                )
    if mode == "coba":
        anchor, _ = execute(
            events, weights, mode, cfg, 100_000, tmp_path / "anchor", matched=False
        )
        np.testing.assert_array_equal(
            r["spike_time_ns"][r["spike_neurons"] == 0],
            anchor["spike_time_ns"][anchor["spike_neurons"] == 0],
        )
        np.testing.assert_array_equal(
            r["output_spike_time_ns"], anchor["output_spike_time_ns"] + 100_000
        )


@pytest.mark.parametrize("dt_ns", [100_000, 50_000, 25_000, 12_500])
def test_physical_input_arrivals_are_identical_across_the_sweep(dt_ns, tmp_path):
    from experiments.exp142.custom.discrete import execute
    from experiments.exp142.custom.inputs import event_digest

    cfg = recipe.configuration()
    assert [row["dt_ns"] for row in cfg["cases"]] == [100_000, 50_000, 25_000, 12_500]
    cfg.update(n_input=1, n_e=1, n_output=1, duration_ns=2_000_000)
    events = dict(
        time_ns=np.array([0, 200_000, 400_000, 1_600_000], np.int64),
        batches=np.zeros(4, np.int64),
        channels=np.zeros(4, np.int64),
    )
    weights = {
        "input_to_E.weight": np.full((1, 1), 0.01, np.float32),
        "readout_projection.weight": np.zeros((1, 1), np.float32),
    }
    result, diagnostics = execute(
        events, weights, "coba", cfg, dt_ns, tmp_path / str(dt_ns), matched=True
    )
    assert diagnostics["input_event_sha256"] == event_digest(events)
    assert diagnostics["max_input_time_error_ns"] == 0
    # Inspect the executed conductance: jumps, including t=0, must be at the
    # exact physical input times for every dt. Never regenerate or round input.
    conductance = result["g_exc_ns"][:, 0]
    jumps = np.r_[0, 1 + np.flatnonzero(np.diff(conductance) > 0)]
    np.testing.assert_array_equal(result["g_time_ns"][jumps], events["time_ns"])


def test_native_anchor_retains_its_own_recording_clock(tmp_path):
    from experiments.exp142.analyse import validate
    from experiments.exp142.custom.discrete import execute

    cfg = recipe.configuration()
    cfg.update(n_input=1, n_e=1, n_output=1, duration_ns=2_000_000)
    weights = {
        "input_to_E.weight": np.zeros((1, 1), np.float32),
        "readout_projection.weight": np.zeros((1, 1), np.float32),
    }
    events = {key: np.array([], np.int64) for key in ("time_ns", "batches", "channels")}
    result, diagnostics = execute(
        events,
        weights,
        "coba",
        cfg,
        cfg["input_grid_ns"],
        tmp_path / "native",
        matched=False,
    )
    validate(result, cfg, 0, sample_dt_ns=cfg["native_sample_dt_ns"])
    assert diagnostics["dt_ns"] == diagnostics["sample_dt_ns"] == 200_000
    assert cfg["sample_dt_ns"] == 100_000


def test_native_encoder_is_repeatable_and_keeps_physical_grid(tmp_path):
    cfg = recipe.configuration()
    cfg.update(n_input=3, n_e=1, n_output=1)
    path = tmp_path / "dataset.npz"
    np.savez_compressed(
        path,
        sample_features=np.array([[0, 0.5, 1]], np.float32),
        sample_labels=np.array([0]),
    )
    graph = recipe.author_network(
        "coba", cfg, cfg["input_grid_ns"], matched=False
    ).graph
    original, _ = encode_sample(
        graph, path, cfg, max_rate_hz=cfg["source_input_rate_hz"]
    )
    higher, protocol = encode_sample(graph, path, cfg, max_rate_hz=cfg["input_rate_hz"])
    repeated, _ = encode_sample(graph, path, cfg, max_rate_hz=cfg["input_rate_hz"])
    assert cfg["input_rate_hz"] == 100 and cfg["source_input_rate_hz"] == 25
    assert event_digest(higher) == event_digest(repeated)
    original_coordinates = set(zip(original["time_ns"], original["channels"]))
    higher_coordinates = set(zip(higher["time_ns"], higher["channels"]))
    assert original_coordinates < higher_coordinates
    assert np.all(higher["channels"] != 0)  # black pixels remain silent
    for case in cfg["cases"]:
        assert np.all(higher["time_ns"] % case["dt_ns"] == 0)
    assert protocol["dataset_binding"]["encoder"]["max_rate_hz"] == 100
    with pytest.raises(ValueError, match="fixed physical input grid"):
        encode_sample(
            recipe.author_network("coba", cfg, 100_000, matched=True).graph,
            path,
            cfg,
            max_rate_hz=100,
        )


@pytest.mark.parametrize("rate", [-1, 0, 5001, float("nan"), float("inf"), True])
def test_invalid_rates_fail_in_configuration(rate):
    with pytest.raises(ValueError, match="input_rate_hz"):
        recipe.configuration(input_rate_hz=rate)


def test_source_control_requires_the_original_spikes_and_scores():
    cfg = recipe.configuration()
    cfg.update(n_e=1, n_output=1)
    source = dict(
        E=np.array([[0], [1]], np.uint8),
        readout=np.array([[0], [1]], np.uint8),
        scores=np.array([1]),
    )
    correct = dict(
        spike_time_ns=np.array([400_000, 400_000]),
        spike_neurons=np.array([0, 1]),
        class_scores=np.array([1]),
    )
    assert all(check_anchor(correct, source, cfg, 0).values())
    changed_input_response = {**correct, "spike_time_ns": np.array([200_000, 400_000])}
    with pytest.raises(AssertionError):
        check_anchor(changed_input_response, source, cfg, 0)


def fixture():
    cfg = recipe.configuration()
    cfg.update(n_e=2, n_i=1, duration_ns=2_000_000, guard_ms=0.01)
    audit = dict(event_count=2, cases={str(k): {} for k in range(14)})
    summary, plots, originals = figure_summary(cfg, audit), {}, {}
    cases = ["native"] + [row["id"] for row in cfg["cases"]] + ["continuous"]
    for mode in cfg["models"]:
        ni = int(mode == "ping")
        hidden = cfg["n_e"] + ni
        recordings, models, measurements = {}, {}, {}
        for index, case in enumerate(cases):
            size = hidden + cfg["n_output"]
            sample = (
                cfg["native_sample_dt_ns"] if case == "native" else cfg["sample_dt_ns"]
            )
            g = np.arange(0, cfg["duration_ns"], sample)
            root_offset = 0.25 if case == "continuous" else 0
            times = np.array([400_000, 800_000, 1_000_000, 1_400_000]) + root_offset
            ids = np.array([0, 1, hidden + (index % 3), hidden + (index % 3)])
            cumulative = np.zeros((len(g), cfg["n_output"]), dtype=int)
            cumulative[:, index % 3] = np.searchsorted(
                times[2:], g + sample, side="right"
            )
            voltage = np.broadcast_to(
                -65 + np.arange(size) * (index + 1) / 10, (len(g), size)
            ).copy()
            recordings[case] = dict(
                g_time_ns=g,
                v_time_ns=g + sample,
                g_exc_ns=np.full((len(g), size), index),
                g_inh_ns=np.zeros((len(g), size)),
                voltage_mv=voltage,
                readout_voltage=voltage[:, hidden:],
                spike_time_ns=times,
                spike_neurons=ids,
                output_spike_time_ns=times[2:],
                output_spike_neurons=ids[2:] - hidden,
                class_scores=cumulative[-1],
                readout_scores=cumulative,
            )
            validate(recordings[case], cfg, ni, sample_dt_ns=sample)
        for case, recording in recordings.items():
            populations, metrics = compare(recording, recordings["continuous"], cfg, ni)
            models[case] = dict(
                classification=decision(recording["class_scores"]),
                populations=populations,
            )
            measurements[case] = metrics
        add_model(summary, plots, mode, cfg, ni, models, recordings, measurements)
        originals[mode] = recordings
    return cfg, summary, plots, originals


def test_same_six_comparison_panels_and_hidden_only_errors():
    cfg, summary, plots, _ = fixture()
    figures._style()
    for mode, ni in [("coba", 0), ("ping", 1)]:
        fig = figures._comparison_figure(summary, plots, mode)
        try:
            fig.canvas.draw()
            assert len(fig.axes) == 6
            a, b, c, d, e, f = fig.axes
            assert len(a.lines) == 10 and len(b.lines) == 5 and len(c.collections) == 5
            assert [text.get_text() for text in c.get_yticklabels()] == [
                "0.1",
                "0.05",
                "0.025",
                "0.0125",
                "CT",
            ]
            np.testing.assert_allclose(
                c.collections[-1].get_offsets()[:, 0], [0.40000025]
            )
            for case in summary["cases"]:
                stem = f"{mode}__{case['id']}__"
                assert plots[stem + "voltage_mv"].shape[1] == cfg["n_e"] + ni
                assert len(plots[stem + "v_reset_aware_mae_mv"]) == cfg["n_e"] + ni
            assert "Receptor error" in d.get_title(loc="left")
            assert "Reset-aware voltage error" in e.get_title(loc="left")
            assert "Ordinal spike shift" in f.get_title(loc="left")
        finally:
            plt.close(fig)


def test_same_classification_layout_uses_counts_for_every_case():
    _, summary, plots, originals = fixture()
    figures._style()
    fig = figures._classification_figure(summary, plots, label=0)
    try:
        fig.canvas.draw()
        assert len(fig.axes) == 8  # six panels plus two rank colorbars
        assert fig.get_figwidth() * 25.4 == pytest.approx(180)
        for column, mode in enumerate(("coba", "ping")):
            top, scores, ranks = [fig.axes[column + 2 * row] for row in range(3)]
            record = summary["models"][mode]
            rows = [record["native_anchor"], *record["cases"]]
            for index, row in enumerate(rows):
                expected = originals[mode][row["id"]]
                # CT winner is class 2, so both true-class and winner traces appear.
                np.testing.assert_array_equal(
                    top.lines[2 * index].get_ydata(), expected["readout_scores"][:, 0]
                )
                np.testing.assert_array_equal(
                    scores.lines[index].get_ydata(), expected["class_scores"]
                )
                assert top.lines[2 * index].get_drawstyle() == "steps-post"
            values = np.stack(
                [originals[mode][row["id"]]["class_scores"] for row in rows]
            )
            np.testing.assert_array_equal(
                ranks.images[0].get_array(),
                1 + (values[:, None, :] > values[:, :, None]).sum(axis=2),
            )
            assert "COBA-LIF output" in top.get_title(loc="left")
            assert scores.get_ylabel() == "Output spike count"
        assert "same timestep" in fig._supxlabel.get_text()
        assert "CT: continuous COBA-LIF" in fig._supxlabel.get_text()
        assert "fixed" not in fig._supxlabel.get_text()
    finally:
        plt.close(fig)


def test_scikit_input_uses_the_same_input_layout_with_64_channels():
    _, summary, _, _ = fixture()
    image = np.arange(64).reshape(8, 8) / 63
    events = dict(
        time_ns=np.array([0, 200_000]),
        channels=np.array([0, 63]),
        batches=np.zeros(2, dtype=int),
    )
    figures._style()
    fig = figures._input_figure(summary, image, events, label=0, image_index=0)
    try:
        np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), image)
        assert "Scikit digits" in fig.axes[0].get_title(loc="left")
        assert fig.axes[1].get_ylim() == (-1, 64)
    finally:
        plt.close(fig)


def test_intermediate_output_counts_are_validated():
    cfg, _, _, originals = fixture()
    invalid = deepcopy(originals["coba"]["continuous"])
    invalid["readout_scores"][0, 0] += 1
    with pytest.raises(AssertionError):
        validate(invalid, cfg, 0)


def test_reject_wrong_native_clock_or_output_type(tmp_path):
    import json
    from types import SimpleNamespace

    cfg = recipe.configuration()
    path = tmp_path / "snapshot.json"
    source = SimpleNamespace(file=lambda *args: path)
    snapshot = dict(
        network=dict(
            timebase=dict(dt=dict(value=0.2)),
            populations=[dict(id="readout", neuron=dict(kind="coba_lif"))],
        ),
        execution=dict(timestep_ms=0.2),
    )
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="timestep"):
        output_clock(source, "coba", "dt0p1", cfg)
    snapshot["network"]["timebase"]["dt"]["value"] = 0.1
    snapshot["execution"]["timestep_ms"] = 0.1
    path.write_text(json.dumps(snapshot))
    clock = output_clock(source, "coba", "dt0p1", cfg)
    assert clock["network_dt_ns"] == clock["output_dt_ns"] == 100_000
    snapshot["network"]["populations"][0]["neuron"]["kind"] = "leaky_integrator"
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="COBA-LIF"):
        output_clock(source, "coba", "dt0p1", cfg)


def test_anchor_errors_use_exact_reference_samples_on_its_own_clock():
    cfg, _, _, originals = fixture()
    native = deepcopy(originals["coba"]["native"])
    continuous = deepcopy(originals["coba"]["continuous"])
    continuous["voltage_mv"][:, 0] = np.arange(len(continuous["v_time_ns"]))
    continuous["g_exc_ns"] = continuous["g_exc_ns"].astype(float)
    continuous["g_exc_ns"][:, 0] = np.arange(len(continuous["g_time_ns"]))
    native["voltage_mv"][:, 0] = continuous["voltage_mv"][1::2, 0] + 0.125
    native["g_exc_ns"] = native["g_exc_ns"].astype(float)
    native["g_exc_ns"][:, 0] = continuous["g_exc_ns"][::2, 0] + 0.25
    _, measurements = compare(native, continuous, cfg, 0)
    assert measurements["voltage_mae_mv"][0] == pytest.approx(0.125)
    assert measurements["ampa_mae_ns"][0] == pytest.approx(0.25)
    invalid = deepcopy(native)
    invalid["g_time_ns"] = invalid["g_time_ns"] + 1
    with pytest.raises(ValueError, match="exact samples"):
        compare(invalid, continuous, cfg, 0)
