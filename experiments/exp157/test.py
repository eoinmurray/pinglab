"""Check the native E-only nine-window dynamics and guard the scientific measurements.

Tests cover schedule endpoints, feedforward isolation, inactive/decaying rhythm
rejection, ramp-end target semantics, fixed-control frequency calibration and numerical
refinement. No pre-existing Pingstore runs are required.
"""

import importlib

import numpy as np
import pytest
from experiments.exp157 import analyse, compute, recipe


@pytest.fixture(scope="module")
def network():
    cfg = recipe.configuration()
    bundle = recipe.author_network(cfg)
    return cfg, bundle, recipe.build_model(cfg, bundle)


def test_exact_nine_window_schedule_and_independent_settings():
    cfg = recipe.configuration()
    controls = recipe.drive_controls(cfg)
    times = np.arange(10) * 500
    expected = dict(
        A=[0, 0, 0, 0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5],
        B=[0, 0, 0, 0, 0, 0, 4.45, 4.45, 5.0, 5.0],
        C=[0, 0, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 2.3],
    )
    for circuit, values in expected.items():
        np.testing.assert_allclose(
            [controls[f"X_{circuit}.rate"](t) for t in times], values
        )
        np.testing.assert_allclose(
            [controls[f"X_{circuit}.rate"](t + 250) for t in times[:-1]],
            (np.array(values[:-1]) + values[1:]) / 2,
        )
    windows = recipe.windows(cfg)
    assert len(windows) == 9
    assert all(w["stop_ms"] - w["start_ms"] == 500 for w in windows)
    cfg["tau_rate_ms"]["A"]["I"] = 99
    assert recipe.configuration()["tau_rate_ms"]["A"]["I"] == 6.0
    cfg["neuron"]["leak_us"] = 99
    assert recipe.configuration()["neuron"]["leak_us"] == 0.0167


@pytest.mark.parametrize(
    "times",
    [
        dict(A=dict(E=24, I=6)),
        dict(A=dict(E=24, I=0), B=dict(E=16, I=4), C=dict(E=20, I=5)),
    ],
)
def test_rejects_incomplete_or_nonpositive_relaxation(times):
    with pytest.raises(ValueError):
        recipe.configuration(tau_rate_ms=times)


def test_native_architecture_has_only_two_feedforward_cross_paths(network):
    cfg, bundle, model = network
    ids = {p["id"] for p in bundle.graph["projections"]}
    cross = {
        name
        for name in ids
        if not name.startswith("X_")
        and name.split("_to_")[0][0] != name.split("_to_")[1][0]
    }
    assert cross == {"A_E_to_C_E", "B_E_to_C_E"}
    for circuit in recipe.CIRCUITS:
        assert f"X_{circuit}_to_{circuit}_E" in ids
        assert f"X_{circuit}_to_{circuit}_I" not in ids
    assert {name for name in ids if name.startswith("X_")} == {
        f"X_{c}_to_{c}_E" for c in recipe.CIRCUITS
    }
    assert set(recipe.initial_state(model, cfg).values()) == {0.0}
    assert not model.delayed


@pytest.mark.parametrize("kind", ["inactive", "decaying", "outside gamma"])
def test_rejects_non_gamma_signals(kind):
    cfg = recipe.configuration()
    time = np.arange(0, 400, 0.25)
    if kind == "inactive":
        rate = np.full(time.shape, 1e-9)
    elif kind == "decaying":
        rate = 5 + np.exp(-time / 60) * np.sin(2 * np.pi * 60 * time / 1000)
    else:
        rate = 5 + np.sin(2 * np.pi * 10 * time / 1000)
    metrics, _, _ = analyse.peak_phase(time, rate, cfg["analysis"])
    assert not metrics["gamma"]


def test_rate_target_remains_at_end_of_ramp_not_new_hold():
    from types import SimpleNamespace

    cfg = recipe.configuration()
    time = np.arange(0, 4500, 0.25)
    state = {
        "A_E.rate": np.full(time.shape, 5.0),
        "B_E.rate": np.where(time < 4000, 10.0, 20.0),
    }
    result = compute.terminal_rates(SimpleNamespace(time_ms=time, state=state), cfg)
    assert result["b_to_a_ratio"] == 2.0


def test_native_fixed_control_run_and_refinement():
    reference = recipe.configuration()
    cfg = compute.comparison_configuration(reference)
    model = recipe.build_model(cfg, recipe.author_network(cfg))
    compute.calibrate_frequencies(model, cfg)
    model = recipe.build_model(cfg, recipe.author_network(cfg))
    autonomous = compute.integrate_autonomous(model, cfg)
    measured = analyse.measure_autonomous(autonomous.time_ms, autonomous.state, cfg)
    assert set(measured) == {f"{c}_{p}" for c in recipe.CIRCUITS for p in ("E", "I")}
    for circuit in recipe.CIRCUITS:
        assert measured[circuit + "_E"]["gamma"]
        assert (
            abs(
                measured[circuit + "_E"]["frequency_hz"]
                - cfg["frequency_calibration"]["targets_hz"][circuit]
            )
            <= cfg["frequency_calibration"]["tolerance_hz"]
        )
    assert recipe.incoming_weights(cfg) == recipe.incoming_weights(reference)
    assert all(
        cfg["tau_rate_ms"][c]["E"] == reference["tau_rate_ms"][c]["E"]
        for c in recipe.CIRCUITS
    )
    solution = compute.integrate(model, cfg)
    refined = compute.integrate(model, cfg, refined=True)
    control = compute.integrate(model, cfg, uncoupled=True)
    assert solution.time_ms[0] == 0 and solution.time_ms[-1] == 4500
    assert all(values[0] == 0 for values in solution.state.values())
    for circuit in ("A", "B"):
        for kind in ("E", "I"):
            np.testing.assert_allclose(
                solution.state[f"{circuit}_{kind}.rate"],
                control.state[f"{circuit}_{kind}.rate"],
                atol=0.002,
                rtol=0.002,
            )
    assert np.isfinite(compute.terminal_rates(solution, cfg)["b_to_a_ratio"])
    assert cfg["b_final_drive_hz"] == reference["b_final_drive_hz"]
    check = analyse.numerical_check(
        solution.time_ms, solution.state, refined.state, cfg
    )
    assert (
        check["same_rhythm_gates"]
        and check["before_hold_normalized_state_error"] < 0.01
    )


def test_frequency_comparison_preserves_weights_controls_and_response_times():
    reference = recipe.configuration()
    cfg = compute.comparison_configuration(reference)
    assert cfg["local_incoming_us"] == dict(
        E_E=0.19285, E_I=0.7709, I_E=1.607, I_I=0.80365
    )
    allowed = {"schema", "frequency_calibration", "calibration_policy"}
    assert all(
        cfg[key] == value for key, value in reference.items() if key not in allowed
    )
    assert recipe.incoming_weights(reference) == recipe.incoming_weights(cfg)
    assert cfg["frequency_calibration"]["targets_hz"] == dict(A=34.0, B=38.0, C=36.0)


def test_phase_not_assigned_across_nonoscillatory_intervals():
    from experiments.exp157 import present

    cfg = recipe.configuration()
    time = np.arange(0, 1000, 0.25)
    state = {
        c + "_E.rate": 5 + np.sin(2 * np.pi * 40 * time / 1000) for c in recipe.CIRCUITS
    }
    centers = np.arange(100, 901, 25.0)
    # Visible peak intervals do not grant phase validity when the gate fails.
    assert len(analyse.interpeak_frequency(time, state["A_E.rate"], cfg["analysis"])[0])
    validity = {c: np.zeros(centers.shape, dtype=bool) for c in recipe.CIRCUITS}
    phases = present.relative_phases(time, state, cfg, centers, validity)
    assert all(np.isnan(values).all() for values in phases.values())


def test_interpeak_frequency_detects_slower_rhythm_after_weight_reduction():
    cfg = recipe.configuration()
    time = np.arange(0, 1000, 0.25)
    state = {
        f"{c}_{p}.rate": 5 + np.sin(2 * np.pi * 29 * time / 1000)
        for c in recipe.CIRCUITS
        for p in ("E", "I")
    }
    for rate in state.values():
        _, values = analyse.interpeak_frequency(time, rate, cfg["analysis"])
        assert np.isfinite(values).all()
        assert np.max(np.abs(values - 29)) < 0.2


def test_interpeak_frequency_keeps_irregular_cycles_without_extrapolation():
    cfg = recipe.configuration()
    time = np.arange(0, 400, 0.25)
    peaks = np.array([25, 50, 90, 110, 165, 190, 220, 245, 290, 315, 350])
    rate = sum(np.maximum(0, 1 - np.abs(time - peak) / 3) for peak in peaks)
    centers, frequency = analyse.interpeak_frequency(time, rate, cfg["analysis"])
    np.testing.assert_array_equal(centers, (peaks[:-1] + peaks[1:]) / 2)
    np.testing.assert_allclose(frequency, 1000 / np.diff(peaks))
    assert frequency.min() < cfg["analysis"]["gamma_band_hz"][0]
    assert not analyse.peak_phase(time, rate, cfg["analysis"])[0]["gamma"]


def test_interpeak_frequency_does_not_connect_across_inactivity():
    cfg = recipe.configuration()
    time = np.arange(0, 1500, 0.25)
    rate = np.where(
        (time < 400) | (time > 1100),
        5 + np.sin(2 * np.pi * 40 * time / 1000),
        5,
    )
    centers, frequency = analyse.interpeak_frequency(time, rate, cfg["analysis"])
    assert np.isnan(frequency[(centers > 500) & (centers < 1000)]).all()
    assert np.isnan(frequency).any()
    centers, frequency = analyse.interpeak_frequency(
        time, np.zeros_like(time), cfg["analysis"]
    )
    assert len(centers) == len(frequency) == 0


def test_imports_do_not_run_simulation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("import executed a stage")

    monkeypatch.setattr(compute.mf, "integrate", forbidden)
    for module in (recipe, compute, analyse):
        importlib.reload(module)
