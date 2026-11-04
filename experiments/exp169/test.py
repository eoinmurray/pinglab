"""Focused checks against retained evidence; no neural simulation is launched."""

from pathlib import Path

import numpy as np
import pytest
import torch
from experiments.exp169 import recipe
from experiments.exp169.analyse import validate
from pingstore.library import file_sha256, source_run
from snnlab import lang
from snnlab.sim import Performance, build

ROOT = Path(__file__).resolve().parents[2]
RETAINED = "exp169-r003-compute"


def evidence():
    if not (ROOT / ".pingstore/runs" / RETAINED).is_dir():
        pytest.skip("Retained migration evidence is not local")
    return source_run(
        ROOT / ".pingstore", RETAINED, experiment="exp169", stage="compute"
    )


def test_import_is_byte_identical_and_preserves_provenance():
    imported = evidence()
    pin = imported.record["inputs"]["retained"]
    if not (ROOT / ".pingstore/runs" / pin["run_id"]).is_dir():
        pytest.skip("Historical pre-migration source is not local")
    source = source_run(
        ROOT / ".pingstore",
        pin["run_id"],
        experiment="exp022",
        stage="compute",
        reference=pin,
        phase="scratch-artificial-inhibition",
    )
    original = {
        p.relative_to(source.outputs): file_sha256(p)
        for p in source.outputs.rglob("*")
        if p.is_file()
    }
    copied = {
        p.relative_to(imported.outputs): file_sha256(p)
        for p in imported.outputs.rglob("*")
        if p.is_file()
    }
    assert original == copied
    assert (
        imported.record["execution"]["configuration"]
        == source.record["execution"]["configuration"]
    )
    assert imported.record["execution"]["no_simulation"] is True


def test_portable_recipe_preserves_native_control_architecture(tmp_path):
    imported = evidence()
    native = lang.load_bundle(
        imported.restore_bundle("network.bundle", tmp_path / "native.bundle")
    )
    assert recipe.author_network(recipe.configuration()).graph == native.graph
    reference = recipe.author_network(recipe.configuration(), include_private=False)
    assert reference.graph["populations"] == native.graph["populations"]
    assert {p["id"] for p in reference.graph["projections"]} == {
        "input_to_E",
        "E_to_I",
        "I_to_E",
        "readout_projection",
    }
    assert {i["id"] for i in reference.graph["inputs"]} == {"drive"}


def test_retained_control_matches_moments_filter_and_frozen_weights():
    results = validate(evidence())
    assert results["checks"]["frozen_checkpoint_parameters_exact"]
    assert results["checks"]["native_baseline_counts_exact"]
    assert results["checks"]["legacy_baseline_mismatched_spike_bins"] == 40
    assert results["matched"]["per_cell_std_relative_error_max"] < 0.01
    assert results["matched"]["pairwise_correlation_mean"] < 0.05
    assert results["conditions"][2]["rate_e_hz"] == pytest.approx(2.275390625)


def test_private_prehistory_preserves_delay_and_fresh_neuronal_state():
    from experiments.exp169.custom.prehistory import filtered_events, initial_state

    cfg = recipe.configuration(prehistory_ms=0.3)
    cfg.update(n_e=4, n_i=2, n_input=3, n_output=2)
    model = build(
        graph=recipe.author_network(cfg).graph,
        performance=Performance(compile_mode=None),
    ).model
    history = np.array([[1, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 1]], np.float32)
    scored = np.array([[0, 1, 0, 0], [1, 0, 0, 0]], np.float32)
    amps = np.array([0.1, 0.2, 0.3, 0.4], np.float32)
    state = initial_state(model, history, amps, cfg)
    decay = np.exp(-cfg["dt_ms"] / cfg["tau_gaba_ms"])
    first = (
        state.conductances["artificial_to_E"].numpy()[0] * decay + history[-1] * amps
    )
    np.testing.assert_allclose(
        first, filtered_events(np.vstack([history, scored]), cfg)[3] * amps, rtol=1e-6
    )
    np.testing.assert_array_equal(
        state.input_histories["artificial"].numpy(), history[-1:, None, :]
    )
    assert (
        state.completed_steps == 0
        and not state.reduction_state
        and not state.reduction_tensors
    )
    for name, voltage in state.voltages.items():
        assert torch.all(voltage == (0 if name == "readout" else -65))
    for group in (state.refractory, state.population_histories):
        assert all(torch.count_nonzero(value) == 0 for value in group.values())
    assert all(
        torch.count_nonzero(value) == 0
        for name, value in state.conductances.items()
        if name != "artificial_to_E"
    )


def test_calibration_matches_scored_window_after_private_prehistory():
    from experiments.exp169.compute import calibrate
    from experiments.exp169.custom.prehistory import filtered_events

    cfg = recipe.configuration()
    rng = np.random.default_rng(99)
    target = filtered_events((rng.random((2600, 4)) < 0.03).astype(float), cfg)[600:]
    events, amplitudes, probabilities, fitted = calibrate(target, cfg)
    assert events.shape == (2600, 4)
    assert np.isfinite(probabilities).all() and np.all(probabilities > 0)
    np.testing.assert_allclose(
        fitted, filtered_events(events, cfg)[600:] * amplitudes, rtol=1e-6
    )
    np.testing.assert_allclose(fitted.mean(0), target.mean(0), rtol=1e-6)
    np.testing.assert_allclose(
        fitted.std(0), target.std(0), rtol=cfg["std_relative_tolerance"]
    )


def test_recipe_rejects_invalid_prehistory():
    for value in (-1, 0.15):
        with pytest.raises(ValueError):
            recipe.configuration(prehistory_ms=value)
