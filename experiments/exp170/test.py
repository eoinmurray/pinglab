"""Validate migration and native evidence without executing a simulation."""

from pathlib import Path

import pytest
from experiments.exp170 import recipe
from experiments.exp170.analyse import validate
from pingstore.library import file_sha256, source_run
from snnlab import lang

ROOT = Path(__file__).resolve().parents[2]


def evidence():
    identity = "exp170-r001-compute"
    if not (ROOT / ".pingstore/runs" / identity).exists():
        pytest.skip("Migration evidence is not local")
    return source_run(
        ROOT / ".pingstore", identity, experiment="exp170", stage="compute"
    )


def test_import_is_byte_identical():
    imported = evidence()
    pin = imported.record["inputs"]["retained"]
    if not (ROOT / ".pingstore/runs" / pin["run_id"]).exists():
        pytest.skip("Historical migration source is not local")
    original = source_run(
        ROOT / ".pingstore",
        pin["run_id"],
        experiment="exp169",
        stage="compute",
        phase="scratch-count-matched-inhibition",
        reference=pin,
    )

    def inventory(source):
        return {
            p.relative_to(source.outputs): file_sha256(p)
            for p in source.outputs.rglob("*")
            if p.is_file()
        }

    assert inventory(imported) == inventory(original)
    assert (
        imported.record["execution"]["configuration"]
        == original.record["execution"]["configuration"]
    )
    assert imported.record["execution"]["no_simulation"] is True


def test_recipe_matches_retained_native_graph(tmp_path):
    imported = evidence()
    bundle = lang.load_bundle(
        imported.restore_bundle("network.bundle", tmp_path / "network.bundle")
    )
    assert (
        recipe.author_network(
            recipe.recorded_configuration(
                imported.record["execution"]["configuration"], bundle.graph
            )
        ).graph
        == bundle.graph
    )


def test_exact_source_counts_conductance_matching_and_frozen_weights():
    result = validate(evidence())
    for name in ["ping", "artificial"]:
        assert result[name]["source_rate_mean_hz"] == 120.0
        assert result[name]["source_rate_sd_hz"] == 0.0
    assert result["checks"]["sd_relative_error_max"] < 0.001
    assert result["checks"]["frozen_checkpoint_exact"]
    assert result["artificial"]["pairwise_conductance_correlation"] < 0.05
    assert result["conditions"][2]["rate_e_hz"] == pytest.approx(3.3984375)


def test_prehistory_default_and_validation():
    assert recipe.configuration()["prehistory_ms"] == 60
    assert recipe.configuration(prehistory_ms=0)["prehistory_ms"] == 0
    for invalid in (-1, 0.15, 201, float("nan")):
        with pytest.raises(ValueError):
            recipe.configuration(prehistory_ms=invalid)


def test_native_prehistory_preserves_queue_and_fresh_neurons():
    import numpy as np
    import torch
    from experiments.exp170.custom.prehistory import filtered_events, initial_state
    from snnlab.sim import GraphExecutor, plan_graph

    cfg = recipe.configuration(prehistory_ms=0.3)
    cfg.update(n_e=3, n_i=2, n_input=4, n_output=2)
    model = GraphExecutor(
        plan_graph(recipe.author_network(cfg).graph), compile_mode=None
    )
    past = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    amplitudes = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    state = initial_state(model, past, amplitudes, cfg)
    np.testing.assert_allclose(
        state.conductances["artificial_to_E"],
        (filtered_events(past, cfg)[-1] * amplitudes)[None],
        rtol=1e-6,
    )
    np.testing.assert_array_equal(state.input_histories["artificial"], past[-1:, None])
    assert state.completed_steps == 0
    for name, values in state.voltages.items():
        assert torch.all(values == (0 if name == "readout" else -65))
    for group in (state.refractory, state.population_histories):
        assert all(not torch.count_nonzero(v) for v in group.values())


def test_poisson_parameters_match_the_discrete_filter_moments():
    import numpy as np
    from experiments.exp170.compute import poisson_expected_moments, poisson_parameters

    cfg = recipe.configuration()
    target = np.array([[0.0, 0.2, 0.5], [0.0, 0.8, 1.5], [0.0, 0.4, 2.0]])
    rates, amplitude = poisson_parameters(target, cfg)
    mu, sd = poisson_expected_moments(rates, amplitude, cfg)
    np.testing.assert_allclose(mu, target.mean(0), rtol=1e-6)
    np.testing.assert_allclose(sd, target.std(0), rtol=1e-6)
    assert rates[0] == amplitude[0] == 0
    assert cfg["inhibition_method"] == recipe.POISSON_METHOD
    assert cfg["target_events_per_source"] is None
    with pytest.raises(ValueError, match="Constant positive"):
        poisson_parameters(np.ones((4, 2)), cfg)


def test_poisson_counts_are_unconditioned_and_include_coincident_events():
    import numpy as np
    from experiments.exp170.compute import fit_events
    from experiments.exp170.custom.prehistory import filtered_events

    cfg = recipe.configuration()
    # Enough independent channels to check the count distribution without simulation.
    target = np.tile(np.array([0.01, 0.03])[:, None], (1000, 256))
    events, amp, prediction, rates, _ = fit_events(target, cfg)
    history = round(cfg["prehistory_ms"] / cfg["dt_ms"])
    expected = rates * cfg["dt_ms"] / 1000
    assert events.dtype == np.int32 and events.max() > 1
    assert np.unique(events[history:].sum(0)).size > 1
    np.testing.assert_allclose(events.mean(0).mean(), expected.mean(), rtol=0.03)
    np.testing.assert_allclose(events.var(0).mean(), expected.mean(), rtol=0.04)
    np.testing.assert_array_equal(fit_events(target, cfg)[0], events)
    assert not np.array_equal(
        fit_events(target, {**cfg, "artificial_seed": cfg["artificial_seed"] + 1})[0],
        events,
    )
    np.testing.assert_allclose(prediction, filtered_events(events, cfg)[history:] * amp)


def test_poisson_graph_accepts_integer_counts_without_changing_historical_graph():
    new = recipe.author_network(recipe.configuration()).graph
    old = recipe.author_network(
        {**recipe.configuration(), "inhibition_method": "historical"}
    ).graph
    new_input = next(row for row in new["inputs"] if row["id"] == "artificial")
    old_input = next(row for row in old["inputs"] if row["id"] == "artificial")
    assert (new_input["signal_type"], new_input["unit"]) == ("continuous", "1")
    assert (old_input["signal_type"], old_input["unit"]) == ("spikes", "spike")
    for key in ("populations", "projections", "parameters"):
        assert new[key] == old[key]
