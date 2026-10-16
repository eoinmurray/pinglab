"""Scientific intervention, baseline, pairing and estimator compatibility checks."""

import numpy as np
import torch
from experiments.exp123 import recipe as reference
from experiments.exp123.overview_analysis import dynamics as reference_dynamics
from experiments.exp134 import compute, recipe
from experiments.exp134.dynamics import dynamics
from snnlab.sim import ExecutionSpec, build


def test_balanced_nested_sample_and_single_shared_baseline():
    images = {i: (np.zeros((28, 28), dtype=np.float32), i // 50) for i in range(500)}
    cfg = recipe.configuration(images)
    assert cfg["presentations"] == 185000
    assert len(cfg["conditions"]) == 37
    assert len(set(cfg["image_indices"])) == 500
    assert np.array_equal(
        np.bincount([images[i][1] for i in cfg["image_indices"]]), [50] * 10
    )
    for sweep in ("leak", "capacitance"):
        rows = recipe.sweep_conditions(sweep)
        assert len(rows) == 19
        assert [r["multiplier"] for r in rows] == list(recipe.MULTIPLIERS)
        assert sum(r["condition_id"] == "baseline" for r in rows) == 1


def test_only_intended_neuron_parameters_change():
    base = reference.network(6.0).graph
    for condition in recipe.conditions():
        graph = recipe.network(condition).graph
        assert graph["projections"] == base["projections"]
        assert graph["parameters"] == base["parameters"]
        assert graph["populations"][2:] == base["populations"][2:]
        for j in range(2):
            cell = graph["populations"][j]["neuron"]
            assert cell["capacitance_nf"] == condition["capacitance_nf"][j]
            assert cell["leak_us"] == condition["leak_us"][j]
            assert cell["tau_mem"]["value"] == cell["capacitance_nf"] / cell["leak_us"]
            for key in set(cell) - {"capacitance_nf", "leak_us", "tau_mem"}:
                assert cell[key] == base["populations"][j]["neuron"][key]


def test_baseline_graph_is_reference_graph_except_name():
    graph = recipe.network(recipe.conditions()[0]).graph
    other = reference.network(6.0).graph
    graph["name"] = other["name"]
    assert graph == other


def test_repeated_calls_reset_state_and_batch_lanes_are_independent():
    torch.set_num_threads(1)
    graph = recipe.network(recipe.conditions()[0]).graph
    model = build(
        ExecutionSpec(kind="simulate", graph=graph, device="cpu", seed=0)
    ).model
    with torch.inference_mode():
        for parameter in model.parameter_map().values():
            parameter.fill_(0.03)
        drive = torch.zeros((120, 2, 784))
        drive[::10, 0] = 1
        drive[::15, 1] = 1
        batch = model({"image": drive}, diagnostics=True).diagnostics
        assert batch["spk_e"].count_nonzero() > 0
        for lane in range(2):
            single = model(
                {"image": drive[:, lane : lane + 1]}, diagnostics=True
            ).diagnostics
            assert all(
                torch.equal(value[:, lane : lane + 1], single[key])
                for key, value in batch.items()
            )
        repeated = model({"image": drive}, diagnostics=True).diagnostics
        assert all(torch.equal(value, repeated[key]) for key, value in batch.items())


def test_fresh_encoding_uses_identical_seed_rule():
    pixels = np.full((28, 28), 0.5, dtype=np.float32)
    a, b = compute.encode(pixels, 140), compute.encode(pixels, 140)
    assert torch.equal(a, b)
    assert not torch.equal(a[:, 0], a[:, 1])
    assert not torch.equal(a, compute.encode(pixels, 141))


def test_dynamics_match_reference_including_silent_trials():
    rng = np.random.default_rng(134)
    cfg = dict(recipe.measurement(), dt_ms=recipe.DT_MS)
    for probability in (0.0, 0.02):
        e = rng.random((2000, 32)) < probability
        i = rng.random((2000, 8)) < probability
        actual, diag = dynamics(e, i, cfg)
        expected, ref_diag = reference_dynamics(e, i, cfg)
        assert actual == expected
        for key in diag:
            np.testing.assert_equal(diag[key], ref_diag[key])


def test_participation_preserves_equal_image_weight_with_missing_draws(
    tmp_path, monkeypatch
):
    from experiments.exp134 import analyse

    def fake_dynamics(e, i, settings):
        participation = [10.0, 30.0, 80.0, None][int(e[0, 0])]
        return dict(
            valid_cycles=participation is not None,
            participation_e_pct=participation,
            rhythmicity_contrast=0.5,
        ), dict(frequency_hz=np.array([5.0, 10.0, 15.0]), psd=np.array([0.0, 1.0, 0.0]))

    monkeypatch.setattr(analyse, "dynamics", fake_dynamics)
    units = {}
    for index in range(2):
        unit = tmp_path / str(index)
        unit.mkdir()
        units[index] = str(unit)
        e = np.array([[[2 * index], [2 * index + 1]]])
        np.savez_compressed(unit / "baseline--spikes.npz", spk_e=e, spk_i=e)
    cfg = dict(image_indices=[0, 1], seeds=[42, 43], dt_ms=0.1, duration_ms=200.0)
    row = analyse.measure_condition(recipe.conditions()[0], cfg, units)
    assert row["participation_e_pct"] == 50.0  # Mean of image means 20 and 80.
    assert row["accepted_draws"] == 3
    assert row["participation_images"] == 2
    assert row["image_coverage"] == [
        dict(image_index=0, accepted_draws=2),
        dict(image_index=1, accepted_draws=1),
    ]
