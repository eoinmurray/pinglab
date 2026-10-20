"""Check membrane interventions, input pairing and measurement conventions.

Run uv run pytest experiments/exp134/test.py. Synthetic inputs exercise the
network and estimators without loading scientific run artifacts or importing
implementation code from another experiment.

Tests check preserved baseline structure, intended parameter changes, fresh
state and independent batch lanes, deterministic encoding and equal image
weighting when participation draws are missing."""

import numpy as np
import torch
from experiments.exp134 import compute, recipe
from experiments.exp134.custom.dynamics import dynamics
from snnlab.sim import Performance, build


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
        rows = sorted(
            (c for c in cfg["conditions"] if c["sweep"] in (sweep, "baseline")),
            key=lambda c: c["multiplier"],
        )
        assert len(rows) == 19
        assert [r["multiplier"] for r in rows] == recipe.configuration()["multipliers"]
        assert sum(r["condition_id"] == "baseline" for r in rows) == 1


def test_only_intended_neuron_parameters_change():
    base = recipe.author_network(recipe.conditions()[0]).graph
    for condition in recipe.conditions():
        graph = recipe.author_network(condition).graph
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


def test_baseline_architecture_and_units():
    graph = recipe.author_network(recipe.conditions()[0]).graph
    assert [(p["id"], p["size"]) for p in graph["populations"]] == [
        ("E", 1024),
        ("I", 256),
        ("readout", 10),
    ]
    neurons = [p["neuron"] for p in graph["populations"]]
    assert [n["tau_mem"]["value"] for n in neurons[:2]] == [20.0, 5.0]
    assert [n["refractory"]["value"] for n in neurons[:2]] == [1.2, 0.6]
    assert neurons[2]["kind"] == "leaky_integrator"
    projections = {item["id"]: item for item in graph["projections"]}
    assert set(projections) == set(
        name.removesuffix(".weight") for name in recipe.PARAMETERS
    )
    for name in ("E_to_I", "I_to_E"):
        assert projections[name]["connection"] == "recurrent"
        assert projections[name]["delay"] == {"value": 0.1, "unit": "ms"}
    assert projections["I_to_E"]["synapse"]["tau"]["value"] == 6.0


def test_repeated_calls_reset_state_and_batch_lanes_are_independent():
    torch.set_num_threads(1)
    graph = recipe.author_network(recipe.conditions()[0]).graph
    model = build(
        graph=graph, device="cpu", seed=0, performance=Performance(compile_mode=None)
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
    a, b = (
        compute.encode(pixels, 140, recipe.configuration()),
        compute.encode(pixels, 140, recipe.configuration()),
    )
    assert torch.equal(a, b)
    assert not torch.equal(a[:, 0], a[:, 1])
    assert not torch.equal(a, compute.encode(pixels, 141, recipe.configuration()))


def test_silent_and_periodic_population_measurements():
    cfg = dict(recipe.measurement(), dt_ms=0.1)
    silent = np.zeros((2000, 64), dtype=bool)
    result, _ = dynamics(silent, silent, cfg)
    assert not result["valid_cycles"]
    assert result["participation_e_pct"] is None
    assert result["rhythmicity_contrast"] is None
    periodic = silent.copy()
    periodic[100::200] = True
    result, _ = dynamics(periodic, periodic, cfg)
    assert result["valid_cycles"]
    assert result["participation_e_pct"] == 100.0


def test_configuration_overrides_are_resolved_and_independent():
    import pytest

    images = {i: (np.zeros((28, 28)), i // 50) for i in range(500)}
    cfg = recipe.configuration(images, images_per_digit=10, batch_images=2)
    assert cfg["presentations"] == 37000
    assert len(cfg["image_indices"]) == 100
    cfg["conditions"][0]["capacitance_nf"][0] = 999
    cfg["seeds"].clear()
    assert recipe.conditions()[0]["capacitance_nf"] == [1.0, 0.5]
    assert len(recipe.configuration()["seeds"]) == 10
    for value in (0, 51, 1.5):
        with pytest.raises(ValueError):
            recipe.configuration(images_per_digit=value)
    with pytest.raises(ValueError):
        recipe.configuration(batch_images=0)


def test_native_start_snapshot_preserves_frozen_weights(tmp_path):
    import json

    cfg = recipe.configuration()
    bundle = recipe.author_network(cfg["conditions"][0], cfg)
    weights = {
        cfg["parameter_mapping"][item["id"]]: torch.full(
            tuple(reversed(item["shape"])), 0.03
        )
        for item in bundle.graph["parameters"]
    }
    bundle, model, metrics = compute.frozen_model(cfg["conditions"][0], weights, cfg)
    compute.check_weights(model, weights, cfg)
    assert metrics["requested"] == cfg["performance"]
    compute.snapshot_start(model, bundle, tmp_path / "execution", cfg)
    snapshot = json.loads((tmp_path / "execution/start/snapshot.json").read_text())
    assert snapshot["kind"] == "simulate"
    assert snapshot["stage"] == "start"
    assert snapshot["parameters_included"]
    assert (tmp_path / "execution/start/parameters.npz").is_file()
    assert not (tmp_path / "execution/end").exists()
    with np.load(
        tmp_path / "execution/start/parameters.npz", allow_pickle=False
    ) as saved:
        for name, parameter in snapshot["parameters"].items():
            np.testing.assert_array_equal(
                saved[parameter["key"]], weights[cfg["parameter_mapping"][name]].numpy()
            )


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
