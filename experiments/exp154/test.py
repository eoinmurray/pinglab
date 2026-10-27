"""Check the independent dynamics reference and the intended learning boundary.

These tests validate continuous E/I timing, gradients and selective constrained
updates. They do not assert that an arbitrarily initialized controller must
reach the scientific accuracy target.

Run: uv run pytest -q experiments/exp154/test.py.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[2] / "tools"),
]
from experiments.exp154.recipe import (
    author_network,
    configuration,
    encode,
    make_model,
    targets,
)


@pytest.fixture
def controller():
    torch.set_num_threads(1)
    cfg = configuration("smoke", genome={"prenatal_windows": 0})
    bundle = author_network(cfg, 7)
    return cfg, bundle, make_model(bundle, cfg, 7)


def reference(model, cfg, bundle, cues):
    weights = {
        name: value.detach().numpy() for name, value in model.parameter_map().items()
    }
    features = np.column_stack((cues, 1 - cues, np.ones_like(cues)))
    e = np.full((len(cues), cfg["e_cells"]), cfg["initial_rate"])
    i = np.full((len(cues), cfg["i_cells"]), cfg["initial_rate"])
    biases = {
        row["id"]: np.asarray(row["neuron"]["config"]["bias_na"])
        for row in bundle.graph["populations"]
    }
    trace = []
    for _ in range(round(cfg["presentation_ms"] / cfg["dt_ms"])):
        e_drive = (
            features @ weights["sensory_E.weight"]
            + e @ weights["EE.weight"]
            - i @ weights["IE.weight"]
            - biases["E"]
        )
        i_drive = (
            features @ weights["sensory_I.weight"]
            + e @ weights["EI.weight"]
            - i @ weights["II.weight"]
            - biases["I"]
        )
        e = e + cfg["dt_ms"] / cfg["tau_e_ms"] * (
            1 / (1 + np.exp(-e_drive / cfg["sigmoid_scale_na"])) - e
        )
        i = i + cfg["dt_ms"] / cfg["tau_i_ms"] * (
            1 / (1 + np.exp(-i_drive / cfg["sigmoid_scale_na"])) - i
        )
        trace.append(e)
    tail = round(cfg["readout_tail_ms"] / cfg["dt_ms"])
    return np.mean(trace[-tail:], axis=0) @ weights["value_weight"]


def test_independent_rate_reference_and_no_hidden_outcome(controller):
    cfg, bundle, model = controller
    cues = np.array([0.0, 0.4, 1.0])
    expected = reference(model, cfg, bundle, cues)
    actual = model(encode(cues, cfg), diagnostics=False).outputs["prediction"]
    np.testing.assert_allclose(
        actual.detach().numpy(), expected, rtol=1e-12, atol=1e-12
    )
    positive, negative = targets(cues, 1), targets(cues, -1)
    np.testing.assert_array_equal(positive, -negative)
    assert actual.shape == (3, 1)
    assert len(bundle.graph["inputs"]) == 2
    assert torch.count_nonzero(encode(cues, cfg)["internal"]) == 0


@pytest.mark.parametrize("name", ["EI.weight", "IE.weight"])
def test_gradient_matches_finite_difference(controller, name):
    cfg, _, model = controller
    # A non-neutral point gives both pathways nonzero gradients to check.
    with torch.no_grad():
        model.parameter_map()["IE.weight"][0, 0] += 0.01
    inputs = encode([0.2, 0.8], cfg)
    output = model(inputs, diagnostics=False).outputs["prediction"].square().mean()
    output.backward()
    parameter = model.parameter_map()[name]
    analytic = parameter.grad[0, 0].item()
    old = parameter[0, 0].item()
    values = []
    with torch.no_grad():
        for delta in (1e-5, -1e-5):
            parameter[0, 0] = old + delta
            values.append(
                model(inputs, diagnostics=False)
                .outputs["prediction"]
                .square()
                .mean()
                .item()
            )
        parameter[0, 0] = old
    np.testing.assert_allclose(
        analytic, (values[0] - values[1]) / 2e-5, rtol=1e-5, atol=1e-8
    )
    assert np.isfinite(analytic) and abs(analytic) > 0


def test_only_cross_pathways_update_and_bounds_are_enforced(controller):
    cfg, _, model = controller
    parameters = model.parameter_map()
    initial = {name: value.detach().clone() for name, value in parameters.items()}
    optimizer = torch.optim.AdamW(
        [parameters[name] for name in ("EI.weight", "IE.weight")],
        lr=0.01,
        weight_decay=0,
    )
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = (
            model(encode([0.1, 0.9], cfg), diagnostics=False)
            .outputs["prediction"]
            .squeeze(-1)
        )
        loss = (prediction - torch.tensor([-0.8, 0.8])).square().mean()
        loss.backward()
        optimizer.step()
        model.enforce_constraints()
    changed = {
        name
        for name, value in parameters.items()
        if not torch.equal(value, initial[name])
    }
    assert changed == {"EI.weight", "IE.weight"}
    with torch.no_grad():
        parameters["EI.weight"][0, 0] = -1
        parameters["IE.weight"][0, 0] = 2
    model.enforce_constraints()
    assert parameters["EI.weight"].min() >= 0
    assert parameters["IE.weight"].max() <= cfg["weight_upper"]


def test_bundle_reload_preserves_frozen_predictions(controller, tmp_path):
    from snnlab import lang

    cfg, bundle, model = controller
    path = bundle.write(tmp_path / "network.bundle", visualise=False)
    restored = make_model(lang.load_bundle(path), cfg, 7)
    inputs = encode([0.2, 0.8], cfg)
    torch.testing.assert_close(
        model(inputs).outputs["prediction"],
        restored(inputs).outputs["prediction"],
        rtol=0,
        atol=0,
    )


def test_native_snapshot_analysis_accepts_exported_parameters(controller, tmp_path):
    from experiments.exp154.analyse import parameter_changes
    from experiments.exp154.custom.snapshot import capture

    cfg, bundle, model = controller
    capture(
        model, bundle, tmp_path / "initial", cfg, 7, stage="start", selection="test"
    )
    with torch.no_grad():
        model.parameter_map()["EI.weight"].add_(0.01)
    capture(model, bundle, tmp_path / "final", cfg, 7, stage="end", selection="test")
    changes, frozen, parameters = parameter_changes(
        tmp_path / "initial", tmp_path / "final"
    )
    assert frozen and changes["E→I"] > 0 and changes["I→E"] == 0
    assert all(isinstance(value, np.ndarray) for value in parameters.values())


def test_rejection_never_updates_or_reveals_target(controller):
    from experiments.exp154.compute import (
        cross_weights,
        optimizer_for,
        predict,
        update_consumed,
    )

    cfg, _, model = controller
    before = cross_weights(model)
    optimizer = optimizer_for(model, cfg)
    for hidden_delta in (-1.0, 1.0):
        prediction = predict(model, encode([0.8], cfg))
        update_consumed(model, optimizer, prediction, hidden_delta, False, cfg)
        np.testing.assert_array_equal(before, cross_weights(model))
    assert not optimizer.state


def test_lethal_encounter_and_birth_reset(controller):
    from experiments.exp154.compute import cross_weights, run_lifetime

    cfg, _, model = controller
    cfg = {**cfg, "initial_energy": 0.001, "metabolic_cost": 1.1}
    inherited = cross_weights(model)
    events, pre, post, _, _ = run_lifetime(
        model, cfg, -1, np.array([0.9, 0.9, 0.9]), np.zeros(3)
    )
    assert all(e["death"] and e["birth"] and e["updated"] for e in events)
    assert [e["life"] for e in events] == [1, 2, 3]
    assert all(e["age"] == 1 for e in events)
    for before, after in zip(pre, post, strict=True):
        np.testing.assert_array_equal(before, inherited)
        assert not np.array_equal(after, inherited)
    # Identical first updates establish that Adam's moments reset at each birth.
    np.testing.assert_array_equal(post[0], post[1])
    np.testing.assert_array_equal(post[1], post[2])
    assert events[0]["energy_after"] == pytest.approx(0.001 - 1.1 - 0.8)


def test_all_consumed_objects_supply_learning_updates(controller):
    from experiments.exp154.compute import consume, run_lifetime

    cfg, _, model = controller
    cues = np.linspace(0.1, 0.9, 5)
    events, pre, post, _, _ = run_lifetime(model, cfg, 1, cues, np.zeros(5))
    assert all(e["accepted"] and e["feedback"] and e["updated"] for e in events)
    assert np.any(pre != post)
    assert consume(-0.2, 0, cfg) and not consume(-0.2, 1, cfg)
    assert consume(0.2, 1, cfg)


def test_neutral_newborn_has_no_cue_preference():
    from experiments.exp154.compute import consume, predict

    cfg = configuration("smoke")
    seed = cfg["seeds"][0]
    model = make_model(author_network(cfg, seed), cfg, seed)
    with torch.no_grad():
        prediction = predict(model, encode(np.linspace(0, 1, 201), cfg)).numpy()
    np.testing.assert_allclose(prediction, 0, atol=1e-12, rtol=0)
    assert cfg["mapping_signs"] == [1]
    assert not any(consume(value, 1.0, cfg) for value in prediction)
    assert all(consume(value, 0.0, cfg) for value in prediction)


def test_native_recording_preserves_decision_and_gradients(controller):
    from experiments.exp154.compute import predict, recorded_prediction

    cfg, bundle, model = controller
    other = make_model(bundle, cfg, 7)
    actual, signals = recorded_prediction(model, 0.8, cfg)
    expected = predict(other, encode([0.8], cfg))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    (actual - 0.6).square().sum().backward()
    (expected - 0.6).square().sum().backward()
    for name in ("EI.weight", "IE.weight"):
        torch.testing.assert_close(
            model.parameter_map()[name].grad,
            other.parameter_map()[name].grad,
            atol=0,
            rtol=0,
        )
    assert signals["E_rate"].shape == (40, 16)
    assert signals["I_rate"].shape == (40, 4)


def test_recorded_activity_reconstructs_currents_and_predictions(controller):
    from experiments.exp154.analyse import activity_checks
    from experiments.exp154.compute import run_lifetime

    cfg, _, model = controller
    initial = {
        name: value.detach().numpy().copy()
        for name, value in model.parameter_map().items()
    }
    recordings = []
    events, pre, _, _, _ = run_lifetime(
        model, cfg, 1, np.array([0.8, 0.2, 0.7]), np.zeros(3), recording_sink=recordings
    )
    data = {
        f"replay_{name}": np.stack([r[name] for r in recordings])
        for name in recordings[0]
    }
    data["weights_pre"] = pre
    assert activity_checks(data, initial, cfg, events)["passed"]
    # Corrupt one recorded current: validation must detect the mismatch.
    data["replay_IE_current"][0, 0, 0] += 0.01
    assert not activity_checks(data, initial, cfg, events)["passed"]


def test_displayed_topology_reconstructs_native_current(controller, tmp_path):
    from experiments.exp154.compute import cross_weights, recorded_prediction
    from experiments.exp154.custom.snapshot import capture
    from experiments.exp154.present import network_payload
    from snnlab.sim import NetworkSnapshot

    cfg, bundle, model = controller
    capture(
        model,
        bundle,
        tmp_path / "initial",
        cfg,
        7,
        stage="start",
        selection="topology test",
    )
    graph = network_payload(
        NetworkSnapshot.read(tmp_path / "initial"),
        {"weights_pre": cross_weights(model)[None]},
        cfg,
    )
    _, signals = recorded_prediction(model, 0.8, cfg)
    sources = np.concatenate(
        (signals["E_rate"][9], signals["I_rate"][9], [0.8, 0.2, 1], [0, 0, 0])
    )
    currents = {name: np.zeros(20) for name in graph["flow_scales"]}
    for edge in graph["edges"]:
        currents[edge["pathway"]][edge["target"]] += (
            sources[edge["source"]] * edge["weight"]
        )
    for name, current in currents.items():
        target = slice(16, 20) if name.endswith("I") else slice(0, 16)
        np.testing.assert_allclose(
            current[target], signals[name + "_current"][10], atol=1e-12, rtol=0
        )
    assert {edge["target"] for edge in graph["edges"]} == set(range(20))


def test_learning_history_aligns_post_update_weights_and_diagnostic_error():
    from experiments.exp154.present import learning_history

    inherited = np.full((2, 64), 0.1)
    pre = np.stack([inherited, inherited + 0.1, inherited])
    post = np.stack([inherited + 0.1, inherited + 0.2, inherited + 0.3])
    post[:, 1] *= 2
    events = [
        dict(trial=1, validation_mse=0.25, birth=True),
        dict(trial=2, validation_mse=0.04, birth=False),
        dict(trial=3, validation_mse=0.09, birth=True),
    ]
    history = learning_history(events, pre, post)
    np.testing.assert_allclose(history["rmse"], [0.5, 0.2, 0.3])
    np.testing.assert_allclose(
        history["weight_norms"], [[1.6, 3.2], [2.4, 4.8], [3.2, 6.4]]
    )
    np.testing.assert_allclose(history["inherited_norms"], [0.8, 0.8])
    np.testing.assert_allclose(history["norm_ratio"], 0.5)
    post[1, 1] = 0
    undefined = learning_history(events, pre, post)
    assert np.isnan(undefined["norm_ratio"][1])
    np.testing.assert_array_equal(history["births"], [0.5, 2.5])
    with pytest.raises(ValueError, match="aligned learning history"):
        learning_history(events[:-1], pre, post)


def test_genome_controls_inherited_weight_ranges():
    from experiments.exp154.recipe import inherited_genome

    genome = inherited_genome()
    genome["recurrent_ranges"]["EI"] = [0.2, 0.25]
    genome["recurrent_ranges"]["IE"] = [0.35, 0.4]
    cfg = configuration("smoke", genome=genome)
    model = make_model(author_network(cfg, 7), cfg, 7)
    for name, low, high in (("EI.weight", 0.2, 0.25), ("IE.weight", 0.35, 0.4)):
        value = model.parameter_map()[name].detach().numpy()
        assert np.all((value >= low) & (value <= high))
    assert inherited_genome()["recurrent_ranges"]["EI"] == [0.01, 0.05]


def test_development_precedes_food_and_maturation_reduces_learning():
    from experiments.exp154.compute import cross_weights, run_lifetime

    cfg = configuration(
        "smoke", genome={"prenatal_windows": 3, "juvenile_encounters": 2}
    )
    model = make_model(author_network(cfg, 7), cfg, 7)
    inherited = cross_weights(model)
    initial = {
        name: value.detach().clone() for name, value in model.parameter_map().items()
    }
    development = []
    events, pre, _, _, _ = run_lifetime(
        model, cfg, 1, np.full(4, 0.9), np.zeros(4), development_sink=development
    )
    assert [event["phase"] for event in events] == [
        "juvenile",
        "juvenile",
        "adult",
        "adult",
    ]
    assert [event["phase_age"] for event in events] == [1, 2, 1, 2]
    assert events[2]["learning_rate"] < events[1]["learning_rate"]
    np.testing.assert_array_equal(development[0]["weights_pre"], inherited)
    np.testing.assert_array_equal(pre[0], development[-1]["weights_post"])
    assert any(np.any(row["weights_pre"] != row["weights_post"]) for row in development)
    assert all(
        "cue" not in row["event"] and "delta" not in row["event"] for row in development
    )
    for name, value in model.parameter_map().items():
        if name not in {"EI.weight", "IE.weight"}:
            torch.testing.assert_close(value, initial[name], rtol=0, atol=0)


def test_each_conception_restores_genome_before_prenatal_experience():
    from experiments.exp154.compute import cross_weights, run_lifetime

    cfg = configuration("smoke", genome={"prenatal_windows": 2})
    cfg = {**cfg, "initial_energy": 0.001, "metabolic_cost": 2.0}
    model = make_model(author_network(cfg, 7), cfg, 7)
    inherited = cross_weights(model)
    development = []
    events, pre, _, _, _ = run_lifetime(
        model, cfg, 1, np.full(3, 0.9), np.zeros(3), development_sink=development
    )
    assert all(event["birth"] and event["death"] for event in events)
    assert len(development) == 6
    for index in range(3):
        np.testing.assert_array_equal(development[2 * index]["weights_pre"], inherited)
        np.testing.assert_array_equal(
            pre[index], development[2 * index + 1]["weights_post"]
        )
    assert not np.array_equal(
        development[0]["internal_drives"], development[2]["internal_drives"]
    )


def test_prenatal_currents_and_loss_are_reconstructed_from_native_activity():
    from experiments.exp154.analyse import activity_checks
    from experiments.exp154.compute import prenatal_development

    cfg = configuration("smoke", genome={"prenatal_windows": 2})
    model = make_model(author_network(cfg, 7), cfg, 7)
    initial = {
        name: value.detach().numpy().copy()
        for name, value in model.parameter_map().items()
    }
    development = prenatal_development(model, cfg, life=1, before_trial=1, seed=7)
    data = {
        f"replay_{name}": np.stack([row["signals"][name] for row in development])
        for name in development[0]["signals"]
    }
    data["weights_pre"] = np.stack([row["weights_pre"] for row in development])
    drives = np.stack([row["internal_drives"] for row in development])
    events = [row["event"] for row in development]
    assert activity_checks(data, initial, cfg, events, internal_drives=drives)["passed"]
    for row in development:
        target = cfg["genome"]["homeostatic_target_rate"]
        loss = (
            sum(
                np.mean((row["signals"][name + "_rate"].mean(axis=0) - target) ** 2)
                for name in ("E", "I")
            )
            / 2
        )
        assert row["event"]["loss"] == pytest.approx(loss, abs=1e-12)
    data["replay_internal_E_current"][0, 0, 0] += 0.01
    assert not activity_checks(data, initial, cfg, events, internal_drives=drives)[
        "passed"
    ]


def test_lifecycle_replay_preserves_phase_order_and_isolates_inputs(tmp_path):
    import json
    from types import SimpleNamespace

    from experiments.exp154.compute import run_lifetime
    from experiments.exp154.present import learning_history, lifecycle_events

    cfg = configuration(
        "smoke", genome={"prenatal_windows": 2, "juvenile_encounters": 2}
    )
    cfg = {**cfg, "initial_energy": 0.6, "metabolic_cost": 0.5}
    model = make_model(author_network(cfg, 7), cfg, 7)
    development, recordings = [], []
    food, pre, post, _, _ = run_lifetime(
        model,
        cfg,
        1,
        np.array([0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.9]),
        np.zeros(7),
        development_sink=development,
        recording_sink=recordings,
    )
    (tmp_path / "events.json").write_text(json.dumps(food))
    (tmp_path / "development-events.json").write_text(
        json.dumps([row["event"] for row in development])
    )
    for filename, rows, before, after in (
        ("data.npz", recordings, pre, post),
        (
            "development.npz",
            [row["signals"] for row in development],
            [row["weights_pre"] for row in development],
            [row["weights_post"] for row in development],
        ),
    ):
        values = {
            f"replay_{name}": np.stack([row[name] for row in rows]) for name in rows[0]
        }
        if filename == "development.npz":
            values["internal_drives"] = np.stack(
                [row["internal_drives"] for row in development]
            )
        np.savez(tmp_path / filename, weights_pre=before, weights_post=after, **values)
    timeline = lifecycle_events(
        SimpleNamespace(file=lambda path: tmp_path / path), dict(directory="."), cfg
    )
    assert [row["step"] for row in timeline] == list(range(1, len(timeline) + 1))
    assert sum(row["phase"] != "prenatal" for row in timeline) == len(food)
    assert {row["phase"] for row in timeline} == {"prenatal", "juvenile", "adult"}
    assert sum(row["conception"] for row in timeline) == 3
    for life in (1, 2, 3):
        rows = [row for row in timeline if row["life"] == life]
        assert [row["phase"] for row in rows[:3]] == [
            "prenatal",
            "prenatal",
            "juvenile",
        ]
        assert rows[0]["conception"] and rows[2]["birth"]
        np.testing.assert_array_equal(rows[1]["weights_after"], rows[2]["weights"])
    for row in timeline:
        drives = np.asarray(row["drives"])
        assert drives.shape == (41, 6)
        assert np.asarray(row["rates"]).shape == (41, 20)
        np.testing.assert_array_equal(drives[0], np.zeros(6))
        if row["phase"] == "prenatal":
            assert row["cue"] is row["delta"] is row["energy_before"] is None
            assert not row["birth"] and not row["death"]
            assert np.count_nonzero(drives[:, :3]) == 0
            assert np.count_nonzero(drives[:, 3:]) > 0
        else:
            assert np.count_nonzero(drives[:, 3:]) == 0
            np.testing.assert_allclose(drives[1:, :3].sum(axis=1), 2)
    history = learning_history(
        timeline,
        np.asarray([row["weights"] for row in timeline]),
        np.asarray([row["weights_after"] for row in timeline]),
    )
    prenatal = np.array([row["phase"] == "prenatal" for row in timeline])
    np.testing.assert_array_equal(
        history["conceptions"],
        [row["step"] - 0.5 for row in timeline if row["conception"]],
    )
    np.testing.assert_array_equal(history["births"] - history["conceptions"], 2)
    assert np.isnan(history["rmse"][prenatal]).all()
    np.testing.assert_allclose(
        history["rmse"][~prenatal], np.sqrt([row["validation_mse"] for row in food])
    )


def test_evolution_configuration_and_reproducible_bounded_mutations():
    import copy

    from experiments.exp154.recipe import (
        mutate_genome,
        next_population,
        validate_evolution_genome,
    )

    pilot = configuration("pilot", "evolution")
    settings = pilot["evolution"]
    assert (settings["population"], settings["generations"], settings["elites"]) == (
        16,
        10,
        4,
    )
    assert settings["search_seeds"] == [7, 8, 9]
    assert settings["fresh_seeds"] == [10, 11, 12, 13, 14]
    assert set(settings["search_seeds"]).isdisjoint(settings["fresh_seeds"])
    assert pilot["lifetime_presentations"] == 300 and pilot["stop_at_death"]
    original = copy.deepcopy(pilot["genome"])
    a, b = np.random.default_rng(154), np.random.default_rng(154)
    genome = original
    for _ in range(100):
        child = mutate_genome(genome, a, settings)
        assert child == mutate_genome(genome, b, settings)
        validate_evolution_genome(child, settings)
        for pathway in ("EE", "II"):
            assert (
                child["recurrent_ranges"][pathway]
                == original["recurrent_ranges"][pathway]
            )
        for pathway in ("EI", "IE"):
            assert (
                child["recurrent_normal"][pathway]["std"]
                == original["recurrent_normal"][pathway]["std"]
            )
        assert child["weight_upper"] == original["weight_upper"]
        assert child["internal_drive_na"] == original["internal_drive_na"]
        genome = child
    assert pilot["genome"] == original
    assert (
        genome["recurrent_normal"]["EI"]["mean"]
        != original["recurrent_normal"]["EI"]["mean"]
    )
    assert set(original["recurrent_ranges"]) == {"EE", "II"}
    assert pilot["schema"] == "exp154/evolution-v3"
    assert settings["reproduction"]["minimum_age_encounters"] == 25
    founder = dict(
        candidate="g001-c000", parent=None, created_generation=1, genome=original
    )
    first = next_population([founder], settings, a, 1)
    elites = first[:4]
    historical = {
        key: value for key, value in settings.items() if key != "reproduction"
    }
    second = next_population(elites, historical, a, 2)
    assert second[:4] == elites
    assert len(second) == 16
    assert all(
        sum(row["parent"] == elite["candidate"] for row in second[4:]) == 3
        for elite in elites
    )
    with pytest.raises(ValueError, match="weight range"):
        configuration(
            "smoke",
            "evolution",
            genome={
                "recurrent_ranges": {
                    **original["recurrent_ranges"],
                    "EI": [0, 0.05],
                }
            },
        )


def test_evolution_fitness_uses_survival_energy_and_identity():
    from experiments.exp154.recipe import rank_candidates

    rows = [
        dict(candidate="c", mean_survival=20, mean_energy=1, diagnostic_mse=0),
        dict(candidate="b", mean_survival=20, mean_energy=2, diagnostic_mse=999),
        dict(candidate="a", mean_survival=20, mean_energy=2, diagnostic_mse=1000),
        dict(candidate="z", mean_survival=21, mean_energy=0, diagnostic_mse=2000),
    ]
    assert [row["candidate"] for row in rank_candidates(rows)] == ["z", "a", "b", "c"]


def test_reproduction_is_primary_fitness_and_diagnostics_do_not_select():
    from experiments.exp154.compute import fitness_scores

    rows = [
        dict(candidate="long", survival=100, remaining_energy=10, offspring=0),
        dict(candidate="fertile", survival=30, remaining_energy=0, offspring=2),
        dict(candidate="fertile", survival=40, remaining_energy=1, offspring=1),
    ]
    scores = fitness_scores(rows, [dict(candidate="long"), dict(candidate="fertile")])
    assert [row["candidate"] for row in scores] == ["fertile", "long"]
    assert scores[0]["mean_offspring"] == 1.5


@pytest.mark.parametrize(
    "energy,age,phase,reproduced,expected",
    [
        (2.0, 24, "adult", False, 0),
        (2.0, 25, "juvenile", False, 0),
        (-0.1, 25, "adult", False, 0),
        (0.99, 25, "adult", False, 0),
        (1.0, 25, "adult", False, 1),
        (1.5, 25, "adult", False, 2),
        (2.0, 25, "adult", False, 3),
        (10.0, 25, "adult", False, 3),
        (2.0, 26, "adult", True, 0),
    ],
)
def test_reproduction_age_energy_cost_and_single_litter(
    energy, age, phase, reproduced, expected
):
    from experiments.exp154.compute import reproduction_after

    policy = configuration("pilot", "evolution")["evolution"]["reproduction"]
    after, event = reproduction_after(energy, age, phase, reproduced, policy)
    assert event["offspring"] == expected
    assert event["reproduced"] == bool(expected)
    assert after == energy - 0.5 * expected
    if expected:
        assert after >= policy["reserve_energy"]


def test_reproduction_charge_does_not_change_neural_learning_or_juvenile_timing():
    import copy

    from experiments.exp154.analyse import reproduction_checks
    from experiments.exp154.compute import run_lifetime

    cfg = configuration(
        "smoke", "evolution", genome={"prenatal_windows": 0, "juvenile_encounters": 0}
    )
    cfg.update(initial_energy=4.0, metabolic_cost=0.0, exploration_probability=1.0)
    bundle = author_network(cfg, 7)
    events, pre, post, _, _ = run_lifetime(
        make_model(bundle, cfg, 7), cfg, 1, np.ones(40), np.zeros(40)
    )
    reproduction = [event for event in events if event["reproduced"]]
    assert len(reproduction) == 1 and reproduction[0]["age"] == 25
    assert reproduction[0]["offspring"] == 3
    assert reproduction_checks(events, cfg)["passed"]
    historical = copy.deepcopy(cfg)
    historical["evolution"].pop("reproduction")
    old_events, old_pre, old_post, _, _ = run_lifetime(
        make_model(bundle, historical, 7), historical, 1, np.ones(40), np.zeros(40)
    )
    np.testing.assert_array_equal(pre, old_pre)
    np.testing.assert_array_equal(post, old_post)
    assert events[-1]["energy_after"] == old_events[-1]["energy_after"] - 1.5
    tampered = copy.deepcopy(events)
    tampered[24]["offspring"] = 2
    assert not reproduction_checks(tampered, cfg)["passed"]


def test_actual_offspring_limit_admission_and_record_each_mutated_genome():
    import copy

    from experiments.exp154.recipe import next_population, reproductive_population

    cfg = configuration("pilot", "evolution")
    settings = {**cfg["evolution"], "population": 6, "elites": 2}
    founder = dict(
        candidate="g001-c000", parent=None, created_generation=1, genome=cfg["genome"]
    )
    population = next_population([founder], settings, np.random.default_rng(154), 1)
    ranked = [dict(candidate=row["candidate"]) for row in population]
    lives = [
        dict(
            candidate=row["candidate"],
            seed=7,
            offspring=2 if index == 0 else 0,
            reproduction_age=25 if index == 0 else None,
        )
        for index, row in enumerate(population)
    ]
    original = copy.deepcopy(population)
    next_a, children_a = reproductive_population(
        population, ranked, lives, settings, np.random.default_rng(154), 1, admit=True
    )
    next_b, children_b = reproductive_population(
        population, ranked, lives, settings, np.random.default_rng(154), 1, admit=True
    )
    assert (next_a, children_a) == (next_b, children_b)
    assert population == original
    assert len(next_a) == 4 and len(children_a) == 2
    assert next_a[:2] == population[:2]
    assert all(
        row["parent"] == founder["candidate"]
        and row["parent_trial"] == 25
        and row["admitted"]
        for row in children_a
    )
    assert children_a[0]["genome"] != children_a[1]["genome"]
    assert all(set(row["genome"]) == set(cfg["genome"]) for row in children_a)
    for life in lives:
        life["offspring"] = 3
        life["reproduction_age"] = 25
    capped, all_children = reproductive_population(
        population, ranked, lives, settings, np.random.default_rng(154), 1, admit=True
    )
    assert len(capped) == 6 and len(all_children) == 18
    assert sum(child["admitted"] for child in all_children) == 4
    assert all(
        child["parent"] in {row["candidate"] for row in population[:2]}
        for child in all_children
        if child["admitted"]
    )
    for life in lives:
        life["offspring"] = 0
        life["reproduction_age"] = None
    stalled, children = reproductive_population(
        population, ranked, lives, settings, np.random.default_rng(154), 1, admit=True
    )
    assert stalled == population[:2] and not children


def test_evolution_life_stops_at_death_and_records_terminal_update(controller):
    from experiments.exp154.compute import run_lifetime

    cfg, _, model = controller
    cfg = {**cfg, "initial_energy": 0.1, "metabolic_cost": 2, "stop_at_death": True}
    events, pre, post, _, _ = run_lifetime(
        model, cfg, 1, np.full(12, 0.1), np.zeros(12)
    )
    assert len(events) == 1
    assert events[0]["birth"] and events[0]["death"] and events[0]["updated"]
    assert events[0]["energy_after"] < 0
    assert not np.array_equal(pre[0], post[0])


def test_evolution_smoke_genealogy_inputs_native_inheritance_and_replay(tmp_path):
    import json
    from types import SimpleNamespace

    from experiments.exp154.analyse import evolution_summary
    from experiments.exp154.compute import compute_evolution
    from experiments.exp154.present import evolution_reproduction_figure, replay_payload

    torch.set_num_threads(1)
    cfg = configuration("smoke", "evolution")
    compute_evolution(
        SimpleNamespace(
            export=tmp_path, record=dict(execution=dict(configuration=cfg))
        ),
        cfg,
    )
    source = SimpleNamespace(
        file=lambda path: tmp_path / path, reference={"run_id": "test"}
    )
    rows = json.loads((tmp_path / "conditions.json").read_text())
    result = evolution_summary(source, rows, cfg)
    assert result["integrity_pass"]
    search = [row for row in result["conditions"] if row["cohort"] == "search"]
    assert len(search) == sum(len(row["members"]) for row in result["generations"])
    assert 1 <= len(result["generations"]) <= 2
    assert result["termination"] in {"generation_limit", "no_offspring"}
    assert all(0 <= row["offspring"] <= 3 for row in result["conditions"])
    assert any(row["offspring"] > 0 for row in search)
    assert len(result["fresh"]) == 2
    assert all(
        row["births"] == 1
        and row["matched_inputs"]
        and row["fresh_inherited_parameters"]
        for row in result["conditions"]
    )
    result["configuration"] = cfg
    figure = evolution_reproduction_figure(result)
    assert all(ax.get_ylabel() == "Offspring per life" for ax in figure.axes)
    np.testing.assert_allclose(
        figure.axes[0].lines[1].get_ydata(),
        [
            np.mean([row["mean_offspring"] for row in generation["ranked"]])
            for generation in result["generations"]
        ],
    )
    figure.savefig(tmp_path / "reproduction-qa.png", dpi=120)
    import matplotlib.pyplot as plt

    plt.close(figure)
    winner = next(
        row for row in result["individuals"] if row["candidate"] == result["winner"]
    )
    life = next(
        row
        for row in result["conditions"]
        if row["cohort"] == "fresh" and row["role"] == "winner"
    )
    resolved = configuration("smoke", "evolution", genome=winner["genome"])
    payload = replay_payload(
        source, dict(configuration=resolved, conditions=[life], scope="smoke")
    )
    events = payload["conditions"][0]["events"]
    assert payload["food_encounters"] == life["survival"]
    assert sum(e["birth"] for e in events) == 1
    assert sum(e["conception"] for e in events) == int(
        resolved["genome"]["prenatal_windows"] > 0
    )
    assert payload["reproduction"] == cfg["evolution"]["reproduction"]
    reproductive_life = next(row for row in search if row["offspring"])
    individual = next(
        row
        for row in result["individuals"]
        if row["candidate"] == reproductive_life["candidate"]
    )
    from experiments.exp154.recipe import individual_configuration

    replay = replay_payload(
        source,
        dict(
            configuration=individual_configuration(
                cfg, individual["genome"], reproductive_life["seed"]
            ),
            conditions=[reproductive_life],
            scope="reproduction test",
        ),
    )
    (tmp_path / "reproduction-qa.json").write_text(json.dumps(replay))
    # Selection records are evidence, not trusted instructions for analysis.
    path = tmp_path / "evolution.json"
    saved = json.loads(path.read_text())
    saved["generations"][0]["ranked"][0]["mean_survival"] += 1
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="ranking"):
        evolution_summary(source, rows, cfg)


def test_reproductive_search_stops_without_fabricating_offspring(tmp_path):
    import json
    from types import SimpleNamespace

    from experiments.exp154.analyse import evolution_summary
    from experiments.exp154.compute import compute_evolution

    cfg = configuration("smoke", "evolution", genome={"prenatal_windows": 0})
    cfg.update(initial_energy=0.1, metabolic_cost=2.0)
    compute_evolution(
        SimpleNamespace(
            export=tmp_path, record=dict(execution=dict(configuration=cfg))
        ),
        cfg,
    )
    source = SimpleNamespace(
        file=lambda path: tmp_path / path, reference={"run_id": "test"}
    )
    rows = json.loads((tmp_path / "conditions.json").read_text())
    result = evolution_summary(source, rows, cfg)
    assert result["integrity_pass"]
    assert result["termination"] == "no_offspring"
    assert len(result["generations"]) == 1
    assert result["offspring"] == []
    assert all(
        row["offspring"] == 0 and row["survival"] == 1 for row in result["conditions"]
    )
    saved = json.loads((tmp_path / "evolution.json").read_text())
    saved["termination"] = "generation_limit"
    (tmp_path / "evolution.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="termination"):
        evolution_summary(source, rows, cfg)


def test_evolution_allows_zero_prenatal_and_juvenile_durations(tmp_path):
    from types import SimpleNamespace

    from experiments.exp154.analyse import evolution_life_summary
    from experiments.exp154.compute import evolution_life
    from experiments.exp154.present import replay_payload

    torch.set_num_threads(1)
    cfg = configuration(
        "smoke", "evolution", genome={"prenatal_windows": 0, "juvenile_encounters": 0}
    )
    individual = dict(candidate="zero-phases", genome=cfg["genome"])
    row = evolution_life(
        SimpleNamespace(export=tmp_path),
        cfg,
        individual,
        7,
        cohort="search",
        generation=1,
        role="candidate",
    )
    source = SimpleNamespace(
        file=lambda path: tmp_path / path, reference={"run_id": "test"}
    )
    result = evolution_life_summary(source, row, cfg, individual)
    assert result["integrity_pass"]
    assert result["development"]["windows"] == 0
    assert result["phase_counts"]["juvenile"] == 0
    payload = replay_payload(
        source, dict(configuration=cfg, conditions=[row], scope="zero-phase test")
    )
    events = payload["conditions"][0]["events"]
    assert all(event["phase"] == "adult" for event in events)
    assert events[0]["conception"] and events[0]["birth"]


@pytest.mark.parametrize("mean", [0.0002, 0.9998])
def test_normal_initial_weights_are_conditioned_not_clipped(mean):
    from experiments.exp154.recipe import initial_weights
    from scipy.special import ndtr

    cfg = configuration("smoke", "evolution")
    std = 0.01
    cfg["genome"]["recurrent_normal"]["EI"] = dict(mean=mean, std=std)
    values = initial_weights(cfg, np.random.default_rng(42), "EI", (100_000,))
    assert np.all((values > 0) & (values < cfg["weight_upper"]))
    a, b = -mean / std, (cfg["weight_upper"] - mean) / std
    phi_a, phi_b = np.exp(-(np.array([a, b]) ** 2) / 2) / np.sqrt(2 * np.pi)
    mass = ndtr(b) - ndtr(a)
    shift = (phi_a - phi_b) / mass
    expected_mean = mean + std * shift
    expected_variance = std**2 * (1 + (a * phi_a - b * phi_b) / mass - shift**2)
    assert abs(values.mean() - expected_mean) < 5 * np.sqrt(
        expected_variance / len(values)
    )
    assert abs(values.var() / expected_variance - 1) < 0.02


def test_evolved_means_preserve_fixed_parameters_and_paired_readout():
    from experiments.exp154.recipe import individual_configuration

    cfg = configuration("smoke", "evolution")
    changed = configuration("smoke", "evolution")
    changed["genome"]["recurrent_normal"]["EI"]["mean"] = 0.2
    changed["genome"]["recurrent_normal"]["IE"]["mean"] = 0.8
    first = make_model(author_network(cfg, 7), cfg, 7)
    second = make_model(author_network(changed, 7), changed, 7)
    legacy_cfg = configuration("smoke", "lifetime")
    legacy = make_model(author_network(legacy_cfg, 7), legacy_cfg, 7)
    for name, value in first.parameter_map().items():
        other = second.parameter_map()[name]
        if name in {"EI.weight", "IE.weight"}:
            assert not torch.equal(value, other)
        else:
            assert torch.equal(value, other), name
            assert torch.equal(value, legacy.parameter_map()[name]), name
    for model in (first, second):
        output = model(encode([0.1, 0.5, 0.9], cfg), diagnostics=False).outputs[
            "prediction"
        ]
        assert torch.max(torch.abs(output)) <= cfg["decision_threshold"]
    # Resolve execution choices from the recorded run rather than today's defaults.
    cfg["metabolic_cost"] = 0.123
    cfg["evolution"]["horizon"] = 99
    resolved = individual_configuration(cfg, changed["genome"], 11)
    assert resolved["metabolic_cost"] == 0.123
    assert resolved["evolution"]["horizon"] == 99
    assert resolved["seeds"] == [11]
    assert cfg["seeds"] == cfg["evolution"]["search_seeds"] == [9]


@pytest.mark.parametrize(
    "distribution",
    [
        dict(mean=0.03, std=0),
        dict(mean=0.03, std=-0.01),
        dict(mean=0.03, std=float("nan")),
        dict(mean=-0.03, std=0.01),
        dict(mean=1.1, std=0.01),
        dict(mean=1e-6, std=0.01),
    ],
)
def test_invalid_normal_initial_weight_settings_are_rejected(distribution):
    cfg = configuration("smoke", "evolution")
    with pytest.raises(ValueError, match="weight mean|normal initial"):
        configuration(
            "smoke",
            "evolution",
            genome={
                "recurrent_normal": {
                    **cfg["genome"]["recurrent_normal"],
                    "EI": distribution,
                },
            },
        )


def test_evolution_overview_averages_the_population_instead_of_the_winner():
    import copy

    import matplotlib.pyplot as plt
    from experiments.exp154.present import evolution_overview_figure

    cfg = configuration("smoke", "evolution")
    cfg["evolution"]["population"] = 2
    founder = copy.deepcopy(cfg["genome"])
    second, third = copy.deepcopy(founder), copy.deepcopy(founder)
    second["recurrent_normal"]["EI"]["mean"] = 0.09
    third["recurrent_normal"]["EI"]["mean"] = 0.15
    result = dict(
        configuration=cfg,
        founder="a",
        individuals=[
            dict(candidate=key, genome=genome)
            for key, genome in zip(
                ("a", "b", "c"),
                (founder, second, third),
                strict=True,
            )
        ],
        generations=[
            dict(
                generation=1,
                members=["a", "b"],
                ranked=[
                    dict(candidate="b", mean_survival=10),
                    dict(candidate="a", mean_survival=2),
                ],
            ),
            dict(
                generation=2,
                members=["b", "c"],
                ranked=[
                    dict(candidate="c", mean_survival=14),
                    dict(candidate="b", mean_survival=10),
                ],
            ),
        ],
    )
    figure = evolution_overview_figure(result)
    try:
        assert len(figure.axes) == 9
        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [6, 12])
        np.testing.assert_allclose(figure.axes[1].lines[0].get_ydata(), [0.06, 0.12])
        np.testing.assert_allclose(figure.axes[0].lines[1].get_ydata(), [2, 2])
        assert "mean" in figure.axes[1].get_title(loc="left")
        assert "fixed σ" in figure.axes[1].get_title(loc="left")
        assert figure.axes[-1].get_xlabel() == "Generation"
    finally:
        plt.close(figure)
