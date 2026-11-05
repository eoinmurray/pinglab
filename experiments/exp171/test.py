"""Focused non-simulation checks of scientific boundaries and input generation."""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from experiments.exp171 import recipe
from experiments.exp171.analyse import summarise
from experiments.exp171.compute import (
    artificial,
    calibrate,
    digest,
    filtered,
    forward,
    select_indices,
    sensory,
)
from experiments.exp171.custom.inhibition import history_state
from snnlab.sim import GraphExecutor, plan_graph


def test_dependency_boundary():
    root = Path(__file__).parent
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            for name in names:
                assert not name.startswith("experiments.exp") or name.startswith(
                    "experiments.exp171"
                )
        assert not any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "__import__"
            for n in ast.walk(tree)
        )
    cfg = recipe.configuration()
    assert cfg["source_run"] == "exp022-r015-compute"
    assert cfg["rate_penalty"] == 0


def test_selection_and_original_membership():
    labels = np.tile(np.arange(10), 6000)
    cfg = recipe.configuration()
    train, val, source_train, source_val = select_indices(labels, cfg)
    assert len(set(train) & set(val)) == 0
    assert len(source_train) == 6300 and len(source_val) == 700
    assert np.isin(train, source_train).all() and np.isin(val, source_val).all()
    np.testing.assert_array_equal(np.bincount(labels[train]), np.full(10, 20))
    np.testing.assert_array_equal(np.bincount(labels[val]), np.full(10, 10))
    for a, b in zip((train, val), select_indices(labels, cfg)[:2], strict=True):
        np.testing.assert_array_equal(a, b)


def test_delay_and_decay():
    cfg = recipe.configuration()
    events = np.zeros((6, 2))
    events[0] = [1, 2]
    g = filtered(events, cfg)
    np.testing.assert_array_equal(g[0], [0, 0])
    np.testing.assert_array_equal(g[1], [1, 2])
    np.testing.assert_allclose(g[2], np.exp(-0.1 / 6) * np.array([1, 2]))


def test_independent_reproducible_draws_and_fixed_encoding():
    cfg = recipe.configuration()
    cfg.update(n_e=3, n_input=4, steps=100)
    p = np.full((2, 3), 0.1)
    q = np.full((2, 3), 0.02)
    indices = np.array([12, 17])
    first = artificial(p, q, indices, cfg, epoch=1)
    assert first.shape == (cfg["inhibition_prehistory_steps"] + cfg["steps"], 2, 3)
    np.testing.assert_array_equal(first, artificial(p, q, indices, cfg, epoch=1))
    assert digest(first) != digest(artificial(p, q, indices, cfg, epoch=2))
    assert digest(first) != digest(artificial(p, q, indices, cfg, draw=1))
    np.testing.assert_array_equal(
        first[:, :1], artificial(p[:1], q[:1], indices[:1], cfg, epoch=1)
    )
    np.testing.assert_array_equal(
        sensory(np.ones(4), 12, cfg), sensory(np.ones(4), 12, cfg)
    )


def test_calibration_uses_only_conductance():
    cfg = recipe.configuration()
    cfg.update(n_e=3, steps=2000)
    events = np.random.default_rng(9).random((2000, 3)) < 0.04
    target = filtered(events, cfg) * np.array([0.01, 0.02, 0.03])
    p, q, mu, sd, fm, fs, error = calibrate(target, cfg, 17)
    assert np.all(p > 0) and np.all(q > 0)
    np.testing.assert_allclose(mu, fm, rtol=1e-12)
    assert np.max(np.abs(fs / sd - 1)) < 0.03 and error < 0.03


def test_metrics_average_draws_without_ensemble_voting():
    scores = np.array(
        [[[4.0, 0.0], [0.0, 4.0]], [[0.0, 4.0], [0.0, 4.0]]], dtype=np.float32
    )
    measured, loss = summarise(scores, np.array([0, 1]), np.ones((2, 2)) * 5)
    assert measured["accuracy_pct"] == 75
    assert measured["e_rate_hz"] == 5
    assert measured["cross_entropy"] == pytest.approx(loss.mean())


def test_native_authoring_and_trainable_membership():
    cfg = recipe.configuration()
    graph = recipe.author_network(cfg).graph
    assert {p["id"] for p in graph["projections"]} == {
        "input_to_E",
        "readout_projection",
        "E_to_I",
        "I_to_E",
        "artificial_to_E",
    }
    assert set(recipe.TRAINABLE) == {"input_to_E.weight", "readout_projection.weight"}
    assert "artificial" not in {
        p["id"] for p in recipe.author_network(cfg, teacher=True).graph["inputs"]
    }
    with pytest.raises(ValueError):
        recipe.configuration(epochs=0)


def test_twenty_epoch_default_and_original_pilot_override():
    assert recipe.configuration()["epochs"] == 20
    assert recipe.configuration(epochs=3)["epochs"] == 3
    assert recipe.configuration()["batch_size"] == 20


def test_cuda_configuration_without_allocating_device():
    cfg = recipe.configuration(device="cuda")
    assert cfg["calibration_batch_size"] == 10
    assert cfg["precision"] == "highest"
    assert cfg["epochs"] == 20
    assert recipe.configuration()["calibration_batch_size"] == 1


def test_raster_outputs_preserve_network_dynamics():
    cfg = recipe.configuration()
    original = recipe.author_network(cfg).graph
    observed = recipe.author_network(cfg, record_spikes=True).graph
    for key in ("populations", "projections", "parameters", "timebase"):
        assert original[key] == observed[key]
    assert {row["id"] for row in observed["outputs"]} >= {"E_spikes", "I_spikes"}


def test_batched_history_preserves_pending_events_and_resets_cells():
    cfg = recipe.configuration(inhibition_prehistory_ms=0.3)
    cfg.update(n_e=3, n_i=2, n_input=4, n_output=2)
    model = GraphExecutor(
        plan_graph(recipe.author_network(cfg).graph), compile_mode=None
    )
    history = np.array(
        [
            [[0.1, 0, 0], [0, 0.2, 0]],
            [[0, 0, 0.3], [0.4, 0, 0]],
            [[0, 0.5, 0], [0, 0, 0.6]],
        ],
        dtype=np.float32,
    )
    state = history_state(model, history)
    decay = np.exp(-cfg["dt_ms"] / cfg["tau_gaba_ms"])
    expected_first = decay * (decay * history[0] + history[1]) + history[-1]
    np.testing.assert_allclose(
        decay * state.conductances["artificial_to_E"].numpy()
        + state.input_histories["artificial"].numpy()[0],
        expected_first,
        rtol=1e-6,
    )
    assert state.completed_steps == 0
    assert not state.reduction_state and not state.reduction_tensors
    for name, value in state.voltages.items():
        assert value.shape[0] == 2
        assert torch.all(value == (0 if name == "readout" else -65))
    for group in (state.refractory, state.population_histories):
        assert all(not torch.count_nonzero(v) for v in group.values())
    assert all(
        not torch.count_nonzero(v)
        for name, v in state.conductances.items()
        if name != "artificial_to_E"
    )
    assert not state.conductances["artificial_to_E"].requires_grad
    state.voltages["E"].fill_(123)
    assert torch.all(history_state(model, history).voltages["E"] == -65)


def test_forward_passes_only_scored_inputs_with_private_initial_state():
    from types import SimpleNamespace

    cfg = recipe.configuration(inhibition_prehistory_ms=0.3)
    cfg.update(n_e=3, n_i=2, n_input=4, n_output=2, steps=10)
    native = GraphExecutor(
        plan_graph(recipe.author_network(cfg).graph), compile_mode=None
    )
    calls = []

    class CaptureOnly:
        plan = native.plan
        projection_mode = native.projection_mode
        parameter_map = native.parameter_map

        def __call__(self, inputs, **kwargs):
            calls.append((inputs, kwargs))
            return SimpleNamespace(outputs={})

    p, q = np.full((2, 3), 0.5), np.full((2, 3), 0.1)
    drive = np.zeros((cfg["steps"], 2, cfg["n_input"]), np.float32)
    for mode in (dict(epoch=1), dict(draw=0)):
        events = artificial(p, q, np.array([12, 17]), cfg, **mode)
        forward(CaptureOnly(), drive, events)
        inputs, kwargs = calls[-1]
        np.testing.assert_array_equal(inputs["drive"].numpy(), drive)
        np.testing.assert_array_equal(inputs["artificial"].numpy(), events[3:])
        np.testing.assert_allclose(
            kwargs["runtime_state"].conductances["artificial_to_E"].numpy(),
            filtered(events[:3], cfg)[-1],
            rtol=1e-6,
        )
        assert inputs["drive"].shape[0] == inputs["artificial"].shape[0] == cfg["steps"]


def test_history_configuration_and_legacy_draws():
    cfg = recipe.configuration()
    assert cfg["inhibition_prehistory_ms"] == 60
    assert cfg["inhibition_prehistory_steps"] == 600
    for invalid in (-1, 0.15, float("nan")):
        with pytest.raises(ValueError):
            recipe.configuration(inhibition_prehistory_ms=invalid)
    # Archived configurations with no history fields retain their original RNG draws.
    cfg.update(n_e=2, steps=10)
    cfg.pop("inhibition_prehistory_steps")
    ev = artificial(np.full((1, 2), 0.5), np.ones((1, 2)), np.array([17]), cfg, draw=0)
    assert ev.shape == (10, 1, 2)


def test_native_coordinated_binding(tmp_path):
    from experiments.exp171.compute import encode_native
    from snnlab import extensions, sim

    cfg = recipe.configuration(scaled=True)
    bundle = recipe.author_network(cfg)
    dense = recipe.author_network(cfg, dense_private=True)
    for authored, fan_in in ((bundle, 1), (dense, cfg["n_e"])):
        private = next(
            row for row in authored.graph["inputs"] if row["id"] == "artificial"
        )
        assert private["signal_type"] == "continuous"
        weight = next(
            row
            for row in authored.graph["parameters"]
            if row["id"] == "artificial_to_E.weight"
        )
        assert weight["shape"] == [cfg["n_e"], fan_in]
    extensions.register_encoder("tests.exp171/v1", encode_native)
    path = tmp_path / "dataset.npz"
    ids = [17, 42, 66]
    np.savez(
        path,
        ids=ids,
        features=np.full((3, 784), 200, dtype=np.uint8),
        labels=[1, 2, 3],
        probability=np.full((3, 1024), 0.05, dtype=np.float32),
        amplitude=np.full((3, 1024), 0.02, dtype=np.float32),
    )
    encoder = sim.DatasetEncoder(
        "custom",
        duration_ms=cfg["duration_ms"],
        definition="tests.exp171/v1",
        rng_policy="sample_v1",
        config={
            "recipe": cfg,
            "signature": sim.runtime_state_signature(plan_graph(bundle.graph)),
        },
    )
    binding = sim.DatasetSnapshotBinding(
        path,
        "drive",
        "test",
        "validation",
        encoder,
        target_id="digit",
        sample_ids_key="ids",
        sample_indices=(2, 0),
    )
    resolved, (target,) = sim.resolve_dataset_snapshot_binding(bundle.graph, binding)
    torch.testing.assert_close(target.value, torch.tensor([3, 1]))
    np.testing.assert_array_equal(
        resolved.tensors["drive"][:, 0],
        sensory(np.full(784, 200 / 255), 66, cfg),
    )
    with np.load(path) as dataset:
        events = artificial(
            dataset["probability"][[2, 0]],
            dataset["amplitude"][[2, 0]],
            [66, 17],
            cfg,
            draw=0,
        )
    history = cfg["inhibition_prehistory_steps"]
    state = resolved.tensors.initial_state
    np.testing.assert_array_equal(resolved.tensors["artificial"], events[history:])
    np.testing.assert_allclose(
        state.conductances["artificial_to_E"],
        filtered(events[:history], cfg)[-1],
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        state.input_histories["artificial"], events[history - 1 : history]
    )
    assert not state.currents
    assert sim.Batching(microbatch_size=16).sizes(6300, cfg["batch_size"]) == [
        256
    ] * 24 + [156]
