"""Checks for spike-readout measurements and the native SNNLab classifier."""

import numpy as np
import torch
from experiments.exp127.analyse import measurements
from experiments.exp127.recipe import CONFIG, network
from snnlab.sim.execution import DenseArrayBinding, ExecutionSpec, simulate, train


def test_latency_requires_both_spike_evidence_and_lead():
    spikes = np.zeros((10, 2, 2))
    spikes[:5, 0, 1] = 1
    spikes[1:3, 0, 0] = 1
    values = measurements(spikes, np.array([1, 0]), CONFIG)
    assert values["latency_ms"] == [4, None]
    assert values["decision_class"] == [1, None]
    assert values["predictions"] == [1, 0]
    assert values["counts"] == [[2, 5], [0, 0]]
    assert values["output_spikes"] == [7, 0]
    assert values["accuracy"] == 1


def test_measured_latency_can_be_wrong_even_when_final_prediction_is_right():
    spikes = np.zeros((16, 1, 2))
    spikes[:5, 0, 0] = 1
    spikes[5:16, 0, 1] = 1
    values = measurements(spikes, np.array([1]), CONFIG)
    assert values["latency_ms"] == [4]
    assert values["decision_class"] == [0]
    assert values["predictions"] == [1]


def test_snnlab_training_changes_spiking_graph_weights(tmp_path):
    torch.set_num_threads(1)
    bundle = network()
    spikes = torch.zeros(200, 2, 3)
    spikes[::10, 0, 0] = 1
    spikes[::10, 1, 1] = 1
    bindings = (DenseArrayBinding("inputs", spikes),)
    before = simulate(
        ExecutionSpec(
            kind="simulate",
            graph=bundle.graph,
            input_bindings=bindings,
            seed=127,
            device="cpu",
        )
    )
    result = train(
        ExecutionSpec(
            kind="train",
            graph=bundle.graph,
            training=bundle.training,
            input_bindings=bindings,
            targets={"class": torch.tensor([0, 1])},
            updates=3,
            seed=127,
            device="cpu",
            save_final_checkpoint=tmp_path / "trained.checkpoint",
        )
    )
    after = simulate(
        ExecutionSpec(
            kind="simulate",
            graph=bundle.graph,
            checkpoint=tmp_path / "trained.checkpoint",
            input_bindings=bindings,
            seed=127,
            device="cpu",
        )
    )
    assert len(result.parameters) == 1
    parameter = next(iter(result.parameters))
    assert not torch.equal(result.parameters[parameter], before.parameters[parameter])
    assert torch.isfinite(result.gradients[parameter]).all()
    assert torch.count_nonzero(result.gradients[parameter]) > 0
    assert (after.parameters[parameter] >= 0).all()
    torch.testing.assert_close(
        after.outputs["scores"],
        after.diagnostics["output_spikes"].sum(0),
        rtol=0,
        atol=0,
    )
    assert bundle.training["surrogate"]["kind"] == "fast_sigmoid"


def test_training_trace_records_loss_gradients_and_weight_updates(tmp_path):
    from experiments.exp127.compute import training_trace

    torch.set_num_threads(1)
    spikes = np.zeros((200, 2, 3), dtype=np.float32)
    spikes[::10, 0, 0] = 1
    spikes[::10, 1, 1] = 1
    trace = training_trace(
        network().write(tmp_path / "bundle"), (spikes, np.array([0, 1]), None, None)
    )
    scores = torch.from_numpy(trace["outputs"].sum(axis=1))
    for update, row in enumerate(scores):
        expected = torch.nn.functional.cross_entropy(row, torch.tensor([0, 1]))
        np.testing.assert_allclose(trace["losses"][update], float(expected), rtol=1e-6)
    gradient = trace["gradients"][0]
    clipped = gradient * min(
        1, CONFIG["gradient_clip"] / (np.linalg.norm(gradient) + 1e-6)
    )
    expected_weights = np.maximum(
        0,
        trace["weights"][0]
        - CONFIG["learning_rate"] * clipped / (np.abs(clipped) + 1e-8),
    )
    np.testing.assert_allclose(trace["weights"][1], expected_weights, atol=1e-6)
    assert trace["losses"][-1] < trace["losses"][0]
    assert np.isfinite(trace["voltages"]).all()
    assert np.count_nonzero(gradient) > 0
