"""Cross-library parity coverage; intentionally not executed before commit/push."""

from types import SimpleNamespace

import torch
from experiments.exp126.compute_parts.matched_models import (
    LabModel,
    TorchModel,
    neuron_step,
)
from experiments.exp126.recipe import comparison_configuration
from torch import nn


def test_shared_events_match_states_spikes_and_parameter_gradients():
    cfg = {
        **comparison_configuration(),
        "inputs": 2,
        "hidden": 2,
        "outputs": 2,
        "steps": 4,
    }
    initial = {
        "input_weight": torch.tensor([[1.2, 0.3], [0.2, 1.1]]),
        "input_bias": torch.tensor([[0.1, 0.1]]),
        "output_weight": torch.tensor([[0.9, 0.3], [0.2, 1.1]]),
        "output_bias": torch.tensor([[0.1, 0.1]]),
    }
    events = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ]
    )
    a, b = TorchModel(cfg, initial), LabModel(cfg, initial)
    logits_a, traces_a = a(events, record=True)
    logits_b, traces_b = b(events, record=True)
    assert set(traces_a) == set(traces_b)
    for key in traces_a:
        torch.testing.assert_close(traces_a[key], traces_b[key], atol=1e-6, rtol=1e-5)
        if key.endswith("spikes"):
            assert torch.equal(traces_a[key], traces_b[key])
    torch.testing.assert_close(logits_a, logits_b)
    labels = torch.tensor([0, 1])
    nn.functional.cross_entropy(logits_a, labels).backward()
    nn.functional.cross_entropy(logits_b, labels).backward()
    for key in initial:
        torch.testing.assert_close(
            a.canonical()[key].grad, b.canonical()[key].grad, atol=1e-6, rtol=1e-5
        )


def test_delayed_subtractive_reset_keeps_residual_voltage():
    context = SimpleNamespace(
        config={"alpha": 0.8, "beta": 0.9, "threshold": 1.0},
        state={
            "voltage": torch.tensor([[1.5]]),
            "synaptic": torch.zeros(1, 1),
            "refractory": torch.zeros(1, 1, dtype=torch.long),
        },
        excitatory=torch.tensor([[2.0]]),
        inhibitory=torch.zeros(1, 1),
        spike=lambda value: (value > 0).float(),
    )
    state, spikes = neuron_step(context)
    torch.testing.assert_close(state["voltage"], torch.tensor([[2.35]]))
    assert spikes.item() == 1
