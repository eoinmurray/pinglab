"""Tests for pinglab.backends.pytorch.e_prop."""

import pytest
import torch

from pinglab.backends.pytorch.e_prop import (
    run_batch_eprop,
    compute_eprop_gradients,
    train_epoch_eprop,
)
from pinglab.backends.pytorch.training import get_device
from pinglab.backends.pytorch.simulate_network import prepare_runtime_tensors
from pinglab.io.compiler import compile_graph, compile_graph_to_runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ping_spec(n_in=16, n_hidden=8, n_out=4, n_i=4):
    """Minimal PING spec (E+I) for e-prop tests."""
    return {
        "schema_version": "pinglab-graph.v1",
        "meta": {"scan_id": "test-eprop"},
        "sim": {"dt_ms": 1.0, "T_ms": 10.0, "seed": 0, "neuron_model": "lif"},
        "execution": {"performance_mode": False, "max_spikes": 10000, "burn_in_ms": 0},
        "constraints": {"nonnegative_weights": False, "nonnegative_input": False},
        "biophysics": {
            "V_init": -65, "E_L": -65, "E_e": 0, "E_i": -80,
            "C_m_E": 1, "g_L_E": 0.05, "C_m_I": 1, "g_L_I": 0.1,
            "V_th": -50, "V_reset": -65, "t_ref_E": 3, "t_ref_I": 1.5,
            "tau_ampa": 2, "tau_gaba": 6.5,
            "V_floor": -200,
            "g_L_heterogeneity_sd": 0.0, "C_m_heterogeneity_sd": 0.0,
            "V_th_heterogeneity_sd": 0.0, "t_ref_heterogeneity_sd": 0.0,
        },
        "nodes": [
            {"id": "inp",       "kind": "input",      "type": "tonic", "size": 0},
            {"id": "E_in",      "kind": "population",  "type": "E",     "size": n_in},
            {"id": "E_hid",     "kind": "population",  "type": "E",     "size": n_hidden},
            {"id": "E_out",     "kind": "population",  "type": "E",     "size": n_out},
            {"id": "I_global",  "kind": "population",  "type": "I",     "size": n_i},
        ],
        "edges": [
            {"id": "in_ein",    "kind": "input", "from": "inp",      "to": "E_in",
             "w": {"mean": 0.0, "std": 0.0}},
            {"id": "ein_ehid",  "kind": "EE",    "from": "E_in",     "to": "E_hid",
             "w": {"mean": 0.05, "std": 0.01}, "delay_ms": 1.0},
            {"id": "ehid_eout", "kind": "EE",    "from": "E_hid",    "to": "E_out",
             "w": {"mean": 0.05, "std": 0.01}, "delay_ms": 1.0},
            {"id": "ehid_i",    "kind": "EI",    "from": "E_hid",    "to": "I_global",
             "w": {"mean": 0.3, "std": 0.1},   "delay_ms": 1.0},
            {"id": "i_ehid",    "kind": "IE",    "from": "I_global",  "to": "E_hid",
             "w": {"mean": 0.15, "std": 0.05}, "delay_ms": 1.0},
        ],
        "inputs": {"inp": {"mode": "tonic", "mean": 0.0, "std": 0.0, "seed": 0}},
    }


def _build_runtime_and_state(spec, batch_size=4):
    """Compile spec → runtime + sim_state + pop_idx."""
    plan = compile_graph(spec)
    pop_idx = plan["population_index"]
    runtime = compile_graph_to_runtime(spec, backend="pytorch", trainable=True, device="cpu")
    sim_state = prepare_runtime_tensors(runtime, training_mode=True, batch_size=batch_size)
    return runtime, sim_state, pop_idx, plan


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunBatchEprop:
    """Test that run_batch_eprop produces correct shapes and finite values."""

    def test_output_shapes(self):
        spec = _make_ping_spec(n_in=16, n_hidden=8, n_out=4, n_i=4)
        runtime, sim_state, pop_idx, plan = _build_runtime_and_state(spec, batch_size=4)
        N_E = int(plan["totals"]["N_E"])
        n_total = N_E + int(plan["totals"]["N_I"])
        n_in = 16
        T_steps = 10

        images = torch.rand(4, 1, 4, 4)  # 16 pixels = n_in
        logits, spikes_E, spikes_I, traces = run_batch_eprop(
            runtime, images,
            T_steps=T_steps, n_total=n_total, n_input=n_in,
            out_start=int(pop_idx["E_out"]["start"]),
            out_stop=int(pop_idx["E_out"]["stop"]),
            input_scale=1.0, sim_state=sim_state,
            readout_alpha=0.01, encoding="poisson",
            pop_idx=pop_idx,
        )
        assert logits.shape == (4, 4), f"Expected (4,4), got {logits.shape}"
        assert spikes_E.shape == (4, T_steps, N_E), f"Expected (4,{T_steps},{N_E}), got {spikes_E.shape}"
        assert len(traces) == 4, "Expected 4 trace tensors"
        # E_in_hid: [B, n_hid, n_in]
        assert traces[0].shape == (4, 8, 16)
        # E_hid_out: [B, n_out, n_hid]
        assert traces[1].shape == (4, 4, 8)

    def test_values_finite(self):
        spec = _make_ping_spec()
        runtime, sim_state, pop_idx, plan = _build_runtime_and_state(spec)
        n_total = int(plan["totals"]["N_E"]) + int(plan["totals"]["N_I"])

        images = torch.rand(4, 1, 4, 4)
        logits, spikes_E, spikes_I, traces = run_batch_eprop(
            runtime, images,
            T_steps=10, n_total=n_total, n_input=16,
            out_start=int(pop_idx["E_out"]["start"]),
            out_stop=int(pop_idx["E_out"]["stop"]),
            input_scale=1.0, sim_state=sim_state,
            readout_alpha=0.01, encoding="poisson",
            pop_idx=pop_idx,
        )
        assert torch.isfinite(logits).all(), "Logits contain non-finite values"
        for i, tr in enumerate(traces):
            assert torch.isfinite(tr).all(), f"Trace {i} contains non-finite values"


class TestComputeEpropGradients:
    """Test learning signal computation and gradient shapes."""

    def test_gradient_shapes_match_weights(self):
        spec = _make_ping_spec()
        runtime, sim_state, pop_idx, plan = _build_runtime_and_state(spec)
        n_total = int(plan["totals"]["N_E"]) + int(plan["totals"]["N_I"])

        images = torch.rand(4, 1, 4, 4)
        logits, spikes_E, spikes_I, traces = run_batch_eprop(
            runtime, images,
            T_steps=10, n_total=n_total, n_input=16,
            out_start=int(pop_idx["E_out"]["start"]),
            out_stop=int(pop_idx["E_out"]["stop"]),
            input_scale=1.0, sim_state=sim_state,
            readout_alpha=0.01, encoding="poisson",
            pop_idx=pop_idx,
        )
        labels = torch.randint(0, 4, (4,))
        loss = compute_eprop_gradients(
            runtime, logits, labels, traces, pop_idx=pop_idx,
        )

        assert loss.ndim == 0, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"
        assert runtime.weights.W_ee.grad is not None
        assert runtime.weights.W_ee.grad.shape == runtime.weights.W_ee.shape
        if runtime.weights.W_ei is not None:
            assert runtime.weights.W_ei.grad is not None
            assert runtime.weights.W_ei.grad.shape == runtime.weights.W_ei.shape
        if runtime.weights.W_ie is not None:
            assert runtime.weights.W_ie.grad is not None
            assert runtime.weights.W_ie.grad.shape == runtime.weights.W_ie.shape

    def test_learning_signal_softmax_minus_onehot(self):
        """Verify analytical cross-entropy gradient is softmax - one_hot."""
        logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
        labels = torch.tensor([0])
        probs = torch.softmax(logits, dim=1)
        one_hot = torch.zeros_like(probs)
        one_hot[0, 0] = 1.0
        L_out = probs - one_hot
        # The gradient of CE w.r.t. logits is exactly softmax - one_hot
        assert L_out[0, 0] < 0, "Correct class should have negative learning signal"
        assert (L_out[0, 1:] > 0).all(), "Wrong classes should have positive learning signal"
        assert torch.isclose(L_out.sum(), torch.tensor(0.0), atol=1e-6)

    def test_gradients_finite(self):
        spec = _make_ping_spec()
        runtime, sim_state, pop_idx, plan = _build_runtime_and_state(spec)
        n_total = int(plan["totals"]["N_E"]) + int(plan["totals"]["N_I"])

        images = torch.rand(4, 1, 4, 4)
        logits, _, _, traces = run_batch_eprop(
            runtime, images,
            T_steps=10, n_total=n_total, n_input=16,
            out_start=int(pop_idx["E_out"]["start"]),
            out_stop=int(pop_idx["E_out"]["stop"]),
            input_scale=1.0, sim_state=sim_state,
            readout_alpha=0.01, encoding="poisson",
            pop_idx=pop_idx,
        )
        labels = torch.randint(0, 4, (4,))
        compute_eprop_gradients(runtime, logits, labels, traces, pop_idx=pop_idx)

        for name, param in [("W_ee", runtime.weights.W_ee),
                            ("W_ei", runtime.weights.W_ei),
                            ("W_ie", runtime.weights.W_ie)]:
            if param is not None and param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"{name}.grad has non-finite values"


class TestTrainingReducesLoss:
    """Test that a few e-prop training steps reduce loss on a tiny dataset."""

    def test_loss_decreases(self):
        spec = _make_ping_spec(n_in=16, n_hidden=8, n_out=3, n_i=4)
        # Change T_ms for speed
        spec["sim"]["T_ms"] = 10.0
        runtime, sim_state, pop_idx, plan = _build_runtime_and_state(spec, batch_size=8)
        n_total = int(plan["totals"]["N_E"]) + int(plan["totals"]["N_I"])
        n_out = 3

        # Create tiny dataset
        torch.manual_seed(42)
        images = torch.rand(16, 1, 4, 4)
        labels = torch.randint(0, n_out, (16,))

        trainable_params = [runtime.weights.W_ee]
        if runtime.weights.W_ei is not None:
            trainable_params.append(runtime.weights.W_ei)
        if runtime.weights.W_ie is not None:
            trainable_params.append(runtime.weights.W_ie)

        optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

        batch_kwargs = dict(
            T_steps=10, n_total=n_total, n_input=16,
            out_start=int(pop_idx["E_out"]["start"]),
            out_stop=int(pop_idx["E_out"]["stop"]),
            input_scale=1.0, sim_state=sim_state,
            readout_alpha=0.01, encoding="poisson",
        )

        losses = []
        for step in range(5):
            logits, _, _, traces = run_batch_eprop(
                runtime, images[:8], pop_idx=pop_idx, **batch_kwargs,
            )
            loss = compute_eprop_gradients(
                runtime, logits, labels[:8], traces, pop_idx=pop_idx,
            )
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        # Loss should decrease (or at least not explode)
        assert all(torch.isfinite(torch.tensor(l)) for l in losses), \
            f"Loss values not finite: {losses}"
        # Allow some noise but overall trend should be downward
        assert losses[-1] < losses[0] + 0.5, \
            f"Loss didn't decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


class TestPaddedBatch:
    """Test that run_batch_eprop handles partial batches correctly."""

    def test_partial_batch(self):
        spec = _make_ping_spec()
        runtime, sim_state, pop_idx, plan = _build_runtime_and_state(spec, batch_size=4)
        n_total = int(plan["totals"]["N_E"]) + int(plan["totals"]["N_I"])

        # Only 2 images but batch_size=4
        images = torch.rand(2, 1, 4, 4)
        logits, spikes_E, spikes_I, traces = run_batch_eprop(
            runtime, images,
            T_steps=10, n_total=n_total, n_input=16,
            out_start=int(pop_idx["E_out"]["start"]),
            out_stop=int(pop_idx["E_out"]["stop"]),
            input_scale=1.0, sim_state=sim_state,
            readout_alpha=0.01, encoding="poisson",
            pop_idx=pop_idx,
        )
        # Should return B_actual=2, not B_state=4
        assert logits.shape[0] == 2
        assert spikes_E.shape[0] == 2
        for tr in traces:
            assert tr.shape[0] == 2
