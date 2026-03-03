"""Tests for pinglab.backends.pytorch.training (get_device, run_batch)."""

import pytest
import torch

from pinglab.backends.pytorch.training import get_device, run_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ff_spec(n_in=16, n_hidden=8, n_out=4):
    """Minimal feedforward-only SNN spec (no I neurons) for run_batch tests."""
    n_e = n_in + n_hidden + n_out
    return {
        "schema_version": "pinglab-graph.v1",
        "meta": {"scan_id": "test-run-batch"},
        "sim": {"dt_ms": 1.0, "T_ms": 10.0, "seed": 0, "neuron_model": "lif"},
        "execution": {"performance_mode": False, "max_spikes": 10000, "burn_in_ms": 0},
        "constraints": {"nonnegative_weights": False, "nonnegative_input": False},
        "biophysics": {
            "V_init": -65, "E_L": -65, "E_e": 0, "E_i": -80,
            "C_m_E": 1, "g_L_E": 0.05, "C_m_I": 1, "g_L_I": 0.1,
            "V_th": -50, "V_reset": -65, "t_ref_E": 3, "t_ref_I": 1.5,
            "tau_ampa": 2, "tau_gaba": 6.5,
            "g_L_heterogeneity_sd": 0.0, "C_m_heterogeneity_sd": 0.0,
            "V_th_heterogeneity_sd": 0.0, "t_ref_heterogeneity_sd": 0.0,
        },
        "nodes": [
            {"id": "inp",   "kind": "input",      "type": "tonic",  "size": 0},
            {"id": "E_in",  "kind": "population",  "type": "E",      "size": n_in},
            {"id": "E_hid", "kind": "population",  "type": "E",      "size": n_hidden},
            {"id": "E_out", "kind": "population",  "type": "E",      "size": n_out},
        ],
        "edges": [
            {"id": "in_ein",   "kind": "input", "from": "inp",   "to": "E_in",
             "w": {"mean": 0.0, "std": 0.0}},
            {"id": "ein_ehid", "kind": "EE",    "from": "E_in",  "to": "E_hid",
             "w": {"mean": 0.05, "std": 0.01}, "delay_ms": 1.0},
            {"id": "ehid_eout","kind": "EE",    "from": "E_hid", "to": "E_out",
             "w": {"mean": 0.05, "std": 0.01}, "delay_ms": 1.0},
        ],
        "inputs": {"inp": {"mode": "tonic", "mean": 0.0, "std": 0.0, "seed": 0}},
    }


def _make_runtime(n_in=16, n_hidden=8, n_out=4, trainable=True):
    from pinglab.io.compiler import compile_graph_to_runtime
    return compile_graph_to_runtime(
        _make_ff_spec(n_in=n_in, n_hidden=n_hidden, n_out=n_out),
        backend="pytorch",
        trainable=trainable,
    )


def _pop_bounds(spec):
    """Return (n_total, n_input, out_start, out_stop) from a compiled plan."""
    from pinglab.io.compiler import compile_graph
    plan = compile_graph(spec)
    idx = plan["population_index"]
    n_total  = int(plan["totals"]["N_E"])
    n_input  = int(idx["E_in"]["stop"] - idx["E_in"]["start"])
    out_start = int(idx["E_out"]["start"])
    out_stop  = int(idx["E_out"]["stop"])
    return n_total, n_input, out_start, out_stop


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_returns_string(self):
        d = get_device()
        assert isinstance(d, str)

    def test_cpu_fallback(self):
        # On most CI machines there is no accelerator; at minimum it should
        # return a valid torch device string.
        d = get_device()
        torch.device(d)  # raises if invalid

    def test_consistent_calls(self):
        assert get_device() == get_device()


# ---------------------------------------------------------------------------
# run_batch
# ---------------------------------------------------------------------------

class TestRunBatch:
    N_IN = 16
    N_HID = 8
    N_OUT = 4
    T_STEPS = 10

    @pytest.fixture(scope="class")
    def setup(self):
        spec = _make_ff_spec(self.N_IN, self.N_HID, self.N_OUT)
        n_total, n_input, out_start, out_stop = _pop_bounds(spec)
        rt = _make_runtime(self.N_IN, self.N_HID, self.N_OUT, trainable=True)
        return rt, n_total, n_input, out_start, out_stop

    def _kwargs(self, n_total, n_input, out_start, out_stop):
        return dict(
            T_steps=self.T_STEPS,
            n_total=n_total,
            n_input=n_input,
            out_start=out_start,
            out_stop=out_stop,
            input_scale=1.0,
        )

    def test_output_shape(self, setup):
        rt, n_total, n_input, out_start, out_stop = setup
        B = 3
        images = torch.rand(B, n_input)
        logits = run_batch(rt, images, **self._kwargs(n_total, n_input, out_start, out_stop))
        assert logits.shape == (B, self.N_OUT)

    def test_single_sample(self, setup):
        rt, n_total, n_input, out_start, out_stop = setup
        images = torch.rand(1, n_input)
        logits = run_batch(rt, images, **self._kwargs(n_total, n_input, out_start, out_stop))
        assert logits.shape == (1, self.N_OUT)

    def test_logits_are_nonneg_spike_counts(self, setup):
        rt, n_total, n_input, out_start, out_stop = setup
        images = torch.rand(4, n_input)
        logits = run_batch(rt, images, **self._kwargs(n_total, n_input, out_start, out_stop))
        assert torch.all(logits >= 0), "spike count logits must be >= 0"

    def test_gradient_flows_to_weights(self, setup):
        rt, n_total, n_input, out_start, out_stop = setup
        images = torch.rand(2, n_input)
        logits = run_batch(rt, images, **self._kwargs(n_total, n_input, out_start, out_stop))
        logits.sum().backward()
        assert rt.weights.W_ee.grad is not None
        assert rt.weights.W_ee.grad.shape == rt.weights.W_ee.shape

    def test_last_batch_padding(self, setup):
        """Smaller batch than sim_state.batch_size should still return correct shape."""
        from pinglab.backends.pytorch import prepare_runtime_tensors

        rt, n_total, n_input, out_start, out_stop = setup
        batch_size = 4
        sim_state = prepare_runtime_tensors(rt, training_mode=True, batch_size=batch_size)

        # Pass only 2 images — smaller than the pre-built state's batch_size=4
        images = torch.rand(2, n_input)
        logits = run_batch(
            rt, images,
            sim_state=sim_state,
            **self._kwargs(n_total, n_input, out_start, out_stop),
        )
        assert logits.shape == (2, self.N_OUT)

    def test_input_scale_affects_output(self, setup):
        rt, n_total, n_input, out_start, out_stop = setup
        images = torch.rand(2, n_input)
        kw = self._kwargs(n_total, n_input, out_start, out_stop)

        logits_low  = run_batch(rt, images, **{**kw, "input_scale": 0.0})
        logits_high = run_batch(rt, images, **{**kw, "input_scale": 5.0})
        # Stronger input should generally produce more (or equal) spikes
        assert logits_high.sum() >= logits_low.sum()


class TestWeightMaskFF:
    """Verify the structural mask prevents weight leaking in feedforward networks."""

    def test_mask_only_covers_defined_edges(self):
        """M_ee should have ones only in (E_in→E_hid) and (E_hid→E_out) blocks."""
        spec = _make_ff_spec(n_in=16, n_hidden=8, n_out=4)
        rt = _make_runtime(n_in=16, n_hidden=8, n_out=4, trainable=True)
        _, n_input, out_start, out_stop = _pop_bounds(spec)

        M = rt.weights.M_ee
        assert M is not None
        n_total = 16 + 8 + 4  # 28
        assert M.shape == (n_total, n_total)

        # Only two blocks should be nonzero:
        # E_in→E_hid: M[16:24, 0:16]
        assert M[16:24, 0:16].sum().item() == 16 * 8
        # E_hid→E_out: M[24:28, 16:24]
        assert M[24:28, 16:24].sum().item() == 8 * 4
        # Everything else should be zero
        total_ones = M.sum().item()
        expected = 16 * 8 + 8 * 4
        assert total_ones == expected, f"mask has {total_ones} ones, expected {expected}"

    def test_no_gradient_leak_after_backward(self):
        """After backward, gradients outside the mask must be exactly zero."""
        from pinglab.backends.pytorch import simulate_network, surrogate_lif_step
        from pinglab.io.training import encode_rate_to_tonic

        rt = _make_runtime(n_in=16, n_hidden=8, n_out=4, trainable=True)
        n_total, n_input, out_start, out_stop = _pop_bounds(
            _make_ff_spec(n_in=16, n_hidden=8, n_out=4)
        )
        img = torch.rand(1, n_input)
        ext = encode_rate_to_tonic(img, T_steps=10, n_total=n_total, n_input=n_input, scale=1.0)
        ext = ext.unsqueeze(0)  # [1, T, N]
        _, spikes = simulate_network(
            rt, external_input=ext, spike_fn=surrogate_lif_step, return_spike_tensor=True,
        )
        spikes.sum().backward()

        grad = rt.weights.W_ee.grad
        mask = rt.weights.M_ee
        assert grad is not None
        outside = mask == 0
        assert torch.all(grad[outside] == 0), "gradient leaked through non-edge block"
