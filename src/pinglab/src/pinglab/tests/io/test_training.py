"""Tests for pinglab.io.training."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pinglab.io.training import encode_rate, train_epoch, eval_epoch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec(n_e=20, n_i=5, seed=0):
    return {
        "schema_version": "pinglab-graph.v1",
        "meta": {"scan_id": "test"},
        "sim": {"dt_ms": 0.1, "T_ms": 50.0, "seed": seed, "neuron_model": "lif"},
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
            {"id": "inp", "kind": "input", "type": "tonic", "size": 0},
            {"id": "E", "kind": "population", "type": "E", "size": n_e},
            {"id": "I", "kind": "population", "type": "I", "size": n_i},
        ],
        "edges": [
            {"id": "in_e", "kind": "input", "from": "inp", "to": "E",
             "w": {"mean": 0.5, "std": 0.0}},
            {"id": "e_i", "kind": "EI", "from": "E", "to": "I",
             "w": {"mean": 0.0, "std": 0.1}, "delay_ms": 0.5},
            {"id": "i_e", "kind": "IE", "from": "I", "to": "E",
             "w": {"mean": 0.0, "std": 0.1}, "delay_ms": 1.0},
            {"id": "e_e", "kind": "EE", "from": "E", "to": "E",
             "w": {"mean": 0.0, "std": 0.05}, "delay_ms": 1.0},
        ],
        "inputs": {"inp": {"mode": "tonic", "mean": 0.5, "std": 0.0, "seed": 0}},
    }


def _make_runtime(n_e=20, n_i=5, seed=0, trainable=False):
    from pinglab.io.compiler import compile_graph_to_runtime
    return compile_graph_to_runtime(_make_spec(n_e=n_e, n_i=n_i, seed=seed), backend="pytorch", trainable=trainable)


# ---------------------------------------------------------------------------
# trainable=True via compile_graph_to_runtime
# ---------------------------------------------------------------------------

class TestTrainableWeights:
    def test_trainable_returns_new_runtime(self):
        rt = _make_runtime()
        rt2 = _make_runtime(trainable=True)
        assert rt2 is not rt

    def test_weight_matrices_require_grad(self):
        rt = _make_runtime(trainable=True)
        for attr in ("W_ee", "W_ei", "W_ie", "W_ii"):
            w = getattr(rt.weights, attr)
            assert w.requires_grad, f"{attr} should require grad"

    def test_non_trainable_weights_no_grad(self):
        rt = _make_runtime(trainable=False)
        for attr in ("W_ee", "W_ei", "W_ie", "W_ii"):
            w = getattr(rt.weights, attr)
            assert not w.requires_grad, f"{attr} should not require grad"

    def test_delay_matrices_not_trainable(self):
        rt = _make_runtime(trainable=True)
        for attr in ("D_ee", "D_ei", "D_ie", "D_ii"):
            d = getattr(rt.weights, attr)
            if d is not None:
                assert not d.requires_grad, f"{attr} should not require grad"

    def test_weight_values_same_trainable_vs_not(self):
        rt = _make_runtime(seed=7)
        rt_tr = _make_runtime(seed=7, trainable=True)
        assert torch.allclose(rt.weights.W_ee, rt_tr.weights.W_ee)

    def test_gradient_flows_through_simulation(self):
        from pinglab.backends.pytorch import simulate_network, surrogate_lif_step

        rt = _make_runtime(n_e=10, n_i=5, trainable=True)
        _, spikes = simulate_network(
            rt,
            spike_fn=surrogate_lif_step,
            return_spike_tensor=True,
        )
        loss = spikes.sum()
        loss.backward()
        assert rt.weights.W_ee.grad is not None
        assert rt.weights.W_ee.grad.shape == rt.weights.W_ee.shape

    def test_weight_masks_present_when_trainable(self):
        rt = _make_runtime(trainable=True)
        assert rt.weights.M_ee is not None
        assert rt.weights.M_ee.shape == rt.weights.W_ee.shape

    def test_gradient_masked_outside_edges(self):
        """Gradients should be zero where the structural mask is zero."""
        from pinglab.backends.pytorch import simulate_network, surrogate_lif_step

        rt = _make_runtime(n_e=20, n_i=5, trainable=True)
        _, spikes = simulate_network(
            rt, spike_fn=surrogate_lif_step, return_spike_tensor=True,
        )
        spikes.sum().backward()
        grad = rt.weights.W_ee.grad
        mask = rt.weights.M_ee
        assert grad is not None
        # Gradients outside the mask must be exactly zero
        outside_mask = mask == 0
        if outside_mask.any():
            assert torch.all(grad[outside_mask] == 0), \
                "gradient leaked into non-edge weight entries"


# ---------------------------------------------------------------------------
# encode_rate
# ---------------------------------------------------------------------------

class TestEncodeRate:
    def test_output_shape(self):
        img = torch.rand(1, 28, 28)
        ext = encode_rate(img, T_steps=100, n_total=800, n_input=784)
        assert ext.shape == (100, 800)

    def test_input_slice_matches_pixels(self):
        img = torch.rand(1, 28, 28)
        scale = 3.0
        ext = encode_rate(img, T_steps=50, n_total=784, n_input=784, scale=scale)
        pixels = img.reshape(-1)
        for t in range(50):
            assert torch.allclose(ext[t], pixels * scale)

    def test_padding_zeros(self):
        img = torch.rand(1, 4, 4)  # 16 pixels
        ext = encode_rate(img, T_steps=10, n_total=32, n_input=16)
        assert torch.all(ext[:, 16:] == 0.0)

    def test_constant_across_timesteps(self):
        img = torch.rand(1, 4, 4)
        ext = encode_rate(img, T_steps=20, n_total=16, n_input=16)
        # Each row should be identical
        assert torch.allclose(ext[0], ext[-1])

    def test_scale_applied(self):
        img = torch.ones(1, 2, 2)
        ext = encode_rate(img, T_steps=5, n_total=4, n_input=4, scale=7.5)
        assert torch.allclose(ext, torch.full((5, 4), 7.5))

    def test_wrong_input_size_raises(self):
        img = torch.rand(1, 4, 4)  # 16 pixels
        with pytest.raises(ValueError, match="n_input"):
            encode_rate(img, T_steps=10, n_total=32, n_input=20)

    def test_flat_input_accepted(self):
        img = torch.rand(784)
        ext = encode_rate(img, T_steps=10, n_total=784, n_input=784)
        assert ext.shape == (10, 784)


# ---------------------------------------------------------------------------
# Helpers for train_epoch / eval_epoch
# ---------------------------------------------------------------------------

def _make_loader(n_samples=20, n_features=8, n_classes=3, batch_size=4):
    X = torch.randn(n_samples, n_features)
    y = torch.randint(0, n_classes, (n_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False)


def _make_linear_forward(n_features=8, n_classes=3):
    """Return a simple differentiable forward_fn (linear layer) for testing."""
    W = torch.randn(n_classes, n_features, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)

    def forward(X: torch.Tensor) -> torch.Tensor:
        return X @ W.T + b

    return forward, W, b


# ---------------------------------------------------------------------------
# train_epoch
# ---------------------------------------------------------------------------

class TestTrainEpoch:
    def test_returns_tuple_of_three(self):
        loader = _make_loader()
        forward, W, b = _make_linear_forward()
        opt = torch.optim.SGD([W, b], lr=0.01)
        result = train_epoch(loader, opt, forward, n_total_samples=20)
        assert isinstance(result, tuple) and len(result) == 3

    def test_avg_loss_is_float(self):
        loader = _make_loader()
        forward, W, b = _make_linear_forward()
        opt = torch.optim.SGD([W, b], lr=0.01)
        avg_loss, _, _ = train_epoch(loader, opt, forward, n_total_samples=20)
        assert isinstance(avg_loss, float)

    def test_iter_losses_length_matches_batches(self):
        n_samples, batch_size = 20, 4
        loader = _make_loader(n_samples=n_samples, batch_size=batch_size)
        forward, W, b = _make_linear_forward()
        opt = torch.optim.SGD([W, b], lr=0.01)
        _, iter_losses, _ = train_epoch(loader, opt, forward, n_total_samples=n_samples)
        assert len(iter_losses) == len(loader)

    def test_iter_accs_length_matches_batches(self):
        n_samples, batch_size = 20, 4
        loader = _make_loader(n_samples=n_samples, batch_size=batch_size)
        forward, W, b = _make_linear_forward()
        opt = torch.optim.SGD([W, b], lr=0.01)
        _, _, iter_accs = train_epoch(loader, opt, forward, n_total_samples=n_samples)
        assert len(iter_accs) == len(loader)

    def test_weights_updated_after_epoch(self):
        loader = _make_loader()
        forward, W, b = _make_linear_forward()
        W_before = W.detach().clone()
        opt = torch.optim.SGD([W, b], lr=0.1)
        train_epoch(loader, opt, forward, n_total_samples=20)
        assert not torch.allclose(W.detach(), W_before), "weights should change after training"

    def test_iter_accs_in_range(self):
        loader = _make_loader()
        forward, W, b = _make_linear_forward()
        opt = torch.optim.SGD([W, b], lr=0.01)
        _, _, iter_accs = train_epoch(loader, opt, forward, n_total_samples=20)
        for acc in iter_accs:
            assert 0.0 <= acc <= 1.0

    def test_loss_decreases_over_multiple_epochs(self):
        loader = _make_loader(n_samples=40, batch_size=8)
        forward, W, b = _make_linear_forward()
        opt = torch.optim.Adam([W, b], lr=0.1)
        losses = []
        for _ in range(5):
            avg, _, _ = train_epoch(loader, opt, forward, n_total_samples=40)
            losses.append(avg)
        assert losses[-1] < losses[0], "loss should decrease with training"


# ---------------------------------------------------------------------------
# eval_epoch
# ---------------------------------------------------------------------------

class TestEvalEpoch:
    def test_returns_tuple_of_two(self):
        loader = _make_loader()
        forward, _, _ = _make_linear_forward()
        result = eval_epoch(loader, forward)
        assert isinstance(result, tuple) and len(result) == 2

    def test_avg_loss_is_positive_float(self):
        loader = _make_loader()
        forward, _, _ = _make_linear_forward()
        avg_loss, _ = eval_epoch(loader, forward)
        assert isinstance(avg_loss, float)
        assert avg_loss >= 0.0

    def test_accuracy_in_range(self):
        loader = _make_loader()
        forward, _, _ = _make_linear_forward()
        _, acc = eval_epoch(loader, forward)
        assert 0.0 <= acc <= 1.0

    def test_perfect_predictor_accuracy_one(self):
        n_classes = 3
        X = torch.eye(n_classes).float()
        y = torch.arange(n_classes)
        loader = DataLoader(TensorDataset(X, y), batch_size=n_classes, shuffle=False)

        # forward_fn returns logits that perfectly predict y
        def perfect_forward(batch_X):
            # one-hot logits: class i gets score 1 for sample i
            return batch_X * 10.0

        _, acc = eval_epoch(loader, perfect_forward)
        assert abs(acc - 1.0) < 1e-6

    def test_no_weight_updates_during_eval(self):
        loader = _make_loader()
        forward, W, b = _make_linear_forward()
        W_before = W.detach().clone()
        eval_epoch(loader, forward)
        assert torch.allclose(W.detach(), W_before), "eval_epoch must not modify weights"
