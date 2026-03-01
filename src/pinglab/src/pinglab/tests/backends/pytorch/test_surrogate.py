"""Tests for surrogate gradient spike function."""

import pytest
import torch

from pinglab.backends.pytorch.surrogate import SpikeFunction, surrogate_lif_step


class TestSpikeFunction:
    def test_forward_fires_above_threshold(self):
        u = torch.tensor([0.1, 1.0, 5.0])
        out = SpikeFunction.apply(u)
        assert torch.all(out == 1.0)

    def test_forward_silent_below_threshold(self):
        u = torch.tensor([-0.1, -1.0, -5.0])
        out = SpikeFunction.apply(u)
        assert torch.all(out == 0.0)

    def test_forward_at_zero_fires(self):
        u = torch.tensor([0.0])
        out = SpikeFunction.apply(u)
        assert out.item() == 1.0

    def test_backward_returns_surrogate_not_zero(self):
        u = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        out = SpikeFunction.apply(u)
        out.sum().backward()
        # Surrogate gradient should be non-zero everywhere (1/(1+|u|)^2)
        assert u.grad is not None
        assert torch.all(u.grad > 0)

    def test_backward_surrogate_shape(self):
        u = torch.randn(10, requires_grad=True)
        SpikeFunction.apply(u).sum().backward()
        assert u.grad is not None
        assert u.grad.shape == u.shape

    def test_backward_surrogate_value(self):
        # At u=0: grad = 1/(1+0)^2 = 1.0
        u = torch.tensor([0.0], requires_grad=True)
        SpikeFunction.apply(u).sum().backward()
        assert u.grad is not None
        assert abs(u.grad.item() - 1.0) < 1e-5

    def test_backward_surrogate_decays_with_distance(self):
        # Gradient should decrease as |u| increases
        u = torch.tensor([0.0, 1.0, 2.0, 5.0], requires_grad=True)
        SpikeFunction.apply(u).sum().backward()
        assert u.grad is not None
        grads = u.grad.tolist()
        assert grads[0] > grads[1] > grads[2] > grads[3]

    def test_gradient_flows_through_weights(self):
        W = torch.randn(5, 5, requires_grad=True)
        x = torch.ones(5)
        u = W @ x
        SpikeFunction.apply(u).sum().backward()
        assert W.grad is not None
        assert W.grad.shape == W.shape


class TestSurrogateLifStep:
    def _default_kwargs(self, n=10):
        return dict(
            E_L=-65.0, E_e=0.0, E_i=-80.0,
            C_m=1.0, g_L=0.05, V_th=-50.0, V_reset=-65.0,
        )

    def test_output_shapes(self):
        n = 10
        V = torch.full((n,), -65.0)
        g_e = torch.zeros(n)
        g_i = torch.zeros(n)
        I_ext = torch.ones(n) * 5.0
        V_new, spiked = surrogate_lif_step(V, g_e, g_i, I_ext, 0.1, **self._default_kwargs(n))
        assert V_new.shape == (n,)
        assert spiked.shape == (n,)

    def test_spiked_values_are_zero_or_one(self):
        n = 20
        V = torch.randn(n) * 10 - 50.0
        g_e = torch.zeros(n)
        g_i = torch.zeros(n)
        I_ext = torch.zeros(n)
        _, spiked = surrogate_lif_step(V, g_e, g_i, I_ext, 0.1, **self._default_kwargs(n))
        unique = spiked.unique().tolist()
        assert all(v in (0.0, 1.0) for v in unique)

    def test_reset_on_spike(self):
        n = 5
        # Drive all neurons well above threshold
        V = torch.full((n,), -45.0)
        g_e = torch.zeros(n)
        g_i = torch.zeros(n)
        I_ext = torch.ones(n) * 100.0
        V_new, spiked = surrogate_lif_step(V, g_e, g_i, I_ext, 0.1, **self._default_kwargs(n))
        # Spiking neurons should be reset
        spiked_bool = spiked.bool()
        if spiked_bool.any():
            assert torch.allclose(V_new[spiked_bool], torch.tensor(-65.0))

    def test_can_spike_mask_respected(self):
        n = 5
        V = torch.full((n,), -45.0)
        g_e = torch.zeros(n)
        g_i = torch.zeros(n)
        I_ext = torch.ones(n) * 100.0
        can_spike = torch.zeros(n, dtype=torch.bool)
        _, spiked = surrogate_lif_step(
            V, g_e, g_i, I_ext, 0.1, **self._default_kwargs(n), can_spike=can_spike
        )
        assert torch.all(spiked == 0.0)

    def test_gradient_flows_to_weights(self):
        n = 5
        W = torch.randn(n, n, requires_grad=True)
        x = torch.ones(n)
        V = torch.full((n,), -55.0)
        g_e = W @ x
        g_i = torch.zeros(n)
        I_ext = torch.zeros(n)
        _, spiked = surrogate_lif_step(V, g_e, g_i, I_ext, 0.1, **self._default_kwargs(n))
        spiked.sum().backward()
        assert W.grad is not None

    def test_matches_lif_step_forward_values(self):
        """surrogate_lif_step forward pass should match lif_step outputs."""
        from pinglab.backends.pytorch.simulate_network import lif_step

        n = 50
        torch.manual_seed(0)
        V = torch.randn(n) * 5 - 55.0
        g_e = torch.rand(n) * 0.5
        g_i = torch.rand(n) * 0.2
        I_ext = torch.randn(n) * 2.0
        kwargs = self._default_kwargs(n)

        V_lif, spiked_lif = lif_step(V, g_e, g_i, I_ext, 0.1, **kwargs)
        V_sur, spiked_sur = surrogate_lif_step(V, g_e, g_i, I_ext, 0.1, **kwargs)

        assert torch.allclose(V_lif, V_sur, atol=1e-5), "V_new mismatch"
        assert torch.allclose(spiked_lif.float(), spiked_sur, atol=1e-5), "spiked mismatch"
