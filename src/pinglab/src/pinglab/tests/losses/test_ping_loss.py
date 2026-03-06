"""Tests for the differentiable PING health loss."""
import torch
import pytest

from pinglab.losses.ping_loss import ping_health_loss


def _make_periodic_spikes(B: int, T: int, N: int, period_steps: int) -> torch.Tensor:
    """Create spikes with a regular periodic pattern."""
    spikes = torch.zeros(B, T, N)
    for t in range(0, T, period_steps):
        spikes[:, t, :] = 1.0
    return spikes


def _make_noise_spikes(B: int, T: int, N: int, rate: float = 0.1) -> torch.Tensor:
    """Create random Bernoulli spikes."""
    return (torch.rand(B, T, N) < rate).float()


class TestPingHealthLoss:
    def test_output_is_scalar(self):
        spikes = _make_noise_spikes(4, 200, 32)
        loss = ping_health_loss(spikes, dt_ms=1.0)
        assert loss.shape == ()

    def test_output_requires_grad(self):
        spikes = _make_noise_spikes(4, 200, 32)
        spikes.requires_grad_(True)
        loss = ping_health_loss(spikes, dt_ms=1.0)
        assert loss.requires_grad

    def test_gradients_flow(self):
        spikes = _make_noise_spikes(4, 200, 32)
        spikes.requires_grad_(True)
        loss = ping_health_loss(spikes, dt_ms=1.0)
        loss.backward()
        assert spikes.grad is not None
        assert not torch.all(spikes.grad == 0)

    def test_periodic_lower_than_noise(self):
        B, N = 4, 64
        dt_ms = 0.1
        T = 2000  # 200 ms at dt=0.1

        # Periodic at ~30 Hz → period = 33.3 ms → 333 steps at dt=0.1
        period_steps = int(1000.0 / 30.0 / dt_ms)
        periodic = _make_periodic_spikes(B, T, N, period_steps)
        noise = _make_noise_spikes(B, T, N, rate=0.05)

        loss_periodic = ping_health_loss(periodic, dt_ms=dt_ms, f_target=30.0)
        loss_noise = ping_health_loss(noise, dt_ms=dt_ms, f_target=30.0)

        assert loss_periodic.item() < loss_noise.item()

    def test_batch_shapes(self):
        for B in [1, 8, 16]:
            spikes = _make_noise_spikes(B, 200, 32)
            loss = ping_health_loss(spikes, dt_ms=1.0)
            assert loss.shape == ()
            assert torch.isfinite(loss)

    def test_dt_01_binning(self):
        """With dt=0.1 and T=200ms → 2000 steps, bin_ms=5 → 400 bins."""
        B, N = 2, 32
        T = 2000
        spikes = _make_noise_spikes(B, T, N)
        loss = ping_health_loss(spikes, dt_ms=0.1, bin_ms=5.0)
        assert loss.shape == ()
        assert torch.isfinite(loss)
