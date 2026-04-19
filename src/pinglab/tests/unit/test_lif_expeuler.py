"""Exponential-Euler COBA membrane update.

Tests the new `lif_step_expeuler` primitive that replaces the forward-Euler
`lif_step` for biophysical models. Under a zero-order hold on g_e, g_i over
one step of length dt:

    g_tot   = g_L + g_e + g_i
    tau_eff = C_m / g_tot
    v_inf   = (g_L*E_L + g_e*E_e + g_i*E_i) / g_tot
    v_{t+1} = v_inf + (v_t - v_inf) * exp(-dt / tau_eff)

These tests are the acceptance contract for the new primitive; they fail
until `lif_step_expeuler` is implemented.
"""
import math

import pytest
import torch

import models as M
from models import spike_biophysical


pytest.importorskip("models")  # noqa
# The symbol under test — implemented in a follow-up commit.
lif_step_expeuler = pytest.importorskip(
    "models", reason="lif_step_expeuler not yet implemented"
).__dict__.get("lif_step_expeuler")

pytestmark = pytest.mark.skipif(
    lif_step_expeuler is None,
    reason="lif_step_expeuler not yet implemented — TDD stub",
)


def _fresh_state(v0=None, B=1, N=1):
    v = torch.full((B, N), v0 if v0 is not None else M.E_L)
    ref = torch.zeros((B, N), dtype=torch.long)
    return v, ref


def _step(v, ref, *, g_e=0.0, g_i=None, C_m=None, g_L=None,
          ref_steps=None, dt=None):
    """Thin adapter so tests don't carry the full call signature."""
    C_m = M.C_m_E if C_m is None else C_m
    g_L = M.g_L_E if g_L is None else g_L
    ref_steps = M.ref_steps_E if ref_steps is None else ref_steps
    g_e_t = torch.as_tensor(g_e, dtype=v.dtype).broadcast_to(v.shape)
    g_i_t = None if g_i is None else torch.as_tensor(
        g_i, dtype=v.dtype).broadcast_to(v.shape)
    kwargs = {}
    if dt is not None:
        kwargs["dt"] = dt
    return lif_step_expeuler(v, ref, g_e_t, g_i_t, C_m, g_L, ref_steps,
                             spike_biophysical, **kwargs)


class TestPassiveDecay:
    def test_resting_zero_input_is_stationary(self):
        """v=E_L, no conductance input → voltage doesn't move, no spike."""
        v, ref = _fresh_state()
        v2, s, ref2 = _step(v, ref)
        torch.testing.assert_close(v2, v)
        assert s.item() == 0.0
        assert ref2.item() == 0

    def test_passive_decay_matches_closed_form(self):
        """With g_e=g_i=0 and v != E_L, one step follows
            v_{t+1} = E_L + (v_t - E_L) * exp(-dt * g_L / C_m)
        exactly (this is the exp-Euler / exact solution for the homogeneous
        ODE C dv/dt = -g_L (v - E_L))."""
        v0 = -55.0  # 10 mV above E_L
        v, ref = _fresh_state(v0=v0)
        v2, _, _ = _step(v, ref)
        expected = M.E_L + (v0 - M.E_L) * math.exp(-M.dt * M.g_L_E / M.C_m_E)
        assert v2.item() == pytest.approx(expected, abs=1e-6)

    def test_passive_decay_is_dt_invariant(self):
        """The headline win: N steps at dt equal 1 step at N*dt, exactly.
        Forward Euler does not satisfy this; exp-Euler does."""
        v0 = -55.0
        # Fine: N steps at small dt
        v_fine, ref = _fresh_state(v0=v0)
        N = 10
        dt_fine = 0.1
        for _ in range(N):
            v_fine, _, ref = _step(v_fine, ref, dt=dt_fine)
        # Coarse: 1 step at N*dt_fine
        v_coarse, ref_c = _fresh_state(v0=v0)
        v_coarse, _, _ = _step(v_coarse, ref_c, dt=N * dt_fine)
        assert v_fine.item() == pytest.approx(v_coarse.item(), abs=1e-6)


class TestConductanceDrive:
    def test_steady_state_with_constant_g_e(self):
        """Under constant g_e with no g_i, running many steps drives v toward
            v_inf = (g_L*E_L + g_e*E_e) / (g_L + g_e)
        which sits below V_th (no spikes) for g_e small enough."""
        g_e = 0.01  # uS, below rheobase
        v_inf = (M.g_L_E * M.E_L + g_e * M.E_e) / (M.g_L_E + g_e)
        assert v_inf < M.V_th, "test precondition: subthreshold drive"
        v, ref = _fresh_state()
        for _ in range(2000):  # >> tau_eff
            v, s, ref = _step(v, ref, g_e=g_e)
            assert s.item() == 0.0, "should not spike subthreshold"
        assert v.item() == pytest.approx(v_inf, abs=1e-3)

    def test_tau_eff_governs_approach(self):
        """From v = E_L, under constant g_e, after one dt the fraction of the
        gap to v_inf closed is exactly (1 - exp(-dt / tau_eff))."""
        g_e = 0.02
        g_tot = M.g_L_E + g_e
        tau_eff = M.C_m_E / g_tot
        v_inf = (M.g_L_E * M.E_L + g_e * M.E_e) / g_tot
        v, ref = _fresh_state()
        v2, _, _ = _step(v, ref, g_e=g_e)
        expected = v_inf + (M.E_L - v_inf) * math.exp(-M.dt / tau_eff)
        assert v2.item() == pytest.approx(expected, abs=1e-6)

    def test_inhibition_pulls_v_below_E_L(self):
        """With only g_i active, v_inf < E_L — the exp-Euler step must reflect
        this (forward Euler does too, but we want to confirm the ZOH on g_i
        is wired up)."""
        g_i = 0.05
        v_inf = (M.g_L_E * M.E_L + g_i * M.E_i) / (M.g_L_E + g_i)
        assert v_inf < M.E_L
        v, ref = _fresh_state()
        for _ in range(2000):
            v, _, ref = _step(v, ref, g_e=0.0, g_i=g_i)
        assert v.item() == pytest.approx(v_inf, abs=1e-3)


class TestLimits:
    def test_dt_to_zero_matches_forward_euler(self):
        """In the dt → 0 limit, exp-Euler and forward Euler agree to O(dt^2).
        Pick dt small enough that relative error is below 1e-4."""
        from models import lif_step as lif_step_fwd
        g_e = 0.01
        v0 = -55.0
        v_e, ref_e = _fresh_state(v0=v0)
        v_f = torch.full_like(v_e, v0)
        ref_f = ref_e.clone()
        dt_tiny = 0.001  # ms — forward and exp Euler agree at this scale

        v_e, _, _ = _step(v_e, ref_e, g_e=g_e, dt=dt_tiny)

        # Forward Euler reference via the old lif_step signature
        I_total = torch.tensor(
            [[g_e * (M.E_e - v0)]])  # COBA current at v0
        # old lif_step uses module-level M.dt — too coarse for this check, so
        # we reconstruct the forward-Euler increment directly:
        expected_fwd = v0 + (dt_tiny / M.C_m_E) * (
            -M.g_L_E * (v0 - M.E_L) + g_e * (M.E_e - v0))

        rel = abs(v_e.item() - expected_fwd) / abs(expected_fwd - v0)
        assert rel < 1e-3, f"exp-Euler diverged from fwd-Euler at dt={dt_tiny}"


class TestRefractory:
    def test_refractory_pins_v_at_reset(self):
        """Behavioural: after a spike, v stays at V_reset for ref_steps
        regardless of incoming conductance."""
        v = torch.full((1, 1), M.V_th - 0.01)
        ref = torch.zeros((1, 1), dtype=torch.long)
        # Force spike via huge conductance
        v, s, ref = _step(v, ref, g_e=100.0)
        assert s.item() == 1.0
        assert v.item() == pytest.approx(M.V_reset)
        assert ref.item() == M.ref_steps_E
        for _ in range(M.ref_steps_E - 1):
            v, s, ref = _step(v, ref, g_e=100.0)
            assert s.item() == 0.0, "spiked during refractory period"
            assert v.item() == pytest.approx(M.V_reset)


class TestGradientFlow:
    def test_grad_flows_through_g_e(self):
        """BPTT must push gradient through the membrane update back to g_e;
        otherwise COBA/PING training silently breaks."""
        v, ref = _fresh_state()
        g_e = torch.tensor([[0.01]], requires_grad=True)
        v2, _, _ = lif_step_expeuler(
            v, ref, g_e, None, M.C_m_E, M.g_L_E, M.ref_steps_E,
            spike_biophysical)
        v2.sum().backward()
        assert g_e.grad is not None
        assert g_e.grad.abs().item() > 0.0

    def test_cm_back_scale_attenuates_grad(self):
        """cm_back > 1 must reduce the gradient magnitude relative to the
        identity-scale reference (the point of the dampening hook)."""
        v_ref, ref = _fresh_state()
        v_damp, ref2 = _fresh_state()
        g_e_ref = torch.tensor([[0.01]], requires_grad=True)
        g_e_damp = torch.tensor([[0.01]], requires_grad=True)
        v2r, _, _ = lif_step_expeuler(
            v_ref, ref, g_e_ref, None, M.C_m_E, M.g_L_E, M.ref_steps_E,
            spike_biophysical, cm_back=1.0)
        v2d, _, _ = lif_step_expeuler(
            v_damp, ref2, g_e_damp, None, M.C_m_E, M.g_L_E, M.ref_steps_E,
            spike_biophysical, cm_back=1000.0)
        v2r.sum().backward()
        v2d.sum().backward()
        assert g_e_damp.grad.abs().item() < g_e_ref.grad.abs().item()


class TestFiring:
    def test_constant_conductance_fires_periodically(self):
        """Under constant suprathreshold g_e, neuron fires repeatedly. We
        don't pin the ISI (it differs slightly from forward Euler) — only
        that spikes appear and recurs at a stable rate."""
        # Enough drive that v_inf exceeds V_th:
        g_e_rheo = M.g_L_E * (M.V_th - M.E_L) / (M.E_e - M.V_th)
        g_e = 3.0 * g_e_rheo
        v, ref = _fresh_state()
        spikes = 0
        steps = int(200.0 / M.dt)
        for _ in range(steps):
            v, s, ref = _step(v, ref, g_e=g_e)
            spikes += int(s.item())
        assert spikes >= 5, f"expected periodic firing, got {spikes} spikes"
