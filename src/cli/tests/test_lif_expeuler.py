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

import models as M
import pytest
import torch
from models import coba_current, spike_biophysical

pytest.importorskip("models")  # noqa
# The symbol under test — implemented in a follow-up commit.
lif_step_expeuler = pytest.importorskip(
    "models", reason="lif_step_expeuler not yet implemented"
).__dict__.get("lif_step_expeuler")

pytestmark = pytest.mark.skipif(
    lif_step_expeuler is None,
    reason="lif_step_expeuler not yet implemented — TDD stub",
)


def _fresh_state(v0=None, B=1, N=1, dtype=torch.float32):
    v = torch.full((B, N), v0 if v0 is not None else M.E_L, dtype=dtype)
    ref = torch.zeros((B, N), dtype=torch.long)
    return v, ref


def _step(v, ref, *, g_e=0.0, g_i=None, C_m=None, g_L=None, ref_steps=None, dt=None):
    """Thin adapter so tests don't carry the full call signature."""
    C_m = M.C_m_E if C_m is None else C_m
    g_L = M.g_L_E if g_L is None else g_L
    ref_steps = M.ref_steps_E if ref_steps is None else ref_steps
    g_e_t = torch.as_tensor(g_e, dtype=v.dtype).broadcast_to(v.shape)
    g_i_t = (
        None
        if g_i is None
        else torch.as_tensor(g_i, dtype=v.dtype).broadcast_to(v.shape)
    )
    kwargs = {}
    if dt is not None:
        kwargs["dt_override"] = dt
    return lif_step_expeuler(
        v, ref, g_e_t, g_i_t, C_m, g_L, ref_steps, spike_biophysical, **kwargs
    )


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
        # Fine: N steps at small dt (float64 to see exact-integrator equality).
        v_fine, ref = _fresh_state(v0=v0, dtype=torch.float64)
        N = 10
        dt_fine = 0.1
        for _ in range(N):
            v_fine, _, ref = _step(v_fine, ref, dt=dt_fine)
        v_coarse, ref_c = _fresh_state(v0=v0, dtype=torch.float64)
        v_coarse, _, _ = _step(v_coarse, ref_c, dt=N * dt_fine)
        assert v_fine.item() == pytest.approx(v_coarse.item(), abs=1e-10)


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
        v, ref = _fresh_state(dtype=torch.float64)
        v2, _, _ = _step(v, ref, g_e=g_e)
        expected = v_inf + (M.E_L - v_inf) * math.exp(-M.dt / tau_eff)
        assert v2.item() == pytest.approx(expected, abs=1e-10)

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
        The *dv* predicted by each integrator has relative error
        ≈ (dt/tau_eff)/2; pick dt small enough that this is tiny."""
        g_e = 0.01
        v0 = -55.0
        dt_tiny = 0.001  # ms
        v_e, ref_e = _fresh_state(v0=v0, dtype=torch.float64)
        v_e, _, _ = _step(v_e, ref_e, g_e=g_e, dt=dt_tiny)

        # Forward-Euler reference reconstructed directly (old lif_step is
        # hard-wired to module-level M.dt):
        expected_fwd = v0 + (dt_tiny / M.C_m_E) * (
            -M.g_L_E * (v0 - M.E_L) + g_e * (M.E_e - v0)
        )

        dv_exp = v_e.item() - v0
        dv_fwd = expected_fwd - v0
        rel = abs(dv_exp - dv_fwd) / abs(dv_fwd)
        # Leading error is (dt/tau_eff)/2 ≈ 3e-5 at dt=0.001, tau_eff≈17 ms
        assert rel < 1e-4, f"exp-Euler diverged from fwd-Euler at dt={dt_tiny}"


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
            v, ref, g_e, None, M.C_m_E, M.g_L_E, M.ref_steps_E, spike_biophysical
        )
        v2.sum().backward()
        assert g_e.grad is not None
        assert g_e.grad.abs().item() > 0.0

    def test_v_grad_dampen_attenuates_grad(self):
        """v_grad_dampen > 1 must reduce the gradient magnitude relative to the
        identity-scale reference (the point of the dampening hook)."""
        v_ref, ref = _fresh_state()
        v_damp, ref2 = _fresh_state()
        g_e_ref = torch.tensor([[0.01]], requires_grad=True)
        g_e_damp = torch.tensor([[0.01]], requires_grad=True)
        v2r, _, _ = lif_step_expeuler(
            v_ref,
            ref,
            g_e_ref,
            None,
            M.C_m_E,
            M.g_L_E,
            M.ref_steps_E,
            spike_biophysical,
            v_grad_dampen=1.0,
        )
        v2d, _, _ = lif_step_expeuler(
            v_damp,
            ref2,
            g_e_damp,
            None,
            M.C_m_E,
            M.g_L_E,
            M.ref_steps_E,
            spike_biophysical,
            v_grad_dampen=1000.0,
        )
        v2r.sum().backward()
        v2d.sum().backward()
        assert g_e_damp.grad is not None and g_e_ref.grad is not None
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


class TestCombinedDrive:
    def test_v_inf_with_both_conductances_closed_form(self):
        """One step with BOTH g_e and g_i active must hit the three-term
        closed form: v_inf = (g_L E_L + g_e E_e + g_i E_i) / (g_L + g_e + g_i)."""
        g_e, g_i = 0.03, 0.02
        g_tot = M.g_L_E + g_e + g_i
        v_inf = (M.g_L_E * M.E_L + g_e * M.E_e + g_i * M.E_i) / g_tot
        tau_eff = M.C_m_E / g_tot
        v, ref = _fresh_state(dtype=torch.float64)
        v2, _, _ = _step(v, ref, g_e=g_e, g_i=g_i)
        expected = v_inf + (M.E_L - v_inf) * math.exp(-M.dt / tau_eff)
        assert v2.item() == pytest.approx(expected, abs=1e-10)

    def test_v_inf_sits_between_reversal_potentials(self):
        """With mixed E/I drive, v_inf is a conductance-weighted average of the
        three reversal potentials, so it lies within [E_i, E_e]."""
        g_e, g_i = 0.04, 0.06
        v, ref = _fresh_state(v0=-60.0, dtype=torch.float64)
        for _ in range(4000):  # >> tau_eff → converged
            v, s, ref = _step(v, ref, g_e=g_e, g_i=g_i)
            assert s.item() == 0.0
        g_tot = M.g_L_E + g_e + g_i
        v_inf = (M.g_L_E * M.E_L + g_e * M.E_e + g_i * M.E_i) / g_tot
        assert M.E_i <= v_inf <= M.E_e
        assert v.item() == pytest.approx(v_inf, abs=1e-3)


class TestZOHConductanceInvariance:
    def test_dt_invariant_under_held_conductance(self):
        """The ZOH exp-Euler property WITH drive: holding g_e constant, N steps
        at dt equal 1 step at N*dt exactly (constant-coefficient linear ODE has
        an exact solution that composes). Forward Euler lacks this."""
        g_e = 0.01  # subthreshold: v_inf ≈ -54 mV < V_th, so no reset intervenes
        v_fine, ref = _fresh_state(dtype=torch.float64)
        N, dt_fine = 8, 0.05
        for _ in range(N):
            v_fine, s, ref = _step(v_fine, ref, g_e=g_e, dt=dt_fine)
            assert s.item() == 0.0
        v_coarse, ref_c = _fresh_state(dtype=torch.float64)
        v_coarse, _, _ = _step(v_coarse, ref_c, g_e=g_e, dt=N * dt_fine)
        assert v_fine.item() == pytest.approx(v_coarse.item(), abs=1e-10)


class TestBatched:
    def test_batched_neurons_follow_independent_closed_form(self):
        """The vectorised step integrates each (batch, neuron) under its own
        g_e — no cross-talk. Checks the closed form per column."""
        g_e_list = [0.005, 0.01, 0.02]
        g_e = torch.tensor([g_e_list], dtype=torch.float64)
        v = torch.full((1, 3), M.E_L, dtype=torch.float64)
        ref = torch.zeros((1, 3), dtype=torch.long)
        v2, s, _ = lif_step_expeuler(
            v, ref, g_e, None, M.C_m_E, M.g_L_E, M.ref_steps_E, spike_biophysical
        )
        for j, ge in enumerate(g_e_list):
            g_tot = M.g_L_E + ge
            v_inf = (M.g_L_E * M.E_L + ge * M.E_e) / g_tot
            expected = v_inf + (M.E_L - v_inf) * math.exp(-M.dt / (M.C_m_E / g_tot))
            assert v2[0, j].item() == pytest.approx(expected, abs=1e-10)
        assert s.sum().item() == 0.0  # one step from rest stays subthreshold


class TestThresholdOffset:
    def test_positive_offset_suppresses_spike(self):
        """threshold_offset raises the effective threshold (v - V_th - offset):
        a voltage that spikes at offset 0 must NOT spike under a large offset."""
        v0 = M.V_th + 1.0  # just above threshold
        v = torch.full((1, 1), v0)
        ref = torch.zeros((1, 1), dtype=torch.long)
        off0 = torch.zeros_like(v)
        _, s_on, _ = lif_step_expeuler(
            v.clone(), ref.clone(), torch.zeros_like(v), None,
            M.C_m_E, M.g_L_E, M.ref_steps_E, spike_biophysical,
            threshold_offset=off0,
        )
        assert s_on.item() == 1.0
        off_hi = torch.full_like(v, 5.0)  # raise threshold by 5 mV
        _, s_off, _ = lif_step_expeuler(
            v.clone(), ref.clone(), torch.zeros_like(v), None,
            M.C_m_E, M.g_L_E, M.ref_steps_E, spike_biophysical,
            threshold_offset=off_hi,
        )
        assert s_off.item() == 0.0


class TestVoltageClamps:
    def test_v_max_caps_the_membrane(self):
        """V_max hard-caps the post-update voltage. With strong depolarising
        drive but V_max below threshold, v pins at V_max and never spikes."""
        V_max = M.V_th - 5.0  # -55 mV, below threshold
        v, ref = _fresh_state()
        g_e = torch.full_like(v, 50.0)  # v_inf ≈ E_e = 0, drives hard up
        for _ in range(50):
            v, s, ref = lif_step_expeuler(
                v, ref, g_e, None, M.C_m_E, M.g_L_E, M.ref_steps_E,
                spike_biophysical, V_max=V_max,
            )
            assert s.item() == 0.0
            assert v.item() <= V_max + 1e-6
        assert v.item() == pytest.approx(V_max, abs=1e-4)  # pinned at the cap


class TestMembraneNoise:
    def _noiseless_passive_step(self, v0):
        return M.E_L + (v0 - M.E_L) * math.exp(-M.dt * M.g_L_E / M.C_m_E)

    def test_zero_noise_is_deterministic_closed_form(self):
        """v_noise_std=0 leaves the deterministic exp-Euler step untouched."""
        v0 = -55.0
        v, ref = _fresh_state(v0=v0, dtype=torch.float64)
        v2, _, _ = lif_step_expeuler(
            v, ref, torch.zeros_like(v), None, M.C_m_E, M.g_L_E, M.ref_steps_E,
            spike_biophysical, v_noise_std=0.0,
        )
        assert v2.item() == pytest.approx(self._noiseless_passive_step(v0), abs=1e-9)

    def test_noise_is_reproducible_under_seed(self):
        """Same seed → identical membrane noise (the Wiener increment is drawn
        from the global torch RNG)."""
        def once():
            torch.manual_seed(1234)
            v, ref = _fresh_state(v0=-60.0, N=64)
            v2, _, _ = lif_step_expeuler(
                v, ref, torch.zeros_like(v), None, M.C_m_E, M.g_L_E,
                M.ref_steps_E, spike_biophysical, v_noise_std=1.0,
            )
            return v2

        torch.testing.assert_close(once(), once())

    def test_noise_perturbs_but_is_zero_mean(self):
        """Membrane noise deflects individual neurons off the deterministic
        value, but its ensemble mean is ≈ 0 (zero-mean Wiener increment)."""
        torch.manual_seed(7)
        n = 40000
        v, ref = _fresh_state(v0=-60.0, N=n, dtype=torch.float64)
        v2, _, _ = lif_step_expeuler(
            v, ref, torch.zeros_like(v), None, M.C_m_E, M.g_L_E, M.ref_steps_E,
            spike_biophysical, v_noise_std=1.0,
        )
        dev = v2 - self._noiseless_passive_step(-60.0)
        assert dev.abs().mean().item() > 0.0          # noise actually applied
        assert abs(dev.mean().item()) < 0.05          # but zero-mean

    def test_stationary_std_matches_v_noise_std_and_is_dt_invariant(self):
        """The docstring's scaling claim: the sqrt(2 dt / tau_leak) factor makes
        the stationary subthreshold std ≈ v_noise_std, roughly independent of dt."""
        target = 2.0

        def stationary_std(dt_step, n=4000, steps=1500):
            torch.manual_seed(42)
            v = torch.full((1, n), M.E_L, dtype=torch.float64)
            ref = torch.zeros((1, n), dtype=torch.long)
            g_e = torch.zeros_like(v)
            for _ in range(steps):
                v, _, ref = lif_step_expeuler(
                    v, ref, g_e, None, M.C_m_E, M.g_L_E, M.ref_steps_E,
                    spike_biophysical, dt_override=dt_step, v_noise_std=target,
                )
            return v.std().item()

        assert stationary_std(0.1) == pytest.approx(target, rel=0.1)
        assert stationary_std(0.5) == pytest.approx(target, rel=0.15)


class TestCobaCurrent:
    """The conductance-based synaptic current I = g_e(E_e - v) [+ g_i(E_i - v)],
    the driving term the exp-Euler step integrates in closed form."""

    def test_excitatory_only_closed_form(self):
        """With no inhibition, I = g_e * (E_e - v)."""
        g_e = torch.tensor([[0.03]])
        v = torch.tensor([[-60.0]])
        expected = 0.03 * (M.E_e - (-60.0))
        assert coba_current(g_e, v).item() == pytest.approx(expected, abs=1e-6)

    def test_excitatory_and_inhibitory_sum(self):
        """Both conductances present: I = g_e(E_e - v) + g_i(E_i - v)."""
        g_e, g_i = 0.03, 0.05
        v = -55.0
        expected = g_e * (M.E_e - v) + g_i * (M.E_i - v)
        got = coba_current(
            torch.tensor([[g_e]]), torch.tensor([[v]]), torch.tensor([[g_i]])
        )
        assert got.item() == pytest.approx(expected, abs=1e-6)

    def test_current_vanishes_at_reversal_potentials(self):
        """At v = E_e the excitatory drive is zero; adding v = E_i zeroes the
        inhibitory drive too, so the net current is zero."""
        g_e = torch.tensor([[0.04]])
        g_i = torch.tensor([[0.06]])
        # Excitatory term alone vanishes at v = E_e.
        assert coba_current(g_e, torch.full_like(g_e, M.E_e)).item() == pytest.approx(
            0.0, abs=1e-6
        )
        # Both terms vanish only where each reversal is met; check the inhibitory
        # term is zero at v = E_i with g_e = 0.
        i_at_ei = coba_current(torch.zeros_like(g_i), torch.full_like(g_i, M.E_i), g_i)
        assert i_at_ei.item() == pytest.approx(0.0, abs=1e-6)

    def test_signs_are_depolarising_and_hyperpolarising(self):
        """Below E_e excitation is an inward (positive, depolarising) current;
        above E_i inhibition is an outward (negative, hyperpolarising) one."""
        v = torch.tensor([[M.E_L]])  # -65, between E_i (-80) and E_e (0)
        assert coba_current(torch.tensor([[0.05]]), v).item() > 0.0
        i_inhib_only = coba_current(torch.zeros_like(v), v, torch.tensor([[0.05]]))
        assert i_inhib_only.item() < 0.0

    def test_batched_matches_per_element(self):
        """Vectorised over a (B, N) tensor with per-element g_e, g_i, v."""
        g_e = torch.tensor([[0.01, 0.02, 0.03]])
        g_i = torch.tensor([[0.00, 0.04, 0.05]])
        v = torch.tensor([[-70.0, -60.0, -50.0]])
        got = coba_current(g_e, v, g_i)
        expected = g_e * (M.E_e - v) + g_i * (M.E_i - v)
        torch.testing.assert_close(got, expected)


class TestExpEulerContract:
    """Integrator invariants not pinned above: the None-vs-zero g_i fast path,
    the hard voltage floor, dt_override's independence from the module dt, the
    full refractory arm→release cycle, and unconditional (no-overshoot) stability.
    """

    def test_g_i_none_matches_explicit_zero_g_i(self):
        """The `g_i is None` fast path must be identical to passing a zero g_i
        tensor — same v, same spike, same refractory. Guards the branch in
        lif_step_expeuler that swaps g_sum / g_E_drive when g_i is absent."""
        v0 = -58.0
        va, ra = _fresh_state(v0=v0)
        vb, rb = _fresh_state(v0=v0)
        v_none, s_none, ref_none = _step(va, ra, g_e=0.03, g_i=None)
        v_zero, s_zero, ref_zero = _step(vb, rb, g_e=0.03, g_i=0.0)
        torch.testing.assert_close(v_none, v_zero)
        assert s_none.item() == s_zero.item()
        torch.testing.assert_close(ref_none, ref_zero)

    def test_v_floor_clamps_membrane(self):
        """Strong inhibition drives v_inf ≈ E_i in one step; the hard V_floor
        caps the membrane below that. Mirror of the V_max cap test."""
        v = torch.full((1, 1), M.E_L)
        ref = torch.zeros((1, 1), dtype=torch.long)
        g_e = torch.zeros((1, 1))
        g_i = torch.full((1, 1), 100.0)  # v_inf ≈ E_i = -80, decay ≈ 0
        floor = -70.0
        v2, s, _ = lif_step_expeuler(
            v, ref, g_e, g_i, M.C_m_E, M.g_L_E, M.ref_steps_E,
            spike_biophysical, V_floor=floor,
        )
        assert v2.item() == pytest.approx(floor)  # clamped up from ≈ -80
        assert s.item() == 0.0

    def test_dt_override_ignores_module_dt(self):
        """The step integrates over dt_override, never the module-global M.dt —
        the Dynamo-safe design that keeps dt out of a graph-break. A wrong M.dt
        must not change the result (conftest restores M.dt afterwards)."""
        v0 = -55.0
        dt_use = 0.3
        M.dt = 999.0  # deliberately wrong; must be ignored
        v, ref = _fresh_state(v0=v0, dtype=torch.float64)
        v2, _, _ = _step(v, ref, dt=dt_use)  # passive (g_e=g_i=0)
        expected = M.E_L + (v0 - M.E_L) * math.exp(-dt_use * M.g_L_E / M.C_m_E)
        assert v2.item() == pytest.approx(expected, abs=1e-10)

    def test_refractory_releases_after_exactly_ref_steps(self):
        """A spike arms the refractory counter to ref_steps; the neuron is
        locked for exactly that many steps, then free to fire again."""
        v = torch.full((1, 1), M.V_th - 0.01)
        ref = torch.zeros((1, 1), dtype=torch.long)
        v, s, ref = _step(v, ref, g_e=100.0)  # force a spike
        assert s.item() == 1.0
        assert ref.item() == M.ref_steps_E
        # Sit with no drive: counter ticks down, no re-fire, until it hits 1.
        for _ in range(M.ref_steps_E - 1):
            v, s, ref = _step(v, ref, g_e=0.0)
            assert s.item() == 0.0
        assert ref.item() == 1
        # Next step clears the lock (ref→0, can_spike) and strong drive fires again.
        v, s, ref = _step(v, ref, g_e=100.0)
        assert s.item() == 1.0
        assert ref.item() == M.ref_steps_E

    def test_no_overshoot_at_large_dt(self):
        """Unconditional stability: even at dt ≫ tau_eff the membrane approaches
        v_inf monotonically from below and never overshoots it — the property
        forward Euler loses (it would oscillate/diverge at this step size)."""
        g_e = 0.01  # subthreshold: v_inf below V_th, so no spikes interrupt
        g_tot = M.g_L_E + g_e
        v_inf = (M.g_L_E * M.E_L + g_e * M.E_e) / g_tot
        assert v_inf < M.V_th
        tau_eff = M.C_m_E / g_tot
        big_dt = 50.0 * tau_eff  # far past forward Euler's stability limit
        v, ref = _fresh_state(dtype=torch.float64)  # starts at E_L, below v_inf
        prev = v.item()
        for _ in range(5):
            v, s, ref = _step(v, ref, g_e=g_e, dt=big_dt)
            val = v.item()
            assert prev <= val <= v_inf + 1e-9, "overshoot past v_inf"
            assert s.item() == 0.0
            prev = val
        assert v.item() == pytest.approx(v_inf, abs=1e-3)
