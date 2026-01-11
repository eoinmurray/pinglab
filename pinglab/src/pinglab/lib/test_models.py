"""Tests for non-LIF neuron model steps."""

import numpy as np
import pytest

from pinglab.lib import (
    adex_step,
    cs_step,
    cs_init_gating,
    fhn_step,
    hh_step,
    hh_init_gating,
    izh_step,
    izh_init_u,
    mqif_step,
)


class TestHH:
    def test_hh_init_gating_bounds(self):
        V = np.array([-65.0, -50.0, 0.0])
        m, h, n = hh_init_gating(V)
        assert np.all((m >= 0.0) & (m <= 1.0))
        assert np.all((h >= 0.0) & (h <= 1.0))
        assert np.all((n >= 0.0) & (n <= 1.0))

    def test_hh_step_increases_with_input(self):
        V = np.array([-65.0])
        m, h, n = hh_init_gating(V)
        zeros = np.zeros_like(V)
        V_low, _, _, _ = hh_step(
            V,
            m,
            h,
            n,
            zeros,
            zeros,
            np.array([0.0]),
            0.05,
            C_m=1.0,
            g_L=0.1,
            g_Na=120.0,
            g_K=36.0,
            E_L=-65.0,
            E_Na=50.0,
            E_K=-77.0,
            E_e=0.0,
            E_i=-80.0,
        )
        V_high, _, _, _ = hh_step(
            V,
            m,
            h,
            n,
            zeros,
            zeros,
            np.array([10.0]),
            0.05,
            C_m=1.0,
            g_L=0.1,
            g_Na=120.0,
            g_K=36.0,
            E_L=-65.0,
            E_Na=50.0,
            E_K=-77.0,
            E_e=0.0,
            E_i=-80.0,
        )
        assert V_high[0] > V_low[0]


class TestAdEx:
    def test_adex_spike_resets_and_adapts(self):
        V = np.array([19.0])
        w = np.array([0.0])
        zeros = np.zeros_like(V)
        V_new, w_new, spiked = adex_step(
            V,
            w,
            zeros,
            zeros,
            np.array([500.0]),
            0.1,
            C_m=1.0,
            g_L=0.1,
            E_L=-65.0,
            E_e=0.0,
            E_i=-80.0,
            V_T=-50.0,
            Delta_T=2.0,
            tau_w=100.0,
            a=2.0,
            b=60.0,
            V_reset=-65.0,
            V_peak=20.0,
        )
        assert spiked[0]
        assert V_new[0] == -65.0
        assert w_new[0] > w[0]

    def test_adex_refractory_mask(self):
        V = np.array([10.0, 10.0])
        w = np.array([0.0, 0.0])
        zeros = np.zeros_like(V)
        can_spike = np.array([True, False])
        V_new, _, spiked = adex_step(
            V,
            w,
            zeros,
            zeros,
            np.array([500.0, 500.0]),
            0.1,
            C_m=1.0,
            g_L=0.1,
            E_L=-65.0,
            E_e=0.0,
            E_i=-80.0,
            V_T=-50.0,
            Delta_T=2.0,
            tau_w=100.0,
            a=2.0,
            b=60.0,
            V_reset=-65.0,
            V_peak=20.0,
            can_spike=can_spike,
        )
        assert spiked[0]
        assert not spiked[1]
        assert V_new[1] == -65.0


class TestFitzHugh:
    def test_fhn_step_updates_state(self):
        V = np.array([-1.0, 0.5])
        W = np.array([0.0, 0.1])
        zeros = np.zeros_like(V)
        V_new, W_new = fhn_step(
            V,
            W,
            zeros,
            zeros,
            np.array([0.5, 0.5]),
            0.01,
            a=0.7,
            b=0.8,
            tau_w=12.5,
            E_e=0.0,
            E_i=-80.0,
        )
        assert V_new.shape == V.shape
        assert W_new.shape == W.shape
        assert np.all(np.isfinite(V_new))
        assert np.all(np.isfinite(W_new))


class TestMQIF:
    def test_mqif_mismatched_terms(self):
        V = np.array([-60.0])
        zeros = np.zeros_like(V)
        with pytest.raises(ValueError, match="same length"):
            mqif_step(
                V,
                zeros,
                zeros,
                np.array([1.0]),
                0.1,
                C_m=1.0,
                g_L=0.1,
                E_L=-65.0,
                E_e=0.0,
                E_i=-80.0,
                a_terms=np.array([1.0, 2.0]),
                V_r_terms=np.array([-50.0]),
                V_th=-40.0,
                V_reset=-65.0,
            )

    def test_mqif_spike_resets(self):
        V = np.array([-39.0])
        zeros = np.zeros_like(V)
        V_new, spiked = mqif_step(
            V,
            zeros,
            zeros,
            np.array([50.0]),
            0.1,
            C_m=1.0,
            g_L=0.1,
            E_L=-65.0,
            E_e=0.0,
            E_i=-80.0,
            a_terms=np.array([1.0]),
            V_r_terms=np.array([-50.0]),
            V_th=-40.0,
            V_reset=-65.0,
        )
        assert spiked[0]
        assert V_new[0] == -65.0


class TestIzhikevich:
    def test_izh_init_u(self):
        V = np.array([-65.0, -50.0])
        U = izh_init_u(V, 0.2)
        np.testing.assert_allclose(U, 0.2 * V)

    def test_izh_spike_resets(self):
        V = np.array([29.0])
        U = np.array([0.0])
        zeros = np.zeros_like(V)
        V_new, U_new, spiked = izh_step(
            V,
            U,
            zeros,
            zeros,
            np.array([20.0]),
            0.1,
            a=0.02,
            b=0.2,
            c=-65.0,
            d=8.0,
            V_th=30.0,
            E_e=0.0,
            E_i=-80.0,
        )
        assert spiked[0]
        assert V_new[0] == -65.0
        assert U_new[0] > U[0]


class TestConnorStevens:
    def test_cs_init_gating_bounds(self):
        V = np.array([-65.0, -40.0])
        m, h, n, a, b = cs_init_gating(V)
        for gate in (m, h, n, a, b):
            assert np.all((gate >= 0.0) & (gate <= 1.0))

    def test_cs_step_increases_with_input(self):
        V = np.array([-65.0])
        m, h, n, a, b = cs_init_gating(V)
        zeros = np.zeros_like(V)
        V_low, *_ = cs_step(
            V,
            m,
            h,
            n,
            a,
            b,
            zeros,
            zeros,
            np.array([0.0]),
            0.05,
            C_m=1.0,
            g_L=0.1,
            g_Na=120.0,
            g_K=36.0,
            g_A=47.7,
            E_L=-65.0,
            E_Na=50.0,
            E_K=-77.0,
            E_e=0.0,
            E_i=-80.0,
        )
        V_high, *_ = cs_step(
            V,
            m,
            h,
            n,
            a,
            b,
            zeros,
            zeros,
            np.array([10.0]),
            0.05,
            C_m=1.0,
            g_L=0.1,
            g_Na=120.0,
            g_K=36.0,
            g_A=47.7,
            E_L=-65.0,
            E_Na=50.0,
            E_K=-77.0,
            E_e=0.0,
            E_i=-80.0,
        )
        assert V_high[0] > V_low[0]
