"""Tests for apply_heterogeneity function."""

import numpy as np
import pytest
from pinglab.run.apply_heterogeneity import apply_heterogeneity


class TestApplyHeterogeneity:
    """Tests for heterogeneous parameter generation."""

    # Default base parameters
    N_E = 80
    N_I = 20
    g_L_E = 0.05
    g_L_I = 0.1
    C_m_E = 1.0
    C_m_I = 0.5
    V_th = -50.0
    t_ref_E = 2.0
    t_ref_I = 1.0

    def test_zero_sd_returns_base_values(self):
        """With sd=0 for all parameters, output should match base values."""
        rng = np.random.RandomState(42)

        V_th_arr, g_L_arr, C_m_arr, t_ref_arr = apply_heterogeneity(
            rng, self.N_E, self.N_I,
            V_th_heterogeneity_sd=0.0,
            g_L_heterogeneity_sd=0.0,
            C_m_heterogeneity_sd=0.0,
            t_ref_heterogeneity_sd=0.0,
            g_L_E=self.g_L_E, g_L_I=self.g_L_I,
            C_m_E=self.C_m_E, C_m_I=self.C_m_I,
            V_th=self.V_th,
            t_ref_E=self.t_ref_E, t_ref_I=self.t_ref_I,
        )

        # All neurons should have identical values
        assert np.all(V_th_arr == self.V_th)
        assert np.all(g_L_arr[:self.N_E] == self.g_L_E)
        assert np.all(g_L_arr[self.N_E:] == self.g_L_I)
        assert np.all(C_m_arr[:self.N_E] == self.C_m_E)
        assert np.all(C_m_arr[self.N_E:] == self.C_m_I)
        assert np.all(t_ref_arr[:self.N_E] == self.t_ref_E)
        assert np.all(t_ref_arr[self.N_E:] == self.t_ref_I)

    def test_correct_output_shapes(self):
        """Output arrays should have shape (N_E + N_I,)."""
        rng = np.random.RandomState(42)
        N = self.N_E + self.N_I

        V_th_arr, g_L_arr, C_m_arr, t_ref_arr = apply_heterogeneity(
            rng, self.N_E, self.N_I,
            V_th_heterogeneity_sd=1.0,
            g_L_heterogeneity_sd=0.1,
            C_m_heterogeneity_sd=0.1,
            t_ref_heterogeneity_sd=0.5,
            g_L_E=self.g_L_E, g_L_I=self.g_L_I,
            C_m_E=self.C_m_E, C_m_I=self.C_m_I,
            V_th=self.V_th,
            t_ref_E=self.t_ref_E, t_ref_I=self.t_ref_I,
        )

        assert V_th_arr.shape == (N,)
        assert g_L_arr.shape == (N,)
        assert C_m_arr.shape == (N,)
        assert t_ref_arr.shape == (N,)

    def test_deterministic_with_same_seed(self):
        """Same seed should produce identical results."""
        results = []
        for _ in range(2):
            rng = np.random.RandomState(123)
            result = apply_heterogeneity(
                rng, self.N_E, self.N_I,
                V_th_heterogeneity_sd=2.0,
                g_L_heterogeneity_sd=0.15,
                C_m_heterogeneity_sd=0.1,
                t_ref_heterogeneity_sd=0.3,
                g_L_E=self.g_L_E, g_L_I=self.g_L_I,
                C_m_E=self.C_m_E, C_m_I=self.C_m_I,
                V_th=self.V_th,
                t_ref_E=self.t_ref_E, t_ref_I=self.t_ref_I,
            )
            results.append(result)

        for arr1, arr2 in zip(results[0], results[1]):
            np.testing.assert_array_equal(arr1, arr2)

    def test_clipping_prevents_negative_conductance(self):
        """Large heterogeneity should not produce negative conductances."""
        rng = np.random.RandomState(42)

        # Use very large heterogeneity to test clipping
        _, g_L_arr, C_m_arr, t_ref_arr = apply_heterogeneity(
            rng, self.N_E, self.N_I,
            V_th_heterogeneity_sd=0.0,
            g_L_heterogeneity_sd=2.0,  # 200% variation - will hit negatives without clipping
            C_m_heterogeneity_sd=2.0,
            t_ref_heterogeneity_sd=5.0,  # Large variation
            g_L_E=self.g_L_E, g_L_I=self.g_L_I,
            C_m_E=self.C_m_E, C_m_I=self.C_m_I,
            V_th=self.V_th,
            t_ref_E=self.t_ref_E, t_ref_I=self.t_ref_I,
        )

        # All values should be positive (clipped)
        assert np.all(g_L_arr >= 0.01), "g_L should be clipped to minimum 0.01"
        assert np.all(C_m_arr >= 0.01), "C_m should be clipped to minimum 0.01"
        assert np.all(t_ref_arr >= 0.1), "t_ref should be clipped to minimum 0.1"

    def test_heterogeneity_produces_variation(self):
        """Non-zero heterogeneity should produce variation in parameters."""
        rng = np.random.RandomState(42)

        V_th_arr, g_L_arr, C_m_arr, t_ref_arr = apply_heterogeneity(
            rng, self.N_E, self.N_I,
            V_th_heterogeneity_sd=2.0,
            g_L_heterogeneity_sd=0.15,
            C_m_heterogeneity_sd=0.1,
            t_ref_heterogeneity_sd=0.3,
            g_L_E=self.g_L_E, g_L_I=self.g_L_I,
            C_m_E=self.C_m_E, C_m_I=self.C_m_I,
            V_th=self.V_th,
            t_ref_E=self.t_ref_E, t_ref_I=self.t_ref_I,
        )

        # Check that there is variation (std > 0)
        assert np.std(V_th_arr) > 0, "V_th should have variation"
        assert np.std(g_L_arr[:self.N_E]) > 0, "g_L_E should have variation"
        assert np.std(C_m_arr[:self.N_E]) > 0, "C_m_E should have variation"
        assert np.std(t_ref_arr[:self.N_E]) > 0, "t_ref_E should have variation"

    def test_e_i_populations_have_different_base_values(self):
        """E and I populations should have different base conductances/capacitances."""
        rng = np.random.RandomState(42)

        _, g_L_arr, C_m_arr, t_ref_arr = apply_heterogeneity(
            rng, self.N_E, self.N_I,
            V_th_heterogeneity_sd=0.0,
            g_L_heterogeneity_sd=0.0,
            C_m_heterogeneity_sd=0.0,
            t_ref_heterogeneity_sd=0.0,
            g_L_E=self.g_L_E, g_L_I=self.g_L_I,
            C_m_E=self.C_m_E, C_m_I=self.C_m_I,
            V_th=self.V_th,
            t_ref_E=self.t_ref_E, t_ref_I=self.t_ref_I,
        )

        # E and I populations should differ
        assert np.mean(g_L_arr[:self.N_E]) != np.mean(g_L_arr[self.N_E:])
        assert np.mean(C_m_arr[:self.N_E]) != np.mean(C_m_arr[self.N_E:])
        assert np.mean(t_ref_arr[:self.N_E]) != np.mean(t_ref_arr[self.N_E:])
