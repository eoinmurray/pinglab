"""Tests for the LIF neuron model."""

import numpy as np
import pytest
from pinglab.lib import lif_step


class TestLIFStep:
    """Tests for lif_step function."""

    # Default parameters for a single neuron
    DEFAULT_PARAMS = {
        "dt": 0.1,
        "E_L": -70.0,
        "E_e": 0.0,
        "E_i": -80.0,
        "C_m": 1.0,
        "g_L": 0.05,
        "V_th": -50.0,
        "V_reset": -70.0,
    }

    def test_subthreshold_no_spike(self):
        """Neuron below threshold should not spike."""
        V = np.array([-60.0])
        g_e = np.zeros(1)
        g_i = np.zeros(1)
        I_ext = np.zeros(1)

        V_new, spiked = lif_step(V, g_e, g_i, I_ext, **self.DEFAULT_PARAMS)

        assert not spiked[0], "Subthreshold neuron should not spike"
        assert V_new[0] < self.DEFAULT_PARAMS["V_th"], "Voltage should remain below threshold"

    def test_suprathreshold_spike_and_reset(self):
        """Neuron above threshold should spike and reset."""
        V = np.array([-49.0])  # Just above threshold
        g_e = np.zeros(1)
        g_i = np.zeros(1)
        I_ext = np.array([10.0])  # Strong input to ensure spike

        V_new, spiked = lif_step(V, g_e, g_i, I_ext, **self.DEFAULT_PARAMS)

        assert spiked[0], "Suprathreshold neuron should spike"
        assert V_new[0] == self.DEFAULT_PARAMS["V_reset"], "Spiked neuron should reset to V_reset"

    def test_refractory_mask_prevents_spike(self):
        """Neurons with can_spike=False should be held at V_reset."""
        V = np.array([-40.0, -40.0])  # Both above threshold
        g_e = np.zeros(2)
        g_i = np.zeros(2)
        I_ext = np.array([10.0, 10.0])
        can_spike = np.array([True, False])

        V_new, spiked = lif_step(V, g_e, g_i, I_ext, can_spike=can_spike, **self.DEFAULT_PARAMS)

        assert spiked[0], "First neuron (can_spike=True) should spike"
        assert not spiked[1], "Second neuron (can_spike=False) should not spike"
        assert V_new[1] == self.DEFAULT_PARAMS["V_reset"], "Refractory neuron held at V_reset"

    def test_heterogeneous_threshold(self):
        """Different thresholds per neuron should work correctly."""
        V = np.array([-48.0, -48.0])
        g_e = np.zeros(2)
        g_i = np.zeros(2)
        I_ext = np.array([5.0, 5.0])
        V_th = np.array([-50.0, -45.0])  # Different thresholds

        params = {**self.DEFAULT_PARAMS, "V_th": V_th}
        V_new, spiked = lif_step(V, g_e, g_i, I_ext, **params)

        assert spiked[0], "First neuron (low threshold) should spike"
        assert not spiked[1], "Second neuron (high threshold) should not spike"

    def test_leak_toward_equilibrium(self):
        """With no input, voltage should decay toward E_L."""
        V = np.array([-60.0])  # Above E_L
        g_e = np.zeros(1)
        g_i = np.zeros(1)
        I_ext = np.zeros(1)

        # Run many steps to allow convergence (tau = C_m/g_L = 1/0.05 = 20ms, dt=0.1ms)
        for _ in range(1000):
            V, _ = lif_step(V, g_e, g_i, I_ext, **self.DEFAULT_PARAMS)

        # Should approach E_L (within 0.1mV after ~5 tau)
        assert abs(V[0] - self.DEFAULT_PARAMS["E_L"]) < 0.1, "Voltage should decay toward E_L"

    def test_excitatory_conductance_depolarizes(self):
        """Excitatory conductance should increase voltage."""
        V_base = np.array([-65.0])
        g_e_zero = np.zeros(1)
        g_e_high = np.array([1.0])
        g_i = np.zeros(1)
        I_ext = np.zeros(1)

        V_no_exc, _ = lif_step(V_base.copy(), g_e_zero, g_i, I_ext, **self.DEFAULT_PARAMS)
        V_with_exc, _ = lif_step(V_base.copy(), g_e_high, g_i, I_ext, **self.DEFAULT_PARAMS)

        assert V_with_exc[0] > V_no_exc[0], "Excitatory conductance should depolarize"

    def test_inhibitory_conductance_hyperpolarizes(self):
        """Inhibitory conductance should decrease voltage (when V > E_i)."""
        V_base = np.array([-65.0])  # Above E_i (-80)
        g_e = np.zeros(1)
        g_i_zero = np.zeros(1)
        g_i_high = np.array([1.0])
        I_ext = np.zeros(1)

        V_no_inh, _ = lif_step(V_base.copy(), g_e, g_i_zero, I_ext, **self.DEFAULT_PARAMS)
        V_with_inh, _ = lif_step(V_base.copy(), g_e, g_i_high, I_ext, **self.DEFAULT_PARAMS)

        assert V_with_inh[0] < V_no_inh[0], "Inhibitory conductance should hyperpolarize"

    def test_vectorized_multiple_neurons(self):
        """Function should handle multiple neurons correctly."""
        N = 100
        V = np.full(N, -60.0)
        g_e = np.zeros(N)
        g_i = np.zeros(N)
        I_ext = np.linspace(0, 20, N)  # Gradient of input

        V_new, spiked = lif_step(V, g_e, g_i, I_ext, **self.DEFAULT_PARAMS)

        assert V_new.shape == (N,), "Output shape should match input"
        assert spiked.shape == (N,), "Spiked array shape should match input"
        # Higher input neurons should have higher voltage (if not spiked)
        non_spiked_mask = ~spiked
        if non_spiked_mask.sum() > 1:
            non_spiked_V = V_new[non_spiked_mask]
            assert non_spiked_V[-1] >= non_spiked_V[0], "Higher input should yield higher V"
