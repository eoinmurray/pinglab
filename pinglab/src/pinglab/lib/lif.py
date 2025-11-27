"""
Vectorized leaky integrate-and-fire neuron model with conductance-based synapses.
"""
import numpy as np

def lif_step(
    V: np.ndarray,
    g_e: np.ndarray,
    g_i: np.ndarray,
    I_ext: np.ndarray,
    dt: float,
    *,
    E_L: float,
    E_e: float,
    E_i: float,
    C_m: float | np.ndarray,
    g_L: float | np.ndarray,
    V_th: float,
    V_reset: float,
    can_spike: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform one Euler integration step of the LIF model.

    Parameters:
        V: membrane potentials (mV)
        g_e: excitatory conductance
        g_i: inhibitory conductance
        I_ext: external input current
        dt: time step (ms)
        E_L: leak reversal potential
        E_e: excitatory reversal potential
        E_i: inhibitory reversal potential
        C_m: membrane capacitance (float or array for heterogeneous C_m)
        g_L: leak conductance (float or array for heterogeneous g_L)
        V_th: spike threshold
        V_reset: reset potential after spike
        can_spike: boolean array indicating which neurons can spike (not in refractory period).
                  If None, all neurons can spike. Neurons not in can_spike are held at V_reset.

    Returns:
        V_new: updated membrane potentials
        spiked: boolean array indicating which neurons spiked (only for can_spike neurons)
    """
    # If no refractory mask provided, assume all can spike
    if can_spike is None:
        can_spike = np.ones(V.shape, dtype=bool)

    # Compute membrane potential derivative
    dVdt = (g_L * (E_L - V) + g_e * (E_e - V) + g_i * (E_i - V) + I_ext) / C_m
    V_new = V + dt * dVdt

    # Detect spikes (only in non-refractory neurons)
    spiked = (V_new >= V_th) & can_spike

    # Reset spiked neurons
    V_new = np.where(spiked, V_reset, V_new)

    # Clamp refractory neurons to V_reset (hold absolute refractory period)
    V_new = np.where(can_spike, V_new, V_reset)

    return V_new, spiked