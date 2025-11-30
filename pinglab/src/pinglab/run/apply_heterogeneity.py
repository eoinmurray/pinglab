import numpy as np


def apply_heterogeneity(
    rng: np.random.RandomState,
    N_E: int,
    N_I: int,
    V_th_heterogeneity_sd: float,
    g_L_heterogeneity_sd: float,
    C_m_heterogeneity_sd: float,
    t_ref_heterogeneity_sd: float,
    g_L_E: float,
    g_L_I: float,
    C_m_E: float,
    C_m_I: float,
    V_th: float,
    t_ref_E: float,
    t_ref_I: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate heterogeneous neural parameters from base values.

    Parameters:
        rng: Random number generator
        N_E, N_I: Population sizes
        V_th_heterogeneity_sd: Spike threshold std (mV)
        g_L_heterogeneity_sd: Leak conductance variation (relative %, e.g., 0.15 = 15%)
        C_m_heterogeneity_sd: Membrane capacitance variation (relative %, e.g., 0.1 = 10%)
        t_ref_heterogeneity_sd: Refractory period variation (ms)
        g_L_E, g_L_I: Base leak conductances
        C_m_E, C_m_I: Base membrane capacitances
        V_th: Base spike threshold
        t_ref_E: Base refractory period for E neurons
        t_ref_I: Base refractory period for I neurons

    Returns:
        V_th_arr: Per-neuron thresholds
        g_L_arr: Per-neuron leak conductances
        C_m_arr: Per-neuron membrane capacitances
        t_ref_arr: Per-neuron refractory periods (in ms)
    """
    N = N_E + N_I

    # Spike threshold heterogeneity
    V_th_arr = np.full(N, V_th)
    if V_th_heterogeneity_sd > 0:
        V_th_het_E = rng.normal(0, V_th_heterogeneity_sd, N_E)
        V_th_het_I = rng.normal(0, V_th_heterogeneity_sd, N_I)
        V_th_arr += np.concatenate([V_th_het_E, V_th_het_I])

    # Leak conductance heterogeneity (multiplicative, as relative variation)
    g_L_arr = np.concatenate([np.full(N_E, g_L_E), np.full(N_I, g_L_I)])
    if g_L_heterogeneity_sd > 0:
        # Apply relative variation: multiply by (1 + N(0, std))
        g_L_het_E = rng.normal(0, g_L_heterogeneity_sd, N_E)
        g_L_het_I = rng.normal(0, g_L_heterogeneity_sd, N_I)
        g_L_het = np.concatenate([g_L_het_E, g_L_het_I])
        # Ensure conductances don't go negative
        g_L_arr *= 1.0 + g_L_het
        g_L_arr = np.clip(g_L_arr, 0.01, None)  # Minimum viable conductance

    # Membrane capacitance heterogeneity (multiplicative)
    C_m_arr = np.concatenate([np.full(N_E, C_m_E), np.full(N_I, C_m_I)])
    if C_m_heterogeneity_sd > 0:
        # Apply relative variation: multiply by (1 + N(0, std))
        C_m_het_E = rng.normal(0, C_m_heterogeneity_sd, N_E)
        C_m_het_I = rng.normal(0, C_m_heterogeneity_sd, N_I)
        C_m_het = np.concatenate([C_m_het_E, C_m_het_I])
        # Ensure capacitances don't go negative
        C_m_arr *= 1.0 + C_m_het
        C_m_arr = np.clip(C_m_arr, 0.01, None)  # Minimum viable capacitance

    # Refractory period heterogeneity
    t_ref_arr = np.concatenate([np.full(N_E, t_ref_E), np.full(N_I, t_ref_I)])
    if t_ref_heterogeneity_sd > 0:
        t_ref_het_E = rng.normal(0, t_ref_heterogeneity_sd, N_E)
        t_ref_het_I = rng.normal(0, t_ref_heterogeneity_sd, N_I)
        t_ref_arr += np.concatenate([t_ref_het_E, t_ref_het_I])
        # Ensure refractory periods don't go negative
        t_ref_arr = np.clip(t_ref_arr, 0.1, None)

    return V_th_arr, g_L_arr, C_m_arr, t_ref_arr
