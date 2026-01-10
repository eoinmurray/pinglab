"""
Adaptive Exponential Integrate-and-Fire neuron model with conductance-based synapses.
"""
from __future__ import annotations

import numpy as np


def adex_step(
    V: np.ndarray,
    w: np.ndarray,
    g_e: np.ndarray,
    g_i: np.ndarray,
    I_ext: np.ndarray,
    dt: float,
    *,
    C_m: float | np.ndarray,
    g_L: float | np.ndarray,
    E_L: float,
    E_e: float,
    E_i: float,
    V_T: float,
    Delta_T: float,
    tau_w: float,
    a: float,
    b: float,
    V_reset: float,
    V_peak: float,
    can_spike: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform one Euler step of the AdEx model.

    Returns updated (V, w, spiked).
    """
    if can_spike is None:
        can_spike = np.ones(V.shape, dtype=bool)

    exp_arg = np.clip((V - V_T) / Delta_T, -100.0, 50.0)
    dVdt = (
        g_L * (E_L - V)
        + g_L * Delta_T * np.exp(exp_arg)
        - w
        + I_ext
        + g_e * (E_e - V)
        + g_i * (E_i - V)
    ) / C_m
    V_new = V + dt * dVdt

    dwdt = (a * (V - E_L) - w) / tau_w
    w_new = w + dt * dwdt

    spiked = (V_new >= V_peak) & can_spike
    V_new = np.where(spiked, V_reset, V_new)
    w_new = np.where(spiked, w_new + b, w_new)
    V_new = np.where(can_spike, V_new, V_reset)

    return V_new, w_new, spiked
