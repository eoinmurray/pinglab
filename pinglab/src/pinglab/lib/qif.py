"""
Quadratic integrate-and-fire neuron model with conductance-based synapses.
"""
from __future__ import annotations

import numpy as np


def qif_step(
    V: np.ndarray,
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
    a: float,
    V_r: float,
    V_t: float,
    V_th: float | np.ndarray,
    V_reset: float,
    can_spike: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform one Euler step of the QIF model.

    Returns updated (V, spiked).
    """
    if can_spike is None:
        can_spike = np.ones(V.shape, dtype=bool)

    quad_term = a * (V - V_r) * (V - V_t)

    dVdt = (
        g_L * (E_L - V)
        + quad_term
        + I_ext
        + g_e * (E_e - V)
        + g_i * (E_i - V)
    ) / C_m
    V_new = V + dt * dVdt

    spiked = (V_new >= V_th) & can_spike
    V_new = np.where(spiked, V_reset, V_new)
    V_new = np.where(can_spike, V_new, V_reset)

    return V_new, spiked
