"""
Izhikevich neuron model with conductance-based synapses.
"""
from __future__ import annotations

import numpy as np


def izh_init_u(V: np.ndarray, b: float) -> np.ndarray:
    """Initialize recovery variable U at steady state."""
    return b * V


def izh_step(
    V: np.ndarray,
    U: np.ndarray,
    g_e: np.ndarray,
    g_i: np.ndarray,
    I_ext: np.ndarray,
    dt: float,
    *,
    a: float,
    b: float,
    c: float,
    d: float,
    V_th: float | np.ndarray,
    E_e: float,
    E_i: float,
    can_spike: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform one Euler step of the Izhikevich model.

    Returns updated (V, U, spiked).
    """
    if can_spike is None:
        can_spike = np.ones(V.shape, dtype=bool)

    dVdt = 0.04 * V ** 2 + 5.0 * V + 140.0 - U + I_ext + g_e * (E_e - V) + g_i * (E_i - V)
    dUdt = a * (b * V - U)

    V_new = V + dt * dVdt
    U_new = U + dt * dUdt

    spiked = (V_new >= V_th) & can_spike
    V_new = np.where(spiked, c, V_new)
    U_new = np.where(spiked, U_new + d, U_new)
    V_new = np.where(can_spike, V_new, c)

    return V_new, U_new, spiked
