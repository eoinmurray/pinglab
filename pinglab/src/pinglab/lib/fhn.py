"""
FitzHugh-Nagumo neuron model with conductance-based synapses.
"""
from __future__ import annotations

import numpy as np


def fhn_step(
    V: np.ndarray,
    W: np.ndarray,
    g_e: np.ndarray,
    g_i: np.ndarray,
    I_ext: np.ndarray,
    dt: float,
    *,
    a: float,
    b: float,
    tau_w: float,
    E_e: float,
    E_i: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform one Euler step of the FitzHugh-Nagumo model.

    Returns updated (V, W).
    """
    dVdt = V - (V ** 3) / 3.0 - W + I_ext + g_e * (E_e - V) + g_i * (E_i - V)
    dWdt = (V + a - b * W) / tau_w

    V_new = V + dt * dVdt
    W_new = W + dt * dWdt
    return V_new, W_new
