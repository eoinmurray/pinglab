"""
Multi-Quadratic Integrate-and-Fire neuron model with conductance-based synapses.
"""
from __future__ import annotations

import numpy as np


def mqif_step(
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
    a_terms: np.ndarray,
    V_r_terms: np.ndarray,
    w_a_terms: np.ndarray,
    w_Vr_terms: np.ndarray,
    w_tau: np.ndarray,
    V_th: float | np.ndarray,
    V_reset: float,
    can_spike: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform one Euler step of the MQIF model.

    Returns updated (V, w, spiked).
    """
    if can_spike is None:
        can_spike = np.ones(V.shape, dtype=bool)

    if a_terms.size != V_r_terms.size:
        raise ValueError("mqif a_terms and V_r_terms must have the same length")

    if a_terms.size == 0:
        quad_term = 0.0
    else:
        quad_term = np.sum(a_terms * (V[:, None] - V_r_terms[None, :]) ** 2, axis=1)

    if w_a_terms.size != w_Vr_terms.size:
        raise ValueError("mqif w_a_terms and w_Vr_terms must have the same length")
    if w_a_terms.size != w_tau.size:
        raise ValueError("mqif w_a_terms and w_tau must have the same length")

    if w_a_terms.size == 0:
        w_sum = 0.0
        w_new = w
    else:
        if w.ndim == 1:
            w = w[:, None]
        if w.shape[1] != w_a_terms.size:
            raise ValueError("mqif w has incompatible shape for w terms")
        w_sum = np.sum(w, axis=1)
        dwdt = (w_a_terms[None, :] * (V[:, None] - w_Vr_terms[None, :]) ** 2 - w) / w_tau[
            None, :
        ]
        w_new = w + dt * dwdt

    dVdt = (
        g_L * (E_L - V)
        + quad_term
        - w_sum
        + I_ext
        + g_e * (E_e - V)
        + g_i * (E_i - V)
    ) / C_m
    V_new = V + dt * dVdt

    spiked = (V_new >= V_th) & can_spike
    V_new = np.where(spiked, V_reset, V_new)
    V_new = np.where(can_spike, V_new, V_reset)

    return V_new, w_new, spiked
