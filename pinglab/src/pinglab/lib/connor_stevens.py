"""
Connor-Stevens neuron model with conductance-based synapses.
"""
from __future__ import annotations

import numpy as np


def _safe_div_nexp(x: np.ndarray, y: float) -> np.ndarray:
    """Compute x / (1 - exp(-x / y)) with a stable small-x limit."""
    small = np.abs(x / y) < 1e-6
    return np.where(small, y, x / (1.0 - np.exp(-x / y)))


def _safe_div_expm1(x: np.ndarray, y: float) -> np.ndarray:
    """Compute x / (exp(x / y) - 1) with a stable small-x limit."""
    small = np.abs(x / y) < 1e-6
    return np.where(small, y, x / np.expm1(x / y))


def _alpha_m(V: np.ndarray) -> np.ndarray:
    return 0.1 * _safe_div_nexp(V + 40.0, 10.0)


def _beta_m(V: np.ndarray) -> np.ndarray:
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


def _alpha_h(V: np.ndarray) -> np.ndarray:
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


def _beta_h(V: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


def _alpha_n(V: np.ndarray) -> np.ndarray:
    return 0.01 * _safe_div_nexp(V + 55.0, 10.0)


def _beta_n(V: np.ndarray) -> np.ndarray:
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


def _alpha_a(V: np.ndarray) -> np.ndarray:
    return 0.02 * _safe_div_nexp(V + 50.0, 10.0)


def _beta_a(V: np.ndarray) -> np.ndarray:
    return 0.0175 * _safe_div_expm1(V + 60.0, 10.0)


def _alpha_b(V: np.ndarray) -> np.ndarray:
    return 0.0016 * np.exp(-(V + 13.0) / 18.0)


def _beta_b(V: np.ndarray) -> np.ndarray:
    return 0.05 / (1.0 + np.exp(-(V + 10.0) / 5.0))


def cs_init_gating(
    V: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize Connor-Stevens gating variables at steady state for a given V."""
    V_rates = np.clip(V, -100.0, 60.0)
    alpha_m = _alpha_m(V_rates)
    beta_m = _beta_m(V_rates)
    alpha_h = _alpha_h(V_rates)
    beta_h = _beta_h(V_rates)
    alpha_n = _alpha_n(V_rates)
    beta_n = _beta_n(V_rates)
    alpha_a = _alpha_a(V_rates)
    beta_a = _beta_a(V_rates)
    alpha_b = _alpha_b(V_rates)
    beta_b = _beta_b(V_rates)

    m = alpha_m / (alpha_m + beta_m)
    h = alpha_h / (alpha_h + beta_h)
    n = alpha_n / (alpha_n + beta_n)
    a = alpha_a / (alpha_a + beta_a)
    b = alpha_b / (alpha_b + beta_b)
    return m, h, n, a, b


def cs_step(
    V: np.ndarray,
    m: np.ndarray,
    h: np.ndarray,
    n: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    g_e: np.ndarray,
    g_i: np.ndarray,
    I_ext: np.ndarray,
    dt: float,
    *,
    C_m: float | np.ndarray,
    g_L: float | np.ndarray,
    g_Na: float,
    g_K: float,
    g_A: float,
    E_L: float,
    E_Na: float,
    E_K: float,
    E_e: float,
    E_i: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform one Euler step of the Connor-Stevens model.

    Returns updated (V, m, h, n, a, b).
    """
    V_rates = np.clip(V, -100.0, 60.0)
    alpha_m = _alpha_m(V_rates)
    beta_m = _beta_m(V_rates)
    alpha_h = _alpha_h(V_rates)
    beta_h = _beta_h(V_rates)
    alpha_n = _alpha_n(V_rates)
    beta_n = _beta_n(V_rates)
    alpha_a = _alpha_a(V_rates)
    beta_a = _beta_a(V_rates)
    alpha_b = _alpha_b(V_rates)
    beta_b = _beta_b(V_rates)

    m = m + dt * (alpha_m * (1.0 - m) - beta_m * m)
    h = h + dt * (alpha_h * (1.0 - h) - beta_h * h)
    n = n + dt * (alpha_n * (1.0 - n) - beta_n * n)
    a = a + dt * (alpha_a * (1.0 - a) - beta_a * a)
    b = b + dt * (alpha_b * (1.0 - b) - beta_b * b)
    m = np.clip(m, 0.0, 1.0)
    h = np.clip(h, 0.0, 1.0)
    n = np.clip(n, 0.0, 1.0)
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)

    I_Na = g_Na * (m ** 3) * h * (V - E_Na)
    I_K = g_K * (n ** 4) * (V - E_K)
    I_A = g_A * (a ** 3) * b * (V - E_K)
    I_L = g_L * (V - E_L)

    dVdt = (I_ext + g_e * (E_e - V) + g_i * (E_i - V) - I_Na - I_K - I_A - I_L) / C_m
    V = V + dt * dVdt
    return V, m, h, n, a, b
