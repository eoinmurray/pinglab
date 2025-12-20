from __future__ import annotations

from typing import Optional
import numpy as np


def calculate_regime_label(
    mean_rate_E: float,
    mean_rate_I: float,
    cv_population: float,
    cv_per_neuron: np.ndarray,
    corr_EE_mean: float,
    corr_II_mean: float,
    corr_EI_mean: float,
    lag_mean_ms: float,
    lag_std_ms: float,
    gamma_peak_freq: Optional[float],
    gamma_peak_power: Optional[float],
    gamma_Q: Optional[float],
    fano_E: float,
    fano_I: float,
    synchrony: float,
    g_e_mean: float,
    g_i_mean: float,
    g_ei_ratio: float,
    g_e_cv: float,
    g_i_cv: float,
) -> tuple[str, str]:
    """
    Classify network regime from scalar metrics.

    Returns one of:
      - 'silent'
      - 'AI'       (asynchronous irregular)
      - 'PING'     (gamma oscillatory)
      - 'burst'
      - 'unstable'
    """

    # ---------- 0. Basic aggregates ----------
    corr_mean = float(np.mean([corr_EE_mean, corr_II_mean, corr_EI_mean]))

    has_gamma = (
        gamma_peak_freq is not None
        and gamma_peak_power is not None
        and gamma_peak_power > 1e8
        and 20.0 <= gamma_peak_freq <= 100.0
        and gamma_Q is not None
        and gamma_Q > 2.0
    )

    # modest helper flags – deliberately loose
    low_corr_sync  = (abs(corr_mean) < 0.1) and (synchrony < 3.0)
    high_corr_sync = (abs(corr_mean) > 0.2) or (synchrony > 5.0)

    cv_bursty = cv_population > 1.8
    fano_big  = (fano_E > 5.0) or (fano_I > 5.0)

    # ---------- 1. Silent ----------
    if mean_rate_E < 0.1 and mean_rate_I < 0.1:
        return "silent", "Low firing rates"

    # ---------- 2. Hard "properly broken" guard ----------
    # Extreme rates or absurd E/I ratio: call it unstable no matter what.
    if mean_rate_E > 300.0 or mean_rate_I > 300.0:
        return "unstable", "Extreme firing rates"
    if g_ei_ratio > 20.0 or g_ei_ratio < 0.05:
        return "unstable", "Extreme E/I conductance ratio"

    # ---------- 3. Gamma present → PING by definition ----------
    # If the PSD says "there is a gamma peak", we respect it.
    if has_gamma:
        return "PING", "Gamma peak detected"

    # From here on: NO gamma peak.

    # ---------- 4. AI (asynchronous irregular) ----------
    # No gamma, weak correlations, moderate CV/Fano.
    if low_corr_sync and (0.5 <= cv_population <= 1.8) and not fano_big:
        return "AI", "Low correlations and moderate irregularity"

    # ---------- 5. Burst ----------
    # No gamma, but big irregularity + elevated correlations/synchrony.
    if (cv_bursty or fano_big) and high_corr_sync:
        return "burst", "High irregularity and high correlations"

    # ---------- 6. Fallbacks ----------

    # If it's loose but doesn't quite meet AI thresholds, still prefer "AI"
    # over calling everything weird "unstable".
    if low_corr_sync:
        return "AI", "Low correlations but irregularity outside AI range"

    # Anything else is genuinely ambiguous / messy.
    return "unstable", "Did not meet any specific regime criteria"
