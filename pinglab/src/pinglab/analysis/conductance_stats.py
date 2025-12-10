
import numpy as np


def conductance_stats(
    g_e: np.ndarray,
    g_i: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    Compute basic statistics of excitatory and inhibitory conductances.

    Parameters
    ----------
    g_e : np.ndarray
        Excitatory conductance array, shape (T, N_E) or (T, N_total).
    g_i : np.ndarray
        Inhibitory conductance array, shape (T, N_I) or (T, N_total).

    Returns
    -------
    g_e_mean : float
        Mean excitatory conductance.
    g_i_mean : float
        Mean inhibitory conductance.
    g_ei_ratio : float
        Ratio g_e_mean / g_i_mean (0.0 if g_i_mean == 0).
    g_e_cv : float
        Coefficient of variation of excitatory conductance (std/mean).
    g_i_cv : float
        Coefficient of variation of inhibitory conductance (std/mean).
    """
    g_e = np.asarray(g_e, dtype=float)
    g_i = np.asarray(g_i, dtype=float)

    # Flatten over time and neurons
    ge_flat = g_e.ravel()
    gi_flat = g_i.ravel()

    # Remove any NaNs/infs just in case
    ge_flat = ge_flat[np.isfinite(ge_flat)]
    gi_flat = gi_flat[np.isfinite(gi_flat)]

    if ge_flat.size == 0:
        g_e_mean = 0.0
        g_e_cv = 0.0
    else:
        g_e_mean = float(np.mean(ge_flat))
        g_e_std = float(np.std(ge_flat))
        g_e_cv = g_e_std / g_e_mean if g_e_mean > 0 else 0.0

    if gi_flat.size == 0:
        g_i_mean = 0.0
        g_i_cv = 0.0
    else:
        g_i_mean = float(np.mean(gi_flat))
        g_i_std = float(np.std(gi_flat))
        g_i_cv = g_i_std / g_i_mean if g_i_mean > 0 else 0.0

    g_ei_ratio = g_e_mean / g_i_mean if g_i_mean > 0 else 0.0

    return g_e_mean, g_i_mean, g_ei_ratio, g_e_cv, g_i_cv
