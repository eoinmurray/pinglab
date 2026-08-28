"""Numerical definitions preserved from the combined exp046 runner."""

import numpy as np

MEASUREMENT = {
    "schema": "exp046.measurement/v1",
    "burst_detection": "population I count; Gaussian sigma 1 ms, +/-4 sigma; height 5% of trial maximum; minimum separation floor(half exp041 cycle period)",
    "cycle_edges": "integer midpoints between peaks; first and last intervals extend to trial boundaries; zero-peak trials skipped",
    "buckets": ["0", "1", "2", ">=3"],
    "aggregation": "pool cell-cycle counts across seeds and conditions",
    "ceiling_fit": "through-origin least squares over 18 network maxima; centred R squared; retained 1e-9 denominator floors",
}


def detect_i_burst_steps(
    s_i_trial: np.ndarray, dt_ms: float, f_gamma_hz: float
) -> np.ndarray:
    """Detect I-burst peak timesteps in a single trial.

    s_i_trial: (T, N_I) bool/int spike tensor.
    Returns: int array of peak timesteps (ascending).
    """
    from scipy.signal import find_peaks

    rate = s_i_trial.sum(axis=1).astype(np.float32)
    # Gaussian smooth with sigma = 1 ms.
    sigma_steps = max(1.0, 1.0 / dt_ms)
    L = int(np.ceil(4 * sigma_steps))
    k = np.arange(-L, L + 1)
    kernel = np.exp(-0.5 * (k / sigma_steps) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(rate, kernel, mode="same")
    cycle_steps = max(1.0, 1000.0 / max(f_gamma_hz, 1e-3) / dt_ms)
    # Require peaks to be ≥ 0.5 cycle apart and to lift above 5% of max.
    height = 0.05 * float(smooth.max())
    peaks, _ = find_peaks(
        smooth,
        distance=max(1, int(0.5 * cycle_steps)),
        height=height,
    )
    return peaks


def count_e_spikes_per_cycle(
    s_e_trial: np.ndarray,
    peak_steps: np.ndarray,
) -> np.ndarray:
    """Count E spikes per (cell, cycle) in one trial.

    s_e_trial: (T, N_E)
    peak_steps: (K,) I-burst peak timesteps.
    Returns: (K, N_E) int array.
    """
    T, N_E = s_e_trial.shape
    K = len(peak_steps)
    if K == 0:
        return np.zeros((0, N_E), dtype=np.int32)
    # Each timestep assigned to its nearest peak. Equivalent: cycle boundaries
    # at midpoints between consecutive peaks.
    edges = np.concatenate(
        [
            [0],
            ((peak_steps[:-1] + peak_steps[1:]) // 2).astype(int),
            [T],
        ]
    )
    counts = np.zeros((K, N_E), dtype=np.int32)
    for k in range(K):
        a, b = edges[k], edges[k + 1]
        if b > a:
            counts[k] = s_e_trial[a:b].sum(axis=0)
    return counts


def measure(R, per_cell_rate_hz, acc, tau_gaba_ms, dt_ms, f_gamma_hz):
    T, n_e, n_i = int(R["T"]), int(R["n_e"]), int(R["n_i"])
    trial_count = int(R["n_trials"])

    def _by_trial(prefix):
        tr = R[f"{prefix}_trial"]
        order = np.argsort(tr, kind="stable")
        return (
            R[f"{prefix}_t"][order],
            R[f"{prefix}_cell"][order],
            np.searchsorted(tr[order], np.arange(trial_count + 1)),
        )

    e_t, e_c, e_b = _by_trial("e")
    i_t, i_c, i_b = _by_trial("i")

    bucket_counts = np.zeros(4, dtype=np.int64)  # 0, 1, 2, ≥3
    cycle_count = 0
    for b in range(trial_count):
        s_i_trial = np.zeros((T, n_i), dtype=np.int8)
        s_i_trial[i_t[i_b[b] : i_b[b + 1]], i_c[i_b[b] : i_b[b + 1]]] = 1
        peaks = detect_i_burst_steps(s_i_trial, dt_ms, f_gamma_hz)
        if len(peaks) == 0:
            continue
        s_e_trial = np.zeros((T, n_e), dtype=np.int8)
        s_e_trial[e_t[e_b[b] : e_b[b + 1]], e_c[e_b[b] : e_b[b + 1]]] = 1
        counts = count_e_spikes_per_cycle(s_e_trial, peaks)  # (K, N_E)
        cycle_count += counts.shape[0]
        flat = counts.ravel()
        bucket_counts[0] += int((flat == 0).sum())
        bucket_counts[1] += int((flat == 1).sum())
        bucket_counts[2] += int((flat == 2).sum())
        bucket_counts[3] += int((flat >= 3).sum())

    return {
        "tau_gaba_ms": tau_gaba_ms,
        "f_gamma_hz": float(f_gamma_hz),
        "acc": acc,
        "n_trials": int(trial_count),
        "n_cycles_observed": int(cycle_count),
        "bucket_counts": bucket_counts.tolist(),
        "per_cell_rate_hz": per_cell_rate_hz.tolist(),
        "per_cell_max_rate_hz": float(per_cell_rate_hz.max()),
        "per_cell_median_rate_hz": float(np.median(per_cell_rate_hz)),
    }


def summarize(rows):
    per_tau = {}
    for tau in sorted({r["tau_gaba_ms"] for r in rows}):
        buckets = np.zeros(4, dtype=np.float64)
        for row in rows:
            if row["tau_gaba_ms"] == tau:
                buckets += np.array(row["bucket_counts"], dtype=np.float64)
        fractions = buckets / max(buckets.sum(), 1.0)
        per_tau[f"tau_{tau:g}"] = dict(
            zip(
                ("frac_zero", "frac_one", "frac_two", "frac_three_plus"),
                map(float, fractions),
                strict=True,
            )
        )
    global_buckets = np.zeros(4, dtype=np.int64)
    for row in rows:
        global_buckets += np.array(row["bucket_counts"], dtype=np.int64)
    total = int(global_buckets.sum())
    global_fracs = {
        name: float(global_buckets[i]) / max(total, 1)
        for i, name in enumerate(("zero", "one", "two", "three_plus"))
    }
    ordered = [
        r
        for tau in sorted({r["tau_gaba_ms"] for r in rows})
        for r in rows
        if r["tau_gaba_ms"] == tau
    ]
    frequencies = np.array([r["f_gamma_hz"] for r in ordered])
    maximum = np.array([r["per_cell_max_rate_hz"] for r in ordered])
    slope = float((frequencies * maximum).sum() / max((frequencies**2).sum(), 1e-9))
    residual = float(((maximum - slope * frequencies) ** 2).sum())
    total_ss = float(((maximum - maximum.mean()) ** 2).sum())
    return {
        "per_tau": per_tau,
        "global_fracs": global_fracs,
        "n_cell_cycle_pairs": total,
        "ceiling": {
            "max_cell_slope_vs_fgamma": slope,
            "max_cell_r2": 1.0 - residual / max(total_ss, 1e-9),
        },
    }
