"""Numerical definitions retained from the flat exp041 runner."""

import numpy as np
from scipy import signal as sp_signal

from .recipe import F_GAMMA_BAND_HZ


def _peak_with_parabolic(psd: np.ndarray, freqs: np.ndarray) -> float:
    """Locate the gamma-band peak with parabolic sub-bin interpolation.

    Returns NaN if the PSD is flat in the gamma band. Welch with
    nperseg = T_steps gives Δf = fs/nperseg (5 Hz at fs=10000, T=200ms)
    — too coarse on its own across six τ_GABA values. Parabolic
    interpolation through (k-1, k, k+1) recovers the analytic peak
    location with error O((Δf)^3) when the peak is well-isolated.
    """
    band_mask = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
    if not band_mask.any() or psd[band_mask].max() <= 0:
        return float("nan")
    in_band = np.where(band_mask)[0]
    peak_local = int(psd[in_band].argmax())
    peak_idx = int(in_band[peak_local])
    if not (0 < peak_idx < len(psd) - 1):
        return float(freqs[peak_idx])
    y0 = float(psd[peak_idx - 1])
    y1 = float(psd[peak_idx])
    y2 = float(psd[peak_idx + 1])
    denom = y0 - 2.0 * y1 + y2
    offset = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    offset = max(-0.5, min(0.5, offset))
    df = float(freqs[1] - freqs[0])
    return float(freqs[peak_idx]) + offset * df


def spectrum(traces, dt_ms):
    """Per-trial Welch density, then peak of the mean PSD; not median peaks."""
    psds = []
    for tr in traces:
        freqs, power = sp_signal.welch(
            tr - tr.mean(),
            fs=1000.0 / dt_ms,
            nperseg=traces[0].size,
            scaling="density",
            detrend=False,
        )
        psds.append(power)
    mean = np.mean(np.stack(psds, axis=0), axis=0)
    per_trial = [_peak_with_parabolic(p, freqs) for p in psds]
    return {
        "f_gamma_hz": _peak_with_parabolic(mean, freqs),
        "freqs_hz": freqs.tolist(),
        "psd": mean.tolist(),
        "per_trial_peaks_hz": [float(x) for x in per_trial if np.isfinite(x)],
    }


def summarize(rows):
    """Six condition means and seed SEMs; retain per-trial diagnostic statistics."""
    aggregate = []
    for tau in sorted({r["tau_gaba_ms"] for r in rows}):
        sub = [r for r in rows if r["tau_gaba_ms"] == tau]
        item = {"tau_gaba_ms": tau}
        for key in ("f_gamma_hz", "e_rate_hz", "acc"):
            values = [r[key] for r in sub]
            item[key] = {
                "mean": float(np.mean(values)),
                "sem": float(np.std(values, ddof=1) / np.sqrt(len(sub)))
                if len(sub) > 1
                else 0.0,
            }
        item["freqs_hz"] = sub[0]["freqs_hz"]
        item["psd"] = np.mean(np.stack([r["psd"] for r in sub]), axis=0).tolist()
        # The PSD panel historically marked the un-interpolated peak of the seed-mean spectrum.
        freqs = np.array(item["freqs_hz"])
        power = np.array(item["psd"])
        band = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
        idx = power[band].argmax()
        item["psd_marker"] = {
            "frequency_hz": float(freqs[band][idx]),
            "power": float(power[band][idx]),
        }
        peaks = np.array([p for row in sub for p in row["per_trial_peaks_hz"]])
        bins = np.arange(F_GAMMA_BAND_HZ[0], F_GAMMA_BAND_HZ[1] + 1.0, 1.0)
        counts, edges = np.histogram(peaks, bins=bins)
        item["trial_peaks"] = {
            "counts": counts.tolist(),
            "bins_hz": edges.tolist(),
            "median_hz": float(np.median(peaks)) if peaks.size else None,
            "iqr_hz": float(np.percentile(peaks, 75) - np.percentile(peaks, 25))
            if peaks.size
            else None,
        }
        aggregate.append(item)
    return aggregate


def fit_law(aggregate):
    """Retained least-squares fits on the six seed means, with centred R squared."""
    fg_arr = np.array([row["f_gamma_hz"]["mean"] for row in aggregate])
    er_arr = np.array([row["e_rate_hz"]["mean"] for row in aggregate])
    slope_aff, intercept_aff = np.polyfit(fg_arr, er_arr, 1)
    p_fit, a_fit = float(slope_aff), float(intercept_aff)
    ss_tot = float(np.sum((er_arr - er_arr.mean()) ** 2))
    ss_res_aff = float(np.sum((er_arr - (p_fit * fg_arr + a_fit)) ** 2))
    p0 = float(np.sum(fg_arr * er_arr) / np.sum(fg_arr**2))
    ss_res_0 = float(np.sum((er_arr - p0 * fg_arr) ** 2))
    return {
        "p_affine": p_fit,
        "a_affine": a_fit,
        "r2_affine": 1.0 - ss_res_aff / ss_tot if ss_tot > 0 else None,
        "p_origin": p0,
        "r2_origin": 1.0 - ss_res_0 / ss_tot if ss_tot > 0 else None,
    }
