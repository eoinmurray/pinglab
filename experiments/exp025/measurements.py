"""Preserved exp025 measurements; no simulation, storage selection or plotting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .recipe import F_GAMMA_BAND_HZ, MODELS, RATE_TARGET_GRID_HZ


def aggregate_frontier(rows: list[dict]) -> list[dict]:
    """Summarise each model/budget point across independent seeds."""
    aggregates: list[dict] = []
    for model in MODELS:
        for rate_target_hz in RATE_TARGET_GRID_HZ:
            cell_rows = [
                row
                for row in rows
                if row["model"] == model and row["rate_target_hz"] == rate_target_hz
            ]
            if not cell_rows:
                continue
            accs = np.asarray([row["final_acc"] for row in cell_rows], dtype=float)
            rates = np.asarray([row["rate_e"] for row in cell_rows], dtype=float)
            n = len(cell_rows)
            aggregates.append(
                {
                    "model": model,
                    "rate_target_hz": rate_target_hz,
                    "rate_target_display": cell_rows[0]["rate_target_display"],
                    "statistic": "mean_across_independent_seeds",
                    "uncertainty": "sem_across_independent_seeds",
                    "n_seeds": n,
                    "seeds": [int(row["seed"]) for row in cell_rows],
                    "cell_names": [row["cell_name"] for row in cell_rows],
                    "acc_mean": float(accs.mean()),
                    "rate_mean": float(rates.mean()),
                    "acc_sem": float(accs.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                    "rate_sem": float(rates.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                }
            )
    return aggregates


def aggregate_low_w_in_seed_rows(w_in: float, seed_rows: list[dict]) -> dict:
    """Aggregate one local control across independent training seeds."""
    if not seed_rows:
        raise ValueError("low-W_in aggregation requires at least one seed")

    def mean_sem(key: str) -> tuple[float, float]:
        values = np.asarray([row[key] for row in seed_rows], dtype=float)
        sem = (
            float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        )
        return float(values.mean()), sem

    final_acc, final_acc_sem = mean_sem("final_acc")
    rate_e, rate_e_sem = mean_sem("rate_e")
    rate_i, rate_i_sem = mean_sem("rate_i")
    return {
        "w_in": float(w_in),
        "seeds": [int(row["seed"]) for row in seed_rows],
        "n_seeds": len(seed_rows),
        "statistic": "mean_across_independent_seeds",
        "uncertainty": "sem_across_independent_seeds",
        "final_acc": final_acc,
        "final_acc_sem": final_acc_sem,
        "rate_e": rate_e,
        "rate_e_sem": rate_e_sem,
        "rate_i": rate_i,
        "rate_i_sem": rate_i_sem,
        "per_seed": seed_rows,
    }


def _f_gamma_from_population(
    pop_traces: list[np.ndarray],
    fs_hz: float,
) -> float:
    """Welch PSD per trial, averaged across trials, peak frequency in band
    via parabolic interpolation. Returns NaN if the spectrum is flat or
    the population is silent."""
    from scipy import signal as sp_signal

    if not pop_traces or pop_traces[0].size == 0:
        return float("nan")
    nperseg = pop_traces[0].size
    psds: list[np.ndarray] = []
    freqs: np.ndarray | None = None
    for tr in pop_traces:
        if tr.std() == 0:
            continue
        f, p = sp_signal.welch(
            tr - tr.mean(),
            fs=fs_hz,
            nperseg=nperseg,
            scaling="density",
            detrend=False,
        )
        psds.append(p)
        freqs = f
    if not psds or freqs is None:
        return float("nan")
    psd_mean = np.mean(np.stack(psds, axis=0), axis=0)
    band = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
    if not band.any() or psd_mean[band].max() <= 0:
        return float("nan")
    in_band = np.where(band)[0]
    peak_local = int(psd_mean[in_band].argmax())
    peak_idx = int(in_band[peak_local])
    if not (0 < peak_idx < len(psd_mean) - 1):
        return float(freqs[peak_idx])
    y0, y1, y2 = (
        float(psd_mean[peak_idx - 1]),
        float(psd_mean[peak_idx]),
        float(psd_mean[peak_idx + 1]),
    )
    denom = y0 - 2.0 * y1 + y2
    offset = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    offset = max(-0.5, min(0.5, offset))
    df = float(freqs[1] - freqs[0])
    return float(freqs[peak_idx] + offset * df)


def _detect_i_burst_peaks(
    s_i_trial: np.ndarray,
    dt_ms: float,
    f_gamma_hz: float,
) -> np.ndarray:
    """Mirrors exp046.detect_i_burst_steps. Smooth I population rate with
    1-ms Gaussian, find peaks separated by at least 0.5 cycle."""
    from scipy.signal import find_peaks

    rate = s_i_trial.sum(axis=1).astype(np.float32)
    sigma_steps = max(1.0, 1.0 / dt_ms)
    L = int(np.ceil(4 * sigma_steps))
    k = np.arange(-L, L + 1)
    kernel = np.exp(-0.5 * (k / sigma_steps) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(rate, kernel, mode="same")
    cycle_steps = max(1.0, 1000.0 / max(f_gamma_hz, 1e-3) / dt_ms)
    height = 0.05 * float(smooth.max()) if smooth.max() > 0 else 0.0
    peaks, _ = find_peaks(
        smooth,
        distance=max(1, int(0.5 * cycle_steps)),
        height=height,
    )
    return peaks


def _count_e_spikes_per_cycle(
    s_e_trial: np.ndarray,
    peak_steps: np.ndarray,
) -> np.ndarray:
    """Mirrors exp046.count_e_spikes_per_cycle."""
    T, N_E = s_e_trial.shape
    K = len(peak_steps)
    if K == 0:
        return np.zeros((0, N_E), dtype=np.int32)
    edges = np.concatenate(
        [
            [0],
            ((peak_steps[:-1] + peak_steps[1:]) // 2).astype(int),
            [T],
        ]
    )
    counts = np.zeros((K, N_E), dtype=np.int32)
    for kk in range(K):
        a, b = edges[kk], edges[kk + 1]
        if b > a:
            counts[kk] = s_e_trial[a:b].sum(axis=0)
    return counts


def measure_p_fgamma(out_dir: Path, dt_ms: float, is_ping: bool) -> dict:
    fs_hz = 1000.0 / dt_ms
    m = json.loads((out_dir / "metrics.json").read_text())
    rates = m.get("rates_hz", {})
    acc = float(m["best_acc"])
    e_rate = float(rates.get("hid", 0.0))
    i_rate = float(rates.get("inh", 0.0))
    if not is_ping:
        return {
            "acc": acc,
            "e_rate": e_rate,
            "i_rate": i_rate,
            "f_gamma": None,
            "p": None,
        }

    pt = np.load(out_dir / "pop_traces.npz")
    pop_e_traces = list(pt["pop_e"])
    f_gamma_val = _f_gamma_from_population(pop_e_traces, fs_hz)
    f_gamma = float(f_gamma_val) if np.isfinite(f_gamma_val) else None

    # Reconstruct per-trial dense E/I rasters from the sparse indices and count
    # (cell, cycle) participation. Cycle boundaries come from the I-burst peaks.
    R = np.load(out_dir / "rasters.npz")
    T, n_e, n_i = int(R["T"]), int(R["n_e"]), int(R["n_i"])
    n_trials = min(int(R["n_trials"]), len(pop_e_traces))

    def _by_trial(prefix):
        tr = R[f"{prefix}_trial"]
        order = np.argsort(tr, kind="stable")
        return (
            R[f"{prefix}_t"][order],
            R[f"{prefix}_cell"][order],
            np.searchsorted(tr[order], np.arange(n_trials + 1)),
        )

    e_t, e_c, e_b = _by_trial("e")
    i_t, i_c, i_b = _by_trial("i")

    n_cycle_pairs = 0
    n_cycle_pairs_active = 0
    for b in range(n_trials):
        s_i_trial = np.zeros((T, n_i), dtype=np.int8)
        s_i_trial[i_t[i_b[b] : i_b[b + 1]], i_c[i_b[b] : i_b[b + 1]]] = 1
        f_gamma_batch = _f_gamma_from_population([pop_e_traces[b]], fs_hz)
        if not np.isfinite(f_gamma_batch) or f_gamma_batch <= 0:
            continue
        peaks = _detect_i_burst_peaks(s_i_trial, dt_ms, f_gamma_batch)
        if peaks.size == 0:
            continue
        s_e_trial = np.zeros((T, n_e), dtype=np.int8)
        s_e_trial[e_t[e_b[b] : e_b[b + 1]], e_c[e_b[b] : e_b[b + 1]]] = 1
        counts = _count_e_spikes_per_cycle(s_e_trial, peaks)  # (K, N_E)
        n_cycle_pairs += counts.size
        n_cycle_pairs_active += int((counts > 0).sum())

    p = (
        float(n_cycle_pairs_active) / float(n_cycle_pairs)
        if n_cycle_pairs > 0
        else None
    )
    return {"acc": acc, "e_rate": e_rate, "i_rate": i_rate, "f_gamma": f_gamma, "p": p}


def scaled_metrics(out_dir: Path, fr_upper_target_hz: float, fr_upper_strength: float):
    m = json.loads((out_dir / "metrics.json").read_text())
    rates = m.get("rates_hz", {})
    penalty = 0.0
    if fr_upper_strength > 0:
        pc = np.load(out_dir / "per_cell_rates.npz")
        sample_rates = pc["rate_e_per_sample"]
        penalty = float(
            fr_upper_strength
            * (np.maximum(sample_rates - fr_upper_target_hz, 0.0) ** 2).mean()
        )
    return (
        float(m["best_acc"]),
        float(m["ce_loss"]),
        penalty,
        float(rates.get("hid", 0.0)),
        float(rates.get("inh", 0.0)),
    )
