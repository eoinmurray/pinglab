"""Compute population averaging and signed-current cancellation diagnostics.

Native SNNLab estimators supply ISI CV and spike counts. Sliding-window counts
use cumulative sums and NumPy Pearson correlations; the native correlation API
supports only disjoint bins.
The additional group-size scaling and current covariance decomposition implement
the diagnostic definitions in Renart's supplement, without an AI classifier.

Currents use conductances at step t and voltage from step t-1, matching the
simulator's held-conductance membrane update. All four covariance components
use the same ordered pairs and time-lag normalization, so their sum is exact.
"""

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import lfilter
from snnlab import analysis


def pair_correlations(spikes, cfg):
    selected = []
    sizes = []
    for population in ("E", "I"):
        values = spikes[population]
        cells = np.linspace(
            0,
            values.shape[1] - 1,
            min(cfg["analysis"]["correlation_cells"], values.shape[1]),
            dtype=int,
        )
        selected.append(values[:, cells])
        sizes.append(len(cells))
    combined = np.concatenate(selected, axis=1)
    step_ms = cfg["analysis"]["correlation_step_ms"]
    # Bin at the slide step, then sum adjacent bins without padding either end.
    # Each row is a complete square counting window inside the measurement period.
    counts = analysis.binned_spike_counts(
        combined, cfg["dt_ms"], bin_ms=step_ms, rounding="exact"
    )["counts"]
    cumulative = np.vstack(
        (np.zeros((1, counts.shape[1]), dtype=np.int64), counts.cumsum(axis=0))
    )
    arrays, summaries = {}, {}
    for width in cfg["analysis"]["correlation_bins_ms"]:
        width_bins = round(width / step_ms)
        if width_bins < 1 or not np.isclose(width_bins * step_ms, width):
            raise ValueError(
                "correlation width must be a positive multiple of its step"
            )
        windows = cumulative[width_bins:] - cumulative[:-width_bins]
        valid = (
            windows.std(axis=0) > 0
            if len(windows) >= 2
            else np.zeros(counts.shape[1], dtype=bool)
        )
        matrix = np.full((counts.shape[1], counts.shape[1]), np.nan)
        if valid.any():
            # A single varying cell has no valid distinct-cell pair.
            if valid.sum() == 1:
                matrix[np.ix_(valid, valid)] = 1
            else:
                matrix[np.ix_(valid, valid)] = np.corrcoef(
                    windows[:, valid], rowvar=False
                )
        n_e, n_i = sizes
        blocks = {
            "EE": matrix[:n_e, :n_e][np.triu_indices(n_e, 1)],
            "II": matrix[n_e:, n_e:][np.triu_indices(n_i, 1)],
            "EI": matrix[:n_e, n_e:].ravel(),
        }
        summaries[str(width)] = {}
        for name, values in blocks.items():
            values = values[np.isfinite(values)]
            arrays[f"correlation_{name}_{width:g}"] = values
            summaries[str(width)][name] = dict(
                mean=float(values.mean()) if values.size else None,
                std=float(values.std()) if values.size else None,
                pairs=int(values.size),
            )
    return summaries, arrays


def population_scaling(spikes, cfg, *, seed):
    settings = cfg["analysis"]
    counts = analysis.binned_spike_counts(
        spikes,
        cfg["dt_ms"],
        bin_ms=settings["scaling_bin_ms"],
    )["counts"].astype(np.float64)
    sizes = np.unique(
        [min(n, counts.shape[1]) for n in settings["group_sizes"]] + [counts.shape[1]]
    )
    rng = np.random.default_rng(seed)
    ratios = np.full((len(sizes), settings["subsample_repeats"]), np.nan)
    for index, size in enumerate(sizes):
        for repeat in range(settings["subsample_repeats"]):
            cells = rng.choice(counts.shape[1], size=int(size), replace=False)
            group = counts[:, cells]
            denominator = group.var(axis=0).mean()
            if denominator > 0:
                ratios[index, repeat] = group.mean(axis=1).var() / denominator
    quantiles = np.full((3, len(sizes)), np.nan)
    for index, values in enumerate(ratios):
        valid = values[np.isfinite(values)]
        if valid.size:
            quantiles[:, index] = np.percentile(valid, [5, 50, 95])
    valid = np.isfinite(quantiles[1]) & (quantiles[1] > 0)
    slope = (
        float(np.polyfit(np.log(sizes[valid]), np.log(quantiles[1, valid]), 1)[0])
        if valid.sum() >= 2
        else None
    )
    return dict(
        loglog_slope=slope,
        bin_ms=settings["scaling_bin_ms"],
        repeats=settings["subsample_repeats"],
    ), dict(
        group_sizes=sizes,
        q_low=quantiles[0],
        q_median=quantiles[1],
        q_high=quantiles[2],
    )


def reconstruct_currents(signals, cfg, population):
    voltage = signals[f"{population}.voltage"].astype(np.float64)
    previous = np.concatenate(
        [np.full_like(voltage[:1], cfg["resting_mv"]), voltage[:-1]], axis=0
    )
    excitation = (
        signals[f"E_to_{population}.conductance"].astype(np.float64)
        + signals[f"X_to_{population}.conductance"]
    )
    inhibition = signals[f"I_to_{population}.conductance"].astype(np.float64)
    # µS × mV = nA. Inhibitory current is signed, not its positive magnitude.
    return excitation * (cfg["excitatory_reversal_mv"] - previous), inhibition * (
        cfg["inhibitory_reversal_mv"] - previous
    )


def pair_covariance(a, b, max_lag):
    """Average Cov(a_i(t), b_j(t+lag)) over ordered distinct-cell pairs.

    Subtract same-cell products from the all-pair sum in Fourier space.
    Global temporal means are removed; each lag uses its actual overlap length.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] < 2:
        raise ValueError(
            "current covariance requires matching time-by-cell arrays and at least two cells"
        )
    steps, cells = a.shape
    if not 0 <= max_lag < steps:
        raise ValueError("lag must be shorter than the measurement window")
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    fft_size = next_fast_len(2 * steps - 1)
    fa, fb = rfft(a, fft_size, axis=0), rfft(b, fft_size, axis=0)
    all_pairs = fa.sum(axis=1).conj() * fb.sum(axis=1)
    same_cell = (fa.conj() * fb).sum(axis=1)
    summed = irfft(all_pairs - same_cell, fft_size)
    lags = np.arange(-max_lag, max_lag + 1)
    return summed[lags % fft_size] / ((steps - np.abs(lags)) * cells * (cells - 1))


def current_cancellation(signals, cfg):
    settings = cfg["analysis"]
    excitation, inhibition = reconstruct_currents(
        signals, cfg, settings["current_population"]
    )
    start = round(cfg["burn_in_ms"] / cfg["dt_ms"])
    e, i = excitation[start:], inhibition[start:]
    max_lag = round(settings["current_max_lag_ms"] / cfg["dt_ms"])
    curves = {
        name: pair_covariance(a, b, max_lag)
        for name, a, b in (
            ("EE", e, e),
            ("II", i, i),
            ("EI", e, i),
            ("IE", i, e),
        )
    }
    total = pair_covariance(e + i, e + i, max_lag)
    np.testing.assert_allclose(total, sum(curves.values()), atol=1e-12, rtol=1e-8)
    decay = np.exp(-cfg["dt_ms"] / settings["current_filter_ms"])
    filtered = lfilter([1 - decay], [1, -decay], excitation + inhibition, axis=0)[
        start:
    ]
    filtered_total = pair_covariance(filtered, filtered, max_lag)
    zero = max_lag
    denominator = curves["EE"][zero] + curves["II"][zero]
    summary = dict(
        population=settings["current_population"],
        cells=e.shape[1],
        zero_lag_nA2={name: float(values[zero]) for name, values in curves.items()},
        total_zero_lag_nA2=float(total[zero]),
        total_to_same_type_zero_lag=float(total[zero] / denominator)
        if denominator > 0
        else None,
        filtered_total_zero_lag_nA2=float(filtered_total[zero]),
    )
    return summary, dict(
        current_lags_ms=np.arange(-max_lag, max_lag + 1) * cfg["dt_ms"],
        **{f"current_{name}": values for name, values in curves.items()},
        current_total=total,
        current_filtered_total=filtered_total,
    )
