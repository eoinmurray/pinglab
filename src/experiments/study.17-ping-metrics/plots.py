"""Plot generation for study.17 — loads spike NPZ data, produces artifacts.

Usage:
    uv run python plots.py --data-dir data/<run_id>
    uv run python plots.py  # auto-discovers latest run
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from pinglab.plots.styles import save_both


def _compute_rate(spike_times, spike_ids, start, stop, edges, bin_ms):
    n_neurons = stop - start
    mask = (spike_ids >= start) & (spike_ids < stop)
    counts, _ = np.histogram(spike_times[mask], bins=edges)
    return counts / (n_neurons * bin_ms / 1000.0)


def _autocorr(x: np.ndarray, max_lag: int, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Full symmetric autocorrelation. Normalized to 1 at zero lag by default."""
    x = x - x.mean()
    norm = np.sum(x ** 2)
    if norm == 0:
        return np.arange(-max_lag, max_lag + 1, dtype=float), np.zeros(2 * max_lag + 1)
    acf = np.correlate(x, x, mode="full")
    mid = len(acf) // 2
    acf = acf[mid - max_lag:mid + max_lag + 1]
    if normalize:
        acf = acf / norm
    else:
        acf = acf / len(x)  # unbiased mean: units of variance (Hz²)
    lags = np.arange(-max_lag, max_lag + 1, dtype=float)
    return lags, acf


def _find_peaks(acf_positive: np.ndarray, min_height: float = 0.0) -> list[int]:
    """Find local maxima above min_height in the positive-lag ACF (excluding lag 0)."""
    peaks = []
    for i in range(1, len(acf_positive) - 1):
        if (acf_positive[i] > acf_positive[i - 1]
                and acf_positive[i] > acf_positive[i + 1]
                and acf_positive[i] > min_height):
            peaks.append(i)
    return peaks


def compute_autocorrelations(
    spike_times: np.ndarray,
    spike_ids: np.ndarray,
    *,
    layer_bounds: list[tuple[int, int, str]],
    T_ms: float,
    bin_ms: float = 5.0,
    burn_in_ms: float = 0.0,
    max_lag_ms: float = 1000.0,
) -> dict[str, dict]:
    """Compute autocorrelation + peaks for each layer. Returns per-layer results."""
    edges = np.arange(burn_in_ms, T_ms + bin_ms, bin_ms)
    n_bins = len(edges) - 1
    max_lag_bins = min(int(max_lag_ms / bin_ms), n_bins - 1)

    mask_time = spike_times >= burn_in_ms
    st = spike_times[mask_time]
    si = spike_ids[mask_time]

    results = {}
    for start, stop, label in layer_bounds:
        rate = _compute_rate(st, si, start, stop, edges, bin_ms)
        lag_bins, acf = _autocorr(rate, max_lag_bins)
        lags_ms = lag_bins * bin_ms

        # Positive-lag half (excluding zero)
        mid = max_lag_bins
        acf_pos = acf[mid + 1:]  # exclude lag=0
        peak_indices = _find_peaks(acf_pos)
        peak_lags_ms = [(i + 1) * bin_ms for i in peak_indices]
        peak_heights = [float(acf_pos[i]) for i in peak_indices]

        oscillation_strength = peak_heights[0] if peak_heights else 0.0
        oscillation_period_ms = peak_lags_ms[0] if peak_lags_ms else 0.0

        # Fit exponential decay A * exp(-t / tau) to peak envelope
        decay_tau_ms = 0.0
        decay_A = 0.0
        if len(peak_lags_ms) >= 2:
            from scipy.optimize import curve_fit
            pl = np.array(peak_lags_ms)
            ph = np.array(peak_heights)
            try:
                def _exp_decay(t, A, tau):
                    return A * np.exp(-t / tau)
                popt, _ = curve_fit(_exp_decay, pl, ph, p0=[ph[0], 200.0], maxfev=5000)
                decay_A, decay_tau_ms = float(popt[0]), float(popt[1])
            except RuntimeError:
                pass

        # SNR from unnormalized autocorrelation (Hz² units)
        _, acf_raw = _autocorr(rate, max_lag_bins, normalize=False)
        raw_mid = max_lag_bins
        acf_raw_pos = acf_raw[raw_mid + 1:]
        first_peak_raw = float(acf_raw_pos[peak_indices[0]]) if peak_indices else 0.0
        noise_mask = np.abs(lags_ms) > 600.0
        noise_floor_raw = float(np.mean(np.abs(acf_raw[noise_mask]))) if noise_mask.any() else 0.0
        snr = first_peak_raw / noise_floor_raw if noise_floor_raw > 0 else 0.0

        # Zero-lag peak FWHM — measure of burstiness
        # Walk outward from center until ACF drops below half-max (0.5)
        half_max = 0.5  # normalized ACF has peak=1.0 at lag 0
        fwhm_bins = 0
        for i in range(1, mid + 1):
            if acf[mid + i] < half_max:
                fwhm_bins = i
                break
        fwhm_ms = 2.0 * fwhm_bins * bin_ms  # full width (both sides)

        results[label] = {
            "lags_ms": lags_ms,
            "acf": acf,
            "peak_lags_ms": peak_lags_ms,
            "peak_heights": peak_heights,
            "oscillation_strength": round(oscillation_strength, 4),
            "oscillation_period_ms": round(oscillation_period_ms, 1),
            "decay_tau_ms": round(decay_tau_ms, 1),
            "decay_A": round(decay_A, 4),
            "snr": round(snr, 2),
            "zero_lag_fwhm_ms": round(fwhm_ms, 1),
            "n_neurons": stop - start,
        }
    return results


def save_population_rates(
    path: Path,
    spike_times: np.ndarray,
    spike_ids: np.ndarray,
    *,
    layer_bounds: list[tuple[int, int, str]],
    T_ms: float,
    bin_ms: float = 5.0,
    burn_in_ms: float = 0.0,
) -> None:
    """Stacked plot of population firing rates (Hz) for each layer."""
    edges = np.arange(burn_in_ms, T_ms + bin_ms, bin_ms)
    mask_time = spike_times >= burn_in_ms
    st = spike_times[mask_time]
    si = spike_ids[mask_time]
    centers = (edges[:-1] + edges[1:]) / 2.0
    n_layers = len(layer_bounds)

    def _plot() -> None:
        fig, axes = plt.subplots(n_layers, 1, figsize=(6, 2.5 * n_layers + 0.5), sharex=True)
        if n_layers == 1:
            axes = [axes]
        for ax, (start, stop, label) in zip(axes, layer_bounds):
            rate_hz = _compute_rate(st, si, start, stop, edges, bin_ms)
            ax.plot(centers, rate_hz, linewidth=0.8)
            ax.set_ylabel("Hz")
            ax.set_title(f"{label} (n={stop - start})")
        axes[-1].set_xlabel("Time (ms)")
        fig.suptitle("Population Firing Rate")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

    save_both(path, _plot)


def save_autocorrelation(
    path: Path,
    acf_results: dict[str, dict],
) -> None:
    """Stacked autocorrelation plot with peaks marked."""
    labels = list(acf_results.keys())
    n_layers = len(labels)

    mono = "monospace"
    ann_fs = 7.5
    label_fs = 8.5

    def _bbox(ec, alpha=0.85):
        bg = plt.rcParams["axes.facecolor"]
        return dict(boxstyle="round,pad=0.25", fc=bg, ec=ec, alpha=alpha, lw=0.8)

    def _arrow(color):
        return dict(arrowstyle="-|>", color=color, lw=1.0, shrinkA=0, shrinkB=2)

    def _plot() -> None:
        fig, axes = plt.subplots(
            n_layers, 1, figsize=(7, 3.2 * n_layers + 0.6), sharex=True,
        )
        if n_layers == 1:
            axes = [axes]
        for ax, label in zip(axes, labels):
            r = acf_results[label]

            # Main ACF trace
            ax.fill_between(r["lags_ms"], 0, r["acf"], alpha=0.08, color="C0")
            ax.plot(r["lags_ms"], r["acf"], linewidth=0.7, color="C0", zorder=4)

            # Peak markers
            for lag, h in zip(r["peak_lags_ms"], r["peak_heights"]):
                ax.plot(lag, h, "o", color="C1", markersize=3.5, zorder=6,
                        markeredgewidth=0.5, markeredgecolor="C1")
                ax.plot(-lag, h, "o", color="C1", markersize=3.5, zorder=6,
                        markeredgewidth=0.5, markeredgecolor="C1")

            if r["peak_lags_ms"]:
                first_lag = r["peak_lags_ms"][0]
                first_h = r["peak_heights"][0]

                # Period vertical line
                ax.axvline(first_lag, color="C2", linewidth=0.7, linestyle=":",
                           alpha=0.5, zorder=3)

                # Expand y for headroom (cap base at 1.0 for normalized ACF)
                ax.set_ylim(bottom=-0.15, top=1.18)
                y_top = ax.get_ylim()[1]

                # T annotation (top, points at vertical line)
                ax.annotate(
                    f"T = {first_lag:.0f} ms",
                    xy=(first_lag, y_top * 0.82),
                    xytext=(first_lag + 220, y_top * 0.88),
                    fontsize=ann_fs, fontfamily=mono, ha="center",
                    bbox=_bbox("C2"),
                    arrowprops=_arrow("C2"),
                )

                # Peak height annotation
                ax.annotate(
                    f"{first_h:.2f}",
                    xy=(first_lag, first_h),
                    xytext=(first_lag + 140, first_h + 0.08),
                    fontsize=ann_fs, fontfamily=mono, ha="center",
                    bbox=_bbox("C1"),
                    arrowprops=_arrow("C1"),
                )

            # Exponential decay envelope
            tau = r.get("decay_tau_ms", 0.0)
            A = r.get("decay_A", 0.0)
            if tau > 0 and A > 0:
                t_env = np.linspace(0, max(r["lags_ms"]), 300)
                env = A * np.exp(-t_env / tau)
                ax.plot(t_env, env, "--", color="C3", linewidth=0.9, alpha=0.7, zorder=5)
                ax.plot(-t_env, env, "--", color="C3", linewidth=0.9, alpha=0.7, zorder=5)
                ax.annotate(
                    f"$\\tau$ = {tau:.0f} ms",
                    xy=(tau, A * np.exp(-1)),
                    xytext=(tau + 120, A * np.exp(-1) + 0.12),
                    fontsize=ann_fs, fontfamily=mono, ha="center",
                    bbox=_bbox("C3"),
                    arrowprops=_arrow("C3"),
                )

            # FWHM of zero-lag peak
            fwhm = r.get("zero_lag_fwhm_ms", 0.0)
            if fwhm > 0:
                lags_arr = r["lags_ms"]
                ax.text(
                    min(lags_arr) * 0.97, 0.92,
                    f"FWHM {fwhm:.0f} ms",
                    fontsize=ann_fs, fontfamily=mono, ha="left", va="top",
                    bbox=_bbox("#888888"),
                )

            # SNR (top right, standalone text)
            snr_val = r.get("snr", 0.0)
            if snr_val > 0:
                y_top_snr = ax.get_ylim()[1]
                lags_arr = r["lags_ms"]
                ax.text(
                    max(lags_arr) * 0.97, y_top_snr * 0.92,
                    f"SNR {snr_val:.1f}",
                    fontsize=ann_fs, fontfamily=mono, ha="right", va="top",
                    bbox=_bbox("#888888"),
                )

            ax.set_ylabel("Autocorrelation", fontsize=label_fs, fontfamily=mono)
            ax.set_title(f"{label}  n={r['n_neurons']}",
                         fontsize=label_fs, fontfamily=mono, pad=8)
            ax.axhline(0, color="grey", linewidth=0.4, linestyle="--", alpha=0.5)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[-1].set_xlabel("Lag (ms)", fontsize=label_fs, fontfamily=mono)
        fig.suptitle("Population Rate Autocorrelation",
                     fontsize=10, fontfamily=mono, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_both(path, _plot)


def compute_psd_metrics(
    spike_times: np.ndarray,
    spike_ids: np.ndarray,
    *,
    layer_bounds: list[tuple[int, int, str]],
    T_ms: float,
    bin_ms: float = 5.0,
    burn_in_ms: float = 0.0,
    f_target: float | None = None,
    f_search_lo: float = 5.0,
    f_search_hi: float = 80.0,
    sigma_hz: float = 1.75,
    n_harmonics: int = 5,
) -> dict[str, dict]:
    """Compute PSD-based PING metrics for each layer."""
    edges = np.arange(burn_in_ms, T_ms + bin_ms, bin_ms)
    dt_s = bin_ms / 1000.0
    mask_time = spike_times >= burn_in_ms
    st = spike_times[mask_time]
    si = spike_ids[mask_time]

    results = {}
    for start, stop, label in layer_bounds:
        rate = _compute_rate(st, si, start, stop, edges, bin_ms)
        rate = rate - rate.mean()

        # PSD via FFT
        n = len(rate)
        spectrum = np.fft.rfft(rate)
        psd = np.abs(spectrum) ** 2 / n
        freqs = np.fft.rfftfreq(n, d=dt_s)

        # Find fundamental: peak of PSD in search band
        band = (freqs >= f_search_lo) & (freqs <= f_search_hi)
        power_band = psd[band]
        freqs_band = freqs[band]
        f0_default = f_target if f_target is not None else 13.0
        f0 = float(freqs_band[np.argmax(power_band)]) if power_band.sum() > 0 else f0_default
        ft = f_target if f_target is not None else f0

        # Gaussian comb mask — teeth at f0, 2*f0, 3*f0, ...
        comb = np.zeros_like(freqs)
        for k in range(1, n_harmonics + 1):
            comb += np.exp(-((freqs - k * f0) ** 2) / (2 * sigma_hz ** 2))
        comb = np.clip(comb, 0, 1)

        p_signal = float(np.sum(comb * psd))
        p_noise = float(np.sum((1 - comb) * psd))
        p_total = float(np.sum(psd))
        snr_psd = p_signal / p_noise if p_noise > 0 else 0.0

        # Loss components
        import math
        l_concentration = -math.log(p_signal / p_total) if p_total > 0 and p_signal > 0 else 0.0
        l_frequency = (f0 - ft) ** 2
        l_noise = p_noise / p_signal if p_signal > 0 else 0.0

        results[label] = {
            "freqs": freqs,
            "psd": psd,
            "comb": comb,
            "signal_psd": comb * psd,
            "noise_psd": (1 - comb) * psd,
            "f0": round(f0, 2),
            "f_target": round(ft, 2),
            "p_signal": round(p_signal, 4),
            "p_noise": round(p_noise, 4),
            "p_total": round(p_total, 4),
            "snr_psd": round(snr_psd, 2),
            "l_concentration": round(l_concentration, 4),
            "l_frequency": round(l_frequency, 4),
            "l_noise": round(l_noise, 4),
            "n_neurons": stop - start,
        }
    return results


def save_psd(
    path: Path,
    psd_results: dict[str, dict],
) -> None:
    """Power spectrum plot for each layer."""
    labels = list(psd_results.keys())
    n_layers = len(labels)
    mono = "monospace"
    ann_fs = 7.5
    label_fs = 8.5

    def _bbox(ec, alpha=0.85):
        bg = plt.rcParams["axes.facecolor"]
        return dict(boxstyle="round,pad=0.25", fc=bg, ec=ec, alpha=alpha, lw=0.8)

    def _plot() -> None:
        fig, axes = plt.subplots(n_layers, 1, figsize=(7, 3.2 * n_layers + 0.6), sharex=True)
        if n_layers == 1:
            axes = [axes]
        for ax, label in zip(axes, labels):
            r = psd_results[label]
            freqs, psd = r["freqs"], r["psd"]
            f_max = 100  # show up to 100 Hz
            mask = freqs <= f_max
            ax.fill_between(freqs[mask], 0, psd[mask], alpha=0.08, color="C0")
            ax.plot(freqs[mask], psd[mask], linewidth=0.7, color="C0", zorder=4)

            # Mark f0 and harmonics
            f0 = r["f0"]
            ax.axvline(f0, color="C2", linewidth=0.7, linestyle=":", alpha=0.6, zorder=3)
            y_top = ax.get_ylim()[1]
            ax.text(
                f0 + 3, y_top * 0.9,
                f"f0 = {f0:.1f} Hz",
                fontsize=ann_fs, fontfamily=mono, ha="left", va="top",
                bbox=_bbox("C2"),
            )
            for k in range(2, 8):
                fk = k * f0
                if fk <= f_max:
                    ax.axvline(fk, color="C2", linewidth=0.5, linestyle=":",
                               alpha=0.3, zorder=3)

            ax.set_ylabel("Power", fontsize=label_fs, fontfamily=mono)
            ax.set_title(f"{label}  n={r['n_neurons']}",
                         fontsize=label_fs, fontfamily=mono, pad=8)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[-1].set_xlabel("Frequency (Hz)", fontsize=label_fs, fontfamily=mono)
        fig.suptitle("Power Spectral Density",
                     fontsize=10, fontfamily=mono, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_both(path, _plot)


def save_psd_comb(
    path: Path,
    psd_results: dict[str, dict],
) -> None:
    """PSD with Gaussian comb mask overlay."""
    labels = list(psd_results.keys())
    n_layers = len(labels)
    mono = "monospace"
    ann_fs = 7.5
    label_fs = 8.5

    def _bbox(ec, alpha=0.85):
        bg = plt.rcParams["axes.facecolor"]
        return dict(boxstyle="round,pad=0.25", fc=bg, ec=ec, alpha=alpha, lw=0.8)

    def _plot() -> None:
        fig, axes = plt.subplots(n_layers, 1, figsize=(7, 3.2 * n_layers + 0.6), sharex=True)
        if n_layers == 1:
            axes = [axes]
        for ax, label in zip(axes, labels):
            r = psd_results[label]
            freqs, psd, comb = r["freqs"], r["psd"], r["comb"]
            mask = freqs <= 200

            # PSD
            ax.plot(freqs[mask], psd[mask], linewidth=0.7, color="C0", zorder=4)

            # Comb mask on secondary y-axis
            ax2 = ax.twinx()
            ax2.fill_between(freqs[mask], 0, comb[mask], alpha=0.15, color="C1", zorder=2)
            ax2.plot(freqs[mask], comb[mask], linewidth=0.8, color="C1", alpha=0.6, zorder=3)
            ax2.set_ylim(0, 1.3)
            ax2.set_ylabel("W(f)", fontsize=label_fs, fontfamily=mono, color="C1")
            ax2.tick_params(labelsize=7, colors="C1")
            ax2.spines["top"].set_visible(False)

            f0 = r["f0"]
            ax.set_xlim(0, 100)
            ax.text(
                0.97, 0.92,
                f"f0 = {f0:.1f} Hz",
                fontsize=ann_fs, fontfamily=mono, ha="right", va="top",
                bbox=_bbox("C2"),
                transform=ax.transAxes,
            )

            ax.set_ylabel("Power", fontsize=label_fs, fontfamily=mono)
            ax.set_title(f"{label}  n={r['n_neurons']}",
                         fontsize=label_fs, fontfamily=mono, pad=8)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)

        axes[-1].set_xlabel("Frequency (Hz)", fontsize=label_fs, fontfamily=mono)
        fig.suptitle("PSD with Gaussian Comb Mask",
                     fontsize=10, fontfamily=mono, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_both(path, _plot)


def save_psd_decomposition(
    path: Path,
    psd_results: dict[str, dict],
) -> None:
    """Signal vs noise decomposition of PSD."""
    labels = list(psd_results.keys())
    n_layers = len(labels)
    mono = "monospace"
    ann_fs = 7.5
    label_fs = 8.5

    def _bbox(ec, alpha=0.85):
        bg = plt.rcParams["axes.facecolor"]
        return dict(boxstyle="round,pad=0.25", fc=bg, ec=ec, alpha=alpha, lw=0.8)

    def _plot() -> None:
        fig, axes = plt.subplots(n_layers, 1, figsize=(7, 3.2 * n_layers + 0.6), sharex=True)
        if n_layers == 1:
            axes = [axes]
        for ax, label in zip(axes, labels):
            r = psd_results[label]
            freqs = r["freqs"]
            sig, noi = r["signal_psd"], r["noise_psd"]
            mask = freqs <= 200

            ax.fill_between(freqs[mask], 0, sig[mask], alpha=0.35, color="C2",
                            label="Signal", zorder=4)
            ax.fill_between(freqs[mask], 0, noi[mask], alpha=0.25, color="C1",
                            label="Noise", zorder=3)
            ax.plot(freqs[mask], sig[mask] + noi[mask], linewidth=0.5, color="C0",
                    alpha=0.4, zorder=5)

            # Annotate SNR and loss components
            y_top = max(np.max(sig[mask]), np.max(noi[mask])) * 1.1
            ax.set_ylim(0, y_top * 1.25)
            y_top = ax.get_ylim()[1]

            ax.set_xlim(0, 100)
            metrics_text = (
                f"SNR   {r['snr_psd']:.1f}\n"
                f"L_con {r['l_concentration']:.2f}\n"
                f"L_frq {r['l_frequency']:.1f}\n"
                f"L_noi {r['l_noise']:.3f}"
            )
            ax.text(
                0.97, 0.95,
                metrics_text,
                fontsize=ann_fs, fontfamily=mono, ha="right", va="top",
                bbox=_bbox("#888888"),
                transform=ax.transAxes,
            )

            ax.legend(fontsize=7, loc="upper center", framealpha=0.7,
                      prop={"family": mono})
            ax.set_ylabel("Power", fontsize=label_fs, fontfamily=mono)
            ax.set_title(f"{label}  n={r['n_neurons']}",
                         fontsize=label_fs, fontfamily=mono, pad=8)
            ax.tick_params(labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[-1].set_xlabel("Frequency (Hz)", fontsize=label_fs, fontfamily=mono)
        fig.suptitle("Signal / Noise Decomposition",
                     fontsize=10, fontfamily=mono, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_both(path, _plot)


def main(
    data_dir: Path | str,
    artifacts_dir: Path | str | None = None,
) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from pinglab.io import layer_bounds_from_spec
    from pinglab.io.graph_renderer import save_graph_diagram

    data_dir = Path(data_dir)
    if artifacts_dir is None:
        from settings import ARTIFACTS_ROOT
        artifacts_dir = ARTIFACTS_ROOT / Path(__file__).parent.name
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with (data_dir / "config.json").open() as f:
        spec = json.load(f)

    layer_bounds = layer_bounds_from_spec(spec)
    T_ms = float(spec.get("sim", {}).get("T_ms", 1000.0))
    burn_in_ms = float(spec.get("execution", {}).get("burn_in_ms", 0.0))

    # Graph diagram
    save_graph_diagram(spec, artifacts_dir / "graph")

    # Load spikes
    spk = np.load(data_dir / "spikes.npz")
    spike_times = spk["times"]
    spike_ids = spk["ids"]

    save_population_rates(
        artifacts_dir / "pop_rates_main_main_00",
        spike_times,
        spike_ids,
        layer_bounds=layer_bounds,
        T_ms=T_ms,
        burn_in_ms=burn_in_ms,
    )

    acf_results = compute_autocorrelations(
        spike_times,
        spike_ids,
        layer_bounds=layer_bounds,
        T_ms=T_ms,
        burn_in_ms=burn_in_ms,
    )

    save_autocorrelation(
        artifacts_dir / "autocorr_main_main_00",
        acf_results,
    )

    # PSD metrics
    psd_results = compute_psd_metrics(
        spike_times,
        spike_ids,
        layer_bounds=layer_bounds,
        T_ms=T_ms,
        burn_in_ms=burn_in_ms,
    )

    save_psd(
        artifacts_dir / "psd_main_main_00",
        psd_results,
    )
    save_psd_comb(
        artifacts_dir / "psd_comb_main_main_00",
        psd_results,
    )
    save_psd_decomposition(
        artifacts_dir / "psd_decomp_main_main_00",
        psd_results,
    )

    # Save results.json with all metrics
    results = {}
    for label in acf_results:
        r = acf_results[label]
        p = psd_results[label]
        results[label] = {
            "oscillation_strength": r["oscillation_strength"],
            "oscillation_period_ms": r["oscillation_period_ms"],
            "decay_tau_ms": r["decay_tau_ms"],
            "decay_A": r["decay_A"],
            "snr": r["snr"],
            "zero_lag_fwhm_ms": r["zero_lag_fwhm_ms"],
            "f0_hz": p["f0"],
            "snr_psd": p["snr_psd"],
            "l_concentration": p["l_concentration"],
            "l_frequency": p["l_frequency"],
            "l_noise": p["l_noise"],
            "peak_lags_ms": [round(x, 1) for x in r["peak_lags_ms"]],
            "peak_heights": [round(x, 4) for x in r["peak_heights"]],
        }
    with open(artifacts_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Copy metadata to artifacts
    for name in ("config.json",):
        src = data_dir / name
        if src.exists():
            shutil.copy2(src, artifacts_dir / name)

    print(f"Plots saved to {artifacts_dir}")


def _find_latest_run(experiment_dir: Path) -> Path:
    data_root = experiment_dir / "data"
    if not data_root.exists():
        raise FileNotFoundError(f"No data directory at {data_root}")
    runs = [d for d in data_root.iterdir() if d.is_dir() and d.name != "MNIST"]
    if not runs:
        raise FileNotFoundError(f"No run directories in {data_root}")
    return max(runs, key=lambda d: d.stat().st_mtime)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate plots from saved run data")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Path to run data dir (default: auto-discover latest)")
    parser.add_argument("--artifacts-dir", type=Path, default=None,
                        help="Where to write plots")
    args = parser.parse_args()

    experiment_dir = Path(__file__).parent.resolve()
    if args.data_dir is None:
        data_dir = _find_latest_run(experiment_dir)
        print(f"Auto-discovered latest run: {data_dir}")
    else:
        data_dir = args.data_dir

    main(data_dir=data_dir, artifacts_dir=args.artifacts_dir)
