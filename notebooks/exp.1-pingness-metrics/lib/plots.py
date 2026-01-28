from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pinglab.plots.raster import save_raster
from pinglab.plots.styles import save_both
from pinglab.types import Spikes


def plot_raster(
    spikes: Spikes,
    out_path: Path,
    *,
    label: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    info_lines: list[str] | None = None,
) -> None:
    save_raster(
        spikes,
        out_path,
        label=label,
        xlim=xlim,
        ylim=ylim,
        info_lines=info_lines,
    )


def plot_weight_histograms(
    W_ee: np.ndarray,
    W_ei: np.ndarray,
    W_ie: np.ndarray,
    W_ii: np.ndarray,
    out_path: Path,
    *,
    title: str,
) -> None:
    def plot_fn() -> None:
        fig, axes = plt.subplots(2, 2, figsize=(6, 6))
        blocks = [
            ("W_ee", W_ee),
            ("W_ei", W_ei),
            ("W_ie", W_ie),
            ("W_ii", W_ii),
        ]
        for ax, (label, block) in zip(axes.flat, blocks):
            data = block.ravel()
            ax.hist(data, bins=50, alpha=0.8)
            ax.set_title(label)
            ax.grid(True, alpha=0.2)
        fig.suptitle(title)
        plt.tight_layout()

    save_both(out_path, plot_fn)


def plot_population_rate(
    times_ms: np.ndarray,
    rate_hz: np.ndarray,
    out_path: Path,
    *,
    title: str,
) -> None:
    def plot_fn() -> None:
        plt.figure(figsize=(6, 6))
        if times_ms.size and rate_hz.size:
            plt.plot(times_ms, rate_hz, linewidth=1.5)
        else:
            plt.text(0.5, 0.5, "No spikes", ha="center", va="center")
        plt.xlabel("Time (ms)")
        plt.ylabel("Rate (Hz)")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    save_both(out_path, plot_fn)


def plot_rate_psd(
    freqs_hz: np.ndarray,
    psd: np.ndarray,
    out_path: Path,
    *,
    title: str,
) -> None:
    def plot_fn() -> None:
        plt.figure(figsize=(6, 6))
        if freqs_hz.size and psd.size:
            plt.plot(freqs_hz, psd, linewidth=1.5)
        else:
            plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (Hz²/Hz)")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    save_both(out_path, plot_fn)


def plot_autocorr(
    lags_ms: np.ndarray,
    corr: np.ndarray,
    out_path: Path,
    *,
    title: str,
    window: tuple[float, float] | None = None,
    peak_lag_ms: float | None = None,
    peak_value: float | None = None,
) -> None:
    def plot_fn() -> None:
        plt.figure(figsize=(6, 6))
        if lags_ms.size and corr.size:
            plt.plot(lags_ms, corr, linewidth=1.5)
            if window is not None:
                plt.axvspan(window[0], window[1], alpha=0.15)
            if peak_lag_ms is not None and peak_value is not None:
                plt.scatter([peak_lag_ms], [peak_value], s=40, zorder=3)
                plt.axvline(peak_lag_ms, linestyle="--", alpha=0.4)
        else:
            plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.xlabel("Lag (ms)")
        plt.ylabel("Correlation")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    save_both(out_path, plot_fn)


def plot_pairwise_xcorr(
    lags_ms: np.ndarray,
    corr: np.ndarray,
    out_path: Path,
    *,
    title: str,
    window: tuple[float, float] | None = None,
    peak_lag_ms: float | None = None,
    peak_value: float | None = None,
) -> None:
    def plot_fn() -> None:
        plt.figure(figsize=(6, 6))
        if lags_ms.size and corr.size:
            plt.plot(lags_ms, corr, linewidth=1.5)
            if window is not None:
                plt.axvspan(window[0], window[1], alpha=0.15)
            if peak_lag_ms is not None and peak_value is not None:
                plt.scatter([peak_lag_ms], [peak_value], s=40, zorder=3)
                plt.axvline(peak_lag_ms, linestyle="--", alpha=0.4)
        else:
            plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.xlabel("Lag (ms)")
        plt.ylabel("Correlation")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    save_both(out_path, plot_fn)


def plot_coherence(
    lags_ms: np.ndarray,
    corr: np.ndarray,
    out_path: Path,
    *,
    title: str,
    peak_lag_ms: float | None = None,
    peak_value: float | None = None,
) -> None:
    def plot_fn() -> None:
        plt.figure(figsize=(6, 6))
        if lags_ms.size and corr.size:
            plt.plot(lags_ms, corr, linewidth=1.5)
            if peak_lag_ms is not None and peak_value is not None:
                plt.scatter([peak_lag_ms], [peak_value], s=40, zorder=3)
                plt.axvline(peak_lag_ms, linestyle="--", alpha=0.4)
        else:
            plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.xlabel("Lag (ms)")
        plt.ylabel("Coherence")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    save_both(out_path, plot_fn)


def plot_lagged_windows(
    times_ms: np.ndarray,
    rate_hz: np.ndarray,
    window_ms: float,
    out_path: Path,
    *,
    title: str,
) -> None:
    def plot_fn() -> None:
        plt.figure(figsize=(6, 6))
        if times_ms.size and rate_hz.size and window_ms > 0:
            plt.plot(times_ms, rate_hz, linewidth=1.2)
            start = times_ms[0]
            end = times_ms[-1]
            t = start
            while t < end:
                plt.axvline(t, linestyle="--", alpha=0.4)
                t += window_ms
        else:
            plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.xlabel("Time (ms)")
        plt.ylabel("Rate (Hz)")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    save_both(out_path, plot_fn)


def plot_lagged_coherence_spectrum(
    freqs_hz: np.ndarray,
    lambda_vals: np.ndarray,
    out_path: Path,
    *,
    title: str,
) -> None:
    def plot_fn() -> None:
        plt.figure(figsize=(6, 6))
        if freqs_hz.size and lambda_vals.size:
            plt.plot(freqs_hz, lambda_vals, linewidth=1.5)
            plt.scatter(freqs_hz, lambda_vals, s=10, alpha=0.6)
        else:
            plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Lagged coherence (λ)")
        plt.ylim(0.0, 1.05)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    save_both(out_path, plot_fn)


def plot_metrics_vs_mu(
    mean_values: np.ndarray,
    rows: list[dict[str, float]],
    out_path: Path,
    *,
    title: str,
) -> None:
    def plot_fn() -> None:
        plt.figure(figsize=(6, 6))
        colors = {
            "autocorr_peak": "#1f77b4",
            "xcorr_peak": "#ff7f0e",
            "coherence_peak": "#9467bd",
            "lagged_lambda": "#2ca02c",
        }
        labels = {
            "autocorr_peak": "Autocorr peak",
            "xcorr_peak": "Pairwise xcorr peak",
            "coherence_peak": "Coherence peak",
            "lagged_lambda": "Lagged coherence",
        }
        for key in ["autocorr_peak", "xcorr_peak", "coherence_peak", "lagged_lambda"]:
            for mu in mean_values:
                points = [
                    row[key]
                    for row in rows
                    if np.isclose(row["mu_g_ei"], float(mu))
                ]
                if not points:
                    continue
                x = np.full(len(points), float(mu))
                plt.scatter(x, points, alpha=0.6, color=colors[key])
            means = [
                float(
                    np.mean(
                        [
                            row[key]
                            for row in rows
                            if np.isclose(row["mu_g_ei"], float(mu))
                        ]
                    )
                )
                for mu in mean_values
            ]
            plt.plot(
                mean_values,
                means,
                linewidth=1.8,
                color=colors[key],
                label=labels[key],
            )
        plt.xlabel("$\\mu_{g_{ei}}$")
        plt.ylabel("Metric value")
        plt.ylim(0.0, 1.0)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=False, fontsize=9)
        plt.tight_layout()

    save_both(out_path, plot_fn)


def plot_metric_std_vs_mu(
    mean_values: np.ndarray,
    rows: list[dict[str, float]],
    out_path: Path,
    *,
    title: str,
) -> None:
    def plot_fn() -> None:
        plt.figure(figsize=(6, 6))
        colors = {
            "autocorr_peak": "#1f77b4",
            "xcorr_peak": "#ff7f0e",
            "coherence_peak": "#9467bd",
            "lagged_lambda": "#2ca02c",
        }
        labels = {
            "autocorr_peak": "Autocorr peak std dev",
            "xcorr_peak": "Pairwise xcorr peak std dev",
            "coherence_peak": "Coherence peak std dev",
            "lagged_lambda": "Lagged coherence std dev",
        }
        for key in ["autocorr_peak", "xcorr_peak", "coherence_peak", "lagged_lambda"]:
            stds = []
            for mu in mean_values:
                points = [
                    row[key]
                    for row in rows
                    if np.isclose(row["mu_g_ei"], float(mu))
                ]
                stds.append(float(np.std(points)) if points else 0.0)
            plt.plot(
                mean_values,
                stds,
                linewidth=1.8,
                color=colors[key],
                marker="o",
                markersize=4,
                label=labels[key],
            )
        plt.xlabel("$\\mu_{g_{ei}}$")
        plt.ylabel("Std dev across seeds")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=False, fontsize=9)
        plt.tight_layout()

    save_both(out_path, plot_fn)
