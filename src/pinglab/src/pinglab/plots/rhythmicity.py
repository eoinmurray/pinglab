from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from matplotlib.ticker import FuncFormatter

from pinglab.analysis import population_rate
from pinglab.plots.styles import save_both


def _apply_monospace_font() -> None:
    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.monospace"] = [
        "Menlo",
        "Monaco",
        "DejaVu Sans Mono",
        "Courier New",
        "monospace",
    ]
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.titlesize"] = 21
    plt.rcParams["figure.titlesize"] = 23
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13
    plt.rcParams["legend.fontsize"] = 12


def _window_style_for_axis(ax) -> str:
    face = ax.get_facecolor()
    luminance = 0.2126 * face[0] + 0.7152 * face[1] + 0.0722 * face[2]
    if luminance < 0.5:
        return "#c7ced8"
    return "#8a93a1"


def normalized_e_rate_trace(*, spikes, T_ms: float, N_E: int) -> tuple[np.ndarray, np.ndarray]:
    t_ms, rate_hz = population_rate(
        spikes=spikes,
        T_ms=T_ms,
        dt_ms=5.0,
        pop="E",
        N_E=N_E,
        smooth_sigma_ms=None,
    )
    max_rate = float(np.max(rate_hz)) if rate_hz.size else 0.0
    if max_rate > 0.0:
        return t_ms, (rate_hz / max_rate)
    return t_ms, np.zeros_like(rate_hz, dtype=float)


def save_e_rate_plot(*, spikes, T_ms: float, N_E: int, parameter: str, scan_value: float, out_path: Path) -> None:
    t_ms, rate_plot = normalized_e_rate_trace(spikes=spikes, T_ms=T_ms, N_E=N_E)

    def _plot() -> None:
        _apply_monospace_font()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(t_ms, rate_plot, linewidth=1.5)
        ax.set_xlim(0.0, T_ms)
        ax.set_xlabel("Time (ms)")
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Norm E rate")
        ax.set_yticks([])
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.set_title(f"Norm E rate | {parameter}={scan_value:.6f}")
        ax.grid(alpha=0.25)
        fig.tight_layout()

    save_both(out_path, _plot)


def save_stacked_e_rates_plot(*, traces: list[np.ndarray], labels: list[str], t_ms: np.ndarray, parameter: str, out_path: Path) -> None:
    if not traces or t_ms.size == 0:
        return

    offset_step = 1.2

    def _plot() -> None:
        _apply_monospace_font()
        fig, ax = plt.subplots(figsize=(8, 8))
        trace_color = plt.rcParams.get("text.color", "#333333")
        for idx, (trace, label) in enumerate(zip(traces, labels)):
            offset = idx * offset_step
            ax.hlines(y=offset, xmin=0.0, xmax=float(t_ms[-1]), linewidth=0.8, linestyles="--", alpha=0.35, color=trace_color)
            ax.plot(t_ms, trace + offset, linewidth=1.1, color=trace_color)
            ax.text(t_ms[-1] * 1.005, offset + 0.5, label, fontsize=10, va="center", ha="left")
        ax.set_xlim(0.0, float(t_ms[-1]))
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Stacked norm E")
        ax.set_yticks([])
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.set_title(f"Stacked E rates | {parameter}")
        ax.grid(alpha=0.2)
        fig.tight_layout()

    save_both(out_path, _plot)


def save_scan_metrics_plot(*, scan_values: np.ndarray, metrics: dict[str, list[float]], parameter: str, out_path: Path) -> None:
    if scan_values.size == 0:
        return

    def _plot() -> None:
        _apply_monospace_font()
        fig, ax = plt.subplots(figsize=(8, 8))
        for metric_name, values in metrics.items():
            arr = np.array(values, dtype=float)
            ax.plot(scan_values, arr, marker="o", linewidth=1.6, markersize=4, label=metric_name)
        ax.set_xlabel(parameter)
        ax.set_ylabel("Metric")
        ax.set_title(f"Rhymicity metric vs {parameter}")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()

    save_both(out_path, _plot)


def save_total_e_spikes_vs_parameter_plot(*, scan_values: np.ndarray, total_e_spikes: list[int], parameter: str, out_path: Path) -> None:
    if scan_values.size == 0 or not total_e_spikes:
        return

    def _plot() -> None:
        _apply_monospace_font()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(scan_values, np.array(total_e_spikes, dtype=float), marker="o", linewidth=1.6, markersize=4)
        ax.set_xlabel(parameter)
        ax.set_ylabel("E spikes")
        ax.set_ylim(bottom=0.0)
        ax.set_title(f"E spikes vs {parameter}")
        ax.grid(alpha=0.25)
        fig.tight_layout()

    save_both(out_path, _plot)


def save_ee_autocorr_heatmap(*, mean_values: np.ndarray, std_values: np.ndarray, autocorr_matrix: np.ndarray, out_path: Path) -> None:
    if mean_values.size == 0 or std_values.size == 0:
        return

    def _plot() -> None:
        _apply_monospace_font()
        fig, ax = plt.subplots(figsize=(8, 8))
        image = ax.imshow(
            autocorr_matrix,
            aspect="auto",
            origin="lower",
            cmap="Reds",
            extent=(float(mean_values[0]), float(mean_values[-1]), float(std_values[0]), float(std_values[-1])),
        )
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("autocorr")
        ax.set_xlabel("EE mean")
        ax.set_ylabel("EE std")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.4g}"))
        ax.set_title("EE mean/std vs autocorr")
        fig.tight_layout()

    save_both(out_path, _plot)


def save_autocorr_curve_plot(*, lags_ms: np.ndarray, corr: np.ndarray, parameter: str, scan_value: float, peak_lag_ms: float, peak_value: float, out_path: Path) -> None:
    if lags_ms.size == 0 or corr.size == 0:
        return

    def _plot() -> None:
        _apply_monospace_font()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(lags_ms, corr, linewidth=1.4)
        ax.axvline(peak_lag_ms, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.scatter([peak_lag_ms], [peak_value], s=18, zorder=3)
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Autocorr")
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(f"Autocorr | {parameter}={scan_value:.6f}")
        ax.grid(alpha=0.25)
        fig.tight_layout()

    save_both(out_path, _plot)


def save_stacked_autocorr_curves_plot(
    *,
    lags_ms: np.ndarray,
    curves: list[np.ndarray],
    labels: list[str],
    peak_lags_ms: list[float],
    peak_values: list[float],
    processing_window_ms: tuple[float, float] | None = None,
    parameter: str,
    out_path: Path,
) -> None:
    if lags_ms.size == 0 or not curves:
        return

    def _plot() -> None:
        _apply_monospace_font()
        fig, ax = plt.subplots(figsize=(8, 8))
        trace_color = plt.rcParams.get("text.color", "#333333")
        offset_step = 2.2
        max_x = float(lags_ms[-1])
        text_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        for idx, (curve, label, peak_lag_ms, peak_value) in enumerate(zip(curves, labels, peak_lags_ms, peak_values)):
            offset = idx * offset_step
            if processing_window_ms is not None:
                window_color = _window_style_for_axis(ax)
                face = ax.get_facecolor()
                luminance = 0.2126 * face[0] + 0.7152 * face[1] + 0.0722 * face[2]
                fill_alpha = 0.008 if luminance < 0.5 else 0.01
                ax.axvspan(float(processing_window_ms[0]), float(processing_window_ms[1]), facecolor=window_color, edgecolor="none", alpha=fill_alpha)
            ax.hlines(y=offset, xmin=float(lags_ms[0]), xmax=float(lags_ms[-1]), linewidth=0.8, linestyles="--", alpha=0.35, color=trace_color)
            ax.plot(lags_ms, curve + offset, linewidth=1.0, color=trace_color)
            ax.scatter([peak_lag_ms], [peak_value + offset], s=16, zorder=3, color=trace_color)
            ax.axvline(peak_lag_ms, ymin=0.0, ymax=1.0, linewidth=0.6, linestyle=":", alpha=0.15, color=trace_color)
            ax.text(1.03, offset, label, transform=text_transform, fontsize=10, va="center", ha="left", clip_on=False)
        min_y = -1.2
        max_y = (len(curves) - 1) * offset_step + 1.2
        ax.set_ylim(min_y, max_y)
        ax.set_xlim(float(lags_ms[0]), max_x)
        ax.set_title(f"Stacked autocorr | {parameter}")
        ax.set_xlabel("Lag (ms)")
        ax.set_ylabel("Stacked autocorr")
        ax.set_yticks([])
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax.grid(alpha=0.2)
        fig.tight_layout()

    save_both(out_path, _plot)


def save_ei_neuron_state_plot(
    *,
    t_ms: np.ndarray,
    V_e: np.ndarray,
    V_i: np.ndarray,
    I_syn_e: np.ndarray,
    I_syn_i: np.ndarray,
    parameter: str,
    scan_value: float,
    e_id: int,
    i_id: int,
    out_path: Path,
) -> None:
    if t_ms.size == 0:
        return

    def _plot() -> None:
        _apply_monospace_font()
        fig, (ax_v, ax_i) = plt.subplots(
            2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.0]}
        )

        color_main = plt.rcParams.get("text.color", "#333333")
        color_e = color_main
        color_i = "#FF3B30"

        ax_v.plot(t_ms, V_e, linewidth=1.2, color=color_e, label=f"E[{e_id}]")
        ax_v.plot(t_ms, V_i, linewidth=1.2, color=color_i, label=f"I[{i_id}]")
        ax_v.set_ylabel("V (mV)")
        ax_v.set_title(f"Neuron states | {parameter}={scan_value:.6f}")
        ax_v.grid(alpha=0.22)
        ax_v.legend(loc="upper right")

        ax_i.plot(t_ms, I_syn_e, linewidth=1.3, color=color_e, alpha=0.95, linestyle="-", label="E syn")
        ax_i.plot(t_ms, I_syn_i, linewidth=1.3, color=color_i, alpha=0.95, linestyle="-", label="I syn")
        ax_i.axhline(0.0, linewidth=0.8, color=color_main, alpha=0.25)
        ax_i.set_ylabel("I (a.u.)")
        ax_i.set_xlabel("Time (ms)")
        ax_i.grid(alpha=0.22)
        ax_i.legend(loc="upper right", ncol=1)

        fig.tight_layout()

    save_both(out_path, _plot)
