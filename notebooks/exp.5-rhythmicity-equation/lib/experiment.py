from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pinglab.analysis import population_mean_rate, population_rate, rate_psd
from pinglab.inputs import tonic
from pinglab.plots.raster import save_raster
from pinglab.plots.styles import save_both, figsize
from pinglab.run import run_network
from pinglab.types import Spikes
from pinglab.utils import slice_spikes

from .model import LocalConfig, SweepConfig, SweepOverrides


def _shift_spikes(spikes: Spikes, offset_ms: float) -> Spikes:
    return Spikes(
        times=spikes.times - offset_ms,
        ids=spikes.ids,
        types=spikes.types,
        populations=spikes.populations,
    )


def _autocorr_rhythmicity(
    rate_hz: np.ndarray,
    dt_ms: float,
    tau_min_ms: float,
    tau_max_ms: float,
) -> tuple[float, float]:
    if rate_hz.size == 0:
        return 0.0, 0.0

    mean = float(np.mean(rate_hz))
    std = float(np.std(rate_hz))
    if std == 0.0:
        return 0.0, 0.0

    x = (rate_hz - mean) / std
    n = x.size
    corr = np.correlate(x, x, mode="full")[n - 1 :]
    norm = np.arange(n, 0, -1, dtype=float)
    C = corr / norm

    lag_min = max(1, int(np.ceil(tau_min_ms / dt_ms)))
    lag_max = min(n - 1, int(np.floor(tau_max_ms / dt_ms)))
    if lag_max < lag_min:
        return 0.0, 0.0

    window = C[lag_min : lag_max + 1]
    peak_idx = int(np.argmax(window))
    lag_idx = lag_min + peak_idx
    tau_star = lag_idx * dt_ms
    rho = float(C[lag_idx])
    return rho, tau_star


def _autocorr_curve(rate_hz: np.ndarray, dt_ms: float) -> tuple[np.ndarray, np.ndarray]:
    if rate_hz.size == 0:
        return np.array([]), np.array([])

    mean = float(np.mean(rate_hz))
    std = float(np.std(rate_hz))
    if std == 0.0:
        return np.array([]), np.array([])

    x = (rate_hz - mean) / std
    n = x.size
    corr = np.correlate(x, x, mode="full")[n - 1 :]
    norm = np.arange(n, 0, -1, dtype=float)
    C = corr / norm
    lags_ms = np.arange(n, dtype=float) * dt_ms
    return lags_ms, C


def _resolve_sweep(defaults: SweepConfig, overrides: SweepOverrides | None) -> SweepConfig:
    if overrides is None:
        return defaults
    return SweepConfig(
        I_E=overrides.I_E or defaults.I_E,
        bin_ms=overrides.bin_ms if overrides.bin_ms is not None else defaults.bin_ms,
        burn_in_ms=overrides.burn_in_ms if overrides.burn_in_ms is not None else defaults.burn_in_ms,
        tau_min_ms=overrides.tau_min_ms if overrides.tau_min_ms is not None else defaults.tau_min_ms,
        tau_max_ms=overrides.tau_max_ms if overrides.tau_max_ms is not None else defaults.tau_max_ms,
    )


def run_experiment(config: LocalConfig, data_path: Path) -> None:
    data_path.mkdir(parents=True, exist_ok=True)

    combined_rates: dict[str, tuple[np.ndarray, list[float]]] = {}
    combined_rhythmicity: dict[str, tuple[np.ndarray, list[float]]] = {}
    combined_psd_peaks: dict[str, tuple[np.ndarray, list[float]]] = {}

    for model in config.models:
        model_path = data_path / model.name
        model_path.mkdir(parents=True, exist_ok=True)

        run_cfg = config.base.model_copy(
            update={"neuron_model": model.neuron_model, **model.overrides}
        )
        sweep_cfg = _resolve_sweep(config.sweep, model.sweep)
        sweep = np.linspace(
            sweep_cfg.I_E.start,
            sweep_cfg.I_E.stop,
            sweep_cfg.I_E.num,
        )
        num_steps = int(np.ceil(run_cfg.T / run_cfg.dt))

        rates_E: list[float] = []
        rates_I: list[float] = []
        spikes_E_counts: list[int] = []
        spikes_I_counts: list[int] = []
        rhythmicity: list[float] = []
        psd_peak_freqs: list[float] = []
        rate_series: list[tuple[float, np.ndarray, np.ndarray]] = []
        autocorr_series: list[tuple[float, np.ndarray, np.ndarray]] = []
        psd_series: list[tuple[float, np.ndarray, np.ndarray]] = []

        start_ms = min(sweep_cfg.burn_in_ms, max(0.0, run_cfg.T - run_cfg.dt))
        stop_ms = run_cfg.T
        analysis_T = stop_ms - start_ms

        print(
            f"[exp.5][{model.name}] sweep I_E={sweep_cfg.I_E.start:.2f}.."
            f"{sweep_cfg.I_E.stop:.2f} ({sweep_cfg.I_E.num} steps) | "
            f"bin_ms={sweep_cfg.bin_ms} burn_in_ms={sweep_cfg.burn_in_ms}"
        )

        for I_E in sweep:
            external_input = tonic(
                N_E=run_cfg.N_E,
                N_I=run_cfg.N_I,
                I_E=float(I_E) * model.input_scale_E,
                I_I=config.default_inputs.I_I * model.input_scale_I,
                noise_std=config.default_inputs.noise,
                num_steps=num_steps,
                seed=run_cfg.seed or 0,
            )
            result = run_network(run_cfg, external_input=external_input)

            sliced = slice_spikes(result.spikes, start_ms, stop_ms)
            shifted = _shift_spikes(sliced, start_ms)

            mean_rate_E, mean_rate_I = population_mean_rate(
                shifted, analysis_T, run_cfg.N_E, run_cfg.N_I
            )
            rates_E.append(float(mean_rate_E))
            rates_I.append(float(mean_rate_I))

            spikes_E = int(np.sum(shifted.ids < run_cfg.N_E))
            spikes_I = int(np.sum(shifted.ids >= run_cfg.N_E))
            spikes_E_counts.append(spikes_E)
            spikes_I_counts.append(spikes_I)

            t_ms, rate_hz = population_rate(
                shifted,
                T_ms=analysis_T,
                dt_ms=sweep_cfg.bin_ms,
                pop="E",
                N_E=run_cfg.N_E,
                N_I=run_cfg.N_I,
            )
            rho, _ = _autocorr_rhythmicity(
                rate_hz,
                dt_ms=sweep_cfg.bin_ms,
                tau_min_ms=sweep_cfg.tau_min_ms,
                tau_max_ms=sweep_cfg.tau_max_ms,
            )
            rhythmicity.append(float(rho))

            if config.plotting is not None:
                raster_start = min(config.plotting.raster.start_time, stop_ms)
                raster_stop = min(config.plotting.raster.stop_time, stop_ms)
                if raster_start < raster_stop:
                    raster_slice = slice_spikes(result.spikes, raster_start, raster_stop)
                    save_raster(
                        raster_slice,
                        model_path / f"raster_I_E_{I_E:.2f}.png",
                        label=f"{model.name} I_E={I_E:.2f}",
                        xlim=(raster_start, raster_stop),
                        ylim=(0.0, float(run_cfg.N_E + run_cfg.N_I)),
                    )

            freqs, psd = rate_psd(rate_hz, dt_ms=sweep_cfg.bin_ms)
            rate_series.append((float(I_E), t_ms, rate_hz))

            lags_ms, C = _autocorr_curve(rate_hz, sweep_cfg.bin_ms)
            autocorr_series.append((float(I_E), lags_ms, C))
            psd_series.append((float(I_E), freqs, psd))

            if psd.size > 0:
                peak_idx = int(np.argmax(psd))
                peak_freq = float(freqs[peak_idx])
                peak_power = float(psd[peak_idx])
            else:
                peak_freq = 0.0
                peak_power = 0.0
            psd_peak_freqs.append(peak_freq)

            print(
                f"[exp.5][{model.name}] I_E={I_E:.2f} | "
                f"rate_E={mean_rate_E:.2f} Hz | rate_I={mean_rate_I:.2f} Hz | "
                f"spikes_E={spikes_E} spikes_I={spikes_I} | rho={rho:.3f} | "
                f"psd_peak={peak_freq:.1f} Hz ({peak_power:.3g})"
            )

        def plot_rate_curve() -> None:
            plt.figure(figsize=figsize)
            plt.plot(sweep, rates_E, "o-", linewidth=1.5)
            plt.xlabel("Tonic Input $I_E$")
            plt.ylabel("E Population Firing Rate (Hz)")
            plt.title(f"Firing Rate vs Input ({model.name})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

        save_both(model_path / "firing_rate_vs_I_E.png", plot_rate_curve)

        def plot_rhythmicity_curve() -> None:
            plt.figure(figsize=figsize)
            plt.plot(sweep, rhythmicity, "o-", linewidth=1.5)
            plt.xlabel("Tonic Input $I_E$")
            plt.ylabel("Rhythmicity (Autocorr Peak)")
            plt.title(f"Rhythmicity vs Input ({model.name})")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

        save_both(model_path / "rhythmicity_vs_I_E.png", plot_rhythmicity_curve)

        combined_rates[model.name] = (sweep, rates_E)
        combined_rhythmicity[model.name] = (sweep, rhythmicity)
        combined_psd_peaks[model.name] = (sweep, psd_peak_freqs)

        metrics_table = np.column_stack(
            [
                sweep,
                np.array(rates_E, dtype=float),
                np.array(rates_I, dtype=float),
                np.array(spikes_E_counts, dtype=int),
                np.array(spikes_I_counts, dtype=int),
                np.array(rhythmicity, dtype=float),
            ]
        )
        np.savetxt(
            model_path / "summary_metrics.csv",
            metrics_table,
            delimiter=",",
            header="I_E,rate_E_hz,rate_I_hz,spikes_E,spikes_I,rhythmicity",
            comments="",
        )

        total_spikes_E = int(np.sum(spikes_E_counts))
        total_spikes_I = int(np.sum(spikes_I_counts))
        print(
            f"[exp.5][{model.name}] totals | "
            f"spikes_E={total_spikes_E} spikes_I={total_spikes_I}"
        )
        totals_table = np.array([[total_spikes_E, total_spikes_I]], dtype=int)
        np.savetxt(
            model_path / "summary_totals.csv",
            totals_table,
            delimiter=",",
            header="total_spikes_E,total_spikes_I",
            comments="",
            fmt="%d",
        )

        if rate_series:
            max_rate = max(float(np.max(rate)) for _, _, rate in rate_series)
            for I_E, t_ms, rate_hz in rate_series:
                def plot_rate() -> None:
                    plt.figure(figsize=figsize)
                    plt.plot(t_ms, rate_hz, linewidth=1.2)
                    plt.ylim(0.0, max_rate * 1.05 if max_rate > 0 else 1.0)
                    plt.xlabel("Time (ms)")
                    plt.ylabel("E Population Rate (Hz)")
                    plt.title(f"Population Rate r(t) {model.name} I_E={I_E:.2f}")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

                save_both(model_path / f"rate_I_E_{I_E:.2f}.png", plot_rate)

        if autocorr_series:
            valid_autocorr = [C for _, _, C in autocorr_series if C.size > 0]
            if valid_autocorr:
                min_auto = min(float(np.min(C)) for C in valid_autocorr)
                max_auto = max(float(np.max(C)) for C in valid_autocorr)
                if min_auto == max_auto:
                    min_auto -= 0.05
                    max_auto += 0.05
                for I_E, lags_ms, C in autocorr_series:
                    def plot_autocorr() -> None:
                        plt.figure(figsize=figsize)
                        plt.plot(lags_ms, C, linewidth=1.2)
                        plt.ylim(min_auto, max_auto)
                        plt.xlabel("Lag (ms)")
                        plt.ylabel("Autocorrelation")
                        plt.title(f"Autocorrelation C(τ) {model.name} I_E={I_E:.2f}")
                        plt.grid(True, alpha=0.3)
                        if C.size > 0:
                            tau_min = sweep_cfg.tau_min_ms
                            tau_max = sweep_cfg.tau_max_ms
                            plt.axvspan(tau_min, tau_max, color="tab:blue", alpha=0.12)
                            plt.axvline(tau_min, color="tab:blue", linewidth=1.0, alpha=0.6)
                            plt.axvline(tau_max, color="tab:blue", linewidth=1.0, alpha=0.6)
                            mask = (lags_ms >= tau_min) & (lags_ms <= tau_max)
                            if np.any(mask):
                                window_lags = lags_ms[mask]
                                window_vals = C[mask]
                                peak_idx = int(np.argmax(window_vals))
                                peak_lag = float(window_lags[peak_idx])
                                peak_val = float(window_vals[peak_idx])
                                plt.plot(peak_lag, peak_val, "o", color="tab:red", markersize=5)
                                plt.text(
                                    peak_lag,
                                    peak_val,
                                    f"  ρ={peak_val:.2f}",
                                    color="tab:red",
                                    fontsize=9,
                                    va="center",
                                )
                        plt.tight_layout()

                    save_both(model_path / f"autocorr_I_E_{I_E:.2f}.png", plot_autocorr)

        if psd_series:
            max_psd = max(float(np.max(psd)) for _, _, psd in psd_series)
            for idx, (I_E, freqs, psd) in enumerate(psd_series):
                def plot_psd() -> None:
                    plt.figure(figsize=figsize)
                    plt.plot(freqs, psd)
                    plt.ylim(0.0, max_psd * 1.05 if max_psd > 0 else 1.0)
                    plt.xlabel("Frequency (Hz)")
                    plt.ylabel("Power Spectral Density")
                    plt.title(f"Population Rate PSD (E) {model.name} I_E={I_E:.2f}")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()

                save_both(model_path / f"psd_I_E_{idx:02d}_{I_E:.2f}.png", plot_psd)

        print(f"[exp.5][{model.name}] done")

    if combined_rates:
        def plot_combined_rates() -> None:
            plt.figure(figsize=figsize)
            for name, (xvals, yvals) in combined_rates.items():
                plt.plot(xvals, yvals, "o-", linewidth=1.5, label=name)
            plt.xlabel("Tonic Input $I_E$")
            plt.ylabel("E Population Firing Rate (Hz)")
            plt.title("Firing Rate vs Input (All Models)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

        save_both(data_path / "compare_firing_rate_vs_I_E.png", plot_combined_rates)

    if combined_rhythmicity:
        def plot_combined_rhythmicity() -> None:
            plt.figure(figsize=figsize)
            for name, (xvals, yvals) in combined_rhythmicity.items():
                plt.plot(xvals, yvals, "o-", linewidth=1.5, label=name)
            plt.xlabel("Tonic Input $I_E$")
            plt.ylabel("Rhythmicity (Autocorr Peak)")
            plt.title("Rhythmicity vs Input (All Models)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

        save_both(data_path / "compare_rhythmicity_vs_I_E.png", plot_combined_rhythmicity)

    if combined_psd_peaks:
        def plot_combined_psd_peaks() -> None:
            plt.figure(figsize=figsize)
            for name, (xvals, yvals) in combined_psd_peaks.items():
                plt.plot(xvals, yvals, "o-", linewidth=1.5, label=name)
            plt.xlabel("Tonic Input $I_E$")
            plt.ylabel("PSD Peak Frequency (Hz)")
            plt.title("PING Rhythm Frequency vs Input (All Models)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()

        save_both(data_path / "compare_psd_peak_vs_I_E.png", plot_combined_psd_peaks)
