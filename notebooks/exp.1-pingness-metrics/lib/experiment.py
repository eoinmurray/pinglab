from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from pinglab.analysis import (
    autocorr_peak,
    coherence_peak,
    lagged_coherence,
    lagged_coherence_spectrum,
    mean_firing_rates,
    mean_pairwise_xcorr_peak,
    population_rate,
    rate_psd,
)
from pinglab.inputs.tonic import tonic
from pinglab.lib.weights_builder import build_adjacency_matrices
from pinglab.run import build_model_from_config, run_network
from pinglab.types import Spikes
from pinglab.utils import slice_spikes

from .model import LocalConfig
from .plots import (
    plot_autocorr,
    plot_coherence,
    plot_lagged_coherence_spectrum,
    plot_lagged_windows,
    plot_metric_std_vs_mu,
    plot_metrics_vs_mu,
    plot_pairwise_xcorr,
    plot_population_rate,
    plot_rate_psd,
    plot_raster,
    plot_weight_histograms,
)


def _shift_spikes(spikes: Spikes, offset_ms: float) -> Spikes:
    return Spikes(
        times=spikes.times - offset_ms,
        ids=spikes.ids,
        types=spikes.types,
        populations=spikes.populations,
    )


def run_experiment(config: LocalConfig, data_path: Path) -> None:
    data_path.mkdir(parents=True, exist_ok=True)

    if config.weights is None:
        raise ValueError("weights must be provided in config for adjacency-only runs.")

    run_cfg = config.base
    scan_cfg = config.scan
    analysis_cfg = config.analysis
    mean_values = np.linspace(
        scan_cfg.mean_ei.start,
        scan_cfg.mean_ei.stop,
        scan_cfg.mean_ei.num,
    )
    num_steps = int(np.ceil(run_cfg.T / run_cfg.dt))

    start_ms = min(scan_cfg.burn_in_ms, max(0.0, run_cfg.T - run_cfg.dt))
    stop_ms = run_cfg.T
    analysis_T = stop_ms - start_ms

    print(
        f"[exp.1] scan mean_g_ei={scan_cfg.mean_ei.start:.3f}.."
        f"{scan_cfg.mean_ei.stop:.3f} ({scan_cfg.mean_ei.num} steps) | "
        f"std_g_ei={scan_cfg.std_ei:.3f} | "
        f"bin_ms={scan_cfg.bin_ms} burn_in_ms={scan_cfg.burn_in_ms}"
    )

    raster_indices: set[int] = set()
    if mean_values.size <= 3:
        raster_indices = set(range(mean_values.size))
    else:
        raster_indices = set(
            np.round(np.linspace(0, mean_values.size - 1, 3)).astype(int).tolist()
        )

    seed_count = 3
    seed_values = [int(run_cfg.seed or 0) + i for i in range(seed_count)]
    metric_rows: list[dict[str, float]] = []

    def _weights_with_ei_params(mean_ei_value: float, std_ei_value: float):
        weights = config.weights.model_copy(deep=True)
        params = dict(weights.ei.dist.params or {})
        params["mean"] = float(mean_ei_value)
        params["std"] = float(std_ei_value)
        weights.ei.dist.params = params
        return weights

    def run_single(mean_ei_value: float, std_ei_value: float, seed: int):
        weights = _weights_with_ei_params(mean_ei_value, std_ei_value)
        matrices = build_adjacency_matrices(
            N_E=run_cfg.N_E,
            N_I=run_cfg.N_I,
            ee=weights.ee,
            ei=weights.ei,
            ie=weights.ie,
            ii=weights.ii,
            clamp_min=weights.clamp_min,
            seed=int(run_cfg.seed or 0),
        )

        external_input = tonic(
            N_E=run_cfg.N_E,
            N_I=run_cfg.N_I,
            I_E=float(config.default_inputs.I_E),
            I_I=float(config.default_inputs.I_I),
            noise_std=float(config.default_inputs.noise),
            num_steps=num_steps,
            seed=seed,
        )
        seeded_cfg = run_cfg.model_copy(update={"seed": seed})
        neuron_model = build_model_from_config(seeded_cfg)
        result = run_network(
            seeded_cfg,
            external_input=external_input,
            model=neuron_model,
            weights=matrices.W,
        )
        return result

    def analyze_run(
        result,
        mean_ei: float,
        seed: int,
        idx: int,
        enable_plots: bool,
    ) -> tuple[float, float]:
        analysis_slice = slice_spikes(result.spikes, start_ms, stop_ms)
        shifted = _shift_spikes(analysis_slice, start_ms)
        mean_rate_E, mean_rate_I = mean_firing_rates(
            shifted, run_cfg.N_E, run_cfg.N_I
        )

        if config.plotting is not None:
            raster_start = min(config.plotting.raster.start_time, stop_ms)
            raster_stop = min(config.plotting.raster.stop_time, stop_ms)

        rel_slice = _shift_spikes(analysis_slice, start_ms)
        peak_value, _, _, lags_ms, corr = autocorr_peak(
            rel_slice,
            T_ms=analysis_T,
            dt_ms=scan_cfg.bin_ms,
            pop="E",
            N_E=run_cfg.N_E,
            N_I=run_cfg.N_I,
            smooth_sigma_ms=10.0,
            smooth_bin_ms=scan_cfg.bin_ms,
            autocorr_max_lag_ms=analysis_cfg.autocorr_max_lag_ms,
            corr_min_lag_ms=analysis_cfg.corr_min_lag_ms,
            corr_max_lag_ms=analysis_cfg.corr_max_lag_ms,
            corr_peak_min=analysis_cfg.corr_peak_min,
            corr_peak_prominence=analysis_cfg.corr_peak_prominence,
        )
        window = (analysis_cfg.corr_min_lag_ms, analysis_cfg.corr_max_lag_ms)
        mask = (lags_ms >= window[0]) & (lags_ms <= window[1])
        if np.any(mask):
            window_lags = lags_ms[mask]
            window_corr = corr[mask]
            matches = np.where(
                np.isclose(window_corr, peak_value, rtol=1e-6, atol=1e-6)
            )[0]
            if matches.size:
                peak_lag_ms = float(window_lags[int(matches[0])])
            else:
                peak_lag_ms = float(window_lags[int(np.argmax(window_corr))])
        else:
            peak_lag_ms = None

        xcorr_peak, xcorr_lags, xcorr_corr = mean_pairwise_xcorr_peak(
            rel_slice,
            T_ms=analysis_T,
            N_E=run_cfg.N_E,
            dt_ms=analysis_cfg.xcorr_bin_ms,
            xcorr_max_lag_ms=analysis_cfg.xcorr_max_lag_ms,
            corr_min_lag_ms=analysis_cfg.corr_min_lag_ms,
            corr_max_lag_ms=analysis_cfg.corr_max_lag_ms,
            corr_peak_min=analysis_cfg.corr_peak_min,
            corr_peak_prominence=analysis_cfg.corr_peak_prominence,
        )
        xcorr_window = (analysis_cfg.corr_min_lag_ms, analysis_cfg.corr_max_lag_ms)
        xcorr_mask = (xcorr_lags >= xcorr_window[0]) & (
            xcorr_lags <= xcorr_window[1]
        )
        if np.any(xcorr_mask):
            window_lags = xcorr_lags[xcorr_mask]
            window_corr = xcorr_corr[xcorr_mask]
            matches = np.where(
                np.isclose(window_corr, xcorr_peak, rtol=1e-6, atol=1e-6)
            )[0]
            if matches.size:
                xcorr_peak_lag = float(window_lags[int(matches[0])])
            else:
                xcorr_peak_lag = float(window_lags[int(np.argmax(window_corr))])
        else:
            xcorr_peak_lag = None

        coh_peak, coh_lags, coh_corr = coherence_peak(
            rel_slice,
            T_ms=analysis_T,
            dt_ms=analysis_cfg.xcorr_bin_ms,
            sigma_ms=10.0,
            max_lag_ms=analysis_cfg.xcorr_max_lag_ms,
            N_E=run_cfg.N_E,
        )
        if coh_corr.size:
            coh_peak_lag = float(coh_lags[int(np.argmax(coh_corr))])
        else:
            coh_peak_lag = None

        freqs = np.arange(2.0, 80.0 + 0.25, 0.5)
        freqs_out, lambda_vals, lag_t_ms, lag_rate = lagged_coherence_spectrum(
            rel_slice,
            T_ms=analysis_T,
            dt_ms=scan_cfg.bin_ms,
            freqs_hz=freqs,
            window_cycles=3.0,
            lag_cycles=3.0,
            pop="E",
            N_E=run_cfg.N_E,
            N_I=run_cfg.N_I,
            remove_mean=True,
            taper="hann",
        )
        if lambda_vals.size:
            peak_idx = int(np.argmax(lambda_vals))
            lagged_lambda = float(lambda_vals[peak_idx])
            lagged_freq = float(freqs_out[peak_idx])
        else:
            lagged_lambda = 0.0
            lagged_freq = 0.0
        lagged_lambda, lag_t_ms, lag_rate, windows, _, _ = lagged_coherence(
            rel_slice,
            T_ms=analysis_T,
            dt_ms=scan_cfg.bin_ms,
            freq_hz=lagged_freq if lagged_freq > 0 else 5.0,
            window_cycles=3.0,
            lag_cycles=3.0,
            pop="E",
            N_E=run_cfg.N_E,
            N_I=run_cfg.N_I,
            remove_mean=True,
            taper="hann",
        )

        metric_rows.append(
            {
                "mu_g_ei": float(mean_ei),
                "seed": float(seed),
                "autocorr_peak": float(peak_value),
                "xcorr_peak": float(xcorr_peak),
                "coherence_peak": float(coh_peak),
                "lagged_lambda": float(lagged_lambda),
                "lagged_freq_peak_hz": float(lagged_freq),
            }
        )

        if (
            enable_plots
            and config.plotting is not None
            and raster_start < raster_stop
            and (idx - 1) in raster_indices
        ):
            raster_slice = slice_spikes(result.spikes, raster_start, raster_stop)
            info_lines = [
                f"$\\mu_{{g_{{ei}}}}$: {mean_ei:.3f}",
                f"E rate: {mean_rate_E:.2f} Hz",
                f"I rate: {mean_rate_I:.2f} Hz",
                f"Autocorr peak: {peak_value:.3f}",
                f"Xcorr peak: {xcorr_peak:.3f}",
                f"Coherence peak: {coh_peak:.3f}",
                f"Lagged coh: {lagged_lambda:.3f}",
            ]
            plot_raster(
                raster_slice,
                data_path / f"raster_g_ei_mean_{idx:02d}_seed_{seed:02d}.png",
                label=(
                    f"$\\mu_{{g_{{ei}}}}={mean_ei:.3f}$, "
                    f"E rate={mean_rate_E:.2f} Hz, "
                    f"I rate={mean_rate_I:.2f} Hz"
                ),
                xlim=(raster_start, raster_stop),
                ylim=(0.0, float(run_cfg.N_E + run_cfg.N_I)),
                info_lines=info_lines,
            )
            t_ms, rate_hz = population_rate(
                rel_slice,
                T_ms=analysis_T,
                dt_ms=scan_cfg.bin_ms,
                pop="E",
                N_E=run_cfg.N_E,
                N_I=run_cfg.N_I,
            )
            plot_population_rate(
                t_ms,
                rate_hz,
                data_path / f"pop_rate_e_g_ei_mean_{idx:02d}_seed_{seed:02d}.png",
                title=f"$\\mu_{{g_{{ei}}}}={mean_ei:.3f}$ E rate",
            )
            _, smooth_rate = population_rate(
                rel_slice,
                T_ms=analysis_T,
                dt_ms=scan_cfg.bin_ms,
                pop="E",
                N_E=run_cfg.N_E,
                N_I=run_cfg.N_I,
                smooth_sigma_ms=10.0,
                smooth_bin_ms=scan_cfg.bin_ms,
            )
            psd_freqs, psd_vals = rate_psd(smooth_rate, dt_ms=scan_cfg.bin_ms)
            plot_rate_psd(
                psd_freqs,
                psd_vals,
                data_path / f"psd_e_g_ei_mean_{idx:02d}_seed_{seed:02d}.png",
                title=f"$\\mu_{{g_{{ei}}}}={mean_ei:.3f}$ smoothed E rate PSD",
            )
            plot_autocorr(
                lags_ms,
                corr,
                data_path / f"autocorr_e_g_ei_mean_{idx:02d}_seed_{seed:02d}.png",
                title=(
                    f"$\\mu_{{g_{{ei}}}}={mean_ei:.3f}$ "
                    f"E rate autocorr, peak={peak_value:.3f}"
                ),
                window=window,
                peak_lag_ms=peak_lag_ms,
                peak_value=peak_value if peak_lag_ms is not None else None,
            )
            plot_pairwise_xcorr(
                xcorr_lags,
                xcorr_corr,
                data_path / f"pairwise_xcorr_e_g_ei_mean_{idx:02d}_seed_{seed:02d}.png",
                title=(
                    f"$\\mu_{{g_{{ei}}}}={mean_ei:.3f}$ "
                    f"E mean pairwise xcorr, peak={xcorr_peak:.3f}"
                ),
                window=xcorr_window,
                peak_lag_ms=xcorr_peak_lag,
                peak_value=xcorr_peak if xcorr_peak_lag is not None else None,
            )
            plot_coherence(
                coh_lags,
                coh_corr,
                data_path / f"coherence_e_g_ei_mean_{idx:02d}_seed_{seed:02d}.png",
                title=f"$\\mu_{{g_{{ei}}}}={mean_ei:.3f}$ coherence",
                peak_lag_ms=coh_peak_lag,
                peak_value=coh_peak if coh_peak_lag is not None else None,
            )
            window_ms = (
                float(windows[0, 1] - windows[0, 0]) if windows.size else 0.0
            )
            plot_lagged_windows(
                lag_t_ms,
                lag_rate,
                window_ms,
                data_path / f"lagged_windows_e_g_ei_mean_{idx:02d}_seed_{seed:02d}.png",
                title=(
                    f"$\\mu_{{g_{{ei}}}}={mean_ei:.3f}$ "
                    f"lagged windows (\\lambda={lagged_lambda:.3f}, "
                    f"f={lagged_freq:.1f} Hz)"
                ),
            )
            plot_lagged_coherence_spectrum(
                freqs_out,
                lambda_vals,
                data_path
                / f"lagged_lambda_spectrum_e_g_ei_mean_{idx:02d}_seed_{seed:02d}.png",
                title=(
                    f"$\\mu_{{g_{{ei}}}}={mean_ei:.3f}$ "
                    f"lagged coherence spectrum"
                ),
            )
        return float(mean_rate_E), float(mean_rate_I)

        for idx, mean_ei in enumerate(mean_values, start=1):
        rate_e_vals = []
        rate_i_vals = []

        if config.plotting is not None:
            weights = _weights_with_ei_params(mean_ei, scan_cfg.std_ei)
            weights_for_hist = build_adjacency_matrices(
                N_E=run_cfg.N_E,
                N_I=run_cfg.N_I,
                ee=weights.ee,
                ei=weights.ei,
                ie=weights.ie,
                ii=weights.ii,
                clamp_min=weights.clamp_min,
                seed=int(run_cfg.seed or 0),
            )
            plot_weight_histograms(
                weights_for_hist.W_ee,
                weights_for_hist.W_ei,
                weights_for_hist.W_ie,
                weights_for_hist.W_ii,
                data_path / f"weights_hist_blocks_g_ei_mean_{idx:02d}.png",
                title=f"Weight histograms ($\\mu_{{g_{{ei}}}}={mean_ei:.3f}$)",
            )

        for seed in seed_values:
            result = run_single(mean_ei, scan_cfg.std_ei, seed)
            mean_rate_E, mean_rate_I = analyze_run(
                result,
                float(mean_ei),
                seed,
                idx,
                enable_plots=True,
            )
            rate_e_vals.append(mean_rate_E)
            rate_i_vals.append(mean_rate_I)

        mean_rate_E = float(np.mean(rate_e_vals)) if rate_e_vals else 0.0
        mean_rate_I = float(np.mean(rate_i_vals)) if rate_i_vals else 0.0

        print(
            f"[exp.1] g_ei_mean={mean_ei:.3f} | "
            f"rate_E={mean_rate_E:.2f} Hz | "
            f"rate_I={mean_rate_I:.2f} Hz"
        )

    metrics_csv = data_path / "metrics_by_seed.csv"
    with metrics_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mu_g_ei",
                "seed",
                "autocorr_peak",
                "xcorr_peak",
                "coherence_peak",
                "lagged_lambda",
                "lagged_freq_peak_hz",
            ],
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    plot_metrics_vs_mu(
        mean_values,
        metric_rows,
        data_path / "metrics_vs_mu.png",
        title="Metrics vs $\\mu_{g_{ei}}$",
    )
    plot_metric_std_vs_mu(
        mean_values,
        metric_rows,
        data_path / "metric_std_vs_mu.png",
        title="Metric std dev vs $\\mu_{g_{ei}}$",
    )
