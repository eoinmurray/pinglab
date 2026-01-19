import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pinglab.analysis import population_plv, population_rate, rate_psd
from pinglab.inputs import tonic
from pinglab.lib.weights_builder import build_adjacency_matrices
from pinglab.plots.styles import save_both, figsize
from pinglab.run import build_model_from_config, run_network
from pinglab.utils import slice_spikes

from .model import LocalConfig

logger = logging.getLogger(__name__)


def _autocorr_rhythmicity(
    rate_hz: np.ndarray,
    dt_ms: float,
    tau_min_ms: float = 5.0,
    tau_max_ms: float = 200.0,
) -> float:
    if rate_hz.size == 0:
        return 0.0
    mean = float(np.mean(rate_hz))
    std = float(np.std(rate_hz))
    if std == 0.0:
        return 0.0

    x = (rate_hz - mean) / std
    n = x.size
    corr = np.correlate(x, x, mode="full")[n - 1 :]
    norm = np.arange(n, 0, -1, dtype=float)
    C = corr / norm

    lag_min = max(1, int(np.ceil(tau_min_ms / dt_ms)))
    lag_max = min(n - 1, int(np.floor(tau_max_ms / dt_ms)))
    if lag_max < lag_min:
        return 0.0

    window = C[lag_min : lag_max + 1]
    peak_idx = int(np.argmax(window))
    lag_idx = lag_min + peak_idx
    return float(C[lag_idx])


def _shift_spikes(spikes, offset_ms: float):
    return spikes.__class__(
        times=spikes.times - offset_ms,
        ids=spikes.ids,
        types=spikes.types,
        populations=spikes.populations,
    )


def run_experiment(config: LocalConfig, data_path: Path) -> None:
    run_cfg = config.base
    if config.weights is None:
        raise ValueError("weights must be provided for this experiment.")

    scan_cfg = config.scan.std_ei
    std_values = np.linspace(
        scan_cfg.start,
        scan_cfg.stop,
        scan_cfg.num,
    )
    logger.info("std_ei scan: %d values from %.3f to %.3f", std_values.size, std_values[0], std_values[-1])

    rhythmicity_vals: list[float] = []
    gamma_snr_vals: list[float] = []
    plv_vals: list[float] = []
    for idx, std_ei in enumerate(std_values, start=1):
        logger.info("running std_ei=%.4f (%d/%d)", std_ei, idx, std_values.size)
        external_input = tonic(
            N_E=int(run_cfg.N_E),
            N_I=int(run_cfg.N_I),
            I_E=float(config.default_inputs.I_E),
            I_I=float(config.default_inputs.I_I),
            noise_std=float(config.default_inputs.noise),
            num_steps=int(np.ceil(run_cfg.T / run_cfg.dt)),
            seed=run_cfg.seed if run_cfg.seed is not None else 0,
        )

        matrices = build_adjacency_matrices(
            N_E=run_cfg.N_E,
            N_I=run_cfg.N_I,
            mean_ee=config.weights.mean_ee,
            mean_ei=config.weights.mean_ei,
            mean_ie=config.weights.mean_ie,
            mean_ii=config.weights.mean_ii,
            std_ee=config.weights.std_ee,
            std_ei=float(std_ei),
            std_ie=config.weights.std_ie,
            std_ii=config.weights.std_ii,
            p_ee=config.weights.p_ee,
            p_ei=config.weights.p_ei,
            p_ie=config.weights.p_ie,
            p_ii=config.weights.p_ii,
            clamp_min=config.weights.clamp_min,
            seed=run_cfg.seed,
        )

        model = build_model_from_config(run_cfg)
        result = run_network(
            run_cfg,
            external_input=external_input,
            model=model,
            weights=matrices.W,
        )

        start_time = float(config.plotting.raster.start_time)
        stop_time = float(config.plotting.raster.stop_time)
        sliced = slice_spikes(result.spikes, start_time=start_time, stop_time=stop_time)

        dt_ms = 5.0
        _, rate_hz = population_rate(
            sliced,
            T_ms=run_cfg.T,
            dt_ms=dt_ms,
            pop="E",
            N_E=int(run_cfg.N_E),
            N_I=int(run_cfg.N_I),
        )
        rhythmicity_vals.append(_autocorr_rhythmicity(rate_hz, dt_ms))

        rate_centered = rate_hz - float(np.mean(rate_hz)) if rate_hz.size else rate_hz
        freqs, psd = rate_psd(rate_centered, dt_ms)
        band_mask = (freqs >= 30.0) & (freqs <= 100.0)
        total_mask = (freqs > 0.0) & (freqs <= 200.0)
        band_power = float(np.trapz(psd[band_mask], freqs[band_mask])) if np.any(band_mask) else 0.0
        total_power = float(np.trapz(psd[total_mask], freqs[total_mask])) if np.any(total_mask) else 0.0
        gamma_snr = band_power / (total_power - band_power) if total_power > band_power else 0.0
        gamma_snr_vals.append(gamma_snr)

        shifted = _shift_spikes(sliced, start_time)
        analysis_T = stop_time - start_time
        plv = population_plv(
            spikes=shifted,
            T_ms=analysis_T,
            dt_ms=5.0,
            fmin=30.0,
            fmax=90.0,
            pop="E",
            N_E=int(run_cfg.N_E),
            N_I=int(run_cfg.N_I),
        )
        plv_vals.append(float(plv))

        label = f"std_ei={std_ei:.3f} (g_ei mean={config.weights.mean_ei:.3f})"

        def plot_raster_with_psd() -> None:
            fig, (ax_ras, ax_psd) = plt.subplots(
                2,
                1,
                figsize=figsize,
                height_ratios=[3, 1],
                gridspec_kw={"hspace": 0.2},
                constrained_layout=True,
            )

            times = sliced.times
            ids = sliced.ids
            types = getattr(sliced, "types", None)
            if types is not None:
                mask_E = types == 0
                mask_I = types == 1
                ax_ras.scatter(times[mask_E], ids[mask_E], s=0.5, marker=".")
                ax_ras.scatter(times[mask_I], ids[mask_I], s=0.5, marker=".", alpha=0.7)
            else:
                ax_ras.scatter(times, ids, s=0.5, marker=".")

            ax_ras.set_xlim(start_time, stop_time)
            ax_ras.set_ylim(0.0, float(run_cfg.N_E + run_cfg.N_I))
            ax_ras.set_xlabel("Time (ms)")
            ax_ras.set_ylabel("Neuron id")
            ax_ras.set_title(label)

            rate_centered = rate_hz - float(np.mean(rate_hz)) if rate_hz.size else rate_hz
            freqs, psd = rate_psd(rate_centered, dt_ms)
            psd_mask = (freqs >= 5.0) & (freqs <= 150.0)
            ax_psd.plot(freqs[psd_mask], psd[psd_mask])
            ax_psd.set_xlabel("Frequency (Hz)")
            ax_psd.set_ylabel("PSD")
            ax_psd.grid(True, alpha=0.3)

        save_both(
            data_path / f"raster_std_ei_{idx:02d}",
            plot_raster_with_psd,
        )

    def plot_rhythmicity() -> None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(std_values, rhythmicity_vals, marker="o")
        ax.set_xlabel("std_g_ei")
        ax.set_ylabel("Rhythmicity (Autocorr Peak)")
        ax.set_title("Rhythmicity vs g_ei stddev")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True)
        plt.tight_layout()

    save_both(data_path / "rhythmicity_vs_g_ei_stddev", plot_rhythmicity)

    def plot_gamma_snr() -> None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(std_values, gamma_snr_vals, marker="o")
        ax.set_xlabel("std_g_ei")
        ax.set_ylabel("Gamma SNR")
        ax.set_title("Gamma SNR vs g_ei stddev")
        ax.grid(True)
        plt.tight_layout()

    save_both(data_path / "gamma_snr_vs_g_ei_stddev", plot_gamma_snr)

    def plot_plv() -> None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(std_values, plv_vals, marker="o")
        ax.set_xlabel("std_g_ei")
        ax.set_ylabel("PLV")
        ax.set_title("PLV vs g_ei stddev")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True)
        plt.tight_layout()

    save_both(data_path / "plv_vs_g_ei_stddev", plot_plv)
