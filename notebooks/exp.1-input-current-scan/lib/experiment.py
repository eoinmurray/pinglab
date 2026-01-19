import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d

from pinglab.analysis import mean_firing_rates, population_rate, rate_psd
from pinglab.inputs import tonic
from pinglab.lib.weights_builder import build_adjacency_matrices
from pinglab.multiprocessing import parallel
from pinglab.plots.styles import save_both, figsize
from pinglab.run import build_model_from_config, run_network
from pinglab.types import NetworkResult
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


def _hotloop(cfg: dict) -> NetworkResult:
    config: LocalConfig = cfg["config"]
    I_E = cfg["I_E"]
    I_I = cfg["I_I"]
    noise_std = cfg["noise_std"]

    run_cfg = config.base
    if config.weights is None:
        raise ValueError("weights must be provided for adjacency-only runs.")

    external_input = tonic(
        N_E=int(run_cfg.N_E),
        N_I=int(run_cfg.N_I),
        I_E=I_E,
        I_I=I_I,
        noise_std=noise_std,
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
        std_ei=config.weights.std_ei,
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
    result: NetworkResult = run_network(
        run_cfg,
        external_input=external_input,
        model=model,
        weights=matrices.W,
    )
    return result


def run_experiment(config: LocalConfig, data_path: Path) -> None:
    scan_cfg = config.input_scan
    values = np.linspace(
        scan_cfg.linspace.start,
        scan_cfg.linspace.stop,
        scan_cfg.linspace.num,
    )
    logger.info("input scan: %d values from %.3f to %.3f", values.size, values[0], values[-1])

    cfgs = []
    base_noise = float(config.default_inputs.noise)
    span = values[-1] - values[0] if values.size > 1 else 1.0
    for value in values:
        norm = (value - values[0]) / span if span > 0 else 0.0
        noise_std = base_noise * (1.5 - 0.9 * norm)
        noise_std = float(np.clip(noise_std, 0.3 * base_noise, 1.5 * base_noise))
        cfgs.append(
            {
                "config": config,
                "I_E": float(value),
                "I_I": scan_cfg.I_I,
                "noise_std": noise_std,
            }
        )

    results = parallel(_hotloop, cfgs, label="exp.5 input scan")

    firing_rates: list[tuple[float, float]] = []
    rhythmicity_vals: list[float] = []
    raster_metrics: list[dict[str, float]] = []

    for i, result in enumerate(results):
        I_E = values[i]
        logger.info(
            "plotting I_E=%.3f noise=%.3f (%d/%d)",
            I_E,
            cfgs[i]["noise_std"],
            i + 1,
            len(results),
        )
        sliced_spikes = slice_spikes(
            result.spikes,
            start_time=config.plotting.raster.start_time,
            stop_time=config.plotting.raster.stop_time,
        )
        raster_rates = mean_firing_rates(
            sliced_spikes,
            N_E=int(config.base.N_E),
            N_I=int(config.base.N_I),
        )
        raster_metrics.append(
            {
                "I_E": float(I_E),
                "rate_E_hz": float(raster_rates[0]),
                "rate_I_hz": float(raster_rates[1]),
            }
        )

        dt_ms = 1.0
        t_ms, rate_e = population_rate(
            result.spikes,
            config.base.T,
            dt_ms,
            pop="E",
            N_E=int(config.base.N_E),
            N_I=int(config.base.N_I),
        )
        _, rate_i = population_rate(
            result.spikes,
            config.base.T,
            dt_ms,
            pop="I",
            N_E=int(config.base.N_E),
            N_I=int(config.base.N_I),
        )
        mask_t = (t_ms >= config.plotting.raster.start_time) & (
            t_ms <= config.plotting.raster.stop_time
        )

        _, rate_hz = population_rate(
            sliced_spikes,
            config.base.T,
            dt_ms,
            pop="E",
            N_E=int(config.base.N_E),
            N_I=int(config.base.N_I),
        )
        rate_smooth = gaussian_filter1d(rate_hz, sigma=2)
        freqs, psd = rate_psd(rate_smooth, dt_ms)
        psd_mask = (freqs >= 5) & (freqs <= 150)

        def plot_raster_rates() -> None:
            psd_peak = float(np.max(psd[psd_mask])) if np.any(psd_mask) else 0.0
            fig, (ax_v, ax_rate, ax_ras, ax_psd) = plt.subplots(
                4,
                1,
                figsize=figsize,
                sharex=False,
                height_ratios=[1, 1, 2, 1],
                gridspec_kw={"hspace": 0.0},
            )
            ax_v.sharex(ax_ras)
            ax_rate.sharex(ax_ras)

            inst = result.instruments
            if inst.V is None:
                raise ValueError(
                    "V traces not recorded. Set instruments.variables to include 'V'."
                )
            v_times = inst.times
            v_traces = inst.V
            v_types = inst.types
            if v_types is None:
                raise ValueError("Instrument types missing for V traces.")
            mask_v = (v_times >= config.plotting.raster.start_time) & (
                v_times <= config.plotting.raster.stop_time
            )
            e_idx = int(np.where(v_types == 0)[0][0])
            i_idx = int(np.where(v_types == 1)[0][0])

            ax_v.plot(v_times[mask_v], v_traces[mask_v, e_idx], label="E", lw=0.8)
            ax_v.plot(
                v_times[mask_v],
                v_traces[mask_v, i_idx],
                label="I",
                lw=1.2,
                alpha=0.5,
            )
            ax_v.set_ylabel("V (mV)", labelpad=-4)
            ax_v.set_title(f"Single-neuron V(t) (I_E={I_E:.2f})", pad=1)
            ax_v.legend(loc="upper right", fontsize=8)
            ax_v.grid(False)
            ax_v.tick_params(labelbottom=False)
            ax_v.tick_params(axis="x", which="both", length=0)
            ax_v.spines["top"].set_visible(True)
            ax_v.spines["bottom"].set_visible(True)

            ax_rate.plot(t_ms[mask_t], rate_e[mask_t], label="E", lw=0.8)
            ax_rate.plot(t_ms[mask_t], rate_i[mask_t], label="I", lw=1.2, alpha=0.5)
            ax_rate.set_ylabel("Rate (Hz)", labelpad=-4)
            ax_rate.set_title("")
            ax_rate.grid(False)
            ax_rate.tick_params(labelbottom=False)
            ax_rate.tick_params(axis="x", which="both", length=0)
            ax_rate.spines["bottom"].set_visible(True)
            ax_rate.spines["top"].set_visible(True)

            times = sliced_spikes.times
            ids = sliced_spikes.ids
            types = getattr(sliced_spikes, "types", None)
            if types is not None:
                mask_E = types == 0
                mask_I = types == 1
                ax_ras.scatter(times[mask_E], ids[mask_E], s=0.5, marker=".")
                ax_ras.scatter(
                    times[mask_I], ids[mask_I], s=0.5, marker=".", alpha=0.7
                )
            else:
                ax_ras.scatter(times, ids, s=0.5, marker=".")

            ax_ras.set_xlabel("Time (ms)")
            ax_ras.set_ylabel("Neuron id", labelpad=-4)
            ax_ras.grid(False)
            ax_ras.spines["top"].set_visible(True)
            ax_ras.spines["bottom"].set_visible(True)

            if psd_peak > 0.0:
                ax_psd.plot(freqs[psd_mask], psd[psd_mask] / psd_peak)
            else:
                ax_psd.plot(freqs[psd_mask], psd[psd_mask])
            ax_psd.set_xlabel("Frequency (Hz)")
            ax_psd.set_ylabel("E PSD")
            ax_psd.set_title("")
            ax_psd.grid(True, alpha=0.3)
            fig.subplots_adjust(hspace=0.0, top=0.96, bottom=0.06)

            pos_v = ax_v.get_position()
            pos_rate = ax_rate.get_position()
            pos_ras = ax_ras.get_position()

            top = 0.955
            bottom = 0.33
            total_height = top - bottom
            h_v = total_height * 0.25
            h_rate = total_height * 0.25
            h_ras = total_height * 0.50

            overlap = 0.004
            ax_v.set_position([pos_v.x0, top - h_v, pos_v.width, h_v])
            ax_rate.set_position(
                [
                    pos_rate.x0,
                    top - h_v - h_rate + overlap,
                    pos_rate.width,
                    h_rate,
                ]
            )
            ax_ras.set_position(
                [pos_ras.x0, bottom + 2 * overlap, pos_ras.width, h_ras]
            )

            pos_psd = ax_psd.get_position()
            ax_psd.set_position([pos_psd.x0, 0.06, pos_psd.width, 0.18])

        save_both(
            data_path / f"input_scan_raster_rates_I_E_{i + 1:02d}",
            plot_raster_rates,
        )

        firing_rates.append(
            mean_firing_rates(
                result.spikes,
                N_E=int(config.base.N_E),
                N_I=int(config.base.N_I),
            )
        )

        rhythm_dt_ms = 5.0
        _, rhythm_rate = population_rate(
            sliced_spikes,
            config.base.T,
            rhythm_dt_ms,
            pop="E",
            N_E=int(config.base.N_E),
            N_I=int(config.base.N_I),
        )
        rhythmicity = _autocorr_rhythmicity(rhythm_rate, rhythm_dt_ms)
        rhythmicity_vals.append(rhythmicity)
        e_rate, i_rate = firing_rates[-1]
        if np.any(psd_mask):
            psd_slice = psd[psd_mask]
            freq_slice = freqs[psd_mask]
            peak_idx = int(np.argmax(psd_slice))
            peak_freq = float(freq_slice[peak_idx])
            peak_val = float(psd_slice[peak_idx])
        else:
            peak_freq = 0.0
            peak_val = 0.0
        logger.info(
            "metrics I_E=%.3f rate_E=%.2fHz rate_I=%.2fHz rhythmicity=%.3f psd_peak=%.2fHz",
            I_E,
            e_rate,
            i_rate,
            rhythmicity,
            peak_freq,
        )

        mask = psd_mask

        def plot_psd() -> None:
            psd_peak = float(np.max(psd[mask])) if np.any(mask) else 0.0
            _, ax = plt.subplots(1, 1, figsize=figsize)
            if psd_peak > 0.0:
                ax.plot(freqs[mask], psd[mask] / psd_peak)
            else:
                ax.plot(freqs[mask], psd[mask])
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power Spectral Density")
            ax.set_title(f"PSD of E population (I_E={I_E:.2f})")
            ax.grid(True)
            plt.tight_layout()

        save_both(
            data_path / f"input_scan_psd_I_E_{i + 1:02d}",
            plot_psd,
        )

    def plot_rates() -> None:
        e_rates = [entry["rate_E_hz"] for entry in raster_metrics]
        i_rates = [entry["rate_I_hz"] for entry in raster_metrics]

        _, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(values, e_rates, marker="o", label="E population")
        ax.plot(values, i_rates, marker="o", label="I population")
        ax.set_xlabel("I_E")
        ax.set_ylabel("Firing Rate (Hz)")
        ax.set_title("Firing rates vs I_E")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

    save_both(data_path / "input_scan_firing_rates", plot_rates)

    def plot_rhythmicity() -> None:
        _, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(values, rhythmicity_vals, marker="o")
        ax.set_xlabel("I_E")
        ax.set_ylabel("Rhythmicity (Autocorr Peak)")
        ax.set_title("Rhythmicity vs I_E")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True)
        plt.tight_layout()

    save_both(data_path / "input_scan_rhythmicity", plot_rhythmicity)

    metrics_path = data_path / "metrics.yaml"
    with metrics_path.open("w") as f:
        yaml.safe_dump(raster_metrics, f, sort_keys=False)
