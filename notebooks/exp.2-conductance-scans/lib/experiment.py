from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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
    g_ei = cfg["g_ei"]
    g_ie = cfg["g_ie"]
    g_ee = cfg["g_ee"]
    g_ii = cfg["g_ii"]
    std_ee = cfg.get("std_ee", config.weights.std_ee)
    I_E = cfg["I_E"]

    run_cfg = config.base
    if config.weights is None:
        raise ValueError("weights must be provided for adjacency-only runs.")

    external_input = tonic(
        N_E=int(run_cfg.N_E),
        N_I=int(run_cfg.N_I),
        I_E=I_E,
        I_I=config.default_inputs.I_I,
        noise_std=config.default_inputs.noise,
        num_steps=int(np.ceil(run_cfg.T / run_cfg.dt)),
        seed=run_cfg.seed if run_cfg.seed is not None else 0,
    )

    matrices = build_adjacency_matrices(
        N_E=run_cfg.N_E,
        N_I=run_cfg.N_I,
        mean_ee=float(g_ee),
        mean_ei=float(g_ei),
        mean_ie=float(g_ie),
        mean_ii=float(g_ii),
        std_ee=float(std_ee),
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
    scans = [
        ("gei", config.gei_scan, "g_ei"),
        ("gie", config.gie_scan, "g_ie"),
        ("gee", config.gee_scan, "g_ee"),
        ("gii", config.gii_scan, "g_ii"),
    ]

    for scan_name, scan_cfg, scan_label in scans:
        values = np.linspace(
            scan_cfg.linspace.start,
            scan_cfg.linspace.stop,
            scan_cfg.linspace.num,
        )

        cfgs = []
        for value in values:
            cfgs.append(
                {
                    "config": config,
                    "g_ee": value if scan_label == "g_ee" else config.weights.mean_ee,
                    "g_ei": value if scan_label == "g_ei" else config.weights.mean_ei,
                    "g_ie": value if scan_label == "g_ie" else config.weights.mean_ie,
                    "g_ii": value if scan_label == "g_ii" else config.weights.mean_ii,
                    "I_E": scan_cfg.I_E,
                }
            )

        results = parallel(_hotloop, cfgs, label=f"exp.1 {scan_name} scan")

        firing_rates: list[tuple[float, float]] = []
        rhythmicity_vals: list[float] = []

        for i, result in enumerate(results):
            sliced_spikes = slice_spikes(
                result.spikes,
                start_time=config.plotting.raster.start_time,
                stop_time=config.plotting.raster.stop_time,
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
                ax_v.set_title(
                    f"Single-neuron V(t) ({scan_label}={values[i]:.2f})",
                    pad=1,
                )
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
                # keep legend only on voltage plot for visual clarity
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
                    ax_ras.scatter(
                        times[mask_E], ids[mask_E], s=0.5, marker=".", label="E"
                    )
                    ax_ras.scatter(
                        times[mask_I],
                        ids[mask_I],
                        s=0.5,
                        marker=".",
                        alpha=0.7,
                        label="I",
                    )
                    # no legend on raster; voltage plot has the only legend
                else:
                    ax_ras.scatter(times, ids, s=0.5, marker=".")

                ax_ras.set_xlabel("Time (ms)")
                ax_ras.set_ylabel("Neuron id", labelpad=-4)
                ax_ras.grid(False)
                ax_ras.spines["top"].set_visible(True)
                ax_ras.spines["bottom"].set_visible(True)

                ax_psd.plot(freqs[psd_mask], psd[psd_mask] / np.max(psd[psd_mask]))
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
                ax_psd.set_position(
                    [
                        pos_psd.x0,
                        0.06,
                        pos_psd.width,
                        0.18,
                    ]
                )

            save_both(
                data_path
                / f"{scan_name}_scan_raster_rates_{scan_label}_{i + 1:02d}",
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
            rhythmicity_vals.append(
                _autocorr_rhythmicity(rhythm_rate, rhythm_dt_ms)
            )

            mask = psd_mask

            def plot_psd() -> None:
                _, ax = plt.subplots(1, 1, figsize=figsize)
                ax.plot(freqs[mask], psd[mask] / np.max(psd[mask]))
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Power Spectral Density")
                ax.set_title(f"PSD of E population ({scan_label}={values[i]:.2f})")
                ax.grid(True)
                plt.tight_layout()

            save_both(
                data_path / f"{scan_name}_scan_psd_{scan_label}_{i + 1:02d}",
                plot_psd,
            )

        def plot_rates() -> None:
            e_rates = [rate[0] for rate in firing_rates]
            i_rates = [rate[1] for rate in firing_rates]
            plot_values = values[: len(firing_rates)]

            _, ax = plt.subplots(1, 1, figsize=figsize)
            ax.plot(plot_values, e_rates, marker="o", label="E population")
            ax.plot(plot_values, i_rates, marker="o", label="I population")
            ax.set_xlabel(scan_label)
            ax.set_ylabel("Firing Rate (Hz)")
            ax.set_title(f"Firing rates vs {scan_label}")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()

        save_both(data_path / f"{scan_name}_scan_firing_rates", plot_rates)

        def plot_rhythmicity() -> None:
            plot_values = values[: len(rhythmicity_vals)]
            _, ax = plt.subplots(1, 1, figsize=figsize)
            ax.plot(plot_values, rhythmicity_vals, marker="o")
            ax.set_xlabel(scan_label)
            ax.set_ylabel("Rhythmicity (Autocorr Peak)")
            ax.set_title(f"Rhythmicity vs {scan_label}")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True)
            plt.tight_layout()

        save_both(data_path / f"{scan_name}_scan_rhythmicity", plot_rhythmicity)
