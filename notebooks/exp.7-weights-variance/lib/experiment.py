import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pinglab.analysis import mean_firing_rates, population_rate, rate_psd
from pinglab.inputs.tonic import tonic
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


def _build_matrices(run_cfg, weights, mean_vals, std_vals):
    return build_adjacency_matrices(
        N_E=run_cfg.N_E,
        N_I=run_cfg.N_I,
        mean_ee=mean_vals["ee"],
        mean_ei=mean_vals["ei"],
        mean_ie=mean_vals["ie"],
        mean_ii=mean_vals["ii"],
        std_ee=std_vals["ee"],
        std_ei=std_vals["ei"],
        std_ie=std_vals["ie"],
        std_ii=std_vals["ii"],
        p_ee=weights.p_ee,
        p_ei=weights.p_ei,
        p_ie=weights.p_ie,
        p_ii=weights.p_ii,
        clamp_min=weights.clamp_min,
        seed=run_cfg.seed,
    )


def run_experiment(config: LocalConfig, data_path: Path) -> None:
    run_cfg = config.base
    if config.weights is None:
        raise ValueError("weights must be provided for this experiment.")

    scan_definitions = [
        ("ei", config.scan.mean_ei, config.scan.std_ei),
        # ("ie", config.scan.mean_ie, config.scan.std_ie),
        # ("ee", config.scan.mean_ee, config.scan.std_ee),
        # ("ii", config.scan.mean_ii, config.scan.std_ii),
    ]

    for scan_key, mean_cfg, std_cfg in scan_definitions:
        mean_values = np.linspace(
            mean_cfg.start,
            mean_cfg.stop,
            mean_cfg.num,
        )
        std_values = np.linspace(
            std_cfg.start,
            std_cfg.stop,
            std_cfg.num,
        )
        logger.info(
            "mean_%s scan: %d values from %.3f to %.3f",
            scan_key,
            mean_values.size,
            mean_values[0],
            mean_values[-1],
        )
        logger.info(
            "std_%s scan: %d values from %.3f to %.3f",
            scan_key,
            std_values.size,
            std_values[0],
            std_values[-1],
        )

        rhythmicity_grid = np.zeros((mean_values.size, std_values.size), dtype=float)
        rate_e_grid = np.zeros((mean_values.size, std_values.size), dtype=float)

        for mean_idx, mean_val in enumerate(mean_values, start=1):
            rhythmicity_vals: list[float] = []
            logger.info(
                "mean_%s=%.4f (%d/%d)",
                scan_key,
                mean_val,
                mean_idx,
                mean_values.size,
            )
            for std_idx, std_val in enumerate(std_values, start=1):
                logger.info(
                    "running std_%s=%.4f (%d/%d)",
                    scan_key,
                    std_val,
                    std_idx,
                    std_values.size,
                )
                mean_ee = config.weights.mean_ee
                mean_ei = config.weights.mean_ei
                mean_ie = config.weights.mean_ie
                mean_ii = config.weights.mean_ii
                std_ee = config.weights.std_ee
                std_ei = config.weights.std_ei
                std_ie = config.weights.std_ie
                std_ii = config.weights.std_ii

                if scan_key == "ei":
                    mean_ei = float(mean_val)
                    std_ei = float(std_val)
                elif scan_key == "ie":
                    mean_ie = float(mean_val)
                    std_ie = float(std_val)
                elif scan_key == "ee":
                    mean_ee = float(mean_val)
                    std_ee = float(std_val)
                elif scan_key == "ii":
                    mean_ii = float(mean_val)
                    std_ii = float(std_val)

                mean_vals = {
                    "ee": mean_ee,
                    "ei": mean_ei,
                    "ie": mean_ie,
                    "ii": mean_ii,
                }
                std_vals = {
                    "ee": std_ee,
                    "ei": std_ei,
                    "ie": std_ie,
                    "ii": std_ii,
                }
                I_E = float(config.default_inputs.I_E)
                I_I = float(config.default_inputs.I_I)
                external_input = tonic(
                    N_E=int(run_cfg.N_E),
                    N_I=int(run_cfg.N_I),
                    I_E=float(I_E),
                    I_I=I_I,
                    noise_std=float(config.default_inputs.noise),
                    num_steps=int(np.ceil(run_cfg.T / run_cfg.dt)),
                    seed=run_cfg.seed if run_cfg.seed is not None else 0,
                )
                matrices = _build_matrices(run_cfg, config.weights, mean_vals, std_vals)

                model = build_model_from_config(run_cfg)
                result = run_network(
                    run_cfg,
                    external_input=external_input,
                    model=model,
                    weights=matrices.W,
                )

                start_time = float(config.plotting.raster.start_time)
                stop_time = float(config.plotting.raster.stop_time)
                sliced = slice_spikes(
                    result.spikes,
                    start_time=start_time,
                    stop_time=stop_time,
                )

                dt_ms = 5.0
                _, rate_hz = population_rate(
                    sliced,
                    T_ms=run_cfg.T,
                    dt_ms=dt_ms,
                    pop="E",
                    N_E=int(run_cfg.N_E),
                    N_I=int(run_cfg.N_I),
                )
                rhythmicity = _autocorr_rhythmicity(rate_hz, dt_ms)
                rhythmicity_vals.append(rhythmicity)
                rhythmicity_grid[mean_idx - 1, std_idx - 1] = rhythmicity

                rate_e, _ = mean_firing_rates(
                    sliced,
                    N_E=int(run_cfg.N_E),
                    N_I=int(run_cfg.N_I),
                )
                rate_e_grid[mean_idx - 1, std_idx - 1] = rate_e
                logger.info(
                    "rate_E=%.2f Hz for mean_%s=%.4f std_%s=%.4f",
                    rate_e,
                    scan_key,
                    mean_val,
                    scan_key,
                    std_val,
                )
                label = (
                    f"mean_{scan_key}={mean_val:.3f} std_{scan_key}={std_val:.3f}"
                    f"\nI_E={I_E:.2f} I_I={I_I:.2f}"
                    f"\nrate_E={rate_e:.2f} Hz"
                )

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
                        ax_ras.scatter(
                            times[mask_I],
                            ids[mask_I],
                            s=0.5,
                            marker=".",
                            alpha=0.7,
                        )
                    else:
                        ax_ras.scatter(times, ids, s=0.5, marker=".")

                    ax_ras.set_xlim(start_time, stop_time)
                    ax_ras.set_ylim(0.0, float(run_cfg.N_E + run_cfg.N_I))
                    ax_ras.set_xlabel("Time (ms)")
                    ax_ras.set_ylabel("Neuron id")
                    ax_ras.set_title(label, fontsize=18)

                    rate_centered = (
                        rate_hz - float(np.mean(rate_hz)) if rate_hz.size else rate_hz
                    )
                    freqs, psd = rate_psd(rate_centered, dt_ms)
                    psd_mask = (freqs >= 5.0) & (freqs <= 150.0)
                    ax_psd.plot(freqs[psd_mask], psd[psd_mask])
                    ax_psd.set_xlabel("Frequency (Hz)")
                    ax_psd.set_ylabel("PSD")
                    ax_psd.grid(True, alpha=0.3)

                save_both(
                    data_path
                    / f"raster_{scan_key}_mean_{mean_idx:02d}_std_{std_idx:02d}",
                    plot_raster_with_psd,
                )

            def plot_rhythmicity() -> None:
                _, ax = plt.subplots(1, 1, figsize=figsize)
                ax.plot(std_values, rhythmicity_vals, marker="o")
                ax.set_xlabel(f"std_g_{scan_key}")
                ax.set_ylabel("Rhythmicity (Autocorr Peak)")
                ax.set_title(
                    f"Rhythmicity vs g_{scan_key} stddev (mean_{scan_key}={mean_val:.3f})",
                    fontsize=18,
                )
                ax.set_ylim(0.0, 1.0)
                ax.grid(True)
                plt.tight_layout()

            save_both(
                data_path / f"rhythmicity_vs_g_{scan_key}_stddev_mean_{mean_idx:02d}",
                plot_rhythmicity,
            )

        def plot_rhythmicity_heatmap() -> None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            im = ax.imshow(
                rhythmicity_grid,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap="gray_r",
            )
            ax.set_title(f"Rhythmicity heatmap (g_{scan_key})", fontsize=18)
            ax.set_xlabel(f"std_g_{scan_key}")
            ax.set_ylabel(f"mean_g_{scan_key}")
            x_step = max(1, int(np.ceil(std_values.size / 10)))
            y_step = max(1, int(np.ceil(mean_values.size / 10)))
            ax.set_xticks(np.arange(0, std_values.size, x_step))
            ax.set_yticks(np.arange(0, mean_values.size, y_step))
            ax.set_xticklabels([f"{v:.3f}" for v in std_values[::x_step]])
            ax.set_yticklabels([f"{v:.3f}" for v in mean_values[::y_step]])
            ax.tick_params(axis="x", labelrotation=45, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8)
            plt.tight_layout()

        save_both(
            data_path / f"rhythmicity_heatmap_{scan_key}",
            plot_rhythmicity_heatmap,
        )

        def plot_rate_e_heatmap() -> None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            im = ax.imshow(
                rate_e_grid,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap="gray_r",
            )
            ax.set_title(f"E rate heatmap (g_{scan_key})", fontsize=18)
            ax.set_xlabel(f"std_g_{scan_key}")
            ax.set_ylabel(f"mean_g_{scan_key}")
            x_step = max(1, int(np.ceil(std_values.size / 10)))
            y_step = max(1, int(np.ceil(mean_values.size / 10)))
            ax.set_xticks(np.arange(0, std_values.size, x_step))
            ax.set_yticks(np.arange(0, mean_values.size, y_step))
            ax.set_xticklabels([f"{v:.3f}" for v in std_values[::x_step]])
            ax.set_yticklabels([f"{v:.3f}" for v in mean_values[::y_step]])
            ax.tick_params(axis="x", labelrotation=45, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            fig.colorbar(im, ax=ax, shrink=0.8)
            plt.tight_layout()

        save_both(
            data_path / f"rate_e_heatmap_{scan_key}",
            plot_rate_e_heatmap,
        )

        def plot_mean_scan_std0() -> None:
            std0_idx = 0
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.plot(mean_values, rhythmicity_grid[:, std0_idx], marker="o")
            ax.set_xlabel(f"mean_g_{scan_key}")
            ax.set_ylabel("Rhythmicity (Autocorr Peak)")
            ax.set_title(
                f"Rhythmicity vs g_{scan_key} mean (std_{scan_key}=0)",
                fontsize=18,
            )
            ax.set_ylim(0.0, 1.0)
            ax.grid(True)
            plt.tight_layout()

        save_both(
            data_path / f"rhythmicity_vs_g_{scan_key}_mean_std0",
            plot_mean_scan_std0,
        )
