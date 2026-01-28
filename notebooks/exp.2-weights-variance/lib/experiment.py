import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from pinglab.analysis import mean_firing_rates, population_rate, rate_psd
from pinglab.inputs.tonic import tonic
from pinglab.lib.weights_builder import build_adjacency_matrices
from pinglab.plots.styles import save_both, figsize
from pinglab.run import build_model_from_config, run_network
from pinglab.types import Spikes
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


def _weights_with_params(weights, mean_vals, std_vals):
    updated = weights.model_copy(deep=True)
    for key in ("ee", "ei", "ie", "ii"):
        block = getattr(updated, key)
        params = dict(block.dist.params or {})
        params["mean"] = float(mean_vals[key])
        params["std"] = float(std_vals[key])
        block.dist.params = params
    return updated


def _build_matrices(run_cfg, weights, mean_vals, std_vals):
    updated = _weights_with_params(weights, mean_vals, std_vals)
    return build_adjacency_matrices(
        N_E=run_cfg.N_E,
        N_I=run_cfg.N_I,
        ee=updated.ee,
        ei=updated.ei,
        ie=updated.ie,
        ii=updated.ii,
        clamp_min=updated.clamp_min,
        seed=run_cfg.seed,
    )


def _rate_for_inputs(
    config: LocalConfig,
    run_cfg,
    weights,
    mean_vals: dict[str, float],
    std_vals: dict[str, float],
    I_E: float,
) -> float:
    rate_cfg = config.rate_match
    match_cfg = run_cfg.model_copy(update={"T": rate_cfg.sim_T})
    I_I = float(config.default_inputs.I_I)
    external_input = tonic(
        N_E=int(match_cfg.N_E),
        N_I=int(match_cfg.N_I),
        I_E=float(I_E),
        I_I=I_I,
        noise_std=float(config.default_inputs.noise),
        num_steps=int(np.ceil(match_cfg.T / match_cfg.dt)),
        seed=match_cfg.seed if match_cfg.seed is not None else 0,
    )
    matrices = _build_matrices(match_cfg, weights, mean_vals, std_vals)
    model = build_model_from_config(match_cfg)
    result = run_network(
        match_cfg,
        external_input=external_input,
        model=model,
        weights=matrices.W,
    )
    sliced = slice_spikes(
        result.spikes,
        start_time=rate_cfg.window_ms.start,
        stop_time=rate_cfg.window_ms.stop,
    )
    rate_e, _ = mean_firing_rates(
        sliced,
        N_E=int(match_cfg.N_E),
        N_I=int(match_cfg.N_I),
    )
    return rate_e


def _rate_match_drive(
    config: LocalConfig,
    run_cfg,
    weights,
    mean_vals: dict[str, float],
    std_vals: dict[str, float],
) -> float:
    rate_cfg = config.rate_match
    target = float(rate_cfg.target_rate_e)

    def score_I_E(I_E: float) -> float:
        rate_e = _rate_for_inputs(
            config,
            run_cfg,
            weights,
            mean_vals,
            std_vals,
            I_E,
        )
        return (rate_e - target) ** 2

    values_e = np.linspace(rate_cfg.I_E.start, rate_cfg.I_E.stop, rate_cfg.I_E.num)
    best_score = float("inf")
    best_I_E = float(config.default_inputs.I_E)
    for I_E in values_e:
        score = score_I_E(float(I_E))
        if score < best_score:
            best_score = score
            best_I_E = float(I_E)

    refine_half_widths = (0.3, 0.12)
    refine_points = 9
    for half_width in refine_half_widths:
        refine_start = max(rate_cfg.I_E.start, best_I_E - half_width)
        refine_stop = min(rate_cfg.I_E.stop, best_I_E + half_width)
        refine_values = np.linspace(refine_start, refine_stop, refine_points)
        for I_E in refine_values:
            score = score_I_E(float(I_E))
            if score < best_score:
                best_score = score
                best_I_E = float(I_E)

    return best_I_E


def _run_scan_cell(
    config: LocalConfig,
    run_cfg,
    weights,
    scan_key: str,
    mean_idx: int,
    std_idx: int,
    mean_val: float,
    std_val: float,
    data_path: Path,
) -> tuple[int, int, float, float]:
    mean_ee = float(weights.ee.dist.params.get("mean", 0.0))
    mean_ei = float(weights.ei.dist.params.get("mean", 0.0))
    mean_ie = float(weights.ie.dist.params.get("mean", 0.0))
    mean_ii = float(weights.ii.dist.params.get("mean", 0.0))
    std_ee = float(weights.ee.dist.params.get("std", 1.0))
    std_ei = float(weights.ei.dist.params.get("std", 1.0))
    std_ie = float(weights.ie.dist.params.get("std", 1.0))
    std_ii = float(weights.ii.dist.params.get("std", 1.0))

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

    I_E = _rate_match_drive(
        config,
        run_cfg,
        weights,
        mean_vals,
        std_vals,
    )
    I_I = float(config.default_inputs.I_I)
    logger.info(
        "matched drive I_E=%.3f (I_I=%.3f) target_E=%.2f",
        I_E,
        I_I,
        float(config.rate_match.target_rate_e),
    )
    external_input = tonic(
        N_E=int(run_cfg.N_E),
        N_I=int(run_cfg.N_I),
        I_E=float(I_E),
        I_I=I_I,
        noise_std=float(config.default_inputs.noise),
        num_steps=int(np.ceil(run_cfg.T / run_cfg.dt)),
        seed=run_cfg.seed if run_cfg.seed is not None else 0,
    )
    matrices = _build_matrices(run_cfg, weights, mean_vals, std_vals)

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
    ri_window_ms = float(config.ri_window_ms)
    ri_start = max(0.0, float(run_cfg.T) - ri_window_ms)
    ri_end = float(run_cfg.T)
    ri_window_ms = max(0.0, ri_end - ri_start)
    ri_spikes = slice_spikes(
        result.spikes,
        start_time=ri_start,
        stop_time=ri_end,
    )
    ri_spikes = Spikes(
        times=ri_spikes.times - ri_start,
        ids=ri_spikes.ids,
        types=getattr(ri_spikes, "types", None),
    )
    _, rate_hz = population_rate(
        ri_spikes,
        T_ms=ri_window_ms,
        dt_ms=dt_ms,
        pop="E",
        N_E=int(run_cfg.N_E),
        N_I=int(run_cfg.N_I),
    )
    rhythmicity = _autocorr_rhythmicity(rate_hz, dt_ms)

    rate_e, _ = mean_firing_rates(
        sliced,
        N_E=int(run_cfg.N_E),
        N_I=int(run_cfg.N_I),
    )
    logger.info(
        "rate_E=%.2f Hz for mean_%s=%.4f std_%s=%.4f",
        rate_e,
        scan_key,
        mean_val,
        scan_key,
        std_val,
    )
    label = (
        f"$\\mu_{{g_{{{scan_key}}}}}={mean_val:.3f}$, "
        f"$\\sigma_{{g_{{{scan_key}}}}}={std_val:.3f}$, "
        f"E rate={rate_e:.2f} Hz, RI={rhythmicity:.3f}"
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

        rate_centered = rate_hz - float(np.mean(rate_hz)) if rate_hz.size else rate_hz
        freqs, psd = rate_psd(rate_centered, dt_ms)
        psd_mask = (freqs >= 5.0) & (freqs <= 150.0)
        ax_psd.plot(freqs[psd_mask], psd[psd_mask])
        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_ylabel("PSD")
        ax_psd.grid(True, alpha=0.3)

    if mean_idx == std_idx:
        save_both(
            data_path / f"raster_{scan_key}_mean_{mean_idx:02d}_std_{std_idx:02d}",
            plot_raster_with_psd,
        )

    return mean_idx - 1, std_idx - 1, float(rhythmicity), float(rate_e)


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

        n_jobs = int(config.parallel.n_jobs)
        if n_jobs == 0:
            n_jobs = 1
        tasks = []
        for mean_idx, mean_val in enumerate(mean_values, start=1):
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
                tasks.append((mean_idx, std_idx, mean_val, std_val))

        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_run_scan_cell)(
                config,
                run_cfg,
                config.weights,
                scan_key,
                mean_idx,
                std_idx,
                mean_val,
                std_val,
                data_path,
            )
            for mean_idx, std_idx, mean_val, std_val in tasks
        )

        for mean_idx, std_idx, rhythmicity, rate_e in results:
            rhythmicity_grid[mean_idx, std_idx] = rhythmicity
            rate_e_grid[mean_idx, std_idx] = rate_e

        # Removed per-mean rhythmicity vs std plots to trim artifacts.

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
