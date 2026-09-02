"""Render retained measurements only; never infer, fit, or resolve training paths."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import theme
from experiments.helpers.figsave import save_figure
from experiments.helpers.operating_point import TAU_GABA_GAMMA_MS

from .recipe import F_GAMMA_BAND_HZ


def plot_quantitative_law(rows: list[dict], fit: dict, out_path: Path) -> None:
    """Seed-mean E rate and accuracy versus gamma frequency, with the saved affine fit."""
    theme.apply()

    fig, (ax_rate, ax_acc) = plt.subplots(
        2,
        1,
        figsize=(6.5, 4.2),
        sharex=True,
        gridspec_kw={"hspace": 0.12, "height_ratios": [1.6, 1.0]},
    )

    f_gammas: list[float] = []
    for row in rows:
        tau = row["tau_gaba_ms"]
        fg_mu, fg_se = row["f_gamma_hz"]["mean"], row["f_gamma_hz"]["sem"]
        er_mu, er_se = row["e_rate_hz"]["mean"], row["e_rate_hz"]["sem"]
        ac_mu, ac_se = row["acc"]["mean"], row["acc"]["sem"]
        f_gammas.append(fg_mu)
        ax_rate.errorbar(
            fg_mu,
            er_mu,
            xerr=fg_se,
            yerr=er_se,
            fmt="o",
            markersize=6,
            color=theme.INK_BLACK,
            capsize=3,
            label=f"τ_GABA = {tau:g} ms" if tau == TAU_GABA_GAMMA_MS else None,
        )
        ax_rate.annotate(
            f" {tau:g} ms",
            (fg_mu, er_mu),
            fontsize=theme.SIZE_ANNOTATION,
            color=theme.MUTED,
            xytext=(9, 0),
            textcoords="offset points",
            va="center",
        )
        ax_acc.errorbar(
            fg_mu,
            ac_mu,
            xerr=fg_se,
            yerr=ac_se,
            fmt="o",
            markersize=6,
            color=theme.INK_BLACK,
            capsize=3,
        )

    fg_arr = np.array(f_gammas)
    p_fit, a_fit, r2 = fit["p_affine"], fit["a_affine"], fit["r2_affine"]
    if p_fit is not None:
        xs = np.linspace(0, fg_arr.max() * 1.1, 200)
        ax_rate.plot(
            xs,
            p_fit * xs + a_fit,
            color=theme.DEEP_RED,
            lw=1.2,
            ls="--",
            label=(
                f"$r_E = a + p · f_γ$  (a = {a_fit:.2f} Hz, "
                f"p = {p_fit:.3f}, R² = {r2:.3f})"
            ),
        )
    ax_rate.set_ylabel("Hidden E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax_acc.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax_acc.set_xlabel(
        "Measured $f_γ$ (Hz) — peak of trained-network population PSD",
        fontsize=theme.SIZE_LABEL,
    )
    ax_acc.set_ylim(0, 100)
    ax_acc.axhline(10.0, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.6)
    for ax in (ax_rate, ax_acc):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=theme.SIZE_TICK)
    ax_rate.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    theme.label_panels((ax_rate, ax_acc))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_training_curves(
    curves: list[dict], taus: list[float], seeds: list[int], out_path: Path
) -> None:
    """Per-cell training-trajectory curves so convergence is auditable
    by inspection. One line per (τ_GABA, seed); colour by τ_GABA."""
    theme.apply()
    cmap = plt.get_cmap("viridis")
    taus_sorted = list(taus)
    fig, (ax_acc, ax_rate) = plt.subplots(
        2,
        1,
        figsize=(6.5, 5.175),
        sharex=True,
        gridspec_kw={"hspace": 0.15},
    )
    for i, tau in enumerate(taus_sorted):
        color = cmap(i / max(1, len(taus_sorted) - 1))
        for j, seed in enumerate(seeds):
            m = next(
                row
                for row in curves
                if row["tau_gaba_ms"] == tau and row["seed"] == seed
            )
            eps = [e["ep"] for e in m["epochs"]]
            accs = [e.get("acc", 0) for e in m["epochs"]]
            rates = [e.get("test_rate_e", 0) for e in m["epochs"]]
            label = f"τ_GABA = {tau:g} ms" if j == 0 else None
            ax_acc.plot(eps, accs, color=color, lw=1.0, alpha=0.85, label=label)
            ax_rate.plot(eps, rates, color=color, lw=1.0, alpha=0.85)
    ax_acc.set_ylabel("Validation accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax_rate.set_ylabel("Validation E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax_rate.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    ax_acc.legend(fontsize=theme.SIZE_LEGEND, frameon=False, ncol=2, loc="lower right")
    ax_acc.set_ylim(0, 100)
    for ax in (ax_acc, ax_rate):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, lw=0.4)
    theme.label_panels((ax_acc, ax_rate))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_psd_panel(rows: list[dict], out_path: Path) -> None:
    """One PSD curve per τ_GABA (mean across seeds), to verify the
    peak-frequency identification is reading the right feature."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(6.5, 3.15))
    cmap = plt.get_cmap("viridis")
    taus_sorted = [r["tau_gaba_ms"] for r in rows]
    for i, row in enumerate(rows):
        tau = row["tau_gaba_ms"]
        freqs = np.array(row["freqs_hz"])
        psd_mean = np.array(row["psd"])
        band_mask = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
        ax.plot(
            freqs[band_mask],
            psd_mean[band_mask],
            color=cmap(i / max(1, len(taus_sorted) - 1)),
            label=f"τ_GABA = {tau:g} ms",
            lw=1.2,
        )
        peak_f = row["psd_marker"]["frequency_hz"]
        peak_p = row["psd_marker"]["power"]
        ax.scatter(
            [peak_f],
            [peak_p],
            color=cmap(i / max(1, len(taus_sorted) - 1)),
            s=20,
            zorder=5,
        )
    ax.set_xlabel("Frequency (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Population PSD (a.u.)", fontsize=theme.SIZE_LABEL)
    ax.set_xlim(F_GAMMA_BAND_HZ)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_per_trial_peaks(
    rows: list[dict],
    out_path: Path,
) -> None:
    """Sanity check: per-trial PSD peak distribution per τ_GABA.

    Pools per-trial peak frequencies across seeds for each τ_GABA value
    and shows their histogram alongside the trial-mean-PSD peak.
    Narrow histograms → trial-mean-PSD f_γ is unbiased; wide histograms
    → trial-mean PSD is a centroid, and per-trial median would differ.
    """
    theme.apply()
    taus_sorted = [r["tau_gaba_ms"] for r in rows]
    n_taus = len(rows)
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(
        n_taus,
        1,
        figsize=(6.5, (1.5 * n_taus + 1.0) * 0.8625),
        sharex=True,
    )
    if n_taus == 1:
        axes = [axes]
    for ax, row in zip(axes, rows):
        tau = row["tau_gaba_ms"]
        stats = row["trial_peaks"]
        color = cmap(taus_sorted.index(tau) / max(1, n_taus - 1))
        if stats["median_hz"] is not None:
            ax.stairs(
                stats["counts"],
                stats["bins_hz"],
                fill=True,
                color=color,
                alpha=0.85,
                edgecolor=theme.INK_BLACK,
                lw=0.4,
            )
            median = stats["median_hz"]
            iqr = stats["iqr_hz"]
            ax.axvline(median, color=theme.INK_BLACK, ls="--", lw=1.0)
            ax.text(
                0.98,
                0.85,
                f"τ_GABA = {tau:g} ms\n"
                f"per-trial: median {median:.1f} Hz, IQR {iqr:.1f} Hz",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=theme.SIZE_LABEL,
            )
        mean_peak = row["f_gamma_hz"]["mean"]
        ax.axvline(
            mean_peak,
            color=theme.DEEP_RED,
            ls="-",
            lw=1.2,
            label=f"trial-mean PSD peak: {mean_peak:.1f} Hz",
        )
        ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
        ax.set_ylabel("trials", fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Per-trial PSD peak frequency (Hz)", fontsize=theme.SIZE_LABEL)
    theme.label_panels(axes)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_raster_strip(
    samples: list[dict],
    out_path: Path,
    t_window_ms: float,
) -> None:
    """One stacked single-trial raster per τ_GABA cluster. X-axis is
    physical time in ms (not steps), so the gamma cadence shift with
    τ_GABA is read by eye — shorter τ_GABA gives faster bursts gives
    more E spikes per unit time."""
    theme.apply()
    # Show τ_GABA in ascending value, top-down (so faster gamma at the top).
    samples = sorted(samples, key=lambda s: s["tau_gaba_ms"])
    n = len(samples)
    n_e = samples[0]["e"].shape[1]
    n_i = samples[0]["i"].shape[1]
    gap = 6
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(6.5, (1.0 * n + 1.0) * 0.69),
        sharex=True,
        gridspec_kw={"hspace": 0.22},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples)):
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt_ms"]
        mask = t_axis <= t_window_ms
        e_t, e_n = np.where(s["e"][mask])
        i_t, i_n = np.where(s["i"][mask])
        ax.scatter(
            t_axis[mask][e_t], e_n, s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.4
        )
        ax.scatter(
            t_axis[mask][i_t],
            i_n + n_e + gap,
            s=2.0,
            c=theme.DEEP_RED,
            marker="|",
            linewidths=0.4,
        )
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, t_window_ms)
        ax.text(
            1.012,
            0.5,
            f"τ_GABA = {s['tau_gaba_ms']:g} ms\nE = {s['e_rate_hz']:.1f} Hz",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=theme.SIZE_LABEL,
        )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    theme.label_panels(axes)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)
