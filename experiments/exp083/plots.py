"""Draw saved response, raster coordinates and spectra; no measurements."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import theme

from .recipe import BURN_MS, DT_MS, N_E, N_I, REPRESENTATIVE_RATES_HZ, T_MS


def plot_representative_rasters(
    recordings: dict[float, dict[str, np.ndarray]],
    summaries: list[dict],
    out: Path,
) -> None:
    theme.apply()
    by_rate = {row["input_rate_hz"]: row for row in summaries}
    gap = 6
    fig, axes = plt.subplots(
        3, 1, figsize=(6.5, 3.5), sharex=True, gridspec_kw={"hspace": 0.18}
    )
    for row, rate in enumerate(REPRESENTATIVE_RATES_HZ):
        arrays = recordings[rate]
        axis = axes[row]
        e_t, e_cells = arrays["e_t"], arrays["e_cells"]
        i_t, i_cells = arrays["i_t"], arrays["i_cells"]
        axis.scatter(
            e_t * DT_MS,
            e_cells,
            s=2.0,
            marker="|",
            linewidths=0.4,
            color=theme.INK_BLACK,
            rasterized=True,
        )
        axis.scatter(
            i_t * DT_MS,
            i_cells + N_E + gap,
            s=2.0,
            marker="|",
            linewidths=0.4,
            color=theme.DEEP_RED,
            rasterized=True,
        )
        axis.axvline(BURN_MS, color=theme.GREY_MID, ls="--", lw=0.8)
        axis.set_ylim(-2, N_E + gap + N_I + 2)
        axis.set_yticks([N_E / 2, N_E + gap + N_I / 2])
        axis.set_yticklabels(["E", "I"])
        axis.tick_params(axis="y", length=0)
        condition = by_rate[rate]
        axis.text(
            1.012,
            0.5,
            (
                f"{rate:g} Hz/channel\n"
                f"E {condition['e_rate_mean_hz']:.1f} · "
                f"I {condition['i_rate_mean_hz']:.1f} Hz"
            ),
            transform=axis.transAxes,
            ha="left",
            va="center",
            fontsize=theme.SIZE_ANNOTATION,
        )
        axis.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlim(0, T_MS)
    axes[-1].set_xlabel("time (ms)")
    fig.subplots_adjust(left=0.08, right=0.78, bottom=0.15, top=0.98, hspace=0.18)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_response(summaries: list[dict], out: Path) -> None:
    theme.apply()
    x = np.array([row["input_rate_hz"] for row in summaries])
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 5.2), sharex=True)
    for key, std, label, colour in (
        ("e_rate_mean_hz", "e_rate_std_hz", "E", theme.INK_BLACK),
        ("i_rate_mean_hz", "i_rate_std_hz", "I", theme.DEEP_RED),
    ):
        axes[0].errorbar(
            x,
            [row[key] for row in summaries],
            yerr=[row[std] for row in summaries],
            marker="o",
            lw=1.3,
            capsize=3,
            color=colour,
            label=label,
        )
    axes[0].set_ylabel("rate (Hz)")
    axes[0].legend(frameon=False)
    rhythmicity = np.array([row["rhythmicity_score_median"] for row in summaries])
    rhythmicity_iqr = np.array([row["rhythmicity_score_iqr"] for row in summaries])
    axes[1].errorbar(
        x,
        rhythmicity,
        yerr=rhythmicity_iqr / 2.0,
        marker="o",
        capsize=3,
        color=theme.INK_BLACK,
    )
    axes[1].set_ylim(-0.04, 1.04)
    axes[1].set_ylabel("rhythmicity score")
    frequencies = [row["rhythm_frequency_median_hz"] for row in summaries]
    axes[2].plot(x, frequencies, marker="o", color=theme.DEEP_RED)
    axes[2].axhspan(5, 30, color=theme.GREY_LIGHT, alpha=0.35)
    axes[2].set_ylim(0, 65)
    axes[2].set_ylabel("frequency (Hz)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("input rate per channel (Hz)")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_spectra(estimates: dict[float, dict[str, np.ndarray]], out: Path) -> None:
    theme.apply()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for rate in REPRESENTATIVE_RATES_HZ:
        estimate = estimates[rate]
        frequencies = estimate["frequencies_hz"]
        keep = (frequencies >= 20) & (frequencies <= 100)
        power = estimate["mean_psd"][keep]
        scale = power.max() if power.size and power.max() > 0 else 1.0
        ax.plot(frequencies[keep], power / scale, label=f"{rate:g} Hz input")
    ax.axvspan(30, 80, color=theme.GREY_LIGHT, alpha=0.45)
    ax.set(xlabel="frequency (Hz)", ylabel="mean PSD (peak-normalized)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
