"""Render retained summaries and raster samples; no simulation or measurement."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import theme
from experiments.helpers.figsave import save_figure


def plot_cell_jitter_sweep(
    cell_rows: list[dict],
    out_path: Path,
) -> None:
    """Per-I-cell jitter sweep — E rate, accuracy, and realised I rate.

    Same twin-axis layout and grey realised-I trace as plot_jitter_sweep, for
    the per-spike jitter family.
    """
    theme.apply()
    sigmas_sorted = [r["sigma_ms"] for r in cell_rows]
    e_means = [r["e_rate_hz"]["mean"] for r in cell_rows]
    e_sems = [r["e_rate_hz"]["sem"] for r in cell_rows]
    acc_means = [r["acc"]["mean"] for r in cell_rows]
    acc_sems = [r["acc"]["sem"] for r in cell_rows]
    i_means = [r["i_rate_hz"]["mean"] for r in cell_rows]

    fig, ax_rate = plt.subplots(figsize=(5.6, 3.11))
    ax_rate.errorbar(
        sigmas_sorted,
        e_means,
        yerr=e_sems,
        marker="D",
        markersize=6,
        lw=1.4,
        color=theme.INK_BLACK,
        capsize=3,
        label="E rate (Hz)",
    )
    # Realised mean I rate — the "held fixed" control, same grey full-trace styling
    # as the cycle-coherent sweep. Per-cell jitter only moves each spike by a small
    # independent offset, so it stays flat near baseline through the E collapse.
    ax_rate.plot(
        sigmas_sorted,
        i_means,
        marker="o",
        markersize=6,
        lw=1.4,
        color=theme.GREY_MID,
        label="realised I rate (Hz)",
    )
    # Symlog x-axis (linthresh matched to plot_jitter_sweep) so the per-cell
    # collapse — all of which happens below σ ≈ 9 ms — spreads across the plot
    # instead of piling into the left margin, and the two paired sweep figures
    # share one x-scale for direct comparison.
    ax_rate.set_xscale("symlog", linthresh=1.0)
    ax_rate.set_xlabel(
        "Per-I-cell jitter σ on the I-stream (ms, symlog)",
        fontsize=theme.SIZE_LABEL,
    )
    ax_rate.set_ylabel(
        "Firing rate (Hz)", fontsize=theme.SIZE_LABEL, color=theme.INK_BLACK
    )
    ax_rate.tick_params(axis="y", labelcolor=theme.INK_BLACK)

    ax_acc = ax_rate.twinx()
    ax_acc.errorbar(
        sigmas_sorted,
        acc_means,
        yerr=acc_sems,
        marker="s",
        markersize=6,
        lw=1.4,
        color=theme.DEEP_RED,
        capsize=3,
        label="Test accuracy (%)",
    )
    ax_acc.set_ylabel(
        "Test accuracy (%)", fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED
    )
    ax_acc.tick_params(axis="y", labelcolor=theme.DEEP_RED)
    ax_acc.set_ylim(0, 100)

    # Self-identify all three traces (twin-axis colours alone don't survive
    # greyscale print): a single legend combining both axes' handles, replacing
    # the earlier inline grey-only label and activating the previously-unused
    # label= kwargs.
    h_rate, l_rate = ax_rate.get_legend_handles_labels()
    h_acc, l_acc = ax_acc.get_legend_handles_labels()
    ax_rate.legend(
        h_rate + h_acc,
        l_rate + l_acc,
        loc="center right",
        frameon=False,
        fontsize=theme.SIZE_LEGEND,
    )

    # H17: caption carries the takeaway
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_jitter_sweep(
    jitter_rows: list[dict],
    out_path: Path,
) -> None:
    """E rate, accuracy, and realised I rate vs cycle-coherent jitter σ.

    jitter_rows: list of dicts with sigma_ms, e_rate_hz, i_rate_hz, acc.
    Aggregated across seeds before plotting.
    """
    theme.apply()
    sigmas_sorted = [r["sigma_ms"] for r in jitter_rows]
    e_means = [r["e_rate_hz"]["mean"] for r in jitter_rows]
    e_sems = [r["e_rate_hz"]["sem"] for r in jitter_rows]
    acc_means = [r["acc"]["mean"] for r in jitter_rows]
    acc_sems = [r["acc"]["sem"] for r in jitter_rows]
    i_means = [r["i_rate_hz"]["mean"] for r in jitter_rows]

    fig, ax_rate = plt.subplots(figsize=(5.6, 3.11))
    # Use a symlog x-axis so both σ = 0 and σ = 100 are visible.
    ax_rate.errorbar(
        sigmas_sorted,
        e_means,
        yerr=e_sems,
        marker="D",
        markersize=6,
        lw=1.4,
        color=theme.INK_BLACK,
        capsize=3,
        label="E rate (Hz)",
    )
    # Realised mean I rate — the "held fixed" control, same grey full-trace styling
    # as the per-cell sweep. Flat near baseline over the rate-matched range; droops at
    # large σ where the Gaussian block offset displaces part of each burst past the
    # trial window (see Methods note).
    ax_rate.plot(
        sigmas_sorted,
        i_means,
        marker="o",
        markersize=6,
        lw=1.4,
        color=theme.GREY_MID,
        label="realised I rate (Hz)",
    )
    ax_rate.set_xscale("symlog", linthresh=1.0)
    ax_rate.set_xlabel(
        "Cycle-coherent jitter σ on the I-stream (ms, symlog)",
        fontsize=theme.SIZE_LABEL,
    )
    ax_rate.set_ylabel(
        "Firing rate (Hz)", fontsize=theme.SIZE_LABEL, color=theme.INK_BLACK
    )
    ax_rate.tick_params(axis="y", labelcolor=theme.INK_BLACK)

    ax_acc = ax_rate.twinx()
    ax_acc.errorbar(
        sigmas_sorted,
        acc_means,
        yerr=acc_sems,
        marker="s",
        markersize=6,
        lw=1.4,
        color=theme.DEEP_RED,
        capsize=3,
        label="Test accuracy (%)",
    )
    ax_acc.set_ylabel(
        "Test accuracy (%)", fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED
    )
    ax_acc.tick_params(axis="y", labelcolor=theme.DEEP_RED)
    ax_acc.set_ylim(0, 100)

    # Self-identify all three traces with one legend combining both axes'
    # handles (replaces the inline grey-only label; matches plot_cell_jitter_sweep).
    h_rate, l_rate = ax_rate.get_legend_handles_labels()
    h_acc, l_acc = ax_acc.get_legend_handles_labels()
    ax_rate.legend(
        h_rate + h_acc,
        l_rate + l_acc,
        loc="center left",
        frameon=False,
        fontsize=theme.SIZE_LEGEND,
    )

    # H17: caption carries the takeaway
    ax_rate.spines["top"].set_visible(False)
    ax_acc.spines["top"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _compound_raster_panel(ax, s: dict, title: str, subtitle: str) -> None:
    """One raster panel — E (black) below, I (red) above, rates annotated."""
    n_e, n_i, gap = s["e"].shape[1], s["i"].shape[1], 6
    T = s["e"].shape[0]
    t_axis = np.arange(T) * s["dt"]
    e_t, e_n = np.where(s["e"])
    i_t, i_n = np.where(s["i"])
    ax.scatter(t_axis[e_t], e_n, s=1.5, c=theme.INK_BLACK, marker="|", linewidths=0.4)
    ax.scatter(
        t_axis[i_t],
        i_n + n_e + gap,
        s=1.5,
        c=theme.DEEP_RED,
        marker="|",
        linewidths=0.4,
    )
    ax.set_ylim(-2, n_e + n_i + gap + 2)
    ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
    ax.set_yticklabels(["E", "I"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, s["t_ms"])
    ax.set_title(title, fontsize=theme.SIZE_LABEL)
    ax.text(
        0.98,
        0.94,
        subtitle + f"\nE = {s['e_rate_hz']:.1f} Hz   I = {s['i_rate_hz']:.1f} Hz",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=theme.SIZE_ANNOTATION,
        color=theme.MUTED,
        # Opaque backing so the annotation reads over the dense I-raster instead
        # of crowding into the red spikes.
        bbox=dict(
            boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.92
        ),
    )
    _despine(ax)


def _compound_sweep_panel(
    ax,
    rows: list[dict],
    *,
    xlabel: str,
    title: str,
    symlog: bool,
    legend_loc: str,
) -> None:
    """One sweep panel — E rate (black) and accuracy (red, twin axis) vs σ."""
    sig = [r["sigma_ms"] for r in rows]
    e_means = [r["e_rate_hz"]["mean"] for r in rows]
    i_means = [r["i_rate_hz"]["mean"] for r in rows]
    a_means = [r["acc"]["mean"] for r in rows]

    ax.plot(
        sig, e_means, marker="D", ms=5, lw=1.4, color=theme.INK_BLACK, label="E rate"
    )
    # Realised (measured) mean I rate on the same Hz axis — makes the "mean
    # inhibition held fixed" control visible directly. It sits flat near
    # baseline over the rate-matched range and only droops where the finite
    # trial window truncates the displaced-burst tail (cycle-coherent, large σ).
    ax.plot(
        sig,
        i_means,
        marker=".",
        ms=4,
        lw=1.0,
        color=theme.GREY_MID,
        ls="-",
        alpha=0.75,
        label="realised I",
    )
    if symlog:
        ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel(xlabel, fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("firing rate (Hz)", color=theme.INK_BLACK, fontsize=theme.SIZE_LABEL)
    ax.tick_params(axis="y", labelcolor=theme.INK_BLACK)
    ax.set_title(title, fontsize=theme.SIZE_LABEL)

    ax_acc = ax.twinx()
    ax_acc.plot(
        sig, a_means, marker="s", ms=5, lw=1.4, color=theme.DEEP_RED, label="accuracy"
    )
    ax_acc.set_ylabel("accuracy (%)", color=theme.DEEP_RED, fontsize=theme.SIZE_LABEL)
    ax_acc.tick_params(axis="y", labelcolor=theme.DEEP_RED)
    ax_acc.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax_acc.spines["top"].set_visible(False)

    # Self-identify all three traces; combine both axes' handles into one legend.
    h_rate, l_rate = ax.get_legend_handles_labels()
    h_acc, l_acc = ax_acc.get_legend_handles_labels()
    ax.legend(
        h_rate + h_acc,
        l_rate + l_acc,
        loc=legend_loc,
        frameon=False,
        fontsize=theme.SIZE_ANNOTATION,
    )


def fig_rhythm_compound(
    cyc_rows: list[dict],
    cell_rows: list[dict],
    raster_cyc: dict,
    raster_cell: dict,
    out_path: Path,
) -> None:
    """2×2 manuscript compound — matched mean I, opposite E response.

    Columns are the two manipulations that both preserve mean I rate:
      left  — cycle-coherent jitter (within-burst synchrony kept, bursts
              displaced) → E fires through the opened gaps, rate rises.
      right — per-I-cell jitter (synchrony destroyed, bursts smeared into
              a continuous shunt) → E silenced, rate falls to zero.
    Top row: example single-trial rasters; bottom row: the full sweeps.
    """
    theme.apply()
    prev_bbox = plt.rcParams["savefig.bbox"]
    plt.rcParams["savefig.bbox"] = "standard"
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 3.88))

    _compound_raster_panel(
        axes[0, 0],
        raster_cell,
        "Smear the bursts — synchrony destroyed",
        f"per-I-cell jitter σ = {raster_cell['sigma_ms']:g} ms",
    )
    _compound_raster_panel(
        axes[0, 1],
        raster_cyc,
        "Move the bursts — synchrony preserved",
        f"cycle-coherent jitter σ = {raster_cyc['sigma_ms']:g} ms",
    )
    _compound_sweep_panel(
        axes[1, 0],
        cell_rows,
        xlabel="per-I-cell jitter σ (ms, symlog)",
        title="Smear bursts → E rate falls to zero",
        # Symlog to match the cycle-coherent panel (and the standalone sweeps):
        # the per-cell collapse all happens below σ ≈ 9 ms and would otherwise
        # pile into the left margin, breaking the side-by-side read.
        symlog=True,
        legend_loc="center right",
    )
    _compound_sweep_panel(
        axes[1, 1],
        cyc_rows,
        xlabel="cycle-coherent jitter σ (ms, symlog)",
        title="Displace bursts → E rate rises",
        symlog=True,
        legend_loc="center left",
    )
    # H17: caption carries the takeaway
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense rasters: PNG, not SVG
    plt.close(fig)
    plt.rcParams["savefig.bbox"] = prev_bbox
