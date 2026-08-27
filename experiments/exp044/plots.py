"""Presentation of saved analysis; no measurements or source-bank access."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import theme
from experiments.helpers.figsave import save_figure


def plot_dt_sweep(rows: list[dict], out_path: Path) -> None:
    """E rate (left axis) and accuracy (right axis) vs Δt (log)."""
    theme.apply()
    dts_sorted = [row["dt_ms"] for row in rows]
    e_means = [row["e_rate_hz"]["mean"] for row in rows]
    e_sems = [row["e_rate_hz"]["sem"] for row in rows]
    acc_means = [row["acc"]["mean"] for row in rows]
    acc_sems = [row["acc"]["sem"] for row in rows]

    fig, ax_rate = plt.subplots(figsize=(5.6, 3.5))
    ax_rate.errorbar(
        dts_sorted,
        e_means,
        yerr=e_sems,
        marker="D",
        markersize=6,
        lw=1.4,
        color=theme.INK_BLACK,
        capsize=3,
        label="E rate (Hz)",
    )
    ax_rate.set_xscale("log")
    ax_rate.set_xlabel("Δt (ms)", fontsize=theme.SIZE_LABEL)
    ax_rate.set_ylabel(
        "Hidden E rate (Hz)", fontsize=theme.SIZE_LABEL, color=theme.INK_BLACK
    )
    ax_rate.tick_params(axis="y", labelcolor=theme.INK_BLACK)
    ax_rate.set_ylim(0, 50)
    ax_rate.set_xticks(dts_sorted)
    ax_rate.set_xticklabels([f"{d:g}" for d in dts_sorted])
    ax_rate.spines["top"].set_visible(False)

    ax_acc = ax_rate.twinx()
    ax_acc.errorbar(
        dts_sorted,
        acc_means,
        yerr=acc_sems,
        marker="s",
        markersize=6,
        lw=1.4,
        color=theme.DEEP_RED,
        capsize=3,
        label="accuracy (%)",
    )
    ax_acc.set_ylabel(
        "Test accuracy (%)", fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED
    )
    ax_acc.tick_params(axis="y", labelcolor=theme.DEEP_RED)
    ax_acc.set_ylim(0, 100)
    ax_acc.spines["top"].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_raster_strip(
    samples: list[dict],
    out_path: Path,
    t_window_ms: float,
) -> None:
    """Single-trial rasters across Δt, one panel per Δt. X-axis is
    physical time in ms (not steps), so gamma cycle alignment is read
    by eye — same physical period if dynamics survive Δt change."""
    theme.apply()
    n = len(samples)
    n_e = samples[0]["e"].shape[1]
    n_i = samples[0]["i"].shape[1]
    gap = 6
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(6.9, 0.7 * n + 0.8),
        sharex=True,
        gridspec_kw={"hspace": 0.22},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples)):
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt_ms"]
        # Truncate display to the first t_window_ms ms so cycles are visible.
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
            f"Δt = {s['dt_ms']:g} ms\nE = {s['e_rate_hz']:.1f} Hz",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=theme.SIZE_LABEL,
        )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


# Training trajectories are retained validation observations.


def plot_training_curves(
    curves: list[dict], dts_sorted: list[float], seeds: list[int], out_path: Path
) -> None:
    """Per-cell training-trajectory curves. One line per (Δt, seed);
    colour by Δt (viridis on log Δt)."""
    theme.apply()
    cmap = plt.get_cmap("viridis")
    fig, (ax_acc, ax_rate) = plt.subplots(
        2,
        1,
        figsize=(5.6, 4.6),
        sharex=True,
        gridspec_kw={"hspace": 0.15},
    )
    for i, dt_ms in enumerate(dts_sorted):
        color = cmap(i / max(1, len(dts_sorted) - 1))
        for j, seed in enumerate(seeds):
            m = next(
                row for row in curves if row["dt_ms"] == dt_ms and row["seed"] == seed
            )
            eps = [e["ep"] for e in m["epochs"]]
            accs = [e["acc"] for e in m["epochs"]]
            rates = [e["test_rate_e"] for e in m["epochs"]]
            label = f"Δt = {dt_ms:g} ms" if j == 0 else None
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
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)
