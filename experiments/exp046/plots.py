"""Render saved exp046 analysis; scientific-reference review remains deferred."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import theme
from experiments.helpers.figsave import save_figure


def plot_distribution(per_tau: dict, out_path: Path) -> None:
    """Bar plot of P(spikes/cell/cycle = k) per τ_GABA, k ∈ {0, 1, 2, ≥3}."""
    theme.apply()
    taus_sorted = sorted(float(k.removeprefix("tau_")) for k in per_tau)
    n = len(taus_sorted)
    fig, axes = plt.subplots(1, n, figsize=(6.9, 4.5 * 6.9 / (2.4 * n)), sharey=True)
    if n == 1:
        axes = [axes]
    labels = ["0", "1", "2", "≥3"]
    cmap = plt.get_cmap("viridis")
    for i, tau in enumerate(taus_sorted):
        values = per_tau[f"tau_{tau:g}"]
        frac = [
            values[k] for k in ("frac_zero", "frac_one", "frac_two", "frac_three_plus")
        ]
        ax = axes[i]
        color = cmap(i / max(1, n - 1))
        ax.bar(labels, frac, color=color, edgecolor=theme.GREY_MID, lw=0.5)
        for k, v in enumerate(frac):
            ax.text(
                k,
                v + 0.01,
                f"{v * 100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=theme.SIZE_ANNOTATION,
                color=theme.INK,
            )
        ax.set_title(f"τ_GABA = {tau:g} ms", fontsize=theme.SIZE_LABEL)
        if i == 0:
            ax.set_ylabel("P(spikes per E cell per cycle)", fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.15, lw=0.4)
    fig.supxlabel("spikes / (cell · cycle)", fontsize=theme.SIZE_LABEL)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)  # bar chart: SVG + PDF
    plt.close(fig)


def plot_ceiling_vs_fgamma(rows: list[dict], out_path: Path) -> None:
    """Per-cell max E rate vs f_γ. y = f_γ line is the 1-spike/cycle ceiling."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(5.6, 3.15))

    # Aggregate by τ_GABA.
    by_tau: dict[float, list[dict]] = {}
    for r in rows:
        by_tau.setdefault(r["tau_gaba_ms"], []).append(r)
    taus_sorted = sorted(by_tau.keys())
    cmap = plt.get_cmap("viridis")

    f_gammas: list[float] = []
    for i, tau in enumerate(taus_sorted):
        group = by_tau[tau]
        color = cmap(i / max(1, len(taus_sorted) - 1))
        for r in group:
            f_gammas.append(r["f_gamma_hz"])
        # Plot the group: scatter.
        xs = [r["f_gamma_hz"] for r in group]
        ys_max = [r["per_cell_max_rate_hz"] for r in group]
        ys_med = [r["per_cell_median_rate_hz"] for r in group]
        ax.scatter(
            xs,
            ys_max,
            marker="^",
            s=60,
            color=color,
            edgecolor=theme.INK,
            lw=0.5,
            label=f"τ_GABA = {tau:g} ms — max cell" if i == 0 else None,
        )
        ax.scatter(
            xs,
            ys_med,
            marker="o",
            s=40,
            color=color,
            edgecolor=theme.INK,
            lw=0.5,
            label=f"τ_GABA = {tau:g} ms — median cell" if i == 0 else None,
        )

    f_arr = np.array(f_gammas)
    fmax = float(f_arr.max()) * 1.05
    xs = np.linspace(0, fmax, 100)
    ax.plot(
        xs,
        xs,
        color=theme.GREY_MID,
        lw=1.0,
        ls="--",
        label="y = f_γ (1 spike / cycle ceiling)",
    )
    ax.plot(
        xs,
        0.20 * xs,
        color=theme.MUTED,
        lw=1.0,
        ls=":",
        label="y = 0.20 · f_γ (exp041 slope p)",
    )

    ax.set_xlim(0, fmax)
    ax.set_ylim(0, fmax)
    ax.set_xlabel("Measured f_γ (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Per-cell E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, lw=0.4)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)  # sparse scatter + line overlays: SVG + PDF
    plt.close(fig)
