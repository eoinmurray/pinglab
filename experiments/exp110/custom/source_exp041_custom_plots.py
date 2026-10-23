"""Render retained measurements only; never infer, fit, or resolve training paths.

The owning experiment calls these local definitions from its explicit stages.
Inputs and outputs retain that stage’s validated scientific contract; importing
this module does not start computation or select a scientific run.
"""

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

from experiments import theme

from experiments.exp110.custom.source_exp041_custom_figures import save_figure

from experiments.exp110.custom.source_exp041_recipe import TAU_GABA_REFERENCE_MS


def plot_quantitative_law(
    rows: list[dict], fit: dict, out_path: Path, *, axes=None
) -> None:
    """Seed-mean E rate and accuracy versus gamma frequency, with the saved affine fit."""
    theme.apply()

    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(6.5, 4.2),
            sharex=True,
            gridspec_kw={"hspace": 0.12, "height_ratios": [1.6, 1.0]},
        )
    ax_rate, ax_acc = axes
    fig = ax_rate.figure

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
            markersize=6 if standalone else 3,
            color=theme.INK_BLACK,
            capsize=3 if standalone else 2,
            label=f"τ_GABA = {tau:g} ms"
            if standalone and tau == TAU_GABA_REFERENCE_MS
            else None,
        )
        ax_rate.annotate(
            f" {tau:g} ms",
            (fg_mu, er_mu),
            fontsize=theme.SIZE_ANNOTATION,
            color=theme.MUTED,
            xytext=(9 if standalone else 4, -3),
            textcoords="offset points",
            va="center",
        )
        ax_acc.errorbar(
            fg_mu,
            ac_mu,
            xerr=fg_se,
            yerr=ac_se,
            fmt="o",
            markersize=6 if standalone else 3,
            color=theme.INK_BLACK,
            capsize=3 if standalone else 2,
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
                "$r_E = a + p · f_γ$"
                + ("  " if standalone else "\n")
                + f"(a = {a_fit:.2f} Hz, p = {p_fit:.3f},"
                + (" " if standalone else "\n")
                + f"R² = {r2:.3f})"
            ),
        )
    ax_rate.set_ylabel("Hidden E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax_acc.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    if standalone:
        ax_acc.set_xlabel(
            "Measured $f_γ$ (Hz) — peak of trained-network population PSD",
            fontsize=theme.SIZE_LABEL,
        )
    else:
        for ax in axes:
            ax.set_xlabel("Measured $f_γ$ (Hz)", fontsize=theme.SIZE_LABEL)
    ax_acc.set_ylim(0, 100)
    ax_acc.axhline(10.0, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.6)
    for ax in (ax_rate, ax_acc):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=theme.SIZE_TICK)
    if not standalone:
        ax_rate.margins(x=0.15, y=0.35)
        ax_acc.set_xlim(ax_rate.get_xlim())
    if not standalone:
        ax_rate.set_ylim(bottom=0)
        for line in ax_rate.lines:
            if line.get_label().startswith("$r_E"):
                line.set_label("Affine fit")
    ax_rate.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    theme.label_panels((ax_rate, ax_acc))
    if standalone:
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)
        plt.close(fig)
