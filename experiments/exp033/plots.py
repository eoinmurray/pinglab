"""Draw retained measurements and coordinates; no numerical model execution."""

import matplotlib.pyplot as plt
import numpy as np
from experiments.helpers import theme

from .recipe import SIGMA_V_MV


def plot_hysteresis(sweep, hopf, out_path, run_id):
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    i_star = hopf["I_ext_star"]
    xu = [d["I_ext"] for d in sweep["up"]]
    yu = [d["amp"] for d in sweep["up"]]
    xd = [d["I_ext"] for d in sweep["down"]]
    yd = [d["amp"] for d in sweep["down"]]
    ax.plot(xu, yu, "o-", color=theme.INK_BLACK, lw=1.2, ms=5, label="drive increasing")
    ax.plot(
        xd,
        yd,
        "s--",
        color=theme.DEEP_RED,
        lw=1.0,
        ms=5,
        markerfacecolor="none",
        label="drive decreasing",
    )
    ax.axvline(i_star, color=theme.AMBER, lw=0.6, ls=":")
    ax.annotate(
        "no resolved hysteresis",
        xy=(i_star, 0.0),
        xytext=(i_star - 0.085, max(yu) * 0.55),
        fontsize=theme.SIZE_ANNOTATION,
        color=theme.GREY_DARK,
        ha="left",
        va="center",
    )
    ax.set_xlabel("$I_\\text{ext}$ (nA)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("E amplitude (pk-pk, ms$^{-1}$)", fontsize=theme.SIZE_LABEL)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_eigenvalues_complex(results, hopf, out_path, run_id):
    """The four 4D eigenvalues in the complex plane, coloured by drive."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    xs = np.array([r["I_ext"] for r in results])
    eig_re = np.array([[e[0] for e in r["eigs"]] for r in results])
    eig_im = np.array([[e[1] for e in r["eigs"]] for r in results])
    sc = None
    for k in range(eig_re.shape[1]):
        sc = ax.scatter(
            eig_re[:, k], eig_im[:, k], c=xs, cmap="magma", s=5, linewidths=0
        )
    assert sc is not None  # eig_re has 4 columns, so the loop always assigns sc
    ax.axvline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    if hopf:
        w = hopf["omega_star"]
        ax.scatter(
            [0, 0],
            [w, -w],
            facecolors="none",
            edgecolors=theme.ELECTRIC_CYAN,
            s=70,
            lw=1.4,
            zorder=5,
        )
        ax.annotate(
            f"crossing at $\\pm i\\omega^\\star$\n"
            f"$f^\\star = {hopf['freq_star_Hz']:.1f}$ Hz",
            xy=(0, w),
            xytext=(0.10 * eig_re.max(), w + 0.12 * w),
            fontsize=theme.SIZE_ANNOTATION,
            color=theme.GREY_DARK,
            ha="left",
            va="bottom",
            arrowprops=dict(arrowstyle="-", color=theme.ELECTRIC_CYAN, lw=0.8),
        )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("$I_\\text{ext}$ (nA)", fontsize=theme.SIZE_LABEL)
    ax.set_xlabel("Re$(\\lambda)$ (ms$^{-1}$)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Im$(\\lambda)$ (rad/ms)", fontsize=theme.SIZE_LABEL)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_limit_cycle(metrics, out_path, run_id):
    """4D limit cycle just above onset: E and I waveforms and the E→I lag."""
    theme.apply()
    I_ext, tt, E, I = (metrics[k] for k in ("I_ext", "t_ms", "E", "I"))
    print(
        f"  limit cycle at I={I_ext:.2f} nA: "
        f"absolute cross-correlation lag {metrics['e_leads_i_ms']:.2f} ms"
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax.plot(tt - tt[0], E, color=theme.INK_BLACK, lw=1.3, label="$E$")
    ax.set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("$E$ rate", fontsize=theme.SIZE_LABEL, color=theme.INK_BLACK)
    ax2 = ax.twinx()
    ax2.plot(tt - tt[0], I, color=theme.DEEP_RED, lw=1.3, label="$I$")
    ax2.set_ylabel("$I$ rate", fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {k: metrics[k] for k in ("I_ext", "e_leads_i_ms", "e_peak_to_peak")}


def plot_frequency_vs_tau_gaba(mf, meas, out_path, run_id):
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    tg = [d["tau_gaba_ms"] for d in mf if d["f_star_Hz"] is not None]
    fs = [d["f_star_Hz"] for d in mf if d["f_star_Hz"] is not None]
    ax.plot(
        tg,
        fs,
        "o-",
        color=theme.INK_BLACK,
        lw=1.4,
        label="reference mean-field $f^\\star$",
    )
    if meas:
        mt = sorted(meas)
        ax.plot(
            mt,
            [meas[t] for t in mt],
            "s--",
            color=theme.DEEP_RED,
            lw=1.3,
            label="spiking $f_\\gamma$",
        )
    ax.set_xlabel("$\\tau_\\text{GABA}$ (ms)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("gamma frequency (Hz)", fontsize=theme.SIZE_LABEL)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_phase_planes(coordinates, out_path, run_id):
    """Project the trajectory; projections alone do not establish a centre manifold."""
    theme.apply()
    Y = coordinates["Y"]
    labels = ["$E$", "$I$", "$g_e^I$", "$g_i^E$"]
    pairs = [(0, 1), (2, 3), (0, 3), (1, 2), (0, 2), (1, 3)]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.5), dpi=150)
    for ax, (a, b) in zip(axes.flat, pairs):
        ax.plot(Y[a], Y[b], color=theme.INK_BLACK, lw=1.0)
        ax.set_xlabel(labels[a], fontsize=theme.SIZE_LABEL)
        ax.set_ylabel(labels[b], fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_timeseries(coordinates, out_path, run_id):
    """The four state variables over the limit cycle: E, g_e^I, I, g_i^E in loop order, sharing a time axis, so
    the round-trip phase lags E -> g_e^I -> I -> g_i^E -> E are visible."""
    theme.apply()
    tt, Y = coordinates["t_ms"], coordinates["Y"]
    t = tt - tt[0]
    # loop order E -> g_e^I -> I -> g_i^E; one trace per panel, so near-black ink
    # throughout (colour would separate nothing — H13).
    rows = [
        (0, "$E$ rate", theme.INK_BLACK),
        (2, "$g_e^I$", theme.INK_BLACK),
        (1, "$I$ rate", theme.INK_BLACK),
        (3, "$g_i^E$", theme.INK_BLACK),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(8.0, 6.5), dpi=150, sharex=True)
    for ax, (idx, lab, col) in zip(axes, rows):
        ax.plot(t, Y[idx], color=col, lw=1.4)
        ax.set_ylabel(lab, fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reduction_ladder(hopf4, hopf3, coordinates, out_path, run_id):
    """g_i^E divergence after a kick at a common supra-threshold drive for
    the full 4D model, the 3D (AMPA-slaved) and 2D (rate-slaved) reductions.
    4D and 3D sustain; 2D rings down — the Hopf survives to 3D, not 2D."""
    theme.apply()
    t4, d4 = coordinates["4d"]["t_ms"], coordinates["4d"]["deviation"]
    t3, d3 = coordinates["3d"]["t_ms"], coordinates["3d"]["deviation"]
    t2, d2 = coordinates["2d"]["t_ms"], coordinates["2d"]["deviation"]
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax.plot(
        t4,
        d4,
        color=theme.INK_BLACK,
        lw=1.4,
        label=f"4D full — Hopf ($f^\\star$ = {hopf4['freq_star_Hz']:.0f} Hz)",
    )
    ax.plot(
        t3,
        d3,
        color=theme.INK_BLACK,
        lw=1.4,
        ls="--",
        label=f"3D, AMPA slaved — Hopf ($f^\\star$ = {hopf3['freq_star_Hz']:.0f} Hz)"
        if hopf3
        else "3D, AMPA slaved",
    )
    ax.plot(
        t2,
        d2,
        color=theme.DEEP_RED,
        lw=1.6,
        label="2D, rates slaved — rings down (no Hopf)",
    )
    ax.axhline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    ax.set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("$g_i^E$ deviation from fixed point", fontsize=theme.SIZE_LABEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_bifurcation_compound(results, hopf, sweep, mf, meas, out_path, run_id):
    """Claim-3 anchor: the recruitment cliff as a predictable Hopf bifurcation.
    A — the 4D eigenvalue pair crossing into the right half-plane at I*.
    B — the hysteresis sweep (supercritical, reversible onset).
    C — gamma frequency vs τ_GABA, reference mean-field vs exp041 spiking."""
    theme.apply()
    from matplotlib.gridspec import GridSpec

    # Wide 3:1 strip — a 1×3 panel row reads better near-square per panel than
    # squeezed portrait into 16:9 (deliberate exception to the house ratio).
    fig = plt.figure(figsize=(13.5, 4.6), dpi=150)
    gs = GridSpec(
        1, 3, figure=fig, wspace=0.42, top=0.86, bottom=0.16, left=0.06, right=0.97
    )

    # A — eigenvalues in the complex plane, coloured by drive
    axA = fig.add_subplot(gs[0, 0])
    xs = np.array([r["I_ext"] for r in results])
    eig_re = np.array([[e[0] for e in r["eigs"]] for r in results])
    eig_im = np.array([[e[1] for e in r["eigs"]] for r in results])
    sc = None
    for k in range(eig_re.shape[1]):
        sc = axA.scatter(
            eig_re[:, k], eig_im[:, k], c=xs, cmap="magma", s=4, linewidths=0
        )
    assert sc is not None  # eig_re has 4 columns, so the loop always assigns sc
    axA.axvline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    if hopf:
        w = hopf["omega_star"]
        axA.scatter(
            [0, 0],
            [w, -w],
            facecolors="none",
            edgecolors=theme.ELECTRIC_CYAN,
            s=60,
            lw=1.4,
            zorder=5,
        )
    cbar = fig.colorbar(sc, ax=axA, fraction=0.046, pad=0.02)
    cbar.set_label("$I_\\text{ext}$ (nA)", fontsize=theme.SIZE_TICK - 1)
    cbar.ax.tick_params(labelsize=theme.SIZE_TICK - 1)
    axA.set_xlabel("Re$(\\lambda)$ (ms$^{-1}$)", fontsize=theme.SIZE_LABEL)
    axA.set_ylabel("Im$(\\lambda)$ (rad/ms)", fontsize=theme.SIZE_LABEL)
    axA.set_title(
        f"A  Hopf crossing at $I^\\star$ = {hopf['I_ext_star']:.2f} nA",
        loc="left",
        fontsize=theme.SIZE_LABEL,
        fontweight="semibold",
    )
    _despine(axA)

    # B — hysteresis sweep
    axB = fig.add_subplot(gs[0, 1])
    xu = [d["I_ext"] for d in sweep["up"]]
    yu = [d["amp"] for d in sweep["up"]]
    xd = [d["I_ext"] for d in sweep["down"]]
    yd = [d["amp"] for d in sweep["down"]]
    axB.plot(xu, yu, "o-", color=theme.INK_BLACK, lw=1.2, ms=4, label="drive ↑")
    axB.plot(
        xd,
        yd,
        "s--",
        color=theme.DEEP_RED,
        lw=1.0,
        ms=4,
        markerfacecolor="none",
        label="drive ↓",
    )
    axB.axvline(hopf["I_ext_star"], color=theme.AMBER, lw=0.6, ls=":")
    axB.set_xlabel("$I_\\text{ext}$ (nA)", fontsize=theme.SIZE_LABEL)
    axB.set_ylabel("E amplitude (pk-pk, ms$^{-1}$)", fontsize=theme.SIZE_LABEL)
    axB.set_title(
        "B  Reversible sampled onset",
        loc="left",
        fontsize=theme.SIZE_LABEL,
        fontweight="semibold",
    )
    axB.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    _despine(axB)

    # C — frequency vs τ_GABA: mean-field prediction vs spiking
    axC = fig.add_subplot(gs[0, 2])
    tg = [d["tau_gaba_ms"] for d in mf if d["f_star_Hz"] is not None]
    fs = [d["f_star_Hz"] for d in mf if d["f_star_Hz"] is not None]
    axC.plot(tg, fs, "o-", color=theme.INK_BLACK, lw=1.4, label="mean-field $f^\\star$")
    if meas:
        mt = sorted(meas)
        axC.plot(
            mt,
            [meas[t] for t in mt],
            "s--",
            color=theme.DEEP_RED,
            lw=1.3,
            label="spiking $f_\\gamma$",
        )
    axC.set_xlabel("$\\tau_\\text{GABA}$ (ms)", fontsize=theme.SIZE_LABEL)
    axC.set_ylabel("gamma frequency (Hz)", fontsize=theme.SIZE_LABEL)
    axC.set_title(
        "C  Frequency from biophysics",
        loc="left",
        fontsize=theme.SIZE_LABEL,
        fontweight="semibold",
    )
    axC.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    _despine(axC)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sigma_sensitivity(sensitivity, out_path, run_id):
    """Compact audit of threshold, frequency, onset state, and amplitude."""
    rows = [row for row in sensitivity["rows"] if row["hopf_exists"]]
    sigma = [row["sigma_V_mV"] for row in rows]
    theme.apply()
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 4.5), dpi=150)
    ax_threshold, ax_frequency, ax_fixed, ax_amplitude = axes.flat
    ax_threshold.plot(
        sigma, [r["hopf"]["I_ext_star"] for r in rows], "o-", color=theme.INK_BLACK
    )
    ax_threshold.set_ylabel("$I^\\star_\\text{ext}$ (nA)")
    ax_frequency.plot(
        sigma, [r["hopf"]["freq_star_Hz"] for r in rows], "o-", color=theme.INK_BLACK
    )
    ax_frequency.set_ylabel("$f^\\star$ (Hz)")
    values = [r["hopf"]["freq_star_Hz"] for r in rows]
    # Show absolute frequency; numerical noise must not fill the vertical axis.
    if values:
        ax_frequency.set_ylim(0, max(40.0, 1.1 * max(values)))
    ax_frequency.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax_fixed.plot(
        sigma,
        [1000 * r["fixed_point_at_hopf"]["E_per_ms"] for r in rows],
        "o-",
        color=theme.INK_BLACK,
        label="E",
    )
    ax_fixed.plot(
        sigma,
        [1000 * r["fixed_point_at_hopf"]["I_per_ms"] for r in rows],
        "s--",
        color=theme.DEEP_RED,
        label="I",
    )
    ax_fixed.set_ylabel("fixed-point rate (Hz)")
    ax_fixed.legend(frameon=False, fontsize=theme.SIZE_LEGEND)
    ax_amplitude.plot(
        sigma,
        [1000 * r["limit_cycle"]["e_peak_to_peak"] for r in rows],
        "o-",
        color=theme.INK_BLACK,
    )
    ax_amplitude.set_ylabel("E amplitude (Hz, pk-pk)")
    verdict = (
        "onset test retained"
        if sensitivity["supercritical_retained"]
        else "verdict changes"
    )
    ax_amplitude.text(
        0.98,
        0.95,
        verdict,
        transform=ax_amplitude.transAxes,
        ha="right",
        va="top",
        fontsize=theme.SIZE_ANNOTATION,
    )
    for ax in axes.flat:
        ax.set_xlabel("$\\sigma_V$ (mV)")
        ax.axvline(SIGMA_V_MV, color=theme.GREY_MID, lw=0.7, ls=":")
        _despine(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
