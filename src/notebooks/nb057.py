"""Notebook runner for entry 057 — the onset, empirics and theory in one figure.

A super-compound that merges two existing anchor figures:
  - nb054 Figure 1 (turn-on): the E-rate / I-rate / contrast maps over the
    W_EI × W_IE plane, with three example rasters (loop-off / weak / strong).
  - nb033 Figure 1 (bifurcation): the 4D mean-field Hopf crossing, the
    supercritical-reversible hysteresis sweep, and gamma frequency vs τ_GABA.

The empirical turn-on (what the network does) sits directly above the
theoretical bifurcation (why it does it), so the smooth onset and its
mean-field explanation read as one story. Data are recomputed by importing
the two source runners; no figures are copied.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "notebooks"))

from cli import theme  # noqa: E402
from helpers.modal import parse_modal_gpu  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402

SLUG = "nb057"
_, FIGURES = artifacts_and_figures(SLUG)

# Tiers only set the nb054 grid sim length (the nb033 numerics are fixed).
TIER_CONFIG = {
    "small": {"sim_ms": 750.0},
    "medium": {"sim_ms": 1000.0},
    "large": {"sim_ms": 2000.0},
}
DEFAULT_TIER = "medium"


def _load_runner(slug: str):
    """Import a sibling notebook runner as a module (without running main)."""
    path = REPO / "src" / "notebooks" / f"{slug}.py"
    spec = importlib.util.spec_from_file_location(slug, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_super_compound(grid, nb054, results, hopf, sweep, mf, meas, out_path, run_id):
    """3×3 super-compound: nb054 maps (row 0) + rasters (row 1) over the nb033
    bifurcation panels (row 2)."""
    from matplotlib.gridspec import GridSpec

    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"

    WEI = nb054.WEI_MEAN_GRID
    WIE = nb054.WIE_MEAN_GRID
    PTS = nb054.TURNON_POINTS
    DT_MS = nb054.DT_MS
    GRID_WIN_MS = nb054.GRID_WIN_MS

    # Tall super-compound — three panel rows; a deliberate exception to the
    # 16:9 house ratio (same reasoning as nb033's wide bifurcation strip).
    fig = plt.figure(figsize=(13.5, 12.0), dpi=150)
    gs = GridSpec(
        3, 3, figure=fig, height_ratios=[1.25, 0.92, 1.05],
        hspace=0.5, wspace=0.32, top=0.95, bottom=0.06, left=0.07, right=0.95,
    )

    # ── Row 0 — the three scalar maps (E rate, I rate, contrast + A/B/C) ──
    e = np.array([[c["rate_hz"] for c in row] for row in grid])
    i = np.array([[c["rate_i_hz"] for c in row] for row in grid])
    ct = np.array([[c["contrast"] for c in row] for row in grid])
    i_fin = i[np.isfinite(i)]
    i_cap = float(np.nanpercentile(i_fin, 92)) if i_fin.size else None
    e_max = float(np.nanmax(e)) if np.isfinite(e).any() else 1.0
    maps = [
        (e, "E firing rate (Hz)", e_max, "{:.0f}", False),
        (i, "I firing rate (Hz; edge clipped)", i_cap, "{:.0f}", False),
        (ct, "lobe–trough contrast", 1.0, "{:.2f}", True),
    ]
    for k, (vals, title, vmax_color, fmt, mark) in enumerate(maps):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(vals, origin="lower", aspect="equal", cmap="Greys",
                  vmin=0.0, vmax=vmax_color)
        xt = range(0, len(WEI), 2)
        ax.set_xticks(list(xt))
        ax.set_xticklabels([f"{WEI[t]:.1f}" for t in xt], fontsize=theme.SIZE_TICK - 1)
        if k == 0:
            yt = range(0, len(WIE), 2)
            ax.set_yticks(list(yt))
            ax.set_yticklabels([f"{WIE[t]:.1f}" for t in yt], fontsize=theme.SIZE_TICK - 1)
            ax.set_ylabel("W_IE mean (μS)", fontsize=theme.SIZE_LABEL)
        else:
            ax.set_yticks([])
        ax.set_xlabel("W_EI mean (μS)", fontsize=theme.SIZE_LABEL)
        ax.set_title(title, loc="center",
                     fontsize=theme.SIZE_LABEL - 1, fontweight="semibold")
        for iy in range(vals.shape[0]):
            for ix in range(vals.shape[1]):
                v = vals[iy, ix]
                if np.isfinite(v):
                    frac = (v / vmax_color) if vmax_color else 0.0
                    ax.text(ix, iy, fmt.format(v), ha="center", va="center",
                            fontsize=4.5, color="white" if frac > 0.55 else theme.INK_BLACK)
        if mark:
            for label, wei_i, wie_i in PTS:
                ax.scatter([wei_i], [wie_i], s=90, facecolor="white",
                           edgecolor=theme.INK_BLACK, linewidths=1.2, zorder=5)
                ax.text(wei_i, wie_i, label, ha="center", va="center", zorder=6,
                        fontsize=theme.SIZE_TICK, fontweight="bold", color=theme.INK_BLACK)

    # ── Row 1 — three representative rasters (A/B/C) ──
    for k, (label, wei_i, wie_i) in enumerate(PTS):
        ax = fig.add_subplot(gs[1, k])
        cell = grid[wie_i][wei_i]
        e_ = cell["e"]
        n_e = e_.shape[1]
        T = e_.shape[0]
        t_ms = np.arange(T) * DT_MS
        e_idx, e_t = np.where(e_.T)
        ax.scatter(t_ms[e_t], e_idx, s=0.5, c=theme.INK_BLACK, marker="|", linewidths=0.35)
        total = n_e
        if cell["i"] is not None:
            ii = cell["i"]
            n_i = ii.shape[1]
            i_idx, i_t = np.where(ii.T)
            ax.scatter(t_ms[i_t], n_e + i_idx, s=0.5, c=theme.DEEP_RED, marker="|", linewidths=0.35)
            ax.axhline(n_e, color=theme.GREY_MID, lw=0.5, alpha=0.6)
            total = n_e + n_i
        ax.set_ylim(0, total)
        ax.set_xlim(0, GRID_WIN_MS)
        ax.set_yticks([])
        ax.set_title(
            f"{label}   W_EI={WEI[wei_i]:.1f}, W_IE={WIE[wie_i]:.1f}"
            f"   contrast={cell['contrast']:.2f}",
            loc="left", fontsize=theme.SIZE_TICK - 1,
        )
        ax.set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
        _despine(ax)

    # ── Row 2 — the nb033 bifurcation panels (Hopf / hysteresis / frequency) ──
    axA = fig.add_subplot(gs[2, 0])
    xs = np.array([r["I_ext"] for r in results])
    eig_re = np.array([[ev[0] for ev in r["eigs"]] for r in results])
    eig_im = np.array([[ev[1] for ev in r["eigs"]] for r in results])
    sc = None
    for k in range(eig_re.shape[1]):
        sc = axA.scatter(eig_re[:, k], eig_im[:, k], c=xs, cmap="magma", s=4, linewidths=0)
    axA.axvline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    if hopf:
        w = hopf["omega_star"]
        axA.scatter([0, 0], [w, -w], facecolors="none",
                    edgecolors=theme.ELECTRIC_CYAN, s=60, lw=1.4, zorder=5)
    cbar = fig.colorbar(sc, ax=axA, fraction=0.046, pad=0.02)
    cbar.set_label("$I_\\text{ext}$ (nA)", fontsize=theme.SIZE_TICK - 1)
    cbar.ax.tick_params(labelsize=theme.SIZE_TICK - 1)
    axA.set_xlabel("Re$(\\lambda)$", fontsize=theme.SIZE_LABEL)
    axA.set_ylabel("Im$(\\lambda)$", fontsize=theme.SIZE_LABEL)
    axA.set_title(f"Hopf crossing at $I^\\star$ = {hopf['I_ext_star']:.2f} nA",
                  loc="left", fontsize=theme.SIZE_LABEL, fontweight="semibold")
    _despine(axA)

    axB = fig.add_subplot(gs[2, 1])
    axB.plot([d["I_ext"] for d in sweep["up"]], [d["amp"] for d in sweep["up"]],
             "o-", color=theme.INK_BLACK, lw=1.2, ms=4, label="drive ↑")
    axB.plot([d["I_ext"] for d in sweep["down"]], [d["amp"] for d in sweep["down"]],
             "s--", color=theme.DEEP_RED, lw=1.0, ms=4, markerfacecolor="none", label="drive ↓")
    axB.axvline(hopf["I_ext_star"], color=theme.AMBER, lw=0.6, ls=":")
    axB.set_xlabel("$I_\\text{ext}$ (nA)", fontsize=theme.SIZE_LABEL)
    axB.set_ylabel("E amplitude (pk-pk)", fontsize=theme.SIZE_LABEL)
    axB.set_title("Supercritical, reversible onset",
                  loc="left", fontsize=theme.SIZE_LABEL, fontweight="semibold")
    axB.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    _despine(axB)

    axC = fig.add_subplot(gs[2, 2])
    tg = [d["tau_gaba_ms"] for d in mf if d["f_star_Hz"] is not None]
    fs = [d["f_star_Hz"] for d in mf if d["f_star_Hz"] is not None]
    axC.plot(tg, fs, "o-", color=theme.INK_BLACK, lw=1.4, label="mean-field $f^\\star$")
    if meas:
        mt = sorted(meas)
        axC.plot(mt, [meas[t] for t in mt], "s--", color=theme.DEEP_RED, lw=1.3,
                 label="spiking $f_\\gamma$ (nb041)")
    axC.set_xlabel("$\\tau_\\text{GABA}$ (ms)", fontsize=theme.SIZE_LABEL)
    axC.set_ylabel("gamma frequency (Hz)", fontsize=theme.SIZE_LABEL)
    axC.set_title("Frequency from biophysics",
                  loc="left", fontsize=theme.SIZE_LABEL, fontweight="semibold")
    axC.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    _despine(axC)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    parse_modal_gpu(sys.argv)  # local CPU; modal unused.
    wipe_dir = "--no-wipe-dir" not in sys.argv
    sim_ms = TIER_CONFIG[tier]["sim_ms"]

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id} tier={tier}")
    prepare_run_dirs(SLUG, notebook_run_id, wipe=wipe_dir, make_artifacts=False)

    nb054 = _load_runner("nb054")
    nb033 = _load_runner("nb033")

    # nb054 — the empirical turn-on grid (private per-cell Poisson input).
    print("nb054 grid: running untrained PING networks over W_EI × W_IE …")
    priv_input = nb054.poisson_input(nb054.DT_MS, sim_ms, nb054.NET_N_E, nb054.NET_INPUT_RATE)
    grid = nb054.run_grid(priv_input)

    # nb033 — the mean-field bifurcation numerics.
    print("nb033 numerics: 4D mean-field sweep, Hopf, hysteresis, frequency …")
    I_grid = np.linspace(0.0, 4.0, 401)
    results = nb033.sweep(I_grid)
    hopf = nb033.find_hopf(results)
    criticality = nb033.hysteresis_sweep(hopf["I_ext_star"])
    mf_freq = nb033.frequency_vs_tau_gaba([4.5, 6.0, 9.0, 12.0, 18.0, 27.0], I_grid)
    meas_fgamma = nb033.load_nb041_fgamma()

    out = FIGURES / "onset_super_compound.png"
    build_super_compound(grid, nb054, results, hopf, criticality, mf_freq,
                         meas_fgamma, out, notebook_run_id)
    print(f"wrote {out}")
    print(f"done in {time.monotonic() - t_start:.1f}s")


if __name__ == "__main__":
    main()
