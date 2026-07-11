"""Notebook runner for entry 054 — A PING rhythmicity metric.

The lobe–trough contrast of the spike-time autocorrelation, applied to untrained
PING networks across the recurrent-weight plane. Each E cell is driven by its OWN
private Poisson channel (one-to-one W_in), so cells share no input — this removes
the input-driven coincidence that otherwise inflates the metric at low firing and
makes it rate-invariant by construction (no post-hoc correction needed).

    A(ℓ) = (1/⟨r⟩²)(1/(n−ℓ)) Σ_t r(t) r(t+ℓ)          (chance = 1)
    contrast = (lobe − trough)/(lobe + trough) ∈ [0, 1)

W_EI (E recruits I) and W_IE (I inhibits E) are swept independently: the two zero
edges are the COBA control (no inhibitory loop), the interior is the PING regime.
The rate-invariance check contrasts private input (flat null) against the rejected
shared-input design (a spurious low-rate hat).

Measurement only: no training, local CPU, no GPU.

Writing: writings/exp054.typ · figures + numbers.json: artifacts/data/exp054/
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.figsave import save_figure  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.rhythmicity import (  # noqa: E402
    iei_histogram,
    population_event_times,
    rhythmicity_scalars,
    spike_autocorrelogram,
)
from helpers.run_cli import run_cli  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp054"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

# ─── recipe (hardcoded literals; the notebook IS the recipe) ────────────
DT_MS = 0.25
MAX_LAG_MS = 100.0
BIN_MS = 1.0

# Untrained PING networks. Each E cell has its OWN private Poisson input channel
# (one-to-one W_in = identity·PRIVATE_W_IN), so no two cells share input.
NET_N_E = 256
NET_INPUT_RATE = 100.0   # Hz, per private channel
PRIVATE_W_IN = 0.5       # identity input weight (one channel → one E cell)
NET_BURN_MS = 100.0

# Rejected shared-input design, kept only for the rate-invariance comparison: a
# sparse many-to-many W_in where ≈13 E cells share each of 200 channels.
SHARED_N_IN = 200
SHARED_W_IN = 0.2
SHARED_W_IN_SP = 0.95

# Independent 2-D sweep of the recurrent weight means (W_EI vs W_IE), 11×11 = 121.
WEI_MEAN_GRID = [round(v, 3) for v in np.linspace(0.0, 3.0, 11)]   # step 0.3
WIE_MEAN_GRID = [round(v, 3) for v in np.linspace(0.0, 6.0, 11)]   # step 0.6
GRID_WIN_MS = 200.0      # raster display window
GRID_E_SHOW = 160        # E cells drawn per raster panel
GRID_I_SHOW = 48         # I cells drawn per raster panel
# The heatmaps show all 100 cells; the per-cell raster/autocorr grids would be
# unreadable at 100 panels, so they display every other cell (a 5×5 subset).
GRID_DISPLAY_STRIDE = 2


def _display_idx(n):
    return list(range(0, n, GRID_DISPLAY_STRIDE))

# Null scans (W_EI = W_IE = 0, no rhythm) for the rate-invariance check. Input
# rates are chosen so the achieved E rate spans ≈1–90 Hz under each input design.
PRIVATE_NULL_INPUT_HZ = [1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 70.0, 100.0]
SHARED_NULL_INPUT_HZ = [8.0, 12.0, 16.0, 20.0, 28.0, 40.0, 60.0, 100.0]

# Simulation length (more spikes ⇒ cleaner autocorrelogram); this is what the
# frozen figures use (the retired "medium" tier).
SIM_MS = 1000.0

# Run scale — stamped into the manifest by run_dirs.prepare and rendered as
# the Methods table via RunScale; the mdx never restates these numbers.
SCALE = {
    "input": "private per-cell Poisson",
    "t_ms": SIM_MS,
    "dt_ms": DT_MS,
    "input_rate_hz": NET_INPUT_RATE,
    "cells": len(WEI_MEAN_GRID) * len(WIE_MEAN_GRID),
    "grid": f"{len(WEI_MEAN_GRID)} W_EI x {len(WIE_MEAN_GRID)} W_IE",
}


def ping_spikes(wei, wie, rate_hz, sim_ms, dt, private=True):
    """Build an untrained PING (W_EI mean wei, W_IE mean wie) via the CLI probe,
    drive it with uniform Poisson at rate_hz for one long trial, and return
    (E raster, I raster) past the burn-in.

    private=True wires each E cell to its own input channel (identity W_in);
    private=False uses the sparse shared W_in. Network build + forward run in the
    CLI; the notebook reconstructs the single-trial raster from rasters.npz.
    """
    n_ch = NET_N_E if private else SHARED_N_IN
    tag = f"{'priv' if private else 'shared'}_wei{wei:g}_wie{wie:g}_r{rate_hz:g}_T{sim_ms:g}"
    out_dir = (ARTIFACTS / "probe" / tag).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "sim",
        "--input", "synthetic-spikes",
        "--model", "ping",
        "--n-hidden", str(NET_N_E),
        "--n-inh", str(NET_N_E),  # equal-fan-in: n_i = n_e
        "--n-in", str(n_ch),
        "--w-ei-mean", str(wei),
        "--w-ie-mean", str(wie),
        "--input-rate", str(rate_hz),
        "--n-batch", "1",
        "--t-ms", str(sim_ms),
        "--dt", str(dt),
        "--seed", "42",
        "--outputs", "rasters",
        "--out-dir", str(out_dir),
    ]
    if private:
        cmd += ["--private-w-in", "--w-in", str(PRIVATE_W_IN)]
    else:
        cmd += ["--w-in", str(SHARED_W_IN), "--w-in-sparsity", str(SHARED_W_IN_SP)]
    run_cli(cmd)

    R = np.load(out_dir / "rasters.npz")
    T, n_e, n_i = int(R["T"]), int(R["n_e"]), int(R["n_i"])

    def _dense(prefix, N):
        m = R[f"{prefix}_trial"] == 0
        d = np.zeros((T, N), dtype=np.int8)
        d[R[f"{prefix}_t"][m], R[f"{prefix}_cell"][m]] = 1
        return d

    b = int(NET_BURN_MS / dt)
    spk = _dense("e", n_e)[b:]
    spk_i = _dense("i", n_i)[b:] if n_i > 0 else None
    return spk, spk_i


def _score(spk, dt):
    """Contrast, located lobe/trough lags, firing rate and autocorrelogram of an
    E-population raster."""
    rate = float(spk.sum() / (spk.shape[1] * spk.shape[0] * dt / 1000.0))
    ac_lags, ac = spike_autocorrelogram(spk, dt, MAX_LAG_MS, BIN_MS)
    iei_lags, iei = iei_histogram(population_event_times(spk, dt), MAX_LAG_MS, BIN_MS)
    sc = rhythmicity_scalars(ac_lags, ac, iei_lags, iei, BIN_MS)
    return dict(
        rate=rate, ac_lags=ac_lags, ac=ac,
        contrast=sc["contrast"] if sc["contrast"] is not None else np.nan,
        lobe_lag=sc["lobe_lag"], trough_lag=sc["trough_lag"],
    )


def run_grid_point(wei, wie, rate_hz, sim_ms, dt=DT_MS):
    """One (W_EI mean, W_IE mean) cell with private input: contrast + E/I rates,
    autocorrelogram, and a small raster for display."""
    spk, spk_i = ping_spikes(wei, wie, rate_hz, sim_ms, dt, private=True)
    s = _score(spk, dt)
    rate_i = (float(spk_i.sum() / (spk_i.shape[1] * spk_i.shape[0] * dt / 1000.0))
              if spk_i is not None else np.nan)
    win = int(GRID_WIN_MS / dt)
    return dict(
        contrast=s["contrast"], rate_hz=s["rate"], rate_i_hz=rate_i,
        ac_lags=s["ac_lags"], ac=s["ac"],
        lobe_lag=s["lobe_lag"], trough_lag=s["trough_lag"],
        e=spk[:win, :GRID_E_SHOW],
        i=spk_i[:win, :GRID_I_SHOW] if spk_i is not None else None,
    )


def run_grid(rate_hz, sim_ms):
    """grid[wie_idx][wei_idx] over WIE_MEAN_GRID × WEI_MEAN_GRID (wie_idx 0 = lowest)."""
    return [[run_grid_point(wei, wie, rate_hz, sim_ms) for wei in WEI_MEAN_GRID]
            for wie in WIE_MEAN_GRID]


def null_scan(sim_ms, private):
    """Scan input rate on a NULL network (W_EI = W_IE = 0, no rhythm at any drive),
    private or shared input. Returns per-rate records sorted by E rate."""
    inputs = PRIVATE_NULL_INPUT_HZ if private else SHARED_NULL_INPUT_HZ
    recs = []
    for r in inputs:
        spk, _ = ping_spikes(0.0, 0.0, r, sim_ms, DT_MS, private=private)
        recs.append(_score(spk, DT_MS))
    recs.sort(key=lambda d: d["rate"])
    return recs


def _contrast_heatmap(vals, title, cbar_label, out_path):
    """A contrast-valued heatmap over the W_EI × W_IE plane (magma, 0→1)."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    im = ax.imshow(vals, origin="lower", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(WEI_MEAN_GRID)))
    ax.set_xticklabels([f"{v:.1f}" for v in WEI_MEAN_GRID])
    ax.set_yticks(range(len(WIE_MEAN_GRID)))
    ax.set_yticklabels([f"{v:.1f}" for v in WIE_MEAN_GRID])
    ax.set_xlabel("W_EI mean (μS)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("W_IE mean (μS)", fontsize=theme.SIZE_LABEL)
    for iy in range(vals.shape[0]):
        for ix in range(vals.shape[1]):
            v = vals[iy, ix]
            if np.isfinite(v):
                ax.text(ix, iy, f"{v:.2f}", ha="center", va="center",
                        fontsize=theme.SIZE_ANNOTATION,
                        color="white" if v > 0.45 else theme.INK_BLACK)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=theme.SIZE_LABEL)
    ax.set_title(title, fontsize=theme.SIZE_TITLE, color=theme.INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_grid_heatmap(grid, out_path):
    """lobe–trough contrast over the W_EI × W_IE plane."""
    ct = np.array([[c["contrast"] for c in row] for row in grid])  # [wie, wei]
    _contrast_heatmap(ct, "Contrast over W_EI × W_IE", "lobe–trough contrast", out_path)


def _rate_heatmap(vals, title, cbar_label, out_path):
    """One firing-rate heatmap over the W_EI × W_IE grid (Hz, viridis)."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    im = ax.imshow(vals, origin="lower", aspect="auto")
    ax.set_xticks(range(len(WEI_MEAN_GRID)))
    ax.set_xticklabels([f"{v:.1f}" for v in WEI_MEAN_GRID])
    ax.set_yticks(range(len(WIE_MEAN_GRID)))
    ax.set_yticklabels([f"{v:.1f}" for v in WIE_MEAN_GRID])
    ax.set_xlabel("W_EI mean (μS)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("W_IE mean (μS)", fontsize=theme.SIZE_LABEL)
    vmax = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
    for iy in range(vals.shape[0]):
        for ix in range(vals.shape[1]):
            v = vals[iy, ix]
            if np.isfinite(v):
                ax.text(ix, iy, f"{v:.0f}", ha="center", va="center",
                        fontsize=theme.SIZE_ANNOTATION,
                        color="white" if v > 0.5 * vmax else theme.INK_BLACK)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=theme.SIZE_LABEL)
    ax.set_title(title, fontsize=theme.SIZE_TITLE, color=theme.INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_grid_rate_e(grid, out_path):
    """E firing-rate heatmap — the gamma-gated sparsity behind the contrast map."""
    e = np.array([[c["rate_hz"] for c in row] for row in grid])
    _rate_heatmap(e, "E firing rate over W_EI × W_IE", "E rate (Hz)", out_path)


def fig_grid_rate_i(grid, out_path):
    """I firing-rate heatmap."""
    i = np.array([[c["rate_i_hz"] for c in row] for row in grid])
    _rate_heatmap(i, "I firing rate over W_EI × W_IE", "I rate (Hz)", out_path)


def fig_grid_rasters(grid, out_path):
    """E/I rasters for a 5×5 subset of the grid (E black, I red)."""
    wie_disp, wei_disp = _display_idx(len(WIE_MEAN_GRID)), _display_idx(len(WEI_MEAN_GRID))
    nr, nc = len(wie_disp), len(wei_disp)
    fig, axes = plt.subplots(nr, nc, figsize=(2.0 * nc, 2.0 * nc * 9 / 16), dpi=150, squeeze=False)
    for r in range(nr):  # display top→bottom = high→low W_IE
        wie_idx = wie_disp[nr - 1 - r]
        for c in range(nc):
            wei_idx = wei_disp[c]
            ax = axes[r][c]
            cell = grid[wie_idx][wei_idx]
            e = cell["e"]
            ne = e.shape[1]
            ax.eventplot([np.nonzero(e[:, k])[0] * DT_MS for k in range(ne)],
                         colors=theme.INK, linewidths=0.5,
                         lineoffsets=np.arange(ne), linelengths=1.0)
            total = ne
            if cell["i"] is not None:
                ii = cell["i"]
                ni = ii.shape[1]
                ax.eventplot([np.nonzero(ii[:, k])[0] * DT_MS for k in range(ni)],
                             colors=theme.DEEP_RED, linewidths=0.5,
                             lineoffsets=np.arange(ne, ne + ni), linelengths=1.0)
                total = ne + ni
            ax.set_xlim(0, GRID_WIN_MS)
            ax.set_ylim(0, total)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"{WIE_MEAN_GRID[wie_idx]:.1f}", fontsize=theme.SIZE_TICK,
                              rotation=0, ha="right", va="center")
            if r == nr - 1:
                ax.set_xlabel(f"{WEI_MEAN_GRID[wei_idx]:.1f}", fontsize=theme.SIZE_TICK)
    fig.supxlabel("W_EI mean (μS)", fontsize=theme.SIZE_LABEL)
    fig.supylabel("W_IE mean (μS)", fontsize=theme.SIZE_LABEL)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_grid_autocorr(grid, out_path):
    """E-population autocorrelogram A(ℓ) per grid cell, matching the heatmap — the
    Mexican hat the contrast is read from, with the located lobe (▲) and trough (▼)."""
    wie_disp, wei_disp = _display_idx(len(WIE_MEAN_GRID)), _display_idx(len(WEI_MEAN_GRID))
    nr, nc = len(wie_disp), len(wei_disp)
    fig, axes = plt.subplots(nr, nc, figsize=(2.0 * nc, 2.0 * nc * 9 / 16), dpi=150, squeeze=False)
    allac = np.concatenate([c["ac"][1:][np.isfinite(c["ac"][1:])] for row in grid for c in row])
    ymax = float(np.nanmax(allac)) if allac.size else 3.0
    for r in range(nr):  # display top→bottom = high→low W_IE
        wie_idx = wie_disp[nr - 1 - r]
        for cc in range(nc):
            wei_idx = wei_disp[cc]
            ax = axes[r][cc]
            cell = grid[wie_idx][wei_idx]
            ac = cell["ac"]
            ax.axhline(1.0, color=theme.FAINT, lw=0.6, ls=":")
            ax.plot(cell["ac_lags"], ac, color=theme.INK, lw=0.9)
            if cell["lobe_lag"] is not None:
                ax.plot(cell["lobe_lag"], ac[int(round(cell["lobe_lag"] / BIN_MS))],
                        "^", color=theme.INK_BLACK, ms=5, zorder=5)
            if cell["trough_lag"] is not None:
                ax.plot(cell["trough_lag"], ac[int(round(cell["trough_lag"] / BIN_MS))],
                        "v", color=theme.DEEP_RED, ms=5, zorder=5)
            ax.set_xlim(-2, 50)
            ax.set_ylim(-0.06 * ymax, ymax * 1.08)
            ax.set_xticks([])
            ax.set_yticks([])
            if cc == 0:
                ax.set_ylabel(f"{WIE_MEAN_GRID[wie_idx]:.1f}", fontsize=theme.SIZE_TICK,
                              rotation=0, ha="right", va="center")
            if r == nr - 1:
                ax.set_xlabel(f"{WEI_MEAN_GRID[wei_idx]:.1f}", fontsize=theme.SIZE_TICK)
    fig.supxlabel("W_EI mean (μS)", fontsize=theme.SIZE_LABEL)
    fig.supylabel("W_IE mean (μS)", fontsize=theme.SIZE_LABEL)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _panel_letter(ax, letter):
    """Stamp a prominent bold panel identifier (A, B, …) in the top-left corner of
    ax, on a small white plate so it stays legible over dense rasters/heatmaps."""
    ax.text(
        0.035, 0.965, letter, transform=ax.transAxes, ha="left", va="top",
        fontsize=theme.SIZE_TITLE + 4, fontweight="bold", color=theme.INK_BLACK,
        zorder=20,
        bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none", alpha=0.85),
    )


# Three representative (W_EI idx, W_IE idx) points for the turn-on callouts.
# Three points along the W_IE = 2·W_EI diagonal (index (i, i), since W_IE spans
# twice the W_EI range): the loop fully off, weak coupling below the 0.5 contrast
# mark, and the strong-coupling corner — a single line through the turn-on.
TURNON_POINTS = [
    ("A", 0, 0),    # loop off: W_EI = W_IE = 0, neither loop engaged
    ("B", 2, 2),    # weak coupling on the diagonal, contrast < 0.5
    ("C", 10, 10),  # strong coupling corner, sharp volleys
]


def fig_turnon_compound(grid, out_path):
    """Claim-1 anchor: the lobe–trough contrast heatmap over W_EI × W_IE with
    three representative rasters (loop-off / weak / strong) called out, so the
    quantitative turn-on map sits beside what off/weak/strong actually look
    like — without the full 6×6 raster grid."""
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(12, 6.75), dpi=150)  # 16:9
    gs = GridSpec(
        3, 2, figure=fig, width_ratios=[1.7, 1.0],
        hspace=0.55, wspace=0.16, top=0.92, bottom=0.11, left=0.07, right=0.97,
    )

    # Contrast heatmap (left, spans all rows) — the turn-on map.
    ct = np.array([[c["contrast"] for c in row] for row in grid])  # [wie, wei]
    ax_hm = fig.add_subplot(gs[:, 0])
    im = ax_hm.imshow(ct, origin="lower", aspect="auto", cmap="Greys", vmin=0.0, vmax=1.0)
    ax_hm.set_xticks(range(len(WEI_MEAN_GRID)))
    ax_hm.set_xticklabels([f"{v:.1f}" for v in WEI_MEAN_GRID], fontsize=theme.SIZE_TICK)
    ax_hm.set_yticks(range(len(WIE_MEAN_GRID)))
    ax_hm.set_yticklabels([f"{v:.1f}" for v in WIE_MEAN_GRID], fontsize=theme.SIZE_TICK)
    ax_hm.set_xlabel("W_EI mean (μS)", fontsize=theme.SIZE_LABEL)
    ax_hm.set_ylabel("W_IE mean (μS)", fontsize=theme.SIZE_LABEL)
    cb = fig.colorbar(im, ax=ax_hm, pad=0.015)
    cb.set_label("lobe–trough contrast", fontsize=theme.SIZE_LABEL)
    for iy in range(ct.shape[0]):
        for ix in range(ct.shape[1]):
            v = ct[iy, ix]
            if np.isfinite(v):
                ax_hm.text(ix, iy, f"{v:.2f}", ha="center", va="center",
                           fontsize=5.5, color="white" if v > 0.55 else theme.INK_BLACK)
    for label, wei_i, wie_i in TURNON_POINTS:
        ax_hm.scatter([wei_i], [wie_i], s=150, facecolor="white",
                      edgecolor=theme.INK_BLACK, linewidths=1.4, zorder=5)
        ax_hm.text(wei_i, wie_i, label, ha="center", va="center", zorder=6,
                   fontsize=theme.SIZE_LABEL - 1, fontweight="bold", color=theme.INK_BLACK)

    # Three representative rasters (right), E black below / I red above.
    for k, (label, wei_i, wie_i) in enumerate(TURNON_POINTS):
        ax = fig.add_subplot(gs[k, 1])
        cell = grid[wie_i][wei_i]
        e = cell["e"]
        n_e = e.shape[1]
        T = e.shape[0]
        t_ms = np.arange(T) * DT_MS
        e_idx, e_t = np.where(e.T)
        ax.scatter(t_ms[e_t], e_idx, s=0.6, c=theme.INK_BLACK, marker="|", linewidths=0.4)
        total = n_e
        if cell["i"] is not None:
            ii = cell["i"]
            n_i = ii.shape[1]
            i_idx, i_t = np.where(ii.T)
            ax.scatter(t_ms[i_t], n_e + i_idx, s=0.6, c=theme.DEEP_RED, marker="|", linewidths=0.4)
            ax.axhline(n_e, color=theme.GREY_MID, lw=0.5, alpha=0.6)
            total = n_e + n_i
        ax.set_ylim(0, total)
        ax.set_xlim(0, GRID_WIN_MS)
        ax.set_yticks([])
        ax.set_title(
            f"{label}   W_EI={WEI_MEAN_GRID[wei_i]:.1f}, W_IE={WIE_MEAN_GRID[wie_i]:.1f}"
            f"   contrast={cell['contrast']:.2f}",
            loc="left", fontsize=theme.SIZE_TICK,
        )
        if k == len(TURNON_POINTS) - 1:
            ax.set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
        else:
            ax.tick_params(labelbottom=False)
        _despine(ax)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_grid_maps_compound(grid, out_path):
    """Three scalar maps over the W_EI × W_IE plane, side by side: E firing
    rate, I firing rate, and lobe–trough contrast — the per-cell summaries the
    rasters and autocorrelograms reduce to, on shared axes."""
    e = np.array([[c["rate_hz"] for c in row] for row in grid])
    i = np.array([[c["rate_i_hz"] for c in row] for row in grid])
    ct = np.array([[c["contrast"] for c in row] for row in grid])
    # The I-rate runaway edge (W_IE = 0) dominates a linear scale; cap the
    # colour range so the interior is legible (the edge saturates darkest).
    i_fin = i[np.isfinite(i)]
    i_cap = float(np.nanpercentile(i_fin, 92)) if i_fin.size else None

    e_max = float(np.nanmax(e)) if np.isfinite(e).any() else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(12, 6.75), dpi=150)
    panels = [
        (axes[0], e, "E firing rate (Hz)", e_max, "{:.0f}"),
        (axes[1], i, "I firing rate (Hz; edge clipped)", i_cap, "{:.0f}"),
        (axes[2], ct, "lobe–trough contrast", 1.0, "{:.2f}"),
    ]
    for k, (ax, vals, title, vmax_color, fmt) in enumerate(panels):
        # Square panels (square cells), grayscale; every cell carries its number.
        ax.imshow(vals, origin="lower", aspect="equal", cmap="Greys",
                  vmin=0.0, vmax=vmax_color)
        xt = range(0, len(WEI_MEAN_GRID), 2)
        ax.set_xticks(list(xt))
        ax.set_xticklabels([f"{WEI_MEAN_GRID[t]:.1f}" for t in xt], fontsize=theme.SIZE_TICK - 1)
        if k == 0:
            yt = range(0, len(WIE_MEAN_GRID), 2)
            ax.set_yticks(list(yt))
            ax.set_yticklabels([f"{WIE_MEAN_GRID[t]:.1f}" for t in yt], fontsize=theme.SIZE_TICK - 1)
            ax.set_ylabel("W_IE mean (μS)", fontsize=theme.SIZE_LABEL)
        else:
            ax.set_yticks([])
        ax.set_xlabel("W_EI mean (μS)", fontsize=theme.SIZE_LABEL)
        ax.set_title(title, loc="center", fontsize=theme.SIZE_LABEL - 1, fontweight="semibold")
        for iy in range(vals.shape[0]):
            for ix in range(vals.shape[1]):
                v = vals[iy, ix]
                if np.isfinite(v):
                    frac = (v / vmax_color) if vmax_color else 0.0
                    ax.text(ix, iy, fmt.format(v), ha="center", va="center",
                            fontsize=4.5, color="white" if frac > 0.55 else theme.INK_BLACK)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.05, wspace=0.12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def turnon_map_panels(grid):
    """The three scalar-map specs (E rate, I rate edge-clipped, contrast) for the
    turn-on grid: (values 2-D array, title, colour vmax, cell-label fmt, mark A/B/C).
    Single source of truth so exp054 Fig 1 and the onset super-compound agree."""
    e = np.array([[c["rate_hz"] for c in row] for row in grid])
    i = np.array([[c["rate_i_hz"] for c in row] for row in grid])
    ct = np.array([[c["contrast"] for c in row] for row in grid])
    i_fin = i[np.isfinite(i)]
    i_cap = float(np.nanpercentile(i_fin, 92)) if i_fin.size else None
    e_max = float(np.nanmax(e)) if np.isfinite(e).any() else 1.0
    return [
        (e, "E firing rate (Hz)", e_max, "{:.0f}", False),
        (i, "I firing rate (Hz; edge clipped)", i_cap, "{:.0f}", False),
        (ct, "lobe–trough contrast", 1.0, "{:.2f}", True),
    ]


def draw_turnon_map(ax, vals, *, title, vmax_color, fmt, mark=False, show_y=True,
                    mark_labels=None, cell_fontsize=4.5):
    """Draw one W_EI × W_IE scalar heatmap into ax (per-cell value labels + optional
    sample-point markers). Shared by exp054 Figure 1 and the onset super-compound —
    restyle here and both figures update. mark_labels overrides the circle letters
    (the super-compound relabels them D/E/F to match its raster panel letters).
    cell_fontsize scales the per-cell value labels down for the small super-compound
    panels, where the default (tuned for Figure 1's wider maps) would overlap."""
    ax.imshow(vals, origin="lower", aspect="equal", cmap="Greys",
              vmin=0.0, vmax=vmax_color)
    xt = range(0, len(WEI_MEAN_GRID), 2)
    ax.set_xticks(list(xt))
    ax.set_xticklabels([f"{WEI_MEAN_GRID[t]:.1f}" for t in xt], fontsize=theme.SIZE_TICK - 1)
    if show_y:
        yt = range(0, len(WIE_MEAN_GRID), 2)
        ax.set_yticks(list(yt))
        ax.set_yticklabels([f"{WIE_MEAN_GRID[t]:.1f}" for t in yt], fontsize=theme.SIZE_TICK - 1)
        ax.set_ylabel("W_IE mean (μS)", fontsize=theme.SIZE_LABEL)
    else:
        ax.set_yticks([])
    ax.set_xlabel("W_EI mean (μS)", fontsize=theme.SIZE_LABEL)
    ax.set_title(title, loc="center", fontsize=theme.SIZE_LABEL - 1, fontweight="semibold")
    for iy in range(vals.shape[0]):
        for ix in range(vals.shape[1]):
            v = vals[iy, ix]
            if np.isfinite(v):
                frac = (v / vmax_color) if vmax_color else 0.0
                ax.text(ix, iy, fmt.format(v), ha="center", va="center",
                        fontsize=cell_fontsize,
                        color="white" if frac > 0.55 else theme.INK_BLACK)
    if mark:
        labels = mark_labels if mark_labels is not None else [p[0] for p in TURNON_POINTS]
        for (_, wei_i, wie_i), disp in zip(TURNON_POINTS, labels):
            ax.scatter([wei_i], [wie_i], s=90, facecolor="white",
                       edgecolor=theme.INK_BLACK, linewidths=1.2, zorder=5)
            ax.text(wei_i, wie_i, disp, ha="center", va="center", zorder=6,
                    fontsize=theme.SIZE_TICK, fontweight="bold", color=theme.INK_BLACK)


def draw_turnon_raster(ax, cell, *, label, wei_i, wie_i, show_label=True):
    """Draw one representative sample-point raster into ax (E black below, I red
    above). Shared by exp054 Figure 1 and the onset super-compound. show_label=False
    drops the leading letter from the title (the super-compound identifies each
    raster by its prominent panel letter instead)."""
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
    prefix = f"{label}   " if show_label else ""
    ax.set_title(
        f"{prefix}W_EI={WEI_MEAN_GRID[wei_i]:.1f}, W_IE={WIE_MEAN_GRID[wie_i]:.1f}"
        f"   contrast={cell['contrast']:.2f}",
        loc="left", fontsize=theme.SIZE_TICK - 1,
    )
    ax.set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    _despine(ax)


def fig_turnon_maps_compound(grid, out_path):
    """Merged turn-on figure: the three scalar maps (E rate, I rate, contrast)
    over W_EI × W_IE on top, with three representative rasters (loop-off / weak /
    strong, marked A/B/C on the contrast map) below. Rendering is shared with
    the onset super-compound via draw_turnon_map / draw_turnon_raster."""
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(12, 6.75), dpi=150)  # 16:9
    gs = GridSpec(
        2, 3, figure=fig, height_ratios=[1.55, 1.0],
        hspace=0.32, wspace=0.12, top=0.95, bottom=0.09, left=0.06, right=0.99,
    )

    # Top row — the three scalar maps; A/B/C marked on the contrast panel.
    for k, (vals, title, vmax_color, fmt, mark) in enumerate(turnon_map_panels(grid)):
        draw_turnon_map(fig.add_subplot(gs[0, k]), vals, title=title,
                        vmax_color=vmax_color, fmt=fmt, mark=mark, show_y=(k == 0))

    # Bottom row — the three representative rasters, E black below / I red above.
    for k, (label, wei_i, wie_i) in enumerate(TURNON_POINTS):
        draw_turnon_raster(fig.add_subplot(gs[1, k]), grid[wie_i][wei_i],
                           label=label, wei_i=wei_i, wie_i=wie_i)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_rate_invariance(grid, priv_null, shared_null, out_path):
    """Contrast vs E firing rate: the private-input null (flat ≈0, used) against the
    shared-input null (climbs at low rate, rejected), with the actual PING grid cells."""
    g_rate = np.array([c["rate_hz"] for row in grid for c in row])
    g_ct = np.array([c["contrast"] for row in grid for c in row])
    pr = np.array([d["rate"] for d in priv_null])
    pc = np.array([d["contrast"] for d in priv_null])
    sr = np.array([d["rate"] for d in shared_null])
    sc = np.array([d["contrast"] for d in shared_null])
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.axhline(0.0, color=theme.FAINT, lw=0.8, ls=":")
    ax.plot(sr, sc, color=theme.GREY_MID, lw=1.4, ls="--", marker="o", ms=4,
            label="null — shared input (rejected)")
    ax.plot(pr, pc, color=theme.INK_BLACK, lw=1.6, marker="o", ms=4,
            label="null — private input (used)")
    ax.scatter(g_rate, g_ct, s=42, color=theme.DEEP_RED, edgecolor="white",
               linewidths=0.5, zorder=5, label="PING grid cells (private)")
    ax.set_xlabel("E firing rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("lobe–trough contrast", fontsize=theme.SIZE_LABEL)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max(g_rate.max(), sr.max(), pr.max()) * 1.03)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_null_autocorr(shared_null, priv_null, out_path):
    """Null-network autocorrelograms at matched low firing rates: shared input (top)
    grows a spurious hat from input coincidence; private input (bottom) stays flat."""
    targets = (1.0, 2.5, 5.0)

    def nearest(recs, t):
        return recs[int(np.argmin([abs(d["rate"] - t) for d in recs]))]

    rows = [("shared input", shared_null), ("private input", priv_null)]
    fig, axes = plt.subplots(2, len(targets), figsize=(3.4 * len(targets), 3.4 * len(targets) * 9 / 16),
                             dpi=150, squeeze=False)
    for ri, (lbl, recs) in enumerate(rows):
        for ci, t in enumerate(targets):
            d = nearest(recs, t)
            ac = d["ac"]
            ax = axes[ri][ci]
            ax.axhline(1.0, color=theme.FAINT, lw=0.8, ls=":")
            ax.plot(d["ac_lags"], ac, color=theme.INK, lw=1.0)
            if d["lobe_lag"] is not None:
                ax.plot(d["lobe_lag"], ac[int(round(d["lobe_lag"] / BIN_MS))], "^",
                        color=theme.INK_BLACK, ms=6, zorder=5)
            if d["trough_lag"] is not None:
                ax.plot(d["trough_lag"], ac[int(round(d["trough_lag"] / BIN_MS))], "v",
                        color=theme.DEEP_RED, ms=6, zorder=5)
            fin = ac[1:][np.isfinite(ac[1:])]
            pmax = float(np.nanmax(fin)) if fin.size else 1.0
            ax.set_xlim(-2, 50)
            ax.set_ylim(-0.06 * pmax, pmax * 1.12)
            ax.set_title(f"E = {d['rate']:.1f} Hz   contrast = {d['contrast']:.2f}",
                         fontsize=theme.SIZE_LABEL, color=theme.INK)
            if ci == 0:
                ax.set_ylabel(f"{lbl}\nA(ℓ)", fontsize=theme.SIZE_LABEL)
            if ri == 1:
                ax.set_xlabel("lag (ms)", fontsize=theme.SIZE_LABEL)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── Onset super-compound: empirics (this notebook) + mean-field (exp033) ─────
# Merges the turn-on maps/rasters above with the exp033 4D conductance mean-field
# bifurcation (Hopf / hysteresis / gamma-vs-τ_GABA) into one figure — the anchor
# for the manuscript's gamma-onset result. Migrated here from the retired nb057.


def _load_runner(slug: str):
    """Import a sibling notebook module (exp033) by slug for cross-notebook reuse.
    Notebooks never import each other's package; this loads the module file so its
    pure-python numerics (the mean-field) can be called directly."""
    path = REPO / "experiments" / f"{slug}.py"
    spec = importlib.util.spec_from_file_location(slug, path)
    # spec_from_file_location returns ModuleSpec | None and spec.loader is
    # Loader | None; both are always set for a real file path — assert to narrow.
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[slug] = mod
    spec.loader.exec_module(mod)
    return mod


def build_super_compound(grid, results, hopf, sweep, mf, meas, out_path):
    """3×3 super-compound: the turn-on maps (row 0) + A/B/C rasters (row 1) over
    the exp033 mean-field bifurcation panels (row 2)."""
    from matplotlib.gridspec import GridSpec

    # Manuscript figure: render in paper mode (smaller print-size typography +
    # 300 dpi) so the dense titles fit the tight 3×3 grid, independent of the
    # notebook's screen-mode default used for its other figures. Restore after.
    prev_paper = theme.PAPER_MODE
    theme.set_paper_mode(True)
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"
    PTS = TURNON_POINTS

    # Tall super-compound — three panel rows; a deliberate exception to the
    # 16:9 house ratio (same reasoning as exp033's wide bifurcation strip).
    fig = plt.figure(figsize=(6.9, 6.13), dpi=150)
    gs = GridSpec(
        3, 3, figure=fig, height_ratios=[1.25, 0.92, 1.05],
        hspace=0.5, wspace=0.32, top=0.95, bottom=0.06, left=0.07, right=0.95,
    )

    # All nine panels carry a prominent A–I identifier (reading order). The three
    # raster panels are D/E/F, so the contrast-map callout circles are relabelled
    # D/E/F to point at them (and the redundant per-raster leading letter dropped).
    RASTER_LETTERS = ["D", "E", "F"]

    # ── Row 0 — turn-on maps (panels A/B/C); contrast map marks the D/E/F points ──
    for k, (vals, title, vmax_color, fmt, mark) in enumerate(turnon_map_panels(grid)):
        ax = fig.add_subplot(gs[0, k])
        draw_turnon_map(ax, vals, title=title, vmax_color=vmax_color, fmt=fmt,
                        mark=mark, show_y=(k == 0),
                        mark_labels=RASTER_LETTERS if mark else None,
                        cell_fontsize=3.0)
        _panel_letter(ax, "ABC"[k])

    # ── Row 1 — the D/E/F rasters (identified by their panel letters) ──
    for k, (label, wei_i, wie_i) in enumerate(PTS):
        ax = fig.add_subplot(gs[1, k])
        draw_turnon_raster(ax, grid[wie_i][wei_i], label=label,
                           wei_i=wei_i, wie_i=wie_i, show_label=False)
        _panel_letter(ax, RASTER_LETTERS[k])

    # ── Row 2 — the exp033 bifurcation panels (Hopf / hysteresis / frequency) ──
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
    # sc is assigned inside the eigenvalue loop (always ≥1 eigenvalue); narrow
    # the None branch. PathCollection from scatter is a valid ScalarMappable.
    assert sc is not None
    # Colour-bar as an inset INSIDE panel G. `fig.colorbar(ax=axA)` steals space
    # from axA's right edge and drops the bar (plus its "I_ext (nA)" label) into the
    # G|H gutter, where it collides with panel H's y-axis label and ticks. An inset
    # keeps it within G's own footprint, in the sparse upper-left of the eigenvalue
    # scatter, so the mean-field row stays legible.
    cax = axA.inset_axes([0.06, 0.56, 0.035, 0.38])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("$I_\\text{ext}$ (nA)", fontsize=theme.SIZE_TICK - 2, labelpad=2)
    cbar.ax.tick_params(labelsize=theme.SIZE_TICK - 2)
    cbar.ax.yaxis.set_ticks_position("right")
    cbar.ax.yaxis.set_label_position("right")
    axA.set_xlabel("Re$(\\lambda)$", fontsize=theme.SIZE_LABEL)
    axA.set_ylabel("Im$(\\lambda)$", fontsize=theme.SIZE_LABEL)
    axA.set_title(f"Hopf crossing at $I^\\star$ = {hopf['I_ext_star']:.2f} nA",
                  loc="left", fontsize=theme.SIZE_LABEL, fontweight="semibold")
    _panel_letter(axA, "G")
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
    axB.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="lower right")
    _panel_letter(axB, "H")
    _despine(axB)

    axC = fig.add_subplot(gs[2, 2])
    tg = [d["tau_gaba_ms"] for d in mf if d["f_star_Hz"] is not None]
    fs = [d["f_star_Hz"] for d in mf if d["f_star_Hz"] is not None]
    axC.plot(tg, fs, "o-", color=theme.INK_BLACK, lw=1.4, label="mean-field $f^\\star$")
    if meas:
        mt = sorted(meas)
        axC.plot(mt, [meas[t] for t in mt], "s--", color=theme.DEEP_RED, lw=1.3,
                 label="spiking $f_\\gamma$ (exp041)")
    axC.set_xlabel("$\\tau_\\text{GABA}$ (ms)", fontsize=theme.SIZE_LABEL)
    axC.set_ylabel("gamma frequency (Hz)", fontsize=theme.SIZE_LABEL)
    axC.set_title("Frequency from biophysics",
                  loc="left", fontsize=theme.SIZE_LABEL, fontweight="semibold")
    axC.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    _panel_letter(axC, "I")
    _despine(axC)

    save_figure(fig, out_path, formats=("png", "pdf"))
    plt.close(fig)
    theme.set_paper_mode(prev_paper)  # restore the notebook's screen-mode default
    theme.apply()


# ── Onset super-compound cache ──────────────────────────────────────
# The onset super-compound (Fig 2 of ar009) is otherwise redrawn only by the full
# run, whose ~6 min cost is the 121-network `run_grid`. Persisting its inputs — the
# grid plus the exp033 mean-field results — lets `--plot-only super` redraw it in
# seconds. The structures are nested dict/list/ndarray, so they go in as a single
# object-array payload, written to the gitignored scratch dir (ARTIFACTS/temp),
# not the published data dir — it is regenerable and shouldn't be committed.
def _save_super_cache(path, grid, results, hopf, criticality, mf_freq, meas_fgamma):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.empty(6, dtype=object)
    payload[:] = [grid, results, hopf, criticality, mf_freq, meas_fgamma]
    np.savez(path, payload=payload)


def _load_super_cache(path):
    payload = np.load(path, allow_pickle=True)["payload"]
    grid, results, hopf, criticality, mf_freq, meas_fgamma = payload
    return grid, results, hopf, criticality, mf_freq, meas_fgamma


def main():
    meta = parse_meta(sys.argv)
    sim_ms = SIM_MS

    if meta.plot_fig == "super":
        # Fast redraw of the onset super-compound from cache — no 121-network grid
        # re-simulation. Mirrors exp048's `--plot-only headline`; requires one prior
        # full run to seed the cache (like exp048's `--replot grid`).
        run_id = next_run_id(SLUG)
        with published_run(
            SLUG, run_id, scale=SCALE, plot_only=True,
        ) as (_artifacts, figures):
            theme.apply()
            plt.rcParams["savefig.bbox"] = "standard"
            cache = _artifacts / "super_compound_cache.npz"
            if not cache.exists():
                raise SystemExit(
                    "--plot-only super: no cache at "
                    f"{cache}; run the full notebook once first."
                )
            grid, results, hopf, criticality, mf_freq, meas_fgamma = _load_super_cache(cache)
            build_super_compound(grid, results, hopf, criticality, mf_freq, meas_fgamma,
                                 figures / "onset_super_compound")
            print("wrote onset_super_compound.{png,pdf} (replot from cache)")
        return

    t_start = time.monotonic()
    run_id = next_run_id(SLUG)
    with published_run(
        SLUG, run_id, scale=SCALE, plot_only=meta.plot_only,
    ) as (_artifacts, figures):
        theme.apply()
        plt.rcParams["savefig.bbox"] = "standard"  # keep the saved 16:9 exact

        n_nets = len(WEI_MEAN_GRID) * len(WIE_MEAN_GRID)
        print(f"exp054 | {n_nets} untrained PING networks, private "
              f"{NET_INPUT_RATE:.0f} Hz per-cell Poisson input, sim {sim_ms:.0f} ms (compiles per net)…")
        grid = run_grid(NET_INPUT_RATE, sim_ms)
        fig_turnon_maps_compound(grid, figures / "turnon_maps_compound.png")
        fig_turnon_compound(grid, figures / "turnon_compound.png")
        fig_grid_maps_compound(grid, figures / "grid_maps.png")
        fig_grid_rasters(grid, figures / "grid_rasters.png")
        fig_grid_autocorr(grid, figures / "grid_autocorr.png")
        print("wrote turnon_maps_compound + turnon_compound + grid_maps + grid_rasters + grid_autocorr")

        rates = [c["rate_hz"] for row in grid for c in row]
        contrasts = [c["contrast"] for row in grid for c in row]
        print(f"  contrast {np.nanmin(contrasts):.2f}–{np.nanmax(contrasts):.2f}; "
              f"E rate {min(rates):.1f}–{max(rates):.1f} Hz")

        print("  rate-invariance: null networks scanned over input rate (private + shared)…")
        priv_null = null_scan(sim_ms, private=True)
        shared_null = null_scan(sim_ms, private=False)
        fig_rate_invariance(grid, priv_null, shared_null, figures / "rate_invariance.png")
        fig_null_autocorr(shared_null, priv_null, figures / "null_autocorr.png")
        pmax = max(d["contrast"] for d in priv_null)
        smax = max(d["contrast"] for d in shared_null)
        print(f"wrote rate_invariance + null_autocorr  (null contrast max: private {pmax:.2f}, shared {smax:.2f})")

        # ── Onset super-compound — empirical turn-on (this notebook) over the exp033
        # 4D mean-field bifurcation. Migrated from the retired nb057; the maps/rasters
        # reuse the shared helpers above, so restyling Figure 1 propagates here. ──
        exp033 = _load_runner("exp033")
        print("exp033 numerics: 4D mean-field sweep, Hopf, hysteresis, frequency …")
        I_grid = np.linspace(0.0, 4.0, 401)
        results = exp033.sweep(I_grid)
        hopf = exp033.find_hopf(results)
        criticality = exp033.hysteresis_sweep(hopf["I_ext_star"])
        mf_freq = exp033.frequency_vs_tau_gaba([4.5, 6.0, 9.0, 12.0, 18.0, 27.0], I_grid)
        meas_fgamma = exp033.load_exp041_fgamma()
        build_super_compound(grid, results, hopf, criticality, mf_freq, meas_fgamma,
                             figures / "onset_super_compound")
        _save_super_cache(_artifacts / "super_compound_cache.npz",
                          grid, results, hopf, criticality, mf_freq, meas_fgamma)
        print("wrote onset_super_compound.{png,pdf} (+ super_compound_cache.npz in scratch)")

        duration_s = time.monotonic() - t_start
        # default=float coerces residual numpy scalars; round-trip so write_numbers
        # (which json.dumps the envelope) receives plain Python types.
        payload = json.loads(json.dumps({
            "config": {
                "source": "untrained PING networks, private per-cell Poisson input",
                "dt_ms": DT_MS,
                "sim_ms": sim_ms,
                "burn_ms": NET_BURN_MS,
                "n_e": NET_N_E,
                "input_rate_hz": NET_INPUT_RATE,
                "private_w_in": PRIVATE_W_IN,
                "max_lag_ms": MAX_LAG_MS,
                "bin_ms": BIN_MS,
            },
            "grid": {
                "wei_mean": list(WEI_MEAN_GRID),
                "wie_mean": list(WIE_MEAN_GRID),
                "contrast": [
                    [(c["contrast"] if np.isfinite(c["contrast"]) else None) for c in row]
                    for row in grid
                ],
                "rate_e_hz": [
                    [(c["rate_hz"] if np.isfinite(c["rate_hz"]) else None) for c in row]
                    for row in grid
                ],
                "rate_i_hz": [
                    [(c["rate_i_hz"] if np.isfinite(c["rate_i_hz"]) else None) for c in row]
                    for row in grid
                ],
                "rate_e_min_hz": float(np.nanmin(rates)),
                "rate_e_max_hz": float(np.nanmax(rates)),
                "contrast_min": float(np.nanmin(contrasts)),
                "contrast_max": float(np.nanmax(contrasts)),
            },
            "rate_invariance": {
                "private_null_max": float(pmax),
                "shared_null_max": float(smax),
                "private_scan": {"rate_hz": [d["rate"] for d in priv_null],
                                 "contrast": [d["contrast"] for d in priv_null]},
                "shared_scan": {"rate_hz": [d["rate"] for d in shared_null],
                                "contrast": [d["contrast"] for d in shared_null]},
            },
        }, default=float))
        write_numbers(figures, run_id=run_id, duration_s=duration_s, payload=payload)
        print(f"wrote {figures / 'numbers.json'}")
        print(f"\nTotal runtime: {int(duration_s // 60)}m {int(duration_s % 60):02d}s")


if __name__ == "__main__":
    main()
