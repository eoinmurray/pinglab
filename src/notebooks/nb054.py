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

Notebook entry: src/docs/src/pages/notebooks/nb054.mdx
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from helpers.modal import parse_modal_gpu  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_id import next_run_id, persist as persist_run_id  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402
import torch  # noqa: E402
from cli import theme  # noqa: E402
from cli import config as C  # noqa: E402
from cli.config import make_net, patch_dt, _extract_records  # noqa: E402
from cli.scan import primary_hid_key, primary_inh_key  # noqa: E402

# The networks read globals (N_IN, T_steps, dt…) from the `models` module that
# `config` imports — reach it via C.M so we mutate the same object build_net does.
M = C.M
from cli.metrics import (  # noqa: E402
    iei_histogram,
    population_event_times,
    rhythmicity_scalars,
    spike_autocorrelogram,
)

SLUG = "nb054"
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

# The size tier scales only the simulation length (more spikes ⇒ cleaner
# autocorrelogram); the default (medium = 1000 ms) is what the frozen figures use.
TIER_CONFIG = {
    "extra small": dict(sim_ms=500.0),
    "small": dict(sim_ms=750.0),
    "medium": dict(sim_ms=1000.0),
    "large": dict(sim_ms=2000.0),
    "extra large": dict(sim_ms=4000.0),
}
DEFAULT_TIER = "medium"


def poisson_input(dt, sim_ms, n_channels, rate_hz):
    """[T, n_channels] tensor: constant homogeneous Poisson input, each channel
    independent at rate_hz. T_steps is set from sim_ms via M.T_ms (what patch_dt
    reads), and M.N_IN is set so the network's W_in matches the channel count."""
    M.N_IN = n_channels
    M.T_ms = sim_ms
    C.cfg.sim_ms = sim_ms
    patch_dt(dt)
    rng = np.random.default_rng(0)
    inp = (rng.random((M.T_steps, n_channels)) < rate_hz * dt / 1000.0).astype(np.float32)
    return torch.tensor(inp, device=C.cfg.torch_device)


def ping_spikes(wei, wie, input_spikes, dt, private=True):
    """Build a PING net (W_EI mean wei, W_IE mean wie), drive it with input_spikes,
    return (E raster, I raster) past the burn-in. private=True wires each E cell to
    its own input channel (identity W_in); private=False uses the sparse shared W_in."""
    C.cfg.n_e = NET_N_E
    C.cfg.w_ei = (wei, wei * 0.1)
    C.cfg.w_ie = (wie, wie * 0.1)
    if private:
        net = make_net(C.cfg, w_in=(PRIVATE_W_IN, 0.0, "normal", 0.0))
        with torch.no_grad():  # one channel → one E cell, no sharing
            net.W_ff[0].copy_(torch.eye(NET_N_E, device=net.W_ff[0].device) * PRIVATE_W_IN)
    else:
        net = make_net(C.cfg, w_in=(SHARED_W_IN, SHARED_W_IN * 0.2, "normal", SHARED_W_IN_SP))
    net.recording = True
    with torch.no_grad():
        net.forward(input_spikes=input_spikes)
    rec = _extract_records(net)
    b = int(NET_BURN_MS / dt)
    spk = np.asarray(rec[primary_hid_key(rec)]).squeeze()[b:]
    ik = primary_inh_key(rec)
    spk_i = np.asarray(rec[ik]).squeeze()[b:] if ik else None
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


def run_grid_point(wei, wie, input_spikes, dt=DT_MS):
    """One (W_EI mean, W_IE mean) cell with private input: contrast + E/I rates,
    autocorrelogram, and a small raster for display."""
    spk, spk_i = ping_spikes(wei, wie, input_spikes, dt, private=True)
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


def run_grid(input_spikes):
    """grid[wie_idx][wei_idx] over WIE_MEAN_GRID × WEI_MEAN_GRID (wie_idx 0 = lowest)."""
    return [[run_grid_point(wei, wie, input_spikes) for wei in WEI_MEAN_GRID]
            for wie in WIE_MEAN_GRID]


def null_scan(sim_ms, private):
    """Scan input rate on a NULL network (W_EI = W_IE = 0, no rhythm at any drive),
    private or shared input. Returns per-rate records sorted by E rate."""
    inputs = PRIVATE_NULL_INPUT_HZ if private else SHARED_NULL_INPUT_HZ
    n_ch = NET_N_E if private else SHARED_N_IN
    recs = []
    for r in inputs:
        inp = poisson_input(DT_MS, sim_ms, n_ch, r)
        spk, _ = ping_spikes(0.0, 0.0, inp, DT_MS, private=private)
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
    fig.suptitle(f"Rasters over W_EI × W_IE  ({nr}×{nc} subset; E black, I red)",
                 fontsize=theme.SIZE_TITLE, color=theme.INK)
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
    fig.suptitle(f"E autocorrelogram A(ℓ) over W_EI × W_IE  ({nr}×{nc} subset; lag 0–50 ms, dotted = chance)",
                 fontsize=theme.SIZE_TITLE, color=theme.INK)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# Three representative (W_EI idx, W_IE idx) points for the turn-on callouts.
TURNON_POINTS = [
    ("A", 0, 6),    # loop off: W_EI = 0, I never recruited
    ("B", 4, 4),    # weak/intermediate coupling, emerging volleys
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
    ax_hm.set_title("Gamma turns on across W_EI × W_IE", loc="left",
                    fontsize=theme.SIZE_TITLE, fontweight="semibold")
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


def fig_turnon_maps_compound(grid, out_path):
    """Merged turn-on figure: the three scalar maps (E rate, I rate, contrast)
    over W_EI × W_IE on top, with three representative rasters (loop-off / weak /
    strong, marked A/B/C on the contrast map) below. Combines the standalone
    turn-on compound and the grid-maps compound into one panel."""
    from matplotlib.gridspec import GridSpec

    e = np.array([[c["rate_hz"] for c in row] for row in grid])
    i = np.array([[c["rate_i_hz"] for c in row] for row in grid])
    ct = np.array([[c["contrast"] for c in row] for row in grid])
    i_fin = i[np.isfinite(i)]
    i_cap = float(np.nanpercentile(i_fin, 92)) if i_fin.size else None
    e_max = float(np.nanmax(e)) if np.isfinite(e).any() else 1.0

    fig = plt.figure(figsize=(12, 6.75), dpi=150)  # 16:9
    gs = GridSpec(
        2, 3, figure=fig, height_ratios=[1.55, 1.0],
        hspace=0.32, wspace=0.12, top=0.95, bottom=0.09, left=0.06, right=0.99,
    )

    # Top row — the three scalar maps; A/B/C marked on the contrast panel.
    panels = [
        (e, "E firing rate (Hz)", e_max, "{:.0f}", False),
        (i, "I firing rate (Hz; edge clipped)", i_cap, "{:.0f}", False),
        (ct, "lobe–trough contrast", 1.0, "{:.2f}", True),
    ]
    for k, (vals, title, vmax_color, fmt, mark) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
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
        if mark:
            for label, wei_i, wie_i in TURNON_POINTS:
                ax.scatter([wei_i], [wie_i], s=90, facecolor="white",
                           edgecolor=theme.INK_BLACK, linewidths=1.2, zorder=5)
                ax.text(wei_i, wie_i, label, ha="center", va="center", zorder=6,
                        fontsize=theme.SIZE_TICK, fontweight="bold", color=theme.INK_BLACK)

    # Bottom row — the three representative rasters, E black below / I red above.
    for k, (label, wei_i, wie_i) in enumerate(TURNON_POINTS):
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
            f"{label}   W_EI={WEI_MEAN_GRID[wei_i]:.1f}, W_IE={WIE_MEAN_GRID[wie_i]:.1f}"
            f"   contrast={cell['contrast']:.2f}",
            loc="left", fontsize=theme.SIZE_TICK - 1,
        )
        ax.set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
        _despine(ax)

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
    ax.set_title("Private per-cell input makes the metric rate-invariant",
                 fontsize=theme.SIZE_LABEL, color=theme.INK)
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
    fig.suptitle("Null autocorrelograms: shared input grows a spurious hat at low firing, private stays flat",
                 fontsize=theme.SIZE_TITLE, color=theme.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    argv = sys.argv[1:]
    tier = parse_tier(argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(argv)  # accepted for contract parity; unused (local CPU)
    sim_ms = TIER_CONFIG[tier]["sim_ms"]

    if modal_gpu:
        print("note: nb054 is local CPU; --modal-gpu ignored.")

    t_start = time.monotonic()
    for d in (ARTIFACTS, FIGURES):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    notebook_run_id = next_run_id(SLUG)
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"  # keep the saved 16:9 exact

    n_nets = len(WEI_MEAN_GRID) * len(WIE_MEAN_GRID)
    print(f"nb054 | tier={tier} | {n_nets} untrained PING networks, private "
          f"{NET_INPUT_RATE:.0f} Hz per-cell Poisson input, sim {sim_ms:.0f} ms (compiles per net)…")
    priv_input = poisson_input(DT_MS, sim_ms, NET_N_E, NET_INPUT_RATE)
    grid = run_grid(priv_input)
    fig_turnon_maps_compound(grid, FIGURES / "turnon_maps_compound.png")
    fig_turnon_compound(grid, FIGURES / "turnon_compound.png")
    fig_grid_maps_compound(grid, FIGURES / "grid_maps.png")
    fig_grid_rasters(grid, FIGURES / "grid_rasters.png")
    fig_grid_autocorr(grid, FIGURES / "grid_autocorr.png")
    print("wrote turnon_maps_compound + turnon_compound + grid_maps + grid_rasters + grid_autocorr")

    rates = [c["rate_hz"] for row in grid for c in row]
    contrasts = [c["contrast"] for row in grid for c in row]
    print(f"  contrast {np.nanmin(contrasts):.2f}–{np.nanmax(contrasts):.2f}; "
          f"E rate {min(rates):.1f}–{max(rates):.1f} Hz")

    print("  rate-invariance: null networks scanned over input rate (private + shared)…")
    priv_null = null_scan(sim_ms, private=True)
    shared_null = null_scan(sim_ms, private=False)
    fig_rate_invariance(grid, priv_null, shared_null, FIGURES / "rate_invariance.png")
    fig_null_autocorr(shared_null, priv_null, FIGURES / "null_autocorr.png")
    pmax = max(d["contrast"] for d in priv_null)
    smax = max(d["contrast"] for d in shared_null)
    print(f"wrote rate_invariance + null_autocorr  (null contrast max: private {pmax:.2f}, shared {smax:.2f})")

    duration_s = time.monotonic() - t_start
    numbers = {
        "notebook_run_id": notebook_run_id,
        "duration_s": duration_s,
        "duration": f"{int(duration_s // 60)}m {int(duration_s % 60):02d}s",
        "tier": tier,
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
    }
    (FIGURES / "numbers.json").write_text(json.dumps(numbers, indent=2, default=float))
    persist_run_id(SLUG, notebook_run_id)
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"\nTotal runtime: {numbers['duration']}")


if __name__ == "__main__":
    main()
