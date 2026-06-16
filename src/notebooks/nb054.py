"""Notebook runner for entry 054 — A PING rhythmicity metric.

The lobe–trough contrast of the spike-time autocorrelation, applied to untrained
PING networks across the recurrent-weight plane. Each network is driven by a
constant homogeneous Poisson spike input through W_in (sparse weights, so the
COBA baseline fires irregularly), simulated at init (no training), and scored on
its E population spike train:

    A(ℓ) = (1/⟨r⟩²)(1/(n−ℓ)) Σ_t r(t) r(t+ℓ)          (chance = 1)
    contrast = (lobe − trough)/(lobe + trough) ∈ [0, 1)

W_EI (E recruits I) and W_IE (I inhibits E) are swept independently: the two zero
edges are the COBA control (no inhibitory loop), the interior is the PING regime.
Outputs the contrast heatmap, the E/I firing-rate maps, and the E/I rasters.

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
# `config` imports — reach it via C.M so we mutate the same object build_net does
# (a second `from cli import models` would be a *different* module under the dual
# src / src/cli path).
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

# Untrained PING networks driven by a CONSTANT homogeneous Poisson spike input
# through W_in (sparse weights keep the COBA baseline irregular ⇒ contrast ≈ 0).
NET_N_E = 256
NET_N_IN = 200           # input channels
NET_INPUT_RATE = 100.0   # Hz, constant per channel
NET_W_IN = 0.2           # input weight mean (sparse, see NET_W_IN_SP)
NET_W_IN_SP = 0.95
NET_BURN_MS = 100.0

# Independent 2-D sweep of the recurrent weight means (W_EI vs W_IE).
WEI_MEAN_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
WIE_MEAN_GRID = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
GRID_WIN_MS = 600.0      # raster display window
GRID_E_SHOW = 160        # E cells drawn per raster panel
GRID_I_SHOW = 48         # I cells drawn per raster panel

# Rate-invariance check: a non-rhythmic NULL network (no inhibitory loop, so no
# rhythm at any drive) scanned over input rate. A rate-invariant metric should
# read ≈0 at every E firing rate; any rise is a finite-sample artifact.
NULL_SCAN_INPUT_HZ = [8.0, 12.0, 16.0, 20.0, 28.0, 40.0, 60.0, 100.0]
NULL_FLOOR_HZ = 5.0      # below this E rate the null contrast exceeds ≈0.1 (unreliable)

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


def constant_poisson_input(dt, sim_ms, rate_hz=NET_INPUT_RATE):
    """[T, N_IN] tensor: constant homogeneous Poisson input, every channel firing
    at rate_hz for the whole trial (same realisation for every net). T_steps is set
    from sim_ms via M.T_ms (what patch_dt reads), so the sim length really changes."""
    M.N_IN = NET_N_IN
    M.T_ms = sim_ms
    C.cfg.sim_ms = sim_ms
    patch_dt(dt)
    rng = np.random.default_rng(0)
    inp = (rng.random((M.T_steps, NET_N_IN)) < rate_hz * dt / 1000.0).astype(np.float32)
    return torch.tensor(inp, device=C.cfg.torch_device)


def ping_spikes(input_spikes, wei, wie, dt):
    """Build a PING net (W_EI mean wei, W_IE mean wie), drive it with the constant
    Poisson input, and return (E raster, I raster) past the burn-in."""
    C.cfg.n_e = NET_N_E
    C.cfg.w_ei = (wei, wei * 0.1)
    C.cfg.w_ie = (wie, wie * 0.1)
    net = make_net(C.cfg, w_in=(NET_W_IN, NET_W_IN * 0.2, "normal", NET_W_IN_SP))
    net.recording = True
    with torch.no_grad():
        net.forward(input_spikes=input_spikes)
    rec = _extract_records(net)
    b = int(NET_BURN_MS / dt)
    spk = np.asarray(rec[primary_hid_key(rec)]).squeeze()[b:]
    ik = primary_inh_key(rec)
    spk_i = np.asarray(rec[ik]).squeeze()[b:] if ik else None
    return spk, spk_i


def run_grid_point(wei, wie, input_spikes, dt=DT_MS):
    """One (W_EI mean, W_IE mean) cell: drive with constant Poisson input, compute
    the contrast + E/I rates, and keep a small raster for display."""
    spk, spk_i = ping_spikes(input_spikes, wei, wie, dt)
    ac_lags, ac = spike_autocorrelogram(spk, dt, MAX_LAG_MS, BIN_MS)
    iei_lags, iei = iei_histogram(population_event_times(spk, dt), MAX_LAG_MS, BIN_MS)
    sc = rhythmicity_scalars(ac_lags, ac, iei_lags, iei, BIN_MS)
    ct = sc["contrast"]
    rate = float(spk.sum() / (spk.shape[1] * spk.shape[0] * dt / 1000.0))
    rate_i = (float(spk_i.sum() / (spk_i.shape[1] * spk_i.shape[0] * dt / 1000.0))
              if spk_i is not None else np.nan)
    win = int(GRID_WIN_MS / dt)
    return dict(
        contrast=ct if ct is not None else np.nan,
        rate_hz=rate,
        rate_i_hz=rate_i,
        ac_lags=ac_lags,
        ac=ac,
        lobe_lag=sc["lobe_lag"],
        trough_lag=sc["trough_lag"],
        e=spk[:win, :GRID_E_SHOW],
        i=spk_i[:win, :GRID_I_SHOW] if spk_i is not None else None,
    )


def run_grid(input_spikes):
    """grid[wie_idx][wei_idx] over WIE_MEAN_GRID × WEI_MEAN_GRID (wie_idx 0 = lowest)."""
    return [[run_grid_point(wei, wie, input_spikes) for wei in WEI_MEAN_GRID]
            for wie in WIE_MEAN_GRID]


def _contrast_heatmap(vals, title, cbar_label, out_path):
    """A contrast-valued heatmap over the W_EI × W_IE plane (magma, 0→1)."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    im = ax.imshow(vals, origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(WEI_MEAN_GRID)))
    ax.set_xticklabels([f"{v:g}" for v in WEI_MEAN_GRID])
    ax.set_yticks(range(len(WIE_MEAN_GRID)))
    ax.set_yticklabels([f"{v:g}" for v in WIE_MEAN_GRID])
    ax.set_xlabel("W_EI mean (μS)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("W_IE mean (μS)", fontsize=theme.SIZE_LABEL)
    for iy in range(vals.shape[0]):
        for ix in range(vals.shape[1]):
            v = vals[iy, ix]
            if np.isfinite(v):
                ax.text(ix, iy, f"{v:.2f}", ha="center", va="center",
                        fontsize=theme.SIZE_ANNOTATION,
                        color="white" if v < 0.5 else "black")
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
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    im = ax.imshow(vals, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(WEI_MEAN_GRID)))
    ax.set_xticklabels([f"{v:g}" for v in WEI_MEAN_GRID])
    ax.set_yticks(range(len(WIE_MEAN_GRID)))
    ax.set_yticklabels([f"{v:g}" for v in WIE_MEAN_GRID])
    ax.set_xlabel("W_EI mean (μS)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("W_IE mean (μS)", fontsize=theme.SIZE_LABEL)
    vmax = np.nanmax(vals) if np.isfinite(vals).any() else 1.0
    for iy in range(vals.shape[0]):
        for ix in range(vals.shape[1]):
            v = vals[iy, ix]
            if np.isfinite(v):
                ax.text(ix, iy, f"{v:.0f}", ha="center", va="center",
                        fontsize=theme.SIZE_ANNOTATION,
                        color="white" if v < 0.55 * vmax else "black")
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
    """E/I rasters laid out to match the heatmap (E black, I red)."""
    nr, nc = len(WIE_MEAN_GRID), len(WEI_MEAN_GRID)
    fig, axes = plt.subplots(nr, nc, figsize=(2.0 * nc, 1.7 * nr), dpi=150, squeeze=False)
    for r in range(nr):  # display top→bottom = high→low W_IE
        wie_idx = nr - 1 - r
        for c in range(nc):
            ax = axes[r][c]
            cell = grid[wie_idx][c]
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
                ax.set_ylabel(f"{WIE_MEAN_GRID[wie_idx]:g}", fontsize=theme.SIZE_TICK,
                              rotation=0, ha="right", va="center")
            if r == nr - 1:
                ax.set_xlabel(f"{WEI_MEAN_GRID[c]:g}", fontsize=theme.SIZE_TICK)
    fig.supxlabel("W_EI mean (μS)", fontsize=theme.SIZE_LABEL)
    fig.supylabel("W_IE mean (μS)", fontsize=theme.SIZE_LABEL)
    fig.suptitle("Rasters over W_EI × W_IE  (E black, I red)",
                 fontsize=theme.SIZE_TITLE, color=theme.INK)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_grid_autocorr(grid, out_path):
    """E-population autocorrelogram A(ℓ) per grid cell, laid out to match the
    heatmap — the Mexican hat the contrast is read from: flat ≈1 at the COBA
    edges, a deep lobe-and-trough through the PING interior."""
    nr, nc = len(WIE_MEAN_GRID), len(WEI_MEAN_GRID)
    fig, axes = plt.subplots(nr, nc, figsize=(2.0 * nc, 1.7 * nr), dpi=150, squeeze=False)
    # Shared y-range spanning the tallest lobe (+ head/foot room) so every lobe (▲)
    # and trough (▼) marker stays inside the panel.
    allac = np.concatenate([c["ac"][1:][np.isfinite(c["ac"][1:])] for row in grid for c in row])
    ymax = float(np.nanmax(allac)) if allac.size else 3.0
    for r in range(nr):  # display top→bottom = high→low W_IE
        wie_idx = nr - 1 - r
        for cc in range(nc):
            ax = axes[r][cc]
            cell = grid[wie_idx][cc]
            ax.axhline(1.0, color=theme.FAINT, lw=0.6, ls=":")
            ax.plot(cell["ac_lags"], cell["ac"], color=theme.INK, lw=0.9)
            ac = cell["ac"]
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
                ax.set_ylabel(f"{WIE_MEAN_GRID[wie_idx]:g}", fontsize=theme.SIZE_TICK,
                              rotation=0, ha="right", va="center")
            if r == nr - 1:
                ax.set_xlabel(f"{WEI_MEAN_GRID[cc]:g}", fontsize=theme.SIZE_TICK)
    fig.supxlabel("W_EI mean (μS)", fontsize=theme.SIZE_LABEL)
    fig.supylabel("W_IE mean (μS)", fontsize=theme.SIZE_LABEL)
    fig.suptitle("E autocorrelogram A(ℓ) over W_EI × W_IE  (lag 0–50 ms, dotted = chance)",
                 fontsize=theme.SIZE_TITLE, color=theme.INK)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def null_rate_scan(sim_ms):
    """Scan input rate on a NULL network (W_EI = W_IE = 0, no inhibitory loop, no
    rhythm at any drive). Returns (E firing rates, contrasts) — a rate-invariant
    metric should sit at ≈0 across the whole range."""
    rates, contrasts = [], []
    for r in NULL_SCAN_INPUT_HZ:
        inp = constant_poisson_input(DT_MS, sim_ms, r)
        spk, _ = ping_spikes(inp, 0.0, 0.0, DT_MS)
        ac_lags, ac = spike_autocorrelogram(spk, DT_MS, MAX_LAG_MS, BIN_MS)
        iei_lags, iei = iei_histogram(population_event_times(spk, DT_MS), MAX_LAG_MS, BIN_MS)
        ct = rhythmicity_scalars(ac_lags, ac, iei_lags, iei, BIN_MS)["contrast"]
        rates.append(float(spk.sum() / (spk.shape[1] * spk.shape[0] * DT_MS / 1000.0)))
        contrasts.append(ct if ct is not None else np.nan)
    order = np.argsort(rates)
    return np.array(rates)[order], np.array(contrasts)[order]


def corrected_grid(grid, scan_rates, scan_contrasts):
    """Rate-corrected contrast: subtract the null-network baseline B at each cell's
    own E firing rate (B interpolated from the null scan), clipped to [0, 1]. A
    non-rhythmic network → 0 at any rate; a rhythmic one keeps its excess over the
    input-coincidence floor."""
    out = np.full((len(WIE_MEAN_GRID), len(WEI_MEAN_GRID)), np.nan)
    for iy, row in enumerate(grid):
        for ix, c in enumerate(row):
            if np.isfinite(c["contrast"]):
                b = float(np.interp(c["rate_hz"], scan_rates, scan_contrasts))
                out[iy, ix] = float(np.clip(c["contrast"] - b, 0.0, 1.0))
    return out


def fig_grid_corrected(corrected, out_path):
    """Rate-corrected contrast heatmap (raw contrast minus the null baseline)."""
    _contrast_heatmap(corrected, "Rate-corrected contrast over W_EI × W_IE",
                      "contrast − null baseline", out_path)


def fig_rate_invariance(grid, scan_rates, scan_contrasts, out_path):
    """Contrast vs E firing rate: the NULL network (black, should be flat ≈0) and
    the actual PING grid cells (red). The null climbs as firing thins and blows up
    near zero — the metric is rate-invariant only above a spike-count floor."""
    g_rate = np.array([c["rate_hz"] for row in grid for c in row])
    g_ct = np.array([c["contrast"] for row in grid for c in row])
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.axvspan(0, NULL_FLOOR_HZ, color=theme.DANGER, alpha=0.10)
    ax.axhline(0.0, color=theme.FAINT, lw=0.8, ls=":")
    ax.scatter(g_rate, g_ct, s=42, color=theme.DEEP_RED, edgecolor="white",
               linewidths=0.5, zorder=5, label="PING grid cells")
    ax.plot(scan_rates, scan_contrasts, color=theme.INK_BLACK, lw=1.6, marker="o",
            label="null network (no loop)")
    ax.text(NULL_FLOOR_HZ / 2, 0.52, "low-rate\nbaseline", ha="center", va="center",
            fontsize=theme.SIZE_ANNOTATION, color=theme.DANGER)
    ax.set_xlabel("E firing rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("lobe–trough contrast", fontsize=theme.SIZE_LABEL)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max(g_rate.max(), scan_rates.max()) * 1.03)
    ax.set_title("Rate-invariance: a non-rhythmic network reads ≈0 only above a low-rate floor",
                 fontsize=theme.SIZE_LABEL, color=theme.INK)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="center right")
    fig.tight_layout()
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

    n_nets = len(WEI_MEAN_GRID) * len(WIE_MEAN_GRID)
    print(f"nb054 | tier={tier} | {n_nets} untrained PING networks, "
          f"constant {NET_INPUT_RATE:.0f} Hz Poisson input, sim {sim_ms:.0f} ms (compiles per net)…")
    net_input = constant_poisson_input(DT_MS, sim_ms)
    grid = run_grid(net_input)
    fig_grid_heatmap(grid, FIGURES / "grid_heatmap.png")
    fig_grid_rate_e(grid, FIGURES / "grid_rate_e.png")
    fig_grid_rate_i(grid, FIGURES / "grid_rate_i.png")
    fig_grid_rasters(grid, FIGURES / "grid_rasters.png")
    fig_grid_autocorr(grid, FIGURES / "grid_autocorr.png")
    print(f"wrote grid_heatmap.png + grid_rate_e.png + grid_rate_i.png + grid_rasters.png + grid_autocorr.png")

    rates = [c["rate_hz"] for row in grid for c in row]
    contrasts = [c["contrast"] for row in grid for c in row]
    print(f"  contrast {np.nanmin(contrasts):.2f}–{np.nanmax(contrasts):.2f}; "
          f"E rate {min(rates):.1f}–{max(rates):.1f} Hz")

    print("  rate-invariance: null network scanned over input rate…")
    scan_rates, scan_contrasts = null_rate_scan(sim_ms)
    fig_rate_invariance(grid, scan_rates, scan_contrasts, FIGURES / "rate_invariance.png")
    print(f"wrote rate_invariance.png  (null contrast "
          f"{np.nanmin(scan_contrasts):.2f}–{np.nanmax(scan_contrasts):.2f})")

    corrected = corrected_grid(grid, scan_rates, scan_contrasts)
    fig_grid_corrected(corrected, FIGURES / "grid_corrected.png")
    print(f"wrote grid_corrected.png  (corrected contrast "
          f"{np.nanmin(corrected):.2f}–{np.nanmax(corrected):.2f})")

    duration_s = time.monotonic() - t_start
    numbers = {
        "notebook_run_id": notebook_run_id,
        "duration_s": duration_s,
        "duration": f"{int(duration_s // 60)}m {int(duration_s % 60):02d}s",
        "tier": tier,
        "config": {
            "source": "untrained PING networks, constant Poisson input",
            "dt_ms": DT_MS,
            "sim_ms": sim_ms,
            "burn_ms": NET_BURN_MS,
            "n_e": NET_N_E,
            "n_in": NET_N_IN,
            "input_rate_hz": NET_INPUT_RATE,
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
        },
        "rate_invariance": {
            "null_floor_hz": NULL_FLOOR_HZ,
            "scan_rate_hz": list(scan_rates),
            "scan_contrast": [float(c) if np.isfinite(c) else None for c in scan_contrasts],
        },
        "corrected": {
            "contrast": [[float(v) if np.isfinite(v) else None for v in row]
                         for row in corrected],
        },
    }
    (FIGURES / "numbers.json").write_text(json.dumps(numbers, indent=2, default=float))
    persist_run_id(SLUG, notebook_run_id)
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"\nTotal runtime: {numbers['duration']}")


if __name__ == "__main__":
    main()
