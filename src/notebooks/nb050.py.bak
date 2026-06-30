"""Notebook runner for entry 050 — PING vs the Vreeswijk-Sompolinsky
asynchronous-irregular state.

Two COBANet cells that share everything — sparse fixed-K connectivity at
p = 0.2, matched E↔I weights, no W^II — and differ only in the external
input: PING gets a shared (correlated) 45 Hz Poisson layer through W_in,
the V&S cell gets per-cell independent drive. Renders two side-by-side
figures: the input each receives, and a four-panel raster comparison
(combined E+I raster, Welch PSD, ISI-CV histogram, cross-correlogram).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cli import theme  # noqa: E402
from helpers.modal import parse_modal_gpu  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402

SLUG = "nb050"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
OSCILLOSCOPE = REPO / "src" / "cli/cli.py"
SCOPE_OUT_PNG = REPO / "src" / "artifacts" / "oscilloscope" / "snapshot.png"
SCOPE_OUT_NPZ = REPO / "src" / "artifacts" / "oscilloscope" / "snapshot.npz"

# Shared knobs across both cells: longer trial (T = 1000 ms) so per-cell
# ISI distributions get enough samples to be meaningful, and so the
# Welch PSD has 1 Hz frequency resolution (sharper bins than the 5 Hz
# of nb041/nb049's 200-ms trials).
COMMON_ARGS = [
    "sim",
    "--image",
    "--model", "ping",
    "--input", "synthetic-spikes",
    "--t-ms", "1000",
]

# Both regimes share EVERYTHING except the input: sparse fixed-K connectivity
# at p = 0.2 (cortically realistic, dense enough for the loop to cohere),
# matched E↔I weights, no W^II. The single remaining control is input correlation.
SHARED_CONN = ["--ei-sparsity", "0.8", "--exact-k"]  # p = 1 - 0.8 = 0.2
SHARED_W = ["--w-ei", "0.6", "0.18", "--w-ie", "3.0", "0.9"]

CELLS: dict[str, dict] = {
    "ping": {
        "args": [
            *SHARED_CONN, *SHARED_W,
            # The ONLY difference: a shared (correlated) input layer — a
            # common 45 Hz Poisson drive read by every E cell through W_in
            # (rate matched to the AI cell's E drive; W_in scaled down so the
            # network is not overdriven).
            "--input-rate", "45", "--w-in", "0.8", "0.16",
        ],
        "title": "PING — gamma (shared, correlated input)",
    },
    "ai": {
        "args": [
            *SHARED_CONN, *SHARED_W,
            # The ONLY difference: a DIAGONAL W_in — one private 45 Hz Poisson
            # input per cell (uncorrelated). Realised as per-cell ext_g, which
            # is mathematically identical to an identity W_in × 0.38 μS; the
            # dense input layer is disabled (--w-in ≈ 0).
            "--input-rate", "1", "--w-in", "0.01", "0.001",
            "--independent-drive", "45", "0.38",
            "--independent-drive-i", "8", "0.25",
        ],
        "title": "V&S AI — asynchronous (independent input)",
    },
}

TIER_CONFIG = {
    "extra small": {},
    "small": {},
    "medium": {},
    "large": {},
    "extra large": {},
}
DEFAULT_TIER = "small"


F_GAMMA_BAND_HZ: tuple[float, float] = (5.0, 150.0)


def _population_psd(spk_2d: np.ndarray, dt_ms: float):
    """Welch periodogram on the population-mean spike trace.
    Matches the nb041 / nb049 / nb023 pipeline: one window per trial,
    density scaling, mean-subtracted. Returns (freqs, psd, f_peak_or_None).
    """
    from scipy import signal as sp_signal
    T, N = spk_2d.shape
    if T < 2 or N == 0:
        return np.array([0.0]), np.array([0.0]), None
    x = spk_2d.mean(axis=1).astype(np.float64)
    x = x - x.mean()
    fs = 1000.0 / dt_ms
    freqs, psd = sp_signal.welch(x, fs=fs, nperseg=T, scaling="density")
    band = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
    if not band.any() or psd[band].max() == 0 or not np.isfinite(psd[band]).any():
        return freqs, psd, None
    abs_idx = int(np.where(band)[0][int(np.argmax(psd[band]))])
    if 0 < abs_idx < len(psd) - 1:
        y0, y1, y2 = psd[abs_idx - 1], psd[abs_idx], psd[abs_idx + 1]
        denom = (y0 - 2 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        delta = float(max(-0.5, min(0.5, delta)))
    else:
        delta = 0.0
    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
    f_peak = float(freqs[abs_idx] + delta * df)
    return freqs, psd, f_peak


def _pair_cross_correlogram(
    spk_2d: np.ndarray, dt_ms: float, *,
    n_pairs: int = 100, max_lag_ms: float = 100.0, bin_ms: float = 1.0,
    seed: int = 0,
):
    """Sample ``n_pairs`` random distinct cell pairs and compute their
    mean cross-correlogram (Pearson correlation between binned spike
    trains) over lags ∈ [-max_lag_ms, +max_lag_ms]. Returns:

    - lags_ms: 1-D array of lag centres in ms
    - mean_corr: cross-correlation averaged across pairs
    - peak_abs: max(|mean_corr|) across all lags — the single-number
      summary. V&S AI predicts peak_abs → 0 (≈ 1/√K); PING predicts a
      strong peak at lag 0 (and ±1/f_γ harmonics).
    """
    from scipy import signal as sp_signal
    T, N = spk_2d.shape
    if N < 2 or T == 0:
        return np.array([0.0]), np.array([0.0]), 0.0

    bin_steps = max(1, int(round(bin_ms / dt_ms)))
    n_bins = T // bin_steps
    if n_bins < 2:
        return np.array([0.0]), np.array([0.0]), 0.0
    binned = (
        spk_2d[: n_bins * bin_steps]
        .reshape(n_bins, bin_steps, N)
        .sum(axis=1)
        .astype(np.float64)
    )

    active = np.where(binned.sum(axis=0) > 0)[0]
    if active.size < 2:
        return np.array([0.0]), np.array([0.0]), 0.0

    max_lag_bins = int(max_lag_ms / bin_ms)
    lags_ms = np.arange(-max_lag_bins, max_lag_bins + 1) * bin_ms
    rng = np.random.default_rng(seed)
    n_pairs = int(min(n_pairs, active.size * (active.size - 1) // 2))

    accum = np.zeros(2 * max_lag_bins + 1, dtype=np.float64)
    n_used = 0
    for _ in range(n_pairs):
        i, j = rng.choice(active, size=2, replace=False)
        si = binned[:, i] - binned[:, i].mean()
        sj = binned[:, j] - binned[:, j].mean()
        norm = np.sqrt((si * si).sum() * (sj * sj).sum())
        if norm == 0:
            continue
        full = sp_signal.correlate(si, sj, mode="full", method="fft")
        center = n_bins - 1
        accum += full[center - max_lag_bins: center + max_lag_bins + 1] / norm
        n_used += 1

    if n_used == 0:
        return lags_ms, np.zeros_like(lags_ms), 0.0
    mean_corr = accum / n_used
    peak_abs = float(np.max(np.abs(mean_corr)))
    return lags_ms, mean_corr, peak_abs


def _isi_cvs(spk_2d: np.ndarray, dt_ms: float, min_spikes: int = 3) -> np.ndarray:
    """Per-neuron coefficient of variation of inter-spike intervals.

    Returns an array of CV values, one per neuron with ≥ min_spikes spikes.
    """
    cvs = []
    T = spk_2d.shape[0]
    times = np.arange(T) * dt_ms
    for n in range(spk_2d.shape[1]):
        idx = np.where(spk_2d[:, n] > 0)[0]
        if idx.size < min_spikes:
            continue
        spike_times = times[idx]
        isis = np.diff(spike_times)
        if isis.size == 0 or isis.std() == 0:
            continue
        cvs.append(isis.std() / isis.mean())
    return np.array(cvs)


def _draw_raster_column(axes: dict, data, col_title: str, *, left: bool) -> None:
    """Draw the four raster-page panels for one regime onto provided axes.
    Row meaning is shared across columns (combined E+I raster with the I
    population stacked above E, PSD, ISI-CV, cross-correlogram); the
    descriptive row titles and y-labels are drawn only on the left column,
    while the per-regime numbers (rates, peak frequency, median CV, peak
    correlation) appear on both."""
    ax_r, ax_psd, ax_cv, ax_xcorr = (
        axes["raster"], axes["psd"], axes["cv"], axes["xcorr"])
    spk_e, spk_i = data["spk_e"], data["spk_i"]
    dt = float(data["dt"])
    T = spk_e.shape[0]
    t_ms = np.arange(T) * dt
    N_E = spk_e.shape[1]
    has_i = spk_i.size > 0 and spk_i.shape[0] == T and spk_i.any()
    N_I = spk_i.shape[1] if has_i else 0

    # Combined raster: E in the lower band (0 … N_E), I stacked above it
    # (N_E … N_E + N_I), so the two populations share one time axis and the
    # I-leads-E timing is read off directly. E black, I red.
    e_idx, e_t = np.where(spk_e.T)
    ax_r.scatter(t_ms[e_t], e_idx, s=1.0, c=theme.INK_BLACK, marker="|",
                 linewidths=0.5)
    e_rate = float(spk_e.mean() * 1000.0 / dt)
    i_rate = 0.0
    if has_i:
        i_idx, i_t = np.where(spk_i.T)
        ax_r.scatter(t_ms[i_t], i_idx + N_E, s=1.0, c=theme.DEEP_RED,
                     marker="|", linewidths=0.5)
        i_rate = float(spk_i.mean() * 1000.0 / dt)
        ax_r.axhline(N_E, color=theme.GREY_MID, lw=0.6, alpha=0.8)
    ax_r.set_ylim(0, N_E + N_I)
    ax_r.set_xlim(0, T * dt)
    ax_r.set_xlabel("time (ms)")
    if left:
        ax_r.set_ylabel("neuron  (E below · I above)")
    ax_r.set_title(f"{col_title}  ·  ⟨r_E⟩ = {e_rate:.1f}, "
                   f"⟨r_I⟩ = {i_rate:.1f} Hz", loc="left")
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)
    if has_i:
        ax_r.text(0.995, 0.985, "I", transform=ax_r.transAxes, ha="right",
                  va="top", fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED,
                  fontweight="semibold")
        ax_r.text(0.995, 0.02,
                  f"E ({N_E})", transform=ax_r.transAxes, ha="right",
                  va="bottom", fontsize=theme.SIZE_LABEL, color=theme.INK_BLACK,
                  fontweight="semibold")

    # PSD on population-mean E spike trace.
    freqs, psd, f_peak = _population_psd(spk_e, dt)
    band = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
    ax_psd.plot(freqs[band], psd[band], color=theme.INK_BLACK, lw=1.4)
    ax_psd.set_xlim(F_GAMMA_BAND_HZ)
    ax_psd.set_xlabel("Frequency (Hz)")
    if left:
        ax_psd.set_ylabel("Pop. E PSD (a.u.)")
        ax_psd.set_title("Welch PSD on population-mean E trace",
                         loc="left", fontsize=theme.SIZE_LABEL, pad=8)
    ax_psd.spines["top"].set_visible(False)
    ax_psd.spines["right"].set_visible(False)
    if f_peak is not None:
        ax_psd.axvline(f_peak, color=theme.DEEP_RED, lw=0.9, ls="--", alpha=0.8)
        ax_psd.text(f_peak, ax_psd.get_ylim()[1] * 0.95,
                    f"  $f_\\gamma$ = {f_peak:.1f} Hz", ha="left", va="top",
                    fontsize=theme.SIZE_LABEL - 1, color=theme.DEEP_RED,
                    fontweight="semibold")
    else:
        ax_psd.text(0.99, 0.95, "no clear peak", transform=ax_psd.transAxes,
                    ha="right", va="top", fontsize=theme.SIZE_LABEL - 1,
                    color=theme.GREY_MID, fontstyle="italic")

    # ISI CV histogram on E cells.
    cvs = _isi_cvs(spk_e, dt)
    if cvs.size > 0:
        ax_cv.hist(cvs, bins=np.linspace(0, 2.0, 40), color=theme.INK_BLACK,
                   alpha=0.7)
        ax_cv.axvline(1.0, color=theme.DEEP_RED, lw=0.9, ls="--", alpha=0.85)
        ax_cv.text(1.0, ax_cv.get_ylim()[1] * 0.95, "  Poisson (CV = 1)",
                   ha="left", va="top", fontsize=theme.SIZE_LABEL - 1,
                   color=theme.DEEP_RED)
        ax_cv.text(0.02, 0.95,
                   f"median CV = {np.median(cvs):.2f}  (n = {cvs.size})",
                   transform=ax_cv.transAxes, ha="left", va="top",
                   fontsize=theme.SIZE_LABEL - 1, color=theme.LABEL)
    ax_cv.set_xlim(0, 2.0)
    ax_cv.set_xlabel("ISI CV per E neuron")
    if left:
        ax_cv.set_ylabel("count")
        ax_cv.set_title("Per-neuron ISI coefficient of variation",
                        loc="left", fontsize=theme.SIZE_LABEL, pad=4)
    ax_cv.spines["top"].set_visible(False)
    ax_cv.spines["right"].set_visible(False)

    # Mean pairwise cross-correlogram on E cells.
    lags_ms, xcorr, peak_abs = _pair_cross_correlogram(spk_e, dt)
    ax_xcorr.plot(lags_ms, xcorr, color=theme.INK_BLACK, lw=1.2)
    ax_xcorr.axhline(0.0, color=theme.GREY_MID, lw=0.6, alpha=0.7)
    ax_xcorr.axvline(0.0, color=theme.GREY_MID, lw=0.6, alpha=0.7)
    ax_xcorr.set_xlim(lags_ms[0], lags_ms[-1])
    ax_xcorr.set_xlabel("lag (ms)")
    if left:
        ax_xcorr.set_ylabel("mean pairwise C(τ)")
        ax_xcorr.set_title(
            "Pairwise cross-correlation (100 random E pairs)",
            loc="left", fontsize=theme.SIZE_LABEL, pad=4)
    ax_xcorr.spines["top"].set_visible(False)
    ax_xcorr.spines["right"].set_visible(False)
    ax_xcorr.text(0.99, 0.95, f"peak |C| = {peak_abs:.3f}",
                  transform=ax_xcorr.transAxes, ha="right", va="top",
                  fontsize=theme.SIZE_LABEL - 1, color=theme.LABEL,
                  fontweight="semibold")


def plot_raster_compare(snaps: dict, out_path: Path) -> None:
    """PING and V&S AI raster pages side by side: four shared rows (combined
    E+I raster with I stacked above E, PSD, ISI-CV, cross-correlogram) across
    two columns, so each diagnostic is read off the two regimes at once."""
    theme.apply()
    from matplotlib.gridspec import GridSpec
    order = [c for c in ("ping", "ai") if c in snaps]
    col_titles = {"ping": "PING", "ai": "V&S AI"}
    fig = plt.figure(figsize=(13.0, 9.2), dpi=150)
    gs = GridSpec(4, len(order), figure=fig,
                  height_ratios=[5.2, 2.4, 1.6, 1.8],
                  hspace=0.55, wspace=0.20,
                  top=0.94, bottom=0.05, left=0.08, right=0.98)
    for col, cell in enumerate(order):
        data = np.load(snaps[cell])
        axes = {
            "raster": fig.add_subplot(gs[0, col]),
            "psd": fig.add_subplot(gs[1, col]),
            "cv": fig.add_subplot(gs[2, col]),
            "xcorr": fig.add_subplot(gs[3, col]),
        }
        _draw_raster_column(axes, data, col_titles.get(cell, cell),
                            left=(col == 0))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_input_compare(snaps: dict, out_path: Path) -> None:
    """The external drive each regime receives, side by side. PING reads a
    1024-neuron input layer at 45 Hz through a dense W_in (a common drive to
    all E cells); the V&S AI cell gets one private 45 Hz Poisson stream per E
    cell. Top row: a raster of the drive over a 300 ms window. Bottom row:
    its population rate over time — featureless, steady Poisson in both, so
    the rhythm in PING is generated by the network, not inherited."""
    theme.apply()
    from matplotlib.gridspec import GridSpec
    INPUT_CLR = "#2c7fb8"
    order = [c for c in ("ping", "ai") if c in snaps]
    titles = {"ping": "PING — input layer, 45 Hz Poisson (dense $W_{in}$ to all E)",
              "ai": "V&S AI — per-cell streams, 45 Hz Poisson (one per E)"}
    fig = plt.figure(figsize=(13.0, 6.2), dpi=150)
    gs = GridSpec(2, len(order), figure=fig, height_ratios=[3.0, 1.3],
                  hspace=0.42, wspace=0.20, top=0.90, bottom=0.10,
                  left=0.08, right=0.98)
    for col, cell in enumerate(order):
        data = np.load(snaps[cell])
        dt = float(data["dt"])
        # Prefer the per-cell independent drive (AI); fall back to the shared
        # input-layer raster (PING).
        if "ind_spikes" in data.files and data["ind_spikes"].size and \
                data["ind_spikes"].any():
            inp = np.asarray(data["ind_spikes"])
            unit = "private stream (per E cell)"
        else:
            inp = np.asarray(data["input_spikes"])
            unit = "shared input neuron"
        T, n_in = inp.shape
        t_ms = np.arange(T) * dt
        win = int(min(300.0 / dt, T))
        sl = slice(0, win)

        ax_r = fig.add_subplot(gs[0, col])
        # Show every input stream (all n_in), vectorised.
        ev_cell, ev_t = np.where(inp[sl].T > 0)
        ax_r.scatter(t_ms[ev_t], ev_cell, s=0.8, c=INPUT_CLR, marker="|",
                     linewidths=0.4)
        ax_r.set_ylim(0, n_in)
        ax_r.set_xlim(0, win * dt)
        ax_r.set_title(titles.get(cell, cell), loc="left",
                       fontsize=theme.SIZE_LABEL)
        if col == 0:
            ax_r.set_ylabel(f"input ({unit})")
        ax_r.spines["top"].set_visible(False)
        ax_r.spines["right"].set_visible(False)
        ax_r.text(0.99, 0.97, f"{n_in} inputs", transform=ax_r.transAxes,
                  ha="right", va="top", fontsize=theme.SIZE_LABEL - 1,
                  color=theme.LABEL)

        # Population input rate over time (5 ms bins), Hz per input.
        ax_p = fig.add_subplot(gs[1, col])
        bin_steps = max(1, int(round(5.0 / dt)))
        nb = T // bin_steps
        binned = inp[: nb * bin_steps].reshape(nb, bin_steps, n_in)
        pop_rate = binned.mean(axis=(1, 2)) * 1000.0 / dt
        tb = (np.arange(nb) + 0.5) * bin_steps * dt
        ax_p.plot(tb, pop_rate, color=INPUT_CLR, lw=1.0)
        ax_p.set_xlim(0, win * dt)
        ax_p.set_ylim(0, max(pop_rate.max() * 1.2, 1.0))
        ax_p.set_xlabel("time (ms)")
        if col == 0:
            ax_p.set_ylabel("pop. input rate (Hz)")
        ax_p.spines["top"].set_visible(False)
        ax_p.spines["right"].set_visible(False)
        ax_p.text(0.99, 0.95,
                  f"⟨rate⟩ = {inp.mean() * 1000.0 / dt:.0f} Hz/input",
                  transform=ax_p.transAxes, ha="right", va="top",
                  fontsize=theme.SIZE_LABEL - 1, color=theme.LABEL)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    parse_modal_gpu(sys.argv)
    wipe_dir = "--no-wipe-dir" not in sys.argv

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id} tier={tier}")

    prepare_run_dirs(SLUG, notebook_run_id, wipe=wipe_dir, make_artifacts=False)

    figures: dict[str, Path] = {}
    summary_rows: list[dict] = []
    snaps: dict[str, Path] = {}  # per-cell snapshot copies for the raster figure
    for cell, spec in CELLS.items():
        for p in (SCOPE_OUT_PNG, SCOPE_OUT_NPZ):
            if p.exists():
                p.unlink()
        scope_argv = [*COMMON_ARGS, *spec["args"]]
        cmd = ["uv", "run", "python", str(OSCILLOSCOPE), *scope_argv]
        print(f"[scope] {cell}: {' '.join(scope_argv)}")
        subprocess.run(cmd, cwd=REPO, check=True)
        if not SCOPE_OUT_NPZ.exists():
            raise SystemExit(f"oscilloscope did not produce {SCOPE_OUT_NPZ}")

        # Stash a copy so the side-by-side raster figure can reload both
        # regimes after the loop (SCOPE_OUT_NPZ is overwritten each cell).
        snap_dst = SCOPE_OUT_NPZ.parent / f"snap_{cell}.npz"
        snap_dst.write_bytes(SCOPE_OUT_NPZ.read_bytes())
        snaps[cell] = snap_dst

        # Summary statistics for the row table — exactly the quantities the
        # raster comparison (Figure 1) reads off: rates, ISI CV, PSD peak,
        # pairwise correlation.
        data = np.load(SCOPE_OUT_NPZ)
        spk_e = data["spk_e"]
        spk_i = data["spk_i"]
        dt = float(data["dt"])
        e_rate = float(spk_e.mean() * 1000.0 / dt)
        i_rate = float(spk_i.mean() * 1000.0 / dt) if spk_i.size > 0 else 0.0
        cvs_e = _isi_cvs(spk_e, dt)
        cvs_i = _isi_cvs(spk_i, dt) if spk_i.size > 0 else np.array([])
        med_cv_e = float(np.median(cvs_e)) if cvs_e.size > 0 else float("nan")
        med_cv_i = float(np.median(cvs_i)) if cvs_i.size > 0 else float("nan")
        _, _, f_peak = _population_psd(spk_e, dt)
        _, _, peak_abs_xcorr = _pair_cross_correlogram(spk_e, dt)

        summary_rows.append({
            "cell": cell,
            "e_rate_hz": e_rate,
            "i_rate_hz": i_rate,
            "median_isi_cv_e": med_cv_e,
            "median_isi_cv_i": med_cv_i,
            "f_psd_peak_hz": f_peak,
            "peak_abs_xcorr_e": peak_abs_xcorr,
        })

    # The figures this entry shows, both side-by-side PING vs V&S AI:
    # the external drive each receives, and the raster page (combined E+I
    # raster, PSD, ISI-CV, cross-correlogram).
    if snaps:
        input_dst = FIGURES / "input_compare.png"
        plot_input_compare(snaps, input_dst)
        figures["input_compare"] = input_dst

        raster_dst = FIGURES / "raster_compare.png"
        plot_raster_compare(snaps, raster_dst)
        figures["raster_compare"] = raster_dst

    duration_s = time.monotonic() - t_start
    summary = {
        "notebook_run_id": notebook_run_id,
        "duration_s": round(duration_s, 1),
        "tier": tier,
        "common_args": COMMON_ARGS,
        "cells": {cell: spec["args"] for cell, spec in CELLS.items()},
        "summary": summary_rows,
    }
    def _clean(o):
        import math
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return None
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        return o
    (FIGURES / "numbers.json").write_text(
        json.dumps(_clean(summary), indent=2, allow_nan=False) + "\n"
    )
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  duration: {duration_s:.1f}s")


if __name__ == "__main__":
    main()
