"""Notebook runner for entry 050 — asynchronous irregular (AI) state.

Reproduces the Vreeswijk-Sompolinsky / Brunel-style balanced-network
regime on top of the existing COBANet by enabling the new W^II matrix
and tuning E↔I weights to a balanced operating point. Free-running with
uniform Poisson input on all channels (no MNIST), single trial, no
training. Per cell we produce raster + Welch PSD (same pipeline as
nb023 / nb041 / nb049) plus an ISI-CV summary.

Two conditions, plotted side-by-side as the "PING vs AI" contrast:
    - ping: canonical PING ([nb023](.)'s `ping` cell, gamma-locked).
    - ai:   balanced E/I with W^II active, sparse connectivity,
            no recurrent gamma cycle.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import theme  # noqa: E402
from _modal import parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402

SLUG = "nb050"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "cli/__main__.py"
SCOPE_OUT_PNG = REPO / "src" / "artifacts" / "oscilloscope" / "snapshot.png"
SCOPE_OUT_NPZ = REPO / "src" / "artifacts" / "oscilloscope" / "snapshot.npz"

# Shared knobs across both cells: longer trial (T = 1000 ms) so per-cell
# ISI distributions get enough samples to be meaningful, and so the
# Welch PSD has 1 Hz frequency resolution (sharper bins than the 5 Hz
# of nb041/nb049's 200-ms trials).
COMMON_ARGS = [
    "image",
    "--model", "ping",
    "--input", "synthetic-spikes",
    "--t-ms", "1000",
]
CELLS: dict[str, dict] = {
    "ping": {
        "args": [
            "--input-rate", "20",
            "--w-in", "1.5", "0.3",
            "--ei-strength", "1.5",
            # No --w-ii ⇒ canonical PING (no I→I).
        ],
        "title": "PING — canonical recurrent loop (no W^II)",
    },
    "ai": {
        "args": [
            # Brunel/Vreeswijk asynchronous-irregular state, full version.
            # Five knobs land it on textbook CV ≈ 1 for both populations:
            #   - --ei-sparsity 0.99 (K ≈ 10) breaks loop synchronisation
            #   - --independent-drive 45 0.38 — large per-spike kicks at low
            #     rate on E (input fluctuations dominate drift)
            #   - --independent-drive-i 8 0.25 — same for I (without this
            #     I-cell CV stays correlated with E via W^EI; this entry
            #     adds the flag)
            #   - --w-ie 3.0 strong I→E shunt so Poisson I activity
            #     propagates noise into E
            #   - --w-ii 0.4 modest I→I self-inhibition
            "--input-rate", "1",
            "--w-in", "0.01", "0.001",
            "--w-ei", "0.6", "0.18",
            "--w-ie", "3.0", "0.9",
            "--w-ii", "0.4", "0.12",
            "--ei-sparsity", "0.99",
            "--independent-drive", "45", "0.38",
            "--independent-drive-i", "8", "0.25",
        ],
        "title": "Balanced E/I (V&S) — per-E + per-I drive, CV ≈ 1",
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


def plot_raster(npz_path: Path, out_path: Path, title: str) -> None:
    theme.apply()
    from matplotlib.gridspec import GridSpec
    data = np.load(npz_path)
    spk_e = data["spk_e"]
    spk_i = data["spk_i"]
    dt = float(data["dt"])
    T = spk_e.shape[0]
    t_ms = np.arange(T) * dt

    has_i = spk_i.size > 0 and spk_i.shape[0] == T and spk_i.any()

    if has_i:
        fig = plt.figure(figsize=(9.0, 8.0), dpi=150)
        gs = GridSpec(
            4, 1, figure=fig,
            height_ratios=[4.0, 1.2, 2.6, 1.6],
            hspace=0.85, top=0.94, bottom=0.06, left=0.12, right=0.97,
        )
        ax_e = fig.add_subplot(gs[0])
        ax_i = fig.add_subplot(gs[1], sharex=ax_e)
        ax_psd = fig.add_subplot(gs[2])
        ax_cv = fig.add_subplot(gs[3])
    else:
        fig = plt.figure(figsize=(9.0, 6.5), dpi=150)
        gs = GridSpec(
            3, 1, figure=fig,
            height_ratios=[4.0, 2.6, 1.6],
            hspace=0.55, top=0.94, bottom=0.08, left=0.12, right=0.97,
        )
        ax_e = fig.add_subplot(gs[0])
        ax_i = None
        ax_psd = fig.add_subplot(gs[1])
        ax_cv = fig.add_subplot(gs[2])

    # E raster
    e_idx, e_t = np.where(spk_e.T)
    ax_e.scatter(
        t_ms[e_t], e_idx, s=1.0, c=theme.INK_BLACK, marker="|", linewidths=0.5,
    )
    e_rate = float(spk_e.mean() * 1000.0 / dt)
    ax_e.set_ylabel("E neuron")
    ax_e.set_ylim(0, spk_e.shape[1])
    ax_e.set_xlim(0, T * dt)
    ax_e.set_title(f"{title}  ·  ⟨r_E⟩ = {e_rate:.1f} Hz", loc="left")
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)

    # I raster
    if has_i:
        i_idx, i_t = np.where(spk_i.T)
        ax_i.scatter(
            t_ms[i_t], i_idx, s=1.0, c=theme.DEEP_RED, marker="|", linewidths=0.5,
        )
        i_rate = float(spk_i.mean() * 1000.0 / dt)
        ax_i.set_ylabel(f"I  ⟨r⟩ = {i_rate:.1f} Hz")
        ax_i.set_ylim(0, spk_i.shape[1])
        ax_i.set_xlim(0, T * dt)
        ax_i.set_xlabel("time (ms)")
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)
        ax_e.tick_params(labelbottom=False)
    else:
        ax_e.set_xlabel("time (ms)")

    # PSD on population-mean E spike trace.
    freqs, psd, f_peak = _population_psd(spk_e, dt)
    band = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
    ax_psd.plot(freqs[band], psd[band], color=theme.INK_BLACK, lw=1.4)
    ax_psd.set_xlim(F_GAMMA_BAND_HZ)
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Pop. E PSD (a.u.)")
    ax_psd.spines["top"].set_visible(False)
    ax_psd.spines["right"].set_visible(False)
    ax_psd.set_title("Welch PSD on population-mean E trace",
                     loc="left", fontsize=theme.SIZE_LABEL, pad=8)
    if f_peak is not None:
        ax_psd.axvline(f_peak, color=theme.DEEP_RED, lw=0.9, ls="--", alpha=0.8)
        ax_psd.text(
            f_peak, ax_psd.get_ylim()[1] * 0.95,
            f"  $f_\\gamma$ = {f_peak:.1f} Hz",
            ha="left", va="top",
            fontsize=theme.SIZE_LABEL - 1, color=theme.DEEP_RED,
            fontweight="semibold",
        )
    else:
        ax_psd.text(
            0.99, 0.95, "no clear peak",
            transform=ax_psd.transAxes, ha="right", va="top",
            fontsize=theme.SIZE_LABEL - 1, color=theme.GREY_MID,
            fontstyle="italic",
        )

    # ISI CV histogram on E cells.
    cvs = _isi_cvs(spk_e, dt)
    if cvs.size > 0:
        ax_cv.hist(
            cvs, bins=np.linspace(0, 2.0, 40),
            color=theme.INK_BLACK, alpha=0.7,
        )
        ax_cv.axvline(1.0, color=theme.DEEP_RED, lw=0.9, ls="--", alpha=0.85)
        ax_cv.text(
            1.0, ax_cv.get_ylim()[1] * 0.95,
            "  Poisson (CV = 1)",
            ha="left", va="top",
            fontsize=theme.SIZE_LABEL - 1, color=theme.DEEP_RED,
        )
        ax_cv.text(
            0.99, 0.95,
            f"median CV = {np.median(cvs):.2f}  (n = {cvs.size} cells)",
            transform=ax_cv.transAxes, ha="right", va="top",
            fontsize=theme.SIZE_LABEL - 1, color=theme.LABEL,
        )
    ax_cv.set_xlim(0, 2.0)
    ax_cv.set_xlabel("ISI CV per E neuron")
    ax_cv.set_ylabel("count")
    ax_cv.spines["top"].set_visible(False)
    ax_cv.spines["right"].set_visible(False)
    ax_cv.set_title("Per-neuron ISI coefficient of variation",
                    loc="left", fontsize=theme.SIZE_LABEL, pad=4)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    parse_modal_gpu(sys.argv)
    wipe_dir = "--no-wipe-dir" not in sys.argv

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id} tier={tier}")

    if wipe_dir:
        for d in (ARTIFACTS, FIGURES):
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)

    figures: dict[str, Path] = {}
    summary_rows: list[dict] = []
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

        raster_dst = FIGURES / f"raster__{cell}.png"
        plot_raster(SCOPE_OUT_NPZ, raster_dst, spec["title"])
        figures[f"raster__{cell}"] = raster_dst
        print(f"wrote {raster_dst}")

        # Extract summary statistics for the row table.
        data = np.load(SCOPE_OUT_NPZ)
        spk_e = data["spk_e"]
        spk_i = data["spk_i"]
        dt = float(data["dt"])
        T = spk_e.shape[0]
        e_rate = float(spk_e.mean() * 1000.0 / dt)
        i_rate = float(spk_i.mean() * 1000.0 / dt) if spk_i.size > 0 else 0.0
        cvs_e = _isi_cvs(spk_e, dt)
        cvs_i = _isi_cvs(spk_i, dt) if spk_i.size > 0 else np.array([])
        med_cv_e = float(np.median(cvs_e)) if cvs_e.size > 0 else float("nan")
        med_cv_i = float(np.median(cvs_i)) if cvs_i.size > 0 else float("nan")
        _, _, f_peak = _population_psd(spk_e, dt)
        summary_rows.append({
            "cell": cell,
            "e_rate_hz": e_rate,
            "i_rate_hz": i_rate,
            "median_isi_cv_e": med_cv_e,
            "median_isi_cv_i": med_cv_i,
            "f_psd_peak_hz": f_peak,
        })

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
