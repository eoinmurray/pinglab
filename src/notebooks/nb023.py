"""Notebook runner for entry 023 — PING fundamentals.

Runs the oscilloscope in *image* mode twice — once with the recurrent
loop disabled (*--ei-strength 0*, "coba") and once with it active
(*--ei-strength 1.5*, "ping") — and renders dedicated per-cell raster
plots from the saved spike arrays. Everything else is held fixed
(MNIST digit 0 sample 0, --input-rate 50, --t-ms 400, --w-in 1.5 0.3).

Per cell, two PNGs are written: raster__<cell>.png (population spike
raster) and traces__<cell>.png (membrane voltage + conductances for a
single representative neuron from each population).

Notebook entry: src/docs/src/pages/notebooks/nb023.mdx
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

from cli import theme  # noqa: E402
from _modal import parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402

SLUG = "nb023"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "cli/cli.py"
SCOPE_OUT_PNG = REPO / "src" / "artifacts" / "oscilloscope" / "snapshot.png"
SCOPE_OUT_NPZ = REPO / "src" / "artifacts" / "oscilloscope" / "snapshot.npz"

COMMON_ARGS = [
    "image",
    "--model", "ping",
    "--input", "dataset",
    "--dataset", "mnist",
    "--digit", "0",
    "--sample", "0",
    "--w-in", "1.5", "0.3",
    "--input-rate", "50",
    "--t-ms", "400",
]
CELLS: dict[str, dict] = {
    "coba": {
        "args": ["--ei-strength", "0"],
        "title": "COBA — no recurrent loop (ei-strength = 0)",
    },
    "ping": {
        "args": ["--ei-strength", "1.5"],
        "title": "PING — recurrent loop active (ei-strength = 1.5)",
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
    """Welch periodogram on the population-mean spike trace, matching
    nb041 / nb049: one window per trial (nperseg = T), density scaling,
    mean-subtracted input. Returns (freqs_hz, psd, f_peak_hz_or_None)
    with parabolic-interpolated peak frequency inside the gamma band."""
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
        fig = plt.figure(figsize=(9.0, 7.5), dpi=150)
        gs = GridSpec(
            3, 1, figure=fig,
            height_ratios=[4.0, 1.2, 2.6],
            hspace=0.85, top=0.94, bottom=0.08, left=0.12, right=0.97,
        )
        ax_e = fig.add_subplot(gs[0])
        ax_i = fig.add_subplot(gs[1], sharex=ax_e)
        ax_psd = fig.add_subplot(gs[2])
    else:
        fig = plt.figure(figsize=(9.0, 6.0), dpi=150)
        gs = GridSpec(
            2, 1, figure=fig,
            height_ratios=[4.0, 2.6],
            hspace=0.55, top=0.94, bottom=0.10, left=0.12, right=0.97,
        )
        ax_e = fig.add_subplot(gs[0])
        ax_i = None
        ax_psd = fig.add_subplot(gs[1])

    # E raster
    e_idx, e_t = np.where(spk_e.T)
    ax_e.scatter(
        t_ms[e_t], e_idx, s=1.0, c=theme.INK_BLACK, marker="|", linewidths=0.5,
    )
    ax_e.set_ylabel("E neuron")
    ax_e.set_ylim(0, spk_e.shape[1])
    ax_e.set_xlim(0, T * dt)
    ax_e.set_title(title, loc="left")
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)

    # I raster
    if has_i:
        i_idx, i_t = np.where(spk_i.T)
        ax_i.scatter(
            t_ms[i_t], i_idx, s=1.0, c=theme.DEEP_RED, marker="|", linewidths=0.5,
        )
        ax_i.set_ylabel("I neuron")
        ax_i.set_ylim(0, spk_i.shape[1])
        ax_i.set_xlim(0, T * dt)
        ax_i.set_xlabel("time (ms)")
        ax_i.spines["top"].set_visible(False)
        ax_i.spines["right"].set_visible(False)
        ax_e.tick_params(labelbottom=False)
    else:
        ax_e.set_xlabel("time (ms)")

    # PSD on the population-mean E spike trace (Welch, nperseg = T)
    freqs, psd, f_peak = _population_psd(spk_e, dt)
    band = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
    ax_psd.plot(freqs[band], psd[band], color=theme.INK_BLACK, lw=1.4)
    ax_psd.set_xlim(F_GAMMA_BAND_HZ)
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Population E PSD (a.u.)")
    ax_psd.spines["top"].set_visible(False)
    ax_psd.spines["right"].set_visible(False)
    ax_psd.set_title("Welch PSD on population-mean E trace", loc="left",
                     fontsize=theme.SIZE_LABEL, pad=8)
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

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _pick_active(spk: np.ndarray) -> int | None:
    if spk.size == 0 or spk.shape[0] == 0:
        return None
    counts = spk.sum(axis=0)
    if not np.any(counts > 0):
        return None
    return int(np.argmax(counts))


def plot_traces(npz_path: Path, out_path: Path, title: str) -> None:
    theme.apply()
    data = np.load(npz_path)
    spk_e = data["spk_e"]
    spk_i = data["spk_i"]
    v_e = data["v_e_1"]
    ge_e = data["ge_e_1"]
    dt = float(data["dt"])
    T = spk_e.shape[0]
    t_ms = np.arange(T) * dt

    has_i_state = (
        "v_i_1" in data.files
        and data["v_i_1"].shape[0] == T
        and spk_i.size > 0
        and spk_i.any()
    )
    has_gi_e = "gi_e_1" in data.files and data["gi_e_1"].any()

    e_idx = _pick_active(spk_e)
    if e_idx is None:
        e_idx = 0
    i_idx = _pick_active(spk_i) if has_i_state else None

    # Reversal potentials (mV) and leak conductances (µS) from models.py.
    E_L = -65.0
    E_E = 0.0
    E_I = -80.0
    G_L_E = 0.05
    G_L_I = 0.10
    PANEL_SIZE = (8.0, 4.5)

    def _save_panel(plot_fn, suffix: str, ylabel: str, panel_title: str):
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        plot_fn(ax)
        ax.set_xlim(0, T * dt)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — {panel_title}")
        ax.legend(loc="upper right", fontsize=theme.SIZE_LEGEND)
        fig.tight_layout()
        panel_path = out_path.with_name(
            f"{out_path.stem}__{suffix}{out_path.suffix}"
        )
        fig.savefig(panel_path, dpi=120)
        plt.close(fig)
        return panel_path

    # ── E neuron ─────────────────────────────────────────────────────
    ge_trace_e = ge_e[:, e_idx]
    gi_trace = data["gi_e_1"][:, e_idx] if has_gi_e else np.zeros_like(ge_trace_e)

    def _draw_v_e(ax):
        ax.plot(t_ms, v_e[:, e_idx], color=theme.INK_BLACK, lw=0.8, label="V_E")
        ax.axhline(-50.0, color=theme.FAINT, lw=0.5, ls="--", label="V_th")

    def _draw_g_e(ax):
        ax.plot(t_ms, ge_trace_e, color=theme.INK_BLACK, lw=0.9, label="g_E (exc)")
        if has_gi_e:
            ax.plot(t_ms, gi_trace, color=theme.DEEP_RED, lw=0.9, label="g_I (inh)")
        ax.axhline(G_L_E, color=theme.FAINT, lw=0.7, ls=":", label="g_L (leak)")

    def _draw_i_e(ax):
        i_e_in = -ge_trace_e * (v_e[:, e_idx] - E_E)
        i_i_in = -gi_trace * (v_e[:, e_idx] - E_I)
        i_l_in = -G_L_E * (v_e[:, e_idx] - E_L)
        ax.axhline(0.0, color=theme.FAINT, lw=0.5)
        ax.plot(t_ms, i_e_in, color=theme.INK_BLACK, lw=0.9,
                label="I_E in (depol.)")
        if has_gi_e:
            ax.plot(t_ms, i_i_in, color=theme.DEEP_RED, lw=0.9,
                    label="I_I in (hyperpol.)")
        ax.plot(t_ms, i_l_in, color=theme.FAINT, lw=0.7, ls=":",
                label="I_L in (leak)")

    written = [
        _save_panel(_draw_v_e, "v_e",
                    f"V_E (neuron {e_idx})  [mV]",
                    "E-neuron membrane voltage"),
        _save_panel(_draw_g_e, "g_e",
                    "g  [µS]",
                    f"E-neuron conductances (cell {e_idx})"),
        _save_panel(_draw_i_e, "i_e",
                    "I in  [+ depol. / − hyperpol.]",
                    f"E-neuron signed currents (cell {e_idx})"),
    ]

    # ── I neuron ─────────────────────────────────────────────────────
    if has_i_state and i_idx is not None:
        v_i = data["v_i_1"]
        ge_i = data["ge_i_1"]
        ge_trace_i = ge_i[:, i_idx]

        def _draw_v_i(ax):
            ax.plot(t_ms, v_i[:, i_idx], color=theme.DEEP_RED, lw=0.8, label="V_I")
            ax.axhline(-50.0, color=theme.FAINT, lw=0.5, ls="--", label="V_th")

        def _draw_g_i(ax):
            ax.plot(t_ms, ge_trace_i, color=theme.INK_BLACK, lw=0.9,
                    label="g_E (exc)")
            ax.axhline(G_L_I, color=theme.FAINT, lw=0.7, ls=":", label="g_L (leak)")

        def _draw_i_i(ax):
            i_e_in_i = -ge_trace_i * (v_i[:, i_idx] - E_E)
            i_l_in_i = -G_L_I * (v_i[:, i_idx] - E_L)
            ax.axhline(0.0, color=theme.FAINT, lw=0.5)
            ax.plot(t_ms, i_e_in_i, color=theme.INK_BLACK, lw=0.9,
                    label="I_E in (depol.)")
            ax.plot(t_ms, i_l_in_i, color=theme.FAINT, lw=0.7, ls=":",
                    label="I_L in (leak)")

        written += [
            _save_panel(_draw_v_i, "v_i",
                        f"V_I (neuron {i_idx})  [mV]",
                        "I-neuron membrane voltage"),
            _save_panel(_draw_g_i, "g_i",
                        "g  [µS]",
                        f"I-neuron conductances (cell {i_idx})"),
            _save_panel(_draw_i_i, "i_i",
                        "I in  [+ depol.]",
                        f"I-neuron signed currents (cell {i_idx})"),
        ]
    return written


def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
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

        traces_dst = FIGURES / f"traces__{cell}.png"
        panel_paths = plot_traces(SCOPE_OUT_NPZ, traces_dst, spec["title"])
        for p in panel_paths:
            figures[p.stem] = p
            print(f"wrote {p}")

    duration_s = time.monotonic() - t_start
    figs_root = FIGURES.parents[2]
    summary = {
        "notebook_run_id": notebook_run_id,
        "duration_s": round(duration_s, 1),
        "tier": tier,
        "config": {
            "tier": tier,
            "model": "ping",
            "input": "mnist d0 s0",
            "cells": list(CELLS),
            "t_ms": 400,
            "input_rate_hz": 50,
            "modal_gpu": modal_gpu,
        },
        "results": [],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {duration_s:.1f}s")


if __name__ == "__main__":
    main()
