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
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cli import theme  # noqa: E402
from helpers.figsave import save_figure  # noqa: E402
from helpers.modal import parse_modal_gpu  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402

SLUG = "nb023"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
OSCILLOSCOPE = REPO / "src" / "cli/cli.py"
SCOPE_OUT_PNG = REPO / "src" / "artifacts" / "oscilloscope" / "snapshot.png"
SCOPE_OUT_NPZ = REPO / "src" / "artifacts" / "oscilloscope" / "snapshot.npz"

COMMON_ARGS = [
    "sim",
    "--image",
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
        fig = plt.figure(figsize=(6.9, 5.75))
        gs = GridSpec(
            3, 1, figure=fig,
            height_ratios=[4.0, 1.2, 2.6],
            hspace=0.85, top=0.94, bottom=0.08, left=0.12, right=0.97,
        )
        ax_e = fig.add_subplot(gs[0])
        ax_i = fig.add_subplot(gs[1], sharex=ax_e)
        ax_psd = fig.add_subplot(gs[2])
    else:
        fig = plt.figure(figsize=(6.9, 4.6))
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

    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


FI_RATES_HZ = [2, 5, 10, 20, 40, 70, 100]   # uniform Poisson input rates (Hz)
FI_EI = {"coba": "0", "ping": "1.5"}         # ei-strength per condition
FI_T_MS = 300


def fi_sweep() -> dict:
    """Free-running f–I curves under uniform Poisson input: mean per-cell E and
    I firing rate vs input rate, for COBA (loop off) and PING (loop on). No
    training — the bare drive-response of the architecture."""
    out = {c: {"in": [], "e": [], "i": []} for c in FI_EI}
    for rate in FI_RATES_HZ:
        for cell, ei in FI_EI.items():
            for p in (SCOPE_OUT_PNG, SCOPE_OUT_NPZ):
                if p.exists():
                    p.unlink()
            argv = [
                "sim", "--image", "--model", "ping", "--input", "synthetic-spikes",
                "--w-in", "1.5", "0.3", "--ei-strength", ei,
                "--input-rate", str(rate), "--t-ms", str(FI_T_MS),
            ]
            subprocess.run(["uv", "run", "python", str(OSCILLOSCOPE), *argv],
                           cwd=REPO, check=True)
            d = np.load(SCOPE_OUT_NPZ)
            se, si, dt = d["spk_e"], d["spk_i"], float(d["dt"])
            T = se.shape[0]
            e = float(se.sum() / (se.shape[1] * T * dt / 1000.0))
            i = float(si.sum() / (si.shape[1] * T * dt / 1000.0)) if si.size else 0.0
            out[cell]["in"].append(rate)
            out[cell]["e"].append(e)
            out[cell]["i"].append(i)
            print(f"[f-I] {cell} input={rate} Hz → E={e:.1f} Hz I={i:.1f} Hz")
    return out


def plot_raster_compound(
    snaps: dict, fi: dict, out_path: Path, titles: dict,
    include_arch: bool = False,
) -> None:
    """Super figure: COBA vs PING side by side.

    Each condition is a column-pair: a single raster (I stacked above E, no gap)
    spans the pair; below it the population-E Welch PSD sits next to the
    free-running f–I curve. COBA has no I population (loop off).

    With include_arch=True a top row carries each column's architecture
    schematic (COBA / PING) directly above its plots.
    """
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"  # keep the saved 16:9 exact
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(6.9, 3.88))  # 16:9, full print width
    if include_arch:
        # Nested gridspecs so the (small) schematic→plots gap is independent of
        # the (larger) raster→PSD gap that has to clear the time-axis label.
        outer = GridSpec(
            2, 1, figure=fig, height_ratios=[2.5, 7.4], hspace=0.10,
            top=0.93, bottom=0.10, left=0.06, right=0.99,
        )
        arch_gs = outer[0].subgridspec(1, 2, wspace=0.5)
        plot_gs = outer[1].subgridspec(
            2, 4, height_ratios=[4.4, 2.6], hspace=0.4, wspace=0.5,
        )
    else:
        arch_gs = None
        plot_gs = GridSpec(
            2, 4, figure=fig, height_ratios=[4.4, 2.6],
            hspace=0.4, wspace=0.5, top=0.92, bottom=0.12, left=0.06, right=0.99,
        )
    fi_max = max(max(fi[c]["e"] + fi[c]["i"]) for c in ("coba", "ping"))

    for col, cell in enumerate(("coba", "ping")):
        c0 = 2 * col
        s = snaps[cell]
        spk_e, spk_i, dt = s["spk_e"], s["spk_i"], s["dt"]
        T = spk_e.shape[0]
        t_ms = np.arange(T) * dt
        has_i = spk_i.size > 0 and spk_i.shape[0] == T and spk_i.any()

        if include_arch:
            ax_arch = fig.add_subplot(arch_gs[0, col])
            _draw_schematic(ax_arch, cell)
            ax_arch.set_title(titles[cell], loc="left", fontweight="semibold")

        ax_r = fig.add_subplot(plot_gs[0, c0:c0 + 2])    # one raster, I above E
        ax_psd = fig.add_subplot(plot_gs[1, c0])         # PSD next to f–I
        ax_fi = fig.add_subplot(plot_gs[1, c0 + 1])

        # Combined raster: E (black) at the bottom, I (red) stacked directly
        # above it in the same axes — no vertical gap between the populations.
        n_e = spk_e.shape[1]
        e_idx, e_t = np.where(spk_e.T)
        ax_r.scatter(t_ms[e_t], e_idx, s=1.0, c=theme.INK_BLACK, marker="|", linewidths=0.5)
        if has_i:
            n_i = spk_i.shape[1]
            i_idx, i_t = np.where(spk_i.T)
            ax_r.scatter(t_ms[i_t], n_e + i_idx, s=1.0, c=theme.DEEP_RED, marker="|", linewidths=0.5)
            ax_r.axhline(n_e, color=theme.GREY_MID, lw=0.6, alpha=0.6)
            total = n_e + n_i
            ax_r.set_yticks([n_e / 2, n_e + n_i / 2])
            ax_r.set_yticklabels(["E", "I"])
        else:
            total = n_e
            ax_r.set_yticks([n_e / 2])
            ax_r.set_yticklabels(["E"])
            ax_r.text(0.99, 0.97, "no I population (loop off)",
                      transform=ax_r.transAxes, ha="right", va="top",
                      color=theme.GREY_MID, fontstyle="italic",
                      fontsize=theme.SIZE_LABEL - 1)
        ax_r.set_ylim(0, total)
        ax_r.set_xlim(0, T * dt)
        ax_r.set_xlabel("time (ms)")
        if not include_arch:
            ax_r.set_title(titles[cell], loc="left", fontweight="semibold")
        _despine(ax_r)

        # Welch PSD on the population-mean E trace
        freqs, psd, f_peak = _population_psd(spk_e, dt)
        band = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
        ax_psd.plot(freqs[band], psd[band], color=theme.INK_BLACK, lw=1.4)
        ax_psd.set_xlim(F_GAMMA_BAND_HZ)
        ax_psd.set_xlabel("frequency (Hz)")
        ax_psd.set_ylabel("E PSD (a.u.)")
        _despine(ax_psd)
        # Only the loop-on condition has a recurrent gamma peak to mark; the
        # loop-off control has scattered input-driven power, not a rhythm.
        if has_i and f_peak is not None:
            ax_psd.axvline(f_peak, color=theme.DEEP_RED, lw=0.9, ls="--", alpha=0.8)
            ax_psd.text(f_peak, ax_psd.get_ylim()[1] * 0.95,
                        f"  $f_\\gamma$ = {f_peak:.1f} Hz", ha="left", va="top",
                        fontsize=theme.SIZE_LABEL - 1, color=theme.DEEP_RED,
                        fontweight="semibold")
        elif not has_i:
            ax_psd.text(0.97, 0.95, "no recurrent gamma\n(loop off)",
                        transform=ax_psd.transAxes, ha="right", va="top",
                        fontsize=theme.SIZE_LABEL - 1, color=theme.GREY_MID,
                        fontstyle="italic")
        else:
            ax_psd.text(0.97, 0.95, "no clear peak", transform=ax_psd.transAxes,
                        ha="right", va="top", fontsize=theme.SIZE_LABEL - 1,
                        color=theme.GREY_MID, fontstyle="italic")

        # f–I curve (bottom-right of the pair); shared y-scale shows the
        # dynamic-range compression PING applies relative to COBA.
        f = fi[cell]
        ax_fi.plot(f["in"], f["e"], color=theme.INK_BLACK, marker="o", ms=3, lw=1.3, label="E")
        ax_fi.plot(f["in"], f["i"], color=theme.DEEP_RED, marker="s", ms=3, lw=1.3, label="I")
        ax_fi.set_ylim(0, fi_max * 1.05)
        ax_fi.set_xlim(0, max(FI_RATES_HZ))
        ax_fi.set_xlabel("input rate (Hz)")
        ax_fi.set_ylabel("rate (Hz)")
        _despine(ax_fi)
        if col == 1:
            ax_fi.legend(frameon=False, fontsize=theme.SIZE_LABEL - 2, loc="upper left")

    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
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
    PANEL_SIZE = (3.0, 1.6875)  # small per-panel trace tile, 8:4.5 aspect

    def _save_panel(plot_fn, suffix: str, ylabel: str, panel_title: str):
        fig, ax = plt.subplots(figsize=PANEL_SIZE)
        plot_fn(ax)
        ax.set_xlim(0, T * dt)
        ax.set_xlabel("time (ms)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — {panel_title}")
        ax.legend(loc="upper right", fontsize=theme.SIZE_LEGEND)
        fig.tight_layout()
        panel_stem = out_path.with_name(f"{out_path.name}__{suffix}")
        save_figure(fig, panel_stem)  # line traces: SVG + PDF
        plt.close(fig)
        return panel_stem

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


# ─── architecture schematic (manuscript Figure 0) ───────────────────


def _arch_box(ax, cx, cy, w, h, label, fontsize=15):
    """A black-edged population box with a centred monospace label."""
    ax.add_patch(Rectangle(
        (cx - w / 2, cy - h / 2), w, h,
        fill=False, edgecolor=theme.INK_BLACK, lw=1.8, zorder=3,
    ))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, color=theme.INK_BLACK, zorder=4)


def _arch_arrow(ax, x0, y0, x1, y1):
    """A solid black arrow from (x0, y0) to (x1, y1)."""
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
        lw=1.6, color=theme.INK_BLACK, shrinkA=0, shrinkB=0, zorder=3,
    ))


def _arch_label(ax, x, y, text, fontsize=12):
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=theme.INK_BLACK, zorder=4)


def _draw_schematic(ax, kind: str) -> None:
    """Draw one architecture schematic (kind = 'coba' or 'ping') into ax.

    The frame fills the axes (16 × 9, no forced square) so the schematic is
    large; both kinds share the frame and box sizes so the COBA and PING
    panels read at the same scale. Weight labels sit clear of the boxes.
    """
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    bw, bh = 3.0, 2.2          # shared box size
    bf, lf = 13, 10            # box-label / weight-label font sizes
    if kind == "coba":
        _arch_box(ax, 8.0, 4.5, bw, bh, "E", fontsize=bf)
        _arch_arrow(ax, 2.4, 4.5, 6.4, 4.5)        # input → E
        _arch_label(ax, 4.2, 5.7, "W_in", fontsize=lf)
        _arch_arrow(ax, 9.6, 4.5, 13.6, 4.5)       # E → output
        _arch_label(ax, 11.6, 5.7, "W_out", fontsize=lf)
    else:  # ping
        _arch_box(ax, 8.0, 6.4, bw, bh, "E", fontsize=bf)
        _arch_box(ax, 8.0, 2.0, bw, bh, "I", fontsize=bf)
        _arch_arrow(ax, 2.4, 6.4, 6.4, 6.4)        # input → E
        _arch_label(ax, 4.2, 7.6, "W_in", fontsize=lf)
        _arch_arrow(ax, 9.6, 6.4, 13.6, 6.4)       # E → output
        _arch_label(ax, 11.6, 7.6, "W_out", fontsize=lf)
        _arch_arrow(ax, 6.9, 5.3, 6.9, 3.1)        # E → I (down, left)
        _arch_label(ax, 5.0, 4.2, "W_ei", fontsize=lf)
        _arch_arrow(ax, 9.1, 3.1, 9.1, 5.3)        # I → E (up, right)
        _arch_label(ax, 11.0, 4.2, "W_ie", fontsize=lf)


def plot_architecture(out_path: Path) -> None:
    """Draw the COBA (left) and PING (right) schematics on one 16:9 figure."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"  # keep the saved 16:9 exact

    fig, ax = plt.subplots(figsize=(6.9, 3.88))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")  # 16:9 data range in a 16:9 frame, undistorted
    ax.axis("off")

    # ── COBA (left), centred on x = 4 ──────────────────────────────
    ax.text(4.0, 7.6, "COBA", ha="center", va="center",
            fontsize=14, color=theme.INK_BLACK)
    _arch_box(ax, 4.0, 5.0, 2.4, 1.7, "E")
    _arch_arrow(ax, 0.7, 5.0, 2.75, 5.0)        # input → E
    _arch_label(ax, 1.7, 5.5, "W_in")
    _arch_arrow(ax, 5.25, 5.0, 7.3, 5.0)        # E → output
    _arch_label(ax, 6.25, 5.5, "W_out")

    # ── PING (right), centred on x = 12 ────────────────────────────
    ax.text(12.0, 7.6, "PING", ha="center", va="center",
            fontsize=14, color=theme.INK_BLACK)
    _arch_box(ax, 12.0, 5.5, 2.6, 1.7, "E")
    _arch_box(ax, 12.0, 2.5, 2.6, 1.7, "I")
    _arch_arrow(ax, 8.5, 5.5, 10.65, 5.5)       # input → E
    _arch_label(ax, 9.55, 6.0, "W_in")
    _arch_arrow(ax, 13.3, 5.5, 15.4, 5.5)       # E → output
    _arch_label(ax, 14.35, 6.0, "W_out")
    _arch_arrow(ax, 11.3, 4.65, 11.3, 3.35)     # E → I (down, left side)
    _arch_label(ax, 10.25, 4.0, "W_ei")
    _arch_arrow(ax, 12.7, 3.35, 12.7, 4.65)     # I → E (up, right side)
    _arch_label(ax, 13.75, 4.0, "W_ie")

    save_figure(fig, out_path)  # schematic line art: SVG + PDF
    plt.close(fig)


def main() -> None:
    # Publication profile: every figure this notebook writes is a print-sized
    # vector, emitted as both SVG (docs) and PDF (manuscript) by save_figure.
    theme.set_paper_mode(True)

    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    wipe_dir = "--no-wipe-dir" not in sys.argv

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id} tier={tier}")

    prepare_run_dirs(SLUG, notebook_run_id, wipe=wipe_dir, make_artifacts=False)

    figures: dict[str, Path] = {}

    # Architecture schematic — manuscript Figure 0 (pure drawing, no compute).
    arch_dst = FIGURES / "architecture"
    plot_architecture(arch_dst)
    figures["architecture"] = arch_dst
    print(f"wrote {arch_dst}.{{svg,pdf}}")

    snaps: dict[str, dict] = {}
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

        data = np.load(SCOPE_OUT_NPZ)
        snaps[cell] = {
            "spk_e": np.array(data["spk_e"]),
            "spk_i": np.array(data["spk_i"]),
            "dt": float(data["dt"]),
        }

        traces_dst = FIGURES / f"traces__{cell}"
        panel_paths = plot_traces(SCOPE_OUT_NPZ, traces_dst, spec["title"])
        for p in panel_paths:
            figures[p.stem] = p
            print(f"wrote {p}.{{svg,pdf}}")

    # Free-running f–I curves (uniform Poisson sweep), folded into the super figure.
    fi = fi_sweep()

    column_titles = {
        "coba": "A   COBA — recurrent loop off",
        "ping": "B   PING — recurrent loop active",
    }

    # Super figure: COBA vs PING side by side (raster, PSD + f–I). Used by the
    # manuscript's Claim 1 (the architecture is its own Claim 0 figure there).
    compound_dst = FIGURES / "raster_compound"
    plot_raster_compound(snaps, fi, compound_dst, column_titles)
    figures["raster_compound"] = compound_dst
    print(f"wrote {compound_dst}.{{png,pdf}}")

    # Merged overview for this notebook: each column's architecture schematic
    # sits directly above its raster / PSD / f–I plots.
    overview_dst = FIGURES / "overview_compound"
    plot_raster_compound(snaps, fi, overview_dst, column_titles, include_arch=True)
    figures["overview_compound"] = overview_dst
    print(f"wrote {overview_dst}.{{png,pdf}}")

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
