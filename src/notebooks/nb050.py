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

from cli import theme  # noqa: E402
from _modal import parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402

SLUG = "nb050"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "cli/cli.py"
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
    # Lyapunov / chaos probe: rerun on identical frozen input with an
    # ε = 2 mV initial membrane perturbation and record the spike-train
    # divergence D(t). PING re-locks to D = 0 (stable limit cycle);
    # AI sustains a small slowly-growing divergence (weak intrinsic chaos
    # riding on top of input entrainment).
    "--lyapunov-eps", "2.0",
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
            #   - --ei-sparsity 0.99 + --exact-k → fixed fan-in K ≈ 10 per
            #     post cell (Brunel/V&S convention; removes the binomial
            #     fan-in variance that otherwise broadens the rate dist)
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
            "--exact-k",
            "--independent-drive", "45", "0.38",
            "--independent-drive-i", "8", "0.25",
        ],
        "title": "Balanced E/I (V&S) — fixed-K, per-E + per-I drive, CV ≈ 1",
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


def _per_cell_fluctuation_drive(ge_e, gi_e, v_e, burn_steps: int):
    """Per-E-cell V&S signature: total synaptic + leak current $I_{tot}(t)$,
    its time-mean $\\mu_I$, time-std $\\sigma_I$, and the fluctuation-vs-mean
    ratio $\\eta = \\sigma_I / |I_{th} - \\mu_I|$ where $I_{th}$ is the
    constant current needed to maintain V at threshold.

    V&S predicts mean drives cancel to leave $\\mu_I$ well below the
    threshold-needed value, with fluctuations $\\sigma_I$ comparable to
    the gap (so the cell fires only when noise pushes V across $V_{th}$).
    PING in contrast has a strong oscillatory $I_{tot}(t)$ where the
    deterministic cycle structure dominates the variance.

    Constants: E_e = 0, E_i = -80, E_L = -65, V_th = -50 mV, g_L = 0.05 μS.
    I_{th} = g_L (V_th - E_L) = 0.75 nA.
    """
    E_e = 0.0
    E_i = -80.0
    E_L = -65.0
    V_th = -50.0
    g_L = 0.05
    I_th = g_L * (V_th - E_L)  # 0.75 nA threshold-current

    ge = ge_e[burn_steps:]
    gi = gi_e[burn_steps:]
    v = v_e[burn_steps:]
    # Total membrane current per cell per timestep (positive depolarises).
    I_tot = ge * (E_e - v) + gi * (E_i - v) + g_L * (E_L - v)
    mu = I_tot.mean(axis=0)       # (N_E,) mean drive
    sigma = I_tot.std(axis=0)     # (N_E,) fluctuation amplitude
    gap = I_th - mu               # how much extra mean current would be needed
    eta = sigma / np.maximum(np.abs(gap), 1e-9)
    return mu, sigma, gap, eta, I_th


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


def plot_current_balance(npz_path: Path, out_path: Path, title: str) -> dict:
    """Two-panel V&S fluctuation-vs-mean diagnostic.

    Left: scatter of per-cell time-mean $\\mu_I$ vs time-std $\\sigma_I$ of
    the total membrane current $I_{tot}(t) = g_e(E_e - V) + g_i(E_i - V)
    + g_L(E_L - V)$. Two reference lines:
      - $\\mu_I = I_{th}$ (vertical): cells right of this would fire from
        the mean alone (no fluctuations needed).
      - $\\sigma_I = I_{th} - \\mu_I$ (diagonal, dashed): on this line the
        fluctuation amplitude equals the distance to threshold, the
        boundary of fluctuation-driven firing. V&S cells sit on or above
        the diagonal *left* of the vertical line. Mean-driven cells sit
        *right* of the vertical line.

    Right: histogram of $\\eta = \\sigma_I / |I_{th} - \\mu_I|$ across cells,
    the V&S single-number signature. Reference line at $\\eta = 1$:
      - $\\eta \\gg 1$: fluctuations dominate the gap to threshold →
        fluctuation-driven firing (V&S balanced state).
      - $\\eta \\ll 1$: cell's mean drive is well below threshold and
        fluctuations are too small to drive it across — cell is silent.
    """
    theme.apply()
    from matplotlib.gridspec import GridSpec
    data = np.load(npz_path)
    spk_e = data["spk_e"]
    if "ge_e_1" not in data:
        return {
            "median_eta_fluct_drive": None, "frac_above_eta_1": None,
            "median_mu_I": None, "median_sigma_I": None,
        }
    ge_e = data["ge_e_1"]
    gi_e = data["gi_e_1"]
    v_e = data["v_e_1"]
    burn_steps = max(0, ge_e.shape[0] - spk_e.shape[0])
    mu, sigma, gap, eta, I_th = _per_cell_fluctuation_drive(
        ge_e, gi_e, v_e, burn_steps,
    )
    valid = np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(eta)
    mu = mu[valid]
    sigma = sigma[valid]
    eta = eta[valid]
    gap = gap[valid]
    if mu.size == 0:
        return {
            "median_eta_fluct_drive": None, "frac_above_eta_1": None,
            "median_mu_I": None, "median_sigma_I": None,
        }

    fig = plt.figure(figsize=(11.0, 4.4), dpi=150)
    gs = GridSpec(
        1, 2, figure=fig, width_ratios=[1.0, 1.0],
        wspace=0.30, top=0.86, bottom=0.16, left=0.08, right=0.97,
    )
    ax_sc = fig.add_subplot(gs[0])
    ax_hi = fig.add_subplot(gs[1])

    # Scatter: mean current vs fluctuation amplitude per cell.
    ax_sc.scatter(mu, sigma, s=4.0, c=theme.INK_BLACK, alpha=0.35, linewidths=0)
    x_lo = float(min(mu.min(), 0.0))
    x_hi = float(max(mu.max(), I_th * 1.2))
    y_hi = float(max(sigma.max(), abs(I_th - mu.min())))
    # Vertical: μ = I_th
    ax_sc.axvline(
        I_th, color=theme.DEEP_RED, lw=0.9, ls=":", alpha=0.7,
    )
    # Diagonal: σ = I_th - μ (only meaningful where μ ≤ I_th)
    xs = np.linspace(x_lo, I_th, 50)
    ax_sc.plot(
        xs, I_th - xs, color=theme.DEEP_RED, lw=0.9, ls="--", alpha=0.7,
    )
    ax_sc.set_xlim(x_lo - 0.5, x_hi + 0.5)
    ax_sc.set_ylim(0, y_hi * 1.1)
    ax_sc.set_xlabel("$\\mu_I$  (time-mean total current, nA)")
    ax_sc.set_ylabel("$\\sigma_I$  (time-std, nA)")
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)
    ax_sc.set_title(
        f"Per-cell drive: mean vs fluctuation (n = {mu.size} E cells)",
        loc="left", fontsize=theme.SIZE_LABEL, pad=4,
    )
    ax_sc.text(
        0.02, 0.97,
        f"I$_{{th}}$ = {I_th:.2f} nA (dotted)\n$\\sigma$ = I$_{{th}}$ − $\\mu$ (dashed)",
        transform=ax_sc.transAxes, ha="left", va="top",
        fontsize=theme.SIZE_LABEL - 1, color=theme.LABEL,
    )

    # Histogram of η = σ / |I_th - μ| (clip extremes for display).
    eta_disp = np.clip(eta, 0, 5.0)
    median_eta = float(np.median(eta))
    frac_above_1 = float((eta >= 1.0).mean())
    ax_hi.hist(
        eta_disp, bins=np.linspace(0, 5.0, 50),
        color=theme.INK_BLACK, alpha=0.7,
    )
    ax_hi.axvline(1.0, color=theme.DEEP_RED, lw=1.0, ls="--", alpha=0.85)
    ax_hi.text(
        1.0, ax_hi.get_ylim()[1] * 0.95,
        "  η = 1  (fluct ≈ gap)",
        ha="left", va="top",
        fontsize=theme.SIZE_LABEL - 1, color=theme.DEEP_RED,
    )
    ax_hi.set_xlim(0, 5.0)
    ax_hi.set_xlabel("$\\eta = \\sigma_I / |I_{th} - \\mu_I|$")
    ax_hi.set_ylabel("E-cell count")
    ax_hi.spines["top"].set_visible(False)
    ax_hi.spines["right"].set_visible(False)
    ax_hi.set_title(
        "Fluctuation-vs-mean ratio (V&S signature)",
        loc="left", fontsize=theme.SIZE_LABEL, pad=4,
    )
    ax_hi.text(
        0.97, 0.97,
        f"median η = {median_eta:.2f}\n{frac_above_1:.0%} of cells have η ≥ 1",
        transform=ax_hi.transAxes, ha="right", va="top",
        fontsize=theme.SIZE_LABEL - 1, color=theme.LABEL,
        fontweight="semibold",
    )

    fig.suptitle(title, fontsize=theme.SIZE_TITLE, x=0.08, ha="left")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "median_eta_fluct_drive": median_eta,
        "frac_above_eta_1": frac_above_1,
        "median_mu_I": float(np.median(mu)),
        "median_sigma_I": float(np.median(sigma)),
    }


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
        fig = plt.figure(figsize=(9.0, 9.5), dpi=150)
        gs = GridSpec(
            5, 1, figure=fig,
            height_ratios=[4.0, 1.2, 2.4, 1.6, 1.8],
            hspace=0.95, top=0.94, bottom=0.05, left=0.12, right=0.97,
        )
        ax_e = fig.add_subplot(gs[0])
        ax_i = fig.add_subplot(gs[1], sharex=ax_e)
        ax_psd = fig.add_subplot(gs[2])
        ax_cv = fig.add_subplot(gs[3])
        ax_xcorr = fig.add_subplot(gs[4])
    else:
        fig = plt.figure(figsize=(9.0, 8.0), dpi=150)
        gs = GridSpec(
            4, 1, figure=fig,
            height_ratios=[4.0, 2.4, 1.6, 1.8],
            hspace=0.70, top=0.94, bottom=0.06, left=0.12, right=0.97,
        )
        ax_e = fig.add_subplot(gs[0])
        ax_i = None
        ax_psd = fig.add_subplot(gs[1])
        ax_cv = fig.add_subplot(gs[2])
        ax_xcorr = fig.add_subplot(gs[3])

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

    # Mean pairwise cross-correlogram on E cells.
    lags_ms, xcorr, peak_abs = _pair_cross_correlogram(spk_e, dt)
    ax_xcorr.plot(lags_ms, xcorr, color=theme.INK_BLACK, lw=1.2)
    ax_xcorr.axhline(0.0, color=theme.GREY_MID, lw=0.6, alpha=0.7)
    ax_xcorr.axvline(0.0, color=theme.GREY_MID, lw=0.6, alpha=0.7)
    ax_xcorr.set_xlim(lags_ms[0], lags_ms[-1])
    ax_xcorr.set_xlabel("lag (ms)")
    ax_xcorr.set_ylabel("mean pairwise C(τ)")
    ax_xcorr.spines["top"].set_visible(False)
    ax_xcorr.spines["right"].set_visible(False)
    ax_xcorr.set_title(
        "Pairwise spike-train cross-correlation (100 random E pairs)",
        loc="left", fontsize=theme.SIZE_LABEL, pad=4,
    )
    ax_xcorr.text(
        0.99, 0.95,
        f"peak |C| = {peak_abs:.3f}",
        transform=ax_xcorr.transAxes, ha="right", va="top",
        fontsize=theme.SIZE_LABEL - 1, color=theme.LABEL,
        fontweight="semibold",
    )

    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lyapunov(lyap_by_cell: dict, out_path: Path, run_id: str) -> None:
    """Compare spike-train divergence D(t) across cells on one axis.

    D(t) = number of E cells whose spike differs between the clean run and
    an ε-perturbed rerun on identical frozen input. PING re-locks to 0
    (stable limit cycle, λ ≤ 0); the V&S AI cell sustains a small
    slowly-growing divergence (weak intrinsic chaos riding on top of the
    strong input entrainment that the shared frozen drive imposes).
    """
    theme.apply()
    colours = {"ping": theme.MUTED, "ai": theme.DEEP_RED}
    labels = {"ping": "PING", "ai": "V&S AI"}
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    for cell, (t_ms, dist) in lyap_by_cell.items():
        # Smooth lightly (10 ms boxcar) so the per-step jitter doesn't
        # swamp the trend.
        w = max(1, int(10.0 / (t_ms[1] - t_ms[0]))) if len(t_ms) > 1 else 1
        if w > 1:
            kernel = np.ones(w) / w
            smooth = np.convolve(dist, kernel, mode="same")
        else:
            smooth = dist
        ax.plot(
            t_ms, smooth,
            color=colours.get(cell, theme.INK_BLACK), lw=1.4,
            label=labels.get(cell, cell),
        )
    ax.set_xlabel("time since perturbation (ms)")
    ax.set_ylabel("spike-train divergence  D(t)  (E cells differing)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    ax.set_title(
        "Chaos probe: spike-train divergence after ε = 2 mV perturbation",
        fontsize=theme.SIZE_TITLE, loc="left",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    lyap_by_cell: dict[str, tuple] = {}
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

        current_dst = FIGURES / f"current_balance__{cell}.png"
        current_stats = plot_current_balance(
            SCOPE_OUT_NPZ, current_dst, spec["title"],
        )
        figures[f"current_balance__{cell}"] = current_dst
        print(f"wrote {current_dst}")

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
        _, _, peak_abs_xcorr = _pair_cross_correlogram(spk_e, dt)

        # Lyapunov / chaos: spike-train divergence curve D(t) from the
        # perturbed rerun (saved in the npz by the snapshot generator).
        # The discriminating quantity is the steady-state divergence level
        # (mean D over the second half): ≈ 0 for a stable limit cycle that
        # re-locks (PING), > 0 and sustained for chaos (AI).
        lyap_final = lyap_max = lyap_steady = None
        if "lyap_dist" in data:
            lt = np.asarray(data["lyap_t_ms"])
            ld = np.asarray(data["lyap_dist"])
            lyap_by_cell[cell] = (lt, ld)
            lyap_final = float(ld[-1])
            lyap_max = float(ld.max())
            half = len(ld) // 2
            lyap_steady = float(ld[half:].mean())

        summary_rows.append({
            "cell": cell,
            "e_rate_hz": e_rate,
            "i_rate_hz": i_rate,
            "median_isi_cv_e": med_cv_e,
            "median_isi_cv_i": med_cv_i,
            "f_psd_peak_hz": f_peak,
            "peak_abs_xcorr_e": peak_abs_xcorr,
            "lyap_final_diff_cells": lyap_final,
            "lyap_max_diff_cells": lyap_max,
            "lyap_steady_diff_cells": lyap_steady,
            **current_stats,
        })

    # Comparison figure: spike-train divergence D(t) for all cells.
    if lyap_by_cell:
        lyap_dst = FIGURES / "lyapunov_divergence.png"
        plot_lyapunov(lyap_by_cell, lyap_dst, notebook_run_id)
        figures["lyapunov_divergence"] = lyap_dst
        print(f"wrote {lyap_dst}")

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
