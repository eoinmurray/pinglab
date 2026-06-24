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

CELLS: dict[str, dict] = {
    "ping": {
        "args": [
            # Shared 20 Hz Poisson input layer through a dense W_in — the
            # COMMON (correlated) drive that lets the recurrent loop phase-lock
            # into gamma. This correlated drive is load-bearing: replacing it
            # with the AI cell's per-cell independent streams desynchronises
            # PING into a near-asynchronous state (gamma cross-correlation
            # collapses from ≈ 0.05 to ≈ 0.01). So the two regimes necessarily
            # differ in their input as well as their wiring.
            "--input-rate", "20",
            "--w-in", "1.5", "0.3",
            # E↔I weights matched to the AI cell — both W_EI and W_IE are now
            # shared parameters. Verified incidental: PING gamma holds (peak
            # ≈ 29 Hz, cross-corr ≈ 0.05 vs the AI floor 0.009). So the E↔I
            # coupling is NOT what switches the two regimes.
            "--w-ei", "0.6", "0.18",
            "--w-ie", "3.0", "0.9",
            # No --w-ii ⇒ canonical PING (no I→I); dense connectivity.
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
            #   - --independent-drive 45 0.38 — per-cell independent (un-
            #     correlated) Poisson on E; the asynchronous state needs this
            #   - --independent-drive-i 8 0.25 — same for I (without this
            #     I-cell CV stays correlated with E via W^EI)
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


def _capture_rep_cell(data, spk_e: np.ndarray) -> dict | None:
    """Pull one representative E cell's voltage + conductance traces from a
    snapshot, for the single-neuron intuition figure. Picks the active cell
    whose spike count is the median of the active population (a typical cell,
    not an outlier). Returns None if the snapshot has no voltage trace."""
    if "v_e_1" not in data:
        return None
    counts = spk_e.sum(axis=0)
    active = np.where(counts >= 5)[0]
    if active.size == 0:
        return None
    # 70th-percentile-by-rate cell: active enough to spike visibly in a short
    # window, still a typical cell rather than the busiest outlier.
    rep = int(active[np.argsort(counts[active])[int(active.size * 0.7)]])
    ge, gi, v = data["ge_e_1"], data["gi_e_1"], data["v_e_1"]
    burn = max(0, ge.shape[0] - spk_e.shape[0])
    return {
        "v": np.asarray(v[burn:, rep], dtype=np.float64),
        "ge": np.asarray(ge[burn:, rep], dtype=np.float64),
        "gi": np.asarray(gi[burn:, rep], dtype=np.float64),
        "spk": np.asarray(spk_e[:, rep], dtype=np.float64),
        "dt": float(data["dt"]),
        "rep_idx": rep,
    }


def plot_single_neuron_intuition(captured: dict, out_path: Path) -> dict:
    """The pedagogical keystone: why balance ⇒ irregularity, at one cell.

    Two columns (PING | AI), two rows:
      - Top: a representative E cell's membrane voltage $V(t)$ over a short
        window, threshold marked. PING rides the gamma pump and fires
        cycle-locked; the AI cell wanders and crosses threshold at random.
      - Bottom: the distribution of that cell's *net* input current
        $I_{tot}(t)$, with the threshold current $I_{th}=0.75$ nA marked.
        The AI cell's mean sits near $I_{th}$ with fluctuations that
        straddle it (fluctuation-driven); the PING cell's distribution is
        the deterministic cycle, not noise.
    """
    theme.apply()
    from matplotlib.gridspec import GridSpec
    E_e, E_i, E_L, V_th, g_L = 0.0, -80.0, -65.0, -50.0, 0.05
    I_th = g_L * (V_th - E_L)  # 0.75 nA

    order = [c for c in ("ping", "ai") if c in captured]
    titles = {"ping": "PING — cycle-locked", "ai": "V&S AI — fluctuation-driven"}
    colors = {"ping": "#7570b3", "ai": "#1b9e77"}

    fig = plt.figure(figsize=(9, 5))
    gs = GridSpec(2, len(order), figure=fig, height_ratios=[1.4, 1.0],
                  hspace=0.42, wspace=0.22)
    stats: dict = {}
    for col, cell in enumerate(order):
        rec = captured[cell]
        dt = rec["dt"]
        v, ge, gi = rec["v"], rec["ge"], rec["gi"]
        T = v.shape[0]
        t_ms = np.arange(T) * dt
        # Net membrane current (positive depolarises).
        I_tot = ge * (E_e - v) + gi * (E_i - v) + g_L * (E_L - v)
        mu, sigma = float(I_tot.mean()), float(I_tot.std())
        eta = sigma / max(abs(I_th - mu), 1e-9)
        stats[cell] = {"mu_I": mu, "sigma_I": sigma, "eta": eta}

        # --- Top: voltage trace over a 300 ms window past the transient ---
        # The integrator stores V *after* reset, so threshold crossings are
        # invisible in the raw trace. Draw each spike back in to a fixed peak
        # so the crossings — the whole point — are unmistakable.
        ax_v = fig.add_subplot(gs[0, col])
        # Center a 300 ms window on this cell's own spiking so the crossings
        # are actually in frame (sparse cells would otherwise look silent).
        win = int(300.0 / dt)
        spk_all = np.where(rec["spk"] > 0)[0]
        if spk_all.size:
            center = int(np.median(spk_all))
            w0 = int(np.clip(center - win // 2, 0, max(0, T - win)))
        else:
            w0 = int(min(150.0 / dt, T * 0.2))
        w1 = int(min(w0 + win, T))
        sl = slice(w0, w1)
        SPIKE_PEAK = -40.0
        v_disp = v.copy()
        v_disp[rec["spk"] > 0] = SPIKE_PEAK
        ax_v.plot(t_ms[sl], v_disp[sl], color=colors[cell], lw=0.9)
        ax_v.axhline(V_th, color="#d95f02", lw=0.9, ls="--",
                     label="threshold")
        ax_v.set_ylim(-82, -36)
        ax_v.set_title(titles[cell], fontsize=10)
        ax_v.set_xlabel("time (ms)")
        if col == 0:
            ax_v.set_ylabel("membrane V (mV)")
            ax_v.legend(loc="lower right", fontsize=7)

        # --- Bottom: net-current distribution, threshold marked ---
        ax_h = fig.add_subplot(gs[1, col])
        ax_h.hist(I_tot, bins=60, color=colors[cell], alpha=0.8,
                  density=True)
        ax_h.axvline(I_th, color="#d95f02", lw=1.0, ls="--",
                     label="$I_{th}$")
        ax_h.axvline(mu, color="#333333", lw=1.0, ls=":",
                     label="mean")
        ax_h.set_xlabel("net input current $I_{tot}$ (nA)")
        if col == 0:
            ax_h.set_ylabel("density")
            ax_h.legend(loc="upper right", fontsize=7)
        ax_h.text(0.04, 0.92, f"σ = {sigma:.1f} nA", transform=ax_h.transAxes,
                  fontsize=8, va="top")

    fig.suptitle("One cell, two regimes: why balance makes firing irregular",
                 fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")
    return stats


def _current_balance_arrays(data):
    """Per-E-cell (mu, sigma, eta, I_th) for the fluctuation-vs-mean
    diagnostic, with non-finite cells dropped. Returns None if the snapshot
    carries no voltage trace."""
    spk_e = data["spk_e"]
    if "ge_e_1" not in data:
        return None
    ge_e, gi_e, v_e = data["ge_e_1"], data["gi_e_1"], data["v_e_1"]
    burn_steps = max(0, ge_e.shape[0] - spk_e.shape[0])
    mu, sigma, gap, eta, I_th = _per_cell_fluctuation_drive(
        ge_e, gi_e, v_e, burn_steps)
    valid = np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(eta)
    if not valid.any():
        return None
    return mu[valid], sigma[valid], eta[valid], I_th


def _current_balance_stats(data) -> dict:
    """Summary-row statistics for the fluctuation-vs-mean diagnostic."""
    arr = _current_balance_arrays(data)
    if arr is None:
        return {"median_eta_fluct_drive": None, "frac_above_eta_1": None,
                "median_mu_I": None, "median_sigma_I": None}
    mu, sigma, eta, _ = arr
    return {
        "median_eta_fluct_drive": float(np.median(eta)),
        "frac_above_eta_1": float((eta >= 1.0).mean()),
        "median_mu_I": float(np.median(mu)),
        "median_sigma_I": float(np.median(sigma)),
    }


def _draw_current_balance_column(ax_sc, ax_hi, data, row_title: str, *,
                                 top: bool) -> None:
    """Draw the two fluctuation-vs-mean panels for one regime.

    Scatter (ax_sc): per-cell time-mean $\\mu_I$ vs time-std $\\sigma_I$ of
    the total membrane current, with $\\mu_I = I_{th}$ (dotted) and the
    fluctuation-driven boundary $\\sigma_I = I_{th} - \\mu_I$ (dashed).
    Histogram (ax_hi): $\\eta = \\sigma_I / |I_{th} - \\mu_I|$, the V&S
    single-number signature, with $\\eta = 1$ marked. Descriptive titles go
    on the top row only; the per-regime label and numbers go on every row."""
    arr = _current_balance_arrays(data)
    if arr is None:
        return
    mu, sigma, eta, I_th = arr

    ax_sc.scatter(mu, sigma, s=4.0, c=theme.INK_BLACK, alpha=0.35, linewidths=0)
    x_lo = float(min(mu.min(), 0.0))
    x_hi = float(max(mu.max(), I_th * 1.2))
    y_hi = float(max(sigma.max(), abs(I_th - mu.min())))
    ax_sc.axvline(I_th, color=theme.DEEP_RED, lw=0.9, ls=":", alpha=0.7)
    xs = np.linspace(x_lo, I_th, 50)
    ax_sc.plot(xs, I_th - xs, color=theme.DEEP_RED, lw=0.9, ls="--", alpha=0.7)
    ax_sc.set_xlim(x_lo - 0.5, x_hi + 0.5)
    ax_sc.set_ylim(0, y_hi * 1.1)
    ax_sc.set_xlabel("$\\mu_I$  (time-mean total current, nA)")
    ax_sc.set_ylabel(f"{row_title}\n$\\sigma_I$  (time-std, nA)")
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)
    if top:
        ax_sc.set_title(
            "Per-cell drive: mean vs fluctuation",
            loc="left", fontsize=theme.SIZE_LABEL, pad=4)
    ax_sc.text(
        0.02, 0.97,
        f"I$_{{th}}$ = {I_th:.2f} nA (dotted)\n$\\sigma$ = I$_{{th}}$ − $\\mu$ (dashed)",
        transform=ax_sc.transAxes, ha="left", va="top",
        fontsize=theme.SIZE_LABEL - 1, color=theme.LABEL)

    eta_disp = np.clip(eta, 0, 5.0)
    median_eta = float(np.median(eta))
    frac_above_1 = float((eta >= 1.0).mean())
    ax_hi.hist(eta_disp, bins=np.linspace(0, 5.0, 50), color=theme.INK_BLACK,
               alpha=0.7)
    ax_hi.axvline(1.0, color=theme.DEEP_RED, lw=1.0, ls="--", alpha=0.85)
    ax_hi.text(1.0, ax_hi.get_ylim()[1] * 0.95, "  η = 1  (fluct ≈ gap)",
               ha="left", va="top", fontsize=theme.SIZE_LABEL - 1,
               color=theme.DEEP_RED)
    ax_hi.set_xlim(0, 5.0)
    ax_hi.set_xlabel("$\\eta = \\sigma_I / |I_{th} - \\mu_I|$")
    ax_hi.set_ylabel("E-cell count")
    ax_hi.spines["top"].set_visible(False)
    ax_hi.spines["right"].set_visible(False)
    if top:
        ax_hi.set_title("Fluctuation-vs-mean ratio (V&S signature)",
                        loc="left", fontsize=theme.SIZE_LABEL, pad=4)
    ax_hi.text(0.97, 0.97,
               f"median η = {median_eta:.2f}\n{frac_above_1:.0%} have η ≥ 1",
               transform=ax_hi.transAxes, ha="right", va="top",
               fontsize=theme.SIZE_LABEL - 1, color=theme.LABEL,
               fontweight="semibold")


def plot_current_balance_compare(snaps: dict, out_path: Path) -> None:
    """PING (top) and V&S AI (bottom), each with its mean-vs-fluctuation
    scatter (left) and η histogram (right), so the two regimes' fluctuation
    structure sits one above the other."""
    theme.apply()
    from matplotlib.gridspec import GridSpec
    order = [c for c in ("ping", "ai") if c in snaps
             and _current_balance_arrays(np.load(snaps[c])) is not None]
    if not order:
        return
    row_titles = {"ping": "PING", "ai": "V&S AI"}
    fig = plt.figure(figsize=(11.0, 8.4), dpi=150)
    gs = GridSpec(len(order), 2, figure=fig, width_ratios=[1.0, 1.0],
                  wspace=0.28, hspace=0.42,
                  top=0.93, bottom=0.09, left=0.10, right=0.97)
    for row, cell in enumerate(order):
        data = np.load(snaps[cell])
        ax_sc = fig.add_subplot(gs[row, 0])
        ax_hi = fig.add_subplot(gs[row, 1])
        _draw_current_balance_column(ax_sc, ax_hi, data,
                                     row_titles.get(cell, cell),
                                     top=(row == 0))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_current_decomposition(npz_path: Path, out_path: Path, title: str) -> dict:
    """The V&S cancellation, made visible (their Eq. 1-3 / Fig. 1 idea).

    Left: one representative E cell's membrane currents over a window —
    the large *excitatory* current $I_E = g_E(E_e - V)$, the large
    *inhibitory* current $I_I = g_I(E_i - V)$, and their sum $I_E + I_I$.
    The two large opposing currents nearly cancel, leaving a small,
    noisy net that hovers around the threshold current $I_{th}$.

    Right: across all E cells, the per-cell time-means $\\langle I_E\\rangle$,
    $|\\langle I_I\\rangle|$ and the net $\\langle I_E + I_I + I_L\\rangle$.
    $\\langle I_E\\rangle \\approx |\\langle I_I\\rangle| \\gg$ net is the balance
    condition: the $O(\\sqrt K)$ excitatory and inhibitory drives cancel to
    an $O(1)$ residue.
    """
    theme.apply()
    from matplotlib.gridspec import GridSpec
    E_e, E_i, E_L, V_th, g_L = 0.0, -80.0, -65.0, -50.0, 0.05
    I_th = g_L * (V_th - E_L)
    data = np.load(npz_path)
    spk_e = data["spk_e"]
    if "ge_e_1" not in data:
        return {"median_I_exc": None, "median_I_inh": None,
                "median_I_net": None, "cancellation_ratio": None}
    ge, gi, v = data["ge_e_1"], data["gi_e_1"], data["v_e_1"]
    dt = float(data["dt"])
    burn = max(0, ge.shape[0] - spk_e.shape[0])
    ge, gi, v = ge[burn:], gi[burn:], v[burn:]
    I_E = ge * (E_e - v)            # (T, N) excitatory, > 0
    I_I = gi * (E_i - v)            # (T, N) inhibitory, < 0
    I_L = g_L * (E_L - v)
    mu_E = I_E.mean(axis=0)         # per-cell time-means
    mu_I = I_I.mean(axis=0)
    mu_net = (I_E + I_I + I_L).mean(axis=0)
    # representative cell: the one whose net drive is closest to the median
    med_net = np.median(mu_net)
    rep = int(np.argmin(np.abs(mu_net - med_net)))

    fig = plt.figure(figsize=(11.0, 4.4), dpi=150)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.35, 1.0],
                  wspace=0.28, top=0.86, bottom=0.16, left=0.08, right=0.97)
    ax_ts = fig.add_subplot(gs[0])
    ax_ba = fig.add_subplot(gs[1])

    # Left: representative-cell current traces over a 250 ms window.
    t = np.arange(ge.shape[0]) * dt
    w = (t >= 300.0) & (t < 550.0)
    ax_ts.plot(t[w], I_E[w, rep], color=theme.ELECTRIC_CYAN, lw=1.0,
               label="$I_E = g_E(E_e - V)$  (excitatory)")
    ax_ts.plot(t[w], I_I[w, rep], color=theme.DEEP_RED, lw=1.0,
               label="$I_I = g_I(E_i - V)$  (inhibitory)")
    ax_ts.plot(t[w], (I_E + I_I + I_L)[w, rep], color=theme.INK_BLACK, lw=1.4,
               label="net  $I_E + I_I + I_L$")
    ax_ts.axhline(I_th, color=theme.AMBER, lw=0.8, ls=":")
    ax_ts.axhline(0.0, color=theme.GREY_MID, lw=0.5)
    ax_ts.text(t[w][-1], I_th, " $I_{th}$", color=theme.AMBER,
               va="center", ha="left", fontsize=theme.SIZE_LABEL - 1)
    ax_ts.set_xlabel("time (ms)")
    ax_ts.set_ylabel("membrane current (nA)")
    ax_ts.spines["top"].set_visible(False)
    ax_ts.spines["right"].set_visible(False)
    ax_ts.legend(fontsize=theme.SIZE_LEGEND - 1, frameon=False, loc="upper right",
                 ncol=1)
    ax_ts.set_title(f"One E cell: two large currents, small sum (cell {rep})",
                    loc="left", fontsize=theme.SIZE_LABEL, pad=4)

    # Right: population balance bars (median +/- IQR).
    def _stat(a):
        return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))
    me, me_lo, me_hi = _stat(mu_E)
    mi, mi_lo, mi_hi = _stat(np.abs(mu_I))
    mn, mn_lo, mn_hi = _stat(mu_net)
    xs = [0, 1, 2]
    vals = [me, mi, mn]
    errs = [[me - me_lo, mi - mi_lo, mn - mn_lo], [me_hi - me, mi_hi - mi, mn_hi - mn]]
    cols = [theme.ELECTRIC_CYAN, theme.DEEP_RED, theme.INK_BLACK]
    ax_ba.bar(xs, vals, yerr=errs, color=cols, alpha=0.8, capsize=3, width=0.6)
    ax_ba.axhline(I_th, color=theme.AMBER, lw=0.8, ls=":")
    ax_ba.text(2.4, I_th, "$I_{th}$", color=theme.AMBER, va="center",
               ha="left", fontsize=theme.SIZE_LABEL - 1)
    ax_ba.set_xticks(xs)
    ax_ba.set_xticklabels(["$\\langle I_E\\rangle$", "$|\\langle I_I\\rangle|$",
                           "net"])
    ax_ba.set_ylabel("per-cell mean current (nA)")
    ax_ba.spines["top"].set_visible(False)
    ax_ba.spines["right"].set_visible(False)
    ratio = abs(mn) / me if me else float("nan")
    ax_ba.set_title(f"Population balance: net / exc = {ratio:.2f}",
                    loc="left", fontsize=theme.SIZE_LABEL, pad=4)

    fig.suptitle(title, fontsize=theme.SIZE_TITLE, x=0.08, ha="left")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"median_I_exc": me, "median_I_inh": float(np.median(mu_I)),
            "median_I_net": mn, "cancellation_ratio": float(ratio)}


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
        ax_cv.text(0.99, 0.95,
                   f"median CV = {np.median(cvs):.2f}  (n = {cvs.size})",
                   transform=ax_cv.transAxes, ha="right", va="top",
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
    1024-neuron input layer at 20 Hz through a dense W_in (a common drive to
    all E cells); the V&S AI cell gets one private 45 Hz Poisson stream per E
    cell. Top row: a raster of the drive over a 300 ms window. Bottom row:
    its population rate over time — featureless, steady Poisson in both, so
    the rhythm in PING is generated by the network, not inherited."""
    theme.apply()
    from matplotlib.gridspec import GridSpec
    INPUT_CLR = "#2c7fb8"
    order = [c for c in ("ping", "ai") if c in snaps]
    titles = {"ping": "PING — input layer, 20 Hz Poisson (dense $W_{in}$ to all E)",
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


def _strip_flag(args: list[str], flag: str, nvals: int) -> list[str]:
    out, i = [], 0
    while i < len(args):
        if args[i] == flag:
            i += 1 + nvals
        else:
            out.append(args[i]); i += 1
    return out


def _run_scope(extra_args: list[str], t_ms: str = "1000"):
    """Run one oscilloscope sim and return the loaded snapshot npz."""
    for p in (SCOPE_OUT_PNG, SCOPE_OUT_NPZ):
        if p.exists():
            p.unlink()
    argv = ["sim", "--image", "--model", "ping", "--input", "synthetic-spikes",
            "--t-ms", t_ms, *extra_args]
    subprocess.run(["uv", "run", "python", str(OSCILLOSCOPE), *argv],
                   cwd=REPO, check=True)
    return np.load(SCOPE_OUT_NPZ)


def _run_scope_rates(extra_args: list[str], t_ms: str = "1000") -> tuple[float, float]:
    """Run one oscilloscope sim and return (r_E, r_I) population rates in Hz."""
    data = _run_scope(extra_args, t_ms)
    spk_e, spk_i = data["spk_e"], data["spk_i"]
    dt = float(data["dt"])
    r_e = float(spk_e.mean() * 1000.0 / dt)
    r_i = float(spk_i.mean() * 1000.0 / dt) if spk_i.size > 0 else 0.0
    return r_e, r_i


def sqrtk_sweep(sparsities) -> list[dict]:
    """Vary K (via connection sparsity) with the recurrent weights scaled
    by 1/sqrt(K) — V&S's defining synaptic scaling. On the balanced
    manifold the AI signatures (CV -> 1, low pairwise correlation) are
    K-invariant, down to the finite-K breakdown where the mean-field
    (central-limit) argument stops holding.
    """
    N_E, base_K = 1024, 10.0
    out = []
    for s in sparsities:
        K = (1.0 - s) * N_E
        sc = float(np.sqrt(base_K / K))   # J ~ 1/sqrt(K)
        args = ["--input-rate", "1", "--w-in", "0.01", "0.001",
                "--w-ei", f"{0.6 * sc:.4f}", f"{0.18 * sc:.4f}",
                "--w-ie", f"{3.0 * sc:.4f}", f"{0.9 * sc:.4f}",
                "--w-ii", f"{0.4 * sc:.4f}", f"{0.12 * sc:.4f}",
                "--ei-sparsity", f"{s}", "--exact-k",
                "--independent-drive", "45", "0.38",
                "--independent-drive-i", "8", "0.25"]
        data = _run_scope(args)
        spk = data["spk_e"]
        dt = float(data["dt"])
        cv = float(np.median(_isi_cvs(spk, dt)))
        _, _, pk = _pair_cross_correlogram(spk, dt)
        out.append({"K": float(K), "sparsity": float(s), "weight_scale": sc,
                    "median_cv_e": cv, "xcorr_peak": float(pk)})
        print(f"  K={K:.0f} (sparsity {s}): medCV={cv:.2f}, xcorr={pk:.4f}")
    out.sort(key=lambda d: d["K"])
    return out


def n_invariance_sweep(n_values, sparsity: float = 0.99) -> list[dict]:
    """True V&S N-invariance test: sweep the network size N at fixed K/N
    (fixed sparsity, so K ∝ N), scaling BOTH the recurrent weights and the
    external drive as the balanced state requires.

    Recurrent weights ∝ 1/√K. The external drive is made V&S-sparse by
    scaling the per-cell Poisson stream so it equals the superposition of
    K_ext ∝ K independent external connections each at strength ∝ 1/√K:
    amplitude ∝ 1/√K and rate ∝ K, giving an external mean O(√K) with O(1)
    fluctuations — matching the recurrent drive. If the state lives on the
    balanced manifold the *rates* (not just the CV) are N-invariant.
    """
    N_ref = 1024
    K_ref = (1.0 - sparsity) * N_ref
    base_re, base_ge = 45.0, 0.38   # external E drive at N_ref
    base_ri, base_gi = 8.0, 0.25    # external I drive at N_ref
    out = []
    for N in n_values:
        K = (1.0 - sparsity) * N
        wsc = float(np.sqrt(K_ref / K))   # weights & ext amplitude ∝ 1/√K
        rsc = K / K_ref                   # ext rate ∝ K
        args = ["--n-hidden", str(int(N)),
                "--input-rate", "1", "--w-in", "0.01", "0.001",
                "--w-ei", f"{0.6 * wsc:.5f}", f"{0.18 * wsc:.5f}",
                "--w-ie", f"{3.0 * wsc:.5f}", f"{0.9 * wsc:.5f}",
                "--w-ii", f"{0.4 * wsc:.5f}", f"{0.12 * wsc:.5f}",
                "--ei-sparsity", f"{sparsity}", "--exact-k",
                "--independent-drive",
                f"{base_re * rsc:.4f}", f"{base_ge * wsc:.5f}",
                "--independent-drive-i",
                f"{base_ri * rsc:.4f}", f"{base_gi * wsc:.5f}"]
        data = _run_scope(args)
        spk_e, spk_i = data["spk_e"], data["spk_i"]
        dt = float(data["dt"])
        n_i = int(data["n_i"]) if "n_i" in data.files else 0
        r_e = float(spk_e.mean() * 1000.0 / dt)
        r_i = float(spk_i.mean() * 1000.0 / dt) if spk_i.size > 0 else 0.0
        cv = float(np.median(_isi_cvs(spk_e, dt)))
        out.append({"N": int(N), "n_i": n_i, "K": float(K),
                    "r_e_hz": r_e, "r_i_hz": r_i, "median_cv_e": cv})
        print(f"  N={N} (K={K:.1f}, N_I={n_i}): "
              f"r_E={r_e:.1f} r_I={r_i:.1f} Hz  medCV={cv:.2f}")
    return out


def plot_n_invariance(sweep: list[dict], out_path: Path, run_id: str) -> None:
    theme.apply()
    ns = [d["N"] for d in sweep]
    re = [d["r_e_hz"] for d in sweep]
    ri = [d["r_i_hz"] for d in sweep]
    cv = [d["median_cv_e"] for d in sweep]
    ks = [d["K"] for d in sweep]
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax.plot(ns, re, "o-", color=theme.INK_BLACK, lw=1.6, ms=7, label="$r_E$")
    ax.plot(ns, ri, "s--", color=theme.DEEP_RED, lw=1.6, ms=7, label="$r_I$")
    for x, y, k in zip(ns, re, ks):
        ax.annotate(f"K={k:.0f}", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center",
                    fontsize=theme.SIZE_ANNOTATION - 1, color=theme.GREY_DARK)
    ax.set_xlabel("network size $N_E$  ($K \\propto N$; weights & external "
                  "drive $\\propto 1/\\sqrt{K}$)")
    ax.set_ylabel("population rate (Hz)")
    ax.set_ylim(0, 38)
    ax.spines["top"].set_visible(False)
    ax2 = ax.twinx()
    ax2.plot(ns, cv, "^:", color=theme.AMBER, lw=1.4, ms=7,
             label="median ISI CV (E)")
    ax2.axhline(1.0, color=theme.AMBER, lw=0.7, ls=":", alpha=0.6)
    ax2.set_ylabel("median ISI CV (E)", color=theme.AMBER)
    ax2.set_ylim(0, 1.6)
    ax2.tick_params(axis="y", colors=theme.AMBER)
    lines = ax.get_lines()[:2] + ax2.get_lines()[:1]
    ax.legend(lines, [ln.get_label() for ln in lines],
              fontsize=theme.SIZE_LEGEND - 1, frameon=False, loc="center right")
    ax.set_title("N-sweep on the balanced manifold: CV and $r_I$ invariant, "
                 "$r_E$ drifts", fontsize=theme.SIZE_TITLE, loc="left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sqrtk(sweep: list[dict], out_path: Path, run_id: str) -> None:
    theme.apply()
    ks = [d["K"] for d in sweep]
    cvs = [d["median_cv_e"] for d in sweep]
    pks = [d["xcorr_peak"] for d in sweep]
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax.plot(ks, cvs, "o-", color=theme.INK_BLACK, lw=1.4, zorder=5,
            label="median ISI CV (E)")
    ax.axhline(1.0, color=theme.AMBER, lw=0.8, ls=":")
    ax.text(ks[0], 1.0, "CV = 1 (Poisson) ", color=theme.AMBER,
            va="bottom", ha="left", fontsize=theme.SIZE_LABEL - 1)
    ax.set_xlabel("K  (presynaptic inputs per cell, weights $\\propto 1/\\sqrt{K}$)")
    ax.set_ylabel("median ISI CV (E)")
    ax.set_ylim(0.0, 1.4)
    ax.spines["top"].set_visible(False)
    ax2 = ax.twinx()
    ax2.plot(ks, pks, "s--", color=theme.DEEP_RED, lw=1.0, alpha=0.8,
             label="peak pairwise $|C(\\tau)|$")
    ax2.set_ylabel("peak pairwise $|C(\\tau)|$", color=theme.DEEP_RED)
    ax2.set_ylim(0, max(pks) * 1.6)
    ax2.tick_params(axis="y", colors=theme.DEEP_RED)
    lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(lines, [ln.get_label() for ln in lines],
              fontsize=theme.SIZE_LEGEND, frameon=False, loc="lower right")
    ax.set_title("$\\sqrt{K}$ scaling: irregularity holds as K grows",
                 fontsize=theme.SIZE_TITLE, loc="left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _balance_currents(data) -> dict:
    """Median per-cell, time-mean membrane currents decomposed by source,
    for both populations. Computed per cell per timestep then time-averaged
    (so the g–V correlation is kept, unlike ⟨g⟩·(E−⟨V⟩)), then median over
    cells. Returns excitatory / inhibitory / leak currents onto E and I,
    plus the mean membrane potentials.
    """
    E_e, E_i, E_L, g_L = 0.0, -80.0, -65.0, 0.05
    spk_e = data["spk_e"]
    ge_e, gi_e, v_e = data["ge_e_1"], data["gi_e_1"], data["v_e_1"]
    b = max(0, ge_e.shape[0] - spk_e.shape[0])
    ge_e, gi_e, v_e = ge_e[b:], gi_e[b:], v_e[b:]
    out = {
        "I_exc_E": float(np.median((ge_e * (E_e - v_e)).mean(axis=0))),
        "I_inh_E": float(np.median((gi_e * (E_i - v_e)).mean(axis=0))),
        "I_leak_E": float(np.median((g_L * (E_L - v_e)).mean(axis=0))),
        "v_e_mean": float(np.median(v_e)),
    }
    if "gi_i_1" in data.files and data["gi_i_1"].size:
        ge_i = data["ge_i_1"][b:]
        gi_i = data["gi_i_1"][b:]
        v_i = data["v_i_1"][b:]
        out.update({
            "I_exc_I": float(np.median((ge_i * (E_e - v_i)).mean(axis=0))),
            "I_inh_I": float(np.median((gi_i * (E_i - v_i)).mean(axis=0))),
            "I_leak_I": float(np.median((g_L * (E_L - v_i)).mean(axis=0))),
            "v_i_mean": float(np.median(v_i)),
        })
    return out


def drive_sweep(ai_args: list[str], scales) -> list[dict]:
    """Sweep the external drive level and record rates + decomposed currents.

    V&S's signature prediction: in the balanced state the rates are a
    *linear* function of the external input, even though the single units
    are strongly nonlinear. We scale both external Poisson rates (E: 45·x,
    I: 8·x Hz) by a common factor x and measure r_E, r_I — and the mean
    membrane currents, so the balance equations can be solved and the
    predicted rates overlaid (see ``balance_predict``).
    """
    static = _strip_flag(_strip_flag(ai_args, "--independent-drive", 2),
                         "--independent-drive-i", 2)
    base_e, base_i = 45.0, 8.0
    out = []
    for x in scales:
        re_ext, ri_ext = base_e * x, base_i * x
        args = [*static,
                "--independent-drive", f"{re_ext:.3f}", "0.38",
                "--independent-drive-i", f"{ri_ext:.3f}", "0.25"]
        data = _run_scope(args)
        spk_e, spk_i = data["spk_e"], data["spk_i"]
        dt = float(data["dt"])
        r_e = float(spk_e.mean() * 1000.0 / dt)
        r_i = float(spk_i.mean() * 1000.0 / dt) if spk_i.size > 0 else 0.0
        out.append({"drive_scale": float(x), "ext_rate_e_hz": re_ext,
                    "ext_rate_i_hz": ri_ext, "r_e_hz": r_e, "r_i_hz": r_i,
                    **_balance_currents(data)})
        print(f"  drive x={x:.2f} (ext_E={re_ext:.0f} Hz): "
              f"r_E={r_e:.1f}, r_I={r_i:.1f} Hz")
    return out


def balance_predict(sweep: list[dict]) -> dict:
    """Solve the leading-order balance equations for the predicted (r_E, r_I).

    The balanced state sets the rates by the requirement that the large
    mean currents cancel to an O(1) residue. From the drive sweep we read
    the realized synaptic *current* gains (slopes of mean current vs the
    driving rate) — the network's effective coefficients, since the flag
    weights do not survive the sparsity / exact-K / 1/√K machinery — and
    use the known external-drive gains, then solve:

      E-balance:  I_exc^E(r_ext) + I_inh^E(r_I) + I_leak^E ≈ 0
      I-balance:  I_exc^I(r_E, r_ext) + I_inh^I(r_I) + I_leak^I ≈ 0

    Returns the predicted r_E, r_I across the sweep and the fitted gains.
    """
    E_e = 0.0
    tau_a, gd_i = 0.002, 0.25
    x_extE = np.array([s["ext_rate_e_hz"] for s in sweep])
    x_extI = np.array([s["ext_rate_i_hz"] for s in sweep])
    r_e = np.array([s["r_e_hz"] for s in sweep])
    r_i = np.array([s["r_i_hz"] for s in sweep])
    I_exc_E = np.array([s["I_exc_E"] for s in sweep])
    I_inh_E = np.array([s["I_inh_E"] for s in sweep])
    I_leak_E = np.array([s["I_leak_E"] for s in sweep])
    # E-balance: inhibition (∝ r_I) must cancel external excitation (∝ r_ext).
    a_E = float(np.polyfit(x_extE, I_exc_E, 1)[0])   # exc current per Hz ext
    b_E = float(np.polyfit(r_i, I_inh_E, 1)[0])      # inh current per Hz r_I (<0)
    leak_E = float(I_leak_E.mean())
    r_i_pred = -(a_E * x_extE + leak_E) / b_E
    pred = {"a_E": a_E, "b_E": b_E, "r_i_pred": r_i_pred.tolist()}
    if all("I_exc_I" in s for s in sweep):
        I_exc_I = np.array([s["I_exc_I"] for s in sweep])
        I_inh_I = np.array([s["I_inh_I"] for s in sweep])
        I_leak_I = np.array([s["I_leak_I"] for s in sweep])
        v_i = np.array([s["v_i_mean"] for s in sweep])
        # subtract the known external-to-I excitatory current, leaving E→I
        ext_I_cur = gd_i * tau_a * x_extI * (E_e - v_i)
        c_I = float(np.polyfit(r_e, I_exc_I - ext_I_cur, 1)[0])  # E→I per Hz r_E
        leak_I = float(I_leak_I.mean())
        r_e_pred = -(ext_I_cur + I_inh_I + leak_I) / c_I
        pred.update({"c_I": c_I, "r_e_pred": r_e_pred.tolist()})
    return pred


def _linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m, c = np.polyfit(x, y, 1)
    yhat = m * x + c
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(m), float(c), r2


def plot_linear_response(sweep: list[dict], pred: dict,
                         out_path: Path, run_id: str) -> dict:
    theme.apply()
    xe = [d["ext_rate_e_hz"] for d in sweep]
    re = [d["r_e_hz"] for d in sweep]
    ri = [d["r_i_hz"] for d in sweep]
    me, ce, r2e = _linfit(xe, re)
    mi, ci, r2i = _linfit(xe, ri)
    # balance-predicted rates and their slopes
    ri_pred = pred.get("r_i_pred")
    re_pred = pred.get("r_e_pred")
    mi_p = _linfit(xe, ri_pred)[0] if ri_pred else None
    me_p = _linfit(xe, re_pred)[0] if re_pred else None
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    # measured data points
    ax.scatter(xe, re, s=34, color=theme.INK_BLACK, zorder=5,
               label=f"$r_E$ measured  (slope {me:.3f})")
    ax.scatter(xe, ri, s=34, color=theme.DEEP_RED, zorder=5, marker="s",
               label=f"$r_I$ measured  (slope {mi:.3f})")
    # balance-equation predictions (solid lines)
    if re_pred is not None:
        ax.plot(xe, re_pred, color=theme.INK_BLACK, lw=1.4, alpha=0.85,
                label=f"$r_E$ balance prediction  (slope {me_p:.3f})")
    if ri_pred is not None:
        ax.plot(xe, ri_pred, color=theme.DEEP_RED, lw=1.4, alpha=0.85,
                label=f"$r_I$ balance prediction  (slope {mi_p:.3f})")
    ax.set_xlabel("external drive to E  (Hz, per-cell Poisson)")
    ax.set_ylabel("population rate (Hz)")
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=theme.SIZE_LEGEND - 1, frameon=False, loc="upper left")
    ax.set_title("Balanced state: linear rates; balance equation pins $r_I$, "
                 "bounds $r_E$", fontsize=theme.SIZE_TITLE, loc="left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"r_e_slope": me, "r_e_r2": r2e, "r_i_slope": mi, "r_i_r2": r2i,
            "r_e_slope_pred": me_p, "r_i_slope_pred": mi_p,
            "r_e_slope_ratio": (me_p / me) if me_p else None,
            "r_i_slope_ratio": (mi_p / mi) if mi_p else None}


def plot_lyapunov(lyap_by_cell: dict, out_path: Path, run_id: str) -> None:
    """Compare spike-train divergence D(t) across cells on one axis.

    D(t) = number of E cells whose spike differs between the clean run and
    an ε-perturbed rerun on identical frozen input. PING re-locks to 0
    (stable limit cycle, λ ≤ 0); the V&S AI cell sustains a small
    slowly-growing divergence (weak intrinsic chaos riding on top of the
    strong input entrainment that the shared frozen drive imposes).
    """
    theme.apply()
    colours = {"ping": theme.MUTED, "ai": theme.DEEP_RED,
               "ai_quenched": theme.AMBER}
    labels = {"ping": "PING", "ai": "V&S AI (Poisson drive)",
              "ai_quenched": "V&S AI (quenched DC)"}
    order = {"ping": 0, "ai": 1, "ai_quenched": 2}
    lyap_by_cell = dict(sorted(lyap_by_cell.items(),
                               key=lambda kv: order.get(kv[0], 9)))
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
