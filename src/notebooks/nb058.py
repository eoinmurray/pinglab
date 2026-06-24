"""Notebook runner for entry 058 — the canonical Vreeswijk-Sompolinsky state.

Pushes the COBANet into the full four-coupling balanced asynchronous-irregular
regime — sparse fixed fan-in (K ≈ 10), per-cell independent drive on E and I,
and all four recurrent matrices (W^EE, W^EI, W^IE, W^II). Two figures:

  Figure 1 (raster page): combined E+I raster, Welch PSD, ISI-CV histogram,
    pairwise cross-correlogram — the hallmark irregular (CV ≈ 1),
    asynchronous, rhythmless firing.
  Figure 2 (N-sweep): hold K fixed, grow N 4×, for the four-coupling network
    and a reduced one with W^EE = 0 — showing the signatures and rates are
    N-invariant and that the fixed-K balance, not W^EE, pins the rates.
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

SLUG = "nb058"
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
# Canonical Brunel/Vreeswijk asynchronous-irregular state, full four-coupling
# architecture (W^EE, W^EI, W^IE, W^II all present). The knobs that land it on
# textbook CV ≈ 1 for both populations:
#   - --ei-sparsity 0.99 + --exact-k → fixed fan-in K ≈ 10 per post cell
#     (Brunel/V&S convention; removes the binomial fan-in variance that
#     otherwise broadens the rate distribution)
#   - --independent-drive 45 0.38 — large per-spike kicks at low rate on E
#     (input fluctuations dominate drift)
#   - --independent-drive-i 8 0.25 — same for I (without it the I-cell CV stays
#     correlated with E via W^EI)
#   - --w-ie 3.0 strong I→E shunt so Poisson I activity propagates noise into E
#   - --w-ii 0.4 modest I→I self-inhibition
#   - --w-ee 0.4 recurrent E→E completing the four-coupling V&S architecture.
#     It does NOT pin the rates (the fixed-K balance equations already do — see
#     the N-sweep) — it is here for canonical completeness and lifts CV
#     slightly above 1.
CANONICAL_ARGS = [
    "--input-rate", "1",
    "--w-in", "0.01", "0.001",
    "--w-ei", "0.6", "0.18",
    "--w-ie", "3.0", "0.9",
    "--w-ii", "0.4", "0.12",
    "--w-ee", "0.4", "0.12",
    "--ei-sparsity", "0.99",
    "--exact-k",
    "--independent-drive", "45", "0.38",
    "--independent-drive-i", "8", "0.25",
]
CELLS: dict[str, dict] = {
    "ai": {
        "args": CANONICAL_ARGS,
        "title": "Canonical V&S — four-coupling, fixed-K, per-E + per-I drive",
    },
}

# N-invariance sweep: hold fan-in K ≈ 10 fixed (sparsity s = 1 - K/N_E) and
# grow the network 4× — N_E = 1024 → 4096, N_I = N_E / 4 — comparing the
# canonical four-coupling network (W^EE = 0.4) against the reduced no-E→E one
# (W^EE = 0). The balance equations depend only on K, so at fixed K the rates
# and signatures should be N-invariant for both — demonstrating that recurrent
# excitation is not what pins r_E. Multiple seeds separate finite-size noise
# from genuine drift.
K_TARGET = 10
N_GRID: list[tuple[int, int]] = [(1024, 256), (2048, 512), (4096, 1024)]
SWEEP_SEEDS = [0, 1, 2]
SWEEP_VARIANTS = {
    "canonical": {"w_ee": ("0.4", "0.12"), "label": "canonical ($W^{EE}$ = 0.4)"},
    "reduced": {"w_ee": ("0.0", "0.0"), "label": "reduced (no E→E)"},
}

# K-sweep: the defining V&S test. At fixed N_E = 1024 grow the fan-in K via
# sparsity (s = 1 - K/N_E) under two coupling conventions:
#   - strong (V&S):  synapse ∝ J/√K → scale recurrent weights by √(K/K0), and
#     fold the drive into the balance (external mean ∝ √K, fluctuation O(1)):
#     rate ∝ K, per-spike g ∝ 1/√K. Fluctuations survive K→∞, CV stays ≈ 1.
#   - weak (mean-field): hold everything fixed → synapse ∝ 1/K, fluctuations
#     ∝ 1/√K → 0, the state regularises (CV drifts down toward a clock).
# The exact-K path already divides each synapse by K, so "scale weights by √K"
# realises J/√K with no model change — pure runner arithmetic.
K_SWEEP_K0 = 10
K_SWEEP_KS = [10, 20, 40, 80, 160]
K_SWEEP_T_MS = 3000  # longer trial so low-rate high-K points get enough ISIs
K_SWEEP_BASE_W = {  # nominal (mean, std) at K0
    "w-ei": (0.6, 0.18), "w-ie": (3.0, 0.9),
    "w-ii": (0.4, 0.12), "w-ee": (0.4, 0.12),
}
K_SWEEP_BASE_DRIVE = {"e": (45.0, 0.38), "i": (8.0, 0.25)}  # (rate_hz, g)
K_SWEEP_VARIANTS = {
    "strong": {"label": "strong $J/\\sqrt{K}$ (V&S)", "scale": True},
    "weak": {"label": "weak (mean-field)", "scale": False},
}

# Self-generation test: drive both networks with a *quenched* DC conductance
# (a frozen per-cell offset — no per-timestep fluctuation, so the input carries
# zero noise) and ask whether irregular firing survives. The balanced recurrent
# network keeps CV ≈ 1 (the irregularity is generated by its own deterministic
# dynamics); a decoupled network (recurrent weights ≈ 0) under the same frozen
# DC fires like a clock (CV ≈ 0) — the negative control. A deterministic
# autonomous system that sustains aperiodic firing is chaotic by elimination.
CHAOS_SEEDS = [0, 1, 2]
CHAOS_QUENCH = ["--quenched-drive", "0.0342", "0.01",
                "--quenched-drive-i", "0.004", "0.005"]
CHAOS_VARIANTS = {
    "balanced": {
        "label": "balanced (full recurrence)",
        "w": ["--w-ei", "0.6", "0.18", "--w-ie", "3.0", "0.9",
              "--w-ii", "0.4", "0.12", "--w-ee", "0.4", "0.12"],
        "quench": CHAOS_QUENCH,
    },
    "decoupled": {
        "label": "decoupled (recurrence ≈ 0)",
        "w": ["--w-ei", "0.001", "0.0", "--w-ie", "0.001", "0.0",
              "--w-ii", "0.001", "0.0", "--w-ee", "0.001", "0.0"],
        # lower DC so the uncoupled clocks fire at a comparable rate
        "quench": ["--quenched-drive", "0.003", "0.01",
                   "--quenched-drive-i", "0.004", "0.005"],
    },
}

# Direct Lyapunov measurement: clone the network, kick every membrane voltage
# by ε at t=0 on identical frozen (quenched-DC) input, and track the voltage
# distance ‖ΔV(t)‖ between the two copies. Spiking dynamics contract between
# spikes and expand only at spike-flips, so ε must be large enough to flip
# spikes (ε too small → pure contraction → false λ < 0); ‖ΔV‖ then grows and
# saturates at the attractor size. The slope of log‖ΔV‖ in the pre-saturation
# window is the largest Lyapunov exponent: > 0 for the chaotic balanced net,
# < 0 for the decoupled control. (Precise asymptotic λ would need a
# renormalised Benettin scheme; this is the initial-growth estimate.)
LYAP_SEEDS = [0, 1, 2, 3, 4]
LYAP_EPS = "0.1"
LYAP_T_MS = "500"
LYAP_FIT_MS = (5.0, 80.0)   # balanced growth window (post-flip, pre-saturation)

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

    fig = plt.figure(figsize=(9.0, 9.0), dpi=150)
    gs = GridSpec(
        4, 1, figure=fig,
        height_ratios=[5.2, 2.4, 1.6, 1.8],
        hspace=0.55, top=0.94, bottom=0.06, left=0.12, right=0.97,
    )
    ax_r = fig.add_subplot(gs[0])
    ax_psd = fig.add_subplot(gs[1])
    ax_cv = fig.add_subplot(gs[2])
    ax_xcorr = fig.add_subplot(gs[3])

    # Combined raster: E in the lower band (0 … N_E), I stacked above
    # (N_E … N_E + N_I), sharing one time axis. E black, I red.
    N_E = spk_e.shape[1]
    N_I = spk_i.shape[1] if has_i else 0
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
        ax_r.text(0.995, 0.985, "I", transform=ax_r.transAxes, ha="right",
                  va="top", fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED,
                  fontweight="semibold")
        ax_r.text(0.995, 0.02, f"E ({N_E})", transform=ax_r.transAxes,
                  ha="right", va="bottom", fontsize=theme.SIZE_LABEL,
                  color=theme.INK_BLACK, fontweight="semibold")
    ax_r.set_ylim(0, N_E + N_I)
    ax_r.set_xlim(0, T * dt)
    ax_r.set_xlabel("time (ms)")
    ax_r.set_ylabel("neuron  (E below · I above)")
    ax_r.set_title(f"{title}  ·  ⟨r_E⟩ = {e_rate:.1f}, ⟨r_I⟩ = {i_rate:.1f} Hz",
                   loc="left")
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)

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
            f"  max {f_peak:.1f} Hz (noise floor)",
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
            0.02, 0.95,
            f"median CV = {np.median(cvs):.2f}  (n = {cvs.size} cells)",
            transform=ax_cv.transAxes, ha="left", va="top",
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


def _sweep_args(n_e: int, n_i: int, sparsity: float,
                w_ee: tuple[str, str], seed: int) -> list[str]:
    """Build the sim args for one N-sweep point — the canonical config with
    N, sparsity (so K stays fixed), W^EE, and seed overridden."""
    return [
        "--n-e", str(n_e), "--n-i", str(n_i),
        "--ei-sparsity", f"{sparsity:.6f}", "--exact-k",
        "--input-rate", "1", "--w-in", "0.01", "0.001",
        "--w-ei", "0.6", "0.18", "--w-ie", "3.0", "0.9", "--w-ii", "0.4", "0.12",
        "--w-ee", w_ee[0], w_ee[1],
        "--independent-drive", "45", "0.38", "--independent-drive-i", "8", "0.25",
        "--seed", str(seed),
    ]


def _run_and_measure(scope_argv: list[str]) -> dict:
    """Run one sim and read back the four scalar diagnostics."""
    for p in (SCOPE_OUT_PNG, SCOPE_OUT_NPZ):
        if p.exists():
            p.unlink()
    cmd = ["uv", "run", "python", str(OSCILLOSCOPE), *COMMON_ARGS, *scope_argv]
    subprocess.run(cmd, cwd=REPO, check=True)
    if not SCOPE_OUT_NPZ.exists():
        raise SystemExit(f"oscilloscope did not produce {SCOPE_OUT_NPZ}")
    data = np.load(SCOPE_OUT_NPZ)
    spk_e, spk_i = data["spk_e"], data["spk_i"]
    dt = float(data["dt"])
    e_rate = float(spk_e.mean() * 1000.0 / dt)
    i_rate = float(spk_i.mean() * 1000.0 / dt) if spk_i.size > 0 else 0.0
    cvs_e = _isi_cvs(spk_e, dt)
    med_cv_e = float(np.median(cvs_e)) if cvs_e.size > 0 else float("nan")
    _, _, xpeak = _pair_cross_correlogram(spk_e, dt)
    return {"e_rate": e_rate, "i_rate": i_rate,
            "median_isi_cv_e": med_cv_e, "peak_abs_xcorr_e": xpeak}


def run_n_sweep() -> dict:
    """Sweep N at fixed K for both architecture variants, averaging over
    seeds. Returns a nested dict variant → metric → (means, stds) over N."""
    n_es = [ne for ne, _ in N_GRID]
    out: dict = {"n_e": n_es, "variants": {}}
    for vkey, vspec in SWEEP_VARIANTS.items():
        per_metric = {m: {"mean": [], "std": []}
                      for m in ("e_rate", "i_rate", "median_isi_cv_e",
                                "peak_abs_xcorr_e")}
        for (n_e, n_i) in N_GRID:
            sparsity = 1.0 - K_TARGET / n_e
            seed_vals = {m: [] for m in per_metric}
            for seed in SWEEP_SEEDS:
                argv = _sweep_args(n_e, n_i, sparsity, vspec["w_ee"], seed)
                print(f"[sweep] {vkey} N_E={n_e} s={sparsity:.4f} seed={seed}")
                res = _run_and_measure(argv)
                for m in per_metric:
                    seed_vals[m].append(res[m])
            for m in per_metric:
                arr = np.array(seed_vals[m], dtype=float)
                per_metric[m]["mean"].append(float(np.nanmean(arr)))
                per_metric[m]["std"].append(float(np.nanstd(arr)))
        out["variants"][vkey] = {"label": vspec["label"], "metrics": per_metric}
    return out


def plot_n_sweep(sweep: dict, out_path: Path) -> None:
    """Three panels vs network size N_E (log2 x): median ISI CV, pairwise
    cross-correlation peak, and the two population rates — canonical
    (four-coupling) vs reduced (no E→E)."""
    theme.apply()
    n_es = np.array(sweep["n_e"], dtype=float)
    fig, (ax_cv, ax_x, ax_r) = plt.subplots(
        1, 3, figsize=(13.5, 4.5), dpi=150)
    styles = {
        "canonical": dict(color=theme.INK_BLACK, ls="-", marker="o"),
        "reduced": dict(color=theme.GREY_MID, ls="--", marker="s"),
    }
    for vkey, vdata in sweep["variants"].items():
        st = styles[vkey]
        lab = vdata["label"]
        m = vdata["metrics"]
        ax_cv.errorbar(n_es, m["median_isi_cv_e"]["mean"],
                       yerr=m["median_isi_cv_e"]["std"], label=lab,
                       capsize=3, lw=1.4, **st)
        ax_x.errorbar(n_es, m["peak_abs_xcorr_e"]["mean"],
                      yerr=m["peak_abs_xcorr_e"]["std"], label=lab,
                      capsize=3, lw=1.4, **st)
        ax_r.errorbar(n_es, m["e_rate"]["mean"], yerr=m["e_rate"]["std"],
                      label=f"{lab} · r_E", capsize=3, lw=1.4, **st)
        ax_r.errorbar(n_es, m["i_rate"]["mean"], yerr=m["i_rate"]["std"],
                      label=f"{lab} · r_I", capsize=3, lw=1.0,
                      markerfacecolor="none",
                      color=st["color"], ls=st["ls"], marker=st["marker"])

    for ax in (ax_cv, ax_x, ax_r):
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_es)
        ax.set_xticklabels([f"{int(n)}" for n in n_es])
        ax.set_xlabel("$N_E$  (fixed $K \\approx 10$)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_cv.axhline(1.0, color=theme.DEEP_RED, lw=0.9, ls=":", alpha=0.85)
    ax_cv.text(n_es[0], 1.0, " Poisson", color=theme.DEEP_RED, va="bottom",
               ha="left", fontsize=theme.SIZE_LABEL - 1)
    ax_cv.set_ylabel("median ISI CV (E)")
    ax_cv.set_title("Irregularity", loc="left", fontsize=theme.SIZE_LABEL)
    ax_cv.legend(fontsize=theme.SIZE_LABEL - 2, frameon=False, loc="best")

    ax_x.axhline(0.0, color=theme.GREY_MID, lw=0.6, alpha=0.7)
    ax_x.set_ylabel("pairwise C(τ) peak (E)")
    ax_x.set_title("Asynchrony", loc="left", fontsize=theme.SIZE_LABEL)
    ax_x.set_ylim(bottom=0.0)

    ax_r.set_ylabel("rate (Hz)")
    ax_r.set_title("Rates  (filled $r_E$, open $r_I$)",
                   loc="left", fontsize=theme.SIZE_LABEL)
    ax_r.legend(fontsize=theme.SIZE_LABEL - 3, frameon=False, loc="best")

    fig.suptitle(
        "N-invariance at fixed fan-in K — signatures hold as the network "
        "grows 4×; W$^{EE}$ does not pin the rates",
        x=0.01, ha="left", fontsize=theme.SIZE_LABEL + 1)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _ksweep_args(k: int, scale_coupling: bool, seed: int) -> list[str]:
    """Build sim args for one K-sweep point at fixed N_E = 1024. Strong
    (V&S) scales recurrent weights ∝ √(K/K0) and the drive into the balance
    (rate ∝ K, g ∝ 1/√K); weak holds everything fixed."""
    s = 1.0 - k / 1024.0
    fac = (k / K_SWEEP_K0) ** 0.5 if scale_coupling else 1.0   # √(K/K0)
    fk = fac * fac if scale_coupling else 1.0                  # K/K0
    args = [
        # Longer trial than the default 1000 ms: under strong coupling the
        # rate drifts down with K (finite-K balance correction), so high-K
        # cells need a longer window to accumulate enough ISIs for an
        # unbiased CV (at K=160, 5 Hz → ~5 spikes/1000 ms biases CV low).
        "--t-ms", str(K_SWEEP_T_MS),
        "--ei-sparsity", f"{s:.6f}", "--exact-k",
        "--input-rate", "1", "--w-in", "0.01", "0.001",
        "--seed", str(seed),
    ]
    for flag, (m, sd) in K_SWEEP_BASE_W.items():
        args += [f"--{flag}", f"{m * fac:.5f}", f"{sd * fac:.5f}"]
    re_, ge = K_SWEEP_BASE_DRIVE["e"]
    ri, gi = K_SWEEP_BASE_DRIVE["i"]
    args += ["--independent-drive", f"{re_ * fk:.4f}", f"{ge / fac:.5f}"]
    args += ["--independent-drive-i", f"{ri * fk:.4f}", f"{gi / fac:.5f}"]
    return args


def run_k_sweep() -> dict:
    """Sweep K at fixed N for the strong (J/√K) and weak conventions,
    averaging over seeds. Returns variant → metric → (means, stds) over K."""
    out: dict = {"k": K_SWEEP_KS, "variants": {}}
    for vkey, vspec in K_SWEEP_VARIANTS.items():
        per_metric = {m: {"mean": [], "std": []}
                      for m in ("e_rate", "i_rate", "median_isi_cv_e",
                                "peak_abs_xcorr_e")}
        for k in K_SWEEP_KS:
            seed_vals = {m: [] for m in per_metric}
            for seed in SWEEP_SEEDS:
                argv = _ksweep_args(k, vspec["scale"], seed)
                print(f"[ksweep] {vkey} K={k} seed={seed}")
                res = _run_and_measure(argv)
                for m in per_metric:
                    seed_vals[m].append(res[m])
            for m in per_metric:
                arr = np.array(seed_vals[m], dtype=float)
                per_metric[m]["mean"].append(float(np.nanmean(arr)))
                per_metric[m]["std"].append(float(np.nanstd(arr)))
        out["variants"][vkey] = {"label": vspec["label"], "metrics": per_metric}
    return out


def plot_k_sweep(sweep: dict, out_path: Path) -> None:
    """Three panels vs fan-in K (log2 x): median ISI CV, pairwise
    cross-correlation peak, and the population rates — strong J/√K (V&S)
    vs weak mean-field coupling."""
    theme.apply()
    ks = np.array(sweep["k"], dtype=float)
    fig, (ax_cv, ax_x, ax_r) = plt.subplots(
        1, 3, figsize=(13.5, 4.5), dpi=150)
    styles = {
        "strong": dict(color=theme.INK_BLACK, ls="-", marker="o"),
        "weak": dict(color=theme.GREY_MID, ls="--", marker="s"),
    }
    for vkey, vdata in sweep["variants"].items():
        st = styles[vkey]
        lab = vdata["label"]
        m = vdata["metrics"]
        ax_cv.errorbar(ks, m["median_isi_cv_e"]["mean"],
                       yerr=m["median_isi_cv_e"]["std"], label=lab,
                       capsize=3, lw=1.4, **st)
        ax_x.errorbar(ks, m["peak_abs_xcorr_e"]["mean"],
                      yerr=m["peak_abs_xcorr_e"]["std"], label=lab,
                      capsize=3, lw=1.4, **st)
        ax_r.errorbar(ks, m["e_rate"]["mean"], yerr=m["e_rate"]["std"],
                      label=f"{lab} · r_E", capsize=3, lw=1.4, **st)
        ax_r.errorbar(ks, m["i_rate"]["mean"], yerr=m["i_rate"]["std"],
                      label=f"{lab} · r_I", capsize=3, lw=1.0,
                      markerfacecolor="none",
                      color=st["color"], ls=st["ls"], marker=st["marker"])

    for ax in (ax_cv, ax_x, ax_r):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([f"{int(n)}" for n in ks])
        ax.set_xlabel("fan-in $K$  (fixed $N_E = 1024$)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_cv.axhline(1.0, color=theme.DEEP_RED, lw=0.9, ls=":", alpha=0.85)
    ax_cv.text(ks[0], 1.0, " Poisson", color=theme.DEEP_RED, va="bottom",
               ha="left", fontsize=theme.SIZE_LABEL - 1)
    ax_cv.set_ylabel("median ISI CV (E)")
    ax_cv.set_title("Irregularity vs K", loc="left", fontsize=theme.SIZE_LABEL)
    ax_cv.legend(fontsize=theme.SIZE_LABEL - 2, frameon=False, loc="best")

    ax_x.axhline(0.0, color=theme.GREY_MID, lw=0.6, alpha=0.7)
    ax_x.set_ylabel("pairwise C(τ) peak (E)")
    ax_x.set_title("Asynchrony vs K", loc="left", fontsize=theme.SIZE_LABEL)
    ax_x.set_ylim(bottom=0.0)

    ax_r.set_ylabel("rate (Hz)")
    ax_r.set_title("Rates  (filled $r_E$, open $r_I$)",
                   loc="left", fontsize=theme.SIZE_LABEL)
    ax_r.legend(fontsize=theme.SIZE_LABEL - 3, frameon=False, loc="best")

    fig.suptitle(
        "The defining V&S limit — $J/\\sqrt{K}$ coupling sustains supra-Poisson "
        "irregularity as K grows; weak coupling decays toward the Poisson floor",
        x=0.01, ha="left", fontsize=theme.SIZE_LABEL + 1)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _chaos_args(variant: dict, seed: int) -> list[str]:
    """Quenched-DC config for one chaos-test variant: fixed fan-in, no Poisson
    drive (frozen per-cell DC instead), recurrent weights per variant."""
    return [
        "--ei-sparsity", "0.99", "--exact-k",
        "--input-rate", "1", "--w-in", "0.01", "0.001",
        *variant["w"], *variant["quench"],
        "--seed", str(seed),
    ]


def run_chaos() -> dict:
    """Drive balanced and decoupled networks with quenched DC and measure the
    per-cell ISI CV over seeds. Keeps the seed-0 spike raster of each for the
    figure. Returns {variant: {label, cv_mean, cv_std, rate, raster, dt}}."""
    out: dict = {}
    for vkey, vspec in CHAOS_VARIANTS.items():
        cv_meds, rates = [], []
        raster = None
        dt = None
        for seed in CHAOS_SEEDS:
            for p in (SCOPE_OUT_PNG, SCOPE_OUT_NPZ):
                if p.exists():
                    p.unlink()
            argv = [*COMMON_ARGS, *_chaos_args(vspec, seed)]
            print(f"[chaos] {vkey} seed={seed}")
            subprocess.run(["uv", "run", "python", str(OSCILLOSCOPE), *argv],
                           cwd=REPO, check=True)
            data = np.load(SCOPE_OUT_NPZ)
            spk_e = data["spk_e"]
            dt = float(data["dt"])
            cvs = _isi_cvs(spk_e, dt)
            cv_meds.append(float(np.median(cvs)) if cvs.size else float("nan"))
            rates.append(float(spk_e.mean() * 1000.0 / dt))
            if seed == CHAOS_SEEDS[0]:
                raster = spk_e
                raster_cvs = cvs
        out[vkey] = {
            "label": vspec["label"],
            "cv_mean": float(np.nanmean(cv_meds)),
            "cv_std": float(np.nanstd(cv_meds)),
            "rate": float(np.mean(rates)),
            "raster": raster, "raster_cvs": raster_cvs, "dt": dt,
        }
    return out


def _example_cells(spk: np.ndarray, dt: float, n: int = 6,
                   win_ms: float = 300.0) -> list[np.ndarray]:
    """Pick n example E cells (those firing closest to the median rate) and
    return their spike times (ms) within the window."""
    T = spk.shape[0]
    keep = int(win_ms / dt)
    counts = spk[:keep].sum(axis=0)
    active = np.where(counts >= 4)[0]
    if active.size == 0:
        return []
    med = np.median(counts[active])
    order = active[np.argsort(np.abs(counts[active] - med))]
    chosen = order[:n]
    t_ms = np.arange(keep) * dt
    return [t_ms[np.where(spk[:keep, c] > 0)[0]] for c in chosen]


def plot_chaos(chaos: dict, out_path: Path) -> None:
    """Example single-cell spike trains (regular vs irregular spacing) plus the
    per-cell ISI CV histograms — the self-generated-irregularity result."""
    theme.apply()
    fig, (ax_ex, ax_cv) = plt.subplots(1, 2, figsize=(13.5, 4.5), dpi=150)
    win_ms = 300.0

    # Example spike trains: balanced cells on top (black), decoupled below
    # (grey), each row one cell over the same window.
    row = 0
    yticks, ylabels = [], []
    for vkey, color in (("balanced", theme.INK_BLACK),
                        ("decoupled", theme.GREY_MID)):
        c = chaos[vkey]
        trains = _example_cells(c["raster"], c["dt"], n=6, win_ms=win_ms)
        for tr in trains:
            ax_ex.scatter(tr, np.full_like(tr, row), s=8, c=color, marker="|",
                          linewidths=1.0)
            row += 1
        yticks.append(row - len(trains) / 2.0)
        ylabels.append(c["label"].split(" (")[0])
        row += 1  # gap between groups
    ax_ex.axhline(6.5, color=theme.GREY_MID, lw=0.6, alpha=0.6)
    ax_ex.set_xlim(0, win_ms)
    ax_ex.set_ylim(-0.5, row - 0.5)
    ax_ex.set_yticks(yticks)
    ax_ex.set_yticklabels(ylabels, rotation=90, va="center")
    ax_ex.set_xlabel("time (ms)")
    ax_ex.set_title("Example E-cell spike trains (frozen DC) — even vs irregular",
                    loc="left", fontsize=theme.SIZE_LABEL)
    ax_ex.spines["top"].set_visible(False)
    ax_ex.spines["right"].set_visible(False)

    for vkey, color in (("balanced", theme.INK_BLACK),
                        ("decoupled", theme.GREY_MID)):
        cvs = chaos[vkey]["raster_cvs"]
        if cvs.size:
            ax_cv.hist(cvs, bins=np.linspace(0, 3.0, 46), color=color,
                       alpha=0.6,
                       label=f"{chaos[vkey]['label']} · CV={chaos[vkey]['cv_mean']:.2f}")
    ax_cv.axvline(1.0, color=theme.DEEP_RED, lw=0.9, ls=":", alpha=0.85)
    ax_cv.text(1.0, ax_cv.get_ylim()[1] * 0.95, " Poisson", color=theme.DEEP_RED,
               ha="left", va="top", fontsize=theme.SIZE_LABEL - 1)
    ax_cv.set_xlim(0, 3.0)
    ax_cv.set_xlabel("ISI CV per E neuron")
    ax_cv.set_ylabel("count")
    ax_cv.set_title("Per-neuron ISI CV (frozen DC)", loc="left",
                    fontsize=theme.SIZE_LABEL)
    ax_cv.legend(fontsize=theme.SIZE_LABEL - 2, frameon=False, loc="upper right")
    ax_cv.spines["top"].set_visible(False)
    ax_cv.spines["right"].set_visible(False)

    fig.suptitle(
        "Self-generated irregularity — with frozen (noise-free) DC input, only "
        "the recurrent network fires aperiodically; decoupled cells are clocks",
        x=0.01, ha="left", fontsize=theme.SIZE_LABEL + 1)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _fit_lambda(t_ms: np.ndarray, vmean: np.ndarray,
                lo: float, hi: float) -> float:
    """Slope of log‖ΔV‖ vs t (1/s) over [lo, hi] ms — the Lyapunov exponent."""
    m = (t_ms >= lo) & (t_ms <= hi) & (vmean > 0)
    if m.sum() < 3:
        return float("nan")
    slope = np.polyfit(t_ms[m], np.log(vmean[m]), 1)[0]  # 1/ms
    return float(slope * 1000.0)                          # 1/s


def run_lyapunov() -> dict:
    """Clone-and-perturb voltage-distance probe for balanced vs decoupled
    networks under quenched DC, averaged over seeds. Returns per-variant the
    common time axis, per-seed ‖ΔV(t)‖, the seed-mean, and the fitted λ."""
    out: dict = {"variants": {}}
    for vkey, vspec in CHAOS_VARIANTS.items():
        traces, t_axis = [], None
        for seed in LYAP_SEEDS:
            for p in (SCOPE_OUT_PNG, SCOPE_OUT_NPZ):
                if p.exists():
                    p.unlink()
            argv = [*COMMON_ARGS, "--t-ms", LYAP_T_MS,
                    *_chaos_args(vspec, seed), "--lyapunov-eps", LYAP_EPS]
            print(f"[lyap] {vkey} seed={seed}")
            subprocess.run(["uv", "run", "python", str(OSCILLOSCOPE), *argv],
                           cwd=REPO, check=True)
            data = np.load(SCOPE_OUT_NPZ)
            if "lyap_vdist" not in data:
                raise SystemExit("snapshot did not save lyap_vdist")
            traces.append(np.asarray(data["lyap_vdist"]))
            t_axis = np.asarray(data["lyap_t_ms"])
        n = min(len(v) for v in traces)
        t = t_axis[:n]
        V = np.array([v[:n] for v in traces])           # (seeds, n)
        vmean = V.mean(axis=0)
        lam = _fit_lambda(t, vmean, *LYAP_FIT_MS)
        out["variants"][vkey] = {
            "label": vspec["label"], "t_ms": t, "traces": V,
            "vmean": vmean, "lambda_per_s": lam,
        }
    return out


def plot_lyapunov(lyap: dict, out_path: Path) -> None:
    """log‖ΔV(t)‖ for balanced (grows, λ > 0) vs decoupled (decays, λ < 0):
    the direct Lyapunov-exponent measurement."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    styles = {"balanced": theme.INK_BLACK, "decoupled": theme.GREY_MID}
    for vkey, vdata in lyap["variants"].items():
        color = styles[vkey]
        t = vdata["t_ms"]
        for tr in vdata["traces"]:                       # faint per-seed
            ax.plot(t, np.clip(tr, 1e-6, None), color=color, lw=0.5, alpha=0.25)
        ax.plot(t, np.clip(vdata["vmean"], 1e-6, None), color=color, lw=2.0,
                label=f"{vdata['label']}  ·  λ = {vdata['lambda_per_s']:.0f}/s")
        # overlay the fit line over the balanced growth window
        if vkey == "balanced":
            lo, hi = LYAP_FIT_MS
            m = (t >= lo) & (t <= hi) & (vdata["vmean"] > 0)
            if m.sum() >= 3:
                b = np.polyfit(t[m], np.log(vdata["vmean"][m]), 1)
                ax.plot(t[m], np.exp(np.polyval(b, t[m])),
                        color=theme.DEEP_RED, lw=1.4, ls="--",
                        label=f"fit ({lo:.0f}–{hi:.0f} ms)")

    ax.set_yscale("log")
    ax.set_xlim(0, float(LYAP_T_MS))
    ax.set_xlabel("time since ε-kick (ms)")
    ax.set_ylabel(r"voltage distance $\Vert \Delta V(t) \Vert$  (mV)")
    ax.set_title(
        "Direct Lyapunov measurement — the balanced net amplifies a microscopic "
        "voltage kick (λ > 0); the decoupled control stays marginal (λ ≈ 0)",
        loc="left", fontsize=theme.SIZE_LABEL)
    ax.legend(fontsize=theme.SIZE_LABEL - 1, frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
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

        # Summary statistics — the quantities the raster page reads off.
        data = np.load(SCOPE_OUT_NPZ)
        spk_e, spk_i = data["spk_e"], data["spk_i"]
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

    # N-invariance sweep → Figure 2.
    sweep = run_n_sweep()
    sweep_dst = FIGURES / "n_sweep.png"
    plot_n_sweep(sweep, sweep_dst)
    figures["n_sweep"] = sweep_dst
    print(f"wrote {sweep_dst}")

    # K-sweep, strong J/√K vs weak coupling → Figure 3.
    ksweep = run_k_sweep()
    ksweep_dst = FIGURES / "k_sweep.png"
    plot_k_sweep(ksweep, ksweep_dst)
    figures["k_sweep"] = ksweep_dst
    print(f"wrote {ksweep_dst}")

    # Self-generation test under quenched DC → Figure 4.
    chaos = run_chaos()
    chaos_dst = FIGURES / "chaos.png"
    plot_chaos(chaos, chaos_dst)
    figures["chaos"] = chaos_dst
    print(f"wrote {chaos_dst}")
    chaos_summary = {
        vk: {"label": v["label"], "cv_mean": v["cv_mean"],
             "cv_std": v["cv_std"], "rate": v["rate"]}
        for vk, v in chaos.items()
    }

    # Direct Lyapunov-exponent measurement → Figure 5.
    lyap = run_lyapunov()
    lyap_dst = FIGURES / "lyapunov.png"
    plot_lyapunov(lyap, lyap_dst)
    figures["lyapunov"] = lyap_dst
    print(f"wrote {lyap_dst}")
    lyap_summary = {
        vk: {"label": v["label"], "lambda_per_s": v["lambda_per_s"]}
        for vk, v in lyap["variants"].items()
    }

    duration_s = time.monotonic() - t_start
    summary = {
        "notebook_run_id": notebook_run_id,
        "duration_s": round(duration_s, 1),
        "tier": tier,
        "common_args": COMMON_ARGS,
        "cells": {cell: spec["args"] for cell, spec in CELLS.items()},
        "summary": summary_rows,
        "n_sweep": sweep,
        "k_sweep": ksweep,
        "chaos": chaos_summary,
        "lyapunov": lyap_summary,
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
