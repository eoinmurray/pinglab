"""Notebook runner for entry 041 — retrained τ_GABA → f_γ → rate-ceiling sweep.

The "cannot escape" version of nb037 Figure 6. nb037 mutated τ_GABA on
a network trained at 9 ms (inference-only). This entry trains one PING
from scratch per τ_GABA value, then measures the realised gamma
frequency f_γ from the trained network's population-spike PSD and the
post-training mean E rate. The forbidding signature is that all
retrained points lie on r_E = p · f_γ — the optimiser cannot find a
solution off the structural-bound line even when given the chance to
try at each timescale.

Adds the --tau-gaba flag to the oscilloscope train subcommand
(symmetric counterpart to --tau-ampa, which doesn't yet exist either —
add later when needed).

Notebook entry: src/docs/src/pages/notebooks/nb041.mdx
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from helpers.figsave import save_figure  # noqa: E402
from helpers.fmt import format_duration  # noqa: E402
from helpers.modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from helpers.operating_point import (  # noqa: E402
    MODELS_DEFAULT_TAU_GABA_MS,
    TAU_GABA_GAMMA_MS,
)
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402
from helpers import theme  # noqa: E402

SLUG = "nb041"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
OSCILLOSCOPE = REPO / "src" / "cli" / "cli.py"

T_MS = 200.0
DT_TRAIN = 0.1

# τ_GABA grid — matches the nb037 inference-only sweep so the retrained
# and inference-only curves can be overlaid in ar008. nb037's default
# τ_GABA was 9.0 ms.
TAU_GABA_SWEEP: tuple[float, ...] = (4.5, 6.0, 9.0, 12.0, 18.0, 27.0)
SEEDS: tuple[int, ...] = (42, 43, 44)

# Search band for the gamma peak. τ_GABA = 4.5 ms predicts f_γ near
# 1/(τ_AMPA + τ_GABA) ≈ 1/(2 + 4.5) ms ≈ 154 Hz at the absurd extreme,
# but the loop physics adds latency and the realised f_γ is much lower.
# nb037 showed f_γ in the 20–50 Hz range across this τ_GABA sweep. Use
# a wide band (5–150 Hz) and pick argmax; record both the band-restricted
# peak and the unconstrained PSD for plotting.
F_GAMMA_BAND_HZ: tuple[float, float] = (5.0, 150.0)

# Raster-strip config — one panel per τ_GABA cluster, seed 42 only.
# 100 ms window long enough to show 2-4 full gamma cycles at every τ_GABA.
RASTER_SAMPLE_IDX: int = 0
RASTER_N_E_PLOT: int = 200
RASTER_N_I_PLOT: int = 64
RASTER_T_WINDOW_MS: float = 100.0

TIER_CONFIG = {
    "extra small": dict(max_samples=100, epochs=2),
    "small": dict(max_samples=500, epochs=10),
    # Medium tier upgraded 2026-06-03 to match nb024's convergence
    # finding: PING rate is a converged operating point at 100 epochs
    # but is still drifting at 30. Re-anchors the affine fit slope p
    # at converged values.
    "medium": dict(max_samples=2000, epochs=100),
    "large": dict(max_samples=5000, epochs=100),
    "extra large": dict(max_samples=10000, epochs=100),
}
DEFAULT_TIER = "small"

# Match nb025 PING recipe — same network, same optimiser,
# same readout; only τ_GABA varies.
PING_RECIPE: dict[str, str] = {
    "--ei-strength": "1",
    "--v-grad-dampen": "1000",
    "--w-in": "1.2",
    "--w-in-sparsity": "0.95",
    "--readout": "mem-mean",
    "--surrogate-slope": "1",
    "--readout-w-out-scale": "500",
    "--lr": "0.0004",
    "--batch-size": "256",
}


def tau_label(tau_ms: float) -> str:
    s = f"{tau_ms:g}".replace(".", "p")
    return f"tg{s}"


def cell_dir(tau_ms: float, seed: int) -> Path:
    """Trained cell — now the shared nb022 cell (train-once / reuse-many)."""
    from nb022 import cell_dir as shared_cell_dir
    return shared_cell_dir(f"ping__{tau_label(tau_ms)}__seed{seed}")


def build_train_args(tau_ms: float, seed: int, tier: str, out_dir: Path) -> list[str]:
    args = [
        "train",
        "--model", "ping",
        "--dataset", "mnist",
        "--max-samples", str(TIER_CONFIG[tier]["max_samples"]),
        "--epochs", str(TIER_CONFIG[tier]["epochs"]),
        "--t-ms", str(T_MS),
        "--dt", str(DT_TRAIN),
        "--seed", str(seed),
        "--tau-gaba", str(tau_ms),
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    for k, v in PING_RECIPE.items():
        args += [k, v]
    return args


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


# ─── inference: measure f_γ and post-training rate ──────────────────


def _cell_tau_gaba(cfg: dict) -> float:
    """The cell's trained τ_GABA (ms), or the models.py default if absent."""
    return float(cfg.get("tau_gaba_ms") or MODELS_DEFAULT_TAU_GABA_MS)


def _infer_cell(train_dir: Path, extra_args: list[str], out_name: str) -> Path:
    """Shell out to the CLI's `sim --infer` for one trained cell; return the out dir.

    Network build, weight load and forward all run in the CLI — the notebook only
    runs it and reads artifacts. The cell's τ_GABA is passed through so inference
    replays under the training-time inhibitory dynamics. extra_args adds mode flags
    (--outputs pop_traces for PSD, --sample-index for a snapshot raster).
    """
    train_dir = train_dir.resolve()
    cfg = json.loads((train_dir / "config.json").read_text())
    out_dir = (ARTIFACTS / out_name / train_dir.name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv", "run", "python", str(OSCILLOSCOPE), "sim", "--infer",
            "--load-config", str(train_dir / "config.json"),
            "--load-weights", str(train_dir / "weights.pth"),
            "--tau-gaba", str(_cell_tau_gaba(cfg)),
            "--out-dir", str(out_dir),
            *extra_args,
        ],
        cwd=REPO,
        check=True,
    )
    return out_dir


def measure_rate_and_psd(train_dir: Path) -> dict:
    """Accuracy, mean E rate, and gamma spectrum for a trained cell, via the CLI.

    Runs `sim --infer --outputs pop_traces` under the cell's τ_GABA (acc + rates in
    metrics.json, per-trial population traces in pop_traces.npz), then computes the
    Welch PSD and f_γ locally. The CLI emits the base signal; the metric is the
    notebook's.
    """
    cfg = json.loads((train_dir / "config.json").read_text())
    tau_gaba_ms = _cell_tau_gaba(cfg)
    out_dir = _infer_cell(train_dir, ["--outputs", "pop_traces"], "infer")
    m = json.loads((out_dir / "metrics.json").read_text())
    rates = m.get("rates_hz", {})

    pt = np.load(out_dir / "pop_traces.npz")
    pop_traces = list(pt["pop_e"])
    fs = 1000.0 / float(cfg["dt"])  # sampling rate in Hz (dt in ms)

    # Welch PSD per trial, averaged. nperseg = trial length gives
    # frequency resolution fs/nperseg ≈ 5 Hz at dt=0.1ms T=200ms.
    nperseg = pop_traces[0].size
    psds: list[np.ndarray] = []
    freqs: np.ndarray | None = None
    for tr in pop_traces:
        f, p = sp_signal.welch(
            tr - tr.mean(),
            fs=fs,
            nperseg=nperseg,
            scaling="density",
            detrend=False,
        )
        psds.append(p)
        freqs = f
    psd_mean = np.mean(np.stack(psds, axis=0), axis=0)
    assert freqs is not None

    f_gamma = _peak_with_parabolic(psd_mean, freqs)
    per_trial_peaks = [_peak_with_parabolic(p, freqs) for p in psds]

    return {
        "tau_gaba_ms": tau_gaba_ms,
        "acc": float(m["best_acc"]),
        "e_rate_hz": float(rates.get("hid", 0.0)),
        "f_gamma_hz": f_gamma,
        "freqs_hz": freqs.tolist(),
        "psd": psd_mean.tolist(),
        "per_trial_peaks_hz": [
            float(x) for x in per_trial_peaks if np.isfinite(x)
        ],
        "n_total": int(m.get("n_total", 0)),
    }


def _peak_with_parabolic(psd: np.ndarray, freqs: np.ndarray) -> float:
    """Locate the gamma-band peak with parabolic sub-bin interpolation.

    Returns NaN if the PSD is flat in the gamma band. Welch with
    nperseg = T_steps gives Δf = fs/nperseg (5 Hz at fs=10000, T=200ms)
    — too coarse on its own across six τ_GABA values. Parabolic
    interpolation through (k-1, k, k+1) recovers the analytic peak
    location with error O((Δf)^3) when the peak is well-isolated.
    """
    band_mask = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
    if not band_mask.any() or psd[band_mask].max() <= 0:
        return float("nan")
    in_band = np.where(band_mask)[0]
    peak_local = int(psd[in_band].argmax())
    peak_idx = int(in_band[peak_local])
    if not (0 < peak_idx < len(psd) - 1):
        return float(freqs[peak_idx])
    y0 = float(psd[peak_idx - 1])
    y1 = float(psd[peak_idx])
    y2 = float(psd[peak_idx + 1])
    denom = y0 - 2.0 * y1 + y2
    offset = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    offset = max(-0.5, min(0.5, offset))
    df = float(freqs[1] - freqs[0])
    return float(freqs[peak_idx]) + offset * df


def capture_raster(train_dir: Path, sample_idx: int) -> dict:
    """Single-trial E and I spike raster from a trained cell, via the CLI snapshot.

    Runs `sim --infer --sample-index N` under the cell's τ_GABA so the forward
    replays under its own dynamics, then reads the rasters + class label from
    snapshot.npz and subsamples cells for the plot.
    """
    cfg = json.loads((train_dir / "config.json").read_text())
    tau_gaba_ms = _cell_tau_gaba(cfg)
    out_dir = _infer_cell(train_dir, ["--sample-index", str(sample_idx)], "snapshot")
    d = np.load(out_dir / "snapshot.npz")
    e_full = d["spk_e"]
    i_full = d["spk_i"]
    if e_full.ndim == 3:
        e_full = e_full[:, 0, :]
    if i_full.ndim == 3:
        i_full = i_full[:, 0, :]
    t_sec = float(cfg["t_ms"]) / 1000.0
    e_rate = float(e_full.sum() / (e_full.shape[1] * t_sec))
    i_rate = float(i_full.sum() / (i_full.shape[1] * t_sec))
    rng = np.random.default_rng(0)
    e_idx = np.sort(rng.choice(e_full.shape[1], RASTER_N_E_PLOT, replace=False))
    i_idx = np.sort(rng.choice(i_full.shape[1], RASTER_N_I_PLOT, replace=False))
    return {
        "tau_gaba_ms": tau_gaba_ms,
        "label": int(d["label"]),
        "e": e_full[:, e_idx].astype(bool),
        "i": i_full[:, i_idx].astype(bool),
        "e_rate_hz": e_rate,
        "i_rate_hz": i_rate,
        "dt_ms": float(cfg["dt"]),
        "t_ms": float(cfg["t_ms"]),
    }


# ─── plotting ───────────────────────────────────────────────────────


def plot_quantitative_law(
    rows: list[dict], out_path: Path, run_id: str
) -> dict:
    """Scatter: f_γ vs E rate, one point per retrained cell. Overlay
    r_E = p · f_γ fit. Sibling panel: f_γ vs accuracy.
    Returns (p_fit, r2)."""
    theme.apply()

    # Group by τ_GABA — colour bands; error bars across seeds.
    by_tau: dict[float, list[dict]] = {}
    for r in rows:
        by_tau.setdefault(r["tau_gaba_ms"], []).append(r)

    fig, (ax_rate, ax_acc) = plt.subplots(
        2, 1, figsize=(5.6, 4.2), sharex=True,
        gridspec_kw={"hspace": 0.12, "height_ratios": [1.6, 1.0]},
    )

    f_gammas: list[float] = []
    e_rates: list[float] = []
    for tau, sub in sorted(by_tau.items()):
        fg_mu = float(np.mean([s["f_gamma_hz"] for s in sub]))
        fg_se = float(
            np.std([s["f_gamma_hz"] for s in sub], ddof=1) / np.sqrt(len(sub))
            if len(sub) > 1 else 0.0
        )
        er_mu = float(np.mean([s["e_rate_hz"] for s in sub]))
        er_se = float(
            np.std([s["e_rate_hz"] for s in sub], ddof=1) / np.sqrt(len(sub))
            if len(sub) > 1 else 0.0
        )
        ac_mu = float(np.mean([s["acc"] for s in sub]))
        ac_se = float(
            np.std([s["acc"] for s in sub], ddof=1) / np.sqrt(len(sub))
            if len(sub) > 1 else 0.0
        )
        f_gammas.append(fg_mu)
        e_rates.append(er_mu)
        ax_rate.errorbar(
            fg_mu, er_mu, xerr=fg_se, yerr=er_se,
            fmt="o", markersize=6, color=theme.INK_BLACK,
            capsize=3,
            label=f"τ_GABA = {tau:g} ms" if tau == TAU_GABA_GAMMA_MS else None,
        )
        ax_rate.annotate(
            f" {tau:g} ms",
            (fg_mu, er_mu),
            fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED,
            xytext=(4, 0), textcoords="offset points",
            va="center",
        )
        ax_acc.errorbar(
            fg_mu, ac_mu, xerr=fg_se, yerr=ac_se,
            fmt="o", markersize=6, color=theme.INK_BLACK, capsize=3,
        )

    # Two fits:
    # (1) Affine r_E = a + p · f_γ — the physically honest form. The
    #     intercept a is the non-rhythmic baseline E rate from feedforward
    #     drive that survives at low f_γ; the slope p is per-cycle
    #     participation probability. This is the fit the figure reports.
    # (2) Through-origin r_E = p · f_γ — the structural-bound *idealised*
    #     form. Reported as a constraint check (does a fit through zero
    #     work?); R² will be poor when the intercept is non-negligible.
    fg_arr = np.array(f_gammas)
    er_arr = np.array(e_rates)
    if (fg_arr > 0).all():
        slope_aff, intercept_aff = np.polyfit(fg_arr, er_arr, 1)
        p_fit = float(slope_aff)
        a_fit = float(intercept_aff)
        er_pred_aff = p_fit * fg_arr + a_fit
        ss_res_aff = float(np.sum((er_arr - er_pred_aff) ** 2))
        ss_tot = float(np.sum((er_arr - er_arr.mean()) ** 2))
        r2 = 1.0 - ss_res_aff / ss_tot if ss_tot > 0 else float("nan")
        # Through-origin: also fit for reporting in numbers.json.
        p0 = float(np.sum(fg_arr * er_arr) / np.sum(fg_arr ** 2))
        er_pred_0 = p0 * fg_arr
        ss_res_0 = float(np.sum((er_arr - er_pred_0) ** 2))
        r2_origin = 1.0 - ss_res_0 / ss_tot if ss_tot > 0 else float("nan")

        xs = np.linspace(0, fg_arr.max() * 1.1, 200)
        ax_rate.plot(
            xs, p_fit * xs + a_fit,
            color=theme.DEEP_RED, lw=1.2, ls="--",
            label=(
                f"$r_E = a + p · f_γ$  (a = {a_fit:.2f} Hz, "
                f"p = {p_fit:.3f}, R² = {r2:.3f})"
            ),
        )
    else:
        p_fit, a_fit, r2, p0, r2_origin = (
            float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
        )

    ax_rate.set_ylabel("Hidden E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax_acc.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax_acc.set_xlabel(
        "Measured $f_γ$ (Hz) — peak of trained-network population PSD",
        fontsize=theme.SIZE_LABEL,
    )
    ax_acc.set_ylim(0, 100)
    ax_acc.axhline(10.0, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.6)
    for ax in (ax_rate, ax_acc):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=theme.SIZE_TICK)
    ax_rate.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    fig.suptitle(
        "Retrained τ_GABA sweep — post-training E rate tracks measured $f_γ$",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)
    return {
        "p_affine": p_fit,
        "a_affine": a_fit,
        "r2_affine": r2,
        "p_origin": p0,
        "r2_origin": r2_origin,
    }


def plot_training_curves(
    out_path: Path, run_id: str
) -> None:
    """Per-cell training-trajectory curves so convergence is auditable
    by inspection. One line per (τ_GABA, seed); colour by τ_GABA."""
    theme.apply()
    cmap = plt.get_cmap("viridis")
    taus_sorted = list(TAU_GABA_SWEEP)
    fig, (ax_acc, ax_rate) = plt.subplots(
        2, 1, figsize=(6.9, 5.175), sharex=True,
        gridspec_kw={"hspace": 0.15},
    )
    for i, tau in enumerate(taus_sorted):
        color = cmap(i / max(1, len(taus_sorted) - 1))
        for j, seed in enumerate(SEEDS):
            mfile = cell_dir(tau, seed) / "metrics.json"
            if not mfile.exists():
                continue
            m = json.loads(mfile.read_text())
            eps = [e["ep"] for e in m["epochs"]]
            accs = [e.get("acc", 0) for e in m["epochs"]]
            rates = [e.get("test_rate_e", 0) for e in m["epochs"]]
            label = f"τ_GABA = {tau:g} ms" if j == 0 else None
            ax_acc.plot(eps, accs, color=color, lw=1.0, alpha=0.85, label=label)
            ax_rate.plot(eps, rates, color=color, lw=1.0, alpha=0.85)
    ax_acc.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax_rate.set_ylabel("Test E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax_rate.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    ax_acc.legend(fontsize=theme.SIZE_LEGEND, frameon=False, ncol=2, loc="lower right")
    ax_acc.set_ylim(0, 100)
    for ax in (ax_acc, ax_rate):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, lw=0.4)
    fig.suptitle(
        "Per-cell training curves — convergence check across τ_GABA sweep",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_psd_panel(
    rows: list[dict], out_path: Path, run_id: str
) -> None:
    """One PSD curve per τ_GABA (mean across seeds), to verify the
    peak-frequency identification is reading the right feature."""
    theme.apply()
    by_tau: dict[float, list[dict]] = {}
    for r in rows:
        by_tau.setdefault(r["tau_gaba_ms"], []).append(r)

    fig, ax = plt.subplots(figsize=(5.6, 3.15))
    cmap = plt.get_cmap("viridis")
    taus_sorted = sorted(by_tau.keys())
    for i, tau in enumerate(taus_sorted):
        sub = by_tau[tau]
        freqs = np.array(sub[0]["freqs_hz"])
        psd_mean = np.mean(np.stack([np.array(s["psd"]) for s in sub], axis=0), axis=0)
        band_mask = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
        ax.plot(
            freqs[band_mask], psd_mean[band_mask],
            color=cmap(i / max(1, len(taus_sorted) - 1)),
            label=f"τ_GABA = {tau:g} ms",
            lw=1.2,
        )
        peak_idx = psd_mean[band_mask].argmax()
        peak_f = freqs[band_mask][peak_idx]
        peak_p = psd_mean[band_mask][peak_idx]
        ax.scatter([peak_f], [peak_p], color=cmap(i / max(1, len(taus_sorted) - 1)),
                   s=20, zorder=5)
    ax.set_xlabel("Frequency (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Population PSD (a.u.)", fontsize=theme.SIZE_LABEL)
    ax.set_xlim(F_GAMMA_BAND_HZ)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(
        "Trained-network population PSDs — peak marks $f_γ$",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_per_trial_peaks(
    rows: list[dict], out_path: Path, run_id: str,
) -> None:
    """Sanity check: per-trial PSD peak distribution per τ_GABA.

    Pools per-trial peak frequencies across seeds for each τ_GABA value
    and shows their histogram alongside the trial-mean-PSD peak.
    Narrow histograms → trial-mean-PSD f_γ is unbiased; wide histograms
    → trial-mean PSD is a centroid, and per-trial median would differ.
    """
    theme.apply()
    by_tau: dict[float, list[dict]] = {}
    for r in rows:
        by_tau.setdefault(r["tau_gaba_ms"], []).append(r)
    taus_sorted = sorted(by_tau.keys())
    n_taus = len(taus_sorted)
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(
        n_taus, 1, figsize=(6.9, (1.5 * n_taus + 1.0) * 0.8625), sharex=True,
    )
    if n_taus == 1:
        axes = [axes]
    bin_edges = np.arange(F_GAMMA_BAND_HZ[0], F_GAMMA_BAND_HZ[1] + 1.0, 1.0)

    for ax, tau in zip(axes, taus_sorted):
        sub = by_tau[tau]
        peaks = np.array(
            [pk for s in sub for pk in s.get("per_trial_peaks_hz", [])],
            dtype=float,
        )
        color = cmap(taus_sorted.index(tau) / max(1, n_taus - 1))
        if peaks.size > 0:
            ax.hist(peaks, bins=bin_edges, color=color, alpha=0.85,
                    edgecolor=theme.INK_BLACK, lw=0.4)
            median = float(np.median(peaks))
            iqr = float(np.percentile(peaks, 75) - np.percentile(peaks, 25))
            ax.axvline(median, color=theme.INK_BLACK, ls="--", lw=1.0)
            ax.text(
                0.98, 0.85,
                f"τ_GABA = {tau:g} ms\n"
                f"per-trial: median {median:.1f} Hz, IQR {iqr:.1f} Hz",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=theme.SIZE_LABEL,
            )
        mean_peak = float(np.mean(
            [s["f_gamma_hz"] for s in sub if np.isfinite(s["f_gamma_hz"])]
        ))
        ax.axvline(
            mean_peak, color=theme.DEEP_RED, ls="-", lw=1.2,
            label=f"trial-mean PSD peak: {mean_peak:.1f} Hz",
        )
        ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
        ax.set_ylabel("trials", fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Per-trial PSD peak frequency (Hz)",
                        fontsize=theme.SIZE_LABEL)
    fig.suptitle(
        "Per-trial peak distribution vs trial-mean-PSD peak",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_raster_strip(
    samples: list[dict], out_path: Path, run_id: str, t_window_ms: float,
) -> None:
    """One stacked single-trial raster per τ_GABA cluster. X-axis is
    physical time in ms (not steps), so the gamma cadence shift with
    τ_GABA is read by eye — shorter τ_GABA gives faster bursts gives
    more E spikes per unit time."""
    theme.apply()
    # Show τ_GABA in ascending value, top-down (so faster gamma at the top).
    samples = sorted(samples, key=lambda s: s["tau_gaba_ms"])
    n = len(samples)
    n_e = RASTER_N_E_PLOT
    n_i = RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n, 1, figsize=(6.9, (1.0 * n + 1.0) * 0.69),
        sharex=True, gridspec_kw={"hspace": 0.22},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples)):
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt_ms"]
        mask = t_axis <= t_window_ms
        e_t, e_n = np.where(s["e"][mask])
        i_t, i_n = np.where(s["i"][mask])
        ax.scatter(t_axis[mask][e_t], e_n,
                   s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.4)
        ax.scatter(t_axis[mask][i_t], i_n + n_e + gap,
                   s=2.0, c=theme.DEEP_RED, marker="|", linewidths=0.4)
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, t_window_ms)
        ax.text(
            1.012, 0.5,
            f"τ_GABA = {s['tau_gaba_ms']:g} ms\nE = {s['e_rate_hz']:.1f} Hz",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=theme.SIZE_LABEL,
        )
        if i == 0:
            ax.set_title(
                "Trained-PING rasters at each τ_GABA (seed 42, MNIST digit 0 sample 0) — "
                "x-axis is physical time in ms"
            )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


# ─── success criteria ───────────────────────────────────────────────

def main() -> None:
    # Publication profile: every figure this notebook writes is a print-sized
    # vector, emitted as both SVG (docs) and PDF (manuscript) by save_figure.
    theme.set_paper_mode(True)

    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    skip_training = "--skip-training" in sys.argv
    only_missing = "--only-missing" in sys.argv
    wipe_dir = "--no-wipe-dir" not in sys.argv

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    n_cells = len(TAU_GABA_SWEEP) * len(SEEDS)
    print(
        f"notebook_run_id = {notebook_run_id} tier={tier} cells={n_cells}"
        + ("  [skip-training]" if skip_training else "")
        + (f"  [modal:{modal_gpu}]" if modal_gpu else "")
    )

    prepare_run_dirs(
        SLUG, notebook_run_id, wipe=wipe_dir, skip_training=skip_training,
        make_artifacts=False,
    )

    # Training lives in nb022 now (train-once / reuse-many): the τ_GABA ladder
    # is a registry family there. This notebook only consumes the cells.

    # Inference: measure (acc, E rate, f_γ) per cell.
    rows: list[dict] = []
    for tau in TAU_GABA_SWEEP:
        for seed in SEEDS:
            run_dir = cell_dir(tau, seed)
            if not (run_dir / "weights.pth").exists():
                raise SystemExit(f"missing weights: {run_dir / 'weights.pth'}")
            t0 = time.monotonic()
            res = measure_rate_and_psd(run_dir)
            res["seed"] = seed
            rows.append(res)
            print(
                f"  τ_GABA={tau:>5.1f}ms seed={seed}  "
                f"acc={res['acc']:5.2f}%  E={res['e_rate_hz']:6.2f} Hz  "
                f"f_γ={res['f_gamma_hz']:6.2f} Hz  ({time.monotonic() - t0:.1f}s)"
            )

    fit = plot_quantitative_law(
        rows, FIGURES / "rate_vs_fgamma", notebook_run_id,
    )
    print(
        f"wrote {FIGURES / 'rate_vs_fgamma'}.{{svg,pdf}}  "
        f"(affine: a={fit['a_affine']:.2f}, p={fit['p_affine']:.3f}, "
        f"R²={fit['r2_affine']:.3f})"
    )
    plot_psd_panel(rows, FIGURES / "psds", notebook_run_id)
    print(f"wrote {FIGURES / 'psds'}.{{svg,pdf}}")
    plot_per_trial_peaks(
        rows, FIGURES / "per_trial_peaks", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'per_trial_peaks'}.{{svg,pdf}}")

    # Raster strip — one panel per τ_GABA cluster, seed 42 only. Makes
    # the affine law visceral: shorter τ_GABA → faster gamma → more
    # E spikes per unit time.
    print(f"[raster] single-trial panels from seed {SEEDS[0]}, "
          f"sample {RASTER_SAMPLE_IDX}")
    raster_samples = []
    for tau_ms in TAU_GABA_SWEEP:
        train_dir = cell_dir(tau_ms, SEEDS[0])
        raster_samples.append(capture_raster(train_dir, RASTER_SAMPLE_IDX))
    plot_raster_strip(
        raster_samples, FIGURES / "raster_strip", notebook_run_id,
        t_window_ms=RASTER_T_WINDOW_MS,
    )
    print(f"wrote {FIGURES / 'raster_strip'}.{{png,pdf}}")

    plot_training_curves(FIGURES / "training_curves", notebook_run_id)
    print(f"wrote {FIGURES / 'training_curves'}.{{svg,pdf}}")

    duration_s = time.monotonic() - t_start
    train_cfg = load_config(cell_dir(TAU_GABA_SWEEP[0], SEEDS[0]))
    summary = {
        "notebook_run_id": notebook_run_id,
        "git_sha": train_cfg.get("git_sha"),
        "duration_s": round(duration_s, 1),
        "duration": format_duration(duration_s),
        "tier": tier,
        "config": {
            "tier": tier,
            "dataset": "mnist",
            "tau_gaba_sweep_ms": list(TAU_GABA_SWEEP),
            "seeds": list(SEEDS),
            "f_gamma_band_hz": list(F_GAMMA_BAND_HZ),
            "max_samples": TIER_CONFIG[tier]["max_samples"],
            "epochs": TIER_CONFIG[tier]["epochs"],
            "t_ms": T_MS,
            "dt": DT_TRAIN,
        },
        "fit": fit,
        # results: drop bulky freqs/psd lists; keep them in per-cell
        # numpy if you want them in the future.
        "results": [
            {k: v for k, v in r.items() if k not in ("freqs_hz", "psd")}
            for r in rows
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")



if __name__ == "__main__":
    main()
