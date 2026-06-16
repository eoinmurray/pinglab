"""Notebook runner for entry 054 — A PING rhythmicity metric (calibration).

Step 1 of the proposal: implement the spike-time-autocorrelation rhythmicity
primitives (metrics.iei_histogram / spike_autocorrelogram / rhythmicity_metrics)
and calibrate the candidate scalars on synthetic non-homogeneous Poisson
references — the ground truth — before any application to trained networks.

The references span a continuous spectrum rather than three samples. Each is a
single non-homogeneous Poisson SPIKE TRAIN with rate λ(t) = λ0·(1 + m·sin φ(t))
— the autocorrelogram is computed from those Poisson-encoded spike events, not
from the analytic rate (a Poisson process is memoryless, so the correlogram's
expectation equals the rate autocorrelation, but a single train carries real
finite-count shot noise — the genuine spike-level object). The modulation depth
m is swept from 0 (flat, asynchronous) to 1 (a deep rhythm), in two flavours:
  - stationary — fixed frequency f (a pure rhythm), and
  - drifting   — f swept as a chirp (a non-stationary rhythm).
m = 0 is the flat baseline shared by both; the two m = 1 endpoints are the
"stationary" and "drifting" cases.

Two figures:
  - references.png  — rate, IEI histogram and autocorrelogram (the Mexican hat)
    at the spectrum endpoints (flat, stationary, drifting).
  - calibration.png — the scalars across the whole m-sweep: the autocorrelation
    lobe-to-trough rises from the baseline 1 and tracks together for the
    stationary and drifting series (stationarity-robust), while the PSD peak
    rockets for the stationary case and stays pinned for the drifting one it
    cannot tell from a weak rhythm.

Synthetic + measurement only: no training, no network inference, local CPU.

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
from helpers.run_id import next_run_id, persist as persist_run_id  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402
from cli import theme  # noqa: E402
from cli.metrics import (  # noqa: E402
    population_rate_nondiff,
    rhythmicity_metrics,
    rhythmicity_scalars,
)

SLUG = "nb054"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG

# ─── recipe (hardcoded literals; the notebook IS the recipe) ────────────
DT_MS = 0.25
LAMBDA0_HZ = 20.0          # per-cell baseline rate
F_STATIONARY_HZ = 40.0     # fixed gamma frequency (stationary case)
F_DRIFT_HZ = (30.0, 55.0)  # chirp endpoints (drifting case)
M_MAX = 1.0                # modulation depth swept over [0, M_MAX]
MAX_LAG_MS = 100.0
BIN_MS = 1.0
PSD_BAND_HZ = (20.0, 80.0)
SCALARS = ("lobe_to_trough", "iei_anchored", "psd_snr")

CONDITIONS = ("stationary", "drifting")
COND_STYLE = {
    "stationary": dict(color=theme.INK_BLACK, ls="-"),
    "drifting": dict(color=theme.DEEP_RED, ls="--"),
}

# Single Poisson spike train per reference; long traces give the correlogram
# enough spike pairs, n_seeds averages out shot noise, n_m sets sweep density.
TIER_CONFIG = {
    "extra small": dict(t_ms=20000.0, n_seeds=2, n_m=5),
    "small": dict(t_ms=60000.0, n_seeds=3, n_m=7),
    "medium": dict(t_ms=120000.0, n_seeds=4, n_m=11),
    "large": dict(t_ms=300000.0, n_seeds=6, n_m=15),
    "extra large": dict(t_ms=600000.0, n_seeds=8, n_m=21),
}
DEFAULT_TIER = "medium"


def rate_trace(kind, m, t_ms, dt):
    """Per-cell instantaneous rate λ(t) in Hz at modulation depth m."""
    t = np.arange(int(t_ms / dt)) * dt / 1000.0  # seconds
    if kind == "stationary":
        phase = 2 * np.pi * F_STATIONARY_HZ * t
    elif kind == "drifting":
        f0, f1 = F_DRIFT_HZ
        f_t = f0 + (f1 - f0) * (t / t[-1])
        phase = 2 * np.pi * np.cumsum(f_t) * (dt / 1000.0)
    else:
        raise ValueError(kind)
    return LAMBDA0_HZ * (1.0 + m * np.sin(phase))


def poisson_spike_train(rate_hz, dt, rng):
    """[T, 1] raster: one non-homogeneous Poisson spike train at rate λ(t)."""
    p = np.clip(rate_hz[:, None] * dt / 1000.0, 0.0, 1.0)
    return (rng.random((rate_hz.size, 1)) < p).astype(np.int8)


def psd_peak_snr(raster, n_neurons, dt, bin_ms=BIN_MS):
    """Rate PSD peak-to-median ratio in the gamma band."""
    _, rate = population_rate_nondiff(raster, n_neurons, bin_ms=bin_ms, dt_ms=dt)
    rate = rate - rate.mean()
    freqs = np.fft.rfftfreq(rate.size, d=bin_ms / 1000.0)
    psd = np.abs(np.fft.rfft(rate)) ** 2
    band = (freqs >= PSD_BAND_HZ[0]) & (freqs <= PSD_BAND_HZ[1])
    med = np.median(psd[band])
    if not band.any() or med <= 0:
        return 0.0
    return float(psd[band].max() / med)


def metrics_for(kind, m, cfg, seed):
    """All scalars (+ curves) for one (kind, m, seed) — single Poisson train."""
    rate = rate_trace(kind, m, cfg["t_ms"], DT_MS)
    rng = np.random.default_rng(1000 + seed)
    raster = poisson_spike_train(rate, DT_MS, rng)
    md = rhythmicity_metrics(raster, DT_MS, max_lag_ms=MAX_LAG_MS, bin_ms=BIN_MS)
    md["psd_snr"] = psd_peak_snr(raster, 1, DT_MS)
    return md


def run_point(kind, m, cfg, seeds):
    """One spectrum point: trial-average the correlograms across seeds, extract
    scalars from the average (unbiased on noisy single-train data), and keep the
    per-seed scalars for the error band."""
    mds = [metrics_for(kind, m, cfg, s) for s in seeds]
    ac_lags, iei_lags = mds[0]["ac_lags"], mds[0]["iei_lags"]
    avg_ac = np.nanmean(np.array([md["ac"] for md in mds]), axis=0)
    avg_iei = np.mean(np.array([md["iei_counts"] for md in mds]), axis=0)
    avg = rhythmicity_scalars(ac_lags, avg_ac, iei_lags, avg_iei, BIN_MS)
    avg.update(ac_lags=ac_lags, ac=avg_ac, iei_lags=iei_lags, iei_counts=avg_iei)
    avg["psd_snr"] = float(np.mean([md["psd_snr"] for md in mds]))
    seed = {
        met: [md[met] for md in mds] for met in ("lobe_to_trough", "iei_anchored")
    }
    seed["psd_snr"] = [md["psd_snr"] for md in mds]
    return {"avg": avg, "seed": seed}


def run_sweep(cfg, m_values, seeds):
    """data[kind][i] = {avg, seed} for m_values[i]."""
    return {
        kind: [run_point(kind, m, cfg, seeds) for m in m_values]
        for kind in CONDITIONS
    }


def curve(data, kind, metric):
    """(mean, std) along the m-sweep: mean from the trial-averaged correlogram,
    SD from the per-seed scalars (the single-train shot-noise spread)."""
    pts = data[kind]
    mean = np.array(
        [(p["avg"][metric] if p["avg"][metric] is not None else np.nan) for p in pts]
    )
    std = np.array(
        [np.nanstd([v for v in p["seed"][metric] if v is not None]) for p in pts]
    )
    return mean, std


def fig_references(shapes, out_path):
    """3×3 grid: rate / IEI / autocorrelation at the spectrum endpoints."""
    fig, axes = plt.subplots(3, 3, figsize=(11, 7.5), dpi=150)
    win_ms = 250.0
    ac_all = np.concatenate([s["curves"]["ac"][1:] for s in shapes])
    ac_lo, ac_hi = np.nanmin(ac_all), np.nanmax(ac_all)
    ac_pad = 0.08 * (ac_hi - ac_lo)
    for col, s in enumerate(shapes):
        c, cur = s["color"], s["curves"]

        ax = axes[0, col]
        n = int(win_ms / DT_MS)
        ax.plot(np.arange(n) * DT_MS, s["rate"][:n], color=c, lw=1.0)
        ax.set_title(s["label"], fontsize=theme.SIZE_TITLE, color=theme.INK)
        ax.set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
        if col == 0:
            ax.set_ylabel("rate λ(t) (Hz)", fontsize=theme.SIZE_LABEL)

        ax = axes[1, col]
        counts = cur["iei_counts"]
        ax.bar(cur["iei_lags"], counts, width=BIN_MS, color=c, alpha=0.85)
        if counts.size > 4 and counts[3:].max() > 0:
            ax.set_ylim(0, 1.25 * counts[3:].max())
        ax.set_xlabel("inter-event interval (ms)", fontsize=theme.SIZE_LABEL)
        if col == 0:
            ax.set_ylabel("count (zoomed)", fontsize=theme.SIZE_LABEL)

        ax = axes[2, col]
        ax.axhline(1.0, color=theme.FAINT, lw=0.8, ls=":")
        ax.plot(cur["ac_lags"], cur["ac"], color=c, lw=1.2)
        if cur["lobe_lag"]:
            ax.plot(cur["lobe_lag"], cur["ac"][int(round(cur["lobe_lag"] / BIN_MS))],
                    "^", color=theme.INK, ms=6)
        if cur["trough_lag"]:
            ax.plot(cur["trough_lag"], cur["ac"][int(round(cur["trough_lag"] / BIN_MS))],
                    "v", color=theme.DEEP_RED, ms=6)
        ax.set_ylim(ac_lo - ac_pad, ac_hi + ac_pad)
        ax.set_xlabel("lag (ms)", fontsize=theme.SIZE_LABEL)
        if col == 0:
            ax.set_ylabel("autocorrelation", fontsize=theme.SIZE_LABEL)
        lt = cur["lobe_to_trough"]
        ax.text(0.97, 0.92, f"lobe/trough = {lt:.1f}" if lt else "lobe/trough = —",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=theme.SIZE_ANNOTATION, color=theme.INK)

    fig.suptitle(
        "Rhythmicity references (spectrum endpoints): rate, IEI, autocorrelation",
        fontsize=theme.SIZE_TITLE, color=theme.INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_spectrum(m_values, sweep, out_path):
    """The scalars across the modulation-depth sweep, both conditions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    for kind in CONDITIONS:
        st = COND_STYLE[kind]
        lt_m, lt_s = sweep[kind]["lobe_to_trough"]
        ax1.fill_between(m_values, lt_m - lt_s, lt_m + lt_s, color=st["color"], alpha=0.15)
        ax1.plot(m_values, lt_m, color=st["color"], ls=st["ls"], lw=1.8,
                 label=f"{kind} · lobe/trough")
        ia_m, _ = sweep[kind]["iei_anchored"]
        ax1.plot(m_values, ia_m, color=st["color"], ls=st["ls"], lw=1.0, alpha=0.5,
                 label=f"{kind} · IEI-anchored")
    ax1.axhline(1.0, color=theme.FAINT, lw=0.8, ls=":")
    ax1.set_xlabel("modulation depth m  (0 = flat → 1 = deep rhythm)", fontsize=theme.SIZE_LABEL)
    ax1.set_ylabel("rhythmicity scalar", fontsize=theme.SIZE_LABEL)
    ax1.set_title("Autocorrelation scalars track together (baseline = 1)",
                  fontsize=theme.SIZE_LABEL)
    ax1.legend(fontsize=theme.SIZE_LEGEND, frameon=False)

    for kind in CONDITIONS:
        st = COND_STYLE[kind]
        sn_m, sn_s = sweep[kind]["psd_snr"]
        ax2.fill_between(m_values, sn_m - sn_s, sn_m + sn_s, color=st["color"], alpha=0.15)
        ax2.plot(m_values, sn_m, color=st["color"], ls=st["ls"], lw=1.8, label=kind)
    ax2.set_xlabel("modulation depth m", fontsize=theme.SIZE_LABEL)
    ax2.set_ylabel("PSD peak / median (gamma band)", fontsize=theme.SIZE_LABEL)
    ax2.set_title("Periodogram peak diverges (drifting ≈ flat)", fontsize=theme.SIZE_LABEL)
    ax2.legend(fontsize=theme.SIZE_LEGEND, frameon=False)

    fig.suptitle(
        "Calibration spectrum: autocorrelation is stationarity-robust; the PSD peak is not",
        fontsize=theme.SIZE_TITLE, color=theme.INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    argv = sys.argv[1:]
    tier = parse_tier(argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(argv)  # accepted for contract parity; unused (local CPU)
    cfg = TIER_CONFIG[tier]
    seeds = list(range(cfg["n_seeds"]))
    m_values = np.linspace(0.0, M_MAX, cfg["n_m"])

    if modal_gpu:
        print("note: nb054 is synthetic + local CPU; --modal-gpu ignored.")

    t_start = time.monotonic()
    for d in (ARTIFACTS, FIGURES):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    notebook_run_id = next_run_id(SLUG)
    theme.apply()

    print(
        f"nb054 calibration | tier={tier} | single Poisson train × "
        f"{cfg['t_ms'] / 1000:.0f} s × {cfg['n_seeds']} seed(s) "
        f"× {cfg['n_m']} m-steps × {len(CONDITIONS)} conditions"
    )
    data = run_sweep(cfg, m_values, seeds)
    sweep = {
        kind: {met: curve(data, kind, met) for met in SCALARS}
        for kind in CONDITIONS
    }

    # Spectrum endpoints for the shape figure (trial-averaged curves): flat =
    # stationary at m = 0; the two rhythms at m = 1.
    shapes = [
        dict(label="flat (m=0)", color=theme.GREY_MID,
             curves=data["stationary"][0]["avg"],
             rate=rate_trace("stationary", m_values[0], cfg["t_ms"], DT_MS)),
        dict(label="stationary (m=1)", color=theme.INK_BLACK,
             curves=data["stationary"][-1]["avg"],
             rate=rate_trace("stationary", m_values[-1], cfg["t_ms"], DT_MS)),
        dict(label="drifting (m=1)", color=theme.DEEP_RED,
             curves=data["drifting"][-1]["avg"],
             rate=rate_trace("drifting", m_values[-1], cfg["t_ms"], DT_MS)),
    ]
    fig_references(shapes, FIGURES / "references.png")
    fig_spectrum(m_values, sweep, FIGURES / "calibration.png")
    print(f"wrote {FIGURES / 'references.png'}")
    print(f"wrote {FIGURES / 'calibration.png'}")

    def endpoint(kind, idx):
        return {met: float(sweep[kind][met][0][idx]) for met in SCALARS}

    # flat = m=0 endpoint (identical for both conditions); rhythms = m=1.
    references = {
        "flat": endpoint("stationary", 0),
        "stationary": endpoint("stationary", -1),
        "drifting": endpoint("drifting", -1),
    }
    for label, vals in references.items():
        print(
            f"  {label:11s}  lobe/trough={vals['lobe_to_trough']:.2f}  "
            f"IEI-anchored={vals['iei_anchored']:.2f}  PSD-SNR={vals['psd_snr']:.1f}"
        )

    duration_s = time.monotonic() - t_start
    numbers = {
        "notebook_run_id": notebook_run_id,
        "duration_s": duration_s,
        "duration": f"{int(duration_s // 60)}m {int(duration_s % 60):02d}s",
        "tier": tier,
        "config": {
            "source": "single non-homogeneous Poisson spike train",
            "t_ms": cfg["t_ms"],
            "n_seeds": cfg["n_seeds"],
            "n_m": cfg["n_m"],
            "dt_ms": DT_MS,
            "lambda0_hz": LAMBDA0_HZ,
            "f_stationary_hz": F_STATIONARY_HZ,
            "f_drift_hz": list(F_DRIFT_HZ),
            "max_lag_ms": MAX_LAG_MS,
            "bin_ms": BIN_MS,
        },
        "m_values": [float(x) for x in m_values],
        "spectrum": {
            kind: {
                met: {
                    "mean": [float(x) for x in sweep[kind][met][0]],
                    "std": [float(x) for x in sweep[kind][met][1]],
                }
                for met in SCALARS
            }
            for kind in CONDITIONS
        },
        "references": references,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(numbers, indent=2, default=float))
    persist_run_id(SLUG, notebook_run_id)
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"\nTotal runtime: {numbers['duration']}")


if __name__ == "__main__":
    main()
