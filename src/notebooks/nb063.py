"""Notebook runner for entry 063 — The Poisson process, pedagogically.

A teaching entry, not a result. Builds the Poisson distribution from the
ground up — Bernoulli coin flips → binomial → Poisson — then shows that
this limit is exactly how the simulator encodes inputs (per-timestep
Bernoulli draws, encoders.encode_images_poisson), and why CV = 1 / Fano = 1
is the "pure randomness" yardstick the rest of the collection measures
departures from.

Four figures, all synthetic + the real encoder; no training, no trained
cells. Safe to run anywhere in seconds.

Notebook entry: src/docs/content/notebooks/nb063.mdx
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, expon, poisson

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from helpers.figsave import save_figure  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402
from cli import theme  # noqa: E402

SLUG = "nb063"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

# Teaching constants. A canonical encoder operating point so the spike-count
# rates land on the same λ ladder used in the distribution panels.
MAX_RATE_HZ = 50.0     # Poisson rate at pixel intensity = 1.0
T_MS = 200.0           # trial length (matches the network's trial duration)
DT_MS = 0.1            # canonical integration timestep
N_TRIALS = 4000        # Monte-Carlo trials for empirical histograms
SEED = 42

TIER_CONFIG = {  # tier only scales the Monte-Carlo sample count
    "extra small": dict(n_trials=400),
    "small": dict(n_trials=2000),
    "medium": dict(n_trials=4000),
    "large": dict(n_trials=8000),
    "extra large": dict(n_trials=16000),
}
DEFAULT_TIER = "medium"

LAMBDAS = (1.0, 4.0, 10.0)
LAMBDA_COLORS = (theme.INK_BLACK, theme.DEEP_RED, theme.AMBER)


# ── Fig 1: coin flips become Poisson ────────────────────────────────
def fig_binomial_to_poisson(out_path: Path, run_id: str) -> None:
    """Fix λ = 4 and let n grow: n Bernoulli coins each with p = λ/n. As the
    events get more numerous and rarer the binomial count converges to
    Poisson(λ) — the limit the encoder lives in (n = timesteps, p = rate·dt)."""
    theme.apply()
    lam = 4.0
    ks = np.arange(0, 14)
    ns = [4, 8, 20, 100]
    fig, axes = plt.subplots(1, len(ns), figsize=(11.0, 3.0), sharey=True)
    pois = poisson.pmf(ks, lam)
    for ax, n in zip(axes, ns):
        ax.bar(ks, binom.pmf(ks, n, lam / n), width=0.8,
               color=theme.GREY_MID, alpha=0.85, label=f"Binomial(n={n})")
        ax.plot(ks, pois, "o-", color=theme.DEEP_RED, lw=1.4, ms=4,
                label="Poisson(λ=4)")
        ax.set_title(f"n = {n},  p = {lam/n:.2g}", fontsize=theme.SIZE_LABEL)
        ax.set_xlabel("count k", fontsize=theme.SIZE_LABEL)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("probability", fontsize=theme.SIZE_LABEL)
    axes[-1].legend(fontsize=theme.SIZE_LEGEND, frameon=False)
    fig.suptitle("Coin flips become Poisson — binomial(n, λ/n) → Poisson(λ) as n grows",
                 fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg",))
    plt.close(fig)


# ── Fig 2: the distribution + mean = variance ───────────────────────
def fig_pmf_and_meanvar(out_path: Path, run_id: str, rng) -> None:
    """Left: the PMF for three λ — bigger λ shifts the peak right and widens
    it. Right: the Poisson signature, mean = variance, shown empirically by
    sampling at many λ and scattering sample variance against sample mean."""
    theme.apply()
    fig, (ax_pmf, ax_mv) = plt.subplots(1, 2, figsize=(9.0, 4.0))
    ks = np.arange(0, 22)
    for lam, c in zip(LAMBDAS, LAMBDA_COLORS):
        ax_pmf.plot(ks, poisson.pmf(ks, lam), "o-", color=c, lw=1.3, ms=4,
                    label=f"λ = {lam:g}")
    ax_pmf.set_xlabel("count k", fontsize=theme.SIZE_LABEL)
    ax_pmf.set_ylabel("P(k)", fontsize=theme.SIZE_LABEL)
    ax_pmf.set_title("Poisson PMF", loc="left", fontweight="semibold")
    ax_pmf.legend(fontsize=theme.SIZE_LEGEND, frameon=False)

    lams = np.linspace(0.5, 20, 25)
    means, varis = [], []
    for lam in lams:
        s = rng.poisson(lam, size=3000)
        means.append(s.mean())
        varis.append(s.var())
    ax_mv.plot([0, 20], [0, 20], color=theme.GREY_MID, lw=1.0, ls="--",
               label="variance = mean")
    ax_mv.scatter(means, varis, s=18, color=theme.DEEP_RED, edgecolors="none",
                  alpha=0.85, label="sampled Poisson")
    ax_mv.set_xlabel("sample mean", fontsize=theme.SIZE_LABEL)
    ax_mv.set_ylabel("sample variance", fontsize=theme.SIZE_LABEL)
    ax_mv.set_title("mean = variance", loc="left", fontweight="semibold")
    ax_mv.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    for ax in (ax_pmf, ax_mv):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("One number, λ, sets both the average and the spread",
                 fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg",))
    plt.close(fig)


# ── Fig 3: spike trains + memoryless intervals ──────────────────────
def fig_spiketrains_isi(out_path: Path, run_id: str, rng) -> None:
    """Left: homogeneous Poisson spike trains at three rates — random in time,
    denser at higher rate. Right: the inter-spike intervals are exponential
    (memoryless), and their coefficient of variation is ≈ 1 — the signature
    of a Poisson process and the reference for 'irregular' spiking."""
    theme.apply()
    fig, (ax_r, ax_isi) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    rates = (10.0, 30.0, 60.0)
    colors = (theme.INK_BLACK, theme.DEEP_RED, theme.AMBER)
    T_s = 1.0  # 1 s window for the raster
    n_per = 8
    row = 0
    yticks, ylabels = [], []
    isi_pool = None
    for rate, c in zip(rates, colors):
        for _ in range(n_per):
            # Exponential ISIs → spike times (the defining construction).
            n_exp = int(rate * T_s * 3 + 20)
            isis = rng.exponential(1.0 / rate, size=n_exp)
            times = np.cumsum(isis)
            times = times[times < T_s]
            ax_r.scatter(times, np.full_like(times, row), s=4, marker="|",
                         color=c, linewidths=0.6)
            row += 1
        yticks.append(row - n_per / 2)
        ylabels.append(f"{rate:g} Hz")
        # Collect a clean ISI sample from the middle rate for the histogram.
        if rate == 30.0:
            long = rng.exponential(1.0 / rate, size=20000)
            isi_pool = long
    ax_r.set_yticks(yticks)
    ax_r.set_yticklabels(ylabels)
    ax_r.set_xlabel("time (s)", fontsize=theme.SIZE_LABEL)
    ax_r.set_title("Poisson spike trains", loc="left", fontweight="semibold")
    ax_r.set_xlim(0, T_s)

    isi_ms = isi_pool * 1000.0
    cv = isi_ms.std() / isi_ms.mean()
    bins = np.linspace(0, isi_ms.mean() * 5, 50)
    ax_isi.hist(isi_ms, bins=bins, density=True, color=theme.GREY_MID,
                alpha=0.8, edgecolor="none", label="ISIs (30 Hz)")
    xs = np.linspace(0, bins[-1], 200)
    ax_isi.plot(xs, expon.pdf(xs, scale=isi_ms.mean()), color=theme.DEEP_RED,
                lw=1.6, label="exponential")
    ax_isi.set_xlabel("inter-spike interval (ms)", fontsize=theme.SIZE_LABEL)
    ax_isi.set_ylabel("density", fontsize=theme.SIZE_LABEL)
    ax_isi.set_title(f"exponential ISIs,  CV = {cv:.2f}", loc="left",
                     fontweight="semibold")
    ax_isi.legend(fontsize=theme.SIZE_LEGEND, frameon=False)
    for ax in (ax_r, ax_isi):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("In continuous time: random spike times, memoryless intervals",
                 fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg",))
    plt.close(fig)


# ── Fig 4: this is the simulator's encoder ──────────────────────────
def fig_encoder_check(out_path: Path, run_id: str, n_trials: int) -> dict:
    """Run the project's own encoder (encoders.encode_images_poisson) on
    constant pixels and confirm the spike counts are Poisson. Left: three
    pixel intensities → λ = intensity·rate·T, empirical counts vs Poisson.
    Right: dt-invariance — the same intensity at dt ∈ {0.1, 0.25, 1.0} ms
    gives the same count distribution because λ = (rate·dt)·(T/dt) is fixed."""
    import torch
    from cli.encoders import encode_images_poisson

    theme.apply()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.0, 4.0))

    def counts_for(intensity: float, dt: float, n: int) -> np.ndarray:
        t_steps = int(round(T_MS / dt))
        g = torch.Generator().manual_seed(SEED)
        px = torch.full((n, 1), float(intensity))
        spk = encode_images_poisson(px, t_steps, dt, MAX_RATE_HZ, generator=g)
        return spk.sum(dim=0).squeeze(-1).numpy()  # (n,) spike counts

    fano = {}
    intensities = (0.25, 0.5, 1.0)
    for intensity, c in zip(intensities, LAMBDA_COLORS):
        cnt = counts_for(intensity, DT_MS, n_trials)
        lam = intensity * MAX_RATE_HZ * T_MS / 1000.0
        fano[f"intensity_{intensity:g}"] = float(cnt.var() / cnt.mean())
        ks = np.arange(0, int(lam * 3) + 4)
        ax_a.hist(cnt, bins=np.arange(-0.5, ks[-1] + 1), density=True,
                  color=c, alpha=0.35, edgecolor="none")
        ax_a.plot(ks, poisson.pmf(ks, lam), "o-", color=c, lw=1.2, ms=3,
                  label=f"intensity {intensity:g} → λ={lam:g}")
    ax_a.set_xlabel("spikes per trial", fontsize=theme.SIZE_LABEL)
    ax_a.set_ylabel("probability", fontsize=theme.SIZE_LABEL)
    ax_a.set_title("encoder output vs Poisson", loc="left", fontweight="semibold")
    ax_a.legend(fontsize=theme.SIZE_LEGEND, frameon=False)

    lam_full = 1.0 * MAX_RATE_HZ * T_MS / 1000.0
    for dt, c in zip((0.1, 0.25, 1.0), LAMBDA_COLORS):
        cnt = counts_for(1.0, dt, n_trials)
        ax_b.hist(cnt, bins=np.arange(-0.5, int(lam_full * 3) + 1), density=True,
                  histtype="step", lw=1.4, color=c, label=f"dt = {dt:g} ms")
    ks = np.arange(0, int(lam_full * 3) + 1)
    ax_b.plot(ks, poisson.pmf(ks, lam_full), color=theme.GREY_MID, lw=1.0,
              ls="--", label=f"Poisson(λ={lam_full:g})")
    ax_b.set_xlabel("spikes per trial", fontsize=theme.SIZE_LABEL)
    ax_b.set_ylabel("probability", fontsize=theme.SIZE_LABEL)
    ax_b.set_title("dt-invariance", loc="left", fontweight="semibold")
    ax_b.legend(fontsize=theme.SIZE_LEGEND, frameon=False)
    for ax in (ax_a, ax_b):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("The input encoder is a Poisson process — pixel intensity sets λ",
                 fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg",))
    plt.close(fig)
    return fano


def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    wipe_dir = "--no-wipe-dir" not in sys.argv
    n_trials = TIER_CONFIG[tier]["n_trials"]

    t_start = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {run_id} tier={tier} n_trials={n_trials}")
    prepare_run_dirs(SLUG, run_id, wipe=wipe_dir, skip_training=True,
                     make_artifacts=False)
    rng = np.random.default_rng(SEED)

    fig_binomial_to_poisson(FIGURES / "binomial_to_poisson", run_id)
    print(f"wrote {FIGURES / 'binomial_to_poisson'}.svg")
    fig_pmf_and_meanvar(FIGURES / "pmf_and_meanvar", run_id, rng)
    print(f"wrote {FIGURES / 'pmf_and_meanvar'}.svg")
    fig_spiketrains_isi(FIGURES / "spike_trains_isi", run_id, rng)
    print(f"wrote {FIGURES / 'spike_trains_isi'}.svg")
    fano = fig_encoder_check(FIGURES / "encoder_check", run_id, n_trials)
    print(f"wrote {FIGURES / 'encoder_check'}.svg")

    summary = {
        "notebook_run_id": run_id,
        "tier": tier,
        "config": {"max_rate_hz": MAX_RATE_HZ, "t_ms": T_MS, "dt_ms": DT_MS,
                   "n_trials": n_trials, "seed": SEED},
        "encoder_fano": fano,
        "duration_s": round(time.monotonic() - t_start, 1),
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}  (encoder Fano factors: {fano})")


if __name__ == "__main__":
    main()
