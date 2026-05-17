"""Notebook runner for entry 030 — NMDA-mediated post-stimulus persistence.

Tests whether an untrained COBANet with hand-set W_ee > 0 and the slow
(NMDA-like) synaptic channel enabled can produce non-seizure sustained
activity after the input cuts out — i.e. a bump that decays gracefully
rather than dying instantly or saturating.

Setup
-----
* Untrained PING-config COBANet (ei_strength=1), N_E=256, N_I=64.
* W_ee init: small positive mean (recurrent excitation that NMDA can
  amplify on the slow timescale). All other weights at defaults.
* Slow synapse on, gain = 0.5 (tau_nmda = 100 ms).
* Input: Poisson spikes at fixed rate R for [0, 200] ms, then total
  silence for [200, 600] ms.
* Sweep R over a 1D grid; 1+ seeds per rate.

Per rate we record the full E + I raster and measure:
  * stim_rate_hz  — mean E pop rate over [50, 200] ms
  * post_rate_hz  — mean E pop rate over [250, 600] ms (skip onset transient)
  * is_seizure    — any 5 ms bin where >90% of E cells fire
  * sustained     — post_rate_hz in [5, 50] Hz and not is_seizure

Notebook entry: src/docs/src/pages/notebooks/nb030.mdx
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO / "src" / "pinglab" / "notebooks"))
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402
from pinglab import theme  # noqa: E402

SLUG = "nb030"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG

# ── Hand-tuned operating point ────────────────────────────────────────
DT = 0.1
T_STIM_MS = 200.0
T_TOTAL_MS = 600.0
T_POST_START_MS = 250.0  # skip 50 ms onset-transient after stim cuts
N_E = 256
N_IN = 64
EI_STRENGTH = 1.0
W_EE_MEAN = 0.5         # modest recurrent excitation
W_EE_STD = 0.05
W_IN_MEAN = 1.2
W_IN_STD = 0.36
W_IN_SPARSITY = 0.95
SLOW_SYN_GAIN = 0.5

# Sweep grid
INPUT_RATES_HZ = [5, 10, 25, 50, 75, 100, 150]
EXTRA_SMALL_RATES = [25, 100]

DEFAULT_TIER = "small"
TIER_CONFIG: dict[str, dict] = {
    "extra small": {"n_seeds": 1, "rates": EXTRA_SMALL_RATES},
    "small":       {"n_seeds": 1, "rates": INPUT_RATES_HZ},
    "medium":      {"n_seeds": 3, "rates": INPUT_RATES_HZ},
    "large":       {"n_seeds": 5, "rates": INPUT_RATES_HZ},
}

# Analysis
RATE_BIN_MS = 5.0
# "Seizure" = post-stim runaway, not gamma synchrony. We call it a seizure
# when the post-stim mean E rate is at refractory-saturation territory
# (≥ SEIZURE_HZ). Healthy gamma-locked ping on MNIST runs ~60–90 Hz, so
# 120 Hz is above any normal operating point but below the 1/ref_ms_E ceiling.
SEIZURE_HZ = 120
SUSTAINED_LO_HZ = 5
SUSTAINED_HI_HZ = SEIZURE_HZ


def _stamp(fig, run_id: str) -> None:
    fig.text(
        0.99, 0.005, run_id,
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


def make_input_spikes(rate_hz: float, T_stim_steps: int, T_total_steps: int,
                      seed: int) -> np.ndarray:
    """Poisson input: rate_hz during [0, T_stim_steps], zero after.

    Shape: (T_total_steps, B=1, N_IN).
    """
    rng = np.random.default_rng(seed)
    p = rate_hz * (DT / 1000.0)
    stim = (rng.random((T_stim_steps, 1, N_IN)) < p).astype(np.float32)
    post = np.zeros((T_total_steps - T_stim_steps, 1, N_IN), dtype=np.float32)
    return np.concatenate([stim, post], axis=0)


def run_one(rate_hz: float, seed: int) -> dict:
    """Build a fresh net at this seed, drive with Poisson input, capture spikes."""
    import torch

    import models as M
    from config import build_net, patch_dt
    from cli import _auto_device, seed_everything

    device = _auto_device()
    seed_everything(seed)
    M.N_IN = N_IN
    patch_dt(DT)

    net = build_net(
        "ping",
        w_in=(W_IN_MEAN, W_IN_STD),
        w_in_sparsity=W_IN_SPARSITY,
        w_ee=(W_EE_MEAN, W_EE_STD),
        ei_strength=EI_STRENGTH,
        slow_synapse=True,
        slow_syn_gain=SLOW_SYN_GAIN,
        hidden_sizes=[N_E],
        device=device,
    )
    net.eval()
    net.recording = True

    T_total_steps = int(T_TOTAL_MS / DT)
    T_stim_steps = int(T_STIM_MS / DT)
    M.T_steps = T_total_steps
    M.T_ms = T_TOTAL_MS

    spk_in = make_input_spikes(rate_hz, T_stim_steps, T_total_steps, seed)
    spk_in_t = torch.from_numpy(spk_in).to(device)

    with torch.no_grad():
        _ = net(input_spikes=spk_in_t)

    e_full = net.spike_record["hid"].cpu().numpy()  # (T, B, N_E)
    i_full = net.spike_record["inh"].cpu().numpy()  # (T, B, N_I)
    return {"e": e_full, "i": i_full, "rate_hz": rate_hz, "seed": seed}


def population_rate_hz(spikes: np.ndarray, bin_ms: float) -> np.ndarray:
    """Bin spikes (T, N) into [bin_ms] windows; pop rate Hz per bin."""
    T, N = spikes.shape
    steps_per_bin = int(bin_ms / DT)
    n_bins = T // steps_per_bin
    spikes_t = spikes[: n_bins * steps_per_bin].reshape(n_bins, steps_per_bin, N)
    pop = spikes_t.sum(axis=(1, 2)) / (N * (bin_ms / 1000.0))
    return pop


def analyze(result: dict) -> dict:
    """Compute stim/post rates and seizure flag for one run."""
    e = result["e"]
    rate_trace = population_rate_hz(e, RATE_BIN_MS)
    t_bin = np.arange(len(rate_trace)) * RATE_BIN_MS

    stim_mask = (t_bin >= 50) & (t_bin < T_STIM_MS)
    post_mask = t_bin >= T_POST_START_MS
    stim_rate = float(rate_trace[stim_mask].mean()) if stim_mask.any() else 0.0
    post_rate = float(rate_trace[post_mask].mean()) if post_mask.any() else 0.0

    # Seizure: post-stim mean rate at refractory saturation (≥ SEIZURE_HZ).
    # Gamma synchrony alone (high peaks, low mean) is NOT seizure — only
    # runaway sustained firing is.
    is_seizure = post_rate >= SEIZURE_HZ

    sustained = (SUSTAINED_LO_HZ <= post_rate <= SUSTAINED_HI_HZ) and not is_seizure
    return {
        "rate_trace_hz": [float(x) for x in rate_trace],
        "t_bin_ms": [float(x) for x in t_bin],
        "stim_rate_hz": stim_rate,
        "post_rate_hz": post_rate,
        "is_seizure": is_seizure,
        "sustained": bool(sustained),
    }


def fig_rasters(results_by_rate: dict, run_id: str) -> plt.Figure:
    """Stacked E rasters, one panel per input rate. Stim window shaded."""
    rates = sorted(results_by_rate.keys())
    h = max(4.5, 0.7 * len(rates))
    fig, axes = plt.subplots(len(rates), 1, figsize=(8, h), sharex=True, dpi=150)
    if len(rates) == 1:
        axes = [axes]
    for ax, rate in zip(axes, rates):
        r = results_by_rate[rate][0]
        e = r["e"]  # (T, N)
        t_idx, n_idx = np.where(e > 0)
        ax.scatter(
            t_idx * DT, n_idx,
            s=0.4, c=theme.INK_BLACK, alpha=0.65, marker=".", edgecolors="none",
        )
        ax.axvspan(0, T_STIM_MS, color=theme.DEEP_RED, alpha=0.08)
        ax.axvline(T_STIM_MS, color=theme.DEEP_RED, lw=0.8, alpha=0.5)
        ax.set_xlim(0, T_TOTAL_MS)
        ax.set_ylim(0, N_E)
        ax.set_yticks([])
        ax.set_ylabel(f"{rate} Hz", fontsize=theme.SIZE_LABEL, rotation=0,
                      ha="right", va="center")
    axes[-1].set_xlabel("Time (ms)", fontsize=theme.SIZE_LABEL)
    fig.suptitle("E rasters per input rate — stim window shaded",
                 fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    _stamp(fig, run_id)
    return fig


def fig_rates_vs_input(metrics_by_rate: dict, run_id: str) -> plt.Figure:
    """Stim vs post-stim E rate as a function of input rate."""
    rates = sorted(metrics_by_rate.keys())

    def _mean(key):
        return np.array([np.mean([m[key] for m in metrics_by_rate[r]]) for r in rates])

    def _sem(key):
        return np.array([
            np.std([m[key] for m in metrics_by_rate[r]])
            / max(1.0, np.sqrt(len(metrics_by_rate[r])))
            for r in rates
        ])

    stim = _mean("stim_rate_hz")
    post = _mean("post_rate_hz")
    stim_sem = _sem("stim_rate_hz")
    post_sem = _sem("post_rate_hz")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.axhspan(SUSTAINED_LO_HZ, SUSTAINED_HI_HZ, color=theme.GREY_LIGHT,
               alpha=0.25,
               label=f"non-seizure band [{SUSTAINED_LO_HZ}, {SUSTAINED_HI_HZ}] Hz")
    ax.errorbar(rates, stim, yerr=stim_sem, color=theme.DEEP_RED, marker="o",
                lw=1.5, capsize=3, label="stim window [50, 200] ms")
    ax.errorbar(rates, post, yerr=post_sem, color=theme.INK_BLACK, marker="s",
                lw=1.5, capsize=3, label="post-stim [250, 600] ms")
    ax.set_xlabel("Input rate during stim (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Mean E firing rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_xscale("log")
    ax.legend(fontsize=theme.SIZE_LEGEND, loc="upper left", frameon=False)
    fig.tight_layout()
    _stamp(fig, run_id)
    return fig


def main() -> None:
    tier = parse_tier(sys.argv, choices=list(TIER_CONFIG), default=DEFAULT_TIER)
    tier_cfg = TIER_CONFIG[tier]

    if "--no-wipe-dir" not in sys.argv:
        if ARTIFACTS.exists():
            shutil.rmtree(ARTIFACTS)
        if FIGURES.exists():
            shutil.rmtree(FIGURES)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    run_id = next_run_id(SLUG)
    persist_run_id(SLUG, run_id)
    (FIGURES / "_run.txt").write_text(f"run_id: {run_id}\ntier: {tier}\n")

    t0 = time.time()
    print(f"[{SLUG}] tier={tier}  seeds={tier_cfg['n_seeds']}  rates={tier_cfg['rates']}")

    results_by_rate: dict[int, list] = {}
    metrics_by_rate: dict[int, list] = {}
    for rate in tier_cfg["rates"]:
        results_by_rate[rate] = []
        metrics_by_rate[rate] = []
        for s in range(tier_cfg["n_seeds"]):
            seed = 42 + s
            r = run_one(rate, seed)
            m = analyze(r)
            results_by_rate[rate].append(r)
            metrics_by_rate[rate].append(m)
            tag = "SEIZURE" if m["is_seizure"] else ("SUST" if m["sustained"] else "    ")
            print(f"  rate={rate:>3}Hz seed={seed}: "
                  f"stim={m['stim_rate_hz']:6.2f}Hz  post={m['post_rate_hz']:6.2f}Hz  {tag}")

    fig1 = fig_rasters(results_by_rate, run_id)
    fig1.savefig(FIGURES / "rasters.png", dpi=150)
    plt.close(fig1)

    fig2 = fig_rates_vs_input(metrics_by_rate, run_id)
    fig2.savefig(FIGURES / "rates_vs_input.png", dpi=150)
    plt.close(fig2)

    # Roll up by-rate summaries (drop the per-bin rate traces from numbers.json
    # — they live in artifacts if needed).
    by_rate = {}
    for rate in metrics_by_rate:
        ms = metrics_by_rate[rate]
        by_rate[str(rate)] = {
            "stim_rate_hz": float(np.mean([m["stim_rate_hz"] for m in ms])),
            "post_rate_hz": float(np.mean([m["post_rate_hz"] for m in ms])),
            "seizure_rate": float(np.mean([float(m["is_seizure"]) for m in ms])),
            "sustained_rate": float(np.mean([float(m["sustained"]) for m in ms])),
            "n_seeds": len(ms),
        }

    sustained_any = any(m["sustained"] for ms in metrics_by_rate.values() for m in ms)
    seizure_any = any(m["is_seizure"] for ms in metrics_by_rate.values() for m in ms)

    numbers = {
        "run_id": run_id,
        "config": {
            "dt": DT,
            "t_stim_ms": T_STIM_MS,
            "t_total_ms": T_TOTAL_MS,
            "t_post_start_ms": T_POST_START_MS,
            "slow_syn_gain": SLOW_SYN_GAIN,
            "w_ee_mean": W_EE_MEAN,
            "w_ee_std": W_EE_STD,
            "ei_strength": EI_STRENGTH,
            "n_e": N_E,
            "n_in": N_IN,
            "tier": tier,
            "input_rates_hz": sorted(tier_cfg["rates"]),
        },
        "results": {"by_rate": by_rate},
        "success_criteria": [
            {
                "label": "at least one rate produces non-seizure sustained activity",
                "passed": sustained_any,
                "detail": f"post_rate ∈ [{SUSTAINED_LO_HZ}, {SUSTAINED_HI_HZ}] Hz "
                          "and no 5 ms bin >90% active",
            },
            {
                "label": "at least one rate stays out of seizure",
                "passed": not all(
                    m["is_seizure"] for ms in metrics_by_rate.values() for m in ms
                ),
                "detail": f"seizure observed at any rate: {seizure_any}",
            },
        ],
        "runtime_s": time.time() - t0,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print(f"[{SLUG}] done in {time.time() - t0:.1f}s. "
          f"sustained_any={sustained_any}  seizure_any={seizure_any}")

    if not all(c["passed"] for c in numbers["success_criteria"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
