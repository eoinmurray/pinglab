"""Notebook runner for entry 055 — spikes to the conductance field.

Shows the transform every other entry takes for granted: how a set of
per-cell input spike trains becomes the excitatory conductance field
g_e(i, t) that actually drives the network. The synapse is a linear
filter — each spike injects a quantum W that decays with tau_AMPA — so
the field is the spike train convolved with a causal exponential kernel
(decay-then-add). No network, no recurrence: just the input pathway.

Three figures, same cells and time window so they line up:
    1. raster      — the input spike trains (discrete events).
    2. field       — g_e(i, t) as a heatmap (each spike a decaying streak).
    3. single_cell — one cell's spikes and the conductance they build,
                     the convolution made explicit.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cli import theme  # noqa: E402
from helpers.modal import parse_modal_gpu  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402

SLUG = "nb055"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

# Hardcoded knobs (the recipe). These match the model's AMPA synapse.
DT_MS = 0.25            # integration step (ms)
TAU_AMPA_MS = 2.0       # AMPA decay — the synaptic kernel time constant (ms)
WEIGHT_US = 0.05        # W: peak conductance injected per input spike (μS)
RATE_HZ = 80.0          # per-cell Poisson input rate (Hz)
SEED = 7

# Tiers only set the figure extent; the job is a trivial numpy convolution.
TIER_CONFIG = {
    "tiny":   {"n_cells": 40,  "window_ms": 200},
    "small":  {"n_cells": 60,  "window_ms": 300},
    "medium": {"n_cells": 100, "window_ms": 400},
}
DEFAULT_TIER = "small"

FIGSIZE = (8, 4.5)  # 16:9


def make_field(n_cells: int, window_ms: float):
    """Generate per-cell Poisson spikes and filter them into g_e(i, t)."""
    rng = np.random.default_rng(SEED)
    n_steps = int(round(window_ms / DT_MS))
    p = RATE_HZ * DT_MS / 1000.0
    spikes = (rng.random((n_steps, n_cells)) < p).astype(np.float64)  # (T, N)
    # Synaptic filter: g[t] = decay·g[t-1] + W·spike[t]  (decay-then-add).
    decay = float(np.exp(-DT_MS / TAU_AMPA_MS))
    g = lfilter([WEIGHT_US], [1.0, -decay], spikes, axis=0)
    t_ms = np.arange(n_steps) * DT_MS
    return spikes, g, t_ms


def plot_raster(spikes, t_ms, dst: Path):
    n = spikes.shape[1]
    events = [t_ms[spikes[:, i] > 0] for i in range(n)]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.eventplot(events, colors=theme.INK_BLACK, lineoffsets=np.arange(n),
                 linelengths=0.8, linewidths=0.7)
    ax.set_xlim(t_ms[0], t_ms[-1])
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("input cell")
    ax.set_title(f"Input spikes — {n} independent Poisson trains at {RATE_HZ:.0f} Hz")
    fig.savefig(dst)
    plt.close(fig)


def plot_field(g, t_ms, dst: Path):
    n = g.shape[1]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(g.T, aspect="auto", origin="lower",
                   extent=[t_ms[0], t_ms[-1], -0.5, n - 0.5],
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label(r"$g_e$ (μS)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("cell")
    ax.set_title(r"Conductance field $g_e(i,t)$ — spikes filtered by the AMPA synapse"
                 f" (τ = {TAU_AMPA_MS:.0f} ms)")
    fig.savefig(dst)
    plt.close(fig)


def plot_single_cell(spikes, g, t_ms, dst: Path, cell: int = 0):
    s_i = spikes[:, cell]
    g_i = g[:, cell]
    fig, (ax_s, ax_g) = plt.subplots(
        2, 1, figsize=FIGSIZE, sharex=True,
        gridspec_kw={"height_ratios": [1, 2.4]},
    )
    ax_s.eventplot(t_ms[s_i > 0], colors=theme.INK_BLACK,
                   lineoffsets=0, linelengths=1.0, linewidths=0.9)
    ax_s.set_yticks([])
    ax_s.set_ylabel("spikes")
    ax_s.set_title(f"One cell: each spike injects W = {WEIGHT_US:g} μS, decays with τ = {TAU_AMPA_MS:.0f} ms")

    ax_g.plot(t_ms, g_i, color=theme.INK_BLACK, lw=1.4)
    ax_g.fill_between(t_ms, g_i, color=theme.INK_BLACK, alpha=0.10)
    ax_g.set_xlim(t_ms[0], t_ms[-1])
    ax_g.set_ylim(bottom=0)
    ax_g.set_xlabel("time (ms)")
    ax_g.set_ylabel(r"$g_e(t)$ (μS)")
    fig.savefig(dst)
    plt.close(fig)


def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    parse_modal_gpu(sys.argv)  # CPU-only numpy job; modal unused.
    wipe_dir = "--no-wipe-dir" not in sys.argv
    cfg = TIER_CONFIG[tier]

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id} tier={tier}")
    prepare_run_dirs(SLUG, notebook_run_id, wipe=wipe_dir, make_artifacts=False)

    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"  # keep the saved 16:9 exact

    spikes, g, t_ms = make_field(cfg["n_cells"], cfg["window_ms"])

    plot_raster(spikes, t_ms, FIGURES / "raster.png")
    plot_field(g, t_ms, FIGURES / "field.png")
    plot_single_cell(spikes, g, t_ms, FIGURES / "single_cell.png")
    for name in ("raster.png", "field.png", "single_cell.png"):
        print(f"wrote {FIGURES / name}")

    # Sanity: empirical vs Campbell-theorem moments for a Poisson drive.
    mu_emp = float(g.mean())
    mu_theory = WEIGHT_US * RATE_HZ * (TAU_AMPA_MS / 1000.0)
    snr_emp = float(g.mean(0).mean() / g.std(0).mean())
    snr_theory = float(np.sqrt(2 * RATE_HZ * (TAU_AMPA_MS / 1000.0)))
    print(f"mean g_e: empirical {mu_emp:.5f} μS vs Campbell {mu_theory:.5f} μS")
    print(f"per-cell SNR: empirical {snr_emp:.3f} vs sqrt(2rho.tau) {snr_theory:.3f}")
    print(f"done in {time.monotonic() - t_start:.1f}s")


if __name__ == "__main__":
    main()
