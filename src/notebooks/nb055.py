"""Notebook runner for entry 055 — spikes to the conductance field.

Shows the transform every other entry takes for granted: how a set of
per-cell input spike trains becomes the excitatory conductance field
g_e(i, t) that actually drives the network. The synapse is a linear
filter — each spike injects a quantum W that decays with tau_AMPA — so
the field is the spike train convolved with a causal exponential kernel
(decay-then-add). No network, no recurrence: just the input pathway.

Figure 1 (compound.png): the population transform — input raster over the
conductance field g_e(i, t) heatmap (shared time axis), and the same filter
unrolled for one representative cell (spikes over conductance).

Figure 2 (regularisation.png): the first genuine dissociation. Swap Poisson
for a Gamma renewal process of shape k at matched rate ρ. Strength μ stays
put (it only sees the rate); σ drops as 1/sqrt(k); SNR rises as sqrt(k).
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

FIGSIZE = (12, 6.75)  # 16:9, wider to hold the two-column compound


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


def make_renewal_field(n_cells: int, window_ms: float, k_shape: float,
                       tau_ms: float = TAU_AMPA_MS, seed_offset: int = 0):
    """Gamma renewal spike trains at rate RATE_HZ, filtered with τ = tau_ms.

    Mean ISI = 1/RATE_HZ for every k (matched rate). ISI CV = 1/sqrt(k):
    k=1 recovers Poisson; large k tightens toward a periodic train. The
    long-window spike-count Fano factor is 1/k, so in the slow-kernel
    regime ρτ ≫ 1 the conductance SNR rises as sqrt(2 ρ τ k). When
    ρτ ≪ 1 the kernel sees at most one spike at a time and regularising
    has no effect on σ — the AMPA kernel at ρ = 80 Hz sits in that
    regime, which is why this figure uses a slower kernel.
    """
    rng = np.random.default_rng(SEED + 101 + seed_offset)
    n_steps = int(round(window_ms / DT_MS))
    dt_s = DT_MS / 1000.0
    window_s = window_ms / 1000.0
    pad_s = max(0.2, 50.0 / RATE_HZ)  # long enough to forget the renewal edge
    span_s = window_s + 2.0 * pad_s
    scale = 1.0 / (RATE_HZ * k_shape)  # so mean ISI = k · scale = 1/ρ
    spikes = np.zeros((n_steps, n_cells), dtype=np.float64)
    # Draw ISIs with generous margin; cumsum; keep events inside [pad, pad+T).
    n_draw = int(2.5 * RATE_HZ * span_s) + 64
    for c in range(n_cells):
        isis = rng.gamma(shape=k_shape, scale=scale, size=n_draw)
        # Top up the rare case where the cumulative sum falls short of span_s.
        while isis.sum() < span_s:
            isis = np.concatenate(
                [isis, rng.gamma(shape=k_shape, scale=scale, size=n_draw)]
            )
        t_abs = np.cumsum(isis)
        in_win = (t_abs >= pad_s) & (t_abs < pad_s + window_s)
        idx = ((t_abs[in_win] - pad_s) / dt_s).astype(np.int64)
        idx = idx[(idx >= 0) & (idx < n_steps)]
        np.add.at(spikes[:, c], idx, 1.0)
    decay = float(np.exp(-DT_MS / tau_ms))
    g = lfilter([WEIGHT_US], [1.0, -decay], spikes, axis=0)
    t_ms = np.arange(n_steps) * DT_MS
    return spikes, g, t_ms


def _stats(g: np.ndarray, burn_ms: float = 40.0):
    """Per-cell mean / sd / snr over a window past the renewal transient."""
    burn = int(round(burn_ms / DT_MS))
    gg = g[burn:]
    mu = float(gg.mean())
    # Temporal SD per cell, then average across cells (matches Figure 1).
    sd = float(gg.std(axis=0).mean())
    return mu, sd, mu / sd


def plot_compound(spikes, g, t_ms, dst: Path, cell: int | None = None):
    """One 16:9 figure: population transform (left) and one-cell filter (right).

    Left column  — input raster over the conductance-field heatmap, sharing a
                   time axis, with a slim colourbar row beneath the field.
    Right column — the highlighted cell's spikes over the conductance they
                   build, the convolution made explicit (one row of the field).
    """
    n = spikes.shape[1]
    if cell is None:
        cell = n // 2  # a central, representative row for the right-hand zoom
    t0, t1 = t_ms[0], t_ms[-1]

    fig = plt.figure(figsize=FIGSIZE)
    sf_l, sf_r = fig.subfigures(1, 2, width_ratios=[1.3, 1.0], wspace=0.04)

    # --- Left: population spikes -> field, shared time axis, colourbar row.
    gs_l = sf_l.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.045], hspace=0.42)
    ax_rast = sf_l.add_subplot(gs_l[0])
    ax_field = sf_l.add_subplot(gs_l[1], sharex=ax_rast)
    ax_cbar = sf_l.add_subplot(gs_l[2])

    events = [t_ms[spikes[:, i] > 0] for i in range(n)]
    ax_rast.eventplot(events, colors=theme.INK_BLACK, lineoffsets=np.arange(n),
                      linelengths=0.8, linewidths=0.6)
    ax_rast.set_ylim(-0.5, n - 0.5)
    ax_rast.set_ylabel("input cell")
    ax_rast.set_title(f"Input spikes — {n} Poisson trains at {RATE_HZ:.0f} Hz")
    ax_rast.tick_params(labelbottom=False)

    im = ax_field.imshow(g.T, aspect="auto", origin="lower",
                         extent=[t0, t1, -0.5, n - 0.5], interpolation="nearest")
    ax_field.set_xlim(t0, t1)
    ax_field.set_ylabel("cell")
    ax_field.set_xlabel("time (ms)")
    ax_field.set_title(r"Conductance field $g_e(i,t)$ — input filtered by the AMPA synapse")

    cb = fig.colorbar(im, cax=ax_cbar, orientation="horizontal")
    cb.set_label(r"$g_e$ (μS)")

    # --- Right: the filter for the highlighted cell, shared time axis.
    gs_r = sf_r.add_gridspec(2, 1, height_ratios=[0.5, 1.5], hspace=0.12)
    ax_s = sf_r.add_subplot(gs_r[0])
    ax_g = sf_r.add_subplot(gs_r[1], sharex=ax_s)

    s_i, g_i = spikes[:, cell], g[:, cell]
    ax_s.eventplot(t_ms[s_i > 0], colors=theme.INK_BLACK,
                   lineoffsets=0, linelengths=1.0, linewidths=0.9)
    ax_s.set_yticks([])
    ax_s.set_ylabel("spikes")
    ax_s.set_title(f"Cell {cell}: spike → W = {WEIGHT_US:g} μS, decay τ = {TAU_AMPA_MS:.0f} ms")
    ax_s.tick_params(labelbottom=False)

    ax_g.plot(t_ms, g_i, color=theme.INK_BLACK, lw=1.4)
    ax_g.fill_between(t_ms, g_i, color=theme.INK_BLACK, alpha=0.10)
    ax_g.set_xlim(t0, t1)
    ax_g.set_ylim(bottom=0)
    ax_g.set_xlabel("time (ms)")
    ax_g.set_ylabel(r"$g_e(t)$ (μS)")

    fig.savefig(dst)
    plt.close(fig)


# Regularisation sweep: shape parameters for the Gamma renewal process.
# k=1 is Poisson; larger k tightens ISIs toward periodic. Log-2 ladder.
K_SHAPES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
K_QUAL = (1.0, 64.0)  # low / high regularity for the trace overlay
# Slower kernel for Figure 2's demonstration. The article's AMPA τ = 2 ms
# sits at ρτ ≈ 0.16 — kernel sees at most one spike at a time, so
# regularising input has essentially no effect on σ. Bumping to τ = 50 ms
# (think NMDA, or the temporal integration of a longer membrane window)
# puts the kernel at ρτ = 4, where it averages many ISIs and the Fano
# reduction is visible. The dissociation is the point; the regime caveat
# is genuine and called out in the writeup.
TAU_DEMO_MS = 50.0


def plot_regularisation_sweep(dst: Path, window_ms: float, n_cells_sweep: int):
    """Figure 2 — strength held, reliability moves.

    Left: input spikes and g_e(t) for one cell at k=1 (Poisson) and k=64
    (nearly periodic), same rate ρ, same Campbell mean μ = W ρ τ — the
    trace tightens around the same line.
    Right: μ, σ, SNR as functions of the Gamma shape k, with theory lines
    μ = W ρ τ (flat), σ = W sqrt(ρ τ / (2k)), SNR = sqrt(2 ρ τ k). Demo
    kernel τ = TAU_DEMO_MS — the AMPA regime is in the suppressed branch.
    """
    rho = RATE_HZ
    tau_s = TAU_DEMO_MS / 1000.0
    mu_th = WEIGHT_US * rho * tau_s
    burn_ms = max(40.0, 4.0 * TAU_DEMO_MS)
    # Sweep.
    mus, sds, snrs = [], [], []
    qual_spikes: dict[float, np.ndarray] = {}
    qual_g: dict[float, np.ndarray] = {}
    t_ms = None
    for ki, k in enumerate(K_SHAPES):
        spikes, g, t_ms_k = make_renewal_field(
            n_cells_sweep, window_ms, k, tau_ms=TAU_DEMO_MS, seed_offset=ki,
        )
        mu, sd, snr = _stats(g, burn_ms=burn_ms)
        mus.append(mu); sds.append(sd); snrs.append(snr)
        if k in K_QUAL:
            qual_spikes[k] = spikes[:, 0]
            qual_g[k] = g[:, 0]
            t_ms = t_ms_k
    mus = np.asarray(mus); sds = np.asarray(sds); snrs = np.asarray(snrs)
    ks = np.asarray(K_SHAPES, dtype=float)
    k_dense = np.geomspace(ks[0], ks[-1], 64)
    sd_th = WEIGHT_US * np.sqrt(rho * tau_s / (2.0 * k_dense))
    snr_th = np.sqrt(2.0 * rho * tau_s * k_dense)

    fig = plt.figure(figsize=FIGSIZE)
    sf_l, sf_r = fig.subfigures(1, 2, width_ratios=[1.15, 1.0], wspace=0.06)

    # --- Left: trace overlay for two k values, sharing a time axis.
    gs_l = sf_l.add_gridspec(2, 1, hspace=0.35)
    colors = {K_QUAL[0]: theme.INK_BLACK, K_QUAL[1]: theme.DEEP_RED}
    axes_l = [sf_l.add_subplot(gs_l[0]), sf_l.add_subplot(gs_l[1])]
    window_show = min(600.0, float(t_ms[-1] - burn_ms))
    t_lo = burn_ms
    t_hi = burn_ms + window_show
    ymax = max(np.max(qual_g[k]) for k in K_QUAL) * 1.05
    for ax, k in zip(axes_l, K_QUAL):
        s = qual_spikes[k]; ge = qual_g[k]; c = colors[k]
        in_win = (t_ms >= t_lo) & (t_ms <= t_hi)
        t_view = t_ms[in_win] - t_lo
        ge_view = ge[in_win]
        spk_times = (t_ms[(s > 0) & in_win]) - t_lo
        ax.eventplot(spk_times, colors=c, lineoffsets=ymax * 1.04,
                     linelengths=ymax * 0.10, linewidths=0.7)
        ax.plot(t_view, ge_view, color=c, lw=1.3)
        ax.fill_between(t_view, ge_view, color=c, alpha=0.10)
        ax.axhline(mu_th, color=theme.GREY_MID, lw=0.9, ls="--", zorder=0)
        cv = 1.0 / np.sqrt(k)
        tag = "k = 1 (Poisson)" if k == 1.0 else f"k = {int(k)} (ISI CV ≈ {cv:.2f})"
        ax.set_title(tag, loc="left", color=c)
        ax.set_xlim(0.0, window_show)
        ax.set_ylim(0.0, ymax * 1.15)
        ax.set_ylabel(r"$g_e(t)$ (μS)")
    axes_l[0].tick_params(labelbottom=False)
    axes_l[1].set_xlabel("time (ms)")

    # --- Right: μ, σ, SNR vs k with theory overlays.
    gs_r = sf_r.add_gridspec(3, 1, hspace=0.6)
    ax_mu = sf_r.add_subplot(gs_r[0])
    ax_sd = sf_r.add_subplot(gs_r[1])
    ax_sn = sf_r.add_subplot(gs_r[2])
    for ax in (ax_mu, ax_sd, ax_sn):
        ax.set_xscale("log")
    ax_mu.axhline(mu_th, color=theme.GREY_MID, ls="--", lw=1.0,
                  label=r"$W\rho\tau$")
    ax_mu.plot(ks, mus, "o-", color=theme.INK_BLACK, lw=1.2, ms=4)
    ax_mu.set_ylabel(r"$\mu$ (μS)")
    ax_mu.set_ylim(0.0, max(1.6 * mu_th, mus.max() * 1.4))
    ax_mu.set_title("Strength is unchanged", loc="left", fontsize=10,
                    color=theme.MUTED)
    ax_mu.legend(frameon=False, fontsize=8, loc="lower right")
    ax_mu.tick_params(labelbottom=False)

    ax_sd.set_yscale("log")
    ax_sd.plot(k_dense, sd_th, color=theme.GREY_MID, ls="--", lw=1.0,
               label=r"$W\sqrt{\rho\tau/(2k)}$")
    ax_sd.plot(ks, sds, "o-", color=theme.INK_BLACK, lw=1.2, ms=4)
    ax_sd.set_ylabel(r"$\sigma$ (μS)")
    ax_sd.set_title(r"Fluctuation falls as $1/\sqrt{k}$", loc="left",
                    fontsize=10, color=theme.MUTED)
    ax_sd.legend(frameon=False, fontsize=8, loc="lower left")
    ax_sd.tick_params(labelbottom=False)

    ax_sn.set_yscale("log")
    ax_sn.plot(k_dense, snr_th, color=theme.GREY_MID, ls="--", lw=1.0,
               label=r"$\sqrt{2\rho\tau\,k}$")
    ax_sn.plot(ks, snrs, "o-", color=theme.DEEP_RED, lw=1.2, ms=4)
    ax_sn.set_ylabel("SNR")
    ax_sn.set_xlabel("Gamma shape k")
    ax_sn.set_title(r"Reliability rises as $\sqrt{k}$", loc="left",
                    fontsize=10, color=theme.MUTED)
    ax_sn.legend(frameon=False, fontsize=8, loc="lower right")

    fig.savefig(dst)
    plt.close(fig)
    return ks, mus, sds, snrs


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

    plot_compound(spikes, g, t_ms, FIGURES / "compound.png")
    print(f"wrote {FIGURES / 'compound.png'}")

    # Sanity: empirical vs Campbell-theorem moments for a Poisson drive.
    mu_emp = float(g.mean())
    mu_theory = WEIGHT_US * RATE_HZ * (TAU_AMPA_MS / 1000.0)
    snr_emp = float(g.mean(0).mean() / g.std(0).mean())
    snr_theory = float(np.sqrt(2 * RATE_HZ * (TAU_AMPA_MS / 1000.0)))
    print(f"mean g_e: empirical {mu_emp:.5f} μS vs Campbell {mu_theory:.5f} μS")
    print(f"per-cell SNR: empirical {snr_emp:.3f} vs sqrt(2rho.tau) {snr_theory:.3f}")

    # Figure 2: matched-rate regularisation sweep.
    sweep_window_ms = max(2000.0, 8.0 * cfg["window_ms"])  # longer for steady-state stats
    n_cells_sweep = max(8, cfg["n_cells"] // 4)
    ks, mus, sds, snrs = plot_regularisation_sweep(
        FIGURES / "regularisation.png", sweep_window_ms, n_cells_sweep
    )
    print(f"wrote {FIGURES / 'regularisation.png'}")
    print(f"sweep window {sweep_window_ms:.0f} ms, {n_cells_sweep} cells per k")
    for k, mu, sd, snr in zip(ks, mus, sds, snrs):
        snr_pred = float(np.sqrt(2.0 * RATE_HZ * TAU_DEMO_MS / 1000.0 * k))
        print(f"  k={k:7.2f}  mu={mu:.5f}  sigma={sd:.5f}  SNR={snr:.3f}  (asym pred {snr_pred:.3f})")

    print(f"done in {time.monotonic() - t_start:.1f}s")


if __name__ == "__main__":
    main()
