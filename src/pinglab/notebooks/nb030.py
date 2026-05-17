"""Notebook runner for entry 030 — coba + NMDA cannot sustain post-stim activity.

The recurrent excitation + NMDA story is, with inhibition, what gives a
working-memory layer its attractor. The structural claim this entry
makes concrete: without inhibition (coba, *ei_strength = 0*), the same
slow channel + recurrent E→E that produces a stable attractor in ping
admits only two outcomes — quench or seizure. There is no intermediate
sustained state.

Setup
-----
* Untrained COBANet at *ei_strength = 0* (no I cells driving E, no E→I→E
  loop). N_E = 256.
* Slow channel on, gain = 0.5 (tau_nmda = 100 ms).
* Hand-set W_ee mean swept over a small grid.
* Input: Poisson spikes at rate R during [0, 200] ms, then silence for
  [200, 600] ms.

For every (W_ee, R) cell we classify the outcome by the late-window
mean E rate over [500, 600] ms:

    DECAY     — late_rate < 1 Hz   (silent — the network quenches)
    SUSTAINED — 1 ≤ late_rate < 120 Hz   (the goal — would be a green cell)
    SEIZURE   — late_rate ≥ 120 Hz  (refractory saturation, locked)

The success-criterion check is whether ANY cell on the grid lands in the
sustained class. Result: none does.

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

# ── Fixed network setup ───────────────────────────────────────────────
DT = 0.1
T_STIM_MS = 200.0
T_TOTAL_MS = 600.0
T_LATE_START_MS = 500.0  # late window for outcome classification (last 100 ms)
N_E = 256
N_IN = 64
EI_STRENGTH = 0.0       # coba — no E↔I↔E loop; the structural point being made
W_EE_STD_FRAC = 0.1     # W_ee std = W_ee_mean × this (keep CV constant across grid)
W_IN_MEAN = 0.3         # coba's lighter W_in (matches nb024 coba recipe); a strong
W_IN_STD = 0.09         # ping-tuned W_in would NMDA-charge the network to seizure on
W_IN_SPARSITY = 0.95    # its own, hiding the W_ee effect we're trying to measure.
SLOW_SYN_GAIN = 0.5

# ── 2D sweep grid: W_ee × input rate ──────────────────────────────────
W_EE_MEAN_GRID = [0.0, 0.05, 0.10, 0.25, 0.50, 1.00]
INPUT_RATES_HZ = [5, 25, 50, 100, 150]
# Small-tier subgrids so smoke runs stay quick
EXTRA_SMALL_W_EE = [0.0, 0.50]
EXTRA_SMALL_RATES = [25, 100]

DEFAULT_TIER = "small"
TIER_CONFIG: dict[str, dict] = {
    "extra small": {"n_seeds": 1, "w_ee": EXTRA_SMALL_W_EE, "rates": EXTRA_SMALL_RATES},
    "small":       {"n_seeds": 1, "w_ee": W_EE_MEAN_GRID,   "rates": INPUT_RATES_HZ},
    "medium":      {"n_seeds": 3, "w_ee": W_EE_MEAN_GRID,   "rates": INPUT_RATES_HZ},
    "large":       {"n_seeds": 5, "w_ee": W_EE_MEAN_GRID,   "rates": INPUT_RATES_HZ},
}

# Outcome classification
DECAY_HZ = 1.0          # late-window rate below this: network has quenched
SEIZURE_HZ = 120.0      # late-window rate at/above this: refractory saturation
DECAY_SLOPE_RATIO = 0.5 # late/early post-stim ratio below this counts as
                        # "still decaying" — a passive NMDA tail rather than
                        # a stable attractor.
# Sustained = anywhere between DECAY_HZ and SEIZURE_HZ AND not still actively
# decaying. The claim of this notebook is that for coba (ei_strength=0) this
# class is empty no matter how you tune W_ee.

# Display
RATE_BIN_MS = 5.0


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


def run_one(rate_hz: float, w_ee_mean: float, seed: int) -> dict:
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
        w_ee=(w_ee_mean, w_ee_mean * W_EE_STD_FRAC),
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

    e_full = net.spike_record["hid"].cpu().numpy()
    return {"e": e_full, "rate_hz": rate_hz, "w_ee_mean": w_ee_mean, "seed": seed}


def population_rate_hz(spikes: np.ndarray, bin_ms: float) -> np.ndarray:
    """Bin spikes (T, N) into [bin_ms] windows; pop rate Hz per bin."""
    T, N = spikes.shape
    steps_per_bin = int(bin_ms / DT)
    n_bins = T // steps_per_bin
    spikes_t = spikes[: n_bins * steps_per_bin].reshape(n_bins, steps_per_bin, N)
    pop = spikes_t.sum(axis=(1, 2)) / (N * (bin_ms / 1000.0))
    return pop


def classify(result: dict) -> dict:
    """Outcome classification with a slope-aware sustained-vs-decay split.

    We measure two post-stim windows:
      * post1 = mean E rate over [T_STIM_MS, T_STIM_MS + 100]
      * post2 = mean E rate over [T_TOTAL_MS - 100, T_TOTAL_MS]

    Classification:
      * SEIZURE — post2 >= SEIZURE_HZ (network locked at refractory saturation)
      * DECAY   — post2 < DECAY_HZ  (network has quenched), OR
                  post2 / post1 < DECAY_SLOPE_RATIO (still actively decaying;
                  we count this as "not sustained" because the trace clearly
                  hasn't settled — a passive NMDA tail isn't an attractor)
      * SUSTAINED — anywhere else: post2 stable relative to post1, in
                    [DECAY_HZ, SEIZURE_HZ). This is the only class that
                    represents genuine recurrent-attractor maintenance.
    """
    e = result["e"]
    rate_trace = population_rate_hz(e, RATE_BIN_MS)
    t_bin = np.arange(len(rate_trace)) * RATE_BIN_MS

    stim_mask = (t_bin >= 50) & (t_bin < T_STIM_MS)
    post1_mask = (t_bin >= T_STIM_MS) & (t_bin < T_STIM_MS + 100.0)
    post2_mask = t_bin >= (T_TOTAL_MS - 100.0)
    stim_rate = float(rate_trace[stim_mask].mean()) if stim_mask.any() else 0.0
    post1_rate = float(rate_trace[post1_mask].mean()) if post1_mask.any() else 0.0
    post2_rate = float(rate_trace[post2_mask].mean()) if post2_mask.any() else 0.0

    if post2_rate >= SEIZURE_HZ:
        outcome = "seizure"
    elif post2_rate < DECAY_HZ:
        outcome = "decay"
    elif post1_rate > 0 and post2_rate / post1_rate < DECAY_SLOPE_RATIO:
        outcome = "decay"  # active decay, not a stable attractor
    else:
        outcome = "sustained"

    return {
        "stim_rate_hz": stim_rate,
        "post1_rate_hz": post1_rate,
        "post2_rate_hz": post2_rate,
        "late_rate_hz": post2_rate,  # kept for back-compat with figure code
        "outcome": outcome,
    }


OUTCOME_COLORS = {
    # Map each outcome class to a palette colour. Sustained is the one we
    # *want* to find — give it the strong accent so an empty grid reads as
    # an empty-by-design plot.
    "decay": "#cfd6db",      # cool light grey — silent network
    "sustained": "#1f9d3a",  # green — the (empty) target band
    "seizure": "#cc0000",    # deep-red — runaway saturation
}


TRACE_BIN_MS = 25.0     # rate-trace smoothing — long enough to wash out gamma peaks


def fig_phase_map(grid_results: dict, run_id: str) -> plt.Figure:
    """Phase map: every (W_ee, input_rate) cell coloured by outcome class.

    A cell is green only if at least one seed produced *sustained* (rare —
    expected to be empty for coba). Red if any seed seized; grey if all
    seeds decayed. The whole point of the figure is the absence of green.
    """
    w_ees = sorted({k[0] for k in grid_results})
    rates = sorted({k[1] for k in grid_results})
    Z = np.zeros((len(w_ees), len(rates)), dtype=int)  # 0=decay, 1=sustained, 2=seizure
    late_rates = np.zeros_like(Z, dtype=float)
    for i, w in enumerate(w_ees):
        for j, rt in enumerate(rates):
            ms = grid_results[(w, rt)]
            outs = [m["outcome"] for m in ms]
            Z[i, j] = 2 if "seizure" in outs else (1 if "sustained" in outs else 0)
            late_rates[i, j] = float(np.mean([m["late_rate_hz"] for m in ms]))

    fig, ax = plt.subplots(figsize=(9, 5.0625), dpi=150)
    cmap = plt.matplotlib.colors.ListedColormap([
        OUTCOME_COLORS["decay"],
        OUTCOME_COLORS["sustained"],
        OUTCOME_COLORS["seizure"],
    ])
    ax.imshow(Z, cmap=cmap, vmin=0, vmax=2, aspect="auto", origin="lower")
    for i in range(len(w_ees)):
        for j in range(len(rates)):
            v = late_rates[i, j]
            txt = f"{v:.0f}" if v < 1000 else f"{v:.0f}"
            color = "white" if Z[i, j] == 2 else theme.INK_STRONG
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=theme.SIZE_ANNOTATION, color=color)
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels([str(r) for r in rates], fontsize=theme.SIZE_TICK)
    ax.set_yticks(range(len(w_ees)))
    ax.set_yticklabels([f"{w:g}" for w in w_ees], fontsize=theme.SIZE_TICK)
    ax.set_xlabel("Input rate during stim (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("$W_{ee}$ mean (μS, pre-fan-in)", fontsize=theme.SIZE_LABEL)
    ax.set_title(
        f"Late-window mean E rate (Hz) — coba (ei_strength=0), slow-syn on\n"
        f"green = sustained {DECAY_HZ:g}–{SEIZURE_HZ:g} Hz | "
        f"grey = decay <{DECAY_HZ:g} | red = seizure ≥{SEIZURE_HZ:g}",
        fontsize=theme.SIZE_LABEL,
    )
    # Legend swatches
    for outcome, x in zip(["decay", "sustained", "seizure"], [0.05, 0.40, 0.78]):
        ax.add_patch(plt.matplotlib.patches.Rectangle(
            (x, -0.18), 0.04, 0.04, transform=ax.transAxes,
            color=OUTCOME_COLORS[outcome], clip_on=False,
        ))
        ax.text(x + 0.05, -0.16, outcome, transform=ax.transAxes,
                fontsize=theme.SIZE_ANNOTATION, va="center")
    fig.tight_layout()
    _stamp(fig, run_id)
    return fig


RASTER_N_NEURONS = 80   # subsampled cells shown in each corner raster


def fig_example_traces(corner_traces: list, run_id: str) -> plt.Figure:
    """For each of the 4 grid corners: rate trace on top, raster underneath.

    Each entry of corner_traces is a dict: {label, w, rate, e, outcome}.
    The rate trace shows level; the raster underneath shows the texture
    (gamma-locked synchrony, asynchronous-dense, silent, saturated).
    """
    fig = plt.figure(figsize=(13, 7.3), dpi=150)
    outer = fig.add_gridspec(
        2, 2, hspace=0.35, wspace=0.15,
        left=0.07, right=0.97, top=0.92, bottom=0.06,
    )
    rng = np.random.default_rng(0)
    cell_pick = rng.choice(N_E, size=min(RASTER_N_NEURONS, N_E), replace=False)
    cell_pick.sort()

    for cell_slot, c in zip(
        [outer[0, 0], outer[0, 1], outer[1, 0], outer[1, 1]], corner_traces
    ):
        inner = cell_slot.subgridspec(2, 1, height_ratios=[1, 2], hspace=0.05)
        ax_rate = fig.add_subplot(inner[0])
        ax_rast = fig.add_subplot(inner[1], sharex=ax_rate)

        # Top: rate trace (gamma-smoothed)
        trace = population_rate_hz(c["e"], TRACE_BIN_MS)
        t = np.arange(len(trace)) * TRACE_BIN_MS
        color = OUTCOME_COLORS[c["outcome"]]
        ax_rate.axvspan(0, T_STIM_MS, color=theme.DEEP_RED, alpha=0.07)
        ax_rate.axvline(T_STIM_MS, color=theme.DEEP_RED, lw=0.6, alpha=0.6)
        ax_rate.axhline(SEIZURE_HZ, color=theme.INK_BLACK, lw=0.6, ls="--", alpha=0.4)
        ax_rate.fill_between(t, trace, 0, color=color, alpha=0.35, linewidth=0)
        ax_rate.plot(t, trace, color=color, lw=1.2)
        ax_rate.set_xlim(0, T_TOTAL_MS)
        ax_rate.set_ylim(0, 360)
        ax_rate.set_yticks([0, SEIZURE_HZ])
        ax_rate.set_yticklabels(["0", f"{int(SEIZURE_HZ)}"], fontsize=theme.SIZE_TICK)
        ax_rate.tick_params(axis="x", labelbottom=False)
        ax_rate.set_ylabel("E rate (Hz)", fontsize=theme.SIZE_ANNOTATION)
        ax_rate.set_title(
            f"{c['label']}  ($W_{{ee}}$={c['w']:g}, R={c['rate']} Hz)  →  {c['outcome']}",
            fontsize=theme.SIZE_LABEL, loc="left", pad=4,
        )

        # Bottom: subsampled raster
        e = c["e"]
        e_sub = e[:, cell_pick]
        t_idx, n_idx = np.where(e_sub > 0)
        ax_rast.scatter(
            t_idx * DT, n_idx,
            s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.5,
        )
        ax_rast.axvspan(0, T_STIM_MS, color=theme.DEEP_RED, alpha=0.07)
        ax_rast.axvline(T_STIM_MS, color=theme.DEEP_RED, lw=0.6, alpha=0.6)
        ax_rast.set_xlim(0, T_TOTAL_MS)
        ax_rast.set_ylim(-1, len(cell_pick))
        ax_rast.set_yticks([])
        ax_rast.set_ylabel(f"{len(cell_pick)} E cells",
                           fontsize=theme.SIZE_ANNOTATION)
        ax_rast.set_xlabel("Time (ms)", fontsize=theme.SIZE_LABEL)
        ax_rast.tick_params(axis="x", labelsize=theme.SIZE_TICK)

    fig.suptitle(
        "Rate trace + raster at four corners of the (W_ee × input rate) grid",
        fontsize=theme.SIZE_TITLE, y=0.985,
    )
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
    w_ee_grid = tier_cfg["w_ee"]
    rate_grid = tier_cfg["rates"]
    n_seeds = tier_cfg["n_seeds"]
    print(f"[{SLUG}] tier={tier}  seeds={n_seeds}  "
          f"W_ee={w_ee_grid}  rates={rate_grid}")

    # 2D sweep: (W_ee × input_rate), n_seeds per cell.
    # grid_metrics[(w, r)] = [metric_dict_per_seed]
    # corner_traces holds one example spike record per grid corner for the
    # trace figure (we don't need to cache the rest).
    grid_metrics: dict[tuple, list[dict]] = {}
    corner_keys = {(w_ee_grid[0], rate_grid[0]),
                   (w_ee_grid[0], rate_grid[-1]),
                   (w_ee_grid[-1], rate_grid[0]),
                   (w_ee_grid[-1], rate_grid[-1])}
    corner_traces_raw: dict[tuple, np.ndarray] = {}

    for w in w_ee_grid:
        for rt in rate_grid:
            grid_metrics[(w, rt)] = []
            for s in range(n_seeds):
                seed = 42 + s
                r = run_one(rt, w, seed)
                m = classify(r)
                grid_metrics[(w, rt)].append(m)
                if s == 0 and (w, rt) in corner_keys and (w, rt) not in corner_traces_raw:
                    corner_traces_raw[(w, rt)] = r["e"]
            outcomes = [m["outcome"] for m in grid_metrics[(w, rt)]]
            late = np.mean([m["late_rate_hz"] for m in grid_metrics[(w, rt)]])
            verdict = (
                "SEIZURE" if "seizure" in outcomes else
                "SUST"    if "sustained" in outcomes else
                "DECAY"
            )
            print(f"  W_ee={w:>4}  R={rt:>3}Hz  late={late:6.2f}Hz  {verdict}")

    # Figures
    fig1 = fig_phase_map(grid_metrics, run_id)
    fig1.savefig(FIGURES / "phase_map.png", dpi=150)
    plt.close(fig1)

    corner_traces = [
        {"label": "low W_ee, low input",
         "w": w_ee_grid[0], "rate": rate_grid[0],
         "e": corner_traces_raw[(w_ee_grid[0], rate_grid[0])],
         "outcome": grid_metrics[(w_ee_grid[0], rate_grid[0])][0]["outcome"]},
        {"label": "low W_ee, high input",
         "w": w_ee_grid[0], "rate": rate_grid[-1],
         "e": corner_traces_raw[(w_ee_grid[0], rate_grid[-1])],
         "outcome": grid_metrics[(w_ee_grid[0], rate_grid[-1])][0]["outcome"]},
        {"label": "high W_ee, low input",
         "w": w_ee_grid[-1], "rate": rate_grid[0],
         "e": corner_traces_raw[(w_ee_grid[-1], rate_grid[0])],
         "outcome": grid_metrics[(w_ee_grid[-1], rate_grid[0])][0]["outcome"]},
        {"label": "high W_ee, high input",
         "w": w_ee_grid[-1], "rate": rate_grid[-1],
         "e": corner_traces_raw[(w_ee_grid[-1], rate_grid[-1])],
         "outcome": grid_metrics[(w_ee_grid[-1], rate_grid[-1])][0]["outcome"]},
    ]
    fig2 = fig_example_traces(corner_traces, run_id)
    fig2.savefig(FIGURES / "example_traces.png", dpi=150)
    plt.close(fig2)

    # Roll up per-cell summaries for numbers.json
    by_cell = []
    sustained_count = 0
    seizure_count = 0
    decay_count = 0
    for (w, rt), ms in grid_metrics.items():
        outcomes = [m["outcome"] for m in ms]
        sustained_count += sum(1 for o in outcomes if o == "sustained")
        seizure_count += sum(1 for o in outcomes if o == "seizure")
        decay_count += sum(1 for o in outcomes if o == "decay")
        by_cell.append({
            "w_ee_mean": float(w),
            "input_rate_hz": float(rt),
            "n_seeds": len(ms),
            "late_rate_hz_mean": float(np.mean([m["late_rate_hz"] for m in ms])),
            "stim_rate_hz_mean": float(np.mean([m["stim_rate_hz"] for m in ms])),
            "outcomes": outcomes,
        })

    numbers = {
        "run_id": run_id,
        "config": {
            "dt": DT,
            "t_stim_ms": T_STIM_MS,
            "t_total_ms": T_TOTAL_MS,
            "t_late_start_ms": T_LATE_START_MS,
            "slow_syn_gain": SLOW_SYN_GAIN,
            "ei_strength": EI_STRENGTH,
            "n_e": N_E,
            "n_in": N_IN,
            "tier": tier,
            "w_ee_grid": w_ee_grid,
            "input_rates_hz": rate_grid,
            "n_seeds": n_seeds,
            "decay_threshold_hz": DECAY_HZ,
            "seizure_threshold_hz": SEIZURE_HZ,
        },
        "results": {
            "by_cell": by_cell,
            "totals": {
                "decay": decay_count,
                "sustained": sustained_count,
                "seizure": seizure_count,
                "total_cells": decay_count + sustained_count + seizure_count,
            },
        },
        "success_criteria": [
            # The headline negative result: coba can't sustain. Pass if the
            # grid is empty of "sustained" outcomes — that confirms the
            # structural claim. (Inverted: this is a *predicted* fail of the
            # sustained-search; passing means the prediction held.)
            {
                "label": "structural claim: coba grid has zero sustained cells",
                "passed": sustained_count == 0,
                "detail": f"sustained cells = {sustained_count} of "
                          f"{decay_count + sustained_count + seizure_count}",
            },
            # The bistability sanity check: both decay AND seizure should
            # appear somewhere on the grid (otherwise we've only tested half
            # the regime).
            {
                "label": "grid covers both regimes (decay and seizure observed)",
                "passed": decay_count > 0 and seizure_count > 0,
                "detail": f"decay={decay_count}, seizure={seizure_count}",
            },
        ],
        "runtime_s": time.time() - t0,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print(f"[{SLUG}] done in {time.time() - t0:.1f}s. "
          f"decay={decay_count} sustained={sustained_count} seizure={seizure_count}")

    if not all(c["passed"] for c in numbers["success_criteria"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
