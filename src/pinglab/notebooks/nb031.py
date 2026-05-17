"""Notebook runner for entry 031 — ping breaks the bistability.

Follow-up to nb030. Same 2D sweep over (W_ee × input rate), same slope-
aware outcome classifier, same untrained-network forward-only setup —
but now we run *two* models side by side:

    * coba — ei_strength = 0, W_in = 0.3 (coba recipe)
    * ping — ei_strength = 1, W_in = 1.2 (ping recipe)

Each is given the W_in init that matches its own working regime; nothing
else differs except the I→E→I loop. The prediction (from nb030's mean-
field argument): ping should have a non-empty sustained band that coba
doesn't, because inhibition adds the negative-feedback equation that
makes intermediate fixed points stable.

Notebook entry: src/docs/src/pages/notebooks/nb031.mdx
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

SLUG = "nb031"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG

# ── Common setup ──────────────────────────────────────────────────────
DT = 0.1
T_STIM_MS = 200.0
T_TOTAL_MS = 600.0
N_E = 256
N_IN = 64
W_EE_STD_FRAC = 0.1
W_IN_SPARSITY = 0.95
SLOW_SYN_GAIN = 0.5

# Per-model W_in (each at the model's own standard operating point).
MODEL_CONFIGS: dict[str, dict] = {
    "coba": {"ei_strength": 0.0, "w_in_mean": 0.3, "w_in_std": 0.09},
    "ping": {"ei_strength": 1.0, "w_in_mean": 1.2, "w_in_std": 0.36},
}

# ── 2D sweep grid ─────────────────────────────────────────────────────
W_EE_MEAN_GRID = [0.0, 0.05, 0.10, 0.25, 0.50, 1.00]
INPUT_RATES_HZ = [5, 25, 50, 100, 150]
EXTRA_SMALL_W_EE = [0.0, 0.50]
EXTRA_SMALL_RATES = [25, 100]

DEFAULT_TIER = "small"
TIER_CONFIG: dict[str, dict] = {
    "extra small": {"n_seeds": 1, "w_ee": EXTRA_SMALL_W_EE, "rates": EXTRA_SMALL_RATES},
    "small":       {"n_seeds": 1, "w_ee": W_EE_MEAN_GRID,   "rates": INPUT_RATES_HZ},
    "medium":      {"n_seeds": 3, "w_ee": W_EE_MEAN_GRID,   "rates": INPUT_RATES_HZ},
    "large":       {"n_seeds": 5, "w_ee": W_EE_MEAN_GRID,   "rates": INPUT_RATES_HZ},
}

# ── Outcome classification (matches nb030) ────────────────────────────
DECAY_HZ = 1.0
SEIZURE_HZ = 120.0
DECAY_SLOPE_RATIO = 0.5
RATE_BIN_MS = 5.0
TRACE_BIN_MS = 25.0
RASTER_N_NEURONS = 80

OUTCOME_COLORS = {
    "decay": "#cfd6db",
    "sustained": "#1f9d3a",
    "seizure": "#cc0000",
}


def _stamp(fig, run_id: str) -> None:
    fig.text(
        0.99, 0.005, run_id,
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


def make_input_spikes(rate_hz: float, T_stim_steps: int, T_total_steps: int,
                      seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rate_hz * (DT / 1000.0)
    stim = (rng.random((T_stim_steps, 1, N_IN)) < p).astype(np.float32)
    post = np.zeros((T_total_steps - T_stim_steps, 1, N_IN), dtype=np.float32)
    return np.concatenate([stim, post], axis=0)


def run_one(model: str, rate_hz: float, w_ee_mean: float, seed: int) -> dict:
    """Build a fresh model-tagged net, drive with Poisson input, capture spikes."""
    import torch

    import models as M
    from config import build_net, patch_dt
    from cli import _auto_device, seed_everything

    mc = MODEL_CONFIGS[model]
    device = _auto_device()
    seed_everything(seed)
    M.N_IN = N_IN
    patch_dt(DT)

    net = build_net(
        "ping",
        w_in=(mc["w_in_mean"], mc["w_in_std"]),
        w_in_sparsity=W_IN_SPARSITY,
        w_ee=(w_ee_mean, w_ee_mean * W_EE_STD_FRAC),
        ei_strength=mc["ei_strength"],
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
    i_full = (
        net.spike_record["inh"].cpu().numpy()
        if "inh" in net.spike_record else None
    )
    return {
        "e": e_full,
        "i": i_full,
        "model": model,
        "rate_hz": rate_hz,
        "w_ee_mean": w_ee_mean,
        "seed": seed,
    }


def population_rate_hz(spikes: np.ndarray, bin_ms: float) -> np.ndarray:
    T, N = spikes.shape
    steps_per_bin = int(bin_ms / DT)
    n_bins = T // steps_per_bin
    spikes_t = spikes[: n_bins * steps_per_bin].reshape(n_bins, steps_per_bin, N)
    pop = spikes_t.sum(axis=(1, 2)) / (N * (bin_ms / 1000.0))
    return pop


def classify(result: dict) -> dict:
    """Slope-aware outcome classifier (same as nb030)."""
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
        outcome = "decay"
    else:
        outcome = "sustained"

    return {
        "stim_rate_hz": stim_rate,
        "post1_rate_hz": post1_rate,
        "post2_rate_hz": post2_rate,
        "late_rate_hz": post2_rate,
        "outcome": outcome,
    }


def fig_phase_maps(grid_by_model: dict, run_id: str) -> plt.Figure:
    """Two phase maps side by side: coba | ping."""
    models = ["coba", "ping"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), dpi=150, sharey=True)
    cmap = plt.matplotlib.colors.ListedColormap([
        OUTCOME_COLORS["decay"],
        OUTCOME_COLORS["sustained"],
        OUTCOME_COLORS["seizure"],
    ])
    for ax, model in zip(axes, models):
        grid = grid_by_model[model]
        w_ees = sorted({k[0] for k in grid})
        rates = sorted({k[1] for k in grid})
        Z = np.zeros((len(w_ees), len(rates)), dtype=int)
        late = np.zeros_like(Z, dtype=float)
        for i, w in enumerate(w_ees):
            for j, rt in enumerate(rates):
                ms = grid[(w, rt)]
                outs = [m["outcome"] for m in ms]
                Z[i, j] = 2 if "seizure" in outs else (1 if "sustained" in outs else 0)
                late[i, j] = float(np.mean([m["late_rate_hz"] for m in ms]))
        ax.imshow(Z, cmap=cmap, vmin=0, vmax=2, aspect="auto", origin="lower")
        for i in range(len(w_ees)):
            for j in range(len(rates)):
                color = "white" if Z[i, j] == 2 else theme.INK_STRONG
                ax.text(j, i, f"{late[i, j]:.0f}", ha="center", va="center",
                        fontsize=theme.SIZE_ANNOTATION, color=color)
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([str(r) for r in rates], fontsize=theme.SIZE_TICK)
        ax.set_yticks(range(len(w_ees)))
        ax.set_yticklabels([f"{w:g}" for w in w_ees], fontsize=theme.SIZE_TICK)
        ax.set_xlabel("Input rate during stim (Hz)", fontsize=theme.SIZE_LABEL)
        ax.set_title(
            f"{model}  (ei_strength = {MODEL_CONFIGS[model]['ei_strength']:.0f}, "
            f"W_in mean = {MODEL_CONFIGS[model]['w_in_mean']})",
            fontsize=theme.SIZE_LABEL,
        )
    axes[0].set_ylabel("$W_{ee}$ mean (μS, pre-fan-in)", fontsize=theme.SIZE_LABEL)

    fig.suptitle(
        "Late-window mean E rate — same sweep, two models",
        fontsize=theme.SIZE_TITLE, y=0.99,
    )
    # Legend
    for outcome, x in zip(["decay", "sustained", "seizure"], [0.05, 0.45, 0.80]):
        fig.patches.append(plt.matplotlib.patches.Rectangle(
            (x, 0.005), 0.025, 0.025, transform=fig.transFigure,
            color=OUTCOME_COLORS[outcome], clip_on=False,
        ))
        fig.text(x + 0.03, 0.018, outcome,
                 fontsize=theme.SIZE_ANNOTATION, va="center")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _stamp(fig, run_id)
    return fig


RASTER_N_I_NEURONS = 30  # subsampled I cells shown above the E cells per panel
RASTER_E_I_GAP = 6       # blank rows between E and I bands


def fig_ping_corners(corner_traces: list, run_id: str) -> plt.Figure:
    """Rate trace + (I+E) raster across the three outcome classes for ping."""
    fig = plt.figure(figsize=(13, 7.3), dpi=150)
    outer = fig.add_gridspec(
        2, 2, hspace=0.40, wspace=0.15,
        left=0.07, right=0.97, top=0.92, bottom=0.06,
    )
    rng = np.random.default_rng(0)
    e_pick = rng.choice(N_E, size=min(RASTER_N_NEURONS, N_E), replace=False)
    e_pick.sort()

    for cell_slot, c in zip(
        [outer[0, 0], outer[0, 1], outer[1, 0], outer[1, 1]], corner_traces
    ):
        inner = cell_slot.subgridspec(2, 1, height_ratios=[1, 2], hspace=0.05)
        ax_rate = fig.add_subplot(inner[0])
        ax_rast = fig.add_subplot(inner[1], sharex=ax_rate)

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
        ax_rate.set_yticklabels(["0", f"{int(SEIZURE_HZ)}"],
                                fontsize=theme.SIZE_TICK)
        ax_rate.tick_params(axis="x", labelbottom=False)
        ax_rate.set_ylabel("E rate (Hz)", fontsize=theme.SIZE_ANNOTATION)
        ax_rate.set_title(
            f"{c['label']}  ($W_{{ee}}$={c['w']:g}, R={c['rate']} Hz)  →  {c['outcome']}",
            fontsize=theme.SIZE_LABEL, loc="left", pad=4,
        )

        # E cells in black at the bottom, I cells in red above them with a gap.
        e_sub = c["e"][:, e_pick]
        e_t, e_n = np.where(e_sub > 0)
        ax_rast.scatter(
            e_t * DT, e_n,
            s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.5,
            rasterized=True,
        )
        n_e_plot = len(e_pick)
        n_i_plot = 0
        if c.get("i") is not None and c["i"].shape[1] > 0:
            i_arr = c["i"]
            n_i_total = i_arr.shape[1]
            i_pick = rng.choice(
                n_i_total, size=min(RASTER_N_I_NEURONS, n_i_total), replace=False
            )
            i_pick.sort()
            i_sub = i_arr[:, i_pick]
            i_t, i_n = np.where(i_sub > 0)
            ax_rast.scatter(
                i_t * DT, i_n + n_e_plot + RASTER_E_I_GAP,
                s=2.0, c=theme.DEEP_RED, marker="|", linewidths=0.5,
                rasterized=True,
            )
            n_i_plot = len(i_pick)
        ax_rast.axvspan(0, T_STIM_MS, color=theme.DEEP_RED, alpha=0.07)
        ax_rast.axvline(T_STIM_MS, color=theme.DEEP_RED, lw=0.6, alpha=0.6)
        ax_rast.set_xlim(0, T_TOTAL_MS)
        total_rows = n_e_plot + (RASTER_E_I_GAP + n_i_plot if n_i_plot else 0)
        ax_rast.set_ylim(-1, total_rows + 1)
        # Side labels: "E" centred on its band, "I" centred on its band
        ytick_pos = [n_e_plot / 2]
        ytick_lbl = [f"E\n({n_e_plot})"]
        if n_i_plot:
            ytick_pos.append(n_e_plot + RASTER_E_I_GAP + n_i_plot / 2)
            ytick_lbl.append(f"I\n({n_i_plot})")
        ax_rast.set_yticks(ytick_pos)
        ax_rast.set_yticklabels(ytick_lbl, fontsize=theme.SIZE_ANNOTATION)
        ax_rast.tick_params(axis="y", length=0)
        ax_rast.set_xlabel("Time (ms)", fontsize=theme.SIZE_LABEL)
        ax_rast.tick_params(axis="x", labelsize=theme.SIZE_TICK)

    fig.suptitle(
        "ping across the three outcome classes — rate trace + raster",
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

    grid_by_model: dict[str, dict[tuple, list[dict]]] = {"coba": {}, "ping": {}}
    # We want fig 2 to actually showcase the sustained band — so we cache
    # *every* ping cell's example raster (both E and I), then pick four
    # informative ones (one decay, two sustained, one seizure) after
    # classification is done.
    ping_all_rasters: dict[tuple, dict] = {}

    for model in ["coba", "ping"]:
        print(f"--- {model} ({MODEL_CONFIGS[model]}) ---")
        for w in w_ee_grid:
            for rt in rate_grid:
                grid_by_model[model][(w, rt)] = []
                for s in range(n_seeds):
                    seed = 42 + s
                    r = run_one(model, rt, w, seed)
                    m = classify(r)
                    grid_by_model[model][(w, rt)].append(m)
                    if model == "ping" and s == 0 and (w, rt) not in ping_all_rasters:
                        ping_all_rasters[(w, rt)] = {"e": r["e"], "i": r["i"]}
                outs = [m["outcome"] for m in grid_by_model[model][(w, rt)]]
                late = np.mean([m["late_rate_hz"] for m in grid_by_model[model][(w, rt)]])
                verdict = (
                    "SEIZURE" if "seizure" in outs else
                    "SUST"    if "sustained" in outs else
                    "DECAY"
                )
                print(f"  W_ee={w:>4}  R={rt:>3}Hz  late={late:6.2f}Hz  {verdict}")

    # Figures
    fig1 = fig_phase_maps(grid_by_model, run_id)
    fig1.savefig(FIGURES / "phase_maps.png", dpi=150)
    plt.close(fig1)

    # Pick 4 informative cells from ping's grid: one decay, two sustained
    # (preferring the highest-rate sustained cell + a complementary one),
    # and one seizure. If a category is empty, fall back to a grid corner.
    def _ping_cell(outcome_filter, sort_key=None, prefer=None):
        cells = [(w, rt) for (w, rt), ms in grid_by_model["ping"].items()
                 if ms[0]["outcome"] == outcome_filter]
        if not cells:
            return None
        if prefer is not None and prefer in cells:
            return prefer
        if sort_key:
            cells.sort(key=sort_key)
        return cells[0]

    decay_cell    = _ping_cell("decay",     sort_key=lambda c: (c[0], c[1]))
    sustained_lo  = _ping_cell("sustained", sort_key=lambda c: (c[1], c[0]))   # lowest input rate
    sustained_hi  = _ping_cell("sustained", sort_key=lambda c: (-c[1], -c[0])) # highest input rate
    seizure_cell  = _ping_cell("seizure",   sort_key=lambda c: (-c[0], -c[1]))

    panel_specs = [
        (decay_cell,   "ping — decay"),
        (sustained_lo, "ping — sustained (low input)"),
        (sustained_hi, "ping — sustained (higher input)"),
        (seizure_cell, "ping — seizure"),
    ]
    ping_corners = []
    for cell, label in panel_specs:
        if cell is None:
            continue
        w, rt = cell
        rast = ping_all_rasters[(w, rt)]
        ping_corners.append({
            "label": label,
            "w": w, "rate": rt,
            "e": rast["e"],
            "i": rast["i"],
            "outcome": grid_by_model["ping"][(w, rt)][0]["outcome"],
        })
    fig2 = fig_ping_corners(ping_corners, run_id)
    fig2.savefig(FIGURES / "ping_corners.png", dpi=150)
    plt.close(fig2)

    # Numbers
    by_model_summary = {}
    for model, grid in grid_by_model.items():
        decay_n = seizure_n = sustained_n = 0
        by_cell = []
        for (w, rt), ms in grid.items():
            outs = [m["outcome"] for m in ms]
            decay_n += sum(1 for o in outs if o == "decay")
            seizure_n += sum(1 for o in outs if o == "seizure")
            sustained_n += sum(1 for o in outs if o == "sustained")
            by_cell.append({
                "w_ee_mean": float(w),
                "input_rate_hz": float(rt),
                "n_seeds": len(ms),
                "late_rate_hz_mean": float(np.mean([m["late_rate_hz"] for m in ms])),
                "outcomes": outs,
            })
        by_model_summary[model] = {
            "totals": {
                "decay": decay_n,
                "sustained": sustained_n,
                "seizure": seizure_n,
                "total_cells": decay_n + sustained_n + seizure_n,
            },
            "by_cell": by_cell,
        }

    coba_sus = by_model_summary["coba"]["totals"]["sustained"]
    ping_sus = by_model_summary["ping"]["totals"]["sustained"]

    numbers = {
        "run_id": run_id,
        "config": {
            "dt": DT,
            "t_stim_ms": T_STIM_MS,
            "t_total_ms": T_TOTAL_MS,
            "slow_syn_gain": SLOW_SYN_GAIN,
            "n_e": N_E,
            "n_in": N_IN,
            "tier": tier,
            "w_ee_grid": w_ee_grid,
            "input_rates_hz": rate_grid,
            "n_seeds": n_seeds,
            "decay_threshold_hz": DECAY_HZ,
            "seizure_threshold_hz": SEIZURE_HZ,
            "model_configs": MODEL_CONFIGS,
        },
        "results": by_model_summary,
        "success_criteria": [
            {
                "label": "coba grid has zero sustained cells (nb030 replication)",
                "passed": coba_sus == 0,
                "detail": f"coba sustained = {coba_sus}",
            },
            {
                "label": "ping grid has at least one sustained cell",
                "passed": ping_sus > 0,
                "detail": f"ping sustained = {ping_sus}",
            },
            {
                "label": "ping has strictly more sustained cells than coba",
                "passed": ping_sus > coba_sus,
                "detail": f"ping={ping_sus} > coba={coba_sus}",
            },
        ],
        "runtime_s": time.time() - t0,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print(f"[{SLUG}] done in {time.time() - t0:.1f}s. "
          f"coba sustained={coba_sus}  ping sustained={ping_sus}")

    if not all(c["passed"] for c in numbers["success_criteria"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
