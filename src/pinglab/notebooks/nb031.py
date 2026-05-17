"""Notebook runner for entry 031 — ping W_ee transition at fixed input.

Hold input fixed at 25 Hz MNIST digit-0 Poisson encoding, sweep W_ee
from 0 to 0.1 in 6 steps, plot one (rate trace + E/I raster) panel per
W_ee. The sweep walks the decay → sustained boundary at the low edge of
ping's sustained band.

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

# ── Network setup ─────────────────────────────────────────────────────
DT = 0.1
T_STIM_MS = 200.0
T_TOTAL_MS = 600.0
N_E = 256
N_IN = 784           # 28×28 MNIST
DATASET = "mnist"
DIGIT_CLASS = 0
SAMPLE_IDX = 0
INPUT_RATE_HZ = 25.0  # peak Poisson rate at pixel intensity = 1
INPUT_SEED = 42
W_EE_STD_FRAC = 0.1
W_IN_SPARSITY = 0.95
W_IN_MEAN = 1.2       # ping recipe
W_IN_STD = 0.36
EI_STRENGTH = 1.0     # ping
SLOW_SYN_GAIN = 0.5

# ── W_ee sweep ────────────────────────────────────────────────────────
W_EE_GRID = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
EXTRA_SMALL_W_EE = [0.00, 0.10]

DEFAULT_TIER = "small"
TIER_CONFIG: dict[str, dict] = {
    "extra small": {"n_seeds": 1, "w_ee": EXTRA_SMALL_W_EE},
    "small":       {"n_seeds": 1, "w_ee": W_EE_GRID},
    "medium":      {"n_seeds": 3, "w_ee": W_EE_GRID},
    "large":       {"n_seeds": 5, "w_ee": W_EE_GRID},
}

# ── Outcome classification (same as nb030) ────────────────────────────
DECAY_HZ = 1.0
SEIZURE_HZ = 120.0
DECAY_SLOPE_RATIO = 0.5
RATE_BIN_MS = 5.0
TRACE_BIN_MS = 25.0
RASTER_N_E = 80
RASTER_N_I = 30
RASTER_E_I_GAP = 6

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


_PIXEL_VEC_CACHE: np.ndarray | None = None


def _get_digit_pixels() -> np.ndarray:
    global _PIXEL_VEC_CACHE
    if _PIXEL_VEC_CACHE is None:
        from cli.datasets import _load_dataset_image

        pixel_vec, _ = _load_dataset_image(DATASET, DIGIT_CLASS, SAMPLE_IDX)
        _PIXEL_VEC_CACHE = pixel_vec.astype(np.float32)
    return _PIXEL_VEC_CACHE


def make_input_spikes(T_stim_steps: int, T_total_steps: int, seed: int) -> np.ndarray:
    """Poisson-encode MNIST digit-0 during [0, T_STIM_MS], silence after."""
    from cli.encoders import encode_image_spikes

    pixel_vec = _get_digit_pixels()
    spikes = encode_image_spikes(
        pixel_vec,
        T_total_steps,
        DT,
        base_rate=0.0,
        stim_rate=INPUT_RATE_HZ,
        step_on_ms=0.0,
        step_off_ms=T_STIM_MS,
        seed=INPUT_SEED + seed,
    )
    return spikes.numpy().reshape(T_total_steps, 1, N_IN)


def run_one(w_ee_mean: float, seed: int) -> dict:
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

    spk_in = make_input_spikes(T_stim_steps, T_total_steps, seed)
    spk_in_t = torch.from_numpy(spk_in).to(device)
    with torch.no_grad():
        _ = net(input_spikes=spk_in_t)

    return {
        "w_ee_mean": w_ee_mean,
        "seed": seed,
        "e": net.spike_record["hid"].cpu().numpy(),
        "i": (net.spike_record["inh"].cpu().numpy()
              if "inh" in net.spike_record else None),
    }


def population_rate_hz(spikes: np.ndarray, bin_ms: float) -> np.ndarray:
    T, N = spikes.shape
    steps_per_bin = int(bin_ms / DT)
    n_bins = T // steps_per_bin
    spikes_t = spikes[: n_bins * steps_per_bin].reshape(n_bins, steps_per_bin, N)
    return spikes_t.sum(axis=(1, 2)) / (N * (bin_ms / 1000.0))


def classify(result: dict) -> dict:
    e = result["e"]
    trace = population_rate_hz(e, RATE_BIN_MS)
    t_bin = np.arange(len(trace)) * RATE_BIN_MS
    stim_mask = (t_bin >= 50) & (t_bin < T_STIM_MS)
    post1_mask = (t_bin >= T_STIM_MS) & (t_bin < T_STIM_MS + 100.0)
    post2_mask = t_bin >= (T_TOTAL_MS - 100.0)
    stim_rate = float(trace[stim_mask].mean()) if stim_mask.any() else 0.0
    post1_rate = float(trace[post1_mask].mean()) if post1_mask.any() else 0.0
    post2_rate = float(trace[post2_mask].mean()) if post2_mask.any() else 0.0

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


def fig_w_ee_sweep(panels: list, run_id: str) -> plt.Figure:
    """One (rate trace + E/I raster) panel per W_ee value, 2×3 layout."""
    n = len(panels)
    n_cols = 3 if n >= 5 else 2
    n_rows = (n + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(5.0 * n_cols, 3.65 * n_rows), dpi=150)
    outer = fig.add_gridspec(
        n_rows, n_cols, hspace=0.45, wspace=0.18,
        left=0.05, right=0.985, top=0.92, bottom=0.06,
    )
    rng = np.random.default_rng(0)
    e_pick = rng.choice(N_E, size=min(RASTER_N_E, N_E), replace=False)
    e_pick.sort()

    slots = [outer[r, c] for r in range(n_rows) for c in range(n_cols)]
    for slot, p in zip(slots, panels):
        inner = slot.subgridspec(2, 1, height_ratios=[1, 2], hspace=0.05)
        ax_rate = fig.add_subplot(inner[0])
        ax_rast = fig.add_subplot(inner[1], sharex=ax_rate)

        e = p["e"]
        trace = population_rate_hz(e, TRACE_BIN_MS)
        t = np.arange(len(trace)) * TRACE_BIN_MS
        color = OUTCOME_COLORS[p["outcome"]]

        # Rate trace
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
            f"$W_{{ee}}$ = {p['w']:.2f}    →    late = {p['late']:.1f} Hz  ({p['outcome']})",
            fontsize=theme.SIZE_LABEL, loc="left", pad=4,
        )

        # Raster (E in black, I in red above with gap)
        e_sub = e[:, e_pick]
        e_t, e_n = np.where(e_sub > 0)
        ax_rast.scatter(
            e_t * DT, e_n,
            s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.5,
            rasterized=True,
        )
        n_e_plot = len(e_pick)
        n_i_plot = 0
        if p.get("i") is not None and p["i"].shape[1] > 0:
            i_arr = p["i"]
            n_i_total = i_arr.shape[1]
            i_pick = rng.choice(
                n_i_total, size=min(RASTER_N_I, n_i_total), replace=False
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
        f"ping — $W_{{ee}}$ sweep at fixed input "
        f"(MNIST digit {DIGIT_CLASS}, R = {INPUT_RATE_HZ:.0f} Hz)",
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
    n_seeds = tier_cfg["n_seeds"]
    print(f"[{SLUG}] tier={tier}  R={INPUT_RATE_HZ}Hz  "
          f"W_ee={w_ee_grid}  seeds={n_seeds}")

    panels = []
    by_w: dict[float, list[dict]] = {}
    for w in w_ee_grid:
        by_w[w] = []
        first_e, first_i = None, None
        for s in range(n_seeds):
            seed = 42 + s
            r = run_one(w, seed)
            m = classify(r)
            by_w[w].append(m)
            if s == 0:
                first_e, first_i = r["e"], r["i"]
        late_mean = float(np.mean([m["late_rate_hz"] for m in by_w[w]]))
        outcomes = [m["outcome"] for m in by_w[w]]
        verdict = (
            "SEIZURE" if "seizure" in outcomes else
            "SUST"    if "sustained" in outcomes else
            "DECAY"
        )
        print(f"  W_ee={w:.2f}  late={late_mean:6.2f}Hz  {verdict}")
        # Use the first-seed outcome for the panel colour (sustained beats
        # decay if any seed sustains).
        panel_outcome = (
            "sustained" if "sustained" in outcomes else
            "seizure"   if "seizure"   in outcomes else
            "decay"
        )
        panels.append({
            "w": w,
            "late": late_mean,
            "outcome": panel_outcome,
            "e": first_e,
            "i": first_i,
        })

    fig = fig_w_ee_sweep(panels, run_id)
    fig.savefig(FIGURES / "w_ee_sweep.png", dpi=150)
    plt.close(fig)

    by_w_summary = []
    sustained_count = decay_count = seizure_count = 0
    for w, ms in by_w.items():
        outs = [m["outcome"] for m in ms]
        decay_count += sum(1 for o in outs if o == "decay")
        seizure_count += sum(1 for o in outs if o == "seizure")
        sustained_count += sum(1 for o in outs if o == "sustained")
        by_w_summary.append({
            "w_ee_mean": float(w),
            "n_seeds": len(ms),
            "late_rate_hz_mean": float(np.mean([m["late_rate_hz"] for m in ms])),
            "stim_rate_hz_mean": float(np.mean([m["stim_rate_hz"] for m in ms])),
            "outcomes": outs,
        })

    numbers = {
        "run_id": run_id,
        "config": {
            "dt": DT,
            "t_stim_ms": T_STIM_MS,
            "t_total_ms": T_TOTAL_MS,
            "slow_syn_gain": SLOW_SYN_GAIN,
            "ei_strength": EI_STRENGTH,
            "w_in_mean": W_IN_MEAN,
            "n_e": N_E,
            "n_in": N_IN,
            "dataset": DATASET,
            "digit_class": DIGIT_CLASS,
            "input_rate_hz": INPUT_RATE_HZ,
            "tier": tier,
            "w_ee_grid": w_ee_grid,
            "n_seeds": n_seeds,
            "decay_threshold_hz": DECAY_HZ,
            "seizure_threshold_hz": SEIZURE_HZ,
        },
        "results": {
            "by_w_ee": by_w_summary,
            "totals": {
                "decay": decay_count,
                "sustained": sustained_count,
                "seizure": seizure_count,
                "total_cells": decay_count + sustained_count + seizure_count,
            },
        },
        "success_criteria": [
            {
                "label": "at least one W_ee sustains",
                "passed": sustained_count > 0,
                "detail": f"sustained cells = {sustained_count}",
            },
            {
                "label": "at least one W_ee decays (lower W_ee edge is reachable)",
                "passed": decay_count > 0,
                "detail": f"decay cells = {decay_count}",
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
