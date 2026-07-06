"""Notebook runner for entry 046 — Per-cycle E-spike count: is the ceiling 1?

Claim under test: each E cell emits ≤ 1 spike per gamma cycle.

Method. For each of exp041's 18 trained cells (6 τ_GABA × 3 seeds):
  1. Run inference on the test set; capture E and I spike tensors.
  2. Detect I-burst times per trial from the smoothed population I rate.
  3. Assign each timestep to its nearest I-burst → cycle index.
  4. For each (cell, cycle, trial), count E spikes within the cycle window.
  5. Aggregate the distribution P(spikes-per-cell-per-cycle = k).

Two figures:
  - spikes_per_cycle_distribution.png — P(k spikes) per τ_GABA.
  - ceiling_vs_fgamma.png — per-cell max rate vs f_γ; the y=x line is
    the "always-participate" ceiling = 1 spike per cycle.

No retraining; pure inference. Local MPS.

Writing: writings/exp046.typ · figures + numbers.json: artifacts/data/exp046/
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import theme  # noqa: E402
from helpers.figsave import save_figure  # noqa: E402
from helpers.operating_point import MODELS_DEFAULT_TAU_GABA_MS  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402

SLUG = "exp046"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
SNN_TOOL = REPO / "tools" / "snn" / "tool.py"

# τ_GABA sweep cells now live in the shared training root (exp022
# train-once / reuse-many), not the retired per-notebook exp041 dir.
NB041_ARTIFACTS = REPO / "temp" / "experiments" / "exp022"
NB041_NUMBERS = (
    REPO / "artifacts" / "data" / "exp041"
    / "numbers.json"
)

TAU_GABA_SWEEP_MS: tuple[float, ...] = (4.5, 6.0, 9.0, 12.0, 18.0, 27.0)
SEEDS: tuple[int, ...] = (42, 43, 44)

# Run scale — stamped into the manifest by run_dirs.prepare and rendered as
# the Methods table via RunScale; the mdx never restates these numbers.
SCALE = {
    "dataset": "mnist",
    "seeds": len(SEEDS),
    "cells": len(TAU_GABA_SWEEP_MS) * len(SEEDS),
    "grid": f"{len(TAU_GABA_SWEEP_MS)} tau_GABA x {len(SEEDS)} seeds",
}


def tau_label(tau_ms: float) -> str:
    return f"tg{f'{tau_ms:g}'.replace('.', 'p')}"


def exp041_cell_dir(tau_ms: float, seed: int) -> Path:
    return NB041_ARTIFACTS / f"ping__{tau_label(tau_ms)}__seed{seed}"


def load_exp041_f_gamma() -> dict[tuple[float, int], float]:
    if not NB041_NUMBERS.exists():
        raise SystemExit(
            f"missing exp041 numbers.json at {NB041_NUMBERS}; "
            "re-render exp041 to produce it."
        )
    data = json.loads(NB041_NUMBERS.read_text())
    out = {}
    for r in data.get("results", []):
        out[(float(r["tau_gaba_ms"]), int(r["seed"]))] = float(r["f_gamma_hz"])
    return out


# ─── cycle-window spike counting ────────────────────────────────────


def detect_i_burst_steps(s_i_trial: np.ndarray, dt_ms: float,
                         f_gamma_hz: float) -> np.ndarray:
    """Detect I-burst peak timesteps in a single trial.

    s_i_trial: (T, N_I) bool/int spike tensor.
    Returns: int array of peak timesteps (ascending).
    """
    from scipy.signal import find_peaks
    rate = s_i_trial.sum(axis=1).astype(np.float32)
    # Gaussian smooth with sigma = 1 ms.
    sigma_steps = max(1.0, 1.0 / dt_ms)
    L = int(np.ceil(4 * sigma_steps))
    k = np.arange(-L, L + 1)
    kernel = np.exp(-0.5 * (k / sigma_steps) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(rate, kernel, mode="same")
    cycle_steps = max(1.0, 1000.0 / max(f_gamma_hz, 1e-3) / dt_ms)
    # Require peaks to be ≥ 0.5 cycle apart and to lift above 5% of max.
    height = 0.05 * float(smooth.max())
    peaks, _ = find_peaks(
        smooth, distance=max(1, int(0.5 * cycle_steps)), height=height,
    )
    return peaks


def count_e_spikes_per_cycle(
    s_e_trial: np.ndarray, peak_steps: np.ndarray,
) -> np.ndarray:
    """Count E spikes per (cell, cycle) in one trial.

    s_e_trial: (T, N_E)
    peak_steps: (K,) I-burst peak timesteps.
    Returns: (K, N_E) int array.
    """
    T, N_E = s_e_trial.shape
    K = len(peak_steps)
    if K == 0:
        return np.zeros((0, N_E), dtype=np.int32)
    # Each timestep assigned to its nearest peak. Equivalent: cycle boundaries
    # at midpoints between consecutive peaks.
    edges = np.concatenate([
        [0],
        ((peak_steps[:-1] + peak_steps[1:]) // 2).astype(int),
        [T],
    ])
    counts = np.zeros((K, N_E), dtype=np.int32)
    for k in range(K):
        a, b = edges[k], edges[k + 1]
        if b > a:
            counts[k] = s_e_trial[a:b].sum(axis=0)
    return counts


def _infer_cell(train_dir: Path, extra_args: list[str], max_samples: int) -> Path:
    """Shell out to `sim --infer` under the cell's τ_GABA; return the out dir.

    Network build/load/forward run in the CLI; the notebook reads artifacts.
    """
    train_dir = train_dir.resolve()
    cfg = json.loads((train_dir / "config.json").read_text())
    tau_gaba_ms = float(cfg.get("tau_gaba_ms") or MODELS_DEFAULT_TAU_GABA_MS)
    out_dir = (ARTIFACTS / "infer" / train_dir.name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv", "run", "python", str(SNN_TOOL), "sim", "--infer",
            "--load-config", str(train_dir / "config.json"),
            "--load-weights", str(train_dir / "weights.pth"),
            "--tau-gaba", str(tau_gaba_ms),
            "--max-samples", str(max_samples),
            "--out-dir", str(out_dir),
            *extra_args,
        ],
        cwd=REPO,
        check=True,
    )
    return out_dir


def evaluate_cell(train_dir: Path, f_gamma_hz: float) -> dict:
    """Spikes-per-cell-per-cycle distribution + per-cell rates, via the CLI.

    Runs `sim --infer --outputs rasters per_cell_rates` under the cell's τ_GABA,
    then reconstructs per-trial E/I rasters from the sparse rasters.npz and bins
    (cell, cycle) spike counts locally (I-burst peaks delimit cycles). Per-cell
    rates come from per_cell_rates.npz; acc from metrics.json.
    """
    cfg = json.loads((train_dir / "config.json").read_text())
    tau_gaba_ms = float(cfg.get("tau_gaba_ms") or MODELS_DEFAULT_TAU_GABA_MS)
    dt_ms = float(cfg["dt"])
    out_dir = _infer_cell(
        train_dir,
        ["--outputs", "rasters", "per_cell_rates"],
        max_samples=int(cfg["max_samples"]),
    )
    m = json.loads((out_dir / "metrics.json").read_text())
    acc = float(m["best_acc"])

    per_cell_rate_hz = np.load(out_dir / "per_cell_rates.npz")["rate_e_per_cell"]

    # Reconstruct per-trial rasters from the sparse indices and bin (cell, cycle).
    R = np.load(out_dir / "rasters.npz")
    T, n_e, n_i = int(R["T"]), int(R["n_e"]), int(R["n_i"])
    trial_count = int(R["n_trials"])

    def _by_trial(prefix):
        tr = R[f"{prefix}_trial"]
        order = np.argsort(tr, kind="stable")
        return R[f"{prefix}_t"][order], R[f"{prefix}_cell"][order], \
            np.searchsorted(tr[order], np.arange(trial_count + 1))

    e_t, e_c, e_b = _by_trial("e")
    i_t, i_c, i_b = _by_trial("i")

    bucket_counts = np.zeros(4, dtype=np.int64)  # 0, 1, 2, ≥3
    cycle_count = 0
    for b in range(trial_count):
        s_i_trial = np.zeros((T, n_i), dtype=np.int8)
        s_i_trial[i_t[i_b[b]:i_b[b + 1]], i_c[i_b[b]:i_b[b + 1]]] = 1
        peaks = detect_i_burst_steps(s_i_trial, dt_ms, f_gamma_hz)
        if len(peaks) == 0:
            continue
        s_e_trial = np.zeros((T, n_e), dtype=np.int8)
        s_e_trial[e_t[e_b[b]:e_b[b + 1]], e_c[e_b[b]:e_b[b + 1]]] = 1
        counts = count_e_spikes_per_cycle(s_e_trial, peaks)  # (K, N_E)
        cycle_count += counts.shape[0]
        flat = counts.ravel()
        bucket_counts[0] += int((flat == 0).sum())
        bucket_counts[1] += int((flat == 1).sum())
        bucket_counts[2] += int((flat == 2).sum())
        bucket_counts[3] += int((flat >= 3).sum())

    return {
        "tau_gaba_ms": tau_gaba_ms,
        "f_gamma_hz": float(f_gamma_hz),
        "acc": acc,
        "n_trials": int(trial_count),
        "n_cycles_observed": int(cycle_count),
        "bucket_counts": bucket_counts.tolist(),
        "per_cell_rate_hz": per_cell_rate_hz.tolist(),
        "per_cell_max_rate_hz": float(per_cell_rate_hz.max()),
        "per_cell_median_rate_hz": float(np.median(per_cell_rate_hz)),
    }


# ─── plotting ───────────────────────────────────────────────────────


def plot_distribution(rows: list[dict], out_path: Path, run_id: str) -> dict:
    """Bar plot of P(spikes/cell/cycle = k) per τ_GABA, k ∈ {0, 1, 2, ≥3}."""
    theme.apply()
    by_tau: dict[float, list[dict]] = {}
    for r in rows:
        by_tau.setdefault(r["tau_gaba_ms"], []).append(r)
    taus_sorted = sorted(by_tau.keys())
    n = len(taus_sorted)
    fig, axes = plt.subplots(1, n, figsize=(6.9, 4.5 * 6.9 / (2.4 * n)), sharey=True)
    if n == 1:
        axes = [axes]
    labels = ["0", "1", "2", "≥3"]
    cmap = plt.get_cmap("viridis")
    summary = {}
    for i, tau in enumerate(taus_sorted):
        group = by_tau[tau]
        # Sum buckets across seeds then normalise.
        b = np.zeros(4, dtype=np.float64)
        for r in group:
            b += np.array(r["bucket_counts"], dtype=np.float64)
        total = b.sum()
        frac = b / max(total, 1.0)
        ax = axes[i]
        color = cmap(i / max(1, n - 1))
        ax.bar(labels, frac, color=color, edgecolor=theme.GREY_MID, lw=0.5)
        for k, v in enumerate(frac):
            ax.text(k, v + 0.01, f"{v * 100:.1f}%",
                    ha="center", va="bottom",
                    fontsize=theme.SIZE_ANNOTATION, color=theme.INK)
        ax.set_title(f"τ_GABA = {tau:g} ms", fontsize=theme.SIZE_LABEL)
        if i == 0:
            ax.set_ylabel("P(spikes per E cell per cycle)",
                          fontsize=theme.SIZE_LABEL)
        ax.set_xlabel("spikes / (cell · cycle)",
                      fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.15, lw=0.4)
        summary[f"tau_{tau:g}"] = {
            "frac_zero": float(frac[0]),
            "frac_one": float(frac[1]),
            "frac_two": float(frac[2]),
            "frac_three_plus": float(frac[3]),
        }
    fig.suptitle(
        "Spike count per E cell per gamma cycle — distribution by τ_GABA",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)  # bar chart: SVG + PDF
    plt.close(fig)
    return summary


def plot_ceiling_vs_fgamma(rows: list[dict], out_path: Path, run_id: str) -> dict:
    """Per-cell max E rate vs f_γ. y = f_γ line is the 1-spike/cycle ceiling."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(5.6, 3.15))

    # Aggregate by τ_GABA.
    by_tau: dict[float, list[dict]] = {}
    for r in rows:
        by_tau.setdefault(r["tau_gaba_ms"], []).append(r)
    taus_sorted = sorted(by_tau.keys())
    cmap = plt.get_cmap("viridis")

    f_gammas: list[float] = []
    max_rates: list[float] = []
    med_rates: list[float] = []
    for i, tau in enumerate(taus_sorted):
        group = by_tau[tau]
        color = cmap(i / max(1, len(taus_sorted) - 1))
        for r in group:
            f_gammas.append(r["f_gamma_hz"])
            max_rates.append(r["per_cell_max_rate_hz"])
            med_rates.append(r["per_cell_median_rate_hz"])
        # Plot the group: scatter.
        xs = [r["f_gamma_hz"] for r in group]
        ys_max = [r["per_cell_max_rate_hz"] for r in group]
        ys_med = [r["per_cell_median_rate_hz"] for r in group]
        ax.scatter(xs, ys_max, marker="^", s=60, color=color,
                   edgecolor=theme.INK, lw=0.5,
                   label=f"τ_GABA = {tau:g} ms — max cell"
                   if i == 0 else None)
        ax.scatter(xs, ys_med, marker="o", s=40, color=color,
                   edgecolor=theme.INK, lw=0.5,
                   label=f"τ_GABA = {tau:g} ms — median cell"
                   if i == 0 else None)

    f_arr = np.array(f_gammas)
    fmax = float(f_arr.max()) * 1.05
    xs = np.linspace(0, fmax, 100)
    ax.plot(xs, xs, color=theme.GREY_MID, lw=1.0, ls="--",
            label="y = f_γ (1 spike / cycle ceiling)")
    ax.plot(xs, 0.20 * xs, color=theme.MUTED, lw=1.0, ls=":",
            label="y = 0.20 · f_γ (exp041 slope p)")

    # Linear fit of max rate through origin.
    f_arr_np = np.array(f_gammas)
    m_arr = np.array(max_rates)
    slope_max = float((f_arr_np * m_arr).sum() / max((f_arr_np ** 2).sum(), 1e-9))
    ss_res = float(((m_arr - slope_max * f_arr_np) ** 2).sum())
    ss_tot = float(((m_arr - m_arr.mean()) ** 2).sum())
    r2_max = 1.0 - ss_res / max(ss_tot, 1e-9)

    ax.set_xlim(0, fmax)
    ax.set_ylim(0, fmax)
    ax.set_xlabel("Measured f_γ (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Per-cell E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, lw=0.4)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    ax.set_title(
        f"Per-cell ceiling: max-cell rate = {slope_max:.2f}·f_γ"
        f"  (R² = {r2_max:.3f})",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)  # sparse scatter + line overlays: SVG + PDF
    plt.close(fig)
    return {
        "max_cell_slope_vs_fgamma": slope_max,
        "max_cell_r2": r2_max,
    }


# ─── main ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-wipe-dir", action="store_true")
    args = parser.parse_args()

    # Publication profile: every figure this notebook writes is a print-sized
    # vector, emitted as both SVG (docs) and PDF (manuscript) by save_figure.
    theme.set_paper_mode(True)

    print("notebook_run_id = (allocating)")

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id}")
    print(f"  τ_GABAs: {TAU_GABA_SWEEP_MS}")
    print(f"  seeds:   {SEEDS}")

    prepare_run_dirs(SLUG, notebook_run_id, wipe=not args.no_wipe_dir,
                     make_artifacts=True, scale=SCALE, host="local")

    f_gamma_map = load_exp041_f_gamma()
    print(f"loaded f_γ for {len(f_gamma_map)} exp041 cells")

    rows: list[dict] = []
    n_cells = len(TAU_GABA_SWEEP_MS) * len(SEEDS)
    cells_done = 0
    for tau in TAU_GABA_SWEEP_MS:
        for seed in SEEDS:
            train_dir = exp041_cell_dir(tau, seed)
            if not (train_dir / "weights.pth").exists():
                print(f"[skip] missing {train_dir.relative_to(REPO)}")
                continue
            f_gamma = f_gamma_map.get((tau, seed))
            if f_gamma is None or f_gamma <= 0:
                print(f"[skip] no f_γ for τ_GABA={tau}, seed={seed}")
                continue
            t0 = time.monotonic()
            res = evaluate_cell(train_dir, f_gamma)
            res["seed"] = seed
            rows.append(res)
            cells_done += 1
            elapsed = time.monotonic() - t0
            buckets = res["bucket_counts"]
            tot = max(sum(buckets), 1)
            p_le_1 = (buckets[0] + buckets[1]) / tot
            print(
                f"[{cells_done}/{n_cells}] τ={tau:>5.1f} seed={seed}  "
                f"f_γ={f_gamma:5.2f}Hz  acc={res['acc']:5.2f}%  "
                f"max={res['per_cell_max_rate_hz']:5.2f}Hz  "
                f"P(≤1)={p_le_1:.4f}  ({elapsed:.1f}s)"
            )

    dist_summary = plot_distribution(
        rows, FIGURES / "spikes_per_cycle_distribution", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'spikes_per_cycle_distribution'}.{{svg,pdf}}")
    ceil_summary = plot_ceiling_vs_fgamma(
        rows, FIGURES / "ceiling_vs_fgamma", notebook_run_id,
    )
    print(
        f"wrote {FIGURES / 'ceiling_vs_fgamma'}.{{svg,pdf}}  "
        f"(max-cell slope = {ceil_summary['max_cell_slope_vs_fgamma']:.3f}, "
        f"R² = {ceil_summary['max_cell_r2']:.3f})"
    )

    # Aggregate global distribution across all cells.
    global_buckets = np.zeros(4, dtype=np.int64)
    for r in rows:
        global_buckets += np.array(r["bucket_counts"], dtype=np.int64)
    g_total = int(global_buckets.sum())
    global_fracs = {
        "zero": float(global_buckets[0]) / max(g_total, 1),
        "one": float(global_buckets[1]) / max(g_total, 1),
        "two": float(global_buckets[2]) / max(g_total, 1),
        "three_plus": float(global_buckets[3]) / max(g_total, 1),
    }

    duration_s = time.monotonic() - t_start
    numbers = {
        "notebook_run_id": notebook_run_id,
        "duration_s": duration_s,
        "duration": f"{int(duration_s // 60)}m {int(duration_s % 60):02d}s",
        "config": {
            "tau_gabas_ms": list(TAU_GABA_SWEEP_MS),
            "seeds": list(SEEDS),
            "exp041_source": "ping__tg{N}__seed{S} (100-epoch baselines)",
        },
        "global_fracs": global_fracs,
        "n_cell_cycle_pairs": g_total,
        "per_tau": dist_summary,
        "ceiling": ceil_summary,
        "results": rows,
    }
    out_json = FIGURES / "numbers.json"
    out_json.write_text(json.dumps(numbers, indent=2, default=float))
    print(f"wrote {out_json}")
    print(
        f"\nGlobal: {global_fracs['zero'] * 100:.2f}% zero, "
        f"{global_fracs['one'] * 100:.2f}% one, "
        f"{global_fracs['two'] * 100:.3f}% two, "
        f"{global_fracs['three_plus'] * 100:.4f}% ≥3  "
        f"(n = {g_total:,} cell·cycle pairs)"
    )
    print(f"\nTotal runtime: {numbers['duration']}")


if __name__ == "__main__":
    main()
