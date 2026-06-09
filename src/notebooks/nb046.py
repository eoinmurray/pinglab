"""Notebook runner for entry 046 — Per-cycle E-spike count: is the ceiling 1?

Claim under test: each E cell emits ≤ 1 spike per gamma cycle.

Method. For each of nb041's 18 trained cells (6 τ_GABA × 3 seeds):
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

Notebook entry: src/docs/src/pages/notebooks/nb046.mdx
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402
from cli import theme  # noqa: E402

sys.path.insert(0, str(REPO / "src" / "notebooks"))
from nb042 import _load_trained_full  # noqa: E402

SLUG = "nb046"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG

NB041_ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / "nb041"
NB041_NUMBERS = (
    REPO / "src" / "docs" / "public" / "figures" / "notebooks" / "nb041"
    / "numbers.json"
)

TAU_GABA_SWEEP_MS: tuple[float, ...] = (4.5, 6.0, 9.0, 12.0, 18.0, 27.0)
SEEDS: tuple[int, ...] = (42, 43, 44)

TIER_CONFIG = {
    "extra small": dict(),
    "small": dict(),
    "medium": dict(),
    "large": dict(),
    "extra large": dict(),
}
DEFAULT_TIER = "medium"


def tau_label(tau_ms: float) -> str:
    return f"tg{f'{tau_ms:g}'.replace('.', 'p')}"


def nb041_cell_dir(tau_ms: float, seed: int) -> Path:
    return NB041_ARTIFACTS / f"ping__{tau_label(tau_ms)}__seed{seed}"


def load_nb041_f_gamma() -> dict[tuple[float, int], float]:
    if not NB041_NUMBERS.exists():
        raise SystemExit(
            f"missing nb041 numbers.json at {NB041_NUMBERS}; "
            "re-render nb041 to produce it."
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


def evaluate_cell(train_dir: Path, f_gamma_hz: float, device) -> dict:
    """Run inference; accumulate spikes/cell/cycle distribution."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    import models as M
    from cli import EVAL_SEED, encode_batch

    net, cfg, X_te, y_te = _load_trained_full(train_dir, device)
    tau_gaba_ms = float(cfg.get("tau_gaba_ms") or 9.0)
    import models as M2
    M2.tau_gaba = tau_gaba_ms
    M2.decay_gaba = float(np.exp(-M2.dt / tau_gaba_ms))
    dt_ms = float(M.dt)
    t_ms = float(cfg["t_ms"])
    n_e = M.N_HID

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=64,
    )
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)

    # Distribution of (spikes per cell per cycle) — clipped at 3+.
    bucket_counts = np.zeros(4, dtype=np.int64)  # 0, 1, 2, ≥3
    # Per-cell max spike rate (Hz), for the ceiling plot.
    per_cell_spike_total = np.zeros(n_e, dtype=np.int64)
    trial_count = 0
    cycle_count = 0
    correct = total = 0

    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
            logits = net(input_spikes=spk)
            correct += (logits.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
            # net.spike_record tensors are shaped (T, B, N).
            s_e = net.spike_record["hid"].detach().cpu().numpy().astype(np.int8)
            s_i = net.spike_record["inh"].detach().cpu().numpy().astype(np.int8)
            T, B, _ = s_e.shape
            per_cell_spike_total += s_e.sum(axis=(0, 1)).astype(np.int64)
            trial_count += B
            for b in range(B):
                peaks = detect_i_burst_steps(s_i[:, b, :], dt_ms, f_gamma_hz)
                if len(peaks) == 0:
                    continue
                counts = count_e_spikes_per_cycle(s_e[:, b, :], peaks)
                # counts shape: (K, N_E)
                cycle_count += counts.shape[0]
                # Bucket each (cell, cycle) count into {0, 1, 2, ≥3}.
                flat = counts.ravel()
                bucket_counts[0] += int((flat == 0).sum())
                bucket_counts[1] += int((flat == 1).sum())
                bucket_counts[2] += int((flat == 2).sum())
                bucket_counts[3] += int((flat >= 3).sum())

    # Per-cell mean firing rate (Hz) over all trials.
    sec_per_trial = t_ms / 1000.0
    per_cell_rate_hz = per_cell_spike_total / (trial_count * sec_per_trial)
    return {
        "tau_gaba_ms": tau_gaba_ms,
        "f_gamma_hz": float(f_gamma_hz),
        "acc": 100.0 * correct / total,
        "n_trials": int(trial_count),
        "n_cycles_observed": int(cycle_count),
        "bucket_counts": bucket_counts.tolist(),
        "per_cell_rate_hz": per_cell_rate_hz.tolist(),
        "per_cell_max_rate_hz": float(per_cell_rate_hz.max()),
        "per_cell_median_rate_hz": float(np.median(per_cell_rate_hz)),
    }


# ─── plotting ───────────────────────────────────────────────────────


def _stamp(fig, run_id: str) -> None:
    fig.text(
        0.995, 0.005, run_id,
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


def plot_distribution(rows: list[dict], out_path: Path, run_id: str) -> dict:
    """Bar plot of P(spikes/cell/cycle = k) per τ_GABA, k ∈ {0, 1, 2, ≥3}."""
    theme.apply()
    by_tau: dict[float, list[dict]] = {}
    for r in rows:
        by_tau.setdefault(r["tau_gaba_ms"], []).append(r)
    taus_sorted = sorted(by_tau.keys())
    n = len(taus_sorted)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 4.5), dpi=150, sharey=True)
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
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return summary


def plot_ceiling_vs_fgamma(rows: list[dict], out_path: Path, run_id: str) -> dict:
    """Per-cell max E rate vs f_γ. y = f_γ line is the 1-spike/cycle ceiling."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)

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
            label="y = 0.20 · f_γ (nb041 slope p)")

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
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "max_cell_slope_vs_fgamma": slope_max,
        "max_cell_r2": r2_max,
    }


# ─── main ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default=DEFAULT_TIER)
    parser.add_argument("--no-wipe-dir", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--tau-gabas", nargs="*", type=float,
                        default=list(TAU_GABA_SWEEP_MS))
    args = parser.parse_args()

    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    print(f"notebook_run_id = (allocating) tier={tier}")

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id}")
    print(f"  τ_GABAs: {args.tau_gabas}")
    print(f"  seeds:   {args.seeds}")

    if not args.no_wipe_dir:
        for d in (ARTIFACTS, FIGURES):
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)

    from cli import _auto_device
    device = _auto_device()
    print(f"device = {device}")

    f_gamma_map = load_nb041_f_gamma()
    print(f"loaded f_γ for {len(f_gamma_map)} nb041 cells")

    rows: list[dict] = []
    n_cells = len(args.tau_gabas) * len(args.seeds)
    cells_done = 0
    for tau in args.tau_gabas:
        for seed in args.seeds:
            train_dir = nb041_cell_dir(tau, seed)
            if not (train_dir / "weights.pth").exists():
                print(f"[skip] missing {train_dir.relative_to(REPO)}")
                continue
            f_gamma = f_gamma_map.get((tau, seed))
            if f_gamma is None or f_gamma <= 0:
                print(f"[skip] no f_γ for τ_GABA={tau}, seed={seed}")
                continue
            t0 = time.monotonic()
            res = evaluate_cell(train_dir, f_gamma, device)
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
        rows, FIGURES / "spikes_per_cycle_distribution.png", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'spikes_per_cycle_distribution.png'}")
    ceil_summary = plot_ceiling_vs_fgamma(
        rows, FIGURES / "ceiling_vs_fgamma.png", notebook_run_id,
    )
    print(
        f"wrote {FIGURES / 'ceiling_vs_fgamma.png'}  "
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
        "tier": tier,
        "config": {
            "tau_gabas_ms": list(args.tau_gabas),
            "seeds": list(args.seeds),
            "nb041_source": "ping__tg{N}__seed{S} (100-epoch baselines)",
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
