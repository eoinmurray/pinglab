"""Notebook runner for entry 045 — Jitter inflection tracks 1/f_γ across τ_GABA.

Cross-cell extension of nb042's jitter sweep. Loads each of nb041's
18 trained PING cells (6 τ_GABA × 3 seeds), reads the cell's measured
f_γ from nb041's numbers.json, runs the cycle-coherent jitter sweep
at inference using the cell's own 1/f_γ as the binning period, and
tests whether the inflection point scales as 1/f_γ across the sweep.

Three figures:
1. raw_sweeps.png — E rate vs σ overlaid, one curve per τ_GABA.
2. dimensional_collapse.png — same data with x-axis rescaled by f_γ
   (units of "fraction of a cycle"). If the law holds, all curves
   should collapse onto a single sigmoid with inflection at σ·f_γ ≈ 1.
3. inflection_vs_period.png — extracted inflection σ vs 1/f_γ. If
   the law holds, this is linear with slope ≈ 1.

No retraining. Pure inference on nb041's checkpoints. Local MPS.

Notebook entry: src/docs/src/pages/notebooks/nb045.mdx
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

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402
from pinglab import theme  # noqa: E402

# Reuse nb042's perturbation infrastructure.
sys.path.insert(0, str(REPO / "src" / "pinglab" / "notebooks"))
from nb042 import (  # noqa: E402
    _build_override,
    _load_trained_full,
    _make_i_override_fn,
    F_GAMMA_REFERENCE_HZ,
)

SLUG = "nb045"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG

# nb041 source: 6 τ_GABA × 3 seeds, all already trained at 100 epochs.
NB041_ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / "nb041"
NB041_NUMBERS = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / "nb041" / "numbers.json"

TAU_GABA_SWEEP_MS: tuple[float, ...] = (4.5, 6.0, 9.0, 12.0, 18.0, 27.0)
SEEDS: tuple[int, ...] = (42, 43, 44)

# σ sweep — same grid as nb042, but in *units of the cell's own cycle period*
# would be cleaner. For now use absolute ms; the dimensional-collapse plot
# rescales to σ·f_γ.
SIGMA_MS_SWEEP: tuple[float, ...] = (
    0.0, 1.0, 3.0, 7.0, 14.0, 21.0, 28.0, 42.0, 60.0, 100.0,
)

TIER_CONFIG = {
    "extra small": dict(),
    "small": dict(),
    "medium": dict(),
    "large": dict(),
    "extra large": dict(),
}
DEFAULT_TIER = "medium"


def tau_label(tau_ms: float) -> str:
    s = f"{tau_ms:g}".replace(".", "p")
    return f"tg{s}"


def nb041_cell_dir(tau_ms: float, seed: int) -> Path:
    return NB041_ARTIFACTS / f"ping__{tau_label(tau_ms)}__seed{seed}"


def load_nb041_f_gamma() -> dict[tuple[float, int], float]:
    """Return {(tau_gaba_ms, seed): f_gamma_hz} from nb041's numbers.json."""
    if not NB041_NUMBERS.exists():
        raise SystemExit(
            f"missing nb041 numbers.json at {NB041_NUMBERS}; "
            "re-render nb041 (skip-training, no-wipe-dir) to produce it."
        )
    data = json.loads(NB041_NUMBERS.read_text())
    out = {}
    for r in data.get("results", []):
        out[(float(r["tau_gaba_ms"]), int(r["seed"]))] = float(r["f_gamma_hz"])
    return out


# ─── per-cell jitter evaluation ─────────────────────────────────────


def evaluate_cell_at_sigma(
    train_dir: Path, sigma_ms: float, cycle_period_ms: float,
    device, seed_offset: int = 0,
) -> dict:
    """Run inference on a trained nb041 cell with cycle-coherent jitter σ.

    cycle_period_ms is the bin size for the jitter — this is the
    cross-cell parameter that nb042's single-cell version did not have.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    import models as M
    from cli import EVAL_SEED, encode_batch

    net, cfg, X_te, y_te = _load_trained_full(train_dir, device)
    # nb041's trained cells save τ_GABA into config.json; nb025's don't.
    tau_gaba_ms = float(cfg.get("tau_gaba_ms") or 9.0)
    # Patch the module-level constant so the loaded network uses the
    # right synaptic decay during the forward pass (nb042's loader uses
    # patch_dt which resets it to the nb025 default of 9.0 ms).
    import models as M2
    M2.tau_gaba = tau_gaba_ms
    M2.decay_gaba = float(np.exp(-M2.dt / tau_gaba_ms))
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=64,
    )
    override_gen = torch.Generator().manual_seed(EVAL_SEED + 17 + seed_offset)
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)

    correct = total = 0
    e_spike_sum = i_spike_sum = 0.0
    n_e = M.N_HID
    n_i = M.N_INH or 1
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            spk = encode_batch(X_b, M.dt, False, generator=eval_gen)

            if sigma_ms <= 0.0:
                logits = net(input_spikes=spk)
            else:
                net._hidden_perturb_fn = None
                _ = net(input_spikes=spk)
                s_i_base = net.spike_record["inh"].detach().clone()
                cond = f"jitter_sigma_{sigma_ms:g}"
                override = _build_override(
                    s_i_base, cond, override_gen,
                    dt_ms=float(M.dt),
                    cycle_period_ms=cycle_period_ms,
                )
                fn = _make_i_override_fn(override)
                fn.reset()
                net._hidden_perturb_fn = fn
                logits = net(input_spikes=spk)
                net._hidden_perturb_fn = None

            correct += (logits.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
            e_spike_sum += float(net.spike_record["hid"].sum().item())
            i_spike_sum += float(net.spike_record["inh"].sum().item())

    t_sec = float(cfg["t_ms"]) / 1000.0
    return {
        "tau_gaba_ms": tau_gaba_ms,
        "sigma_ms": float(sigma_ms),
        "cycle_period_ms": float(cycle_period_ms),
        "acc": 100.0 * correct / total,
        "e_rate_hz": e_spike_sum / (total * n_e * t_sec),
        "i_rate_hz": i_spike_sum / (total * n_i * t_sec),
        "n_total": total,
    }


# ─── plotting ───────────────────────────────────────────────────────


def _stamp(fig, run_id: str) -> None:
    fig.text(
        0.995, 0.005, run_id,
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


def _aggregate(rows: list[dict]) -> dict:
    """Aggregate by (tau_gaba_ms, sigma_ms): mean ± sem across seeds."""
    by_key: dict[tuple[float, float], list[dict]] = {}
    for r in rows:
        by_key.setdefault((r["tau_gaba_ms"], r["sigma_ms"]), []).append(r)
    agg: dict = {}
    for key, group in by_key.items():
        e = [g["e_rate_hz"] for g in group]
        a = [g["acc"] for g in group]
        f = [g.get("f_gamma_hz", 0.0) for g in group]
        agg[key] = {
            "e_rate_mean": float(np.mean(e)),
            "e_rate_sem": float(np.std(e, ddof=1) / np.sqrt(len(e))
                                if len(e) > 1 else 0.0),
            "acc_mean": float(np.mean(a)),
            "f_gamma_mean": float(np.mean(f)),
        }
    return agg


def plot_raw_sweeps(rows: list[dict], out_path: Path, run_id: str) -> None:
    """E rate vs σ, one curve per τ_GABA, on a symlog x-axis."""
    theme.apply()
    agg = _aggregate(rows)
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=150)
    cmap = plt.get_cmap("viridis")
    taus_sorted = sorted({k[0] for k in agg.keys()})
    for i, tau in enumerate(taus_sorted):
        color = cmap(i / max(1, len(taus_sorted) - 1))
        sigmas = sorted({k[1] for k in agg.keys() if k[0] == tau})
        e_means = [agg[(tau, s)]["e_rate_mean"] for s in sigmas]
        e_sems = [agg[(tau, s)]["e_rate_sem"] for s in sigmas]
        f_gamma = agg[(tau, sigmas[0])]["f_gamma_mean"]
        ax.errorbar(
            sigmas, e_means, yerr=e_sems,
            marker="o", markersize=5, lw=1.2, color=color,
            capsize=3, label=f"τ_GABA = {tau:g} ms  (f_γ = {f_gamma:.0f} Hz)",
        )
    ax.set_xlabel("Cycle-coherent jitter σ (ms)",
                  fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Hidden E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, lw=0.4)
    fig.suptitle(
        "Jitter sweep across τ_GABA — raw σ axis",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dimensional_collapse(rows: list[dict], out_path: Path, run_id: str) -> None:
    """E rate (normalised) vs σ·f_γ. If the law holds, curves collapse
    onto a single sigmoid with inflection at σ·f_γ ≈ 1."""
    theme.apply()
    agg = _aggregate(rows)
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=150)
    cmap = plt.get_cmap("viridis")
    taus_sorted = sorted({k[0] for k in agg.keys()})

    for i, tau in enumerate(taus_sorted):
        color = cmap(i / max(1, len(taus_sorted) - 1))
        sigmas = sorted({k[1] for k in agg.keys() if k[0] == tau})
        f_gamma = agg[(tau, sigmas[0])]["f_gamma_mean"]
        # Normalise rate to (r - baseline) / (max - baseline).
        baseline = agg[(tau, sigmas[0])]["e_rate_mean"]
        rate_max = max(agg[(tau, s)]["e_rate_mean"] for s in sigmas)
        rate_range = max(rate_max - baseline, 1e-6)
        x_scaled = [s * f_gamma / 1000.0 for s in sigmas]  # σ·f_γ dimensionless
        y_norm = [
            (agg[(tau, s)]["e_rate_mean"] - baseline) / rate_range for s in sigmas
        ]
        ax.plot(
            x_scaled, y_norm,
            marker="o", markersize=5, lw=1.2, color=color,
            label=f"τ_GABA = {tau:g} ms",
        )

    # Annotate predicted inflection at σ·f_γ = 1.
    ax.axvline(1.0, color=theme.GREY_MID, lw=0.7, ls=":", alpha=0.8)
    ax.text(
        1.0, 0.97, " predicted inflection: σ·f_γ = 1",
        transform=ax.get_xaxis_transform(),
        fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED,
        ha="left", va="top",
    )
    ax.set_xlabel(
        "σ · f_γ  (dimensionless, units of one cycle)",
        fontsize=theme.SIZE_LABEL,
    )
    ax.set_ylabel(
        "Normalised E rate (r − baseline) / (max − baseline)",
        fontsize=theme.SIZE_LABEL,
    )
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, lw=0.4)
    ax.set_ylim(-0.05, 1.1)
    fig.suptitle(
        "Dimensional collapse — σ rescaled by f_γ",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_inflection_vs_period(rows: list[dict], out_path: Path, run_id: str) -> dict:
    """Extract per-curve inflection σ (σ at which normalised rate = 0.5)
    and plot vs 1/f_γ. Returns the slope of the linear fit through origin.
    """
    theme.apply()
    agg = _aggregate(rows)
    taus_sorted = sorted({k[0] for k in agg.keys()})
    inflections = []
    periods = []
    f_gammas = []
    for tau in taus_sorted:
        sigmas = sorted({k[1] for k in agg.keys() if k[0] == tau})
        baseline = agg[(tau, sigmas[0])]["e_rate_mean"]
        rate_max = max(agg[(tau, s)]["e_rate_mean"] for s in sigmas)
        rate_range = max(rate_max - baseline, 1e-6)
        ys = [
            (agg[(tau, s)]["e_rate_mean"] - baseline) / rate_range for s in sigmas
        ]
        # Linear interpolation to find σ where y = 0.5.
        target = 0.5
        sigma_inflection = None
        for k in range(len(sigmas) - 1):
            y0, y1 = ys[k], ys[k + 1]
            if (y0 <= target <= y1) or (y1 <= target <= y0):
                frac = (target - y0) / (y1 - y0 + 1e-12)
                sigma_inflection = sigmas[k] + frac * (sigmas[k + 1] - sigmas[k])
                break
        if sigma_inflection is None:
            continue
        f_gamma = agg[(tau, sigmas[0])]["f_gamma_mean"]
        period_ms = 1000.0 / f_gamma
        inflections.append(sigma_inflection)
        periods.append(period_ms)
        f_gammas.append(f_gamma)

    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=150)
    if inflections:
        ax.scatter(periods, inflections, s=60, color=theme.INK_BLACK, zorder=3,
                   label="measured inflection σ")
        # Fit through the origin: σ_inflection = α · (1/f_γ).
        p = np.array(periods)
        s = np.array(inflections)
        alpha = float(np.sum(p * s) / np.sum(p * p))
        ss_res = float(np.sum((s - alpha * p) ** 2))
        ss_tot = float(np.sum((s - s.mean()) ** 2)) if len(s) > 1 else 1.0
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        xs_fit = np.linspace(0, max(p) * 1.1, 100)
        ax.plot(xs_fit, alpha * xs_fit, color=theme.DEEP_RED, ls="--", lw=1.2,
                label=f"σ = α · (1/f_γ)  (α = {alpha:.2f}, R² = {r2:.3f})")
        # Reference line: slope = 1 (the law's prediction).
        ax.plot(xs_fit, xs_fit, color=theme.GREY_MID, ls=":", lw=0.8,
                label="predicted: σ = 1/f_γ")
    ax.set_xlabel("1/f_γ  (ms)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Measured inflection σ  (ms)", fontsize=theme.SIZE_LABEL)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, lw=0.4)
    fig.suptitle(
        "Inflection σ tracks 1/f_γ across τ_GABA",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"alpha": alpha if inflections else float("nan"),
            "r2": r2 if inflections else float("nan"),
            "n_points": len(inflections)}


# ─── main ───────────────────────────────────────────────────────────


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default=DEFAULT_TIER)
    parser.add_argument("--no-wipe-dir", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--tau-gabas", nargs="*", type=float,
                        default=list(TAU_GABA_SWEEP_MS))
    parser.add_argument("--sigmas", nargs="*", type=float,
                        default=list(SIGMA_MS_SWEEP))
    args = parser.parse_args()

    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    print(f"notebook_run_id = (allocating) tier={tier}")

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id}")
    print(f"  τ_GABAs: {args.tau_gabas}")
    print(f"  seeds:   {args.seeds}")
    print(f"  σ grid:  {args.sigmas}")

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
            cycle_period_ms = 1000.0 / f_gamma
            print(
                f"[cell] τ_GABA={tau:>5.1f}ms seed={seed}  "
                f"f_γ={f_gamma:5.2f} Hz  cycle={cycle_period_ms:.2f} ms"
            )
            for sigma in args.sigmas:
                t0 = time.monotonic()
                res = evaluate_cell_at_sigma(
                    train_dir, sigma, cycle_period_ms, device,
                    seed_offset=seed + int(sigma),
                )
                res["seed"] = seed
                res["f_gamma_hz"] = f_gamma
                rows.append(res)
                print(
                    f"    σ={sigma:>5.1f}ms  acc={res['acc']:5.2f}%  "
                    f"E={res['e_rate_hz']:6.2f} Hz  "
                    f"({time.monotonic() - t0:.1f}s)"
                )

    plot_raw_sweeps(rows, FIGURES / "raw_sweeps.png", notebook_run_id)
    print(f"wrote {FIGURES / 'raw_sweeps.png'}")
    plot_dimensional_collapse(
        rows, FIGURES / "dimensional_collapse.png", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'dimensional_collapse.png'}")
    fit = plot_inflection_vs_period(
        rows, FIGURES / "inflection_vs_period.png", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'inflection_vs_period.png'}  "
          f"(α = {fit['alpha']:.3f}, R² = {fit['r2']:.3f})")

    duration_s = time.monotonic() - t_start
    crits = [
        {
            "label": "raw sweeps rendered",
            "passed": (FIGURES / "raw_sweeps.png").exists(),
            "detail": f"{(FIGURES / 'raw_sweeps.png').stat().st_size} bytes"
                       if (FIGURES / "raw_sweeps.png").exists() else "missing",
        },
        {
            "label": "dimensional-collapse plot rendered",
            "passed": (FIGURES / "dimensional_collapse.png").exists(),
            "detail": f"{(FIGURES / 'dimensional_collapse.png').stat().st_size} bytes"
                       if (FIGURES / "dimensional_collapse.png").exists() else "missing",
        },
        {
            "label": "inflection-vs-period plot rendered",
            "passed": (FIGURES / "inflection_vs_period.png").exists(),
            "detail": (
                f"α = {fit['alpha']:.3f}, R² = {fit['r2']:.3f}, "
                f"n = {fit['n_points']}"
            ),
        },
        {
            "label": "linear-fit R² ≥ 0.8 (inflection σ scales linearly with 1/f_γ)",
            "passed": bool(fit["r2"] >= 0.8),
            "detail": f"R² = {fit['r2']:.3f}",
        },
        {
            "label": "α > 0 (positive scaling, i.e. inflection grows with cycle period)",
            "passed": bool(fit["alpha"] > 0),
            "detail": f"α = {fit['alpha']:.3f} (smoke prediction was α ≈ 1; "
                       f"empirical may differ — what matters is linearity)",
        },
    ]

    summary = {
        "notebook_run_id": notebook_run_id,
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "tier": tier,
        "config": {
            "tau_gabas_ms": args.tau_gabas,
            "seeds": args.seeds,
            "sigmas_ms": args.sigmas,
            "nb041_source": "ping__tg{N}__seed{S} (100-epoch baselines)",
        },
        "fit": fit,
        "results": rows,
        "success_criteria": crits,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")

    for c in crits:
        mark = "pass" if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['label']} — {c['detail']}")
    if any(not c["passed"] for c in crits):
        sys.exit(1)


if __name__ == "__main__":
    main()
