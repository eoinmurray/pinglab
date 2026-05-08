"""Notebook runner for entry 019 — spike-penalty θ_u sweep across cuba / coba / ping.

For each rung of the biophysical ladder (cuba, coba, ping), trains
the same calibrated recipe at six values of the upper-bound spike
budget θ_u: off (no penalty) plus θ_u ∈ {5, 2, 1, 0.5, 0.2}
spikes/trial = {25, 10, 5, 2.5, 1} Hz. Same recipe in every other
respect — only the regulariser flag changes — so each (model, θ_u)
cell is one point on that model's accuracy / rate Pareto frontier.

The fr-reg path was originally CUBANet-only — coba and ping run
through COBANet, which had no last_spike_counts hook. This notebook
exercises the matching hook added to COBANet.forward (mirrors the
CUBANet rate_counts list, gradient-attached), so the regulariser
applies uniformly across the ladder.

Notebook entry: src/docs/src/pages/notebooks/nb020.mdx
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402
from pinglab import theme  # noqa: E402

SLUG = "nb020"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope/__main__.py"

TIER_CONFIG = {
    "extra small": dict(max_samples=100, epochs=1),
    "small": dict(max_samples=500, epochs=5),
    "medium": dict(max_samples=2000, epochs=10),
    "large": dict(max_samples=5000, epochs=40),
    "extra large": dict(max_samples=10000, epochs=40),
}
DEFAULT_TIER = "small"
T_MS = 200.0
DT_TRAIN = 0.1
SEED = 42

# θ_u sweep grid in spikes-per-trial. None = no penalty (baseline).
# At T = 200 ms, spikes/trial × 5 = Hz.  The grid spans from no
# pressure (off → ~80–90 Hz baselines for cuba/coba) down to 1 Hz —
# below ping's natural 4 Hz and well into the regime where every
# model loses accuracy.
THETA_U_GRID: list[float | None] = [None, 5.0, 2.0, 1.0, 0.5, 0.2]
FR_STRENGTH_UPPER = 1e-3

MODELS = ["cuba", "coba", "ping"]

MODEL_RECIPES: dict[str, dict] = {
    # Recipes mirror nb010 (cuba) / nb011 (coba) / nb012 (ping). Drift
    # would invalidate the comparison: any per-model rate change must be
    # attributable to the penalty, not to a separate hyperparameter.
    "cuba": {
        "__build_as": "cuba",
        "--kaiming-init": True,
        "--readout": "mem-mean",
        "--surrogate-slope": "1",
        "--lr": "0.04",
        "--batch-size": "256",
    },
    "coba": {
        "__build_as": "ping",
        "--ei-strength": "0",
        "--v-grad-dampen": "1000",
        "--w-in": "0.3",
        "--w-in-sparsity": "0.95",
        "--readout": "mem-mean",
        "--surrogate-slope": "1",
        "--readout-w-out-scale": "100",
        "--lr": "0.0004",
        "--batch-size": "256",
    },
    "ping": {
        "__build_as": "ping",
        "--ei-strength": "1",
        "--v-grad-dampen": "1000",
        "--w-in": "1.2",
        "--w-in-sparsity": "0.95",
        "--readout": "mem-mean",
        "--surrogate-slope": "1",
        "--readout-w-out-scale": "500",
        "--lr": "0.0004",
        "--batch-size": "256",
    },
}

MODEL_COLORS = {
    "cuba": theme.DEEP_RED,
    "coba": theme.AMBER,
    "ping": theme.ELECTRIC_CYAN,
}

MODEL_MARKERS = {"cuba": "o", "coba": "s", "ping": "D"}

MIN_ACC_BY_TIER = {
    "extra small": 15.0,
    "small": 30.0,
    "medium": 50.0,
    "large": 70.0,
    "extra large": 70.0,
}


def theta_label(theta_u: float | None) -> str:
    """Filesystem-safe label for an out-dir / video filename."""
    if theta_u is None:
        return "off"
    s = f"{theta_u:g}".replace(".", "p")
    return f"tu{s}"


def theta_display(theta_u: float | None) -> str:
    """Human label for plots / numbers.json."""
    if theta_u is None:
        return "off"
    return f"{theta_u:g}"


def theta_hz(theta_u: float | None) -> float | None:
    if theta_u is None:
        return None
    return theta_u * (1000.0 / T_MS)


def cell_dir(model: str, theta_u: float | None) -> Path:
    return ARTIFACTS / f"{model}__{theta_label(theta_u)}"


def build_train_args(
    model: str, theta_u: float | None, tier: str, out_dir: Path
) -> list[str]:
    recipe = MODEL_RECIPES[model]
    args = [
        "train",
        "--model",
        recipe["__build_as"],
        "--dataset",
        "mnist",
        "--max-samples",
        str(TIER_CONFIG[tier]["max_samples"]),
        "--epochs",
        str(TIER_CONFIG[tier]["epochs"]),
        "--t-ms",
        str(T_MS),
        "--dt",
        str(DT_TRAIN),
        "--seed",
        str(SEED),
        "--observe",
        "video",
        "--frame-rate",
        "1",
        "--out-dir",
        str(out_dir),
        "--wipe-dir",
    ]
    for k, v in recipe.items():
        if k.startswith("__"):
            continue
        if v is True:
            args.append(k)
        elif v is not None:
            args += [k, v]
    if theta_u is not None:
        args += [
            "--fr-reg-upper-theta",
            str(theta_u),
            "--fr-reg-upper-strength",
            str(FR_STRENGTH_UPPER),
        ]
    return args


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def _stamp(fig, run_id: str) -> None:
    fig.text(
        0.995,
        0.005,
        run_id,
        ha="right",
        va="bottom",
        fontsize=theme.SIZE_CAPTION,
        color=theme.LABEL,
        family="monospace",
    )


def plot_frontier(rows: list[dict], out_path: Path, run_id: str) -> None:
    """Pareto-style frontier: one line per model from baseline (highest
    rate) toward the most-aggressive penalty (lowest rate). Both axes are
    the final-epoch state — best-acc from earlier epochs would pair with
    a rate from a different network state, which doesn't physically
    correspond to anything you can observe in the training video."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in MODELS:
        pts = sorted(
            (r for r in rows if r["model"] == model),
            key=lambda r: r["rate_e"],
        )
        xs = [p["rate_e"] for p in pts]
        ys = [p["final_acc"] for p in pts]
        ax.plot(
            xs, ys,
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            label=model,
        )
        for p in pts:
            ax.annotate(
                p["theta_display"],
                (p["rate_e"], p["final_acc"]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED,
            )
    ax.set_xlabel("hidden E firing rate (Hz, final epoch)")
    ax.set_ylabel("test accuracy (%, final epoch)")
    ax.set_title("Accuracy / rate frontier across the cuba → coba → ping ladder")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_acc_vs_theta(rows: list[dict], out_path: Path, run_id: str) -> None:
    """Two stacked panels: accuracy and achieved rate, both vs θ_u (Hz).
    Reads the regulariser as the independent variable (what was asked
    of the network) rather than the achieved rate (what came out)."""
    theme.apply()
    grid_hz = [theta_hz(t) or float("inf") for t in THETA_U_GRID]
    finite = [(t, h) for t, h in zip(THETA_U_GRID, grid_hz) if h != float("inf")]
    fig, (ax_acc, ax_rate) = plt.subplots(2, 1, figsize=(8, 9), sharex=True)
    for model in MODELS:
        accs_finite, rates_finite, xs_finite = [], [], []
        for t, h in finite:
            r = next(
                row for row in rows if row["model"] == model and row["theta_u"] == t
            )
            accs_finite.append(r["final_acc"])
            rates_finite.append(r["rate_e"])
            xs_finite.append(h)
        ax_acc.plot(
            xs_finite, accs_finite,
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model], label=model,
        )
        ax_rate.plot(
            xs_finite, rates_finite,
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model], label=model,
        )
        base = next(
            row for row in rows if row["model"] == model and row["theta_u"] is None
        )
        ax_acc.axhline(base["final_acc"], color=MODEL_COLORS[model], lw=0.8, ls=":", alpha=0.7)
        ax_rate.axhline(base["rate_e"], color=MODEL_COLORS[model], lw=0.8, ls=":", alpha=0.7)
    # diagonal: rate = θ_u
    budget_xs = sorted(xs_finite)
    ax_rate.plot(
        budget_xs, budget_xs,
        color=theme.LABEL, lw=1.0, ls="--",
        label="rate = θ_u",
    )
    ax_acc.set_ylabel("test accuracy (%, final epoch)")
    ax_acc.set_title("Accuracy vs spike budget θ_u")
    ax_acc.set_ylim(0, 100)
    ax_acc.grid(True, alpha=0.3, which="both")
    ax_acc.legend(loc="lower right")
    ax_rate.set_xlabel("θ_u (Hz)  [smaller = tighter budget]")
    ax_rate.set_ylabel("achieved rate (Hz)")
    ax_rate.set_title("Achieved rate vs spike budget — dotted lines mark each model's λ=off rate")
    ax_rate.grid(True, alpha=0.3)
    ax_rate.legend(loc="upper left")
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def evaluate_success(rows: list[dict], tier: str, figures: Path) -> list[dict]:
    floor = float(MIN_ACC_BY_TIER[tier])
    figs_root = figures.parents[2]

    def artifact(name: str, label: str) -> dict:
        path = figures / name
        ok = path.exists() and path.stat().st_size > 0
        href = "/" + str(path.relative_to(figs_root)) if ok else None
        return {
            "label": label,
            "passed": bool(ok),
            "detail": f"{path.name} ({path.stat().st_size} bytes)"
            if ok
            else f"missing {path.name}",
            "detail_href": href,
        }

    crits: list[dict] = [
        artifact("frontier.png", "frontier figure rendered"),
        artifact("acc_vs_theta.png", "θ_u-axis figure rendered"),
    ]
    for model in MODELS:
        for theta_u in THETA_U_GRID:
            crits.append(
                artifact(
                    f"training__{model}__{theta_label(theta_u)}.mp4",
                    f"{model} θ_u={theta_display(theta_u)}: training video",
                )
            )

    # baseline accuracy must clear the tier floor for every model
    for model in MODELS:
        base = next(
            r for r in rows if r["model"] == model and r["theta_u"] is None
        )
        crits.append(
            {
                "label": f"{model} baseline acc ≥ {floor:.0f}% ({tier} floor)",
                "passed": bool(base["best_acc"] >= floor),
                "detail": f"{model}={base['best_acc']:.2f}%",
            }
        )
    # The frontier should be monotone-ish: tighter budget cannot give
    # higher rate. Allow a tolerance because of stochastic optimisation.
    for model in MODELS:
        ordered = [
            next(r for r in rows if r["model"] == model and r["theta_u"] == t)
            for t in [tu for tu in THETA_U_GRID if tu is not None]
        ]
        # Sort by θ_u descending (loosest → tightest) and check rate is non-increasing.
        ordered.sort(key=lambda r: -r["theta_u"])
        rates = [r["rate_e"] for r in ordered]
        non_increasing = all(
            b <= a + 1.0 for a, b in zip(rates, rates[1:])
        )  # 1 Hz tolerance
        crits.append(
            {
                "label": f"{model} rate non-increasing as θ_u tightens (±1 Hz)",
                "passed": non_increasing,
                "detail": "rates(loose→tight): "
                + ", ".join(f"{r:.1f}" for r in rates),
            }
        )
    return crits


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def copy_video(run_dir: Path, out_path: Path) -> None:
    src = run_dir / "training.mp4"
    if not src.exists():
        raise SystemExit(f"missing training video: {src}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_path)
    print(f"wrote {out_path}")


def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    skip_training = "--skip-training" in sys.argv
    wipe_dir = "--no-wipe-dir" not in sys.argv

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    n_cells = len(MODELS) * len(THETA_U_GRID)
    print(
        f"notebook_run_id = {notebook_run_id} tier={tier} "
        f"cells={n_cells}"
        + ("  [skip-training]" if skip_training else "")
    )

    if wipe_dir:
        if skip_training:
            if FIGURES.exists():
                print(f"[wipe] {FIGURES.relative_to(REPO)}")
                shutil.rmtree(FIGURES)
        else:
            for d in (ARTIFACTS, FIGURES):
                if d.exists():
                    print(f"[wipe] {d.relative_to(REPO)}")
                    shutil.rmtree(d)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)

    if not skip_training:
        dispatcher = BatchDispatcher(modal_gpu, REPO, OSCILLOSCOPE)
        for model in MODELS:
            for theta_u in THETA_U_GRID:
                out = cell_dir(model, theta_u)
                build_as = MODEL_RECIPES[model]["__build_as"]
                gpu_override = None
                if modal_gpu in ("T4", "L4", "A10G") and build_as == "ping":
                    gpu_override = "A100"
                print(
                    f"[train] {model}/θ_u={theta_display(theta_u)} → "
                    f"{out.relative_to(REPO)}"
                    + (f"  [modal:{modal_gpu}]" if modal_gpu else "")
                )
                dispatcher.submit(
                    build_train_args(model, theta_u, tier, out),
                    out,
                    gpu_override=gpu_override,
                )
        dispatcher.drain()

    rows: list[dict] = []
    for model in MODELS:
        for theta_u in THETA_U_GRID:
            run_dir = cell_dir(model, theta_u)
            if not (run_dir / "metrics.json").exists():
                raise SystemExit(f"missing metrics: {run_dir / 'metrics.json'}")
            metrics = load_metrics(run_dir)
            last = metrics["epochs"][-1]
            rows.append(
                {
                    "model": model,
                    "theta_u": theta_u,
                    "theta_display": theta_display(theta_u),
                    "theta_u_hz": theta_hz(theta_u),
                    "best_acc": float(metrics["best_acc"]),
                    "best_epoch": int(metrics["best_epoch"]),
                    "final_acc": float(last["acc"]),
                    "rate_e": float(last.get("rate_e") or 0.0),
                }
            )
            copy_video(
                run_dir,
                FIGURES / f"training__{model}__{theta_label(theta_u)}.mp4",
            )

    print("  results:")
    for r in rows:
        theta_str = (
            f"θ_u={r['theta_display']:>4} ({r['theta_u_hz']:>4.1f} Hz)"
            if r["theta_u"] is not None
            else "θ_u= off"
        )
        print(
            f"    {r['model']:<5}  {theta_str}  "
            f"acc(final)={r['final_acc']:6.2f}%  best={r['best_acc']:6.2f}%  "
            f"rate_e={r['rate_e']:6.1f} Hz"
        )

    plot_frontier(rows, FIGURES / "frontier.png", notebook_run_id)
    print(f"wrote {FIGURES / 'frontier.png'}")
    plot_acc_vs_theta(rows, FIGURES / "acc_vs_theta.png", notebook_run_id)
    print(f"wrote {FIGURES / 'acc_vs_theta.png'}")

    duration_s = time.monotonic() - t_start
    train_cfg = load_config(cell_dir(MODELS[0], None))
    crits = evaluate_success(rows, tier, FIGURES)
    summary = {
        "notebook_run_id": notebook_run_id,
        "git_sha": train_cfg.get("git_sha"),
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "tier": tier,
        "config": {
            "tier": tier,
            "dataset": "mnist",
            "models": MODELS,
            "theta_u_grid_spikes": [t for t in THETA_U_GRID if t is not None],
            "theta_u_grid_hz": [
                theta_hz(t) for t in THETA_U_GRID if t is not None
            ],
            "max_samples": TIER_CONFIG[tier]["max_samples"],
            "epochs": TIER_CONFIG[tier]["epochs"],
            "t_ms": T_MS,
            "dt": DT_TRAIN,
            "seed": SEED,
            "fr_strength_upper": FR_STRENGTH_UPPER,
        },
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
