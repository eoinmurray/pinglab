"""Notebook runner for entry 064 — the gradient-dampening asymmetry: PING needs
it, COBA does not.

Claim under test (ar010 item 1, "damping requirements per architecture"):
the surrogate-gradient voltage dampening (--v-grad-dampen, theory in ar006) is
NOT a global training trick — it is specifically what the recurrent E→I→E loop
needs. Concretely:

  • COBA (loop off, ei_strength = 0) trains fine at dampen = 1 — accuracy at
    dampen = 1 matches accuracy at the canonical dampen = 1000.
  • PING (loop on, ei_strength = 1) does NOT train at dampen = 1 — its BPTT
    gradient explodes through the inhibitory feedback and accuracy collapses;
    it only trains once dampening is applied.

Method: train both architectures across a dampening ladder
{1, 10, 100, 1000} × 1 seed on MNIST (each cell is otherwise the canonical
nb025 recipe for that architecture — only --v-grad-dampen varies), then read
best accuracy and the per-epoch gradient norm from each cell's metrics.json.
The signature is a double dissociation: COBA's accuracy curve is flat across the
ladder while PING's rises from chance; COBA's gradient norm stays bounded while
PING's blows up at low dampening.

Everything is shelled out to cli.py train (no cli/models imports); the notebook
only reads the emitted artifacts. These cells are bespoke to this validation
(deliberately-unstable PING included), so they are trained here rather than in
the nb022 hub.

Notebook entry: src/docs/content/notebooks/nb064.mdx
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from helpers import theme  # noqa: E402
from helpers.figsave import save_figure  # noqa: E402
from helpers.fmt import format_duration  # noqa: E402
from helpers.modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402

SLUG = "nb064"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
PINGLAB_CLI = REPO / "src" / "cli" / "cli.py"

T_MS = 200.0
DT_TRAIN = 0.1

# Dampening ladder. 1 = no dampening (the null); 1000 = the canonical nb025
# training value. Log-spaced so the transition (if any) is visible.
DAMPEN_SWEEP: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0)
MODELS: tuple[str, ...] = ("coba", "ping")
# One seed: the claim is a large qualitative dissociation (chance vs converged),
# not an effect-size measurement, so a single seed demonstrates it. Add more here
# if a replicate is wanted — the aggregation + error bars already handle n > 1.
SEEDS: tuple[int, ...] = (42,)

# MNIST 10-way — chance accuracy, the floor a collapsed cell falls to.
CHANCE_ACC = 10.0

# Epochs is a convergence CONTROL here, not a budget dial: the claim is "does
# each architecture reach its fate?", so every cell trains for the same fixed
# number of epochs — a low epoch budget could make PING look untrainable for
# the wrong reason.
EPOCHS = 30
# This runner's declared data budget (the tier system is retired; each
# notebook decides its own scale). 500 samples/epoch is enough for the
# qualitative dissociation — chance vs converged — this entry claims; bump
# this literal if a replicate at larger N is wanted.
MAX_SAMPLES = 500

# Canonical nb025 recipes, verbatim EXCEPT --v-grad-dampen (swept here, so it is
# omitted and supplied per-cell). Only ei_strength (loop on/off) and the matched
# per-architecture w_in / readout scale differ between the two — the same choices
# the trained baselines use, so this asks "does each architecture, at its own
# recipe, need dampening?".
MODEL_RECIPES: dict[str, dict[str, str]] = {
    "coba": {
        "--ei-strength": "0",
        "--w-in": "0.3",
        "--w-in-sparsity": "0.95",
        "--readout": "mem-mean",
        "--surrogate-slope": "1",
        "--readout-w-out-scale": "100",
        "--lr": "0.0004",
        "--batch-size": "256",
    },
    "ping": {
        "--ei-strength": "1",
        "--w-in": "1.2",
        "--w-in-sparsity": "0.95",
        "--readout": "mem-mean",
        "--surrogate-slope": "1",
        "--readout-w-out-scale": "500",
        "--lr": "0.0004",
        "--batch-size": "256",
    },
}

# Run scale — declared once, stamped into the figures-dir manifest by
# run_dirs.prepare, and rendered as the Methods table in nb064.mdx via the
# RunScale component. Single source of truth: the mdx never restates these
# numbers by hand. Display-facing strings (dataset, grid) are written for
# the table; numeric fields are what the runner actually passes to train.
SCALE = {
    "dataset": "MNIST (10-way)",
    "max_samples": MAX_SAMPLES,
    "epochs": EPOCHS,
    "t_ms": T_MS,
    "dt_ms": DT_TRAIN,
    "batch_size": int(MODEL_RECIPES["coba"]["--batch-size"]),  # same for both
    "hidden": 1024,  # pinglab-cli's MNIST default (not overridden here)
    "seeds": len(SEEDS),
    "cells": len(MODELS) * len(DAMPEN_SWEEP) * len(SEEDS),
    "grid": "2 architectures × 4 dampenings × 1 seed",
}

MODEL_COLORS = {"coba": theme.DEEP_RED, "ping": theme.INK_BLACK}
MODEL_MARKERS = {"coba": "s", "ping": "D"}


def dampen_label(dampen: float) -> str:
    return "d" + f"{dampen:g}".replace(".", "p")


def cell_dir(model: str, dampen: float, seed: int) -> Path:
    return ARTIFACTS / "train" / f"{model}__{dampen_label(dampen)}__seed{seed}"


def build_train_args(model: str, dampen: float, seed: int,
                     out_dir: Path) -> list[str]:
    """Compose a `train` invocation for one (model, dampen, seed) cell."""
    args = [
        "train",
        "--model", "ping",  # both architectures are built as the PING topology
        "--dataset", "mnist",
        "--max-samples", str(MAX_SAMPLES),
        "--epochs", str(EPOCHS),
        "--t-ms", str(T_MS),
        "--dt", str(DT_TRAIN),
        "--seed", str(seed),
        "--v-grad-dampen", str(dampen),   # the swept knob
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    for k, v in MODEL_RECIPES[model].items():
        args += [k, v]
    return args


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def _finite(x) -> float:
    """Coerce to float; map None / NaN / inf to NaN so plots + criteria can
    treat a diverged cell uniformly."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def extract_cell(model: str, dampen: float, seed: int) -> dict:
    """Pull the load-bearing scalars from one trained cell's metrics.json:
    best accuracy (did it learn) and the gradient norm (did BPTT stay sane)."""
    m = load_metrics(cell_dir(model, dampen, seed))
    epochs = m.get("epochs", [])
    grad_norms = [_finite(e.get("grad_norm")) for e in epochs]
    grad_norms = [g for g in grad_norms if np.isfinite(g)]
    return {
        "model": model,
        "dampen": float(dampen),
        "seed": seed,
        "best_acc": _finite(m.get("best_acc")),
        "final_acc": _finite(epochs[-1].get("acc")) if epochs else float("nan"),
        "grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else float("nan"),
        "grad_norm_max": float(np.max(grad_norms)) if grad_norms else float("nan"),
    }


def _agg(
    rows: list[dict], model: str, key: str
) -> tuple[list[float], list[float], list[float]]:
    """Per-dampen (mean, sem, all-seeds) of `key` for one model, in ladder order."""
    xs, mus, sems = [], [], []
    for d in DAMPEN_SWEEP:
        vals = [r[key] for r in rows if r["model"] == model and r["dampen"] == d
                and np.isfinite(r[key])]
        if not vals:
            continue
        xs.append(d)
        mus.append(float(np.mean(vals)))
        sems.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
    return xs, mus, sems


def fig_accuracy_vs_dampen(rows: list[dict], out_path: Path, run_id: str) -> None:
    """Best accuracy vs dampening, one line per architecture. The claim's
    positive half: COBA flat across the ladder, PING rising from chance."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in MODELS:
        xs, mus, sems = _agg(rows, model, "best_acc")
        ax.errorbar(
            xs, mus, yerr=sems, fmt=MODEL_MARKERS[model] + "-",
            color=MODEL_COLORS[model], capsize=3, lw=1.4, markersize=6,
            label=model.upper(),
        )
    ax.axhline(CHANCE_ACC, color=theme.FAINT, lw=0.8, ls=":", zorder=1)
    ax.text(DAMPEN_SWEEP[0], CHANCE_ACC + 1.5, "chance",
            fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("Gradient dampening  1 / ∂v-scale  (--v-grad-dampen)",
                  fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Best test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="center right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, lw=0.4)
    fig.suptitle("Only PING needs gradient dampening to train",
                 fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def fig_gradnorm_vs_dampen(rows: list[dict], out_path: Path, run_id: str) -> None:
    """Mean gradient norm vs dampening (log–log). The claim's mechanism:
    PING's BPTT gradient explodes at low dampening; COBA's stays bounded."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in MODELS:
        xs, mus, sems = _agg(rows, model, "grad_norm_mean")
        ax.errorbar(
            xs, mus, yerr=sems, fmt=MODEL_MARKERS[model] + "-",
            color=MODEL_COLORS[model], capsize=3, lw=1.4, markersize=6,
            label=model.upper(),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Gradient dampening (--v-grad-dampen)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Mean per-epoch gradient norm", fontsize=theme.SIZE_LABEL)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, lw=0.4)
    fig.suptitle("PING's BPTT gradient explodes without dampening; COBA's does not",
                 fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def evaluate_success(rows: list[dict]) -> dict:
    """Reduce the sweep to the paper's binary claim.

    COBA-does-not-need-it: COBA best_acc at dampen = 1 is within TOL of its best
        across the ladder.
    PING-needs-it: PING best_acc at dampen = 1 is near chance AND far below PING
        at the canonical dampen = 1000.
    """
    def mean_at(model: str, dampen: float, key: str = "best_acc") -> float:
        vals = [r[key] for r in rows if r["model"] == model and r["dampen"] == dampen
                and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    TOL_PP = 5.0              # "matches" tolerance, percentage points
    NEAR_CHANCE_PP = 15.0     # collapsed if within this of chance

    coba_lo = mean_at("coba", DAMPEN_SWEEP[0])
    coba_hi = max(mean_at("coba", d) for d in DAMPEN_SWEEP)
    ping_lo = mean_at("ping", DAMPEN_SWEEP[0])
    ping_hi = mean_at("ping", DAMPEN_SWEEP[-1])

    coba_ok = (
        np.isfinite(coba_lo) and np.isfinite(coba_hi)
        and (coba_hi - coba_lo) <= TOL_PP
    )
    ping_ok = (
        np.isfinite(ping_lo) and np.isfinite(ping_hi)
        and ping_lo <= CHANCE_ACC + NEAR_CHANCE_PP
        and (ping_hi - ping_lo) > TOL_PP
    )
    return {
        "coba_acc_at_dampen1": coba_lo,
        "coba_acc_best": coba_hi,
        "ping_acc_at_dampen1": ping_lo,
        "ping_acc_at_dampen1000": ping_hi,
        "coba_does_not_need_dampening": bool(coba_ok),
        "ping_needs_dampening": bool(ping_ok),
        "claim_validated": bool(coba_ok and ping_ok),
    }


def main() -> None:
    theme.set_paper_mode(True)

    modal_gpu = parse_modal_gpu(sys.argv)
    skip_training = "--skip-training" in sys.argv
    only_missing = "--only-missing" in sys.argv
    wipe_dir = "--no-wipe-dir" not in sys.argv

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(
        f"notebook_run_id = {notebook_run_id} cells={SCALE['cells']}"
        + ("  [skip-training]" if skip_training else "")
        + (f"  [modal:{modal_gpu}]" if modal_gpu else "")
    )

    prepare_run_dirs(
        SLUG, notebook_run_id, wipe=wipe_dir, skip_training=skip_training,
        make_artifacts=True, scale=SCALE,
        host=f"modal:{modal_gpu}" if modal_gpu else "local",
    )

    # ── Train the dampen × model × seed grid (bespoke cells, trained here) ──
    if not skip_training:
        dispatcher = BatchDispatcher(modal_gpu, REPO, PINGLAB_CLI)
        for model in MODELS:
            for dampen in DAMPEN_SWEEP:
                for seed in SEEDS:
                    out = cell_dir(model, dampen, seed)
                    if only_missing and (out / "metrics.json").exists():
                        print(f"[skip] {out.name} already trained")
                        continue
                    print(f"[train] {out.name}"
                          + (f"  [modal:{modal_gpu}]" if modal_gpu else ""))
                    dispatcher.submit(
                        build_train_args(model, dampen, seed, out), out,
                    )
        dispatcher.drain()

    # ── Read artifacts ─────────────────────────────────────────────────────
    rows: list[dict] = []
    for model in MODELS:
        for dampen in DAMPEN_SWEEP:
            for seed in SEEDS:
                run_dir = cell_dir(model, dampen, seed)
                if not (run_dir / "metrics.json").exists():
                    raise SystemExit(f"missing metrics: {run_dir / 'metrics.json'}")
                r = extract_cell(model, dampen, seed)
                rows.append(r)
                print(
                    f"  {model:<4} dampen={dampen:>7.1f} seed={seed}  "
                    f"best_acc={r['best_acc']:5.2f}%  "
                    f"grad_norm≈{r['grad_norm_mean']:.2e}"
                )

    fig_accuracy_vs_dampen(rows, FIGURES / "accuracy_vs_dampen", notebook_run_id)
    print(f"wrote {FIGURES / 'accuracy_vs_dampen'}.{{svg,pdf}}")
    fig_gradnorm_vs_dampen(rows, FIGURES / "gradnorm_vs_dampen", notebook_run_id)
    print(f"wrote {FIGURES / 'gradnorm_vs_dampen'}.{{svg,pdf}}")

    verdict = evaluate_success(rows)
    print("  success criteria:")
    print(f"    COBA does not need dampening: {verdict['coba_does_not_need_dampening']}"
          f"  (acc@1={verdict['coba_acc_at_dampen1']:.1f}%, "
          f"best={verdict['coba_acc_best']:.1f}%)")
    print(f"    PING needs dampening:         {verdict['ping_needs_dampening']}"
          f"  (acc@1={verdict['ping_acc_at_dampen1']:.1f}%, "
          f"acc@1000={verdict['ping_acc_at_dampen1000']:.1f}%)")
    print(f"    CLAIM VALIDATED: {verdict['claim_validated']}")

    duration_s = time.monotonic() - t_start
    train_cfg = load_config(cell_dir(MODELS[0], DAMPEN_SWEEP[0], SEEDS[0]))
    summary = {
        "notebook_run_id": notebook_run_id,
        "git_sha": train_cfg.get("git_sha"),
        "duration_s": round(duration_s, 1),
        "duration": format_duration(duration_s),
        "config": {
            "dataset": "mnist",
            "dampen_sweep": list(DAMPEN_SWEEP),
            "models": list(MODELS),
            "seeds": list(SEEDS),
            "max_samples": MAX_SAMPLES,
            "epochs": EPOCHS,
            "t_ms": T_MS,
            "dt": DT_TRAIN,
        },
        "success": verdict,
        "results": rows,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")


if __name__ == "__main__":
    main()
