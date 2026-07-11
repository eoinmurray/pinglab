"""Notebook runner for entry 062 — is Dale's law the implicit stabiliser?

The SHD program's second stability experiment (plan: ar063). exp061 killed the
Δt hypothesis: the free signed-recurrent net's NaN divergence is a gradient
explosion over the recurrent BPTT unroll, not coarse-Δt stiffness, and it gets
worse, not better, with finer Δt. So what, if anything, keeps a matched net in a
trainable regime? The plan's next candidate is Dale's law: its non-negativity
projection bounds the recurrent weights (and fixes their sign), which may hold
the loop gain below the runaway threshold the free net crosses.

This tests it directly. Two cells, identical in every way except the constraint —
free (signed, --no-dales-law, the exp060/exp061 setting that diverges) vs
constrained (--dales-law) — at the coarse Δt = 1.0 ms where the free net is known
to NaN, single seed, full exp060 scale. Measures the plan's registered contrast:
NaN-epoch rate, max recurrent-weight norm, and best accuracy, free vs constrained.

  KILL: if the Dale's-law net also NaNs at matched settings, the constraint is
  not what stabilises — the free-vs-constrained difference lies elsewhere.

Compute: RunPod fan-out (one pod per cell), results collected over the S3 volume
API (helpers/runpod.py); local-CPU + plumbing scales for cheap iteration.

Writing: writings/exp062.typ · figures + numbers.json: artifacts/data/exp062/
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import runpod, theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.figsave import save_figure  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp062"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
SNN_TOOL = REPO / "tools" / "snn" / "tool.py"

N_CLASSES = 20
CHANCE_PCT = 100.0 / N_CLASSES  # 5% on 20 classes

# ── The contrast and the fixed recipe ────────────────────────────────
# Dale's law is the ONLY thing that differs between the two cells. Everything
# else is exp060/exp061's Rung A recipe verbatim, at the coarse Δt = 1.0 ms where
# the free net is known to diverge (exp061: 13/30 NaN epochs), so any difference
# in the stability metrics is attributable to the constraint alone.
SEED = 42
DT_MS = 1.0
T_MS = 1000.0
SUBSET = 1000
EPOCHS = 30

PLUMBING_SUBSET = 64
PLUMBING_EPOCHS = 2
LOCAL_SUBSET = 128
LOCAL_EPOCHS = 15

# Recipe shared by both cells (minus the Dale's-law flag, added per cell).
RECIPE: list[str] = [
    "--dataset", "shd",
    "--t-ms", str(int(T_MS)),
    "--dt", str(DT_MS),
    "--n-hidden", "256",
    "--model", "ping",
    "--batch-size", "32",
    "--lr", "0.001",
    "--weight-decay", "0.001",
    "--w-ee", "0.3", "0.1",
    "--trainable-w-ee", "--trainable-w-ei", "--trainable-w-ie", "--trainable-w-ii",
    "--fr-reg-upper-theta", "100",
    "--fr-reg-upper-strength", "0.06",
]

# Two cells: free (signed) vs constrained (Dale's law). dales_law drives the flag.
CELLS: list[dict] = [
    {"name": f"free__seed{SEED}", "dales_law": False, "label": "free (signed)"},
    {"name": f"dales__seed{SEED}", "dales_law": True, "label": "Dale's law"},
]

SCALE = {
    "dataset": "shd",
    "contrast": "free (--no-dales-law) vs constrained (--dales-law)",
    "max_samples": SUBSET,
    "t_ms": T_MS,
    "dt_ms": DT_MS,
    "epochs": EPOCHS,
    "n_hidden": 256,
    "seeds": 1,
    "seed": SEED,
}

POD_VOLUME_SUBDIR = f"training/{SLUG}"
LOCAL_TRAINING_ROOT = REPO / "temp" / "experiments" / f"{SLUG}_cells"


def training_root() -> Path:
    return Path(os.environ.get("PINGLAB_TRAINING_ROOT", str(LOCAL_TRAINING_ROOT)))


def cell_dir(name: str) -> Path:
    return training_root() / name


def _plumbing() -> bool:
    return os.environ.get("PINGLAB_EXP062_PLUMBING") == "1"


def _local() -> bool:
    return os.environ.get("PINGLAB_EXP062_LOCAL") == "1"


def cell_samples_epochs() -> tuple[int, int]:
    if _plumbing():
        return PLUMBING_SUBSET, PLUMBING_EPOCHS
    if _local():
        return LOCAL_SUBSET, LOCAL_EPOCHS
    return SUBSET, EPOCHS


def _compute_label() -> str:
    if _plumbing():
        return "runpod-plumbing"
    if _local():
        return "local-cpu"
    return "runpod"


def build_train_args(cell: dict, out_dir: Path) -> list[str]:
    """CLI `train` args for one cell — the fixed recipe plus this cell's Dale's-law
    flag, seed, scale and output dir."""
    ms, ep = cell_samples_epochs()
    dales_flag = "--dales-law" if cell["dales_law"] else "--no-dales-law"
    return [
        "train",
        *RECIPE,
        dales_flag,
        "--seed", str(SEED),
        "--max-samples", str(ms),
        "--epochs", str(ep),
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]


def _cell_by_name(name: str) -> dict | None:
    return next((c for c in CELLS if c["name"] == name), None)


# ── RunPod fan-out (one pod per cell) ────────────────────────────────

def runpod_is_done(cell: dict) -> bool:
    """A cell is done iff its metrics.json exists AND was trained at the scale
    (max_samples, epochs, dales_law) THIS run expects."""
    p = cell_dir(cell["name"]) / "metrics.json"
    if not p.exists():
        return False
    try:
        cfg = json.loads(p.read_text()).get("config", {})
    except (json.JSONDecodeError, OSError):
        return False
    want_ms, want_ep = cell_samples_epochs()
    return (cfg.get("max_samples") == want_ms
            and cfg.get("epochs") == want_ep
            and cfg.get("dales_law") == cell["dales_law"])


def _train_one_cell(cell: dict) -> None:
    """Train ONE cell by invoking the SNN CLI directly (sys.executable, like
    exp022/exp061) so the pod's baked venv is used with no redundant uv sync."""
    ms, ep = cell_samples_epochs()
    print(f"[train-cell] {cell['name']} (dales_law={cell['dales_law']}, n={ms}, "
          f"{ep} ep) → {cell_dir(cell['name'])}")
    subprocess.run(
        [sys.executable, str(SNN_TOOL), *build_train_args(cell, cell_dir(cell["name"]))],
        cwd=REPO, check=True,
    )


def pod_run() -> None:
    """Pod-side entrypoint (image start script runs `exp062.py --pod-run`)."""
    print(f"[pod-run] plumbing={_plumbing()} root={training_root()}")

    def is_done(name: str) -> bool:
        c = _cell_by_name(name)
        return c is not None and runpod_is_done(c)

    def run_job(name: str) -> None:
        c = _cell_by_name(name)
        if c is not None:
            _train_one_cell(c)

    runpod.pod_run_loop(
        job_ids=[c["name"] for c in CELLS], is_done=is_done, run_job=run_job,
    )


def run_via_runpod(argv: list[str]) -> None:
    """`--runpod` dispatch: one pod per cell, fire-and-forget to the shared
    volume; collect over the S3 API (--runpod --collect). Dry-run by default."""
    meta = parse_meta(argv, allow_dispatch=True)
    buckets = [{"name": c["name"], "cells": [c["name"]]} for c in CELLS]
    runpod.dispatch(
        slug=SLUG, runner=SLUG,
        buckets=buckets,
        gpu=meta.gpu, live=meta.live, plumbing=meta.plumbing, collect=meta.collect,
        collect_subdir=POD_VOLUME_SUBDIR,
        local_collect_dir=str(LOCAL_TRAINING_ROOT),
        extra_env={"PINGLAB_TRAINING_ROOT": f"{runpod.VOLUME_MOUNT}/{POD_VOLUME_SUBDIR}"},
        plumbing_env={"PINGLAB_EXP062_PLUMBING": "1"},
    )


# ── Stability metrics (same reduction as exp061) ─────────────────────

def _wee_norm(weight_norms: dict) -> float:
    vals = [float(v) for k, v in weight_norms.items() if k.startswith("W_ee.")]
    return max(vals) if vals else float("nan")


def cell_stability(cell: dict) -> dict:
    """Reduce one cell's metrics.json to the plan's stability metrics: NaN-epoch
    rate (non-finite train/test loss or a NaN forward pass), max recurrent-weight
    norm, and best accuracy."""
    p = cell_dir(cell["name"]) / "metrics.json"
    if not p.exists():
        return {"trained": False, "name": cell["name"], "label": cell["label"],
                "dales_law": cell["dales_law"]}
    m = json.loads(p.read_text())
    eps = m.get("epochs") or []

    def _finite(x) -> bool:
        return isinstance(x, (int, float)) and math.isfinite(x)

    nan_epochs = sum(
        1 for e in eps
        if not _finite(e.get("loss")) or not _finite(e.get("test_loss"))
        or (e.get("nan_forward_batches") or 0) > 0
    )
    grad_maxes = [e.get("grad_norm_max", e.get("grad_norm")) for e in eps]
    grad_maxes = [g for g in grad_maxes if _finite(g)]
    wee_maxes = [_wee_norm(e.get("weight_norms", {})) for e in eps]
    wee_maxes = [w for w in wee_maxes if _finite(w)]
    return {
        "trained": True,
        "name": cell["name"],
        "label": cell["label"],
        "dales_law": cell["dales_law"],
        "epochs": len(eps),
        "nan_epochs": nan_epochs,
        "nan_epoch_rate": (nan_epochs / len(eps)) if eps else float("nan"),
        "nan_forward_batches": sum(e.get("nan_forward_batches", 0) or 0 for e in eps),
        "max_grad_norm": max(grad_maxes) if grad_maxes else float("nan"),
        "max_wee_norm": max(wee_maxes) if wee_maxes else float("nan"),
        "best_acc_pct": m.get("best_acc"),
        "best_epoch": m.get("best_epoch"),
    }


# ── Figures ──────────────────────────────────────────────────────────

def plot_comparison(stats: list[dict], stem: Path) -> None:
    """Free vs constrained across the three registered metrics: NaN-epoch rate,
    max recurrent-weight norm, best accuracy."""
    theme.apply()
    trained = [s for s in stats if s.get("trained")]
    labels = [s["label"] for s in trained]
    x = range(len(trained))
    colours = [theme.DEEP_RED if not s["dales_law"] else theme.INK_BLACK for s in trained]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), dpi=150)

    axes[0].bar(x, [100 * s["nan_epoch_rate"] for s in trained], color=colours, width=0.6)
    axes[0].set_ylabel("NaN-epoch rate (%)")

    axes[1].bar(x, [s["max_wee_norm"] for s in trained], color=colours, width=0.6)
    axes[1].set_ylabel("max ‖W_ee‖")

    axes[2].bar(x, [s["best_acc_pct"] for s in trained], color=colours, width=0.6)
    axes[2].axhline(CHANCE_PCT, color=theme.GREY_MID, lw=1.0, ls="--")
    axes[2].set_ylabel("best test accuracy (%)")

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def plot_loss_traces(stem: Path) -> None:
    """Per-epoch train loss, free vs constrained — NaN epochs show as gaps."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    for cell in CELLS:
        p = cell_dir(cell["name"]) / "metrics.json"
        if not p.exists():
            continue
        eps = json.loads(p.read_text()).get("epochs") or []
        x = [e["ep"] for e in eps]
        y = [e.get("test_loss") if isinstance(e.get("test_loss"), (int, float))
             and math.isfinite(e.get("test_loss")) else float("nan") for e in eps]
        colour = theme.DEEP_RED if not cell["dales_law"] else theme.INK_BLACK
        ax.plot(x, y, "-", color=colour, lw=1.8, label=cell["label"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("test cross-entropy loss")
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    meta = parse_meta(sys.argv, allow_dispatch=True)

    if meta.pod_run:
        pod_run()
        return
    if meta.reap:
        runpod.reap_all_pods()
        return
    if meta.runpod:
        run_via_runpod(sys.argv)
        return

    skip_training = meta.skip_training or meta.plot_only

    t_start = time.monotonic()
    run_id = next_run_id(SLUG)
    ms, ep = cell_samples_epochs()
    scale = {**SCALE, "max_samples": ms, "epochs": ep, "compute": _compute_label()}
    print(f"notebook_run_id = {run_id} cells={len(CELLS)} "
          f"scale=({ms} samples, {ep} ep, {_compute_label()})"
          + ("  [skip-training]" if skip_training else ""))

    with published_run(
        SLUG, run_id, make_artifacts=True, scale=scale,
        skip_training=skip_training, plot_only=meta.plot_only,
    ) as (_artifacts, figures):
        if not skip_training:
            LOCAL_TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
            for c in CELLS:
                if meta.only_missing and runpod_is_done(c):
                    print(f"[skip] {c['name']} already trained")
                    continue
                _train_one_cell(c)

        stats = [cell_stability(c) for c in CELLS]
        for s in stats:
            if s.get("trained"):
                print(f"  {s['label']:<14s} NaN-epochs={s['nan_epochs']}/{s['epochs']} "
                      f"max‖Wee‖={s['max_wee_norm']:.3g} best={s['best_acc_pct']}%")
            else:
                print(f"  {s['label']:<14s} [not trained]")

        plot_comparison(stats, figures / "free_vs_dales")
        print(f"wrote {figures / 'free_vs_dales.svg'}")
        plot_loss_traces(figures / "loss_traces")
        print(f"wrote {figures / 'loss_traces.svg'}")

        by_law = {s["dales_law"]: s for s in stats if s.get("trained")}
        free = by_law.get(False)
        dales = by_law.get(True)
        free_diverges = bool(free and free["nan_epochs"] > 0)
        dales_diverges = bool(dales and dales["nan_epochs"] > 0)
        # The hypothesis: Dale's law stabilises where the free net diverges. It
        # needs the free net to actually diverge (else the contrast is vacuous),
        # then asks whether the constrained net is NaN-free.
        if not (free and dales):
            verdict = "incomplete — both cells must train to compare"
        elif not free_diverges:
            verdict = ("inconclusive — the free net did not diverge at this scale, so "
                       "there is nothing for the constraint to stabilise (needs the "
                       "full scale where the free net NaNs)")
        elif dales_diverges:
            verdict = ("kill — the Dale's-law net also diverges at matched settings, so "
                       "the constraint is not what stabilises")
        else:
            verdict = ("supported — the free net diverges and the Dale's-law net trains "
                       "NaN-free at matched settings")

        payload = {
            "seed": SEED,
            "dt_ms": DT_MS,
            "max_samples": ms,
            "epochs": ep,
            "compute": _compute_label(),
            "n_cells": len(CELLS),
            "n_trained": sum(1 for s in stats if s.get("trained")),
            "cells": stats,
            "free_diverges": free_diverges,
            "dales_diverges": dales_diverges,
            "verdict": verdict,
            "chance_pct": CHANCE_PCT,
        }
        duration_s = time.monotonic() - t_start
        write_numbers(figures, run_id=run_id, duration_s=duration_s, payload=payload)
        print(f"wrote {figures / 'numbers.json'}")
        print(f"  duration: {duration_s:.1f}s · verdict: {verdict}")


if __name__ == "__main__":
    main()
