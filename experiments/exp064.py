"""Notebook runner for entry 064 — does the forward-pass state clamp stabilise
the free net (and keep its accuracy)?

The lead the whole stability queue pointed to. exp061–063 showed the free signed
net's NaN divergence is unreachable by any soft knob: Δt makes it worse, weight
decay only regularises, and only Dale's law stabilises — at an accuracy cost
(exp062: 53.2% vs the free net's 56.8%, and exp063's decay reached 63.4% without
stability). The divergence's mechanism is the exp-Euler v_inf = (…)/g_tot blowing
up when signed weights drive g_tot ≤ 0, so the plan's reserved fix is a
forward-pass *state clamp*: floor conductances at 0 (physical) each timestep so
g_tot ≥ g_L > 0, while the weights stay signed and expressive.

This tests it, three cells at the coarse Δt = 1.0 ms where the free net diverges,
single seed, full scale:
  • free (baseline, --no-dales-law) — diverges;
  • free + state clamp (--state-clamp) — does the clamp remove the NaN?
  • free + state clamp + strong decay (λ = 0.1) — decay lifted the free net to
    63.4% but could not stabilise; with the clamp doing the stabilising, does
    clamp+decay beat both the free net's accuracy and Dale's law's stability?

  SUCCESS: the clamp trains the free net NaN-free with bounded dynamics, at
  accuracy at least matching Dale's law — a stable recipe that keeps the signed
  net's expressivity. The clamp+decay cell is the bid to top the program.

Compute: RunPod fan-out, one pod per cell, collected over the S3 volume API.

Writing: writings/exp064.typ · figures + numbers.json: artifacts/data/exp064/
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

SLUG = "exp064"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
SNN_TOOL = REPO / "tools" / "snn" / "tool.py"

N_CLASSES = 20
CHANCE_PCT = 100.0 / N_CLASSES

SEED = 42
DT_MS = 1.0
T_MS = 1000.0
SUBSET = 1000
EPOCHS = 30

PLUMBING_SUBSET = 64
PLUMBING_EPOCHS = 2
LOCAL_SUBSET = 128
LOCAL_EPOCHS = 15

# Recipe shared by every cell — the free (signed) net at Δt = 1.0 where it
# diverges. The state clamp and weight decay vary per cell.
RECIPE: list[str] = [
    "--dataset", "shd",
    "--t-ms", str(int(T_MS)),
    "--dt", str(DT_MS),
    "--n-hidden", "256",
    "--model", "ping",
    "--batch-size", "32",
    "--lr", "0.001",
    "--no-dales-law",
    "--w-ee", "0.3", "0.1",
    "--trainable-w-ee", "--trainable-w-ei", "--trainable-w-ie", "--trainable-w-ii",
    "--fr-reg-upper-theta", "100",
    "--fr-reg-upper-strength", "0.06",
]

# Three cells: state_clamp and weight decay drive the contrast. The baseline uses
# the exp061/062 free recipe (wd 1e-3); the last adds exp063's strongest decay.
CELLS: list[dict] = [
    {"name": f"free__seed{SEED}", "clamp": False, "wd": 1e-3,
     "label": "free (baseline)"},
    {"name": f"clamp__seed{SEED}", "clamp": True, "wd": 1e-3,
     "label": "free + state clamp"},
    {"name": f"clampwd__seed{SEED}", "clamp": True, "wd": 1e-1,
     "label": "clamp + decay 0.1"},
]

SCALE = {
    "dataset": "shd",
    "contrast": "free vs free+clamp vs free+clamp+decay",
    "max_samples": SUBSET,
    "t_ms": T_MS,
    "dt_ms": DT_MS,
    "epochs": EPOCHS,
    "n_hidden": 256,
    "seeds": 1,
    "seed": SEED,
    "dales_law": False,
}

POD_VOLUME_SUBDIR = f"training/{SLUG}"
LOCAL_TRAINING_ROOT = REPO / "temp" / "experiments" / f"{SLUG}_cells"


def training_root() -> Path:
    return Path(os.environ.get("PINGLAB_TRAINING_ROOT", str(LOCAL_TRAINING_ROOT)))


def cell_dir(name: str) -> Path:
    return training_root() / name


def _plumbing() -> bool:
    return os.environ.get("PINGLAB_EXP064_PLUMBING") == "1"


def _local() -> bool:
    return os.environ.get("PINGLAB_EXP064_LOCAL") == "1"


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
    ms, ep = cell_samples_epochs()
    args = [
        "train",
        *RECIPE,
        "--weight-decay", str(cell["wd"]),
        "--seed", str(SEED),
        "--max-samples", str(ms),
        "--epochs", str(ep),
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    if cell["clamp"]:
        args.append("--state-clamp")
    return args


def _cell_by_name(name: str) -> dict | None:
    return next((c for c in CELLS if c["name"] == name), None)


# ── RunPod fan-out (one pod per cell) ────────────────────────────────

def runpod_is_done(cell: dict) -> bool:
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
            and cfg.get("state_clamp") == cell["clamp"]
            and cfg.get("weight_decay") == cell["wd"])


def _train_one_cell(cell: dict) -> None:
    ms, ep = cell_samples_epochs()
    print(f"[train-cell] {cell['name']} (clamp={cell['clamp']}, wd={cell['wd']}, "
          f"n={ms}, {ep} ep) → {cell_dir(cell['name'])}")
    subprocess.run(
        [sys.executable, str(SNN_TOOL), *build_train_args(cell, cell_dir(cell["name"]))],
        cwd=REPO, check=True,
    )


def pod_run() -> None:
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
    meta = parse_meta(argv, allow_dispatch=True)
    cells = CELLS
    if meta.only_cells:                       # fire a subset (e.g. a cell that missed on no-stock)
        wanted = set(meta.only_cells)
        cells = [c for c in CELLS if c["name"] in wanted]
        missing = wanted - {c["name"] for c in cells}
        if missing:
            raise SystemExit(f"unknown cell(s): {sorted(missing)}")
    buckets = [{"name": c["name"], "cells": [c["name"]]} for c in cells]
    runpod.dispatch(
        slug=SLUG, runner=SLUG,
        buckets=buckets,
        gpu=meta.gpu, live=meta.live, plumbing=meta.plumbing, collect=meta.collect,
        collect_subdir=POD_VOLUME_SUBDIR,
        local_collect_dir=str(LOCAL_TRAINING_ROOT),
        extra_env={"PINGLAB_TRAINING_ROOT": f"{runpod.VOLUME_MOUNT}/{POD_VOLUME_SUBDIR}"},
        plumbing_env={"PINGLAB_EXP064_PLUMBING": "1"},
    )


# ── Stability metrics ────────────────────────────────────────────────

def _wee_norm(weight_norms: dict) -> float:
    vals = [float(v) for k, v in weight_norms.items() if k.startswith("W_ee.")]
    return max(vals) if vals else float("nan")


def cell_stability(cell: dict) -> dict:
    p = cell_dir(cell["name"]) / "metrics.json"
    if not p.exists():
        return {"trained": False, "name": cell["name"], "label": cell["label"],
                "clamp": cell["clamp"], "wd": cell["wd"]}
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
        "clamp": cell["clamp"],
        "wd": cell["wd"],
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

def plot_bars(stats: list[dict], stem: Path) -> None:
    """NaN-epoch rate, max ‖W_ee‖, and best accuracy across the three cells."""
    theme.apply()
    trained = [s for s in stats if s.get("trained")]
    labels = [s["label"] for s in trained]
    x = range(len(trained))
    colours = [theme.DEEP_RED if not s["clamp"] else theme.INK_BLACK for s in trained]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), dpi=150)

    axes[0].bar(x, [100 * s["nan_epoch_rate"] for s in trained], color=colours, width=0.6)
    axes[0].set_ylabel("NaN-epoch rate (%)")

    axes[1].bar(x, [s["max_wee_norm"] for s in trained], color=colours, width=0.6)
    axes[1].set_ylabel("max ‖W_ee‖")

    axes[2].bar(x, [s["best_acc_pct"] for s in trained], color=colours, width=0.6)
    axes[2].axhline(CHANCE_PCT, color=theme.GREY_MID, lw=1.0, ls="--")
    axes[2].set_ylabel("best test accuracy (%)")

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def plot_loss_traces(stem: Path) -> None:
    """Test loss per epoch across the three cells — NaN epochs show as gaps."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    colours = [theme.DEEP_RED, theme.INK_BLACK, theme.ELECTRIC_CYAN]
    for cell, colour in zip(CELLS, colours):
        p = cell_dir(cell["name"]) / "metrics.json"
        if not p.exists():
            continue
        eps = json.loads(p.read_text()).get("epochs") or []
        x = [e["ep"] for e in eps]
        y = [e.get("test_loss") if isinstance(e.get("test_loss"), (int, float))
             and math.isfinite(e.get("test_loss")) else float("nan") for e in eps]
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
                print(f"  {s['label']:<20s} NaN={s['nan_epochs']}/{s['epochs']} "
                      f"max‖Wee‖={s['max_wee_norm']:.3g} best={s['best_acc_pct']}%")
            else:
                print(f"  {s['label']:<20s} [not trained]")

        plot_bars(stats, figures / "clamp_bars")
        print(f"wrote {figures / 'clamp_bars.svg'}")
        plot_loss_traces(figures / "loss_traces")
        print(f"wrote {figures / 'loss_traces.svg'}")

        by_name = {s["name"]: s for s in stats if s.get("trained")}
        free = by_name.get(f"free__seed{SEED}")
        clamp = by_name.get(f"clamp__seed{SEED}")
        clampwd = by_name.get(f"clampwd__seed{SEED}")
        free_diverges = bool(free and free["nan_epochs"] > 0)
        clamp_stable = bool(clamp and clamp["nan_epochs"] == 0)
        clampwd_stable = bool(clampwd and clampwd["nan_epochs"] == 0)
        best_stable_acc = max(
            [s["best_acc_pct"] for s in (clamp, clampwd)
             if s and s["nan_epochs"] == 0 and s["best_acc_pct"] is not None],
            default=None,
        )
        if not (free and clamp):
            verdict = "incomplete — need the free and clamp cells to compare"
        elif not free_diverges:
            verdict = ("inconclusive — the free baseline did not diverge at this scale, "
                       "so the clamp has nothing to fix")
        elif not clamp_stable:
            verdict = ("kill — the state clamp did not remove the divergence; the "
                       "g_tot ≤ 0 mechanism is not the (whole) cause")
        else:
            verdict = ("stabilises — the state clamp trains the free net NaN-free while "
                       "keeping the weights signed"
                       + (f"; best stable accuracy {best_stable_acc:.1f}%"
                          if best_stable_acc is not None else ""))

        payload = {
            "seed": SEED,
            "dt_ms": DT_MS,
            "max_samples": ms,
            "epochs": ep,
            "compute": _compute_label(),
            "n_cells": len(CELLS),
            "n_trained": len(by_name),
            "cells": stats,
            "free_diverges": free_diverges,
            "clamp_stable": clamp_stable,
            "clampwd_stable": clampwd_stable,
            "best_stable_acc_pct": best_stable_acc,
            "verdict": verdict,
            "chance_pct": CHANCE_PCT,
        }
        duration_s = time.monotonic() - t_start
        write_numbers(figures, run_id=run_id, duration_s=duration_s, payload=payload)
        print(f"wrote {figures / 'numbers.json'}")
        print(f"  duration: {duration_s:.1f}s · verdict: {verdict}")


if __name__ == "__main__":
    main()
