"""Notebook runner for entry 063 — does weight decay bound the free recurrence?

The SHD program's third stability probe (plan: ar063), and the last of the
no-code-change ones. exp062 found a stable recipe — Dale's law — but it costs a
few points of accuracy against the free signed net (exp062: 53.2% vs 56.8%). So
the question is no longer whether stability is *achievable* but whether it can be
had *without* the constraint's accuracy tax: can decoupled weight decay, strong
enough, bound the free net's runaway W_ee and remove the NaN while keeping the
signed recurrence's extra expressivity?

This sweeps the AdamW weight decay λ ∈ {0, 1e-3, 1e-2, 1e-1} on the free net
(--no-dales-law), everything else the exp060 recipe at Δt = 1.0 where it
diverges, single seed, full scale. It measures the plan's registered contrast:
max recurrent-weight norm and NaN-epoch rate vs decay strength. (A first pass at
1e-3 tamed the gradient explosion but did not bound W_ee or remove the NaN — so
the sweep reaches up to 1e-1.)

  KILL: if even the strongest decay leaves W_ee growing and NaN present, decay is
  not part of the recipe — it regularises but does not stabilise.

Compute: RunPod fan-out (one pod per λ), results collected over the S3 volume API.

Writing: writings/exp063.typ · figures + numbers.json: artifacts/data/exp063/
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

SLUG = "exp063"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
SNN_TOOL = REPO / "tools" / "snn" / "tool.py"

N_CLASSES = 20
CHANCE_PCT = 100.0 / N_CLASSES

# ── The swept variable and the fixed recipe ──────────────────────────
# Weight decay is the ONLY thing that varies. Everything else is exp060's free
# (signed, --no-dales-law) recipe at Δt = 1.0 ms, where the free net diverges.
WD_SWEEP = (0.0, 1e-3, 1e-2, 1e-1)   # AdamW λ; 1e-3 was the earlier partial pass
SEED = 42
DT_MS = 1.0
T_MS = 1000.0
SUBSET = 1000
EPOCHS = 30

PLUMBING_SUBSET = 64
PLUMBING_EPOCHS = 2
LOCAL_SUBSET = 128
LOCAL_EPOCHS = 15

# Recipe shared by every cell (minus --weight-decay, added per cell). Free net.
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


def _recipe_val(flag: str) -> str:
    """Read the value following a flag in RECIPE (so config numbers in the
    writeup are interpolated from the same source that drives training)."""
    return RECIPE[RECIPE.index(flag) + 1]


def _wd_label(wd: float) -> str:
    """Cell-name-safe label for a decay value: 0 → wd0, 1e-3 → wd1em3."""
    if wd == 0:
        return "wd0"
    return "wd" + f"{wd:g}".replace("-", "m").replace(".", "p").replace("+", "")


CELLS: list[dict] = [
    {"name": f"{_wd_label(wd)}__seed{SEED}", "wd": wd} for wd in WD_SWEEP
]

SCALE = {
    "dataset": "shd",
    "sweep": "weight decay λ ∈ {0, 1e-3, 1e-2, 1e-1}",
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
    return os.environ.get("PINGLAB_EXP063_PLUMBING") == "1"


def _local() -> bool:
    return os.environ.get("PINGLAB_EXP063_LOCAL") == "1"


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
    return [
        "train",
        *RECIPE,
        "--weight-decay", str(cell["wd"]),
        "--seed", str(SEED),
        "--max-samples", str(ms),
        "--epochs", str(ep),
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]


def _cell_by_name(name: str) -> dict | None:
    return next((c for c in CELLS if c["name"] == name), None)


# ── RunPod fan-out (one pod per λ) ───────────────────────────────────

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
            and cfg.get("weight_decay") == cell["wd"])


def _train_one_cell(cell: dict) -> None:
    ms, ep = cell_samples_epochs()
    print(f"[train-cell] {cell['name']} (wd={cell['wd']}, n={ms}, {ep} ep) "
          f"→ {cell_dir(cell['name'])}")
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
    buckets = [{"name": c["name"], "cells": [c["name"]]} for c in CELLS]
    runpod.dispatch(
        slug=SLUG, runner=SLUG,
        buckets=buckets,
        gpu=meta.gpu, live=meta.live, plumbing=meta.plumbing, collect=meta.collect,
        collect_subdir=POD_VOLUME_SUBDIR,
        local_collect_dir=str(LOCAL_TRAINING_ROOT),
        extra_env={"PINGLAB_TRAINING_ROOT": f"{runpod.VOLUME_MOUNT}/{POD_VOLUME_SUBDIR}"},
        plumbing_env={"PINGLAB_EXP063_PLUMBING": "1"},
    )


# ── Stability metrics ────────────────────────────────────────────────

def _wee_norm(weight_norms: dict) -> float:
    vals = [float(v) for k, v in weight_norms.items() if k.startswith("W_ee.")]
    return max(vals) if vals else float("nan")


def cell_stability(cell: dict) -> dict:
    p = cell_dir(cell["name"]) / "metrics.json"
    if not p.exists():
        return {"trained": False, "name": cell["name"], "wd": cell["wd"]}
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

def _wd_ticks(ax, wds: list[float]) -> None:
    # weight decay spans 0 and decades; plot on a symlog-ish categorical axis by
    # index, labelled with the actual values.
    ax.set_xticks(range(len(wds)))
    ax.set_xticklabels([("0" if w == 0 else f"{w:g}") for w in wds])
    ax.set_xlabel("weight decay λ")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_decay_sweep(stats: list[dict], stem: Path) -> None:
    """NaN-epoch rate, max ‖W_ee‖, and best accuracy vs weight decay."""
    theme.apply()
    trained = sorted([s for s in stats if s.get("trained")], key=lambda s: s["wd"])
    wds = [s["wd"] for s in trained]
    x = range(len(trained))
    # Column-width multi-panel row (H11/H12: ≈6.5 in wide, one aspect). dpi and
    # linewidth come from theme.apply(), not per-call overrides (H15).
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.8))

    # Separate subplots stay near-black — no decorative per-panel colour (H13).
    axes[0].plot(x, [100 * s["nan_epoch_rate"] for s in trained], "o-",
                 color=theme.INK_BLACK, ms=6)
    axes[0].set_ylabel("NaN-epoch rate (%)")
    axes[0].set_ylim(bottom=0)

    # Zero-based y-axis: ‖W_ee‖ barely moves across the sweep, so an auto-zoomed
    # axis would visually exaggerate a flat quantity (plot honesty, H13/H2).
    axes[1].plot(x, [s["max_wee_norm"] for s in trained], "o-",
                 color=theme.INK_BLACK, ms=6)
    axes[1].set_ylabel("max ‖W_ee‖")
    axes[1].set_ylim(bottom=0)

    axes[2].plot(x, [s["best_acc_pct"] for s in trained], "o-",
                 color=theme.INK_BLACK, ms=6)
    axes[2].axhline(CHANCE_PCT, color=theme.GREY_MID, lw=1.0, ls="--")
    axes[2].set_ylabel("best test accuracy (%)")

    for ax in axes:
        _wd_ticks(ax, wds)
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
                print(f"  λ={s['wd']:<7g} NaN-epochs={s['nan_epochs']}/{s['epochs']} "
                      f"max‖Wee‖={s['max_wee_norm']:.3g} best={s['best_acc_pct']}%")
            else:
                print(f"  λ={s['wd']:<7g} [not trained]")

        plot_decay_sweep(stats, figures / "decay_sweep")
        print(f"wrote {figures / 'decay_sweep.svg'}")

        trained = sorted([s for s in stats if s.get("trained")], key=lambda s: s["wd"])
        strongest = trained[-1] if trained else None
        weakest = trained[0] if trained else None
        # The plan's kill: if even the strongest decay leaves W_ee growing AND NaN
        # present, decay does not stabilise. "Stabilised" here = the strongest λ
        # reaches zero NaN epochs.
        nan_at_strongest = bool(strongest and strongest["nan_epochs"] > 0)
        any_nan_free = any(s["nan_epochs"] == 0 for s in trained)
        # First (smallest) λ that trains NaN-free, if any.
        first_stable = next((s["wd"] for s in trained if s["nan_epochs"] == 0), None)
        if not trained:
            verdict = "no cells trained"
        elif nan_at_strongest and not any_nan_free:
            verdict = ("kill — even the strongest decay leaves NaN present; weight decay "
                       "regularises but does not stabilise the free net")
        elif first_stable is not None:
            verdict = (f"stabilises — weight decay λ ≥ {first_stable:g} trains the free "
                       "net NaN-free, so the signed net can be stabilised without Dale's law")
        else:
            verdict = "mixed — see per-λ metrics"

        payload = {
            "wd_sweep": list(WD_SWEEP),
            "seed": SEED,
            "dt_ms": DT_MS,
            "max_samples": ms,
            "epochs": ep,
            "compute": _compute_label(),
            # Config inputs the writeup interpolates instead of hand-typing them:
            # every fixed recipe number the prose/table/captions cite, sourced
            # from the same RECIPE/SCALE that drives training.
            "config": {
                "n_hidden": int(_recipe_val("--n-hidden")),
                "batch_size": int(_recipe_val("--batch-size")),
                "lr": float(_recipe_val("--lr")),
                "theta_u": int(_recipe_val("--fr-reg-upper-theta")),
                "s_u": float(_recipe_val("--fr-reg-upper-strength")),
                "t_ms": T_MS,
                "dt_ms": DT_MS,
                "seeds": SCALE["seeds"],
            },
            "n_cells": len(CELLS),
            "n_trained": len(trained),
            "cells": stats,
            "nan_at_strongest_decay": nan_at_strongest,
            "first_stable_decay": first_stable,
            "weakest_decay": weakest["wd"] if weakest else None,
            "verdict": verdict,
            "chance_pct": CHANCE_PCT,
        }
        duration_s = time.monotonic() - t_start
        write_numbers(figures, run_id=run_id, duration_s=duration_s, payload=payload)
        print(f"wrote {figures / 'numbers.json'}")
        print(f"  duration: {duration_s:.1f}s · verdict: {verdict}")


if __name__ == "__main__":
    main()
