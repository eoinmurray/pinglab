"""Notebook runner for entry 061 — does finer Δt stabilise the integration?

The first stability experiment in the SHD program (plan: ar063). exp060 showed
the free signed-recurrent ceiling trains to 61% but diverges intermittently:
scattered epochs return NaN, the pre-clip gradient norm spikes into the millions,
and W_ee grows without bound — and a NaN appeared at epoch 2 with tiny weights,
so it is NOT weight-growth runaway. The plan's hypothesis is exp-Euler stiffness
at the coarse Δt = 1.0 ms: halving then quartering Δt should drop the NaN-epoch
rate toward zero.

So this sweeps Δt ∈ {1.0, 0.5, 0.25} ms on ONE seed (fast; we expect breakage
and want a quick first read, not error bars), holding the rest of exp060's Rung A
recipe fixed, and measures the plan's registered stability metrics against Δt:
NaN-epoch rate, max pre-clip gradient norm, and max recurrent-weight (W_ee) norm.

  KILL: if NaN persists at Δt = 0.25 (finest), coarse integration is not the
  cause — drop Δt from the recipe and look upstream (state clamping).

Single seed keeps the sweep to three training runs. Δt = 1.0 reproduces exp060's
known failure (same subset, same recipe) so the finer-Δt arms are read against a
grounded baseline, not an abstract one.

Compute: RunPod fan-out (one pod per Δt) via helpers/runpod.py — local torch here
is CPU-only, and the plan's night-shift policy (ar065) authorises RunPod for this
program. Dry-run by default; --live to spend. Δt = 0.25 unrolls 4000 BPTT steps
(T = 1000 ms), 4x the Δt = 1.0 cost, so each arm gets its own pod.

Writing: writings/exp061.typ · figures + numbers.json: artifacts/data/exp061/
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

SLUG = "exp061"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
SNN_TOOL = REPO / "tools" / "snn" / "tool.py"

N_CLASSES = 20
CHANCE_PCT = 100.0 / N_CLASSES  # 5% on 20 classes

# ── The swept variable and the fixed recipe ──────────────────────────
# Δt is the ONLY thing that varies across cells. Everything else is exp060's
# Rung A recipe verbatim (free signed recurrence, no Dale's law, all four blocks
# trainable, the load-bearing firing-rate regulariser, weight decay 1e-3), so any
# change in the stability metrics is attributable to Δt alone.
DT_SWEEP_MS = (1.0, 0.5, 0.25)   # coarse → fine; 0.25 unrolls 4000 BPTT steps
SEED = 42                         # single seed — fast first read (see module docstring)
T_MS = 1000.0
SUBSET = 1000                     # matches exp060 so Δt = 1.0 reproduces its failure
EPOCHS = 30                       # matches exp060; enough epochs for NaNs to surface

# Plumbing scale — a cheap wiring-check that validates the whole pod path (image
# checkout, SHD download on the pod, training, volume write, self-terminate,
# collect) in minutes for cents, before the real spend. PINGLAB_EXP061_PLUMBING=1.
PLUMBING_SUBSET = 64
PLUMBING_EPOCHS = 2

# Local-CPU scale (PINGLAB_EXP061_LOCAL=1) — a reduced first-read scale that runs
# the whole sweep on CPU in ~40 min when the RunPod fan-out is unavailable (this
# lab's cloud sandbox blocks outbound SSH, so the rsync-over-SSH collect step
# cannot pull results back — see writings/exp061.typ §Compute). The stiffness the
# plan targets is a per-forward-pass property (exp060 diverged at epoch 2 with
# tiny weights), so it is expected to show at this scale; the RunPod path above
# runs the full exp060-matching scale from an unrestricted network.
LOCAL_SUBSET = 128
LOCAL_EPOCHS = 15

# Rung A recipe (from exp060) as flag/value pairs, minus --dt / --seed / --out-dir
# which build_train_args supplies per cell.
RECIPE: list[str] = [
    "--dataset", "shd",
    "--t-ms", str(int(T_MS)),
    "--n-hidden", "256",
    "--model", "ping",
    "--batch-size", "32",
    "--lr", "0.001",
    "--weight-decay", "0.001",
    "--no-dales-law",
    "--w-ee", "0.3", "0.1",
    "--trainable-w-ee", "--trainable-w-ei", "--trainable-w-ie", "--trainable-w-ii",
    "--fr-reg-upper-theta", "100",
    "--fr-reg-upper-strength", "0.06",
]

# ── Cell registry: one training run per Δt ───────────────────────────
CELLS: list[dict] = [
    {"name": f"dt{_dt:g}".replace(".", "p") + f"__seed{SEED}", "dt": _dt, "seed": SEED}
    for _dt in DT_SWEEP_MS
]


def _recipe_val(flag: str) -> str:
    """The value that follows `flag` in the RECIPE list — a single source of truth
    so the writeup can interpolate config inputs (n-hidden, lr, …) straight from
    the recipe rather than re-typing them in prose."""
    return RECIPE[RECIPE.index(flag) + 1]

SCALE = {
    "dataset": "shd",
    "sweep": "Δt ∈ {1.0, 0.5, 0.25} ms",
    "max_samples": SUBSET,
    "t_ms": T_MS,
    "epochs": EPOCHS,
    "n_hidden": 256,
    "seeds": 1,
    "seed": SEED,
    "recipe": "Rung A (free signed-recurrent ceiling, no Dale's law, fr-reg on)",
}


# ── Local ↔ volume training root (isolated from the exp022 bank) ──────
# On a pod, dispatch sets PINGLAB_TRAINING_ROOT to /shared/training/exp061 so the
# fan-out writes durable cells to the network volume under this experiment's own
# subdir (never mixing with exp022's shared bank). Local runs / collected cells
# land under temp/experiments/exp061_cells. cell_dir reads through this.
POD_VOLUME_SUBDIR = f"training/{SLUG}"                    # → /shared/training/exp061
LOCAL_TRAINING_ROOT = REPO / "temp" / "experiments" / f"{SLUG}_cells"


def training_root() -> Path:
    return Path(os.environ.get("PINGLAB_TRAINING_ROOT", str(LOCAL_TRAINING_ROOT)))


def cell_dir(name: str) -> Path:
    return training_root() / name


def _plumbing() -> bool:
    return os.environ.get("PINGLAB_EXP061_PLUMBING") == "1"


def _local() -> bool:
    return os.environ.get("PINGLAB_EXP061_LOCAL") == "1"


def cell_samples_epochs() -> tuple[int, int]:
    """(max_samples, epochs): the tiny plumbing scale under
    PINGLAB_EXP061_PLUMBING, the reduced local-CPU scale under
    PINGLAB_EXP061_LOCAL, else the full RunPod scale."""
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
    """CLI `train` args for one Δt cell — the fixed recipe plus this cell's Δt,
    seed, scale and output dir."""
    ms, ep = cell_samples_epochs()
    return [
        "train",
        *RECIPE,
        "--dt", str(cell["dt"]),
        "--seed", str(cell["seed"]),
        "--max-samples", str(ms),
        "--epochs", str(ep),
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]


def _cell_by_name(name: str) -> dict | None:
    return next((c for c in CELLS if c["name"] == name), None)


# ── RunPod fan-out (one pod per Δt) ──────────────────────────────────

def runpod_is_done(cell: dict) -> bool:
    """A cell is done iff its metrics.json exists AND was trained at the scale
    (max_samples, epochs, dt) THIS run expects — a stale-scale cell reads as
    pending and is retrained (mirrors exp022's scale-aware marker)."""
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
            and cfg.get("dt") == cell["dt"])


def _train_one_cell(cell: dict) -> None:
    """Train ONE Δt cell by invoking the SNN CLI directly (like exp022).

    Shells out to the active interpreter (sys.executable) rather than a nested
    `uv run`: on a pod the image's baked venv is already active, so this avoids
    the redundant `uv sync` a fresh `uv run` would trigger (the image sets no
    PINGLAB_NO_SYNC for nested calls). Local and pod paths are then identical."""
    ms, ep = cell_samples_epochs()
    print(f"[train-cell] {cell['name']} (dt={cell['dt']}, n={ms}, {ep} ep) "
          f"→ {cell_dir(cell['name'])}")
    subprocess.run(
        [sys.executable, str(SNN_TOOL), *build_train_args(cell, cell_dir(cell["name"]))],
        cwd=REPO, check=True,
    )


def pod_run() -> None:
    """Pod-side entrypoint (image start script runs `exp061.py --pod-run`).

    Trains every Δt cell named in the CELLS env var to the shared volume,
    skipping any already done, then self-terminates — the loop / skip-done /
    always-self-terminate contract lives in runpod.pod_run_loop."""
    print(f"[pod-run] plumbing={_plumbing()} root={training_root()}")

    def is_done(name: str) -> bool:
        c = _cell_by_name(name)
        return c is not None and runpod_is_done(c)

    def run_job(name: str) -> None:
        c = _cell_by_name(name)
        if c is not None:   # pod_run_loop only passes valid job ids; guard anyway
            _train_one_cell(c)

    runpod.pod_run_loop(
        job_ids=[c["name"] for c in CELLS], is_done=is_done, run_job=run_job,
    )


def run_via_runpod(argv: list[str]) -> None:
    """`--runpod` dispatch: one pod per Δt cell, fire-and-forget to the shared
    volume. Dry-run by default; --live to spend. Retrieve with --runpod --collect,
    then render with --skip-training."""
    meta = parse_meta(argv, allow_dispatch=True)
    buckets = [{"name": c["name"], "cells": [c["name"]]} for c in CELLS]
    runpod.dispatch(
        slug=SLUG, runner=SLUG,
        buckets=buckets,
        gpu=meta.gpu, live=meta.live, plumbing=meta.plumbing, collect=meta.collect,
        collect_subdir=POD_VOLUME_SUBDIR,
        local_collect_dir=str(LOCAL_TRAINING_ROOT),
        extra_env={"PINGLAB_TRAINING_ROOT": f"{runpod.VOLUME_MOUNT}/{POD_VOLUME_SUBDIR}"},
        plumbing_env={"PINGLAB_EXP061_PLUMBING": "1"},
    )


# ── Stability metrics: read each cell's metrics.json ─────────────────

def _wee_norm(weight_norms: dict) -> float:
    """The E→E recurrent-weight Frobenius norm from a per-epoch weight_norms dict.
    The block is an nn.ParameterDict keyed by layer index, so named_parameters
    yields `W_ee.<k>` (e.g. W_ee.1). Take the largest matching block."""
    vals = [float(v) for k, v in weight_norms.items() if k.startswith("W_ee.")]
    return max(vals) if vals else float("nan")


def cell_stability(cell: dict) -> dict:
    """Reduce one cell's metrics.json to the plan's stability metrics.

    NaN-epoch: an epoch is counted as NaN if its train or test loss is non-finite
    (the eval loss has no NaN guard, so a diverged forward pass surfaces there) OR
    it saw a NaN-forward batch. max grad norm is the peak pre-clip global norm
    across epochs; max W_ee norm the peak E→E recurrent-weight norm."""
    p = cell_dir(cell["name"]) / "metrics.json"
    if not p.exists():
        return {"trained": False, "dt": cell["dt"], "name": cell["name"]}
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
        "dt": cell["dt"],
        "epochs": len(eps),
        "nan_epochs": nan_epochs,
        "nan_epoch_rate": (nan_epochs / len(eps)) if eps else float("nan"),
        "nan_forward_batches": sum(e.get("nan_forward_batches", 0) or 0 for e in eps),
        "skipped_steps": sum(e.get("skipped_steps", 0) or 0 for e in eps),
        "max_grad_norm": max(grad_maxes) if grad_maxes else float("nan"),
        "max_wee_norm": max(wee_maxes) if wee_maxes else float("nan"),
        "best_acc_pct": m.get("best_acc"),
        "best_epoch": m.get("best_epoch"),
    }


# ── Figures ──────────────────────────────────────────────────────────

def _dt_ticks(ax, dts: list[float]) -> None:
    ax.set_xticks(dts)
    ax.set_xticklabels([f"{d:g}" for d in dts])
    ax.set_xlabel("Δt (ms)")
    ax.invert_xaxis()   # coarse → fine, left → right
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_stability(stats: list[dict], stem: Path) -> None:
    """Three panels vs Δt: NaN-epoch rate, max pre-clip grad norm (log),
    max W_ee norm — the plan's registered stability metrics."""
    theme.apply()
    trained = [s for s in stats if s.get("trained")]
    dts = [s["dt"] for s in trained]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))

    # H13: separate subplots stay near-black; no decorative per-panel accent.
    axes[0].plot(dts, [100 * s["nan_epoch_rate"] for s in trained],
                 "o-", color=theme.INK_BLACK, lw=2.0, ms=7)
    axes[0].set_ylabel("NaN-epoch rate (%)")
    axes[0].set_ylim(bottom=-2)

    axes[1].plot(dts, [s["max_grad_norm"] for s in trained],
                 "o-", color=theme.INK_BLACK, lw=2.0, ms=7)
    axes[1].set_ylabel("max pre-clip grad norm")
    axes[1].set_yscale("log")

    axes[2].plot(dts, [s["max_wee_norm"] for s in trained],
                 "o-", color=theme.INK_BLACK, lw=2.0, ms=7)
    axes[2].set_ylabel("max ‖W_ee‖")

    for ax in axes:
        _dt_ticks(ax, dts)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def plot_grad_trace(stem: Path) -> None:
    """Per-epoch max pre-clip grad norm, one line per Δt (log y) — shows WHEN the
    gradient spikes fire, not just their peak."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(6.5, 3.66))
    # H13: near-black by default, at most one accent (red for the finest Δt), and a
    # distinct line style per Δt so the three traces survive a grayscale print and
    # read for colour-blind viewers — not colour alone.
    styles = [
        (theme.INK_BLACK, "-"),
        (theme.INK_BLACK, "--"),
        (theme.DEEP_RED, ":"),
    ]
    drew = False
    for cell, (colour, ls) in zip(CELLS, styles):
        p = cell_dir(cell["name"]) / "metrics.json"
        if not p.exists():
            continue
        eps = json.loads(p.read_text()).get("epochs") or []
        x = [e["ep"] for e in eps]
        y = [e.get("grad_norm_max", e.get("grad_norm")) for e in eps]
        ax.plot(x, y, ls=ls, color=colour, lw=1.8, label=f"Δt = {cell['dt']:g} ms")
        drew = True
    if drew:
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("max pre-clip grad norm")
        ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    meta = parse_meta(sys.argv, allow_dispatch=True)

    # RunPod backend + kill switch, before the local path.
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
        # Local training (CPU here; the real sweep runs on RunPod). Skipped when
        # rendering from already-collected cells.
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
                print(f"  Δt={s['dt']:<5g} NaN-epochs={s['nan_epochs']}/{s['epochs']} "
                      f"maxGrad={s['max_grad_norm']:.3g} max‖Wee‖={s['max_wee_norm']:.3g} "
                      f"best={s['best_acc_pct']}%")
            else:
                print(f"  Δt={s['dt']:<5g} [not trained]")

        plot_stability(stats, figures / "stability_vs_dt")
        print(f"wrote {figures / 'stability_vs_dt.svg'}")
        plot_grad_trace(figures / "grad_trace")
        print(f"wrote {figures / 'grad_trace.svg'}")

        trained = [s for s in stats if s.get("trained")]
        by_dt = {s["dt"]: s for s in trained}
        coarse = by_dt.get(max(by_dt)) if by_dt else None   # Δt = 1.0
        finest = by_dt.get(min(by_dt)) if by_dt else None    # Δt = 0.25

        # The hypothesis is specifically about NaN divergence: does finer Δt drop
        # the NaN-epoch rate toward zero? Reasoning honestly about it needs THREE
        # facts, not just "no NaN at the finest Δt" — which is vacuously true when
        # nothing diverged anywhere:
        #   • nan_reproduced — did the coarse Δt (the exp060 setting) actually
        #     diverge here? If not, the mechanism was never exercised → inconclusive.
        #   • nan_at_finest — the kill criterion: NaN persisting at Δt = 0.25.
        #   • grad_falls_with_finer_dt — the secondary signal: does the peak
        #     pre-clip gradient norm fall monotonically as Δt shrinks? Finer Δt
        #     quarters the step but quadruples the BPTT unroll, so it can trade
        #     integration stiffness for a longer, more explosive gradient chain.
        nan_reproduced = bool(coarse and coarse["nan_epochs"] > 0)
        nan_at_finest = bool(finest and finest["nan_epochs"] > 0)
        grads = [by_dt[d]["max_grad_norm"] for d in sorted(by_dt, reverse=True)]  # coarse→fine
        grad_falls_with_finer_dt = (
            all(a >= b for a, b in zip(grads, grads[1:])) if len(grads) > 1 else None
        )
        if not trained:
            verdict = "no cells trained"
        elif not nan_reproduced:
            verdict = ("inconclusive — the coarse Δt did not diverge at this scale, so "
                       "the NaN mechanism was not exercised; needs the full scale where "
                       "exp060 diverged")
        elif nan_at_finest:
            verdict = ("kill — NaN persists at the finest Δt, so coarse integration is "
                       "not the cause; look upstream (state clamping)")
        else:
            verdict = "supported — NaN present at the coarse Δt and absent at the finest Δt"

        payload = {
            "dt_sweep": list(DT_SWEEP_MS),
            "seed": SEED,
            "max_samples": ms,
            "epochs": ep,
            "compute": _compute_label(),
            # Config inputs held fixed across the Δt sweep (exp060's Rung A recipe),
            # surfaced here so the writeup interpolates every one from data instead
            # of hand-typing them. t_ms drives T_steps = T/Δt in the writeup; the
            # dt values themselves come from each cell's `dt`.
            "config": {
                "n_hidden": int(_recipe_val("--n-hidden")),
                "n_seeds": len({c["seed"] for c in CELLS}),
                "t_ms": int(T_MS),
                "lr": float(_recipe_val("--lr")),
                "weight_decay": float(_recipe_val("--weight-decay")),
                "batch_size": int(_recipe_val("--batch-size")),
                "fr_reg_upper_theta": int(_recipe_val("--fr-reg-upper-theta")),
                "fr_reg_upper_strength": float(_recipe_val("--fr-reg-upper-strength")),
                "local_subset": LOCAL_SUBSET,
            },
            "n_cells": len(CELLS),
            "n_trained": len(trained),
            "cells": stats,
            "finest_dt": finest["dt"] if finest else None,
            "nan_reproduced_at_coarse_dt": nan_reproduced,
            "nan_at_finest_dt": nan_at_finest,
            "grad_falls_with_finer_dt": grad_falls_with_finer_dt,
            "verdict": verdict,
            "chance_pct": CHANCE_PCT,
        }
        duration_s = time.monotonic() - t_start
        write_numbers(figures, run_id=run_id, duration_s=duration_s, payload=payload)
        print(f"wrote {figures / 'numbers.json'}")
        print(f"  duration: {duration_s:.1f}s · verdict: {verdict}")


if __name__ == "__main__":
    main()
