"""Notebook runner for entry 053 — is gradient damping needed without a loop?

A controlled 2×2 smoke test of the claim in [ar006] that the BPTT
gradient explosion in conductance-based SNNs is caused specifically by
the recurrent E→I→E loop — so a network with the loop disabled (COBA,
ei-strength 0) should train fine with the `--v-grad-dampen` stabiliser
turned OFF, while the same network with the loop on (PING, ei-strength 1)
should blow up.

Grid: {coba (ei 0), ping (ei 1)} × {damping off (γ=1), on (γ=1000)},
four otherwise-identical MNIST runs on nb025's canonical recipes. The
decisive measurement is the *pre-clip* gradient norm (train logs it
before the unit-norm clip), which is exactly the quantity ar006 predicts
diverges. We also record final accuracy and whether any NaN/Inf gradient
or loss appeared.

Prediction: ping@γ=1 explodes (huge/NaN grad norm, accuracy at chance);
coba@γ=1 stays bounded and trains; both γ=1000 controls train.

Only `--tier` (size) and `--modal-gpu` are accepted; every other knob is
hardcoded below per the runner contract.

Notebook entry: src/docs/src/pages/notebooks/nb053.mdx
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from _modal import parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402
from cli import theme  # noqa: E402

SLUG = "nb053"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
CLI = REPO / "src" / "cli" / "cli.py"

TIER_CONFIG = {
    "extra small": dict(max_samples=100, epochs=4),
    "small": dict(max_samples=500, epochs=12),
    "medium": dict(max_samples=2000, epochs=30),
    "large": dict(max_samples=5000, epochs=60),
    "extra large": dict(max_samples=10000, epochs=100),
}
DEFAULT_TIER = "small"
T_MS = 200.0
DT_TRAIN = 0.1
SEED = 42

# nb025's canonical recipes, verbatim except --v-grad-dampen, which is
# the variable under test. Both build as the COBANet ("ping") class;
# coba just zeroes the E→I weight (ei-strength 0), breaking the loop.
BASE_RECIPE = {
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

# The 2×2 grid: each cell is (model, gamma). gamma=1 → damping OFF.
DAMPENS = [1, 1000]
CELLS = [(m, g) for m in ("coba", "ping") for g in DAMPENS]

CHANCE_ACC = 10.0  # 10-class MNIST


def cell_label(model: str, gamma: int) -> str:
    return f"{model}__damp_{'off' if gamma == 1 else 'on'}"


def cell_dir(model: str, gamma: int) -> Path:
    return ARTIFACTS / cell_label(model, gamma)


def build_train_args(model: str, gamma: int, tier: str, out_dir: Path) -> list[str]:
    cfg = TIER_CONFIG[tier]
    args = [
        "train",
        "--model", "ping",
        "--dataset", "mnist",
        "--max-samples", str(cfg["max_samples"]),
        "--epochs", str(cfg["epochs"]),
        "--t-ms", str(T_MS),
        "--dt", str(DT_TRAIN),
        "--seed", str(SEED),
        "--v-grad-dampen", str(gamma),
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    for k, v in BASE_RECIPE[model].items():
        args += [k, v]
    return args


def run_cell(model: str, gamma: int, tier: str) -> dict:
    out_dir = cell_dir(model, gamma)
    argv = build_train_args(model, gamma, tier, out_dir)
    cmd = ["uv", "run", "python", str(CLI), *argv]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    dur = time.time() - t0
    crashed = proc.returncode != 0

    # Per-epoch trajectory from metrics.jsonl. The decisive signal is
    # skipped_steps: train.py skips any optimiser step whose pre-clip
    # gradient norm is non-finite (Inf/NaN), so skipped_steps > 0 IS the
    # explosion — and it zeroes the logged grad_norm for that epoch, which
    # is why an exploded run shows grad_norm 0 while frozen at chance.
    accs, grad_norms = [], []
    total_skipped = 0
    jsonl = out_dir / "metrics.jsonl"
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "acc" in rec:
                accs.append(float(rec["acc"]))
            if rec.get("grad_norm") is not None:
                grad_norms.append(float(rec["grad_norm"]))
            total_skipped += int(rec.get("skipped_steps", 0) or 0)

    finite_gn = [g for g in grad_norms if math.isfinite(g) and g > 0]
    nan_seen = any(not math.isfinite(g) for g in grad_norms)
    exploded = bool(total_skipped > 0 or nan_seen or crashed)
    # True pre-clip norm of an exploded step is Inf (it was skipped, so not
    # in finite_gn); report Inf so the figure/table don't read it as healthy.
    if finite_gn:
        peak_gn = max(finite_gn)
    elif exploded:
        peak_gn = float("inf")
    else:
        peak_gn = float("nan")
    final_acc = accs[-1] if accs else float("nan")
    best_acc = max(accs) if accs else float("nan")
    learned = math.isfinite(best_acc) and best_acc > CHANCE_ACC + 5.0

    return {
        "model": model,
        "gamma": gamma,
        "label": cell_label(model, gamma),
        "accs": accs,
        "grad_norms": grad_norms,
        "peak_grad_norm": peak_gn,
        "skipped_steps": total_skipped,
        "nan_seen": nan_seen,
        "crashed": crashed,
        "final_acc": final_acc,
        "best_acc": best_acc,
        "learned": bool(learned),
        "exploded": exploded,
        "duration_s": dur,
    }


# ── plotting ─────────────────────────────────────────────────────────────────
STYLE = {
    "coba": dict(color=theme.DEEP_RED),
    "ping": dict(color=theme.INK_BLACK),
}
DAMP_STYLE = {1: dict(ls="--"), 1000: dict(ls="-")}


def plot_results(rows: list[dict], out_path: Path, run_id: str) -> None:
    theme.apply()
    fig, (axa, axg) = plt.subplots(1, 2, figsize=(8.0, 4.5))

    # Left: test accuracy vs epoch
    for r in rows:
        if not r["accs"]:
            continue
        lbl = f"{r['model']} γ={'off' if r['gamma'] == 1 else r['gamma']}"
        axa.plot(
            range(1, len(r["accs"]) + 1), r["accs"],
            label=lbl, **STYLE[r["model"]], **DAMP_STYLE[r["gamma"]], lw=1.8,
        )
    axa.axhline(CHANCE_ACC, color=theme.GREY_MID, lw=0.8, ls=":")
    axa.text(0.02, CHANCE_ACC + 1, "chance", transform=axa.get_yaxis_transform(),
             color=theme.GREY_MID, fontsize=theme.SIZE_ANNOTATION)
    axa.set_xlabel("epoch", fontsize=theme.SIZE_LABEL)
    axa.set_ylabel("test accuracy (%)", fontsize=theme.SIZE_LABEL)
    axa.set_title("Training proceeds — or doesn't", fontsize=theme.SIZE_TITLE)
    axa.tick_params(labelsize=theme.SIZE_TICK)
    axa.legend(fontsize=theme.SIZE_LEGEND, loc="center right")
    for sp in ("top", "right"):
        axa.spines[sp].set_visible(False)

    # Right: peak pre-clip gradient norm, log10 on a linear axis (no log axis).
    # Exploded cells (Inf — all steps skipped) get a full-height sentinel bar.
    labels = [f"{r['model']}\nγ={'off' if r['gamma'] == 1 else r['gamma']}" for r in rows]
    xs = np.arange(len(rows))
    finite_logs = [math.log10(r["peak_grad_norm"]) for r in rows
                   if math.isfinite(r["peak_grad_norm"]) and r["peak_grad_norm"] > 0]
    ceil = (max(finite_logs) if finite_logs else 1.0) + 2.0
    vals, txts = [], []
    for r in rows:
        g = r["peak_grad_norm"]
        if r["exploded"]:
            vals.append(ceil)
            txts.append("∞ — all\nsteps skipped")
        elif math.isfinite(g) and g > 0:
            vals.append(max(math.log10(g), 0.0))   # floor at log10(GRAD_CLIP=1)
            txts.append(f"{g:.1f}")
        else:
            vals.append(0.0)
            txts.append("—")
    colors = [STYLE[r["model"]]["color"] for r in rows]
    axg.bar(xs, vals, color=colors, edgecolor=theme.INK_BLACK,
            hatch=["" if r["gamma"] == 1000 else "///" for r in rows])
    for x, v, t in zip(xs, vals, txts):
        axg.text(x, v + 0.08, t, ha="center", va="bottom",
                 fontsize=theme.SIZE_ANNOTATION, color=theme.INK_BLACK)
    axg.set_xticks(xs)
    axg.set_xticklabels(labels, fontsize=theme.SIZE_TICK)
    axg.set_ylim(0, ceil + 1.2)
    axg.set_ylabel("log₁₀ peak pre-clip gradient norm", fontsize=theme.SIZE_LABEL)
    axg.set_title("The gradient explosion", fontsize=theme.SIZE_TITLE)
    axg.tick_params(labelsize=theme.SIZE_TICK)
    for sp in ("top", "right"):
        axg.spines[sp].set_visible(False)

    fig.text(0.995, 0.005, run_id, ha="right", va="bottom",
             fontsize=theme.SIZE_CAPTION, color=theme.GREY_MID, family="monospace")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    modal_gpu = parse_modal_gpu(sys.argv)
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    run_id = next_run_id(SLUG)
    t0 = time.time()

    print(f"notebook_run_id = {run_id}  tier={tier}"
          + ("  (modal-gpu ignored: nb053 runs locally)" if modal_gpu else ""))
    cfg = TIER_CONFIG[tier]
    print(f"  4 runs × (max_samples={cfg['max_samples']}, epochs={cfg['epochs']}), "
          f"T={T_MS} ms — ETA a few minutes on CPU")

    if "--no-wipe-dir" not in sys.argv:
        for d in (ARTIFACTS, FIGURES):
            if d.exists():
                shutil.rmtree(d)
            print(f"[wipe] {d}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    rows = []
    for model, gamma in CELLS:
        print(f"  → training {cell_label(model, gamma)} …", flush=True)
        r = run_cell(model, gamma, tier)
        verdict = ("EXPLODED" if r["exploded"] else "ok") + (
            " / learned" if r["learned"] else " / no-learn")
        print(f"    peak|grad|={r['peak_grad_norm']:.3g}  skipped={r['skipped_steps']}  "
              f"best_acc={r['best_acc']:.1f}%  [{verdict}]  ({r['duration_s']:.0f}s)")
        rows.append(r)

    plot_results(rows, FIGURES / "loop_vs_damping.png", run_id)
    print(f"  wrote {FIGURES / 'loop_vs_damping.png'}")

    def _jsafe(v):
        # JSON has no Infinity/NaN; emit null so the mdx import parses.
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v

    summary = {
        "slug": SLUG,
        "notebook_run_id": run_id,
        "tier": tier,
        "config": {"max_samples": cfg["max_samples"], "epochs": cfg["epochs"],
                   "t_ms": T_MS, "dt": DT_TRAIN, "seed": SEED},
        "cells": [
            {k: _jsafe(r[k]) for k in (
                "label", "model", "gamma", "peak_grad_norm", "skipped_steps",
                "nan_seen", "crashed", "final_acc", "best_acc", "learned",
                "exploded")}
            for r in rows
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    persist_run_id(SLUG, run_id)
    print(f"  wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
