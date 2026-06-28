"""Notebook runner for entry 062 — SHD baseline for the autoresearch loop.

Establishes the frozen baseline that *autoresearch* (Karpathy-style overnight
agent loop) iterates against. The baseline is a small, biologically-structured
PING/COBA net trained on SHD with the standard leaky-integrator readout:

  --model ping --dataset shd --readout li
  --n-hidden 128, single layer
  --ei-strength 0.5, --ei-ratio 2.0  (Dale-clamped)
  --dt 2 ms, --t-ms 1000  →  500 timesteps per trial
  --lr 0.001, --batch-size 256
  --v-grad-dampen 80, --w-in 0.3 0.06
  seed 42

This file is the *recipe*, not the playground. Per house rules a notebook
runner is reproducible and is not mutated mid-experiment — autoresearch
experiments happen on their own branches (autoresearch/<tag>) and any
finding worth keeping gets promoted to its own notebook entry
(nb063, nb064, …) carrying *collection: autoresearch* in the frontmatter.

A single training run is dispatched to src/cli/cli.py; metrics.json
+ weights.pth land in src/artifacts/notebooks/nb062/baseline/. A 16:9
two-panel training-curve figure + numbers.json land in
src/docs/public/figures/notebooks/nb062/.

Notebook entry: src/docs/content/notebooks/nb062.mdx
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

from helpers.fmt import format_duration  # noqa: E402
from helpers.modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402
from cli import theme  # noqa: E402

SLUG = "nb062"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
OSCILLOSCOPE = REPO / "src" / "cli/cli.py"

# Tier sizing — max-samples × epochs. ETAs are rough single-run MPS estimates
# at the dt=2 ms / t=1000 ms grid (500 timesteps per trial, n_hidden=128):
TIER_CONFIG = {
    "extra small": dict(max_samples=300,  epochs=2),   # ≈ 5 s   (smoke)
    "small":       dict(max_samples=1000, epochs=8),   # ≈ 20 s
    "medium":      dict(max_samples=3000, epochs=20),  # ≈ 50 s  (default)
    "large":       dict(max_samples=5000, epochs=40),  # ≈ 150 s (headline)
    "extra large": dict(max_samples=8500, epochs=60),  # ≈ 270 s (full train)
}
DEFAULT_TIER = "medium"

SEED = 42

# Hardcoded recipe. New scientific knobs go on the oscilloscope CLI (ar011);
# this runner just passes the recipe values inline. To explore variants, use
# autoresearch on a branch, NOT by editing this file.
RECIPE: dict[str, str] = {
    "--model":          "ping",
    "--dataset":        "shd",
    "--readout":        "li",
    "--dt":             "2.0",
    "--t-ms":           "1000",
    "--n-hidden":       "128",
    "--ei-strength":    "0.5",
    "--ei-ratio":       "2.0",
    "--lr":             "0.001",
    "--batch-size":     "256",
    "--v-grad-dampen":  "80",
    "--w-in":           "0.3 0.06",
    "--w-in-sparsity":  "0.0",
}

CELL_NAME = "baseline"


# ── Paths ───────────────────────────────────────────────────────────
def cell_dir() -> Path:
    return ARTIFACTS / CELL_NAME


def build_train_args(tier: str, out_dir: Path) -> list[str]:
    cfg = TIER_CONFIG[tier]
    args = [
        "train",
        "--max-samples", str(cfg["max_samples"]),
        "--epochs",      str(cfg["epochs"]),
        "--seed",        str(SEED),
        "--out-dir",     str(out_dir),
        "--wipe-dir",
    ]
    for flag, value in RECIPE.items():
        # --w-in takes two floats; everything else a single token.
        args += [flag, *value.split()]
    return args


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


# ── Figure ──────────────────────────────────────────────────────────
def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_training_curves(metrics: dict, out_path: Path, run_id: str) -> None:
    """Two panels: test accuracy (left), train + test loss (right). The CLI
    train loop reports a single 'acc' field (held-out test accuracy) per
    epoch, plus train + test loss separately."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"

    epochs = metrics.get("epochs") or []
    eps   = np.array([e["ep"]   for e in epochs])
    acc   = np.array([e["acc"]  for e in epochs])               # test acc
    loss  = np.array([e["loss"] for e in epochs])               # train loss
    tloss = np.array([e.get("test_loss", np.nan) for e in epochs])

    fig, (ax_a, ax_l) = plt.subplots(1, 2, figsize=(12.0, 6.75), dpi=150)

    best_acc   = float(metrics.get("best_acc", float(np.nanmax(acc))))
    best_epoch = int(metrics.get("best_epoch", int(eps[int(np.nanargmax(acc))])))

    ax_a.plot(eps, acc, color=theme.INK_BLACK, lw=1.6, label="test")
    ax_a.axhline(best_acc, color=theme.GREY_MID, lw=0.8, ls=":")
    ax_a.text(eps[-1], best_acc, f"  best ≈ {best_acc:.1f}%",
              ha="right", va="bottom",
              fontsize=theme.SIZE_CAPTION, color=theme.LABEL)
    ax_a.set_xlabel("epoch", fontsize=theme.SIZE_LABEL)
    ax_a.set_ylabel("test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax_a.set_title("accuracy", loc="left", fontsize=theme.SIZE_LABEL, pad=4)
    ax_a.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="lower right")
    _despine(ax_a)

    ax_l.plot(eps, loss,  color=theme.INK_BLACK, lw=1.6, label="train")
    if not np.all(np.isnan(tloss)):
        ax_l.plot(eps, tloss, color=theme.DEEP_RED, lw=1.6, label="test")
    ax_l.set_xlabel("epoch", fontsize=theme.SIZE_LABEL)
    ax_l.set_ylabel("loss", fontsize=theme.SIZE_LABEL)
    ax_l.set_title("loss", loc="left", fontsize=theme.SIZE_LABEL, pad=4)
    ax_l.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    _despine(ax_l)

    fig.suptitle(
        f"SHD baseline (PING, n_hidden=128, dt=2 ms, t=1000 ms)  ·  "
        f"best ≈ {best_acc:.1f}% at epoch {best_epoch}",
        fontsize=theme.SIZE_TITLE, x=0.06, ha="left", y=0.97,
    )
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.07, right=0.97, wspace=0.22)
    stamp_figure(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────
def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    skip_training = "--no-train" in sys.argv
    wipe_dir = "--no-wipe-dir" not in sys.argv

    run_id = next_run_id(SLUG)
    prepare_run_dirs(SLUG, run_id, wipe=wipe_dir, make_artifacts=True)

    t_start = time.monotonic()
    out = cell_dir()

    if not skip_training:
        dispatcher = BatchDispatcher(modal_gpu, REPO, OSCILLOSCOPE)
        print(f"[train] tier={tier}  →  {out.relative_to(REPO)}")
        dispatcher.submit(build_train_args(tier, out), out)
        dispatcher.drain()

    metrics_path = out / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"missing {metrics_path}; train run did not complete.")
    metrics = load_metrics(out)

    fig_training_curves(metrics, FIGURES / "training_curves.png", run_id)
    print(f"wrote {FIGURES / 'training_curves.png'}")

    duration_s = time.monotonic() - t_start
    best_acc   = float(metrics.get("best_acc", 0.0))
    best_epoch = int(metrics.get("best_epoch", 0))
    train_elapsed = float(metrics.get("total_elapsed_s") or 0.0)

    summary = {
        "notebook_run_id":  run_id,
        "duration_s":       round(duration_s, 1),
        "duration_human":   format_duration(duration_s),
        "train_elapsed_s":  round(train_elapsed, 1),
        "train_elapsed_human": format_duration(train_elapsed),
        "tier":             tier,
        "tier_config":      TIER_CONFIG[tier],
        "recipe":           RECIPE,
        "seed":             SEED,
        "best_acc":         round(best_acc, 2),
        "best_epoch":       best_epoch,
        "n_epochs_run":     len(metrics.get("epochs") or []),
    }
    (FIGURES / "numbers.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  best_acc = {best_acc:.2f}%  at epoch {best_epoch}")
    print(f"  notebook duration: {format_duration(duration_s)}  "
          f"(training: {format_duration(train_elapsed)})")


if __name__ == "__main__":
    main()
