"""Notebook runner for entry 060 — SHD training smoke test.

The first training run in the Spiking Heidelberg Digits program, and a low bar
on purpose: does the pipeline actually *train*? Runs Rung A of the plan — the
free signed-recurrent ceiling (all four recurrent blocks trainable,
--no-dales-law, gamma untuned) — on a 1000-sample SHD subset for a few dozen
epochs and plots the learning curves. Success is only: loss falls and test
accuracy climbs off the 5% chance floor. This is engineering validation, NOT a
registered result — the real baseline number comes from a full-scale run.

Writing: writings/exp060.typ · figures + numbers.json: artifacts/data/exp060/
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.figsave import save_figure  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_cli import run_cli  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp060"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

N_CLASSES = 20
CHANCE_PCT = 100.0 / N_CLASSES  # 5% on 20 classes

# Rung A on a subset: the RSNN ceiling (signed, all four blocks trainable, no
# Dale's law) at coarse dt so a full 1 s window still trains in minutes. The
# firing-rate regulariser (Cramer et al.'s SHD values) keeps the signed
# recurrence from running away — the 526 Hz inhibitory blow-up the unregularised
# one-epoch smoke showed. lr is the biophysical-model value, not the 0.01 default.
TRAIN_ARGS = [
    "train",
    "--dataset", "shd",
    "--max-samples", "1000",
    "--t-ms", "1000",
    "--dt", "1.0",
    "--n-hidden", "256",
    "--model", "ping",
    "--epochs", "30",
    "--batch-size", "32",
    "--lr", "0.001",
    "--weight-decay", "0.001",
    "--no-dales-law",
    "--w-ee", "0.3", "0.1",
    "--trainable-w-ee", "--trainable-w-ei", "--trainable-w-ie", "--trainable-w-ii",
    "--fr-reg-upper-theta", "100",
    "--fr-reg-upper-strength", "0.06",
]

SCALE = {
    "dataset": "shd",
    "rung": "A (free signed-recurrent ceiling)",
    "max_samples": 1000,
    "t_ms": 1000,
    "dt": 1.0,
    "n_hidden": 256,
    "epochs": 30,
    "lr": 0.001,
    "weight_decay": 0.001,
    "dales_law": False,
    "trainable_blocks": ["W_ee", "W_ei", "W_ie", "W_ii"],
}


def _epochs(metrics: dict) -> list[dict]:
    eps = metrics.get("epochs") or []
    if not eps:
        raise SystemExit("no per-epoch metrics — did training run?")
    return eps


def plot_loss(metrics: dict, stem: Path) -> None:
    """Train + test loss against epoch — the 'is it learning at all' curve."""
    theme.apply()
    eps = _epochs(metrics)
    x = [e["ep"] for e in eps]
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax.plot(x, [e["loss"] for e in eps], color=theme.INK_BLACK, lw=2.0, label="train")
    ax.plot(
        x, [e.get("test_loss") for e in eps],
        color=theme.DEEP_RED, lw=2.0, label="test",
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_xlim(min(x), max(x))
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def plot_accuracy(metrics: dict, stem: Path) -> None:
    """Test accuracy against epoch, with the 5% chance floor drawn in."""
    theme.apply()
    eps = _epochs(metrics)
    x = [e["ep"] for e in eps]
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax.axhline(
        CHANCE_PCT, color=theme.GREY_MID, lw=1.0, ls="--",
        label=f"chance ({CHANCE_PCT:.0f}%)",
    )
    ax.plot(x, [e["acc"] for e in eps], color=theme.INK_BLACK, lw=2.0, label="test accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (%)")
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(0, max(CHANCE_PCT * 2, max(e["acc"] for e in eps) * 1.15))
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def main() -> None:
    meta = parse_meta(sys.argv)

    t_start = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {run_id}")

    with published_run(
        SLUG, run_id, make_artifacts=True, scale=SCALE,
        skip_training=meta.skip_training, plot_only=meta.plot_only,
    ) as (artifacts, figures):
        train_dir = artifacts / "rung_a"
        metrics_path = train_dir / "metrics.json"

        if meta.skip_training and metrics_path.exists():
            print(f"[skip-training] reusing {metrics_path}")
        else:
            print(f"[train] {' '.join(TRAIN_ARGS)}")
            run_cli([*TRAIN_ARGS, "--out-dir", str(train_dir)])

        metrics = json.loads(metrics_path.read_text())
        eps = _epochs(metrics)

        plot_loss(metrics, figures / "loss")
        print(f"wrote {figures / 'loss.svg'}")
        plot_accuracy(metrics, figures / "accuracy")
        print(f"wrote {figures / 'accuracy.svg'}")

        first_acc = eps[0]["acc"]
        payload = {
            "n_train_subset": SCALE["max_samples"],
            "epochs": len(eps),
            "chance_pct": CHANCE_PCT,
            "first_epoch_acc_pct": first_acc,
            "best_acc_pct": metrics.get("best_acc"),
            "best_epoch": metrics.get("best_epoch"),
            "final_acc_pct": eps[-1]["acc"],
            "first_epoch_loss": eps[0]["loss"],
            "final_loss": eps[-1]["loss"],
            "loss_fell": eps[-1]["loss"] < eps[0]["loss"],
            "beat_chance": (metrics.get("best_acc") or 0) > CHANCE_PCT,
        }

        duration_s = time.monotonic() - t_start
        write_numbers(figures, run_id=run_id, duration_s=duration_s, payload=payload)
        print(f"wrote {figures / 'numbers.json'}")
        print(f"  best acc {payload['best_acc_pct']}% · loss "
              f"{payload['first_epoch_loss']:.3f} → {payload['final_loss']:.3f}")
        print(f"  duration: {duration_s:.1f}s")


if __name__ == "__main__":
    main()
