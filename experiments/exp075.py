"""Experiment 075 — train a small MNIST PING graph through its snnlang bundle.

This is an integration gate, not a benchmark.  The runner authors a graph and
training recipe, compiles both to a bundle, invokes ``tools/snnsim train --bundle``
on a deterministic 1,000-example MNIST subset, and plots the resulting loss and
accuracy trajectory.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# snnlang is the authoring/compiler library; the simulator remains a subprocess.
from tools import snnlang as snn  # noqa: E402, TID251
from tools.snnlang import training  # noqa: E402, TID251

from helpers import theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp075"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

DT_MS = 0.5
T_MS = 100.0
N_E = 128
N_I = 32
N_INPUT = 784
N_CLASSES = 10
MAX_SAMPLES = 1_000
BATCH_SIZE = 64
EPOCHS = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 75

SCALE = {
    "dt_ms": DT_MS,
    "t_ms": T_MS,
    "n_e": N_E,
    "n_i": N_I,
    "max_samples": MAX_SAMPLES,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "seed": SEED,
}


def author_bundle() -> snn.Bundle:
    net = snn.Network("mnist_ping_training_demo", dt=DT_MS * snn.ms)
    image_spikes = net.input(
        "image_spikes",
        shape=("time", "batch", N_INPUT),
        signal_type="spikes",
        unit="spike",
    )
    cell = snn.components.ping(
        net,
        name="sensory_ping",
        n_e=N_E,
        n_i=N_I,
        source=image_spikes,
    )
    logits = snn.readouts.MeanVoltage(
        source=cell.E.spikes,
        classes=N_CLASSES,
        name="classifier",
        tau=2 * snn.ms,
        weight=snn.Normal(5.1, 3.8),
    )
    net.output("class_logits", logits)
    net.expose(cell.E.spikes, cell.I.spikes, name="cell")

    recurrent_ids = {
        "sensory_ping_E_to_I.weight",
        "sensory_ping_I_to_E.weight",
    }
    feedforward_ids = [
        row["id"] for row in net.parameters if row["id"] not in recurrent_ids
    ]
    recipe = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=logits, target="digit")],
        parameter_groups=[
            training.ParameterGroup(
                feedforward_ids,
                name="feedforward_trainable",
                lr=LEARNING_RATE,
            ),
            training.ParameterGroup(
                sorted(recurrent_ids),
                name="recurrent_frozen",
                lr=0.0,
                frozen=True,
            ),
        ],
        optimizer=training.AdamW(weight_decay=WEIGHT_DECAY),
        epochs=EPOCHS,
        gradient_clip=1.0,
    )
    return snn.compile(net, training=recipe, target="tools/snnsim")


def run_training(bundle_dir: Path, out_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(REPO / "tools/snnsim/tool.py"),
        "train",
        "--bundle",
        str(bundle_dir),
        "--max-samples",
        str(MAX_SAMPLES),
        "--batch-size",
        str(BATCH_SIZE),
        "--t-ms",
        str(T_MS),
        "--seed",
        str(SEED),
        "--out-dir",
        str(out_dir),
        "--wipe-dir",
    ]
    print("[train]", " ".join(cmd))
    env = dict(os.environ)
    env.setdefault("PINGLAB_NO_COMPILE", "1")
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


def plot_training(metrics: dict, out_path: Path) -> None:
    theme.apply()
    rows = metrics["epochs"]
    epochs = np.asarray([row["ep"] for row in rows])
    train_loss = np.asarray([row["loss"] for row in rows])
    test_loss = np.asarray([row["test_loss"] for row in rows])
    accuracy = np.asarray([row["acc"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
    axes[0].plot(
        epochs,
        train_loss,
        marker="o",
        color=theme.INK_BLACK,
        label="train",
    )
    axes[0].plot(
        epochs,
        test_loss,
        marker="o",
        color=theme.DEEP_RED,
        label="held-out",
    )
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].legend(frameon=False)

    axes[1].plot(
        epochs,
        accuracy,
        marker="o",
        color=theme.INK_BLACK,
    )
    axes[1].axhline(
        100.0 / N_CLASSES,
        color=theme.DEEP_RED,
        linestyle="--",
        linewidth=1.0,
        label="chance",
    )
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("held-out accuracy (%)")
    axes[1].set_ylim(0, max(30.0, float(accuracy.max()) * 1.15))
    axes[1].legend(frameon=False)

    for ax in axes:
        ax.set_xticks(epochs)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    meta = parse_meta(sys.argv)
    started = time.monotonic()
    run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {run_id}")

    with published_run(
        SLUG,
        run_id,
        scale=SCALE,
        plot_only=meta.plot_only,
    ) as (artifacts, figures):
        bundle_dir = artifacts / "network.bundle"
        train_dir = artifacts / "training"

        bundle = author_bundle()
        bundle.write(bundle_dir, visualise=True)
        run_training(bundle_dir, train_dir)

        metrics = json.loads((train_dir / "metrics.json").read_text())
        rows = metrics["epochs"]
        if len(rows) != EPOCHS:
            raise RuntimeError(
                f"training emitted {len(rows)} epochs; expected {EPOCHS}"
            )
        plot_training(metrics, figures / "training_curves.png")
        shutil.copytree(bundle_dir, figures / "network.bundle")
        shutil.copy2(
            bundle_dir / "reports/circuit.svg",
            figures / "network_graph.svg",
        )
        shutil.copy2(
            train_dir / "metrics.json",
            figures / "training_metrics.json",
        )

        train_losses = [float(row["loss"]) for row in rows]
        test_losses = [float(row["test_loss"]) for row in rows]
        accuracies = [float(row["acc"]) for row in rows]
        payload = {
            "purpose": "bundle-driven training integration demonstration",
            "graph": {
                "name": bundle.graph["name"],
                "digest": bundle.manifest["graph_digest"],
                "training_digest": next(
                    row["digest"]
                    for row in bundle.manifest["files"]
                    if row["path"] == "training.json"
                ),
            },
            "config": {
                **SCALE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "train_count": int(rows[0]["samples"]),
                "held_out_count": MAX_SAMPLES - int(rows[0]["samples"]),
            },
            "trajectory": {
                "epochs": [int(row["ep"]) for row in rows],
                "train_loss": train_losses,
                "test_loss": test_losses,
                "accuracy_pct": accuracies,
                "train_loss_change": train_losses[-1] - train_losses[0],
                "test_loss_change": test_losses[-1] - test_losses[0],
                "accuracy_change_pct_points": accuracies[-1] - accuracies[0],
                "best_accuracy_pct": float(metrics["best_acc"]),
                "best_epoch": int(metrics["best_epoch"]),
            },
            "training": {
                "total_elapsed_s": float(metrics["total_elapsed_s"]),
                "weights_written": (train_dir / "weights.pth").is_file(),
                "final_weights_written": (
                    train_dir / "weights_final.pth"
                ).is_file(),
            },
        }
        write_numbers(
            figures,
            run_id=run_id,
            duration_s=time.monotonic() - started,
            payload=payload,
        )


if __name__ == "__main__":
    main()
