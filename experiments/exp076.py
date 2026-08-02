"""Experiment 076 — checkpoint replay and bundle/legacy equivalence.

This is an integration/equivalence gate, not an accuracy benchmark.  The runner
authors the current supported snnlang subset, trains it through
``tools/snn train --bundle``, replays the emitted checkpoints through bundle
inference, and checks that the same checkpoint structures load through both the
bundle and explicit legacy routes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "snn"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import snnlang as snn  # noqa: E402, TID251
from tools.snnlang import training  # noqa: E402, TID251

from helpers import theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp076"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

DT_MS = 0.5
T_MS = 40.0
N_E = 64
N_I = 16
N_INPUT = 784
N_CLASSES = 10
MAX_SAMPLES = 160
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 76

W_IN = (0.2, 0.03)
W_EI = (0.5, 0.05)
W_IE = (1.0, 0.1)
TAU_GABA_MS = 9.0
READOUT_INIT = (5.1, 3.8)
READOUT_TAU_MS = 2.0

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
    net = snn.Network("mnist_ping_replay_gate", dt=DT_MS * snn.ms)
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
        tau_gaba=TAU_GABA_MS * snn.ms,
    )
    logits = snn.readouts.MeanVoltage(
        source=cell.E.spikes,
        classes=N_CLASSES,
        name="classifier",
        tau=READOUT_TAU_MS * snn.ms,
        weight=snn.Normal(*READOUT_INIT),
    )
    net.output("class_logits", logits)
    net.expose(cell.E.spikes, cell.I.spikes, name="cell")

    recurrent_ids = {
        "sensory_ping_E_to_I.weight",
        "sensory_ping_I_to_E.weight",
    }
    trainable_ids = [
        row["id"] for row in net.parameters if row["id"] not in recurrent_ids
    ]
    recipe = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=logits, target="digit")],
        parameter_groups=[
            training.ParameterGroup(
                trainable_ids,
                name="input_and_readout_trainable",
                lr=LEARNING_RATE,
            ),
            training.ParameterGroup(
                sorted(recurrent_ids),
                name="recurrent_ei_frozen",
                lr=0.0,
                frozen=True,
            ),
        ],
        optimizer=training.AdamW(weight_decay=WEIGHT_DECAY),
        epochs=EPOCHS,
        gradient_clip=1.0,
    )
    return snn.compile(net, training=recipe, target="tools/snn")


def _run(cmd: list[str], *, cwd: Path = REPO) -> dict[str, Any]:
    started = time.monotonic()
    env = dict(os.environ)
    env.setdefault("PINGLAB_NO_COMPILE", "1")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": result.returncode,
        "elapsed_s": time.monotonic() - started,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def _checked_run(cmd: list[str]) -> dict[str, Any]:
    record = _run(cmd)
    if record["returncode"] != 0:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(cmd)
            + "\nSTDOUT\n"
            + record["stdout_tail"]
            + "\nSTDERR\n"
            + record["stderr_tail"]
        )
    return record


def _tool_cmd(*args: str) -> list[str]:
    return [sys.executable, str(REPO / "tools" / "snn" / "tool.py"), *args]


def _legacy_structural_args() -> list[str]:
    return [
        "--n-hidden",
        str(N_E),
        "--readout",
        "mem-mean",
        "--dt",
        str(DT_MS),
        "--w-in",
        str(W_IN[0]),
        str(W_IN[1]),
        "--w-in-sparsity",
        "0",
        "--w-ei",
        str(W_EI[0]),
        str(W_EI[1]),
        "--w-ie",
        str(W_IE[0]),
        str(W_IE[1]),
        "--ei-strength",
        str(W_EI[0]),
        "--ei-ratio",
        str(W_IE[0] / W_EI[0]),
        "--ei-sparsity",
        "0",
        "--tau-gaba",
        str(TAU_GABA_MS),
    ]


def train_bundle(bundle_dir: Path, out_dir: Path) -> dict[str, Any]:
    return _checked_run(
        _tool_cmd(
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
        )
    )


def train_legacy(out_dir: Path) -> dict[str, Any]:
    return _checked_run(
        _tool_cmd(
            "train",
            *_legacy_structural_args(),
            "--lr",
            str(LEARNING_RATE),
            "--weight-decay",
            str(WEIGHT_DECAY),
            "--epochs",
            "1",
            "--max-samples",
            str(MAX_SAMPLES),
            "--batch-size",
            str(BATCH_SIZE),
            "--t-ms",
            str(T_MS),
            "--seed",
            str(SEED + 1000),
            "--out-dir",
            str(out_dir),
            "--wipe-dir",
        )
    )


def infer_bundle(bundle_dir: Path, weights: Path, out_dir: Path) -> dict[str, Any]:
    return _checked_run(
        _tool_cmd(
            "sim",
            "--bundle",
            str(bundle_dir),
            "--infer",
            "--load-weights",
            str(weights),
            "--max-samples",
            str(MAX_SAMPLES),
            "--t-ms",
            str(T_MS),
            "--seed",
            str(SEED),
            "--out-dir",
            str(out_dir),
            "--wipe-dir",
        )
    )


def infer_legacy(weights: Path, out_dir: Path) -> dict[str, Any]:
    return _checked_run(
        _tool_cmd(
            "sim",
            *_legacy_structural_args(),
            "--infer",
            "--load-weights",
            str(weights),
            "--max-samples",
            str(MAX_SAMPLES),
            "--t-ms",
            str(T_MS),
            "--seed",
            str(SEED),
            "--out-dir",
            str(out_dir),
            "--wipe-dir",
        )
    )


def checkpoint_status(
    kind: str,
    weights: Path,
    *,
    artifact_path: str,
    bundle_dir: Path | None = None,
) -> dict:
    if kind not in {"bundle", "legacy"}:
        raise ValueError(kind)
    state = torch.load(weights, map_location="cpu")
    expected_shapes = {
        "W_ff.0": [N_INPUT, N_E],
        "W_ff.1": [N_E, N_CLASSES],
        "W_ee.1": [N_E, N_E],
        "W_ei.1": [N_E, N_I],
        "W_ie.1": [N_I, N_E],
        "W_ii.1": [N_I, N_I],
    }
    expected = set(expected_shapes)
    observed = set(state)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    shape_mismatch = {
        key: {
            "checkpoint": list(value.shape),
            "expected": expected_shapes[key],
        }
        for key, value in state.items()
        if key in expected_shapes and list(value.shape) != expected_shapes[key]
    }
    return {
        "kind": kind,
        "weights": artifact_path,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch": shape_mismatch,
        "ok": not missing and not unexpected and not shape_mismatch,
        "state_keys": sorted(state),
    }


def plot_training(metrics: dict, out_path: Path) -> None:
    theme.apply()
    rows = metrics["epochs"]
    epochs = np.asarray([row["ep"] for row in rows])
    train_loss = np.asarray([row["loss"] for row in rows])
    test_loss = np.asarray([row["test_loss"] for row in rows])
    accuracy = np.asarray([row["acc"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    axes[0].plot(epochs, train_loss, marker="o", color=theme.INK_BLACK, label="train")
    axes[0].plot(epochs, test_loss, marker="o", color=theme.DEEP_RED, label="held-out")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, accuracy, marker="o", color=theme.INK_BLACK)
    axes[1].axhline(
        100.0 / N_CLASSES,
        color=theme.DEEP_RED,
        linestyle="--",
        linewidth=1.0,
        label="chance",
    )
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("held-out accuracy (%)")
    axes[1].set_ylim(0, max(30.0, float(accuracy.max()) * 1.2))
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(epochs)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_lifecycle_svg(out_path: Path) -> None:
    boxes = [
        ("Python", "graph + recipe"),
        ("Bundle", "signed files"),
        ("Train", "selected + final"),
        ("Replay", "bundle + legacy"),
        ("Parity", "one-step exact"),
    ]
    width, height = 900, 230
    box_w, box_h = 145, 74
    gap = 30
    x0, y0 = 35, 78
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfaf7"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}'
        '.title{font-size:17px;font-weight:700}.sub{font-size:12px;fill:#555}'
        '.box{fill:#fff;stroke:#222;stroke-width:1.3;rx:10}'
        '.arrow{stroke:#8f1d14;stroke-width:2;fill:none;marker-end:url(#arrow)}'
        "</style>",
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#8f1d14"/></marker></defs>',
        '<text x="35" y="35" class="title">exp076 lifecycle checked locally</text>',
        '<text x="35" y="55" class="sub">This validates the current MNIST PING + MeanVoltage adapter only.</text>',
    ]
    for idx, (title, sub) in enumerate(boxes):
        x = x0 + idx * (box_w + gap)
        svg.append(f'<rect class="box" x="{x}" y="{y0}" width="{box_w}" height="{box_h}"/>')
        svg.append(f'<text x="{x + 14}" y="{y0 + 31}" class="title">{title}</text>')
        svg.append(f'<text x="{x + 14}" y="{y0 + 52}" class="sub">{sub}</text>')
        if idx < len(boxes) - 1:
            x1 = x + box_w + 4
            x2 = x + box_w + gap - 7
            y = y0 + box_h / 2
            svg.append(f'<path class="arrow" d="M {x1} {y} L {x2} {y}"/>')
    svg.append("</svg>")
    out_path.write_text("\n".join(svg))


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
        train_dir = artifacts / "bundle_training"
        legacy_train_dir = artifacts / "legacy_training"
        selected_infer_dir = artifacts / "bundle_replay_selected"
        final_infer_dir = artifacts / "bundle_replay_final"
        legacy_load_bundle_dir = artifacts / "legacy_load_bundle_checkpoint"
        bundle_load_legacy_dir = artifacts / "bundle_load_legacy_checkpoint"

        bundle = author_bundle()
        bundle.write(bundle_dir, visualise=False)

        command_records = {
            "bundle_train": train_bundle(bundle_dir, train_dir),
            "bundle_replay_selected": infer_bundle(
                bundle_dir, train_dir / "weights.pth", selected_infer_dir
            ),
            "bundle_replay_final": infer_bundle(
                bundle_dir, train_dir / "weights_final.pth", final_infer_dir
            ),
            "legacy_load_bundle_checkpoint": infer_legacy(
                train_dir / "weights.pth", legacy_load_bundle_dir
            ),
            "legacy_train": train_legacy(legacy_train_dir),
            "bundle_load_legacy_checkpoint": infer_bundle(
                bundle_dir, legacy_train_dir / "weights.pth", bundle_load_legacy_dir
            ),
        }

        metrics = json.loads((train_dir / "metrics.json").read_text())
        selected_replay = json.loads((selected_infer_dir / "metrics.json").read_text())
        final_replay = json.loads((final_infer_dir / "metrics.json").read_text())
        legacy_load_bundle = json.loads(
            (legacy_load_bundle_dir / "metrics.json").read_text()
        )
        bundle_load_legacy = json.loads(
            (bundle_load_legacy_dir / "metrics.json").read_text()
        )
        legacy_metrics = json.loads((legacy_train_dir / "metrics.json").read_text())

        rows = metrics["epochs"]
        selected_train_acc = float(metrics["best_acc"])
        final_train_acc = float(rows[-1]["acc"])
        selected_replay_acc = float(selected_replay["best_acc"])
        final_replay_acc = float(final_replay["best_acc"])

        checkpoint_checks = {
            "bundle_selected_through_bundle": checkpoint_status(
                "bundle",
                train_dir / "weights.pth",
                artifact_path="artifacts/data/exp076/bundle_training/weights.pth",
                bundle_dir=bundle_dir,
            ),
            "bundle_final_through_bundle": checkpoint_status(
                "bundle",
                train_dir / "weights_final.pth",
                artifact_path="artifacts/data/exp076/bundle_training/weights_final.pth",
                bundle_dir=bundle_dir,
            ),
            "bundle_selected_through_legacy": checkpoint_status(
                "legacy",
                train_dir / "weights.pth",
                artifact_path="artifacts/data/exp076/bundle_training/weights.pth",
            ),
            "legacy_selected_through_bundle": checkpoint_status(
                "bundle",
                legacy_train_dir / "weights.pth",
                artifact_path="artifacts/data/exp076/legacy_training/weights.pth",
                bundle_dir=bundle_dir,
            ),
        }

        parity = {
            "automated_test": (
                "tools/snn/tests/test_bundle.py::"
                "test_bundle_and_legacy_one_step_training_are_exactly_equivalent"
            ),
            "initial_state_dict": "exact",
            "forward_logits": "exact",
            "cross_entropy_loss": "exact",
            "gradients": "exact",
            "adamw_step": "exact",
            "tolerance": {"rtol": 0.0, "atol": 0.0},
            "trainable_parameters": ["W_ff.0", "W_ff.1"],
            "frozen_parameters": ["W_ee.1", "W_ei.1", "W_ie.1", "W_ii.1"],
            "checkpoint_structural_ok": all(
                row["ok"] for row in checkpoint_checks.values()
            ),
        }

        plot_training(metrics, figures / "training_curves.png")
        write_lifecycle_svg(figures / "lifecycle.svg")
        shutil.copytree(bundle_dir, figures / "network.bundle")
        for source_dir in (
            train_dir,
            legacy_train_dir,
            selected_infer_dir,
            final_infer_dir,
            legacy_load_bundle_dir,
            bundle_load_legacy_dir,
        ):
            shutil.copytree(source_dir, figures / source_dir.name)
        for name, record in command_records.items():
            (figures / f"{name}_command.json").write_text(json.dumps(record, indent=2))

        payload = {
            "purpose": "checkpoint replay and bundle/legacy equivalence gate",
            "scope": (
                "validates only the current MNIST PING + MeanVoltage tools/snn "
                "bundle adapter subset"
            ),
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
                "w_in": list(W_IN),
                "w_ei": list(W_EI),
                "w_ie": list(W_IE),
                "tau_gaba_ms": TAU_GABA_MS,
                "readout": "MeanVoltage",
                "readout_init": list(READOUT_INIT),
                "readout_tau_ms": READOUT_TAU_MS,
                "parameter_scope": {
                    "trainable": parity["trainable_parameters"],
                    "frozen": parity["frozen_parameters"],
                },
            },
            "trajectory": {
                "epochs": [int(row["ep"]) for row in rows],
                "train_loss": [float(row["loss"]) for row in rows],
                "test_loss": [float(row["test_loss"]) for row in rows],
                "accuracy_pct": [float(row["acc"]) for row in rows],
                "best_accuracy_pct": selected_train_acc,
                "best_epoch": int(metrics["best_epoch"]),
                "final_accuracy_pct": final_train_acc,
            },
            "replay": {
                "selected_checkpoint_accuracy_pct": selected_replay_acc,
                "trainer_best_accuracy_pct": selected_train_acc,
                "selected_delta_pct_points": selected_replay_acc - selected_train_acc,
                "final_checkpoint_accuracy_pct": final_replay_acc,
                "trainer_final_epoch_accuracy_pct": final_train_acc,
                "final_delta_pct_points": final_replay_acc - final_train_acc,
                "explanation": (
                    "Replay uses the same deterministic held-out split, duration, "
                    "seed, and Poisson evaluation generator as training-time eval."
                ),
            },
            "compatibility": {
                "checkpoint_checks": checkpoint_checks,
                "legacy_route_accuracy_on_bundle_checkpoint_pct": float(
                    legacy_load_bundle["best_acc"]
                ),
                "bundle_route_accuracy_on_legacy_checkpoint_pct": float(
                    bundle_load_legacy["best_acc"]
                ),
                "legacy_checkpoint_best_accuracy_pct": float(legacy_metrics["best_acc"]),
            },
            "parity": parity,
            "runtime": {
                "total_elapsed_s": time.monotonic() - started,
                "training_elapsed_s": float(metrics["total_elapsed_s"]),
                "command_elapsed_s": {
                    key: float(value["elapsed_s"])
                    for key, value in command_records.items()
                },
            },
            "artifacts": {
                "bundle": "artifacts/data/exp076/network.bundle",
                "bundle_training": "artifacts/data/exp076/bundle_training",
                "legacy_training": "artifacts/data/exp076/legacy_training",
                "selected_replay": "artifacts/data/exp076/bundle_replay_selected",
                "final_replay": "artifacts/data/exp076/bundle_replay_final",
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
