from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools"), str(REPO / "tools/snnsim")]

from experiments.exp076 import recipe
from experiments.exp076.recipe import (
    BATCH_SIZE,
    DT_MS,
    LEARNING_RATE,
    MAX_SAMPLES,
    N_E,
    SEED,
    T_MS,
    TAU_GABA_MS,
    W_EI,
    W_IE,
    W_IN,
    WEIGHT_DECAY,
    author_bundle,
)
from experiments.helpers import snnlang_stages as stages
from pingstore.contracts import PingstoreError, write_json_atomic


def _tool_cmd(*args: str) -> list[str]:
    return [sys.executable, str(REPO / "tools" / "snnsim" / "tool.py"), *args]


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
        "--w-in-initial-zero-fraction",
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
        "--recurrent-initial-zero-fraction",
        "0",
        "--tau-gaba",
        str(TAU_GABA_MS),
    ]


def train_bundle(bundle_dir: Path, out_dir: Path, provenance: Path) -> dict:
    return stages.command(
        REPO,
        provenance,
        out_dir.name,
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
        ),
    )


def train_legacy(out_dir: Path, provenance: Path) -> dict:
    return stages.command(
        REPO,
        provenance,
        out_dir.name,
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
        ),
    )


def infer_bundle(
    bundle_dir: Path, weights: Path, out_dir: Path, provenance: Path
) -> dict:
    return stages.command(
        REPO,
        provenance,
        out_dir.name,
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
        ),
    )


def infer_legacy(weights: Path, out_dir: Path, provenance: Path) -> dict:
    return stages.command(
        REPO,
        provenance,
        out_dir.name,
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
        ),
    )


def compute(*, run_id=None):
    with stages.execution(REPO, recipe, "compute", run_id=run_id) as run:
        bundle_dir = run.export / "network.bundle"
        train_dir = run.export / "bundle_training"
        legacy_train_dir = run.export / "legacy_training"
        selected_infer_dir = run.export / "bundle_replay_selected"
        final_infer_dir = run.export / "bundle_replay_final"
        legacy_load_bundle_dir = run.export / "legacy_load_bundle_checkpoint"
        bundle_load_legacy_dir = run.export / "bundle_load_legacy_checkpoint"

        bundle = author_bundle()
        bundle.write(bundle_dir, visualise=False)

        command_records = {
            "bundle_train": train_bundle(bundle_dir, train_dir, run.scratch),
            "bundle_replay_selected": infer_bundle(
                bundle_dir,
                train_dir / "weights.pth",
                selected_infer_dir,
                run.scratch,
            ),
            "bundle_replay_final": infer_bundle(
                bundle_dir,
                train_dir / "weights_final.pth",
                final_infer_dir,
                run.scratch,
            ),
            "legacy_load_bundle_checkpoint": infer_legacy(
                train_dir / "weights.pth", legacy_load_bundle_dir, run.scratch
            ),
            "legacy_train": train_legacy(legacy_train_dir, run.scratch),
            "bundle_load_legacy_checkpoint": infer_bundle(
                bundle_dir,
                legacy_train_dir / "weights.pth",
                bundle_load_legacy_dir,
                run.scratch,
            ),
        }

        parity = stages.test_evidence(
            REPO, run, "one-step-parity", [recipe.PARITY_TEST]
        )
        write_json_atomic(run.export / "parity.json", parity)
        run.record["execution"]["commands"] = command_records
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="unused source-neutral reservation")
    args = parser.parse_args()
    try:
        compute(run_id=args.run_id)
    except (PingstoreError, OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
