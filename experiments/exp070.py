"""Registered first exp070 attempt: matched 2 ms SHD temporal resolution.

This runner reuses exp069's validated split, checkpoint-selection, training,
diagnostic, raster, and RunPod plumbing.  It changes only the experiment paths,
the integration/input-bin width (1 ms -> 2 ms), and the registered short-run
duration (80 epochs -> 5 epochs).  The official SHD test has no route here.

Default execution is the local 128/128, two-epoch smoke.  Cloud execution needs
the explicit ``--runpod --live`` spending gate.  Collection publishes the
attempt and derives promotion or kill from the locked epoch-five thresholds.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp069 as baseline  # noqa: E402
from helpers import runpod  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp070"
ATTEMPT = "temporal_2ms"
SHORT_EPOCHS = 5
DT_MS = 2.0
PROMOTION_ACCURACY_GAIN_PP = 3.0

ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
SCRATCH = runpod.artifacts_scratch(SLUG)
CELL_ROOT = SCRATCH / "cells" / ATTEMPT
FROZEN_ROOT = SCRATCH / "frozen" / ATTEMPT
FAILURE_ROOT = SCRATCH / "failures" / ATTEMPT
SMOKE_ROOT = REPO / "temp" / "experiments" / SLUG / "smoke" / ATTEMPT
LOCAL_SPLIT_ROOT = REPO / "temp" / "experiments" / SLUG / "split"
INSTALLED_SPLIT_ROOT = REPO / "temp" / "experiments" / SLUG / "installed_shd"

EXP069_RAW = REPO / "artifacts" / "data" / "exp069" / "raw"
EXP069_NUMBERS = REPO / "artifacts" / "data" / "exp069" / "numbers.json"


def configure_baseline() -> None:
    """Point exp069's validated machinery at the locked exp070 candidate."""
    baseline.__dict__["SLUG"] = SLUG
    baseline.ARTIFACTS = ARTIFACTS
    baseline.FIGURES = FIGURES
    baseline.SCRATCH = SCRATCH
    baseline.CELL_ROOT = CELL_ROOT
    baseline.FROZEN_ROOT = FROZEN_ROOT
    baseline.FAILURE_ROOT = FAILURE_ROOT
    baseline.SMOKE_ROOT = SMOKE_ROOT
    baseline.LOCAL_SPLIT_ROOT = LOCAL_SPLIT_ROOT
    baseline.SHD_DIR = INSTALLED_SPLIT_ROOT
    baseline.__dict__["EPOCHS"] = SHORT_EPOCHS
    baseline.DT_MS = DT_MS
    baseline.SCALE = {
        **baseline.SCALE,
        "experiment": SLUG,
        "attempt": ATTEMPT,
        "epochs": SHORT_EPOCHS,
        "dt_ms": DT_MS,
    }
    # The engine historically resolves SHD through a module-level directory.
    # Redirect it to exp070's staged development-only alias.  This prevents the
    # validation filename expected by the CLI from touching /tmp/shd's official
    # test file, even for a backup/restore copy.
    _, _, snn_train, _, _, _ = baseline.import_snn_modules()
    snn_datasets = sys.modules[snn_train.load_dataset.__module__]
    setattr(snn_datasets, "_SHD_DIR", str(INSTALLED_SPLIT_ROOT))


configure_baseline()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_audit() -> dict[str, Any]:
    """Quantify duration truncation and deterministic bin collisions on train."""
    import h5py

    source = Path(baseline._shd_h5("train"))
    expected = "2bddb4bd46732f09982b7d1631b7c29c19853c73d3d240e3eb32bba909bdd6c1"
    if sha256_file(source) != expected:
        raise RuntimeError("SHD development source hash differs from exp069")
    event_count = 0
    late_event_count = 0
    late_sample_count = 0
    collision_event_count = 0
    audit_indices: list[int] = []
    with h5py.File(source, "r") as handle:
        sample_count = len(handle["labels"])
        audit_indices = list(range(0, sample_count, 8))
        for index in range(sample_count):
            times = np.asarray(handle["spikes/times"][index])
            event_count += int(times.size)
            n_late = int(np.count_nonzero(times >= baseline.T_MS / 1000.0))
            late_event_count += n_late
            late_sample_count += int(n_late > 0)
            if index not in audit_indices:
                continue
            units = np.asarray(handle["spikes/units"][index], dtype=np.int64)
            bins = np.floor(times / (DT_MS / 1000.0)).astype(np.int64)
            keys = bins * baseline.N_INPUT + units
            collision_event_count += int(keys.size - np.unique(keys).size)
    audited_events = 0
    with h5py.File(source, "r") as handle:
        audited_events = sum(int(np.asarray(handle["spikes/times"][i]).size) for i in audit_indices)
    return {
        "source_sha256": expected,
        "source_sample_count": sample_count,
        "source_event_count": event_count,
        "events_at_or_after_window": late_event_count,
        "events_at_or_after_window_pct": 100.0 * late_event_count / event_count,
        "samples_with_events_at_or_after_window": late_sample_count,
        "samples_with_events_at_or_after_window_pct": 100.0 * late_sample_count / sample_count,
        "collision_audit_selection": "every eighth development utterance from index zero",
        "collision_audit_sample_count": len(audit_indices),
        "collision_audit_event_count": audited_events,
        "same_channel_bin_collision_event_count": collision_event_count,
        "same_channel_bin_collision_pct": 100.0 * collision_event_count / audited_events,
        "candidate_dt_ms": DT_MS,
        "window_ms": baseline.T_MS,
    }


def run_smoke() -> None:
    baseline.run_smoke()
    summary_path = SMOKE_ROOT / "smoke_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["attempt"] = ATTEMPT
    summary["candidate_dt_ms"] = DT_MS
    summary["input_audit"] = input_audit()
    baseline.atomic_json(summary_path, summary)


def run_full_cell(model: str) -> None:
    baseline.run_full_cell(model)


def cell_done(model: str) -> bool:
    return baseline.cell_done(model)


def pod_run() -> None:
    runpod.pod_run_loop(
        job_ids=list(baseline.MODELS),
        is_done=cell_done,
        run_job=run_full_cell,
        label="exp070-temporal-2ms-short",
    )


def run_via_runpod(meta: Any) -> None:
    runpod.dispatch(
        slug=SLUG,
        runner=SLUG,
        buckets=[
            {"name": "exp070-2ms-coba", "cells": ["coba"]},
            {"name": "exp070-2ms-ping", "cells": ["ping"]},
        ],
        gpu=meta.gpu,
        live=meta.live,
        collect=meta.collect,
        plumbing=False,
        collect_subdir=f"{runpod.ARTIFACTS_SUBDIR}/{SLUG}",
        local_collect_dir=str(SCRATCH),
        extra_env={
            "PINGLAB_ARTIFACTS_ROOT": f"{runpod.VOLUME_MOUNT}/{runpod.ARTIFACTS_SUBDIR}/{SLUG}",
        },
        max_runtime_s=3600,
    )


def epoch_five_baseline(model: str) -> dict[str, float]:
    metrics = json.loads((EXP069_RAW / model / "metrics.json").read_text())
    row = metrics["epochs"][4]
    return {"accuracy_pct": float(row["acc"]), "cross_entropy": float(row["test_loss"])}


def attempt_decision(cells: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    diagnostics: dict[str, Any] = {}
    promoted = True
    for model in baseline.MODELS:
        epochs = cells[model]["training"]["epochs"]
        row = epochs[4]
        old = epoch_five_baseline(model)
        gain = float(row["acc"]) - old["accuracy_pct"]
        loss_change = float(row["test_loss"]) - old["cross_entropy"]
        finite = all(
            math.isfinite(float(row[key]))
            for key in ("loss", "test_loss", "acc", "grad_norm", "test_rate_e")
        )
        active = float(row["test_rate_e"]) > 0.0
        if model == "ping":
            active = active and float(row.get("test_rate_i") or 0.0) > 0.0
        clean = (
            finite
            and active
            and int(row.get("skipped_steps", 0)) == 0
            and int(row.get("nan_forward_batches", 0)) == 0
        )
        clears = gain >= PROMOTION_ACCURACY_GAIN_PP and loss_change <= 0.0 and clean
        promoted = promoted and clears
        diagnostics[model] = {
            "epoch": 5,
            "validation_accuracy_pct": float(row["acc"]),
            "validation_cross_entropy": float(row["test_loss"]),
            "validation_e_rate_hz": float(row["test_rate_e"]),
            "validation_i_rate_hz": float(row.get("test_rate_i") or 0.0),
            "gradient_norm": float(row["grad_norm"]),
            "skipped_steps": int(row.get("skipped_steps", 0)),
            "nan_forward_batches": int(row.get("nan_forward_batches", 0)),
            "exp069_epoch_five_accuracy_pct": old["accuracy_pct"],
            "exp069_epoch_five_cross_entropy": old["cross_entropy"],
            "accuracy_gain_pp": gain,
            "cross_entropy_change": loss_change,
            "clears_promotion_gate": clears,
        }
    return ("promoted" if promoted else "killed"), diagnostics


def publish_attempt() -> None:
    cells = baseline.validate_collected()
    decision, diagnostics = attempt_decision(cells)
    smoke = json.loads((SMOKE_ROOT / "smoke_summary.json").read_text())
    run_id = next_run_id(SLUG)
    t0 = time.monotonic()
    with published_run(
        SLUG,
        run_id,
        make_artifacts=True,
        scale=baseline.SCALE,
        skip_training=True,
    ) as (_, figures):
        baseline.plot_validation_curves(cells, figures / "short_validation_curves")
        baseline.plot_activity_curves(cells, figures / "short_activity_curves")
        baseline.plot_matched_rasters(figures / "matched_rasters.png")
        raw = figures / "raw" / ATTEMPT
        raw.mkdir(parents=True)
        shutil.copy2(SMOKE_ROOT / "smoke_summary.json", raw / "smoke_summary.json")
        for model in baseline.MODELS:
            source = CELL_ROOT / model
            destination = raw / model
            destination.mkdir()
            for name in ("config.json", "metrics.json", "metrics.jsonl", "checkpoint_selection.json"):
                shutil.copy2(source / name, destination / name)
            shutil.copy2(source / "validation_probe/matched_input.npz", destination / "matched_input.npz")
            shutil.copy2(
                source / "validation_probe/matched_rasters/rasters.npz",
                destination / "matched_rasters.npz",
            )
        payload = {
            "result_status": decision,
            "stage": "short_candidate",
            "attempt": ATTEMPT,
            "seed": baseline.SEED,
            "promotion_accuracy_gain_pp": PROMOTION_ACCURACY_GAIN_PP,
            "cells": diagnostics,
            "split": {
                "development_train_count": 7340,
                "validation_count": 816,
                "train_indices_sha256": baseline.EXP068_TRAIN_HASH,
                "validation_indices_sha256": baseline.EXP068_VALIDATION_HASH,
            },
            "input_audit": smoke["input_audit"],
            "config": baseline.SCALE,
            "integrity": {
                "official_test_access": "forbidden and absent from runner",
                "matched_partitions": True,
                "selection_rule": cells["coba"]["selection"]["rule"],
            },
            "runpod": {"spend_usd": None, "active_pods_after_collection": None},
        }
        write_numbers(figures, run_id=run_id, duration_s=time.monotonic() - t0, payload=payload)
        (figures / "reproduce.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "uv run python experiments/exp070.py\n"
            "uv run python experiments/exp070.py --runpod\n"
            "# Add --live only with fresh, explicit RunPod spending authority.\n"
            "uv run python experiments/exp070.py --runpod --collect\n"
            "uv run python experiments/exp070.py --skip-training\n"
        )


def main() -> None:
    meta = parse_meta(sys.argv, allow_dispatch=True)
    if meta.reap:
        runpod.reap_all_pods()
        return
    if meta.pod_run:
        pod_run()
        return
    if meta.runpod:
        run_via_runpod(meta)
        return
    if meta.skip_training or meta.plot_only:
        publish_attempt()
        return
    run_smoke()


if __name__ == "__main__":
    main()
