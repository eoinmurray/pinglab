"""Exploratory exp071 SHD ladder using the cumulative-potential readout.

This runner reuses exp069's validation-only SHD split, checkpoint selection,
diagnostics, matched rasters, and RunPod plumbing.  The official SHD test has no
route here.  The default command runs a local 128/128 two-epoch smoke for the
selected registered candidate; cloud execution still requires the explicit
``--runpod --live`` spending gate.

Select the registered candidate with ``EXP071_ATTEMPT``:

* ``cumulative_baseline``: baseline 256-cell architecture with the newly merged
  signed cumulative-potential readout and readout bias.
* ``cumulative_256_256``: same readout with two matched 256-cell hidden layers.

Select ``EXP071_STAGE=short`` for the 8-epoch screening run and
``EXP071_STAGE=final40`` for the promoted 40-epoch run.  The runner fails closed
for any unregistered attempt or stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp069 as baseline  # noqa: E402
from helpers import runpod  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp071"
SHORT_EPOCHS = 8
FINAL_EPOCHS = 40
READOUT_MODE = "cumulative-potential"
SIGNED_READOUT = True
READOUT_BIAS = True
READOUT_SCALE = 1.0
PROMOTION_ACCURACY_GAIN_PP = 3.0
BASELINE_ATTEMPT = "cumulative_baseline"
ATTEMPT_SPECS: dict[str, dict[str, Any]] = {
    "cumulative_baseline": {
        "hidden_sizes": [256],
        "description": "baseline architecture with signed cumulative-potential readout",
        "pod_label": "cum-base",
    },
    "cumulative_256_256": {
        "hidden_sizes": [256, 256],
        "description": "two 256-cell hidden layers with signed cumulative-potential readout",
        "pod_label": "cum-256-256",
    },
}
STAGE_SPECS = {
    "short": {"epochs": SHORT_EPOCHS, "pod_suffix": "short"},
    "final40": {"epochs": FINAL_EPOCHS, "pod_suffix": "final40"},
}
ATTEMPT = os.environ.get("EXP071_ATTEMPT", BASELINE_ATTEMPT)
STAGE = os.environ.get("EXP071_STAGE", "short")
if ATTEMPT not in ATTEMPT_SPECS:
    raise RuntimeError(f"unregistered exp071 attempt: {ATTEMPT}")
if STAGE not in STAGE_SPECS:
    raise RuntimeError(f"unregistered exp071 stage: {STAGE}")
ATTEMPT_SPEC = ATTEMPT_SPECS[ATTEMPT]
STAGE_SPEC = STAGE_SPECS[STAGE]
HIDDEN_SIZES = [int(x) for x in ATTEMPT_SPEC["hidden_sizes"]]
EPOCHS = int(STAGE_SPEC["epochs"])
POD_LABEL = f"{ATTEMPT_SPEC['pod_label']}-{STAGE_SPEC['pod_suffix']}"

ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
SCRATCH = runpod.artifacts_scratch(SLUG)
CELL_ROOT = SCRATCH / "cells" / STAGE / ATTEMPT
FROZEN_ROOT = SCRATCH / "frozen" / STAGE / ATTEMPT
FAILURE_ROOT = SCRATCH / "failures" / STAGE / ATTEMPT
SMOKE_ROOT = REPO / "temp" / "experiments" / SLUG / "smoke" / ATTEMPT
LOCAL_SPLIT_ROOT = REPO / "temp" / "experiments" / SLUG / "split"
INSTALLED_SPLIT_ROOT = REPO / "temp" / "experiments" / SLUG / "installed_shd"

EXP070_RAW = REPO / "artifacts" / "data" / "exp070" / "raw"
COMMITTED_SMOKE = REPO / "artifacts" / "data" / SLUG / "raw" / "smoke" / ATTEMPT / "smoke_summary.json"
COMPUTE_LEDGER = SCRATCH / "compute_ledgers" / f"{STAGE}-{ATTEMPT}.json"
ORIGINAL_VALIDATE_TRAINING = baseline.validate_training

GOAL_PROMPT = """/goal Design and execute the next exploratory SHD experiment on branch night/spiking-heidelberg-digits/ar071, using the newly merged tools/snn cumulative-potential readout additions to try to raise validation accuracy while keeping the comparison between matched Dale-constrained COBA and PING networks scientifically clean.

Start from updated main at merge commit 9527162d7a0ef80deac0e3f606ce513b9280ba6c. Do not modify main. Open a new draft PR for this experiment when the first meaningful commit is ready.

Scientific objective:
Run a one-seed exploratory validation-only ladder on SHD to identify whether stronger temporal readout and modest capacity changes materially improve matched COBA/PING learning beyond exp068-exp070.

Constraints:
- Use one seed only.
- Do not use the official SHD test set during exploratory screening.
- Use a held-out validation split from the training set for all selection decisions.
- Keep COBA and PING matched except for their registered cell-specific voltage-gradient dampening.
- Preserve matched input, readout, training split, optimizer, batch, and preprocessing settings across COBA and PING within each candidate.
- Use short runs first for iteration speed, targeting about 8-10 epochs per candidate.
- Promote only the most promising candidate to a 40-epoch validation run.
- Do not run multiple seeds unless a later claim is worth defending and I explicitly authorize it.
- RunPod spending requires explicit approval before pod creation; use at most one pod at a time unless separately authorized.
- Reap any pod when finished.
- Do not merge the PR.
- Avoid the full local test suite on the 4 GB Hetzner host; use focused checks and demolab build.

Candidate ladder:
1. Baseline architecture with the new signed cumulative-potential readout:
   --readout cumulative-potential --signed-readout --readout-bias
2. If candidate 1 is finite, active, and improves validation learning, try modest increased capacity with two hidden layers, e.g. --n-hidden 256 256, keeping the same readout.
3. Optionally test one conservative recurrence/readout-adjacent setting if supported by existing tools/snn CLI and scientifically justified before seeing final results.
4. Promote the best candidate, if any, to a matched 40-epoch COBA/PING validation run.

Deliverables:
- Create the next numbered experiment, likely exp071, with artifacts under artifacts/data/exp071.
- Produce numbers.json, provenance, reproducer, training curves, firing-rate diagnostics, and matched input/E/I rasters where relevant.
- Write writings/exp071.typ as the canonical cold-readable experiment report.
- Include the goal prompt below the abstract and append timestamped activity-log/thread checkpoints in the experiment appendix, following the simplified SHD organization.
- Record scientific decisions, failures, anomalies, costs, commits, results, and pending work in the experiment article.
- Sanitize any publishable transcript/log content before committing.
- Build and validate with focused checks plus demolab build.
- Commit meaningful attempts, including killed attempts, with clear messages.
- Push branch night/spiking-heidelberg-digits/ar071 and update the new PR with evidence.
- Finish at the human review gate with a concise evidence summary, exact compute spend, validation performed, and links to the rendered exp071 file and PR."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_args(model: str, out: Path, *, smoke: bool) -> list[str]:
    recipe = baseline.RECIPES[model]
    args = [
        "train", "--model", "ping", "--dataset", "shd",
        "--epochs", str(baseline.SMOKE_EPOCHS if smoke else EPOCHS),
        "--t-ms", str(baseline.T_MS), "--dt", str(baseline.DT_MS),
        "--n-hidden", *(str(size) for size in HIDDEN_SIZES),
        "--batch-size", str(baseline.BATCH_SIZE),
        "--lr", str(baseline.LEARNING_RATE), "--seed", str(baseline.SEED),
        "--ei-strength", str(recipe["ei_strength"]),
        "--v-grad-dampen", str(recipe["v_grad_dampen"]),
        "--w-in", str(baseline.INPUT_SCALE),
        "--w-in-sparsity", str(baseline.INPUT_SPARSITY),
        "--readout", READOUT_MODE,
        "--readout-w-out-scale", str(READOUT_SCALE),
        "--out-dir", str(out), "--wipe-dir",
    ]
    if SIGNED_READOUT:
        args.append("--signed-readout")
    if READOUT_BIAS:
        args.append("--readout-bias")
    return args


def validate_training(model: str, out: Path, *, smoke: bool) -> list[str]:
    errors = ORIGINAL_VALIDATE_TRAINING(model, out, smoke=smoke)
    if not (out / "metrics.json").exists():
        return errors
    metrics = json.loads((out / "metrics.json").read_text())
    cfg = metrics.get("config", {})
    full_cfg = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else cfg
    hidden_sizes = full_cfg.get("hidden_sizes")
    if hidden_sizes is None and full_cfg.get("n_hidden") is not None:
        hidden_sizes = [int(full_cfg["n_hidden"])]
    if hidden_sizes != HIDDEN_SIZES:
        errors.append(f"hidden_sizes: got {hidden_sizes!r}, want {HIDDEN_SIZES!r}")
    if cfg.get("readout_mode") != READOUT_MODE:
        errors.append(f"readout_mode: got {cfg.get('readout_mode')!r}, want {READOUT_MODE!r}")
    if bool(cfg.get("signed_readout")) is not SIGNED_READOUT:
        errors.append("signed_readout mismatch")
    if bool(cfg.get("readout_bias")) is not READOUT_BIAS:
        errors.append("readout_bias mismatch")
    return errors


def capture_matched_rasters(model: str, train_out: Path, validation_h5: Path) -> None:
    import torch

    snn_tool, _, _, _, _, _ = baseline.import_snn_modules()
    probe_root = train_out / "validation_probe"
    probe_root.mkdir(parents=True, exist_ok=True)
    input_path = probe_root / "matched_input.npz"
    baseline.prepare_matched_input(validation_h5, input_path)
    state = torch.load(train_out / "weights.pth", map_location="cpu")
    final_readout_key = f"W_ff.{len(HIDDEN_SIZES)}"
    state.pop(final_readout_key, None)
    raster_weights = probe_root / "raster_weights.pth"
    torch.save(state, raster_weights)
    raster_out = probe_root / "matched_rasters"
    rc = snn_tool.main(
        [
            "sim", "--load-config", str(train_out / "config.json"),
            "--load-weights", str(raster_weights), "--input-file", str(input_path),
            "--n-in", str(baseline.N_INPUT), "--n-batch", str(len(baseline.RASTER_POSITIONS)),
            "--outputs", "rasters", "--out-dir", str(raster_out), "--wipe-dir",
        ]
    )
    if rc or not (raster_out / "rasters.npz").exists():
        raise RuntimeError("matched raster capture failed")


def configure_baseline() -> None:
    """Point exp069's validated machinery at the selected exp071 candidate."""
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
    baseline.__dict__["EPOCHS"] = EPOCHS
    baseline.N_HIDDEN = HIDDEN_SIZES[-1]
    baseline.N_INHIBITORY = HIDDEN_SIZES[-1] // 4
    baseline.READOUT_MODE = READOUT_MODE
    baseline.READOUT_SCALE = READOUT_SCALE
    baseline.train_args = train_args
    baseline.validate_training = validate_training
    baseline.capture_matched_rasters = capture_matched_rasters
    baseline.SCALE = {
        **baseline.SCALE,
        "experiment": SLUG,
        "attempt": ATTEMPT,
        "stage": STAGE,
        "epochs": EPOCHS,
        "hidden_sizes": HIDDEN_SIZES,
        "n_hidden": HIDDEN_SIZES[-1],
        "n_inhibitory": HIDDEN_SIZES[-1] // 4,
        "readout": READOUT_MODE,
        "signed_readout": SIGNED_READOUT,
        "readout_bias": READOUT_BIAS,
        "readout_scale": READOUT_SCALE,
    }
    _, _, snn_train, _, _, _ = baseline.import_snn_modules()
    snn_datasets = sys.modules[snn_train.load_dataset.__module__]
    setattr(snn_datasets, "_SHD_DIR", str(INSTALLED_SPLIT_ROOT))


configure_baseline()


def load_metrics(model: str) -> dict[str, Any]:
    return json.loads((CELL_ROOT / model / "metrics.json").read_text())


def run_smoke() -> None:
    baseline.run_smoke()
    summary_path = SMOKE_ROOT / "smoke_summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(
        experiment=SLUG,
        attempt=ATTEMPT,
        stage="smoke",
        hidden_sizes=HIDDEN_SIZES,
        readout=READOUT_MODE,
        signed_readout=SIGNED_READOUT,
        readout_bias=READOUT_BIAS,
        readout_scale=READOUT_SCALE,
    )
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
        label=f"exp071-{POD_LABEL}",
    )


def run_via_runpod(meta: Any) -> None:
    runpod.dispatch(
        slug=SLUG,
        runner=SLUG,
        buckets=[
            {"name": f"exp071-{POD_LABEL}-coba", "cells": ["coba"]},
            {"name": f"exp071-{POD_LABEL}-ping", "cells": ["ping"]},
        ],
        gpu=meta.gpu,
        live=meta.live,
        collect=meta.collect,
        plumbing=False,
        collect_subdir=f"{runpod.ARTIFACTS_SUBDIR}/{SLUG}",
        local_collect_dir=str(SCRATCH),
        extra_env={
            "PINGLAB_ARTIFACTS_ROOT": f"{runpod.VOLUME_MOUNT}/{runpod.ARTIFACTS_SUBDIR}/{SLUG}",
            "EXP071_ATTEMPT": ATTEMPT,
            "EXP071_STAGE": STAGE,
        },
        max_runtime_s=7200 if STAGE == "final40" else 3600,
    )


def attempt_diagnostics(cells: dict[str, Any]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for model in baseline.MODELS:
        training = cells[model]["training"]
        selection = cells[model]["selection"]
        selected_epoch = int(selection["epoch"])
        selected_record = training["epochs"][selected_epoch - 1]
        final = training["epochs"][-1]
        finite_keys = ("loss", "test_loss", "acc", "grad_norm", "test_rate_e")
        finite = all(math.isfinite(float(final.get(key, math.nan))) for key in finite_keys)
        if model == "ping":
            finite = finite and math.isfinite(float(final.get("test_rate_i", math.nan)))
        saturation_hz = 0.95 * (1000.0 / baseline.DT_MS)
        active = 0.0 < float(final["test_rate_e"]) < saturation_hz
        if model == "ping":
            active = active and 0.0 < float(final.get("test_rate_i") or 0.0) < saturation_hz
        clean = (
            finite
            and active
            and sum(int(epoch.get("skipped_steps", 0)) for epoch in training["epochs"]) == 0
            and sum(int(epoch.get("nan_forward_batches", 0)) for epoch in training["epochs"]) == 0
        )
        diagnostics[model] = {
            "selected_epoch": selected_epoch,
            "selected_validation_accuracy_pct": float(selection["accuracy_pct"]),
            "selected_validation_cross_entropy": float(selection["cross_entropy"]),
            "selected_validation_e_rate_hz": float(selected_record.get("test_rate_e") or 0.0),
            "selected_validation_i_rate_hz": float(selected_record.get("test_rate_i") or 0.0),
            "final_validation_accuracy_pct": float(final["acc"]),
            "final_validation_cross_entropy": float(final["test_loss"]),
            "final_validation_e_rate_hz": float(final["test_rate_e"]),
            "final_validation_i_rate_hz": float(final.get("test_rate_i") or 0.0),
            "gradient_norm": float(final["grad_norm"]),
            "finite": finite,
            "active": active,
            "clean": clean,
            "training_elapsed_s": training.get("total_elapsed_s"),
            "training_peak_gpu_memory_bytes": training.get("perf", {}).get("peak_memory_bytes"),
        }
    return diagnostics


def publish_pre_result() -> None:
    run_id = next_run_id(SLUG)
    t0 = time.monotonic()
    smoke_path = SMOKE_ROOT / "smoke_summary.json"
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    with published_run(SLUG, run_id, make_artifacts=True, scale=baseline.SCALE,
                       skip_training=True) as (_, figures):
        raw = figures / "raw"
        raw.mkdir()
        if smoke_path.exists():
            smoke_raw = raw / "smoke" / ATTEMPT
            smoke_raw.mkdir(parents=True)
            shutil.copy2(smoke_path, smoke_raw / "smoke_summary.json")
        (figures / "goal.txt").write_text(GOAL_PROMPT + "\n")
        payload = {
            "result_status": "preregistered",
            "stage": "local_smoke_passed" if smoke and smoke.get("passed") else "pre_result_scaffold",
            "seed": baseline.SEED,
            "attempt_order": list(ATTEMPT_SPECS),
            "attempts": {
                name: {
                    "description": spec["description"],
                    "hidden_sizes": spec["hidden_sizes"],
                    "readout": READOUT_MODE,
                    "signed_readout": SIGNED_READOUT,
                    "readout_bias": READOUT_BIAS,
                }
                for name, spec in ATTEMPT_SPECS.items()
            },
            "screening_epochs": SHORT_EPOCHS,
            "promotion_epochs": FINAL_EPOCHS,
            "promotion_accuracy_gain_pp": PROMOTION_ACCURACY_GAIN_PP,
            "split": {
                "development_train_count": 7340,
                "validation_count": 816,
                "train_indices_sha256": baseline.EXP068_TRAIN_HASH,
                "validation_indices_sha256": baseline.EXP068_VALIDATION_HASH,
                "official_test_access": "forbidden and absent from runner",
            },
            "config": baseline.SCALE,
            "smoke": smoke,
            "runpod": {
                "total_spend_usd": 0.0,
                "active_pods_after_collection": 0,
                "paid_compute_started": False,
            },
            "activity": {
                "latest_checkpoint_id": "cp002" if smoke and smoke.get("passed") else "cp001",
                "latest_checkpoint_time_utc": utc_now(),
                "publishable_log": (
                    "artifacts/data/exp071/activity/messages_cp002.json"
                    if smoke and smoke.get("passed")
                    else "artifacts/data/exp071/activity/messages_cp001.json"
                ),
            },
        }
        baseline.atomic_json(raw / "pre_result_status.json", payload)
        write_numbers(figures, run_id=run_id, duration_s=time.monotonic() - t0, payload=payload)
        reproducer = figures / "reproduce.sh"
        reproducer.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "uv run python experiments/exp071.py --plot-only\n"
            "EXP071_ATTEMPT=cumulative_baseline uv run python experiments/exp071.py\n"
            "EXP071_ATTEMPT=cumulative_baseline EXP071_STAGE=short uv run python experiments/exp071.py --runpod\n"
            "# Add --live only with fresh, explicit RunPod spending authority.\n"
            "EXP071_ATTEMPT=cumulative_baseline EXP071_STAGE=short uv run python experiments/exp071.py --runpod --live\n"
            "EXP071_ATTEMPT=cumulative_baseline EXP071_STAGE=short uv run python experiments/exp071.py --runpod --collect\n"
            "EXP071_ATTEMPT=cumulative_baseline EXP071_STAGE=short uv run python experiments/exp071.py --skip-training\n"
        )
        reproducer.chmod(0o755)


def publish_attempt() -> None:
    if not COMPUTE_LEDGER.exists():
        raise SystemExit("missing compute_ledger.json with exact observed RunPod spend")
    compute_ledger = json.loads(COMPUTE_LEDGER.read_text())
    if int(compute_ledger.get("active_pods_after_collection", -1)) != 0:
        raise SystemExit("compute ledger does not confirm zero active pods")
    if not math.isfinite(float(compute_ledger.get("total_spend_usd", math.nan))):
        raise SystemExit("compute ledger does not contain finite total_spend_usd")
    cells = baseline.validate_collected()
    diagnostics = attempt_diagnostics(cells)
    smoke_path = SMOKE_ROOT / "smoke_summary.json"
    if not smoke_path.exists():
        smoke_path = COMMITTED_SMOKE
    run_id = next_run_id(SLUG)
    t0 = time.monotonic()
    with published_run(SLUG, run_id, make_artifacts=True, scale=baseline.SCALE,
                       skip_training=True) as (_, figures):
        baseline.plot_validation_curves(cells, figures / f"{STAGE}_{ATTEMPT}_validation_curves")
        baseline.plot_activity_curves(cells, figures / f"{STAGE}_{ATTEMPT}_activity_curves")
        baseline.plot_matched_rasters(figures / f"{STAGE}_{ATTEMPT}_matched_rasters.png")
        raw = figures / "raw" / STAGE / ATTEMPT
        raw.mkdir(parents=True)
        if smoke_path.exists():
            shutil.copy2(smoke_path, raw / "smoke_summary.json")
        shutil.copy2(COMPUTE_LEDGER, raw / "compute_ledger.json")
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
            "result_status": "done",
            "stage": STAGE,
            "attempt": ATTEMPT,
            "attempt_order": list(ATTEMPT_SPECS),
            "seed": baseline.SEED,
            "cells": diagnostics,
            "split": {
                "development_train_count": 7340,
                "validation_count": 816,
                "train_indices_sha256": baseline.EXP068_TRAIN_HASH,
                "validation_indices_sha256": baseline.EXP068_VALIDATION_HASH,
            },
            "config": baseline.SCALE,
            "integrity": {
                "official_test_access": "forbidden and absent from runner",
                "matched_partitions": True,
                "matched_raster_inputs": True,
                "selection_rule": cells["coba"]["selection"]["rule"],
            },
            "runpod": compute_ledger,
        }
        baseline.atomic_json(raw / "attempt_decision.json", payload)
        write_numbers(figures, run_id=run_id, duration_s=time.monotonic() - t0, payload=payload)


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
    if meta.plot_only:
        publish_pre_result()
        return
    if meta.skip_training:
        publish_attempt()
        return
    run_smoke()


if __name__ == "__main__":
    main()
