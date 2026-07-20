"""Exploratory exp073 SHD run with nonzero trainable recurrent excitation.

This runner reuses exp069's validation-only SHD split, checkpoint selection,
diagnostics, matched rasters, and RunPod plumbing.  The official SHD test has no
route here.  The default command runs a local 128/128 two-epoch smoke for the
selected registered candidate; cloud execution still requires the explicit
``--runpod --live`` spending gate.

Select the registered candidate with ``EXP073_ATTEMPT``:

* ``plastic_wee``: exp071 architecture/readout with nonzero Dale-constrained
  trainable E→E recurrence.

Select ``EXP073_STAGE=short`` for the 8-epoch screening run and
``EXP073_STAGE=final40`` for the promoted 40-epoch run.  The runner fails closed
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

SLUG = "exp073"
SHORT_EPOCHS = 8
FINAL_EPOCHS = 40
READOUT_MODE = "cumulative-potential"
SIGNED_READOUT = True
READOUT_BIAS = True
READOUT_SCALE = 1.0
PROMOTION_ACCURACY_GAIN_PP = 3.0
W_EE_INIT = (0.0003, 0.0001)
BASELINE_ATTEMPT = "plastic_wee"
ATTEMPT_SPECS: dict[str, dict[str, Any]] = {
    "plastic_wee": {
        "hidden_sizes": [256],
        "description": (
            "exp071 architecture/readout with nonzero Dale-constrained "
            "trainable E→E recurrence"
        ),
        "pod_label": "plastic-wee",
        "flags": ["--trainable-w-ee"],
        "w_ee": W_EE_INIT,
    },
}
STAGE_SPECS = {
    "short": {"epochs": SHORT_EPOCHS, "pod_suffix": "short"},
    "final40": {"epochs": FINAL_EPOCHS, "pod_suffix": "final40"},
}
ATTEMPT = os.environ.get("EXP073_ATTEMPT", BASELINE_ATTEMPT)
STAGE = os.environ.get("EXP073_STAGE", "short")
PING_ONLY = os.environ.get("EXP073_PING_ONLY") == "1"
if ATTEMPT not in ATTEMPT_SPECS:
    raise RuntimeError(f"unregistered exp073 attempt: {ATTEMPT}")
if STAGE not in STAGE_SPECS:
    raise RuntimeError(f"unregistered exp073 stage: {STAGE}")
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

GOAL_PROMPT = """/goal Design and execute a new exploratory SHD experiment on a new branch/PR, testing whether nonzero Dale-constrained trainable recurrent excitation moves the matched COBA/PING models beyond the current ~70% regime.

Use current main as the base. Build exp073 around the best surviving SHD recipe from exp071/exp072, especially the cumulative-potential readout. Do not rerun the old baseline unless needed for debugging; use exp071/exp072 as the historical baseline.

Experiment:
- run matched COBA and PING cells with nonzero W_EE initialization and --trainable-w-ee;
- keep W_in and W_out trainable as in the current training path;
- keep Dale signs enforced; do not use free signed recurrence;
- start with a modest nonzero W_EE initialization, e.g. --w-ee 0.03 0.01, unless local inspection of current parameter scales suggests a safer nearby value;
- keep input preprocessing, dataset split, readout, optimizer, batch size, and evaluation protocol matched across COBA and PING;
- keep the comparison exploratory and single-seed unless I explicitly ask for replication.

Protocol:
- implement through experiments/exp073.py and existing tools/snn CLI capabilities if possible;
- do not edit tools/snn unless the current CLI cannot express nonzero trainable W_EE cleanly; if a tools/snn change is required, stop and explain exactly why before editing;
- run a local smoke/scout stage first with few epochs/samples to confirm finite loss, active spiking, and W_EE actually changes during training;
- if both cells are finite and active, run a short RunPod exploratory stage, fewer than 40 epochs for iteration speed;
- if accuracy looks meaningfully better than the exp071/exp072 historical ~70% regime without pathological firing or NaNs, run a 40-epoch confirmation for the promising cell(s);
- COBA and PING may run in parallel on separate RunPod pods when useful;
- RunPod spending is authorized up to $40 total; reap all pods when finished.

Deliverables:
- artifacts/data/exp073 with provenance, numbers.json, reproducer, training curves, firing-rate diagnostics, W_EE diagnostics, and matched rasters where relevant;
- writings/exp073.typ as a cold-readable Demolab experiment report comparing exp073 to the historical exp071/exp072 baseline;
- update the Spiking Heidelberg Digits collection so exp073 is the next canonical experiment;
- build successfully;
- commit focused implementation/results changes, push the branch, and open a new PR;
- do not merge unless I explicitly ask.

End at the review gate with:
- best/selected COBA and PING accuracies;
- whether trainable nonzero W_EE appears to move us beyond the ~70% regime;
- W_EE diagnostics showing it was nonzero and trained;
- firing-rate/pathology summary;
- exact compute spend;
- validation performed;
- PR link and rendered exp073 link."""


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
        "--w-ee", str(W_EE_INIT[0]), str(W_EE_INIT[1]),
        "--readout", READOUT_MODE,
        "--readout-w-out-scale", str(READOUT_SCALE),
        "--out-dir", str(out), "--wipe-dir",
    ]
    if SIGNED_READOUT:
        args.append("--signed-readout")
    if READOUT_BIAS:
        args.append("--readout-bias")
    args.extend(str(flag) for flag in ATTEMPT_SPEC["flags"])
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
    if bool(full_cfg.get("dales_law")) is not True:
        errors.append("dales_law must remain enabled")
    if bool(full_cfg.get("trainable_w_ee")) is not True:
        errors.append("trainable_w_ee must be enabled")
    if any(bool(full_cfg.get(key)) for key in ("trainable_w_ei", "trainable_w_ie", "trainable_w_ii")):
        errors.append("only W_EE may be trainable among recurrent E/I blocks")
    if full_cfg.get("w_ee") != list(W_EE_INIT):
        errors.append(f"w_ee: got {full_cfg.get('w_ee')!r}, want {list(W_EE_INIT)!r}")
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
    """Point exp069's validated machinery at the selected exp073 candidate."""
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
    baseline.__dict__["N_HIDDEN"] = HIDDEN_SIZES[-1]
    baseline.__dict__["N_INHIBITORY"] = HIDDEN_SIZES[-1] // 4
    baseline.__dict__["READOUT_MODE"] = READOUT_MODE
    baseline.__dict__["READOUT_SCALE"] = READOUT_SCALE
    baseline.__dict__["train_args"] = train_args
    baseline.__dict__["validate_training"] = validate_training
    baseline.__dict__["capture_matched_rasters"] = capture_matched_rasters
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
        "candidate_flags": list(ATTEMPT_SPEC["flags"]),
        "w_ee_init": list(W_EE_INIT),
        "trainable_w_ee": True,
        "dales_law": True,
    }
    _, _, snn_train, _, _, _ = baseline.import_snn_modules()
    snn_datasets = sys.modules[snn_train.load_dataset.__module__]
    setattr(snn_datasets, "_SHD_DIR", str(INSTALLED_SPLIT_ROOT))


configure_baseline()


def load_metrics(model: str) -> dict[str, Any]:
    return json.loads((CELL_ROOT / model / "metrics.json").read_text())


def run_smoke() -> None:
    error: BaseException | None = None
    try:
        baseline.run_smoke()
    except BaseException as exc:
        error = exc
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
        flags=list(ATTEMPT_SPEC["flags"]),
        w_ee_init=list(W_EE_INIT),
        trainable_w_ee=True,
        w_ee_diagnostics={
            model: parameter_diagnostics(SMOKE_ROOT / model)
            for model in baseline.MODELS
            if (SMOKE_ROOT / model / "weights.pth").exists()
        },
    )
    baseline.atomic_json(summary_path, summary)
    if error is not None:
        raise error


def run_full_cell(model: str) -> None:
    baseline.run_full_cell(model)


def cell_done(model: str) -> bool:
    return baseline.cell_done(model)


def pod_run() -> None:
    runpod.pod_run_loop(
        job_ids=list(baseline.MODELS),
        is_done=cell_done,
        run_job=run_full_cell,
        label=f"exp073-{POD_LABEL}",
    )


def run_via_runpod(meta: Any) -> None:
    cells = meta.only_cells or list(baseline.MODELS)
    unknown = sorted(set(cells) - set(baseline.MODELS))
    if unknown:
        raise SystemExit(f"unknown exp073 cells for --only-cells: {unknown}")
    buckets = [
        {"name": f"exp073-{POD_LABEL}-{model}", "cells": [model]}
        for model in cells
    ]
    runpod.dispatch(
        slug=SLUG,
        runner=SLUG,
        buckets=buckets,
        gpu=meta.gpu,
        live=meta.live,
        collect=meta.collect,
        plumbing=False,
        collect_subdir=f"{runpod.ARTIFACTS_SUBDIR}/{SLUG}",
        local_collect_dir=str(SCRATCH),
        extra_env={
            "PINGLAB_ARTIFACTS_ROOT": f"{runpod.VOLUME_MOUNT}/{runpod.ARTIFACTS_SUBDIR}/{SLUG}",
            "EXP073_ATTEMPT": ATTEMPT,
            "EXP073_STAGE": STAGE,
            "EXP073_PING_ONLY": "1" if PING_ONLY else "0",
        },
        max_runtime_s=14400 if STAGE == "final80" else (7200 if STAGE == "final40" else 3600),
    )


def attempt_diagnostics(cells: dict[str, Any]) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    for model in cells:
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


def validate_collected_subset(models: tuple[str, ...]) -> dict[str, Any]:
    errors: list[str] = []
    cells: dict[str, Any] = {}
    for model in models:
        model_errors: list[str] = []
        out = CELL_ROOT / model
        frozen_path = FROZEN_ROOT / f"{model}.json"
        if not frozen_path.exists():
            errors.append(f"{model}: missing frozen record")
            continue
        frozen = baseline.load_json(frozen_path)
        model_errors.extend(f"{model}: {error}" for error in validate_training(model, out, smoke=False))
        for relative in (
            "complete.json",
            "validation_probe/matched_input.npz",
            "validation_probe/matched_rasters/rasters.npz",
        ):
            if not (out / relative).exists():
                model_errors.append(f"{model}: missing {relative}")
        if frozen.get("split_train_indices_sha256") != baseline.EXP068_TRAIN_HASH:
            model_errors.append(f"{model}: development-training indices differ from exp068")
        if frozen.get("split_validation_indices_sha256") != baseline.EXP068_VALIDATION_HASH:
            model_errors.append(f"{model}: validation indices differ from exp068")
        if model_errors:
            errors.extend(model_errors)
        else:
            cells[model] = {
                "training": baseline.load_json(out / "metrics.json"),
                "selection": baseline.load_json(out / "checkpoint_selection.json"),
                "frozen": frozen,
            }
    if errors:
        raise SystemExit("collected exp073 subset failed integrity checks:\n" + "\n".join(errors))
    return cells


def plot_ping_only_validation_curves(cells: dict[str, Any], stem: Path) -> None:
    import matplotlib.pyplot as plt
    from helpers.figsave import save_figure
    from helpers import theme

    theme.apply()
    epochs = cells["ping"]["training"]["epochs"]
    selected = cells["ping"]["selection"]
    x = [epoch["ep"] for epoch in epochs]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25))
    axes[0].plot(x, [epoch["loss"] for epoch in epochs], color=theme.INK_BLACK, label="PING train")
    axes[0].plot(x, [epoch["test_loss"] for epoch in epochs], color=theme.INK_BLACK,
                 linestyle=":", label="PING validation")
    axes[1].plot(x, [epoch["acc"] for epoch in epochs], color=theme.INK_BLACK, marker="D",
                 markevery=5, label="PING")
    axes[1].scatter([selected["epoch"]], [selected["accuracy_pct"]],
                    color=theme.INK_BLACK, marker="D", s=32, zorder=4)
    axes[0].set(xlabel="epoch", ylabel="cross-entropy loss")
    axes[1].set(xlabel="epoch", ylabel="validation accuracy (%)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def plot_ping_only_activity_curves(cells: dict[str, Any], stem: Path) -> None:
    import matplotlib.pyplot as plt
    from helpers.figsave import save_figure
    from helpers import theme

    theme.apply()
    epochs = cells["ping"]["training"]["epochs"]
    x = [epoch["ep"] for epoch in epochs]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25), sharex=True)
    axes[0].plot(x, [epoch.get("test_rate_e", 0.0) for epoch in epochs],
                 color=theme.INK_BLACK, label="PING E")
    axes[1].plot(x, [epoch.get("test_rate_i", 0.0) or 0.0 for epoch in epochs],
                 color=theme.INK_BLACK, label="PING I")
    axes[0].set(xlabel="epoch", ylabel="validation E rate (Hz)")
    axes[1].set(xlabel="epoch", ylabel="validation I rate (Hz)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def plot_ping_only_matched_rasters(stem: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from helpers import theme

    theme.apply()
    input_path = CELL_ROOT / "ping" / "validation_probe" / "matched_input.npz"
    input_data = np.load(input_path)
    input_spikes = input_data["input_spikes"]
    data = np.load(CELL_ROOT / "ping" / "validation_probe" / "matched_rasters" / "rasters.npz")
    fig, axes = plt.subplots(3, len(baseline.RASTER_POSITIONS), figsize=(6.5, 3.2), sharex=True)
    e = np.zeros((int(data["T"]), int(data["n_e"])), dtype=np.uint8)
    i = np.zeros((int(data["T"]), int(data["n_i"])), dtype=np.uint8)
    for col, position in enumerate(baseline.RASTER_POSITIONS):
        e.fill(0)
        i.fill(0)
        emask = data["e_trial"] == col
        imask = data["i_trial"] == col
        e[data["e_t"][emask], data["e_cell"][emask]] = 1
        i[data["i_t"][imask], data["i_cell"][imask]] = 1
        for row, (array, label) in enumerate(zip((input_spikes[:, col, :], e, i), ("input", "E", "I"))):
            axis = axes[row, col]
            timesteps, units = np.nonzero(array)
            axis.scatter(timesteps * float(input_data["dt"]), units, s=0.25,
                         color=theme.INK_BLACK, linewidths=0, rasterized=True)
            if col == 0:
                axis.set_ylabel(f"PING {label}", fontsize=7)
            if row == 0:
                speaker = int(input_data["speakers"][col])
                axis.set_title(f"position {position} · speaker {speaker}", fontsize=7)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=6)
    for axis in axes[-1]:
        axis.set_xlabel("time (ms)")
    fig.tight_layout(h_pad=0.3, w_pad=0.5)
    fig.savefig(stem, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _summary(values: Any) -> dict[str, float]:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    return {"min": float(arr.min()), "mean": float(arr.mean()), "max": float(arr.max())}


def _matrix_delta(current: Any, reference: Any) -> dict[str, float]:
    import numpy as np

    delta = np.asarray(current, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        "mean_abs": float(np.abs(delta).mean()),
        "max_abs": float(np.abs(delta).max()),
        "l2": float(np.sqrt(np.square(delta).sum())),
    }


def reconstructed_initial_w_ee(train_out: Path) -> dict[str, Any]:
    """Reconstruct the registered initial W_EE from config/seed for diagnostics."""
    import torch

    _snn_tool, _snn_runlog, snn_train, snn_config, snn_models, _snn_encoders = (
        baseline.import_snn_modules()
    )
    cfg = json.loads((train_out / "config.json").read_text())
    hidden_sizes = [int(value) for value in cfg["hidden_sizes"]]
    snn_train.seed_everything(int(cfg["seed"]))
    snn_config.set_sim_dt(float(cfg["dt"]), float(cfg["t_ms"]))
    snn_config.setup_model_globals(hidden_sizes)
    snn_models.N_IN = int(cfg["n_in"])
    snn_models.N_OUT = 20
    net = snn_config.build_net(
        cfg["model"],
        w_in=tuple(cfg["w_in"]) if cfg.get("w_in") else None,
        w_in_sparsity=float(cfg["w_in_sparsity"]),
        w_ee=tuple(cfg["w_ee"]) if cfg.get("w_ee") else None,
        ei_strength=cfg.get("ei_strength"),
        ei_ratio=float(cfg["ei_ratio"]),
        device=torch.device("cpu"),
        randomize_init=True,
        dales_law=bool(cfg["dales_law"]),
        hidden_sizes=hidden_sizes,
        readout_mode=cfg["readout_mode"],
        signed_readout=bool(cfg["signed_readout"]),
        readout_bias=bool(cfg["readout_bias"]),
        trainable_w_ee=bool(cfg["trainable_w_ee"]),
        trainable_w_ei=bool(cfg["trainable_w_ei"]),
        trainable_w_ie=bool(cfg["trainable_w_ie"]),
        trainable_w_ii=bool(cfg["trainable_w_ii"]),
        state_clamp=bool(cfg["state_clamp"]),
    )
    return {
        layer: tensor.detach().cpu().numpy()
        for layer, tensor in net.W_ee.items()
    }


def parameter_diagnostics(train_out: Path) -> dict[str, Any]:
    """Summarise nonzero trainable W_EE and its movement during training."""
    import torch

    selected_state = torch.load(train_out / "weights.pth", map_location="cpu")
    final_state = torch.load(train_out / "weights_final.pth", map_location="cpu")
    initial = reconstructed_initial_w_ee(train_out)
    diagnostics: dict[str, Any] = {
        "w_ee_init": list(W_EE_INIT),
        "trainable_w_ee": True,
        "dales_law": True,
        "layers": {},
    }
    for layer in range(1, len(HIDDEN_SIZES) + 1):
        key = str(layer)
        state_key = f"W_ee.{key}"
        selected = selected_state[state_key].detach().cpu().numpy()
        final = final_state[state_key].detach().cpu().numpy()
        init = initial[key]
        selected_delta = _matrix_delta(selected, init)
        final_delta = _matrix_delta(final, init)
        diagnostics["layers"][key] = {
            "initial": _summary(init),
            "selected": _summary(selected),
            "final": _summary(final),
            "selected_delta_from_initial": selected_delta,
            "final_delta_from_initial": final_delta,
            "selected_changed_from_initial": selected_delta["mean_abs"] > 1e-8,
            "final_changed_from_initial": final_delta["mean_abs"] > 1e-8,
            "selected_nonnegative": bool(selected.min() >= 0.0),
            "final_nonnegative": bool(final.min() >= 0.0),
        }
    return diagnostics


def existing_activity_logs() -> dict[str, str]:
    """Return already-published sanitized activity logs before publication rewrites."""
    activity = REPO / "artifacts" / "data" / SLUG / "activity"
    if not activity.exists():
        return {}
    return {
        path.name: path.read_text()
        for path in sorted(activity.glob("messages_cp*.json"))
        if path.is_file()
    }


def restore_activity_logs(figures: Path, logs: dict[str, str]) -> dict[str, Any]:
    """Copy sanitized activity logs into the freshly published artifact tree."""
    activity = figures / "activity"
    activity.mkdir(parents=True, exist_ok=True)
    for name, content in logs.items():
        (activity / name).write_text(content)
    checkpoint_ids = [Path(name).stem.removeprefix("messages_") for name in logs]
    latest = checkpoint_ids[-1] if checkpoint_ids else None
    return {
        "checkpoint_ids": checkpoint_ids,
        "latest_checkpoint_id": latest,
        "latest_checkpoint_time_utc": utc_now(),
        "publishable_log": (
            f"artifacts/data/{SLUG}/activity/messages_{latest}.json"
            if latest
            else None
        ),
    }


def write_reproducer(figures: Path) -> None:
    (figures / "goal.txt").write_text(GOAL_PROMPT + "\n")
    reproducer = figures / "reproduce.sh"
    reproducer.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "uv run python experiments/exp073.py --plot-only\n"
        "EXP073_ATTEMPT=plastic_wee uv run python experiments/exp073.py\n"
        "# The matched local gate failed for COBA. With explicit follow-up authority,\n"
        "# continue the surviving PING cell only by setting EXP073_PING_ONLY=1.\n"
        "EXP073_PING_ONLY=1 EXP073_ATTEMPT=plastic_wee EXP073_STAGE=short uv run python experiments/exp073.py --runpod --only-cells ping\n"
        "EXP073_PING_ONLY=1 EXP073_ATTEMPT=plastic_wee EXP073_STAGE=short uv run python experiments/exp073.py --runpod --only-cells ping --live\n"
        "EXP073_PING_ONLY=1 EXP073_ATTEMPT=plastic_wee EXP073_STAGE=short uv run python experiments/exp073.py --runpod --collect\n"
        "EXP073_PING_ONLY=1 EXP073_ATTEMPT=plastic_wee EXP073_STAGE=short uv run python experiments/exp073.py --skip-training\n"
        "# Historical matched commands retained for the original pre-pivot design:\n"
        "EXP073_ATTEMPT=plastic_wee EXP073_STAGE=short uv run python experiments/exp073.py --runpod\n"
        "EXP073_ATTEMPT=plastic_wee EXP073_STAGE=short uv run python experiments/exp073.py --runpod --collect\n"
        "EXP073_ATTEMPT=plastic_wee EXP073_STAGE=short uv run python experiments/exp073.py --skip-training\n"
        "# Promote only if the short run materially improves over the exp071/exp072 ~70% regime.\n"
        "EXP073_ATTEMPT=plastic_wee EXP073_STAGE=final40 uv run python experiments/exp073.py --runpod\n"
        "EXP073_ATTEMPT=plastic_wee EXP073_STAGE=final40 uv run python experiments/exp073.py --runpod --collect\n"
        "EXP073_ATTEMPT=plastic_wee EXP073_STAGE=final40 uv run python experiments/exp073.py --skip-training\n"
    )
    reproducer.chmod(0o755)


def archived_killed_scouts() -> dict[str, Any]:
    scouts_root = REPO / "temp" / "experiments" / SLUG / "killed_scouts"
    if not scouts_root.exists():
        return {}
    scouts: dict[str, Any] = {}
    for scout in sorted(path for path in scouts_root.iterdir() if path.is_dir()):
        payload: dict[str, Any] = {}
        summary_path = scout / "smoke_summary.json"
        if summary_path.exists():
            payload["smoke_summary"] = json.loads(summary_path.read_text())
        for model in baseline.MODELS:
            metrics_path = scout / f"{model}_metrics.json"
            if not metrics_path.exists():
                continue
            metrics = json.loads(metrics_path.read_text())
            payload[f"{model}_metrics_summary"] = {
                "w_ee": metrics.get("config", {}).get("w_ee"),
                "trainable_w_ee": metrics.get("config", {}).get("trainable_w_ee"),
                "epochs": [
                    {
                        "ep": epoch.get("ep"),
                        "loss": _finite_or_none(epoch.get("loss")),
                        "test_loss": _finite_or_none(epoch.get("test_loss")),
                        "acc": _finite_or_none(epoch.get("acc")),
                        "test_rate_e": _finite_or_none(epoch.get("test_rate_e")),
                        "test_rate_i": _finite_or_none(epoch.get("test_rate_i")),
                        "skipped_steps": epoch.get("skipped_steps"),
                        "nan_forward_batches": epoch.get("nan_forward_batches"),
                        "grad_norm": _finite_or_none(epoch.get("grad_norm")),
                    }
                    for epoch in metrics.get("epochs", [])
                ],
            }
        scouts[scout.name] = payload
    return scouts


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    return _finite_or_none(value)


def copy_json_sanitized(source: Path, destination: Path) -> None:
    payload = json.loads(source.read_text())
    baseline.atomic_json(destination, _json_sanitize(payload))


def copy_jsonl_sanitized(source: Path, destination: Path) -> None:
    with destination.open("w") as handle:
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            handle.write(json.dumps(_json_sanitize(payload), sort_keys=True) + "\n")


def copy_killed_scouts(raw_root: Path) -> None:
    scouts_root = REPO / "temp" / "experiments" / SLUG / "killed_scouts"
    if not scouts_root.exists():
        return
    destination_root = raw_root / "killed_scouts"
    for scout in sorted(path for path in scouts_root.iterdir() if path.is_dir()):
        destination = destination_root / scout.name
        destination.mkdir(parents=True, exist_ok=True)
        for source in sorted(scout.glob("*.json")):
            copy_json_sanitized(source, destination / source.name)


def copy_smoke_cell_artifacts(raw_root: Path) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    smoke_raw = raw_root / "smoke" / ATTEMPT
    for model in baseline.MODELS:
        source = SMOKE_ROOT / model
        if not source.exists():
            continue
        destination = smoke_raw / model
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "metrics.json", "checkpoint_selection.json"):
            candidate = source / name
            if candidate.exists():
                copy_json_sanitized(candidate, destination / name)
        metrics_jsonl = source / "metrics.jsonl"
        if metrics_jsonl.exists():
            copy_jsonl_sanitized(metrics_jsonl, destination / "metrics.jsonl")
        if (source / "weights.pth").exists() and (source / "weights_final.pth").exists():
            diag = parameter_diagnostics(source)
            diagnostics[model] = diag
            baseline.atomic_json(destination / "w_ee_diagnostics.json", diag)
    return diagnostics


def collected_cell_summary(stage: str, attempt: str, model: str) -> dict[str, Any] | None:
    cell = SCRATCH / "cells" / stage / attempt / model
    selection = cell / "checkpoint_selection.json"
    run_log = cell / "run.jsonl"
    if not selection.exists() or not run_log.exists():
        return None
    selected = json.loads(selection.read_text())
    summary: dict[str, Any] = {}
    for line in run_log.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "summary":
            summary = event
    return {
        "selected_epoch": selected.get("epoch"),
        "selected_validation_accuracy_pct": selected.get("accuracy_pct"),
        "selected_validation_cross_entropy": selected.get("cross_entropy"),
        "final_validation_accuracy_pct": summary.get("final_acc"),
        "training_elapsed_s": summary.get("runtime_s"),
        "dynamics": summary.get("dynamics"),
    }


def collected_ladder_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for stage in STAGE_SPECS:
        stage_payload: dict[str, Any] = {}
        for attempt in ATTEMPT_SPECS:
            cells = {
                model: collected_cell_summary(stage, attempt, model)
                for model in baseline.MODELS
            }
            if any(value is not None for value in cells.values()):
                stage_payload[attempt] = {"cells": cells}
        if stage_payload:
            summary[stage] = stage_payload
    return summary


def collected_compute_summary() -> dict[str, Any]:
    ledgers: dict[str, Any] = {}
    total = 0.0
    exact = True
    for path in sorted((SCRATCH / "compute_ledgers").glob("*.json")):
        if path.name == "provider_billing_reconciliation.json":
            continue
        payload = json.loads(path.read_text())
        ledgers[path.stem] = {
            "total_spend_usd": payload.get("total_spend_usd"),
            "exact_provider_billing": bool(payload.get("exact_provider_billing", False)),
            "billing_status": payload.get("billing_status"),
        }
        total += float(payload.get("total_spend_usd", 0.0))
        exact = exact and bool(payload.get("exact_provider_billing", False))
    summary = {
        "total_spend_usd": round(total, 3),
        "exact_provider_billing": exact,
        "status": "exact" if exact else "timestamp_estimate_pending_provider_reconciliation",
        "ledgers": ledgers,
    }
    reconciliation = SCRATCH / "compute_ledgers" / "provider_billing_reconciliation.json"
    if reconciliation.exists():
        payload = json.loads(reconciliation.read_text())
        summary["total_spend_usd"] = float(payload["total_spend_usd"])
        summary["exact_provider_billing"] = bool(payload.get("exact_provider_billing", True))
        summary["status"] = payload.get("billing_status", "provider_reconciled")
        summary["provider_reconciliation"] = payload
    return summary


def archive_collected_attempts(raw_root: Path) -> None:
    baseline.atomic_json(raw_root / "collected_ladder_summary.json", collected_ladder_summary())
    reconciliation = SCRATCH / "compute_ledgers" / "provider_billing_reconciliation.json"
    if reconciliation.exists():
        shutil.copy2(reconciliation, raw_root / "provider_billing_reconciliation.json")
    for attempt in ATTEMPT_SPECS:
        smoke = REPO / "temp" / "experiments" / SLUG / "smoke" / attempt / "smoke_summary.json"
        if smoke.exists():
            destination = raw_root / "smoke" / attempt
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(smoke, destination / "smoke_summary.json")
    for stage in STAGE_SPECS:
        for attempt in ATTEMPT_SPECS:
            ledger = SCRATCH / "compute_ledgers" / f"{stage}-{attempt}.json"
            if ledger.exists():
                ledger_dest = raw_root / stage / attempt
                ledger_dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ledger, ledger_dest / "compute_ledger.json")
            for model in baseline.MODELS:
                source = SCRATCH / "cells" / stage / attempt / model
                if not source.exists():
                    continue
                destination = raw_root / stage / attempt / model
                destination.mkdir(parents=True, exist_ok=True)
                for name in ("config.json", "metrics.json", "metrics.jsonl", "checkpoint_selection.json"):
                    candidate = source / name
                    if candidate.exists():
                        shutil.copy2(candidate, destination / name)
                probe = source / "validation_probe"
                if (probe / "matched_input.npz").exists():
                    shutil.copy2(probe / "matched_input.npz", destination / "matched_input.npz")
                rasters = probe / "matched_rasters" / "rasters.npz"
                if rasters.exists():
                    shutil.copy2(rasters, destination / "matched_rasters.npz")
                if (source / "weights.pth").exists():
                    baseline.atomic_json(
                        destination / "parameter_diagnostics.json",
                        parameter_diagnostics(source),
                    )


def publish_pre_result() -> None:
    run_id = next_run_id(SLUG)
    t0 = time.monotonic()
    activity_logs = existing_activity_logs()
    smoke_path = SMOKE_ROOT / "smoke_summary.json"
    smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    smoke_ladder: dict[str, Any] = {}
    for name in ATTEMPT_SPECS:
        candidate_smoke = REPO / "temp" / "experiments" / SLUG / "smoke" / name / "smoke_summary.json"
        if candidate_smoke.exists():
            smoke_ladder[name] = json.loads(candidate_smoke.read_text())
    all_smokes_passed = bool(smoke_ladder) and all(
        bool(summary.get("passed")) for summary in smoke_ladder.values()
    )
    with published_run(SLUG, run_id, make_artifacts=True, scale=baseline.SCALE,
                       skip_training=True) as (_, figures):
        activity_payload = restore_activity_logs(figures, activity_logs)
        raw = figures / "raw"
        raw.mkdir()
        for name in ATTEMPT_SPECS:
            candidate_smoke = REPO / "temp" / "experiments" / SLUG / "smoke" / name / "smoke_summary.json"
            if not candidate_smoke.exists():
                continue
            smoke_raw = raw / "smoke" / name
            smoke_raw.mkdir(parents=True)
            shutil.copy2(candidate_smoke, smoke_raw / "smoke_summary.json")
        smoke_parameter_diagnostics = copy_smoke_cell_artifacts(raw)
        copy_killed_scouts(raw)
        write_reproducer(figures)
        payload = {
            "result_status": "preregistered",
            "stage": "local_smoke_ladder_passed" if all_smokes_passed else (
                "local_smoke_passed" if smoke and smoke.get("passed") else (
                    "local_smoke_failed" if smoke else "pre_result_scaffold"
                )
            ),
            "seed": baseline.SEED,
            "attempt_order": list(ATTEMPT_SPECS),
            "attempts": {
                name: {
                    "description": spec["description"],
                    "hidden_sizes": spec["hidden_sizes"],
                    "readout": READOUT_MODE,
                    "signed_readout": SIGNED_READOUT,
                    "readout_bias": READOUT_BIAS,
                    "flags": spec["flags"],
                    "w_ee": list(spec["w_ee"]),
                }
                for name, spec in ATTEMPT_SPECS.items()
            },
            "screening_epochs": SHORT_EPOCHS,
            "promotion_epochs": FINAL_EPOCHS,
            "promotion_accuracy_gain_pp": PROMOTION_ACCURACY_GAIN_PP,
            "historical_baseline": {
                "source": "exp071/exp072 validation-only one-seed runs",
                "accuracy_regime_pct": 70.0,
            },
            "split": {
                "development_train_count": 7340,
                "validation_count": 816,
                "train_indices_sha256": baseline.EXP068_TRAIN_HASH,
                "validation_indices_sha256": baseline.EXP068_VALIDATION_HASH,
                "official_test_access": "forbidden and absent from runner",
            },
            "config": baseline.SCALE,
            "smoke": smoke,
            "smoke_ladder": smoke_ladder,
            "smoke_w_ee_diagnostics": smoke_parameter_diagnostics,
            "killed_scouts": archived_killed_scouts(),
            "runpod": {
                "total_spend_usd": 0.0,
                "active_pods_after_collection": 0,
                "paid_compute_started": False,
            },
            "activity": activity_payload,
        }
        baseline.atomic_json(raw / "pre_result_status.json", payload)
        write_numbers(figures, run_id=run_id, duration_s=time.monotonic() - t0, payload=payload)

def publish_attempt() -> None:
    if not COMPUTE_LEDGER.exists():
        raise SystemExit("missing compute_ledger.json with exact observed RunPod spend")
    compute_ledger = json.loads(COMPUTE_LEDGER.read_text())
    if int(compute_ledger.get("active_pods_after_collection", -1)) != 0:
        raise SystemExit("compute ledger does not confirm zero active pods")
    if not math.isfinite(float(compute_ledger.get("total_spend_usd", math.nan))):
        raise SystemExit("compute ledger does not contain finite total_spend_usd")
    model_scope = ("ping",) if PING_ONLY else tuple(baseline.MODELS)
    cells = validate_collected_subset(model_scope) if PING_ONLY else baseline.validate_collected()
    diagnostics = attempt_diagnostics(cells)
    activity_logs = existing_activity_logs()
    smoke_path = SMOKE_ROOT / "smoke_summary.json"
    if not smoke_path.exists():
        smoke_path = COMMITTED_SMOKE
    run_id = next_run_id(SLUG)
    t0 = time.monotonic()
    with published_run(SLUG, run_id, make_artifacts=True, scale=baseline.SCALE,
                       skip_training=True) as (_, figures):
        activity_payload = restore_activity_logs(figures, activity_logs)
        write_reproducer(figures)
        if PING_ONLY:
            plot_ping_only_validation_curves(cells, figures / f"{STAGE}_{ATTEMPT}_ping_only_validation_curves")
            plot_ping_only_activity_curves(cells, figures / f"{STAGE}_{ATTEMPT}_ping_only_activity_curves")
            plot_ping_only_matched_rasters(figures / f"{STAGE}_{ATTEMPT}_ping_only_matched_rasters.png")
        else:
            baseline.plot_validation_curves(cells, figures / f"{STAGE}_{ATTEMPT}_validation_curves")
            baseline.plot_activity_curves(cells, figures / f"{STAGE}_{ATTEMPT}_activity_curves")
            baseline.plot_matched_rasters(figures / f"{STAGE}_{ATTEMPT}_matched_rasters.png")
        raw = figures / "raw" / STAGE / ATTEMPT
        raw.mkdir(parents=True)
        if smoke_path.exists():
            shutil.copy2(smoke_path, raw / "smoke_summary.json")
        shutil.copy2(COMPUTE_LEDGER, raw / "compute_ledger.json")
        for model in model_scope:
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
            baseline.atomic_json(destination / "parameter_diagnostics.json", parameter_diagnostics(source))
        archive_collected_attempts(figures / "raw")
        ladder_summary = collected_ladder_summary()
        compute_summary = collected_compute_summary()
        payload = {
            "result_status": "done",
            "design_pivot": (
                "ping_only_after_matched_local_gate_failure" if PING_ONLY else "matched_coba_ping"
            ),
            "pivot_authorized": bool(PING_ONLY),
            "cell_scope": list(model_scope),
            "stage": STAGE,
            "attempt": ATTEMPT,
            "attempt_order": list(ATTEMPT_SPECS),
            "screening_epochs": SHORT_EPOCHS,
            "promotion_epochs": FINAL_EPOCHS,
            "seed": baseline.SEED,
            "cells": diagnostics,
            "parameters": {
                model: parameter_diagnostics(CELL_ROOT / model)
                for model in model_scope
            },
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
                "matched_raster_inputs": not PING_ONLY,
                "selection_rule": cells[model_scope[0]]["selection"]["rule"],
                "note": (
                    "PING-only continuation after the matched local gate killed COBA"
                    if PING_ONLY
                    else "matched COBA/PING comparison"
                ),
            },
            "runpod": compute_ledger,
            "compute": compute_summary,
            "ladder": ladder_summary,
            "activity": activity_payload,
            "spend": {
                "total_spend_usd": float(compute_summary["total_spend_usd"]),
                "exact_provider_billing": bool(compute_summary["exact_provider_billing"]),
                "status": compute_summary["status"],
            },
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
