"""Validation-only 80-epoch SHD comparison of matched COBA and PING.

The official SHD training file is deterministically split by speaker and class.
The existing CLI sees development-train as ``train`` and validation as ``test``
during fitting, but this runner contains no route to the official SHD test.
Default execution is the registered local plumbing smoke. Cloud compute needs
the explicit ``--runpod --live`` gate.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import h5py
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SNN = REPO / "tools" / "snn"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import runpod, theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.datasets import _shd_h5  # noqa: E402
from helpers.figsave import save_figure  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp069"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
SCRATCH = runpod.artifacts_scratch(SLUG)
CELL_ROOT = SCRATCH / "cells"
FROZEN_ROOT = SCRATCH / "frozen"
FAILURE_ROOT = SCRATCH / "failures"
SMOKE_ROOT = REPO / "temp" / "experiments" / SLUG / "smoke"
LOCAL_SPLIT_ROOT = REPO / "temp" / "experiments" / SLUG / "split"

MODELS = ("coba", "ping")
SEED = 42
EPOCHS = 80
SMOKE_EPOCHS = 2
SMOKE_SAMPLES = 128
VALIDATION_FRACTION = 0.10
INPUT_SCALE = 0.9
DT_MS = 1.0
T_MS = 1000.0
N_INPUT = 700
N_CLASSES = 20
N_HIDDEN = 256
N_INHIBITORY = 64
BATCH_SIZE = 32
LEARNING_RATE = 0.0004
INPUT_SPARSITY = 0.95
READOUT_SCALE = 225.0
READOUT_MODE = "mem-mean"
RASTER_POSITIONS = (9, 39, 40)  # fixed validation indices from the registered split
EXP068_TRAIN_HASH = "2fee22abaf3629ec1d10179847380b2829968d748fea60607b8f7bb2e3b1de71"
EXP068_VALIDATION_HASH = "5b9f014ae3a4ab48b6586e7d014b8cdbd73b182c4c9cba2b9492ba203cd4ee3a"
PING_BASELINE_PCT = 43.50
PRIMARY_THRESHOLD_PP = 3.0
SHD_DIR = Path("/tmp/shd")

RECIPES: dict[str, dict[str, float]] = {
    "coba": {"ei_strength": 0.0, "v_grad_dampen": 1.0},
    "ping": {"ei_strength": 1.0, "v_grad_dampen": 1000.0},
}
SCALE = {
    "seed": SEED,
    "epochs": EPOCHS,
    "validation_fraction": VALIDATION_FRACTION,
    "models": list(MODELS),
    "n_hidden": N_HIDDEN,
    "n_inhibitory": N_INHIBITORY,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "input_weight_mean": INPUT_SCALE,
    "input_sparsity": INPUT_SPARSITY,
    "readout_scale": READOUT_SCALE,
    "readout": READOUT_MODE,
    "dt_ms": DT_MS,
    "t_ms": T_MS,
    "raster_positions": list(RASTER_POSITIONS),
    "recipes": RECIPES,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def cell_dir(root: Path, model: str) -> Path:
    return root / model


def read_speaker_and_labels(source: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(source, "r") as handle:
        speakers = np.asarray(handle["extra/speaker"], dtype=np.int64)
        labels = np.asarray(handle["labels"], dtype=np.int64)
    return speakers, labels


def development_indices(source: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return the deterministic joint speaker×class stratified split."""
    from sklearn.model_selection import train_test_split

    speakers, labels = read_speaker_and_labels(source)
    indices = np.arange(len(labels), dtype=np.int64)
    strata = np.asarray([f"{s}:{y}" for s, y in zip(speakers, labels)])
    train_idx, val_idx = train_test_split(
        indices,
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
        shuffle=True,
        stratify=strata,
    )
    train_idx = np.sort(train_idx.astype(np.int64))
    val_idx = np.sort(val_idx.astype(np.int64))
    if len(np.intersect1d(train_idx, val_idx)) or len(train_idx) + len(val_idx) != len(indices):
        raise RuntimeError("development split is not a disjoint partition")
    payload = {
        "seed": SEED,
        "method": "sklearn.train_test_split stratified by speaker:class",
        "validation_fraction": VALIDATION_FRACTION,
        "official_train_count": int(len(indices)),
        "development_train_count": int(len(train_idx)),
        "validation_count": int(len(val_idx)),
        "official_train_sha256": sha256_file(source),
        "train_indices_sha256": hashlib.sha256(train_idx.tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(val_idx.tobytes()).hexdigest(),
        "train_indices": train_idx.tolist(),
        "validation_indices": val_idx.tolist(),
    }
    return train_idx, val_idx, payload


def write_h5_subset(source: Path, destination: Path, indices: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(".tmp.h5")
    if tmp.exists():
        tmp.unlink()
    with h5py.File(source, "r") as src, h5py.File(tmp, "w") as dst:
        spikes = dst.create_group("spikes")
        # HDF5 fancy indexing accepts the sorted indices and performs the copy
        # in C.  Assigning thousands of vlen rows in Python adds minutes of
        # staging latency on each pod without changing a byte of science.
        times = src["spikes/times"][indices]
        units = src["spikes/units"][indices]
        spikes.create_dataset("times", data=times, dtype=src["spikes/times"].dtype)
        spikes.create_dataset("units", data=units, dtype=src["spikes/units"].dtype)
        dst.create_dataset("labels", data=np.asarray(src["labels"])[indices])
        extra = dst.create_group("extra")
        extra.create_dataset("speaker", data=np.asarray(src["extra/speaker"])[indices])
        extra.create_dataset("keys", data=np.asarray(src["extra/keys"]))
        src.copy("extra/meta_info", extra)
    tmp.replace(destination)


def prepare_staged_split(root: Path, *, smoke: bool) -> tuple[Path, Path, dict[str, Any]]:
    source = Path(_shd_h5("train"))
    train_idx, val_idx, provenance = development_indices(source)
    if smoke:
        train_idx = train_idx[:SMOKE_SAMPLES]
        val_idx = val_idx[:SMOKE_SAMPLES]
    train_h5 = root / "shd_train.h5"
    val_h5 = root / "shd_test.h5"
    write_h5_subset(source, train_h5, train_idx)
    write_h5_subset(source, val_h5, val_idx)
    provenance = {
        **provenance,
        "stage": "smoke" if smoke else "full",
        "staged_train_count": int(len(train_idx)),
        "staged_validation_count": int(len(val_idx)),
        "staged_train_sha256": sha256_file(train_h5),
        "staged_validation_sha256": sha256_file(val_h5),
    }
    atomic_json(root / "split_provenance.json", provenance)
    return train_h5, val_h5, provenance


@contextlib.contextmanager
def installed_split(train_h5: Path, val_h5: Path) -> Iterator[None]:
    """Temporarily expose staged development data at the CLI's fixed paths."""
    SHD_DIR.mkdir(parents=True, exist_ok=True)
    backup = train_h5.parent / "local_preexisting_backup"
    backup.mkdir(parents=True, exist_ok=True)
    targets = {"train": SHD_DIR / "shd_train.h5", "test": SHD_DIR / "shd_test.h5"}
    existed: dict[str, bool] = {}
    for name, target in targets.items():
        existed[name] = target.exists()
        if target.exists():
            shutil.copy2(target, backup / target.name)
    shutil.copy2(train_h5, targets["train"])
    shutil.copy2(val_h5, targets["test"])
    try:
        yield
    finally:
        for name, target in targets.items():
            if existed[name]:
                shutil.copy2(backup / target.name, target)
            elif target.exists():
                target.unlink()


def train_args(model: str, out: Path, *, smoke: bool) -> list[str]:
    recipe = RECIPES[model]
    return [
        "train", "--model", "ping", "--dataset", "shd",
        "--epochs", str(SMOKE_EPOCHS if smoke else EPOCHS),
        "--t-ms", str(T_MS), "--dt", str(DT_MS),
        "--n-hidden", str(N_HIDDEN), "--batch-size", str(BATCH_SIZE),
        "--lr", str(LEARNING_RATE), "--seed", str(SEED),
        "--ei-strength", str(recipe["ei_strength"]),
        "--v-grad-dampen", str(recipe["v_grad_dampen"]),
        "--w-in", str(INPUT_SCALE), "--w-in-sparsity", str(INPUT_SPARSITY),
        "--readout", READOUT_MODE,
        "--readout-w-out-scale", str(READOUT_SCALE),
        "--out-dir", str(out), "--wipe-dir",
    ]


def import_snn_modules() -> tuple[Any, Any, Any, Any, Any, Any]:
    if str(SNN) not in sys.path:
        sys.path.insert(0, str(SNN))
    import tool as snn_tool

    # Reach engine modules only through the supported tool entrypoint.  This
    # keeps the experiment import gate intact while allowing runner-side
    # orchestration around the same functions the CLI calls.
    snn_runlog = snn_tool.runlog
    snn_train = sys.modules[snn_tool.train.__module__]
    snn_config = sys.modules[snn_tool.build_config.__module__]
    snn_models = snn_tool.M
    snn_encoders = sys.modules[snn_tool.encode_batch.__module__]

    return snn_tool, snn_runlog, snn_train, snn_config, snn_models, snn_encoders


def run_train_with_registered_selection(model: str, out: Path, *, smoke: bool) -> None:
    """Run the existing trainer while retaining the registered validation winner."""
    import torch

    snn_tool, snn_runlog, _, _, _, _ = import_snn_modules()
    original_write = snn_runlog.MetricsJsonl.write
    selection: dict[str, Any] = {
        "accuracy_pct": -math.inf,
        "cross_entropy": math.inf,
        "epoch": 0,
        "rule": "max validation accuracy; tie lower validation cross-entropy; tie earlier epoch",
    }
    checkpoint_dir = out / "validation_checkpoints"

    def registered_write(writer: Any, **fields: Any) -> None:
        original_write(writer, **fields)
        if not {"ep", "acc", "test_loss"}.issubset(fields):
            return
        acc = float(fields["acc"])
        loss = float(fields["test_loss"])
        epoch = int(fields["ep"])
        rank = (acc, -loss, -epoch)
        current = (
            float(selection["accuracy_pct"]),
            -float(selection["cross_entropy"]),
            -int(selection["epoch"]),
        )
        if rank <= current:
            return
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        net = caller.f_locals.get("net") if caller is not None else None
        if net is None:
            raise RuntimeError("could not access live network for validation checkpoint")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        chosen = checkpoint_dir / f"epoch_{epoch:03d}.pth"
        torch.save({k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, chosen)
        selection.update(
            accuracy_pct=acc,
            cross_entropy=loss,
            epoch=epoch,
            checkpoint=chosen.name,
        )
        atomic_json(out / "checkpoint_selection.live.json", selection)

    snn_runlog.MetricsJsonl.write = registered_write
    try:
        rc = snn_tool.main(train_args(model, out, smoke=smoke))
        if rc:
            raise RuntimeError(f"SNN trainer returned {rc}")
    finally:
        snn_runlog.MetricsJsonl.write = original_write

    selected = checkpoint_dir / str(selection.get("checkpoint", ""))
    if not selected.exists():
        raise RuntimeError("registered validation selection produced no checkpoint")
    shutil.copy2(selected, out / "weights.pth")
    metrics_path = out / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["validation_selection"] = selection
    metrics["best_acc"] = selection["accuracy_pct"]
    metrics["best_epoch"] = selection["epoch"]
    metrics["best_validation_loss"] = selection["cross_entropy"]
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    predictions = out / "test_predictions.json"
    if predictions.exists():
        predictions.replace(out / "validation_predictions.cli_default.json")
    atomic_json(out / "checkpoint_selection.json", selection)


def validate_training(model: str, out: Path, *, smoke: bool) -> list[str]:
    errors: list[str] = []
    for name in ("config.json", "metrics.json", "weights.pth", "checkpoint_selection.json"):
        if not (out / name).exists():
            errors.append(f"missing {name}")
    if errors:
        return errors
    metrics = json.loads((out / "metrics.json").read_text())
    cfg = metrics.get("config", {})
    expected = {
        "dataset": "shd",
        "epochs": SMOKE_EPOCHS if smoke else EPOCHS,
        "seed": SEED,
        "dt": DT_MS,
        "t_ms": T_MS,
        "n_hidden": N_HIDDEN,
        "n_inh": N_INHIBITORY,
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "ei_strength": RECIPES[model]["ei_strength"],
        "v_grad_dampen": RECIPES[model]["v_grad_dampen"],
        "w_in_sparsity": INPUT_SPARSITY,
        "readout_w_out_scale": READOUT_SCALE,
        "readout_mode": READOUT_MODE,
        "dales_law": True,
        "fr_reg_upper_theta": 0.0,
        "fr_reg_upper_strength": 0.0,
    }
    for key, want in expected.items():
        if cfg.get(key) != want:
            errors.append(f"{key}: got {cfg.get(key)!r}, want {want!r}")
    if not cfg.get("w_in") or float(cfg["w_in"][0]) != INPUT_SCALE:
        errors.append("input weight mean mismatch")
    epochs = metrics.get("epochs", [])
    if len(epochs) != expected["epochs"]:
        errors.append(f"got {len(epochs)} epoch records")
    for epoch in epochs:
        for key in ("loss", "test_loss", "acc", "rate_e", "test_rate_e"):
            if not math.isfinite(float(epoch.get(key, math.nan))):
                errors.append(f"epoch {epoch.get('ep')} non-finite {key}")
        if int(epoch.get("skipped_steps", 0)) or int(epoch.get("nan_forward_batches", 0)):
            errors.append(f"epoch {epoch.get('ep')} skipped/non-finite update")
    if epochs:
        final_e = float(epochs[-1].get("rate_e", 0.0))
        if final_e <= 0:
            errors.append("silent excitatory population")
        if final_e >= 0.95 * (1000.0 / DT_MS):
            errors.append("saturated excitatory population")
        if model == "ping" and float(epochs[-1].get("rate_i") or 0.0) <= 0:
            errors.append("silent inhibitory population")
    return errors


def run_smoke() -> None:
    split_root = LOCAL_SPLIT_ROOT / "smoke"
    train_h5, val_h5, split = prepare_staged_split(split_root, smoke=True)
    summary: dict[str, Any] = {"seed": SEED, "split": split, "cells": {}}
    with installed_split(train_h5, val_h5):
        for model in MODELS:
            out = cell_dir(SMOKE_ROOT, model)
            run_train_with_registered_selection(model, out, smoke=True)
            errors = validate_training(model, out, smoke=True)
            metrics = json.loads((out / "metrics.json").read_text())
            final = metrics["epochs"][-1]
            summary["cells"][model] = {
                "passed": not errors,
                "errors": errors,
                "selected_validation_accuracy_pct": metrics["best_acc"],
                "selected_validation_loss": metrics["best_validation_loss"],
                "selected_epoch": metrics["best_epoch"],
                "final_train_loss": final["loss"],
                "final_e_rate_hz": final.get("rate_e"),
                "final_i_rate_hz": final.get("rate_i"),
            }
    summary["passed"] = all(cell["passed"] for cell in summary["cells"].values())
    atomic_json(SMOKE_ROOT / "smoke_summary.json", summary)
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit("registered local plumbing smoke failed")


def prepare_matched_input(validation_h5: Path, destination: Path) -> None:
    spikes = np.zeros((int(T_MS / DT_MS), len(RASTER_POSITIONS), N_INPUT), dtype=np.float32)
    labels: list[int] = []
    speakers: list[int] = []
    with h5py.File(validation_h5, "r") as handle:
        for trial, idx in enumerate(RASTER_POSITIONS):
            units = np.asarray(handle["spikes/units"][idx], dtype=np.int64)
            times = np.asarray(handle["spikes/times"][idx], dtype=np.float32)
            bins = np.floor(times / (DT_MS / 1000.0)).astype(np.int64)
            keep = (bins >= 0) & (bins < spikes.shape[0]) & (units >= 0) & (units < N_INPUT)
            spikes[bins[keep], trial, units[keep]] = 1.0
            labels.append(int(handle["labels"][idx]))
            speakers.append(int(handle["extra/speaker"][idx]))
    np.savez(
        destination,
        input_spikes=spikes,
        labels=np.asarray(labels, dtype=np.int64),
        speakers=np.asarray(speakers, dtype=np.int64),
        sample_indices=np.asarray(RASTER_POSITIONS, dtype=np.int64),
        dt=np.float32(DT_MS),
    )


def capture_matched_rasters(model: str, train_out: Path, validation_h5: Path) -> None:
    import torch

    snn_tool, _, _, _, _, _ = import_snn_modules()
    probe_root = train_out / "validation_probe"
    probe_root.mkdir(parents=True, exist_ok=True)
    input_path = probe_root / "matched_input.npz"
    prepare_matched_input(validation_h5, input_path)
    state = torch.load(train_out / "weights.pth", map_location="cpu")
    state.pop("W_ff.1")  # dynamics-only probe has a ten-class throwaway head
    raster_weights = probe_root / "raster_weights.pth"
    torch.save(state, raster_weights)
    raster_out = probe_root / "matched_rasters"
    rc = snn_tool.main(
        [
            "sim", "--load-config", str(train_out / "config.json"),
            "--load-weights", str(raster_weights), "--input-file", str(input_path),
            "--n-in", str(N_INPUT), "--n-batch", str(len(RASTER_POSITIONS)),
            "--outputs", "rasters", "--out-dir", str(raster_out), "--wipe-dir",
        ]
    )
    if rc or not (raster_out / "rasters.npz").exists():
        raise RuntimeError("matched raster capture failed")


def run_full_cell(model: str) -> None:
    out = cell_dir(CELL_ROOT, model)
    split_root = Path("/tmp") / f"exp069-{model}-split"
    started = utc_now()
    try:
        train_h5, val_h5, split = prepare_staged_split(split_root, smoke=False)
        SHD_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(train_h5, SHD_DIR / "shd_train.h5")
        shutil.copy2(val_h5, SHD_DIR / "shd_test.h5")
        run_train_with_registered_selection(model, out, smoke=False)
        errors = validate_training(model, out, smoke=False)
        if errors:
            raise RuntimeError(f"training validation failed: {errors}")
        selection = json.loads((out / "checkpoint_selection.json").read_text())
        frozen = {
            "model": model,
            "started_at": started,
            "frozen_at": utc_now(),
            "checkpoint_sha256": sha256_file(out / "weights.pth"),
            "selection": selection,
            "split_train_indices_sha256": split["train_indices_sha256"],
            "split_validation_indices_sha256": split["validation_indices_sha256"],
        }
        atomic_json(FROZEN_ROOT / f"{model}.json", frozen)
        capture_matched_rasters(model, out, val_h5)
        atomic_json(out / "complete.json", {"model": model, "completed_at": utc_now()})
    except Exception as error:
        atomic_json(
            FAILURE_ROOT / f"{model}.json",
            {"model": model, "failed_at": utc_now(), "error": str(error)},
        )
        raise


def cell_done(model: str) -> bool:
    out = cell_dir(CELL_ROOT, model)
    return (
        (out / "complete.json").exists()
        and (out / "validation_probe/matched_input.npz").exists()
        and (out / "validation_probe/matched_rasters/rasters.npz").exists()
    )


def pod_run() -> None:
    runpod.pod_run_loop(
        job_ids=list(MODELS),
        is_done=cell_done,
        run_job=run_full_cell,
        label="exp069-full-shd",
    )


def run_via_runpod(meta: Any) -> None:
    runpod.dispatch(
        slug=SLUG,
        runner=SLUG,
        buckets=[
            {"name": "exp069-coba", "cells": ["coba"]},
            {"name": "exp069-ping", "cells": ["ping"]},
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
        max_runtime_s=14400,
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_collected() -> dict[str, Any]:
    errors: list[str] = []
    cells: dict[str, Any] = {}
    frozen = {model: load_json(FROZEN_ROOT / f"{model}.json") for model in MODELS}
    for model in MODELS:
        out = cell_dir(CELL_ROOT, model)
        errors.extend(f"{model}: {error}" for error in validate_training(model, out, smoke=False))
        for relative in (
            "complete.json",
            "validation_probe/matched_input.npz",
            "validation_probe/matched_rasters/rasters.npz",
        ):
            if not (out / relative).exists():
                errors.append(f"{model}: missing {relative}")
        if not errors:
            cells[model] = {
                "training": load_json(out / "metrics.json"),
                "selection": load_json(out / "checkpoint_selection.json"),
                "frozen": frozen[model],
            }
    if frozen["coba"]["split_train_indices_sha256"] != frozen["ping"]["split_train_indices_sha256"]:
        errors.append("cells used different development-training indices")
    if frozen["coba"]["split_validation_indices_sha256"] != frozen["ping"]["split_validation_indices_sha256"]:
        errors.append("cells used different validation indices")
    if frozen["coba"]["split_train_indices_sha256"] != EXP068_TRAIN_HASH:
        errors.append("development-training indices differ from exp068")
    if frozen["coba"]["split_validation_indices_sha256"] != EXP068_VALIDATION_HASH:
        errors.append("validation indices differ from exp068")
    input_hashes = {
        sha256_file(cell_dir(CELL_ROOT, model) / "validation_probe/matched_input.npz")
        for model in MODELS
    }
    if len(input_hashes) != 1:
        errors.append("matched validation raster inputs differ")
    if errors:
        raise SystemExit("collected exp069 failed integrity checks:\n" + "\n".join(errors))
    return cells


def plot_validation_curves(cells: dict[str, Any], stem: Path) -> None:
    theme.apply()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25))
    styles = {"coba": (theme.DEEP_RED, "--", "s"), "ping": (theme.INK_BLACK, "-", "D")}
    for model in MODELS:
        epochs = cells[model]["training"]["epochs"]
        x = [epoch["ep"] for epoch in epochs]
        color, linestyle, marker = styles[model]
        axes[0].plot(x, [epoch["loss"] for epoch in epochs], color=color, linestyle=linestyle,
                     marker=marker, markevery=5, label=f"{model.upper()} train")
        axes[0].plot(x, [epoch["test_loss"] for epoch in epochs], color=color,
                     linestyle=":", label=f"{model.upper()} validation")
        axes[1].plot(x, [epoch["acc"] for epoch in epochs], color=color,
                     linestyle=linestyle, marker=marker, markevery=5, label=model.upper())
        selected = cells[model]["selection"]
        axes[1].scatter([selected["epoch"]], [selected["accuracy_pct"]], color=color,
                        marker=marker, s=32, zorder=4)
    axes[0].set(xlabel="epoch", ylabel="cross-entropy loss")
    axes[1].set(xlabel="epoch", ylabel="validation accuracy (%)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def plot_activity_curves(cells: dict[str, Any], stem: Path) -> None:
    theme.apply()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25), sharex=True)
    for model, color, linestyle in (
        ("coba", theme.DEEP_RED, "--"),
        ("ping", theme.INK_BLACK, "-"),
    ):
        epochs = cells[model]["training"]["epochs"]
        x = [epoch["ep"] for epoch in epochs]
        axes[0].plot(x, [epoch.get("test_rate_e", 0.0) for epoch in epochs],
                     color=color, linestyle=linestyle, label=model.upper())
        axes[1].plot(x, [epoch.get("test_rate_i", 0.0) or 0.0 for epoch in epochs],
                     color=color, linestyle=linestyle, label=model.upper())
    axes[0].set(xlabel="epoch", ylabel="validation E rate (Hz)")
    axes[1].set(xlabel="epoch", ylabel="validation I rate (Hz)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def plot_matched_rasters(stem: Path) -> None:
    theme.apply()
    input_path = cell_dir(CELL_ROOT, "coba") / "validation_probe/matched_input.npz"
    input_data = np.load(input_path)
    input_spikes = input_data["input_spikes"]
    fig, axes = plt.subplots(6, len(RASTER_POSITIONS), figsize=(6.5, 5.5), sharex=True)
    for col, position in enumerate(RASTER_POSITIONS):
        for block, model in enumerate(MODELS):
            data = np.load(cell_dir(CELL_ROOT, model) / "validation_probe/matched_rasters/rasters.npz")
            e = np.zeros((int(data["T"]), int(data["n_e"])), dtype=np.uint8)
            i = np.zeros((int(data["T"]), int(data["n_i"])), dtype=np.uint8)
            emask = data["e_trial"] == col
            imask = data["i_trial"] == col
            e[data["e_t"][emask], data["e_cell"][emask]] = 1
            i[data["i_t"][imask], data["i_cell"][imask]] = 1
            for row_offset, (array, label) in enumerate(zip((input_spikes[:, col, :], e, i),
                                                            ("input", "E", "I"))):
                axis = axes[block * 3 + row_offset, col]
                timesteps, units = np.nonzero(array)
                axis.scatter(timesteps * float(input_data["dt"]), units, s=0.25,
                             color=theme.INK_BLACK, linewidths=0, rasterized=True)
                if col == 0:
                    axis.set_ylabel(f"{model.upper()} {label}", fontsize=7)
                if block == 0 and row_offset == 0:
                    speaker = int(input_data["speakers"][col])
                    axis.set_title(f"position {position} · speaker {speaker}", fontsize=7)
                axis.spines[["top", "right"]].set_visible(False)
                axis.tick_params(labelsize=6)
    for axis in axes[-1]:
        axis.set_xlabel("time (ms)")
    fig.tight_layout(h_pad=0.3, w_pad=0.5)
    fig.savefig(stem, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def interpretation(delta: float) -> str:
    if delta >= PRIMARY_THRESHOLD_PP:
        return "registered longer-training signal"
    if delta > 0.0:
        return "weak positive gain below the registered threshold"
    return "no evidence that forty epochs caused the prior ceiling"


def publish() -> None:
    cells = validate_collected()
    ledger_path = SCRATCH / "compute_ledger.json"
    if not ledger_path.exists():
        raise SystemExit("missing compute_ledger.json with exact observed RunPod spend")
    compute_ledger = load_json(ledger_path)
    if int(compute_ledger.get("active_pods_after_collection", -1)) != 0:
        raise SystemExit("compute ledger does not confirm zero active pods")
    run_id = next_run_id(SLUG)
    t0 = time.monotonic()
    with published_run(SLUG, run_id, make_artifacts=True, scale=SCALE,
                       skip_training=True) as (_, figures):
        plot_validation_curves(cells, figures / "validation_curves")
        plot_activity_curves(cells, figures / "activity_curves")
        plot_matched_rasters(figures / "matched_rasters.png")
        raw = figures / "raw"
        raw.mkdir()
        if (SMOKE_ROOT / "smoke_summary.json").exists():
            shutil.copy2(SMOKE_ROOT / "smoke_summary.json", raw / "smoke_summary.json")
        shutil.copy2(ledger_path, raw / "compute_ledger.json")
        shutil.copy2(FROZEN_ROOT / "coba.json", raw / "coba_frozen.json")
        shutil.copy2(FROZEN_ROOT / "ping.json", raw / "ping_frozen.json")
        for model in MODELS:
            source = cell_dir(CELL_ROOT, model)
            destination = raw / model
            destination.mkdir()
            for name in ("config.json", "metrics.json", "metrics.jsonl", "checkpoint_selection.json"):
                shutil.copy2(source / name, destination / name)
            shutil.copy2(source / "validation_probe/matched_input.npz", destination / "matched_input.npz")
            shutil.copy2(
                source / "validation_probe/matched_rasters/rasters.npz",
                destination / "matched_rasters.npz",
            )
        delta = cells["ping"]["selection"]["accuracy_pct"] - PING_BASELINE_PCT
        cell_payload: dict[str, Any] = {}
        for model in MODELS:
            training = cells[model]["training"]
            selected_epoch = int(cells[model]["selection"]["epoch"])
            selected_record = training["epochs"][selected_epoch - 1]
            cell_payload[model] = {
                "selected_epoch": selected_epoch,
                "selected_validation_accuracy_pct": cells[model]["selection"]["accuracy_pct"],
                "selected_validation_loss": cells[model]["selection"]["cross_entropy"],
                "selected_validation_e_rate_hz": selected_record.get("test_rate_e"),
                "selected_validation_i_rate_hz": selected_record.get("test_rate_i"),
                "training_elapsed_s": training.get("total_elapsed_s"),
                "training_peak_gpu_memory_bytes": training.get("perf", {}).get("peak_memory_bytes"),
                "skipped_steps": sum(int(epoch.get("skipped_steps", 0)) for epoch in training["epochs"]),
                "nan_forward_batches": sum(
                    int(epoch.get("nan_forward_batches", 0)) for epoch in training["epochs"]
                ),
            }
        payload = {
            "result_status": "done",
            "seed": SEED,
            "primary": {
                "ping_change_from_exp068_pp": delta,
                "interpretation": interpretation(delta),
                "baseline_accuracy_pct": PING_BASELINE_PCT,
                "registered_threshold_pp": PRIMARY_THRESHOLD_PP,
                "criterion_met": delta >= PRIMARY_THRESHOLD_PP,
            },
            "cells": cell_payload,
            "split": {
                "source_development_count": 8156,
                "development_train_count": 7340,
                "validation_count": 816,
                "train_indices_sha256": cells["coba"]["frozen"]["split_train_indices_sha256"],
                "validation_indices_sha256": cells["coba"]["frozen"]["split_validation_indices_sha256"],
            },
            "config": SCALE,
            "integrity": {
                "checkpoint_selection_rule": cells["coba"]["selection"]["rule"],
                "official_test_access": "forbidden and absent from runner",
                "matched_partitions": True,
                "matched_raster_inputs": True,
            },
            "runpod": compute_ledger,
        }
        write_numbers(figures, run_id=run_id, duration_s=time.monotonic() - t0, payload=payload)
        (figures / "reproduce.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "uv run python experiments/exp069.py\n"
            "uv run python experiments/exp069.py --runpod\n"
            "# Add --live only with explicit RunPod spending authority.\n"
            "uv run python experiments/exp069.py --runpod --collect\n"
            "uv run python experiments/exp069.py --skip-training\n"
        )
        hashes: dict[str, str] = {}
        for path in sorted(raw.rglob("*")):
            if path.is_file():
                hashes[str(path.relative_to(figures))] = sha256_file(path)
        atomic_json(figures / "raw_sha256.json", hashes)


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
        publish()
        return
    run_smoke()


if __name__ == "__main__":
    main()
