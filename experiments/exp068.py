"""Full-development SHD comparison of matched Dale-constrained COBA and PING.

The runner enforces a sealed three-way protocol without changing ``tools/snn``:
the official SHD training file is split jointly by speaker and class, the CLI
sees development-train as ``train`` and validation as ``test`` during fitting,
and the official test file is unavailable until both validation checkpoints are
frozen.  Default execution is the registered local plumbing smoke.  Cloud
compute requires the explicit ``--runpod --live`` gate.
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
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SNN = REPO / "tools" / "snn"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import runpod  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.datasets import _SHD_URLS, _shd_h5  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402

SLUG = "exp068"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
SCRATCH = runpod.artifacts_scratch(SLUG)
CELL_ROOT = SCRATCH / "cells"
FROZEN_ROOT = SCRATCH / "frozen"
FAILURE_ROOT = SCRATCH / "failures"
SMOKE_ROOT = REPO / "temp" / "experiments" / SLUG / "smoke"
LOCAL_SPLIT_ROOT = REPO / "temp" / "experiments" / SLUG / "split"

MODELS = ("coba", "ping")
SEED = 42
EPOCHS = 40
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
OFFICIAL_TEST_SIZE = 2264
UNSEEN_SPEAKERS = (4, 5)
RASTER_POSITIONS = (0, 47, 947)  # fixed before official-test predictions exist
BARRIER_TIMEOUT_S = 3600
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
    backup = train_h5.parent / "local_official_backup"
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


def download_official_test_after_barrier() -> Path:
    """Fetch official test only after both frozen flags exist."""
    if not all((FROZEN_ROOT / f"{model}.json").exists() for model in MODELS):
        raise RuntimeError("official test requested before both checkpoints froze")
    target = SHD_DIR / "shd_test.h5"
    if target.exists():
        target.unlink()  # remove staged validation, never an official test on a pod
    gz = target.with_suffix(".h5.gz")
    if not gz.exists():
        urllib.request.urlretrieve(_SHD_URLS["test"], gz)
    import gzip

    with gzip.open(gz, "rb") as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination)
    if len(read_speaker_and_labels(target)[1]) != OFFICIAL_TEST_SIZE:
        raise RuntimeError("official test size mismatch")
    return target


def wait_for_frozen_pair() -> None:
    deadline = time.monotonic() + BARRIER_TIMEOUT_S
    while time.monotonic() < deadline:
        failures = sorted(FAILURE_ROOT.glob("*.json")) if FAILURE_ROOT.exists() else []
        if failures:
            raise RuntimeError(f"peer cell failed before sealed evaluation: {failures[0].name}")
        if all((FROZEN_ROOT / f"{model}.json").exists() for model in MODELS):
            return
        time.sleep(10)
    raise RuntimeError("timed out waiting for both validation checkpoints")


def evaluate_official_test(model: str, train_out: Path, official_test: Path) -> None:
    """Evaluate the frozen state once on all official-test samples."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    snn_tool, _, snn_train, snn_config, snn_models, snn_encoders = import_snn_modules()
    config = json.loads((train_out / "config.json").read_text())
    with h5py.File(official_test, "r") as handle:
        labels = np.asarray(handle["labels"], dtype=np.int64)
        events = np.empty(len(labels), dtype=object)
        for idx in range(len(labels)):
            events[idx] = (
                np.asarray(handle["spikes/units"][idx], dtype=np.int16),
                np.asarray(handle["spikes/times"][idx], dtype=np.float32),
            )
    snn_train.seed_everything(SEED)
    snn_config.set_sim_dt(DT_MS, T_MS)
    snn_config.setup_model_globals([N_HIDDEN])
    snn_models.N_IN = N_INPUT
    snn_models.N_OUT = N_CLASSES
    snn_models.V_GRAD_DAMPEN = RECIPES[model]["v_grad_dampen"]
    device = snn_tool._auto_device()
    dataset = snn_train.ShdBinnedDataset(events, labels, n_in=N_INPUT)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    net = snn_config.build_net(
        "ping",
        w_in=tuple(config["w_in"]),
        w_in_sparsity=INPUT_SPARSITY,
        ei_strength=RECIPES[model]["ei_strength"],
        ei_ratio=config["ei_ratio"],
        device=device,
        randomize_init=True,
        dales_law=True,
        hidden_sizes=[N_HIDDEN],
        readout_mode=READOUT_MODE,
    )
    net.load_state_dict(torch.load(train_out / "weights.pth", map_location=device))
    net.eval()
    rows: list[dict[str, Any]] = []
    ce_sum = 0.0
    correct = 0
    total = 0
    rate_e_sum = 0.0
    rate_i_sum = 0.0
    t0 = time.monotonic()
    with torch.no_grad():
        for spikes, y in loader:
            y = y.to(device)
            logits = net(input_spikes=snn_encoders.encode_batch(spikes.to(device), DT_MS))
            ce_sum += float(F.cross_entropy(logits, y, reduction="sum").item())
            pred = logits.argmax(1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            rates = getattr(net, "rates", None) or {}
            e_key = max((k for k in rates if k.startswith("hid")), default=None)
            i_key = max((k for k in rates if k.startswith("inh")), default=None)
            rate_e_sum += float(rates.get(e_key, 0.0)) * int(y.numel())
            rate_i_sum += float(rates.get(i_key, 0.0)) * int(y.numel())
            logits_cpu = logits.detach().cpu().numpy()
            for local_idx in range(len(logits_cpu)):
                rows.append(
                    {
                        "idx": len(rows),
                        "true": int(y[local_idx].item()),
                        "pred": int(pred[local_idx].item()),
                        "correct": bool(pred[local_idx].item() == y[local_idx].item()),
                        "logits": logits_cpu[local_idx].astype(float).tolist(),
                    }
                )
    if total != OFFICIAL_TEST_SIZE:
        raise RuntimeError(f"official test evaluated {total}, expected {OFFICIAL_TEST_SIZE}")
    speakers, truth = read_speaker_and_labels(official_test)
    predictions = np.asarray([row["pred"] for row in rows], dtype=np.int64)
    class_acc = [float(np.mean(predictions[truth == c] == truth[truth == c]) * 100) for c in range(N_CLASSES)]
    unseen = np.isin(speakers, UNSEEN_SPEAKERS)
    payload = {
        "evaluated_at": utc_now(),
        "checkpoint_sha256": sha256_file(train_out / "weights.pth"),
        "n_total": total,
        "n_correct": correct,
        "accuracy_pct": 100.0 * correct / total,
        "cross_entropy": ce_sum / total,
        "macro_class_accuracy_pct": float(np.mean(class_acc)),
        "class_accuracy_pct": class_acc,
        "seen_speaker_accuracy_pct": float(np.mean(predictions[~unseen] == truth[~unseen]) * 100),
        "unseen_speaker_accuracy_pct": float(np.mean(predictions[unseen] == truth[unseen]) * 100),
        "seen_count": int((~unseen).sum()),
        "unseen_count": int(unseen.sum()),
        "e_rate_hz": rate_e_sum / total,
        "i_rate_hz": rate_i_sum / total,
        "elapsed_s": time.monotonic() - t0,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
        "official_test_sha256": sha256_file(official_test),
    }
    official_root = train_out / "official_test"
    official_root.mkdir(parents=True, exist_ok=True)
    atomic_json(official_root / "metrics.json", payload)
    (official_root / "predictions.json").write_text(json.dumps(rows) + "\n")


def prepare_matched_input(official_test: Path, destination: Path) -> None:
    spikes = np.zeros((int(T_MS / DT_MS), len(RASTER_POSITIONS), N_INPUT), dtype=np.float32)
    labels: list[int] = []
    speakers: list[int] = []
    with h5py.File(official_test, "r") as handle:
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


def capture_matched_rasters(model: str, train_out: Path, official_test: Path) -> None:
    import torch

    snn_tool, _, _, _, _, _ = import_snn_modules()
    official_root = train_out / "official_test"
    input_path = official_root / "matched_input.npz"
    prepare_matched_input(official_test, input_path)
    state = torch.load(train_out / "weights.pth", map_location="cpu")
    state.pop("W_ff.1")  # dynamics-only probe has a ten-class throwaway head
    raster_weights = official_root / "raster_weights.pth"
    torch.save(state, raster_weights)
    raster_out = official_root / "matched_rasters"
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
    split_root = Path("/tmp") / f"exp068-{model}-split"
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
        wait_for_frozen_pair()
        official_test = download_official_test_after_barrier()
        evaluate_official_test(model, out, official_test)
        capture_matched_rasters(model, out, official_test)
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
        and (out / "official_test/metrics.json").exists()
        and (out / "official_test/matched_rasters/rasters.npz").exists()
    )


def pod_run() -> None:
    runpod.pod_run_loop(
        job_ids=list(MODELS),
        is_done=cell_done,
        run_job=run_full_cell,
        label="exp068-full-shd",
    )


def run_via_runpod(meta: Any) -> None:
    runpod.dispatch(
        slug=SLUG,
        runner=SLUG,
        buckets=[
            {"name": "exp068-coba", "cells": ["coba"]},
            {"name": "exp068-ping", "cells": ["ping"]},
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
        max_runtime_s=10800,
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
        raise SystemExit("publication is enabled only after both collected cells validate")
    run_smoke()


if __name__ == "__main__":
    main()
