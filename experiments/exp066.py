"""Matched Dale-constrained COBA/PING feasibility test on SHD.

Default execution runs the registered local smoke only.  The pilot is dispatched
explicitly through ``--runpod --live`` after the smoke gate has passed and this
runner has been committed and pushed.  Pilot cells share every training setting
except the two preregistered architectural recipe differences: loop strength and
voltage-gradient dampening.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import runpod, theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.datasets import load_shd_events  # noqa: E402
from helpers.figsave import save_figure  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_cli import run_cli  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402

SLUG = "exp066"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
SCRATCH = runpod.artifacts_scratch(SLUG)
PILOT_ROOT = SCRATCH / "pilot"
SMOKE_ROOT = REPO / "temp" / "experiments" / SLUG / "smoke"

MODELS = ("coba", "ping")
SAMPLE_INDICES = (0, 1, 2)  # fixed before predictions are inspected
SEED = 42
CHANCE_PCT = 5.0
SUCCESS_PCT = 20.0
INPUT_SCALE = 0.9

COMMON = {
    "dataset": "shd", "max_samples": 1000, "epochs": 20,
    "t_ms": 1000.0, "dt_ms": 1.0, "n_hidden": 256,
    "n_inhibitory": 64, "batch_size": 32, "learning_rate": 0.0004,
    "input_weight_mean": INPUT_SCALE, "input_sparsity": 0.95,
    "readout_scale": 225.0, "readout": "mem-mean", "seed": SEED,
    "firing_rate_regularizer": False, "dales_law": True,
}
RECIPES = {
    "coba": {"ei_strength": 0.0, "v_grad_dampen": 1.0},
    "ping": {"ei_strength": 1.0, "v_grad_dampen": 1000.0},
}
SCALE = {**COMMON, "models": list(MODELS), "test_sample_indices": list(SAMPLE_INDICES)}


def cell_dir(root: Path, model: str) -> Path:
    return root / model


def train_args(model: str, out: Path, *, smoke: bool) -> list[str]:
    recipe = RECIPES[model]
    return [
        "train", "--model", "ping", "--dataset", "shd",
        "--max-samples", "128" if smoke else str(COMMON["max_samples"]),
        "--epochs", "2" if smoke else str(COMMON["epochs"]),
        "--t-ms", str(COMMON["t_ms"]), "--dt", str(COMMON["dt_ms"]),
        "--n-hidden", str(COMMON["n_hidden"]), "--batch-size", str(COMMON["batch_size"]),
        "--lr", str(COMMON["learning_rate"]), "--seed", str(SEED),
        "--ei-strength", str(recipe["ei_strength"]),
        "--v-grad-dampen", str(recipe["v_grad_dampen"]),
        "--w-in", str(INPUT_SCALE), "--w-in-sparsity", str(COMMON["input_sparsity"]),
        "--readout", COMMON["readout"],
        "--readout-w-out-scale", str(COMMON["readout_scale"]),
        "--out-dir", str(out), "--wipe-dir",
    ]


def infer_args(model: str, train: Path, out: Path, sample_idx: int) -> list[str]:
    return [
        "sim", "--infer", "--load-config", str(train / "config.json"),
        "--load-weights", str(train / "weights.pth"),
        "--dataset", "shd",
        "--sample-index", str(sample_idx), "--out-dir", str(out), "--wipe-dir",
    ]


def matched_input_path(root: Path) -> Path:
    return root / "matched_input.npz"


def prepare_matched_input(root: Path) -> Path:
    """Bin the preregistered official-test positions exactly as SHD training."""
    path = matched_input_path(root)
    if path.exists():
        return path
    events, labels = load_shd_events(split="test")
    spikes = np.zeros(
        (int(COMMON["t_ms"] / COMMON["dt_ms"]), len(SAMPLE_INDICES), 700),
        dtype=np.float32,
    )
    for trial, idx in enumerate(SAMPLE_INDICES):
        units, times = events[idx]
        bins = np.floor(times / (COMMON["dt_ms"] / 1000.0)).astype(np.int64)
        keep = (bins >= 0) & (bins < spikes.shape[0]) & (units >= 0) & (units < 700)
        spikes[bins[keep], trial, units[keep].astype(np.int64)] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, input_spikes=spikes,
        labels=np.asarray([labels[i] for i in SAMPLE_INDICES], dtype=np.int64),
        sample_indices=np.asarray(SAMPLE_INDICES, dtype=np.int64),
        dt=np.float32(COMMON["dt_ms"]),
    )
    return path


def batch_raster_args(train: Path, out: Path, input_path: Path) -> list[str]:
    """Existing CLI arbitrary-input probe: saved weights in, sparse rasters out."""
    return [
        "sim", "--load-config", str(train / "config.json"),
        "--load-weights", str(raster_weights(train)),
        "--input-file", str(input_path), "--n-in", "700", "--n-batch", "3",
        "--outputs", "rasters", "--out-dir", str(out), "--wipe-dir",
    ]


def raster_weights(train: Path) -> Path:
    """Omit only the unused 20-class readout from the 10-class probe builder."""
    import torch

    source = train / "weights.pth"
    derived = train / "raster_weights.pth"
    if not derived.exists():
        state = torch.load(source, map_location="cpu")
        state.pop("W_ff.1")
        torch.save(state, derived)
    return derived


def load_metrics(root: Path, model: str) -> dict:
    return json.loads((cell_dir(root, model) / "metrics.json").read_text())


def finite_and_active(metrics: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    epochs = metrics.get("epochs", [])
    if not epochs:
        return False, ["no epoch records"]
    numeric = ("loss", "test_loss", "acc", "rate_e", "test_rate_e")
    for ep in epochs:
        for key in numeric:
            value = ep.get(key)
            if value is None or not math.isfinite(float(value)):
                errors.append(f"epoch {ep.get('ep')} has non-finite {key}")
        if ep.get("nan_forward_batches", 0) or ep.get("skipped_steps", 0):
            errors.append(f"epoch {ep.get('ep')} has skipped/non-finite updates")
    final = epochs[-1]
    if float(final.get("rate_e", 0.0)) <= 0.0:
        errors.append("excitatory population is silent")
    # ``act`` is the fraction of cells that fired at least once over the whole
    # utterance, so 100% active is not saturation.  Saturation means firing on
    # nearly every integration step; at dt=1 ms that is near 1000 Hz.
    if float(final.get("rate_e", 0.0)) >= 0.95 * (1000.0 / COMMON["dt_ms"]):
        errors.append("excitatory population is saturated")
    if RECIPES[metrics.get("cell_model", "coba")]["ei_strength"] > 0:
        rate_i = final.get("rate_i")
        if rate_i is None or float(rate_i) <= 0.0:
            errors.append("inhibitory population is silent")
    return not errors, errors


def validate_config(metrics: dict, model: str, *, smoke: bool) -> list[str]:
    cfg = metrics.get("config", {})
    want = {
        "dataset": "shd", "max_samples": 128 if smoke else 1000,
        "epochs": 2 if smoke else 20, "seed": SEED, "dt": 1.0,
        "t_ms": 1000.0, "n_hidden": 256, "n_inh": 64, "batch_size": 32,
        "lr": 0.0004, "ei_strength": RECIPES[model]["ei_strength"],
        "v_grad_dampen": RECIPES[model]["v_grad_dampen"],
        "w_in_sparsity": 0.95, "readout_w_out_scale": 225.0,
        "readout_mode": "mem-mean", "dales_law": True,
        "fr_reg_upper_theta": 0.0, "fr_reg_upper_strength": 0.0,
    }
    errors = []
    for key, expected in want.items():
        got = cfg.get(key)
        if got != expected:
            errors.append(f"{key}: got {got!r}, want {expected!r}")
    w_in = cfg.get("w_in") or []
    if not w_in or float(w_in[0]) != INPUT_SCALE:
        errors.append(f"w_in mean: got {w_in!r}, want {INPUT_SCALE}")
    return errors


def validate_cell(root: Path, model: str, *, smoke: bool, rasters: bool = False) -> list[str]:
    d = cell_dir(root, model)
    errors = [f"missing {name}" for name in ("config.json", "metrics.json", "weights.pth")
              if not (d / name).exists()]
    if errors:
        return errors
    metrics = load_metrics(root, model)
    metrics["cell_model"] = model
    errors += validate_config(metrics, model, smoke=smoke)
    _, activity_errors = finite_and_active(metrics)
    errors += activity_errors
    if rasters:
        raster_path = d / "matched_rasters" / "rasters.npz"
        input_path = matched_input_path(root)
        if not raster_path.exists():
            errors.append("missing matched sparse rasters")
        if not input_path.exists():
            errors.append("missing matched input spikes")
        if raster_path.exists() and int(np.load(raster_path)["n_trials"]) != len(SAMPLE_INDICES):
            errors.append("matched raster trial count is wrong")
    return errors


def capture_rasters(root: Path, model: str) -> None:
    train = cell_dir(root, model)
    raster_out = train / "matched_rasters"
    if (raster_out / "rasters.npz").exists():
        return
    input_path = prepare_matched_input(root)
    print(f"[raster] {model} official SHD test positions {SAMPLE_INDICES}")
    run_cli(batch_raster_args(train, raster_out, input_path))


def train_cell(root: Path, model: str, *, smoke: bool, capture: bool) -> None:
    out = cell_dir(root, model)
    print(f"[train] {model} ({'smoke' if smoke else 'pilot'})")
    run_cli(train_args(model, out, smoke=smoke))
    if capture:
        capture_rasters(root, model)


def run_smoke() -> None:
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {"input_scale": INPUT_SCALE, "adjustment_used": False, "cells": {}}
    for model in MODELS:
        train_cell(SMOKE_ROOT, model, smoke=True, capture=False)
        metrics = load_metrics(SMOKE_ROOT, model)
        errors = validate_cell(SMOKE_ROOT, model, smoke=True)
        final = metrics["epochs"][-1]
        summary["cells"][model] = {
            "passed": not errors, "errors": errors,
            "final_loss": final["loss"], "final_test_loss": final["test_loss"],
            "final_accuracy_pct": final["acc"], "final_e_rate_hz": final.get("rate_e"),
            "final_i_rate_hz": final.get("rate_i"), "skipped_steps": sum(
                int(ep.get("skipped_steps", 0)) for ep in metrics["epochs"]),
            "nan_forward_batches": sum(
                int(ep.get("nan_forward_batches", 0)) for ep in metrics["epochs"]),
        }
    summary["passed"] = all(c["passed"] for c in summary["cells"].values())
    (SMOKE_ROOT / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit("smoke gate failed; shared input-scale adjustment is required before pilot")


def pilot_done(model: str) -> bool:
    return not validate_cell(PILOT_ROOT, model, smoke=False, rasters=True)


def pod_run() -> None:
    def run_job(model: str) -> None:
        training_errors = validate_cell(PILOT_ROOT, model, smoke=False, rasters=False)
        if training_errors:
            train_cell(PILOT_ROOT, model, smoke=False, capture=False)
        else:
            print(f"[reuse] {model} validated training cell")
        capture_rasters(PILOT_ROOT, model)
        errors = validate_cell(PILOT_ROOT, model, smoke=False, rasters=True)
        if errors:
            raise RuntimeError(f"{model} pilot validation failed: {errors}")

    runpod.pod_run_loop(job_ids=list(MODELS), is_done=pilot_done, run_job=run_job,
                        label="exp066-pilot")


def run_via_runpod(meta) -> None:
    runpod.dispatch(
        slug=SLUG, runner=SLUG,
        buckets=[{"name": "exp066-pilot", "cells": list(MODELS)}],
        gpu=meta.gpu, live=meta.live, collect=meta.collect, plumbing=False,
        collect_subdir=f"{runpod.ARTIFACTS_SUBDIR}/{SLUG}",
        local_collect_dir=str(SCRATCH),
        extra_env={"PINGLAB_ARTIFACTS_ROOT": f"{runpod.VOLUME_MOUNT}/{runpod.ARTIFACTS_SUBDIR}/{SLUG}"},
        max_runtime_s=7200,
    )


def plot_learning(root: Path, stem: Path) -> None:
    theme.apply()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25))
    styles = {"coba": (theme.DEEP_RED, "--", "s"), "ping": (theme.INK_BLACK, "-", "D")}
    for model in MODELS:
        eps = load_metrics(root, model)["epochs"]
        x = [e["ep"] for e in eps]
        color, ls, marker = styles[model]
        axes[0].plot(x, [e["loss"] for e in eps], color=color, ls=ls, marker=marker,
                     markevery=4, label=f"{model.upper()} train")
        axes[0].plot(x, [e["test_loss"] for e in eps], color=color, ls=":",
                     label=f"{model.upper()} test")
        axes[1].plot(x, [e["acc"] for e in eps], color=color, ls=ls, marker=marker,
                     markevery=4, label=model.upper())
    axes[1].axhline(CHANCE_PCT, color=theme.GREY_MID, ls=":", label="chance")
    axes[1].axhline(SUCCESS_PCT, color=theme.GREY_MID, ls="--", label="criterion")
    axes[0].set(xlabel="epoch", ylabel="cross-entropy loss")
    axes[1].set(xlabel="epoch", ylabel="test accuracy (%)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def plot_rates(root: Path, stem: Path) -> None:
    theme.apply()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25), sharex=True)
    for model, color, ls in (("coba", theme.DEEP_RED, "--"), ("ping", theme.INK_BLACK, "-")):
        eps = load_metrics(root, model)["epochs"]
        x = [e["ep"] for e in eps]
        axes[0].plot(x, [e.get("rate_e", 0) for e in eps], color=color, ls=ls, label=model.upper())
        axes[1].plot(x, [e.get("rate_i", 0) or 0 for e in eps], color=color, ls=ls, label=model.upper())
    axes[0].set(xlabel="epoch", ylabel="excitatory rate (Hz)")
    axes[1].set(xlabel="epoch", ylabel="inhibitory rate (Hz)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, stem)
    plt.close(fig)


def plot_rasters(root: Path, out: Path) -> None:
    theme.apply()
    fig, axes = plt.subplots(6, len(SAMPLE_INDICES), figsize=(6.5, 5.5), sharex=True)
    input_data = np.load(matched_input_path(root))
    input_spikes = input_data["input_spikes"]
    for col, idx in enumerate(SAMPLE_INDICES):
        for block, model in enumerate(MODELS):
            data = np.load(cell_dir(root, model) / "matched_rasters" / "rasters.npz")
            e = np.zeros((int(data["T"]), int(data["n_e"])), dtype=np.uint8)
            i = np.zeros((int(data["T"]), int(data["n_i"])), dtype=np.uint8)
            em = data["e_trial"] == col
            im = data["i_trial"] == col
            e[data["e_t"][em], data["e_cell"][em]] = 1
            i[data["i_t"][im], data["i_cell"][im]] = 1
            arrays = (input_spikes[:, col, :], e, i)
            for rowoff, (arr, label) in enumerate(zip(arrays, ("input", "E", "I"))):
                ax = axes[block * 3 + rowoff, col]
                ts, units = np.nonzero(arr)
                ax.scatter(ts * float(input_data["dt"]), units, s=0.25, color=theme.INK_BLACK,
                           linewidths=0, rasterized=True)
                if col == 0:
                    ax.set_ylabel(f"{model.upper()} {label}", fontsize=7)
                if block == 0 and rowoff == 0:
                    ax.set_title(f"test position {idx}", fontsize=8)
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(labelsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("time (ms)")
    fig.tight_layout(h_pad=0.3, w_pad=0.5)
    fig.savefig(out, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def publish() -> None:
    errors = {m: validate_cell(PILOT_ROOT, m, smoke=False, rasters=True) for m in MODELS}
    if any(errors.values()):
        raise SystemExit(f"pilot artifacts incomplete or invalid: {errors}")
    run_id = next_run_id(SLUG)
    t0 = time.monotonic()
    with published_run(SLUG, run_id, make_artifacts=True, scale=SCALE,
                       skip_training=True) as (artifacts, figures):
        plot_learning(PILOT_ROOT, figures / "learning_curves")
        plot_rates(PILOT_ROOT, figures / "firing_rates")
        plot_rasters(PILOT_ROOT, figures / "matched_rasters.png")
        raw = figures / "raw"
        raw.mkdir()
        shutil.copy2(matched_input_path(PILOT_ROOT), raw / "matched_input.npz")
        for model in MODELS:
            model_raw = raw / model
            model_raw.mkdir()
            train = cell_dir(PILOT_ROOT, model)
            for name in ("metrics.json", "metrics.jsonl", "test_predictions.json"):
                shutil.copy2(train / name, model_raw / name)
            shutil.copy2(
                train / "matched_rasters" / "rasters.npz",
                model_raw / "matched_rasters.npz",
            )
            shutil.copy2(
                train / "matched_rasters" / "metrics.json",
                model_raw / "replay_metrics.json",
            )
        if (SMOKE_ROOT / "smoke_summary.json").exists():
            shutil.copy2(SMOKE_ROOT / "smoke_summary.json", raw / "smoke_summary.json")
        cells = {}
        for model in MODELS:
            metrics = load_metrics(PILOT_ROOT, model)
            eps = metrics["epochs"]
            cells[model] = {
                "best_accuracy_pct": metrics["best_acc"], "best_epoch": metrics["best_epoch"],
                "final_accuracy_pct": eps[-1]["acc"], "final_train_loss": eps[-1]["loss"],
                "final_test_loss": eps[-1]["test_loss"], "final_e_rate_hz": eps[-1].get("rate_e"),
                "final_i_rate_hz": eps[-1].get("rate_i"),
                "nan_forward_batches": sum(int(e.get("nan_forward_batches", 0)) for e in eps),
                "skipped_steps": sum(int(e.get("skipped_steps", 0)) for e in eps),
                "beat_chance": metrics["best_acc"] > CHANCE_PCT,
                "passed_registered_accuracy": metrics["best_acc"] > SUCCESS_PCT,
                "total_elapsed_s": metrics.get("total_elapsed_s"),
            }
        payload = {
            "chance_accuracy_pct": CHANCE_PCT, "success_criterion_pct": SUCCESS_PCT,
            "seed": SEED, "input_scale_adjustment_used": False,
            "preselected_test_positions": list(SAMPLE_INDICES), "cells": cells,
            "experiment_passed": all(c["passed_registered_accuracy"] for c in cells.values()),
            "config": {**SCALE, "recipes": RECIPES},
            "runpod": {
                "hourly_rate_usd": 0.99,
                "attempt_1_observed_s": 1435.063,
                "attempt_2_observed_s": 78.492,
                "total_observed_s": 1513.555,
                "total_spend_usd": 0.416227625,
                "active_pods_after_collection": 0,
            },
        }
        input_data = np.load(matched_input_path(PILOT_ROOT))
        payload["raster_evidence"] = {
            "labels": input_data["labels"].astype(int).tolist(),
            "input_event_counts": input_data["input_spikes"].sum(axis=(0, 2)).astype(int).tolist(),
            "cells": {},
        }
        for model in MODELS:
            replay = json.loads(
                (cell_dir(PILOT_ROOT, model) / "matched_rasters" / "metrics.json").read_text()
            )
            payload["raster_evidence"]["cells"][model] = {
                "e_rate_hz": replay["rate_e_hz"],
                "i_rate_hz": replay["rate_i_hz"],
            }
        write_numbers(figures, run_id=run_id, duration_s=time.monotonic() - t0, payload=payload)
        (figures / "reproduce.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\nuv run python experiments/exp066.py --runpod\n"
            "# Add --live only with explicit RunPod spending authority.\n"
            "uv run python experiments/exp066.py --runpod --collect\n"
            "uv run python experiments/exp066.py --skip-training\n"
        )
        hashes = {}
        for path in sorted(raw.rglob("*")):
            if path.is_file():
                hashes[str(path.relative_to(figures))] = hashlib.sha256(path.read_bytes()).hexdigest()
        (figures / "raw_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


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
