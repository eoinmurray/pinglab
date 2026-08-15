"""Notebook runner for entry 038 — functional probes of trained
PING vs COBA.

Consumes COBA/PING checkpoints from exp022, then runs inference-only probes:
- input-rate sweep: per-cell f-I curve on MNIST digit 0 + uniform
  Poisson input;
- COBA → PING I-loop transfer: replay trained COBA at eval-time
  ei_strength ∈ [0, 1] and watch the gamma cycle self-assemble; and
- (readout latency removed)
  wall-clock time and cumulative spike count.

Figures land in /figures/notebooks/exp038/ and the success-criteria
summary in exp038/numbers.json.

Writing: writings/exp038.typ · figures + numbers.json: artifacts/data/exp038/
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp022 import RATE_TARGET_GRID_HZ as SHARED_RATE_TARGET_GRID_HZ  # noqa: E402
from exp022 import SEEDS_BASELINE as SHARED_SEEDS  # noqa: E402
from exp022 import cell_dir as shared_cell_dir  # noqa: E402
from exp022 import cell_name  # noqa: E402
from helpers import theme  # noqa: E402
from helpers.checkpoints import (  # noqa: E402
    cache_tag,
    checkpoint_provenance,
    resolve_checkpoint,
)
from helpers.cli import parse_meta, replot_target  # noqa: E402
from helpers.datasets import MNIST_REDUCED_EVAL_SAMPLES  # noqa: E402
from helpers.figsave import save_figure  # noqa: E402
from helpers.frontier import summarize_frontier  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.paths import artifacts_and_figures, runner_paths  # noqa: E402
from helpers.run_cli import run_cli  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402

SLUG = "exp038"
CHECKPOINT_ROLE = "best_validation"
RUN_PATHS = runner_paths(SLUG)
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

MAX_SAMPLES = 7000  # exp022 sweep-cell scale (10% of MNIST); reporting only
SMOKE = os.environ.get("PINGLAB_SMOKE") == "1"
EVAL_MAX_SAMPLES = 100 if SMOKE else MNIST_REDUCED_EVAL_SAMPLES
EVAL_CORPUS_SAMPLES = 10000
T_MS = 200.0
DT_TRAIN = 0.1
BASELINE_EPOCHS: int = 50  # baseline cell training horizon (in exp022 now)

# Run scale — stamped into the manifest by run_dirs.prepare and rendered as
# the Methods table via RunScale; the mdx never restates these numbers.
SCALE = {
    "dataset": "mnist",
    "max_samples": MAX_SAMPLES,
    "evaluation_samples": EVAL_MAX_SAMPLES,
    "epochs": BASELINE_EPOCHS,
    "t_ms": T_MS,
    "dt_ms": DT_TRAIN,
    "batch_size": 256,
    "seeds": 3,  # SEEDS_BASELINE
    "cells": 36,
    "grid": "2 models × 6 rate targets × 3 seeds",
}

# Every activity-frontier point uses the same independent seeds.
SEEDS_BASELINE: list[int] = list(SHARED_SEEDS)

# Inference-time ei_strength sweep on all three COBA baselines.
# Subsumes the now-retired nb019 — trains nothing new; just runs the
# already-trained coba weights forward through the test set with a
# fresh ping-arch I-loop at progressively higher ei_strength.
EI_SWEEP: list[float] = [round(0.1 * i, 1) for i in range(11)]  # 0.0–1.0
EI_RASTER: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
EI_RASTER_SAMPLE_IDX: int = 0
EI_RASTER_N_E_PLOT: int = 200
EI_RASTER_N_I_PLOT: int = 64
if SMOKE:
    EI_SWEEP = [0.0, 0.5, 1.0]
    EI_RASTER = [0.0, 1.0]

# Hidden-E population-rate targets in Hz. None = no penalty (baseline).
# pressure (off → ~80 Hz coba baseline) down to 1 Hz —
# below ping's natural 5 Hz and into the regime where every model
# loses accuracy.
RATE_TARGET_GRID_HZ: list[float | None] = list(SHARED_RATE_TARGET_GRID_HZ)
FR_STRENGTH_UPPER = 1e-3

MODELS = ["coba", "ping"]

MODEL_COLORS = {
    "coba": theme.DEEP_RED,
    "ping": theme.INK_BLACK,
}
MODEL_MARKERS = {"coba": "s", "ping": "D"}


def rate_target_label(rate_target_hz: float | None) -> str:
    """Filesystem-safe label for an out-dir."""
    if rate_target_hz is None:
        return "off"
    s = f"{rate_target_hz:g}".replace(".", "p")
    return f"tu{s}"


def rate_target_display(rate_target_hz: float | None) -> str:
    """Human label for plots / numbers.json."""
    if rate_target_hz is None:
        return "off"
    return f"{rate_target_hz:g}"


def rate_target_hz_value(rate_target_hz: float | None) -> float | None:
    if rate_target_hz is None:
        return None
    return rate_target_hz


def seeds_for(rate_target_hz: float | None) -> list[int]:
    """Return the independent seeds used at every frontier point."""
    return list(SEEDS_BASELINE)


def cell_dir(model: str, rate_target_hz: float | None, seed: int) -> Path:
    """Trained cell — now the shared exp022 cell (train-once / reuse-many).
    exp022 owns the rate target sweep; this notebook only consumes it."""
    if RUN_PATHS.isolated and not os.environ.get("PINGLAB_TRAINING_ROOT"):
        raise RuntimeError("isolated exp038 requires explicit PINGLAB_TRAINING_ROOT")
    return shared_cell_dir(cell_name(model, rate_target_hz, seed))


def _log_event(event: str, **fields: object) -> None:
    RUN_PATHS.logs.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "experiment": SLUG, **fields}
    with (RUN_PATHS.logs / f"{SLUG}.jsonl").open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def baseline_dir(model: str, seed: int = SEEDS_BASELINE[0]) -> Path:
    return cell_dir(model, None, seed)


def checkpoint_path(train_dir: Path) -> Path:
    return resolve_checkpoint(train_dir, CHECKPOINT_ROLE)["path"]


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())







# ── COBA→PING ei_strength sweep (subsumes nb019) ──────────────────────


def _ei_sweep_dir(seed: int | None = None) -> Path:
    root = ARTIFACTS / "ei_sweep"
    return root if seed is None else root / f"seed{seed}"


def run_inproc_infer(train_dir: Path, ei_strength: float, out_dir: Path) -> dict:
    """Transfer-load W_ff/W_ee from the trained COBA checkpoint into a fresh ping
    net at the requested ei_strength (skip W_ei/W_ie so the fresh I-loop survives),
    evaluate accuracy + mean E/I rate. Runs `sim --infer --skip-load W_ei. W_ie.`.
    """
    out_dir = Path(out_dir) / cache_tag(resolve_checkpoint(train_dir, CHECKPOINT_ROLE))
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cli(
        [
            "sim", "--infer",
            "--load-config", str((train_dir / "config.json").resolve()),
            "--load-weights", str(checkpoint_path(train_dir)),
            "--ei-strength", str(ei_strength),
            "--skip-load", "W_ei.", "W_ie.",
            "--max-samples", str(EVAL_MAX_SAMPLES),
            "--out-dir", str(out_dir.resolve()),
        ]
    )
    m = json.loads((out_dir / "metrics.json").read_text())
    rates_hz = m.get("rates_hz", {})
    hid = next((k for k in rates_hz if k.startswith("hid")), None)
    inh = next((k for k in rates_hz if k.startswith("inh")), None)
    metrics = {
        "mode": "infer",
        "ei_strength": ei_strength,
        "best_acc": float(m["best_acc"]),
        "n_correct": int(m.get("n_correct", 0)),
        "n_total": int(m.get("n_total", 0)),
        "rates_hz": rates_hz,
        "hid_rate_hz": rates_hz.get(hid) if hid else None,
        "inh_rate_hz": rates_hz.get(inh) if inh else None,
    }
    print(f"  ei={ei_strength:g}: acc={metrics['best_acc']:.2f}%  "
          f"hid={metrics['hid_rate_hz']:.1f}Hz")
    return metrics


def capture_ei_raster(
    train_dir: Path, ei_strength: float, sample_idx: int, *, seed: int
) -> dict:
    """Single-trial raster: fresh ping at ei_strength with W_ei/W_ie skipped on load
    (same transfer-load as run_inproc_infer), via `sim --infer --skip-load ...
    --sample-index`. Reads spk_e/spk_i + label from snapshot.npz."""
    cfg = json.loads((train_dir / "config.json").read_text())
    out_dir = (
        ARTIFACTS
        / "ei_raster"
        / f"seed{seed}"
        / f"ei{ei_strength:g}_s{sample_idx}"
        / cache_tag(resolve_checkpoint(train_dir, CHECKPOINT_ROLE))
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cli(
        [
            "sim", "--infer",
            "--load-config", str((train_dir / "config.json").resolve()),
            "--load-weights", str(checkpoint_path(train_dir)),
            "--ei-strength", str(ei_strength),
            "--skip-load", "W_ei.", "W_ie.",
            "--max-samples", str(EVAL_MAX_SAMPLES),
            "--sample-index", str(sample_idx),
            "--out-dir", str(out_dir),
        ]
    )
    d = np.load(out_dir / "snapshot.npz")
    e_full, i_full = d["spk_e"], d["spk_i"]
    if e_full.ndim == 3:
        e_full = e_full[:, 0, :]
    if i_full.ndim == 3:
        i_full = i_full[:, 0, :]
    rng = np.random.default_rng(0)
    e_idx = np.sort(rng.choice(e_full.shape[1], EI_RASTER_N_E_PLOT, replace=False))
    i_idx = np.sort(rng.choice(i_full.shape[1], EI_RASTER_N_I_PLOT, replace=False))
    return {
        "seed": seed,
        "ei_strength": float(ei_strength),
        "label": int(d["label"]),
        "e": e_full[:, e_idx].astype(bool),
        "i": i_full[:, i_idx].astype(bool),
        "dt": float(cfg["dt"]),
        "t_ms": float(cfg["t_ms"]),
    }


def capture_rate_raster(train_dir: Path, spike_rate: float, sample_idx: int) -> dict:
    """Single-trial raster of the trained ping baseline at a given input rate, via
    `sim --infer --input-rate R --sample-index N` (input-rate sets M.max_rate_hz).
    Reads spk_e/spk_i + label from snapshot.npz."""
    cfg = json.loads((train_dir / "config.json").read_text())
    out_dir = (
        ARTIFACTS / "rate_raster" / f"r{spike_rate:g}_s{sample_idx}"
        / cache_tag(resolve_checkpoint(train_dir, CHECKPOINT_ROLE))
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cli(
        [
            "sim", "--infer",
            "--load-config", str((train_dir / "config.json").resolve()),
            "--load-weights", str(checkpoint_path(train_dir)),
            "--input-rate", str(spike_rate),
            "--sample-index", str(sample_idx),
            "--out-dir", str(out_dir),
        ]
    )
    d = np.load(out_dir / "snapshot.npz")
    e_full, i_full = d["spk_e"], d["spk_i"]
    if e_full.ndim == 3:
        e_full = e_full[:, 0, :]
    if i_full.ndim == 3:
        i_full = i_full[:, 0, :]
    t_sec = float(cfg["t_ms"]) / 1000.0
    e_rate_hz = float(e_full.sum() / (e_full.shape[1] * t_sec))
    i_rate_hz = float(i_full.sum() / (i_full.shape[1] * t_sec)) if i_full.shape[1] else 0.0
    rng = np.random.default_rng(0)
    e_idx = np.sort(rng.choice(e_full.shape[1], EI_RASTER_N_E_PLOT, replace=False))
    i_idx = np.sort(rng.choice(i_full.shape[1], EI_RASTER_N_I_PLOT, replace=False))
    return {
        "spike_rate": float(spike_rate),
        "e_rate_hz": e_rate_hz,
        "i_rate_hz": i_rate_hz,
        "label": int(d["label"]),
        "e": e_full[:, e_idx].astype(bool),
        "i": i_full[:, i_idx].astype(bool),
        "dt": float(cfg["dt"]),
        "t_ms": float(cfg["t_ms"]),
    }


def plot_rate_rasters(samples: list[dict], out_path: Path, run_id: str) -> None:
    """One row per input-rate value; same E-over-I stacked layout as
    plot_ei_rasters so the two figures are visually comparable."""
    theme.apply()
    n = len(samples)
    n_e = EI_RASTER_N_E_PLOT
    n_i = EI_RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n, 1, figsize=(5.6, 3.15),
        sharex=True, gridspec_kw={"hspace": 0.18},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples)):
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt"]
        e_t, e_n = np.where(s["e"])
        i_t, i_n = np.where(s["i"])
        ax.scatter(
            t_axis[e_t], e_n,
            s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.4,
        )
        ax.scatter(
            t_axis[i_t], i_n + n_e + gap,
            s=2.0, c=theme.DEEP_RED, marker="|", linewidths=0.4,
        )
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, s["t_ms"])
        i_rate_str = (
            f"\nI = {s['i_rate_hz']:.1f} Hz" if "i_rate_hz" in s else ""
        )
        ax.text(
            1.012, 0.5,
            f"input = {s['spike_rate']:.1f} Hz\nE = {s['e_rate_hz']:.1f} Hz" + i_rate_str,
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=theme.SIZE_ANNOTATION,
        )
        if i == 0:
            ax.set_title(
                "E (black) and I (red) spikes — trained ping, MNIST digit 0, "
                "input-rate sweep"
            )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


FI_UNIFORM_RATES_HZ: list[float] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0,
    50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
]
FI_UNIFORM_ZOOM_RATES_HZ: list[float] = [round(r, 2) for r in np.linspace(0.0, 10.0, 101)]
FI_UNIFORM_BATCH: int = 32  # batch of uniform-1 inputs per rate; average over.
if SMOKE:
    FI_UNIFORM_RATES_HZ = [0.0, 10.0, 100.0]
    FI_UNIFORM_ZOOM_RATES_HZ = [0.0, 5.0, 10.0]
    FI_UNIFORM_BATCH = 2


def run_fi_sweep_uniform(notebook_run_id: str, rates: list[float] | None = None) -> list[dict]:
    """Population f-I curves on trained PING and COBA baselines with spatially
    uniform Poisson input, via `cli.py probe --load-weights --input-rate R`. For
    each rate, averages per-cell E/I firing over FI_UNIFORM_BATCH trials."""
    if rates is None:
        rates = FI_UNIFORM_RATES_HZ
    rows: list[dict] = []
    for model in MODELS:
        train_dir = baseline_dir(model)
        if not (train_dir / "weights.pth").exists():
            print(f"  [fi-uniform] skip {model} (no weights)")
            continue
        for rate in rates:
            out_dir = (ARTIFACTS / "fi_uniform" / f"{model}_r{rate:g}").resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            run_cli(
                [
                    "sim",
                    "--input", "synthetic-spikes",
                    "--load-config", str((train_dir / "config.json").resolve()),
                    "--load-weights", str(checkpoint_path(train_dir)),
                    "--n-in", "784",
                    "--input-rate", str(rate),
                    "--n-batch", str(FI_UNIFORM_BATCH),
                    "--out-dir", str(out_dir),
                ]
            )
            m = json.loads((out_dir / "metrics.json").read_text())
            rows.append({
                "model": model,
                "input_rate_hz": float(rate),
                "e_rate_hz": float(m["rate_e_hz"]),
                "i_rate_hz": float(m["rate_i_hz"]),
            })
        e_all = [r["e_rate_hz"] for r in rows if r["model"] == model]
        if e_all:
            print(f"  {model:<5} {len(rates)} rates; E range {min(e_all):.2f}-{max(e_all):.2f} Hz")
    return rows


def plot_fi_curve_uniform(
    rows: list[dict], out_path: Path, run_id: str,
    zoom_rows: list[dict] | None = None,
) -> None:
    """Two-panel f-I figure under spatially uniform Poisson input:
    COBA (E + I) on the left, PING (E + I) on the right. If `zoom_rows`
    is provided, a third panel below adds the 0-10 Hz zoom overlaying
    both models' E curves to expose the recruitment cliff."""
    theme.apply()
    if zoom_rows is None:
        fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.1))
        top_axes = list(axes)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.2))
        top_axes = list(axes[0])
    titles = {"ping": "PING (I-loop active)", "coba": "COBA (no I-loop)"}
    for ax, model in zip(top_axes, MODELS):
        msel = sorted(
            [r for r in rows if r["model"] == model],
            key=lambda r: r["input_rate_hz"],
        )
        xs = [r["input_rate_hz"] for r in msel]
        e_ys = [r["e_rate_hz"] for r in msel]
        i_ys = [r["i_rate_hz"] for r in msel]
        ax.plot(xs, e_ys, marker="o", color=theme.INK_BLACK, lw=1.5, label="E")
        ax.plot(xs, i_ys, marker="s", color=theme.DEEP_RED, lw=1.5, label="I")
        if model == "ping":
            ax.plot(xs, [e + i for e, i in zip(e_ys, i_ys)],
                    marker="^", color=theme.AMBER, lw=1.5, ls="--",
                    label="E + I")
        ax.set_xlabel("Input Poisson rate (Hz, per channel)",
                      fontsize=theme.SIZE_LABEL)
        ax.set_ylabel("Per-cell firing rate (Hz)", fontsize=theme.SIZE_LABEL)
        ax.set_title(titles[model], fontsize=theme.SIZE_TITLE)
        ax.legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="upper left")

    # share the y-axis across the two panels so COBA's saturation and PING's
    # compression are read on one scale (the whole point of the comparison)
    top_max = max(
        (max(r["e_rate_hz"], r["i_rate_hz"],
             r["e_rate_hz"] + r["i_rate_hz"] if r["model"] == "ping" else 0.0)
         for r in rows),
        default=1.0,
    )
    for ax in top_axes:
        ax.set_ylim(0, top_max * 1.05)

    if zoom_rows is not None:
        # Bottom row: zoom 0-10 Hz, one panel per model, same scheme
        # as the top row.
        for ax, model in zip(axes[1], MODELS):
            msel = sorted(
                [r for r in zoom_rows if r["model"] == model],
                key=lambda r: r["input_rate_hz"],
            )
            xs = [r["input_rate_hz"] for r in msel]
            e_ys = [r["e_rate_hz"] for r in msel]
            i_ys = [r["i_rate_hz"] for r in msel]
            ax.plot(xs, e_ys, color=theme.INK_BLACK, lw=1.5, label="E")
            ax.plot(xs, i_ys, color=theme.DEEP_RED, lw=1.5, label="I")
            if model == "ping":
                ax.plot(xs, [e + i for e, i in zip(e_ys, i_ys)],
                        color=theme.AMBER, lw=1.5, ls="--", label="E + I")
            ax.set_xlabel("Input Poisson rate (Hz, per channel)",
                          fontsize=theme.SIZE_LABEL)
            ax.set_ylabel("Per-cell firing rate (Hz)", fontsize=theme.SIZE_LABEL)
            ax.set_title(
                f"{titles[model]} — 0–10 Hz zoom",
                fontsize=theme.SIZE_TITLE,
            )
            ax.set_xlim(0, 10)
            ax.legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="upper left")

    fig.suptitle(
        "Population f-I curves: trained PING and COBA, uniform Poisson input",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    save_figure(fig, out_path)  # line/curve plot: SVG + PDF
    plt.close(fig)


def plot_fi_curve(samples: list[dict], out_path: Path, run_id: str) -> None:
    """f-I curve from the same data that plot_rate_rasters consumed.
    x-axis: input Poisson rate (Hz, per channel). y-axis: per-cell mean
    firing rate of E (black) and I (red) populations across the trial."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(5.6, 3.15))
    xs = [s["spike_rate"] for s in samples]
    e_ys = [s["e_rate_hz"] for s in samples]
    i_ys = [s["i_rate_hz"] for s in samples]
    ax.plot(xs, e_ys, marker="o", color=theme.INK_BLACK, lw=1.5, label="E")
    ax.plot(xs, i_ys, marker="s", color=theme.DEEP_RED, lw=1.5, label="I")
    ax.set_xlabel("Input Poisson rate (Hz, per channel)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Per-cell firing rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.legend(fontsize=theme.SIZE_LABEL, frameon=False)
    fig.suptitle(
        "Trained PING f-I curve (MNIST digit 0)",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    save_figure(fig, out_path)  # line/curve plot: SVG + PDF
    plt.close(fig)


PERTURB_DROP_LEVELS: list[float] = [round(x * 0.1, 2) for x in range(11)]  # 0.0..1.0
PERTURB_ADD_LEVELS: list[float] = [float(x) for x in range(0, 41, 2)]  # 0..40 Hz, 2 Hz steps
PERTURB_RASTER_DROP_LEVELS: list[float] = [0.0, 0.3, 0.6, 0.8, 0.9, 1.0]
PERTURB_RASTER_ADD_LEVELS: list[float] = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0]


def plot_ei_rasters(samples: list[dict], out_path: Path, run_id: str) -> None:
    """One row per ei value; I units stack over E units so the PING-style
    E-then-I cadence reads as alternating bursts when it appears."""
    theme.apply()
    n = len(samples)
    n_e = EI_RASTER_N_E_PLOT
    n_i = EI_RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n, 1, figsize=(5.6, 3.15),
        sharex=True, gridspec_kw={"hspace": 0.18},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples)):
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt"]
        e_t, e_n = np.where(s["e"])
        i_t, i_n = np.where(s["i"])
        ax.scatter(
            t_axis[e_t], e_n,
            s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.4,
        )
        ax.scatter(
            t_axis[i_t], i_n + n_e + gap,
            s=2.0, c=theme.DEEP_RED, marker="|", linewidths=0.4,
        )
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, s["t_ms"])
        ax.text(
            1.012, 0.5, f"ei = {s['ei_strength']:g}",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=theme.SIZE_LABEL,
        )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


def plot_ei_acc_sweep(points: list[dict], out_path: Path, run_id: str) -> None:
    theme.apply()
    eis = [p["ei_strength"] for p in points]
    accs = [p["acc"] for p in points]
    base_acc = points[0]["acc"]
    worst = min(points, key=lambda p: p["acc"])
    y_hi = min(max(accs + [base_acc]) + 6, 100)
    fig, ax = plt.subplots(figsize=(5.6, 3.15))
    ax.axhline(
        base_acc, color=theme.LABEL, lw=1.0, ls="--",
        label=f"baseline {base_acc:.1f}%",
    )
    ax.axhline(
        10.0, color=theme.FAINT, lw=1.0, ls=":", label="chance (10%)",
    )
    ax.plot(eis, accs, marker="o", color=theme.DEEP_RED, label="transfer")
    ax.annotate(
        f"{worst['acc']:.1f}%  (Δ {worst['acc'] - base_acc:+.1f} pp)",
        xy=(worst["ei_strength"], worst["acc"]),
        xytext=(8, -14), textcoords="offset points",
        fontsize=theme.SIZE_ANNOTATION,
    )
    ax.set_xlabel("inference E→I strength")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Transfer accuracy across the I-loop sweep")
    ax.set_ylim(0, y_hi)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([round(0.1 * i, 1) for i in range(11)])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)  # line/curve plot: SVG + PDF
    plt.close(fig)


def plot_ei_rates_sweep(points: list[dict], out_path: Path, run_id: str) -> None:
    theme.apply()
    eis = [p["ei_strength"] for p in points]
    hid = [p.get("hid_rate_hz") or 0.0 for p in points]
    inh = [p.get("inh_rate_hz") or 0.0 for p in points]
    fig, ax = plt.subplots(figsize=(5.6, 3.15))
    ax.plot(eis, hid, marker="o", color=theme.INK_BLACK, label="E (hidden)")
    ax.plot(eis, inh, marker="s", color=theme.DEEP_RED, label="I (inhibitory)")
    ax.set_xlabel("inference E→I strength")
    ax.set_ylabel("mean population rate (Hz)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    save_figure(fig, out_path)  # line/curve plot: SVG + PDF
    plt.close(fig)


def run_ei_sweep(notebook_run_id: str) -> list[dict]:
    """Run the inference E→I sweep across all trained COBA seeds."""
    points: list[dict] = []
    for seed in SEEDS_BASELINE:
        train_dir = baseline_dir("coba", seed)
        if not (train_dir / "weights.pth").exists():
            raise SystemExit(f"ei-sweep needs trained COBA weights at {train_dir}")
        sweep_root = _ei_sweep_dir(seed)
        sweep_root.mkdir(parents=True, exist_ok=True)
        for ei in EI_SWEEP:
            out = sweep_root / f"infer_ei{ei:g}"
            print(f"[ei-sweep] seed={seed} ei={ei} → {out}")
            m = run_inproc_infer(train_dir, ei, out)
            points.append(
                {
                    "seed": seed,
                    "ei_strength": ei,
                    "acc": m["best_acc"],
                    "hid_rate_hz": m.get("hid_rate_hz"),
                    "inh_rate_hz": m.get("inh_rate_hz"),
                    "n_total": m.get("n_total"),
                }
            )

    illustrative_seed = SEEDS_BASELINE[0]
    train_dir = baseline_dir("coba", illustrative_seed)
    print(
        f"[ei-sweep] capturing seed-{illustrative_seed} illustrative rasters "
        f"for ei ∈ {EI_RASTER}"
    )
    raster_samples = [
        capture_ei_raster(
            train_dir, ei, EI_RASTER_SAMPLE_IDX, seed=illustrative_seed
        )
        for ei in EI_RASTER
    ]

    plot_ei_rasters(raster_samples, FIGURES / "ei_rasters", notebook_run_id)
    print(f"wrote {FIGURES / 'ei_rasters'}.{{png,pdf}}")
    # Compound (Figure 1): the rate/accuracy sweeps fold into it, so the two
    # standalone sweep plots are no longer emitted.
    fig_loop_transfer_compound(
        points, raster_samples[0], raster_samples[-1],
        FIGURES / "loop_transfer_compound", notebook_run_id)
    print(f"wrote {FIGURES / 'loop_transfer_compound'}.{{png,pdf}}")
    return points


# ── End ei sweep ────────────────────────────────────────────────────


# ── tau_GABA sweep (inference-only, trained ping) ───────────────────

TAU_GABA_VALUES: list[float] = [4.5, 6.0, 9.0, 12.0, 18.0, 27.0]  # ms; default 9.0




def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def summarize_ei_points(points: list[dict]) -> list[dict]:
    """Aggregate the E→I sweep across independently trained seeds."""
    summary = []
    for ei in sorted({float(point["ei_strength"]) for point in points}):
        rows = [point for point in points if float(point["ei_strength"]) == ei]
        row = {"ei_strength": ei}
        for field in ("acc", "hid_rate_hz", "inh_rate_hz"):
            values = np.asarray(
                [float(point.get(field) or 0.0) for point in rows], dtype=float
            )
            row[field] = float(values.mean())
            row[f"{field}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        summary.append(row)
    return summary


def fig_loop_transfer_compound(points, raster_lo, raster_hi, out_path, run_id):
    """Claim-5 anchor: switching the I-loop on at inference (ei 0→1) on a
    trained COBA cuts the hidden-E rate ~10× at matched accuracy — the gating
    is architectural, not learned. Top: single-trial rasters at ei = 0 and
    ei = 1. Bottom: the ei sweep — E/I rate (left) and accuracy (right)."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"  # keep the saved 16:9 exact
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(6.9, 3.88))  # 16:9, full text width
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3.0, 2.6],
                  hspace=0.5, wspace=0.22, top=0.93, bottom=0.1, left=0.07, right=0.96)

    n_e, n_i, gap = EI_RASTER_N_E_PLOT, EI_RASTER_N_I_PLOT, 6
    for col, s in enumerate((raster_lo, raster_hi)):
        ax = fig.add_subplot(gs[0, col])
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt"]
        e_t, e_n = np.where(s["e"])
        i_t, i_n = np.where(s["i"])
        ax.scatter(t_axis[e_t], e_n, s=1.6, c=theme.INK_BLACK, marker="|", linewidths=0.4)
        ax.scatter(t_axis[i_t], i_n + n_e + gap, s=1.6, c=theme.DEEP_RED, marker="|", linewidths=0.4)
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, s["t_ms"])
        ax.set_xlabel("time (ms)")
        tag = "loop off (COBA)" if s["ei_strength"] == 0 else "loop on (PING)"
        ax.set_title(f"ei = {s['ei_strength']:g}  —  {tag}", loc="left", fontweight="semibold")
        _despine(ax)

    summary = summarize_ei_points(points)

    eis = np.asarray([p["ei_strength"] for p in summary])
    ax_r = fig.add_subplot(gs[1, 0])
    hid = np.asarray([p["hid_rate_hz"] for p in summary])
    inh = np.asarray([p["inh_rate_hz"] for p in summary])
    hid_sd = np.asarray([p["hid_rate_hz_sd"] for p in summary])
    inh_sd = np.asarray([p["inh_rate_hz_sd"] for p in summary])
    ax_r.plot(eis, hid, marker="o", ms=3, color=theme.INK_BLACK, label="E (hidden)")
    ax_r.plot(eis, inh, marker="s", ms=3, color=theme.DEEP_RED, label="I")
    ax_r.fill_between(eis, hid - hid_sd, hid + hid_sd,
                      color=theme.INK_BLACK, alpha=0.15, linewidth=0)
    ax_r.fill_between(eis, inh - inh_sd, inh + inh_sd,
                      color=theme.DEEP_RED, alpha=0.15, linewidth=0)
    ax_r.set_xlabel("inference E→I strength")
    ax_r.set_ylabel("rate (Hz)")
    ax_r.legend(fontsize=theme.SIZE_LEGEND, frameon=False)
    _despine(ax_r)

    ax_a = fig.add_subplot(gs[1, 1])
    accs = np.asarray([p["acc"] for p in summary])
    acc_sds = np.asarray([p["acc_sd"] for p in summary])
    base_acc = summary[0]["acc"]
    ax_a.axhline(base_acc, color=theme.LABEL, lw=1.0, ls="--",
                 label=f"COBA baseline {base_acc:.0f}%")
    ax_a.plot(eis, accs, marker="o", ms=3, color=theme.DEEP_RED, label="transfer")
    ax_a.fill_between(eis, accs - acc_sds, accs + acc_sds,
                      color=theme.DEEP_RED, alpha=0.15, linewidth=0)
    ax_a.set_ylim(0, 100)
    ax_a.set_xlabel("inference E→I strength")
    ax_a.set_ylabel("test accuracy (%)")
    ax_a.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="lower left")
    _despine(ax_a)

    stamp_figure(fig, run_id)
    # Compound contains dense single-trial raster panels: rasterise as PNG, not SVG.
    save_figure(fig, out_path, formats=("png", "pdf"))
    plt.close(fig)


def _load_cached_ei_points() -> list[dict]:
    """Read the ei-sweep results from the previous run's cached metrics.json —
    no inference. Raises with a clear message if the cache is absent."""
    points: list[dict] = []
    for seed in SEEDS_BASELINE:
        sweep_root = _ei_sweep_dir(seed)
        for ei in EI_SWEEP:
            mfile = sweep_root / f"infer_ei{ei:g}" / "metrics.json"
            if not mfile.exists():
                raise SystemExit(
                    f"--replot needs cached ei-sweep data at {mfile.parent}; "
                    "run the notebook once without --replot first."
                )
            m = json.loads(mfile.read_text())
            rates_hz = m.get("rates_hz", {})
            hid = next((k for k in rates_hz if k.startswith("hid")), None)
            inh = next((k for k in rates_hz if k.startswith("inh")), None)
            points.append({
                "seed": seed,
                "ei_strength": ei,
                "acc": float(m["best_acc"]),
                "hid_rate_hz": rates_hz.get(hid) if hid else None,
                "inh_rate_hz": rates_hz.get(inh) if inh else None,
                "n_total": int(m.get("n_total", 0)),
            })
    return points


def _load_cached_ei_raster(
    ei_strength: float, sample_idx: int, *, seed: int
) -> dict:
    """Read a single-trial raster from the previous run's cached snapshot.npz —
    no inference. Mirrors capture_ei_raster's parsing without the sim call."""
    cfg = json.loads((baseline_dir("coba", seed) / "config.json").read_text())
    snap = (
        ARTIFACTS
        / "ei_raster"
        / f"seed{seed}"
        / f"ei{ei_strength:g}_s{sample_idx}"
        / "snapshot.npz"
    )
    if not snap.exists():
        raise SystemExit(
            f"--replot needs cached raster at {snap}; "
            "run the notebook once without --replot first."
        )
    d = np.load(snap)
    e_full, i_full = d["spk_e"], d["spk_i"]
    if e_full.ndim == 3:
        e_full = e_full[:, 0, :]
    if i_full.ndim == 3:
        i_full = i_full[:, 0, :]
    rng = np.random.default_rng(0)
    e_idx = np.sort(rng.choice(e_full.shape[1], EI_RASTER_N_E_PLOT, replace=False))
    i_idx = np.sort(rng.choice(i_full.shape[1], EI_RASTER_N_I_PLOT, replace=False))
    return {
        "seed": seed,
        "ei_strength": float(ei_strength),
        "label": int(d["label"]),
        "e": e_full[:, e_idx].astype(bool),
        "i": i_full[:, i_idx].astype(bool),
        "dt": float(cfg["dt"]),
        "t_ms": float(cfg["t_ms"]),
    }


def replot_figures(run_id: str = "replot") -> None:
    """Regenerate the two published figures (ei_rasters + loop_transfer_compound)
    from the previous run's cached inference outputs — no inference, no training.
    Use when only the figure rendering changed (labels, style, layout); a full run
    is only needed when the underlying numbers change."""
    points = _load_cached_ei_points()
    illustrative_seed = SEEDS_BASELINE[0]
    raster_samples = [
        _load_cached_ei_raster(
            ei, EI_RASTER_SAMPLE_IDX, seed=illustrative_seed
        )
        for ei in EI_RASTER
    ]
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_ei_rasters(raster_samples, FIGURES / "ei_rasters", run_id)
    print(f"wrote {FIGURES / 'ei_rasters'}.{{png,pdf}}")
    fig_loop_transfer_compound(
        points, raster_samples[0], raster_samples[-1],
        FIGURES / "loop_transfer_compound", run_id)
    print(f"wrote {FIGURES / 'loop_transfer_compound'}.{{png,pdf}}")


def main() -> None:
    # Publication profile: every figure this notebook writes is a print-sized
    # vector, emitted as both SVG (docs) and PDF (manuscript) by save_figure.
    theme.set_paper_mode(True)

    # `--replot <name>` re-renders the published figures from cached inference
    # outputs and exits — no training, no inference. For regenerating figures after
    # a style/label change; a full run is only needed when the numbers change.
    if replot_target(sys.argv) is not None:
        replot_figures()
        return
    meta = parse_meta(sys.argv)

    t_start = time.monotonic()
    run_id = next_run_id(SLUG)
    n_cells = len(MODELS) * len(RATE_TARGET_GRID_HZ) * len(SEEDS_BASELINE)
    print(
        f"notebook_run_id = {run_id} cells={n_cells}"
        + ("  [skip-training]" if meta.skip_training else "")
    )
    _log_event("started", run_id=run_id)

    # Training lives in exp022 now (train-once / reuse-many). This notebook
    # consumes the shared cells via cell_dir → exp022.load_cell. Atomic publish:
    # everything lands in `figures` (a staging dir) and swaps into place only if
    # the run completes.
    with published_run(
        SLUG, run_id, skip_training=meta.skip_training, make_artifacts=False,
        scale=SCALE, plot_only=meta.plot_only,
    ) as (_artifacts, figures):
        global FIGURES
        FIGURES = figures   # atomic-publish: point the module path at the staging dir for this run

        rows: list[dict] = []
        for model in MODELS:
            for rate_target_hz in RATE_TARGET_GRID_HZ:
                for seed in seeds_for(rate_target_hz):
                    run_dir = cell_dir(model, rate_target_hz, seed)
                    if not (run_dir / "metrics.json").exists():
                        raise SystemExit(f"missing metrics: {run_dir / 'metrics.json'}")
                    metrics = load_metrics(run_dir)
                    last = metrics["epochs"][-1]
                    rows.append(
                        {
                            "cell_name": cell_name(model, rate_target_hz, seed),
                            "model": model,
                            "rate_target_display": rate_target_display(rate_target_hz),
                            "rate_target_hz": rate_target_hz,
                            "seed": seed,
                            "best_acc": float(metrics["best_acc"]),
                            "best_epoch": int(metrics["best_epoch"]),
                            "final_acc": float(last["acc"]),
                            "rate_e": float(last.get("rate_e") or 0.0),
                        }
                    )

        print("  results:")
        for r in rows:
            theta_str = (
                f"rate target={r['rate_target_display']:>4} ({r['rate_target_hz']:>4.1f} Hz)"
                if r["rate_target_hz"] is not None
                else "rate target= off"
            )
            print(
                f"    {r['model']:<5}  {theta_str}  "
                f"acc(final)={r['final_acc']:6.2f}%  best={r['best_acc']:6.2f}%  "
                f"rate_e={r['rate_e']:6.1f} Hz"
            )

        # Stacked raster snapshot at the first 10 frames of the rate sweep —
        # same panel style as the ei-sweep rasters so the two read as a pair.
        rate_grid = (
            np.asarray([0.0, 10.0, 100.0])
            if SMOKE else np.linspace(0.0, 100.0, 40)[:10]
        )
        print(f"[rate-rasters] capturing rates {[round(r, 2) for r in rate_grid]}")
        rate_samples = [
            capture_rate_raster(baseline_dir("ping"), float(r), sample_idx=0)
            for r in rate_grid
        ]
        plot_rate_rasters(
            rate_samples, figures / "rate_rasters__ping", run_id
        )
        print(f"wrote {figures / 'rate_rasters__ping'}.{{png,pdf}}")
        plot_fi_curve(rate_samples, figures / "fi_curve__ping", run_id)
        print(f"wrote {figures / 'fi_curve__ping'}.{{svg,pdf}}")

        # Uniform-input f-I curves for PING and COBA — no MNIST structure.
        print("[fi-sweep] uniform Poisson input on trained PING and COBA (wide)")
        fi_rows = run_fi_sweep_uniform(run_id)
        plot_fi_curve_uniform(
            fi_rows, figures / "fi_curve_uniform", run_id,
        )
        print(f"wrote {figures / 'fi_curve_uniform'}.{{svg,pdf}}")

        # EI-strength sweep (subsumes nb019): replay COBA-trained weights
        # with progressively stronger I-loop.
        ei_points = run_ei_sweep(run_id)

        duration_s = time.monotonic() - t_start
        train_cfg = load_config(baseline_dir(MODELS[0]))
        write_numbers(
            figures, run_id=run_id, duration_s=duration_s,
            payload={
                "git_sha_train": train_cfg.get("git_sha"),
                "checkpoint_provenance": checkpoint_provenance(
                    [
                        cell_dir(model, target, seed)
                        for model in MODELS
                        for target in RATE_TARGET_GRID_HZ
                        for seed in SEEDS_BASELINE
                    ],
                    CHECKPOINT_ROLE,
                ),
                "config": {
                    "dataset": "mnist",
                    "models": MODELS,
                    "rate_target_grid_hz": [
                        rate_target_hz_value(t) for t in RATE_TARGET_GRID_HZ if t is not None
                    ],
                    "max_samples": MAX_SAMPLES,
                    "evaluation_pool_samples": EVAL_CORPUS_SAMPLES,
                    "epochs": BASELINE_EPOCHS,
                    "t_ms": T_MS,
                    "dt": DT_TRAIN,
                    "frontier_seeds": SEEDS_BASELINE,
                    "quantitative_inference_seeds": SEEDS_BASELINE,
                    "illustrative_raster_seed": SEEDS_BASELINE[0],
                    "evaluation_samples_per_seed": sorted(
                        {int(point["n_total"]) for point in ei_points}
                    ),
                    "fr_strength_upper": FR_STRENGTH_UPPER,
                },
                "baseline_results": rows,
                "frontier_summary": summarize_frontier(rows),
                "ei_sweep": ei_points,
                "ei_sweep_summary": summarize_ei_points(ei_points),
                "fi_sweep_uniform": fi_rows,
            },
        )
        print(f"wrote {figures / 'numbers.json'}")
        _log_event("completed", run_id=run_id, quantitative_rows=len(ei_points))



if __name__ == "__main__":
    main()
