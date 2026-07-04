"""Notebook runner for entry 044 — PING Δt audit.

Trains PING from scratch at five integration timesteps Δt ∈ {0.05, 0.1,
0.25, 0.5, 1.0} ms, holding total physical time T = 200 ms constant.
Measures post-training mean E rate (Hz) and accuracy per cell. Plots
the sweep on a log-Δt axis and grabs single-trial rasters at each Δt
to verify the gamma cycle stays at the same physical period in ms
(rather than the same step count).

Scaffolded by ar009 §Leg 1 item 4. Protects the ≈ 7 Hz headline rate
in ar008 against a discretisation-artefact reading.

Writing: writings/nb044.typ · figures + numbers.json: artifacts/data/nb044/
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import theme  # noqa: E402
from helpers.figsave import save_figure  # noqa: E402
from helpers.fmt import format_duration  # noqa: E402
from helpers.modal import parse_modal_gpu  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402

SLUG = "nb044"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
SNN_TOOL = REPO / "tools" / "snn" / "tool.py"

T_MS = 200.0

# Δt sweep — spans 20× across the integration timescale. nb025 trained
# at Δt = 0.1 ms; nb040 trained CUBA-PING at Δt = 1.0 ms. This audit
# trains the same recipe across the full range to verify the rate
# ceiling is a physical (Hz) feature rather than a step-count artefact.
DT_SWEEP_MS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0)
SEEDS: tuple[int, ...] = (42, 43, 44)

# Batch size is held at 64 across all Δt so per-step compute and memory
# stay comparable, and the Δt = 0.05 cells (4000 timesteps × N_E × N_I)
# fit in a single A100. nb025 used 256; smaller batches are slower but
# the recipe still trains.
BATCH_SIZE: int = 64

# Single-trial raster capture — same convention as nb025 / nb042.
RASTER_SAMPLE_IDX: int = 0
RASTER_N_E_PLOT: int = 200
RASTER_N_I_PLOT: int = 64
RASTER_T_WINDOW_MS: float = 100.0  # show first 100 ms so the cycle is visible

# Baked run scale (the retired "small" tier).
MAX_SAMPLES: int = 500
EPOCHS: int = 10

# Run scale — stamped into the manifest by run_dirs.prepare and rendered as
# the Methods table via RunScale; the mdx never restates these numbers.
SCALE = {
    "dataset": "mnist",
    "max_samples": MAX_SAMPLES,
    "epochs": EPOCHS,
    "t_ms": T_MS,
    "batch_size": BATCH_SIZE,
    "seeds": len(SEEDS),
    "cells": len(DT_SWEEP_MS) * len(SEEDS),
    "grid": "5 Δt × 3 seeds (Δt ∈ {0.05, 0.1, 0.25, 0.5, 1.0} ms)",
}

def dt_label(dt_ms: float) -> str:
    s = f"{dt_ms:g}".replace(".", "p")
    return f"dt{s}"


def cell_dir(dt_ms: float, seed: int) -> Path:
    """Trained cell — now the shared nb022 cell (train-once / reuse-many)."""
    from nb022 import cell_dir as shared_cell_dir
    return shared_cell_dir(f"ping__{dt_label(dt_ms)}__seed{seed}")


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


# ─── inference: rate, accuracy, raster ──────────────────────────────


def _infer_cell(train_dir: Path, extra_args: list[str], out_name: str) -> Path:
    """Shell out to the CLI's `sim --infer` for one trained cell; return the out dir.

    Network construction, weight loading and the forward pass all happen inside the
    CLI — this notebook only runs it and reads the artifacts it writes. extra_args
    tacks on mode-specific flags (e.g. --sample-index for a snapshot).
    """
    out_dir = (ARTIFACTS / out_name / train_dir.name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv", "run", "python", str(SNN_TOOL), "sim", "--infer",
            "--load-config", str((train_dir / "config.json").resolve()),
            "--load-weights", str((train_dir / "weights.pth").resolve()),
            "--out-dir", str(out_dir),
            *extra_args,
        ],
        cwd=REPO,
        check=True,
    )
    return out_dir


def measure_rate_acc(train_dir: Path) -> dict:
    """Accuracy + mean E/I firing rate (Hz) over the test set, via the CLI.

    Reads metrics.json emitted by `sim --infer` (best_acc + per-population
    rates_hz), rather than rebuilding the net and forwarding in-process.
    """
    cfg = json.loads((train_dir / "config.json").read_text())
    out_dir = _infer_cell(train_dir, [], "infer")
    m = json.loads((out_dir / "metrics.json").read_text())
    rates = m.get("rates_hz", {})
    return {
        "dt_ms": float(cfg["dt"]),
        "t_ms": float(cfg["t_ms"]),
        "acc": float(m["best_acc"]),
        "e_rate_hz": float(rates.get("hid", 0.0)),
        "i_rate_hz": float(rates.get("inh", 0.0)),
        "n_total": int(m.get("n_total", 0)),
    }


def capture_raster(train_dir: Path, sample_idx: int) -> dict:
    """Single-trial raster from a trained cell, via the CLI snapshot.

    Runs `sim --infer --sample-index N` (raw test-set index) and reads the
    full E/I rasters from snapshot.npz, then subsamples cells for the plot.
    """
    cfg = json.loads((train_dir / "config.json").read_text())
    out_dir = _infer_cell(train_dir, ["--sample-index", str(sample_idx)], "snapshot")
    d = np.load(out_dir / "snapshot.npz")
    e_full = d["spk_e"]  # (T, N_E)
    i_full = d["spk_i"]  # (T, N_I)
    t_sec = float(cfg["t_ms"]) / 1000.0
    e_rate = float(e_full.sum() / (e_full.shape[1] * t_sec))
    i_rate = float(i_full.sum() / (i_full.shape[1] * t_sec))
    rng = np.random.default_rng(0)
    e_idx = np.sort(rng.choice(e_full.shape[1], RASTER_N_E_PLOT, replace=False))
    i_idx = np.sort(rng.choice(i_full.shape[1], RASTER_N_I_PLOT, replace=False))
    return {
        "dt_ms": float(cfg["dt"]),
        "e": e_full[:, e_idx].astype(bool),
        "i": i_full[:, i_idx].astype(bool),
        "e_rate_hz": e_rate,
        "i_rate_hz": i_rate,
        "t_ms": float(cfg["t_ms"]),
    }


# ─── plotting ───────────────────────────────────────────────────────


def plot_dt_sweep(rows: list[dict], out_path: Path, run_id: str) -> None:
    """E rate (left axis) and accuracy (right axis) vs Δt (log)."""
    theme.apply()
    by_dt: dict[float, list[dict]] = {}
    for r in rows:
        by_dt.setdefault(r["dt_ms"], []).append(r)
    dts_sorted = sorted(by_dt.keys())
    e_means = [
        float(np.mean([r["e_rate_hz"] for r in by_dt[d]])) for d in dts_sorted
    ]
    e_sems = [
        float(np.std([r["e_rate_hz"] for r in by_dt[d]], ddof=1)
              / np.sqrt(max(1, len(by_dt[d]))))
        if len(by_dt[d]) > 1 else 0.0 for d in dts_sorted
    ]
    acc_means = [
        float(np.mean([r["acc"] for r in by_dt[d]])) for d in dts_sorted
    ]
    acc_sems = [
        float(np.std([r["acc"] for r in by_dt[d]], ddof=1)
              / np.sqrt(max(1, len(by_dt[d]))))
        if len(by_dt[d]) > 1 else 0.0 for d in dts_sorted
    ]

    fig, ax_rate = plt.subplots(figsize=(5.6, 3.5))
    ax_rate.errorbar(
        dts_sorted, e_means, yerr=e_sems,
        marker="D", markersize=6, lw=1.4, color=theme.INK_BLACK,
        capsize=3, label="E rate (Hz)",
    )
    ax_rate.set_xscale("log")
    ax_rate.set_xlabel("Δt (ms)", fontsize=theme.SIZE_LABEL)
    ax_rate.set_ylabel("Hidden E rate (Hz)",
                       fontsize=theme.SIZE_LABEL, color=theme.INK_BLACK)
    ax_rate.tick_params(axis="y", labelcolor=theme.INK_BLACK)
    ax_rate.set_ylim(0, 50)
    ax_rate.set_xticks(dts_sorted)
    ax_rate.set_xticklabels([f"{d:g}" for d in dts_sorted])
    ax_rate.spines["top"].set_visible(False)

    ax_acc = ax_rate.twinx()
    ax_acc.errorbar(
        dts_sorted, acc_means, yerr=acc_sems,
        marker="s", markersize=6, lw=1.4, color=theme.DEEP_RED,
        capsize=3, label="accuracy (%)",
    )
    ax_acc.set_ylabel("Test accuracy (%)",
                      fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED)
    ax_acc.tick_params(axis="y", labelcolor=theme.DEEP_RED)
    ax_acc.set_ylim(0, 100)
    ax_acc.spines["top"].set_visible(False)

    fig.suptitle(
        "PING Δt audit — post-training E rate and accuracy vs integration timestep",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_raster_strip(
    samples: list[dict], out_path: Path, run_id: str, t_window_ms: float,
) -> None:
    """Single-trial rasters across Δt, one panel per Δt. X-axis is
    physical time in ms (not steps), so gamma cycle alignment is read
    by eye — same physical period if dynamics survive Δt change."""
    theme.apply()
    n = len(samples)
    n_e = RASTER_N_E_PLOT
    n_i = RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n, 1, figsize=(6.9, 0.7 * n + 0.8),
        sharex=True, gridspec_kw={"hspace": 0.22},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples)):
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt_ms"]
        # Truncate display to the first t_window_ms ms so cycles are visible.
        mask = t_axis <= t_window_ms
        e_t, e_n = np.where(s["e"][mask])
        i_t, i_n = np.where(s["i"][mask])
        ax.scatter(t_axis[mask][e_t], e_n,
                   s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.4)
        ax.scatter(t_axis[mask][i_t], i_n + n_e + gap,
                   s=2.0, c=theme.DEEP_RED, marker="|", linewidths=0.4)
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, t_window_ms)
        ax.text(
            1.012, 0.5,
            f"Δt = {s['dt_ms']:g} ms\nE = {s['e_rate_hz']:.1f} Hz",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=theme.SIZE_LABEL,
        )
        if i == 0:
            ax.set_title(
                "Trained-PING rasters at each Δt (seed 42, MNIST digit 0 sample 0) — "
                "x-axis is physical time in ms"
            )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


# ─── success criteria ───────────────────────────────────────────────


def plot_training_curves(out_path: Path, run_id: str) -> None:
    """Per-cell training-trajectory curves. One line per (Δt, seed);
    colour by Δt (viridis on log Δt)."""
    theme.apply()
    cmap = plt.get_cmap("viridis")
    dts_sorted = list(DT_SWEEP_MS)
    fig, (ax_acc, ax_rate) = plt.subplots(
        2, 1, figsize=(5.6, 4.6), sharex=True,
        gridspec_kw={"hspace": 0.15},
    )
    for i, dt_ms in enumerate(dts_sorted):
        color = cmap(i / max(1, len(dts_sorted) - 1))
        for j, seed in enumerate(SEEDS):
            mfile = cell_dir(dt_ms, seed) / "metrics.json"
            if not mfile.exists():
                continue
            m = json.loads(mfile.read_text())
            eps = [e["ep"] for e in m["epochs"]]
            accs = [e.get("acc", 0) for e in m["epochs"]]
            rates = [e.get("test_rate_e", 0) for e in m["epochs"]]
            label = f"Δt = {dt_ms:g} ms" if j == 0 else None
            ax_acc.plot(eps, accs, color=color, lw=1.0, alpha=0.85, label=label)
            ax_rate.plot(eps, rates, color=color, lw=1.0, alpha=0.85)
    ax_acc.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax_rate.set_ylabel("Test E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax_rate.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    ax_acc.legend(fontsize=theme.SIZE_LEGEND, frameon=False, ncol=2, loc="lower right")
    ax_acc.set_ylim(0, 100)
    for ax in (ax_acc, ax_rate):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, lw=0.4)
    fig.suptitle(
        "Per-cell training curves — convergence check across Δt sweep",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)

def main() -> None:
    modal_gpu = parse_modal_gpu(sys.argv)
    skip_training = "--skip-training" in sys.argv
    wipe_dir = "--no-wipe-dir" not in sys.argv

    # Publication profile: every figure this notebook writes is a print-sized
    # vector, emitted as both SVG (docs) and PDF (manuscript) by save_figure.
    theme.set_paper_mode(True)

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    n_cells = len(DT_SWEEP_MS) * len(SEEDS)
    print(
        f"notebook_run_id = {notebook_run_id} cells={n_cells}"
        + ("  [skip-training]" if skip_training else "")
        + (f"  [modal:{modal_gpu}]" if modal_gpu else "")
    )

    prepare_run_dirs(
        SLUG, notebook_run_id, wipe=wipe_dir, skip_training=skip_training,
        make_artifacts=False,
        scale=SCALE,
        host=f"modal:{modal_gpu}" if modal_gpu else "local",
    )

    # Training lives in nb022 now (train-once / reuse-many): the dt sweep is a
    # registry family there (the documented dt exception). This notebook only
    # consumes the cells.

    rows: list[dict] = []
    for dt_ms in DT_SWEEP_MS:
        for seed in SEEDS:
            run_dir = cell_dir(dt_ms, seed)
            if not (run_dir / "weights.pth").exists():
                raise SystemExit(f"missing weights: {run_dir / 'weights.pth'}")
            t0 = time.monotonic()
            res = measure_rate_acc(run_dir)
            res["seed"] = seed
            rows.append(res)
            print(
                f"  Δt={dt_ms:>5.2f}ms seed={seed}  "
                f"acc={res['acc']:5.2f}%  E={res['e_rate_hz']:6.2f} Hz  "
                f"I={res['i_rate_hz']:6.2f} Hz  ({time.monotonic() - t0:.1f}s)"
            )

    plot_dt_sweep(rows, FIGURES / "dt_sweep", notebook_run_id)
    print(f"wrote {FIGURES / 'dt_sweep'}.{{svg,pdf}}")

    # Raster strip — one trial per Δt, all from seed 42.
    raster_seed = SEEDS[0]
    print(f"[raster] single-trial panels from seed {raster_seed}, "
          f"sample {RASTER_SAMPLE_IDX}")
    samples = []
    for dt_ms in DT_SWEEP_MS:
        run_dir = cell_dir(dt_ms, raster_seed)
        samples.append(capture_raster(run_dir, RASTER_SAMPLE_IDX))
    plot_raster_strip(
        samples, FIGURES / "raster_strip", notebook_run_id,
        t_window_ms=RASTER_T_WINDOW_MS,
    )
    print(f"wrote {FIGURES / 'raster_strip'}.{{png,pdf}}")
    plot_training_curves(FIGURES / "training_curves", notebook_run_id)
    print(f"wrote {FIGURES / 'training_curves'}.{{svg,pdf}}")

    duration_s = time.monotonic() - t_start
    train_cfg = load_config(cell_dir(DT_SWEEP_MS[0], SEEDS[0]))
    summary = {
        "notebook_run_id": notebook_run_id,
        "git_sha": train_cfg.get("git_sha"),
        "duration_s": round(duration_s, 1),
        "duration": format_duration(duration_s),
        "config": {
            "dataset": "mnist",
            "dt_sweep_ms": list(DT_SWEEP_MS),
            "seeds": list(SEEDS),
            "batch_size": BATCH_SIZE,
            "max_samples": MAX_SAMPLES,
            "epochs": EPOCHS,
            "t_ms": T_MS,
        },
        "results": rows,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")



if __name__ == "__main__":
    main()
