"""Notebook runner for entry 035 — why PING has a rate floor.

Standalone runner with no cross-notebook helpers. Inlines everything
it needs (constants, baseline training, plot fns, sweeps).

For each rung of the biophysical ladder (coba, ping), reads the shared exp022
cells at six hidden-E population-rate ceilings: off (no penalty) plus
{25, 10, 5, 2.5, 1} Hz. The recipe is otherwise fixed, so each
regulariser flag changes — so each (model, rate target) cell is one point on
that model's accuracy / rate Pareto frontier. The unpenalised "off"
cell of each model feeds the foundation figures (acc-vs-rate bar,
learning curves, post-training rasters).

Then runs:
- the low-W_in alternate-schedule sweep (four PING networks
  initialised across the recruitment cliff) showing all four converge
  to PING; and
- the inference-time W_in scale sweep on the heaviest-budget cells
  (PING and COBA at the 1 Hz target) projecting the cliff into the loss
  landscape.

All figures land in /figures/notebooks/exp025/ and the success-criteria
summary in exp025/numbers.json.

Writing: writings/exp025.typ · figures + numbers.json: .artifacts/exp025/
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

from exp022 import (
    FR_STRENGTH_UPPER,  # noqa: E402
    training_run_cell,  # noqa: E402
    training_run_values,  # noqa: E402
)
from exp022 import (
    cell_dir as shared_cell_dir,  # noqa: E402
)
from helpers import theme  # noqa: E402
from helpers.checkpoints import (  # noqa: E402
    cache_tag,
    checkpoint_policy,
    checkpoint_provenance,
    resolve_checkpoint,
)
from helpers.cli import parse_meta  # noqa: E402
from helpers.datasets import MNIST_REDUCED_EVAL_SAMPLES  # noqa: E402
from helpers.figsave import save_figure  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.paths import (  # noqa: E402
    artifacts_and_figures,
    log_runner_event,
    runner_paths,
)
from helpers.run_cli import run_cli  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402

ANALYSIS_PURPOSE = "endpoint_dynamics"
CHECKPOINT_POLICY = checkpoint_policy(ANALYSIS_PURPOSE)
CHECKPOINT_ROLE = CHECKPOINT_POLICY["role"]

SLUG = "exp025"
RUN_PATHS = runner_paths(SLUG)
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

SMOKE = os.environ.get("PINGLAB_SMOKE") == "1"
MAX_SAMPLES = 100 if SMOKE else 7000
EVAL_MAX_SAMPLES = 100 if SMOKE else MNIST_REDUCED_EVAL_SAMPLES
EPOCHS = 2 if SMOKE else 50
T_MS = 200.0
DT_TRAIN = 0.1

# Every quantitative frontier point uses the three independent TR-02 seeds.
# Mechanism-only diagnostics use one explicitly representative checkpoint.
SEEDS: list[int] = list(training_run_values("TR-02", "seed"))
REPRESENTATIVE_SEED: int = 42
LOW_W_IN_SEEDS: list[int] = [REPRESENTATIVE_SEED] if SMOKE else list(SEEDS)

# Inference-time ei_strength sweep on the coba__off__seed42 baseline.
# Subsumes the now-retired nb019 — trains nothing new; just runs the
# already-trained coba weights forward through the test set with a
# fresh ping-arch I-loop at progressively higher ei_strength.
EI_SWEEP: list[float] = [round(0.1 * i, 1) for i in range(11)]  # 0.0–1.0
EI_RASTER: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
EI_RASTER_SAMPLE_IDX: int = 0
EI_RASTER_N_E_PLOT: int = 200
EI_RASTER_N_I_PLOT: int = 64

# Hidden-E population-rate ceiling in Hz. None = no penalty (baseline).
# The grid spans from no
# pressure (off → ~80 Hz coba baseline) down to 1 Hz —
# below ping's natural 5 Hz and into the regime where every model
# loses accuracy.
RATE_TARGET_GRID_HZ: list[float | None] = list(
    training_run_values("TR-02", "rate_target_hz")
)
LOW_W_IN_VALUES = training_run_values("TR-07", "w_in")
LOW_W_IN_RATE_TARGET_HZ: float = 1.0

MODELS = list(training_run_values("TR-02", "model"))


def cell_name(model: str, rate_target_hz: float | None, seed: int) -> str:
    return training_run_cell(
        "TR-02", model=model, rate_target_hz=rate_target_hz, seed=seed
    )["name"]

# Run scale — stamped into the manifest by run_dirs.prepare and rendered as
# the Methods table via RunScale; the mdx never restates these numbers.
SCALE = {
    "dataset": "mnist",
    "max_samples": MAX_SAMPLES,
    "evaluation_samples": EVAL_MAX_SAMPLES,
    "epochs": EPOCHS,
    "t_ms": T_MS,
    "dt_ms": DT_TRAIN,
    "batch_size": 256,
    "seeds": len(SEEDS),
    "cells": (
        len(MODELS) * len(RATE_TARGET_GRID_HZ) * len(SEEDS)
        + len(LOW_W_IN_VALUES) * len(LOW_W_IN_SEEDS)
    ),
    "grid": (
        "shared TR-02: 2 models × 6 targets × 3 seeds; "
        f"local controls: 4 W_in values × {len(LOW_W_IN_SEEDS)} seeds"
    ),
}

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
    return f"rt{s}hz"


def rate_target_display(rate_target_hz: float | None) -> str:
    """Human label for plots / numbers.json."""
    if rate_target_hz is None:
        return "off"
    return f"{rate_target_hz:g}"


def rate_target_hz_value(rate_target_hz: float | None) -> float | None:
    return rate_target_hz


def seeds_for(rate_target_hz: float | None) -> list[int]:
    """Return the independent seeds contributing to every frontier point."""
    return list(SEEDS)


def cell_dir(model: str, rate_target_hz: float | None, seed: int) -> Path:
    """Per-cell artifact directory.

    All cells are seed-suffixed by the shared exp022 registry.
    """
    # exp022 owns the shared training bank; exp025 only analyzes its checkpoints.
    if RUN_PATHS.isolated and not os.environ.get("PINGLAB_TRAINING_ROOT"):
        raise RuntimeError("isolated exp025 requires explicit PINGLAB_TRAINING_ROOT")
    return shared_cell_dir(cell_name(model, rate_target_hz, seed))


def baseline_dir(model: str, seed: int = SEEDS[0]) -> Path:
    return cell_dir(model, None, seed)


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def _baseline_seed_stats(rows: list[dict], model: str) -> tuple[float, float, float, float, int]:
    """Mean/SEM of final-epoch accuracy and rate_e across all seeds for
    one model's rate target = off cells. Returns (acc_mean, acc_sem, rate_mean,
    rate_sem, n_seeds). With n=1, SEM is reported as 0 (no error bar)."""
    baseline_rows = [
        r for r in rows if r["model"] == model and r["rate_target_hz"] is None
    ]
    accs = np.array([r["final_acc"] for r in baseline_rows], dtype=float)
    rates = np.array([r["rate_e"] for r in baseline_rows], dtype=float)
    n = len(accs)
    acc_sem = float(accs.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    rate_sem = float(rates.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return float(accs.mean()), acc_sem, float(rates.mean()), rate_sem, n


def _draw_acc_rate_bars(ax_acc, rows, *, compact: bool = False):
    """Draw the twin-y accuracy/rate bars (rate target = off) into ax_acc and return the
    twin rate axis. Shared by the standalone Figure 2 and the results compound so
    they look identical: coba grey / ping red, accuracy solid + rate hatched."""
    ax_rate = ax_acc.twinx()
    n = len(MODELS)
    xs = np.arange(n)
    width = 0.35
    stats = [_baseline_seed_stats(rows, m) for m in MODELS]
    accs = [s[0] for s in stats]
    acc_sems = [s[1] for s in stats]
    rates = [s[2] for s in stats]
    rate_sems = [s[3] for s in stats]
    n_seeds = stats[0][4] if stats else 0

    # Local grey/red scheme; the two-bars-per-model pairing is told apart by
    # hatch (accuracy solid, rate hatched), not colour.
    bar_colors = {"coba": theme.GREY_MID, "ping": theme.DEEP_RED}
    ax_acc.bar(
        xs - width / 2, accs, width=width,
        color=[bar_colors[m] for m in MODELS],
        edgecolor=theme.INK_BLACK, yerr=acc_sems, ecolor=theme.INK_BLACK, capsize=4,
    )
    ax_rate.bar(
        xs + width / 2, rates, width=width,
        color=[bar_colors[m] for m in MODELS],
        edgecolor=theme.INK_BLACK, hatch="///",
        yerr=rate_sems, ecolor=theme.INK_BLACK, capsize=4,
    )
    lab_fs = theme.SIZE_ANNOTATION - (1 if compact else 0)
    for x, a, ae in zip(xs, accs, acc_sems):
        label = f"{a:.1f}%" if n_seeds <= 1 else f"{a:.1f}±{ae:.1f}%"
        ax_acc.text(x - width / 2, a + ae + 1.5, label, ha="center", va="bottom",
                    fontsize=lab_fs, color=theme.INK_BLACK)
    for x, r, re_ in zip(xs, rates, rate_sems):
        label = f"{r:.1f} Hz" if n_seeds <= 1 else f"{r:.1f}±{re_:.1f} Hz"
        ax_rate.text(x + width / 2, r + re_ + (max(rates) * 0.02 if rates else 0.0),
                     label, ha="center", va="bottom", fontsize=lab_fs, color=theme.INK_BLACK)

    ax_acc.set_xticks(xs)
    ax_acc.set_xticklabels(MODELS)
    ax_acc.set_ylim(0, max(100, max(accs) + 10))
    ax_rate.set_ylim(0, max(rates) * 1.2 if rates else 1.0)
    ax_acc.grid(True, axis="y", alpha=0.3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=theme.PAPER, edgecolor=theme.INK_BLACK),
        plt.Rectangle((0, 0), 1, 1, facecolor=theme.PAPER, edgecolor=theme.INK_BLACK, hatch="///"),
    ]
    ax_acc.legend(handles, ["accuracy (left)", "rate (right)"],
                  loc="upper right", fontsize=theme.SIZE_LEGEND - (1 if compact else 0))
    if compact:
        ax_acc.set_ylabel("test acc (%)")
        ax_rate.set_ylabel("E rate (Hz)")
        ax_acc.set_title("Accuracy vs rate (rate target = off)", loc="left", fontsize=theme.SIZE_LABEL)
    else:
        ax_acc.set_ylabel("test accuracy (%, final epoch)")
        ax_rate.set_ylabel("hidden-E firing rate (Hz, final epoch)")
        title_suffix = (
            f" — accuracy vs firing rate (rate target = off, n={n_seeds} seeds, mean ± SEM)"
            if n_seeds > 1 else " — accuracy vs firing rate (rate target = off)"
        )
        ax_acc.set_title("coba / ping" + title_suffix)
    return ax_rate


def plot_acc_rate_bars(rows: list[dict], out_path: Path, run_id: str) -> None:
    """Twin-y bar chart on the baseline (rate target = off) cells: per model,
    side-by-side bars for accuracy (left y-axis) and mean hidden-E rate
    (right y-axis). Error bars are ±SEM across baseline seeds."""
    theme.apply()
    fig, ax_acc = plt.subplots(figsize=(5.6, 3.15))
    _draw_acc_rate_bars(ax_acc, rows, compact=False)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_learning_curves(out_path: Path, run_id: str) -> None:
    """Train loss + test accuracy per epoch, one curve per model
    (rate target = off). With multiple seeds, plot mean and shade ±SEM."""
    theme.apply()
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(6.9, 3.881))
    n_seeds = len(SEEDS)
    for m in MODELS:
        per_seed_loss: list[list[float]] = []
        per_seed_acc: list[list[float]] = []
        epochs_ref: list[int] = []
        for seed in SEEDS:
            metrics_path = baseline_dir(m, seed) / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = load_metrics(baseline_dir(m, seed))
            eps = [e["ep"] for e in metrics["epochs"]]
            if not epochs_ref:
                epochs_ref = eps
            per_seed_loss.append([e["loss"] for e in metrics["epochs"]])
            per_seed_acc.append([e["acc"] for e in metrics["epochs"]])
        loss_arr = np.asarray(per_seed_loss, dtype=float)
        acc_arr = np.asarray(per_seed_acc, dtype=float)
        loss_mean = loss_arr.mean(axis=0)
        acc_mean = acc_arr.mean(axis=0)
        ax_loss.plot(epochs_ref, loss_mean, marker="o", color=MODEL_COLORS[m], label=m)
        ax_acc.plot(epochs_ref, acc_mean, marker="o", color=MODEL_COLORS[m], label=m)
        if loss_arr.shape[0] > 1:
            loss_sem = loss_arr.std(axis=0, ddof=1) / np.sqrt(loss_arr.shape[0])
            acc_sem = acc_arr.std(axis=0, ddof=1) / np.sqrt(acc_arr.shape[0])
            ax_loss.fill_between(
                epochs_ref, loss_mean - loss_sem, loss_mean + loss_sem,
                color=MODEL_COLORS[m], alpha=0.18, linewidth=0,
            )
            ax_acc.fill_between(
                epochs_ref, acc_mean - acc_sem, acc_mean + acc_sem,
                color=MODEL_COLORS[m], alpha=0.18, linewidth=0,
            )
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("train loss")
    title_suffix = f" (n={n_seeds} seeds, mean ± SEM)" if n_seeds > 1 else ""
    ax_loss.set_title("Train loss per epoch" + title_suffix)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=theme.SIZE_LEGEND)
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("test accuracy (%)")
    ax_acc.set_title("Test accuracy per epoch" + title_suffix)
    ax_acc.set_ylim(0, 100)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(fontsize=theme.SIZE_LEGEND)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def render_raster(npz_path: Path, out_path: Path, title: str) -> None:
    """Population spike raster from snapshot.npz."""
    theme.apply()
    data = np.load(npz_path)
    spk_e = data["spk_e"]
    spk_i = data["spk_i"]
    dt = float(data["dt"])
    T = spk_e.shape[0]
    t_ms = np.arange(T) * dt
    has_i = spk_i.size > 0 and spk_i.shape[0] == T and spk_i.any()
    if has_i:
        fig, (ax_e, ax_i) = plt.subplots(
            2, 1, figsize=(5.6, 3.15), sharex=True,
            gridspec_kw={"height_ratios": [4, 1]},
        )
    else:
        fig, ax_e = plt.subplots(1, 1, figsize=(5.6, 3.15))
        ax_i = None
    e_idx, e_t = np.where(spk_e.T)
    ax_e.scatter(
        t_ms[e_t], e_idx, s=1.0, c=theme.INK_BLACK, marker="|", linewidths=0.5
    )
    ax_e.set_ylabel("E neuron")
    ax_e.set_ylim(0, spk_e.shape[1])
    ax_e.set_xlim(0, T * dt)
    ax_e.set_title(title)
    if has_i:
        assert ax_i is not None  # has_i is only True when the 2-axes branch ran
        i_idx, i_t = np.where(spk_i.T)
        ax_i.scatter(
            t_ms[i_t], i_idx, s=1.0, c=theme.DEEP_RED, marker="|", linewidths=0.5
        )
        ax_i.set_ylabel("I neuron")
        ax_i.set_ylim(0, spk_i.shape[1])
        ax_i.set_xlim(0, T * dt)
        ax_i.set_xlabel("time (ms)")
    else:
        ax_e.set_xlabel("time (ms)")
    fig.tight_layout()
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


def render_baseline_rasters_combined(
    npz_coba: Path, npz_ping: Path, out_path: Path,
) -> None:
    """Publication-quality 2-panel raster comparing trained COBA and
    trained PING on the same MNIST input. E spikes (black) and I spikes
    (red) stacked per panel; COBA's I region is empty by construction
    (loop disabled). Shared x-axis so the temporal contrast reads at a
    glance — COBA fires asynchronously across the trial; PING fires in
    gamma-locked bursts."""
    theme.apply()
    fig, (ax_coba, ax_ping) = plt.subplots(
        2, 1, figsize=(6.9, 4.278), dpi=150,
        sharex=True, gridspec_kw={"hspace": 0.22, "left": 0.07,
                                  "right": 0.985, "top": 0.93, "bottom": 0.09},
    )
    cells = [
        (ax_coba, npz_coba, "COBA — recurrent inhibitory loop disabled"),
        (ax_ping, npz_ping, "PING — recurrent inhibitory loop active"),
    ]
    for ax, npz_path, title in cells:
        data = np.load(npz_path)
        spk_e = data["spk_e"]  # (T, N_E)
        spk_i = data["spk_i"]  # (T, N_I)
        dt = float(data["dt"])
        T = spk_e.shape[0]
        N_E = spk_e.shape[1]
        N_I = spk_i.shape[1] if spk_i.size > 0 and spk_i.ndim == 2 else 0
        t_ms = np.arange(T) * dt
        gap = max(8, N_E // 40)
        e_t, e_n = np.where(spk_e)
        ax.scatter(
            t_ms[e_t], e_n, s=1.4, c=theme.INK_BLACK,
            marker="|", linewidths=0.45,
        )
        if N_I > 0 and spk_i.any():
            i_t, i_n = np.where(spk_i)
            ax.scatter(
                t_ms[i_t], i_n + N_E + gap,
                s=1.6, c=theme.DEEP_RED, marker="|", linewidths=0.55,
            )
            ax.set_ylim(-2, N_E + N_I + gap + 2)
            ax.set_yticks([N_E / 2, N_E + gap + N_I / 2])
            ax.set_yticklabels(["E\n(1024)", "I\n(256)"], fontsize=theme.SIZE_LABEL)
        else:
            ax.set_ylim(-2, N_E + 2)
            ax.set_yticks([N_E / 2])
            ax.set_yticklabels(["E\n(1024)"], fontsize=theme.SIZE_LABEL)
            ax.text(
                T * dt * 0.985, N_E - 30,
                "I population silent\n($W^{ei} = W^{ie} = 0$)",
                ha="right", va="top", fontsize=theme.SIZE_LABEL - 1,
                color=theme.MUTED, fontstyle="italic",
            )
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, T * dt)
        ax.set_title(title, fontsize=theme.SIZE_LABEL, loc="left", pad=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_ping.set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    fig.suptitle(
        "Trained-baseline single-trial rasters — same MNIST input, "
        "same trial duration",
        fontsize=theme.SIZE_TITLE,
    )
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


def generate_raster(model: str, out_path: Path) -> None:
    """Replay the trained baseline (rate target = off) network on MNIST digit 0
    for 400 ms and render its raster figure."""
    infer_dir = baseline_dir(model) / "infer"
    infer_dir.mkdir(parents=True, exist_ok=True)
    npz_path = infer_dir / "snapshot.npz"
    if npz_path.exists():
        npz_path.unlink()
    baseline = baseline_dir(model)
    checkpoint = resolve_checkpoint(baseline, CHECKPOINT_ROLE)
    argv = [
        "sim",
        "--infer",
        "--load-config", str(baseline / "config.json"),
        "--load-weights", str(checkpoint["path"]),
        "--input", "dataset",
        "--dataset", "mnist",
        "--digit", "0",
        "--sample", "0",
        "--t-ms", "400",
        "--out-dir", str(infer_dir),
    ]
    print(f"[raster] {model}: {' '.join(argv)}")
    run_cli(argv)
    if not npz_path.exists():
        raise SystemExit(f"pinglab-cli did not produce {npz_path}")
    render_raster(npz_path, out_path, f"{model} — trained network, MNIST digit 0, 400 ms")



def aggregate_frontier(rows: list[dict]) -> list[dict]:
    """Summarise each model/budget point across independent seeds."""
    aggregates: list[dict] = []
    for model in MODELS:
        for rate_target_hz in RATE_TARGET_GRID_HZ:
            cell_rows = [
                row for row in rows
                if row["model"] == model and row["rate_target_hz"] == rate_target_hz
            ]
            if not cell_rows:
                continue
            accs = np.asarray([row["final_acc"] for row in cell_rows], dtype=float)
            rates = np.asarray([row["rate_e"] for row in cell_rows], dtype=float)
            n = len(cell_rows)
            aggregates.append({
                "model": model,
                "rate_target_hz": rate_target_hz,
                "rate_target_display": cell_rows[0]["rate_target_display"],
                "statistic": "mean_across_independent_seeds",
                "uncertainty": "sem_across_independent_seeds",
                "n_seeds": n,
                "seeds": [int(row["seed"]) for row in cell_rows],
                "cell_names": [row["cell_name"] for row in cell_rows],
                "acc_mean": float(accs.mean()),
                "rate_mean": float(rates.mean()),
                "acc_sem": float(accs.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
                "rate_sem": float(rates.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            })
    return aggregates


def plot_frontier(rows: list[dict], out_path: Path, run_id: str) -> None:
    """Pareto-style frontier: one line per model, baseline → tightest
    penalty. Both axes are final-epoch means ± SEM across three seeds."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(5.6, 3.15))
    aggregates = aggregate_frontier(rows)
    for model in MODELS:
        agg_pts = [point for point in aggregates if point["model"] == model]
        agg_pts.sort(key=lambda p: p["rate_mean"])
        xs = [p["rate_mean"] for p in agg_pts]
        ys = [p["acc_mean"] for p in agg_pts]
        x_err = [p["rate_sem"] for p in agg_pts]
        y_err = [p["acc_sem"] for p in agg_pts]
        ax.errorbar(
            xs, ys, xerr=x_err, yerr=y_err,
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            label=model, capsize=3,
        )
        for p in agg_pts:
            ax.annotate(
                p["rate_target_display"],
                (p["rate_mean"], p["acc_mean"]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED,
            )
    ax.set_xlabel("hidden E firing rate (Hz, final epoch)")
    ax.set_ylabel("test accuracy (%, final epoch)")
    ax.set_title("Accuracy / rate frontier across the coba → ping ladder")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)




def _infer_cell(
    train_dir: Path,
    extra_args: list[str] | None = None,
    out_name: str = "infer",
    max_samples: int | None = None,
) -> Path:
    """Shell out to `sim --infer` for one trained cell; return the out dir.

    Network build, weight load and the forward pass run in the CLI — the notebook
    only runs it and reads artifacts. extra_args adds flags (--scale-w-in,
    --outputs, ...); max_samples caps the evaluation set.
    """
    train_dir = train_dir.resolve()
    checkpoint = resolve_checkpoint(train_dir, CHECKPOINT_ROLE)
    out_dir = (ARTIFACTS / out_name / f"{train_dir.name}__{cache_tag(checkpoint)}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "sim", "--infer",
        "--load-config", str(train_dir / "config.json"),
        "--load-weights", str(checkpoint["path"]),
        "--out-dir", str(out_dir),
    ]
    if max_samples is not None:
        cmd += ["--max-samples", str(max_samples)]
    cmd += list(extra_args or [])
    run_cli(cmd)
    return out_dir


def _eval_scaled(
    train_dir: Path,
    scale_w_in: float = 1.0,
    fr_upper_target_hz: float = 0.0,
    fr_upper_strength: float = 0.0,
) -> tuple[float, float, float, float, float]:
    """Evaluate a trained cell with W_in scaling; return
    (acc, ce_loss, penalty, e_rate, i_rate).

    acc / ce_loss / rates come from metrics.json. The firing-rate-upper penalty is
    recomputed from the emitted per-presentation hidden-E population rates using
    the same sample-wise mean squared overshoot as the trainer.
    """
    out_dir = _infer_cell(
        train_dir,
        ["--scale-w-in", str(scale_w_in), "--outputs", "per_cell_rates"],
        out_name=f"win_scale/s{scale_w_in:g}",
        max_samples=EVAL_MAX_SAMPLES,
    )
    m = json.loads((out_dir / "metrics.json").read_text())
    rates = m.get("rates_hz", {})
    penalty = 0.0
    if fr_upper_strength > 0:
        pc = np.load(out_dir / "per_cell_rates.npz")
        sample_rates = pc["rate_e_per_sample"]
        penalty = float(
            fr_upper_strength
            * (np.maximum(sample_rates - fr_upper_target_hz, 0.0) ** 2).mean()
        )
    return (
        float(m["best_acc"]),
        float(m["ce_loss"]),
        penalty,
        float(rates.get("hid", 0.0)),
        float(rates.get("inh", 0.0)),
    )


def evaluate_frontier_cell(train_dir: Path) -> dict:
    """Measure one TR-02 endpoint on a single, shared evaluation contract.

    Both coordinates in Figure 1's frontier come from the same deterministic
    subset of the official MNIST test partition, using the final-epoch
    checkpoint selected by ``_infer_cell``. Training's per-epoch ``acc`` and
    ``rate_e`` fields describe different validation/reference measurements and
    therefore cannot form a coherent endpoint together.
    """
    out_dir = _infer_cell(
        train_dir,
        out_name="frontier",
        max_samples=EVAL_MAX_SAMPLES,
    )
    metrics = json.loads((out_dir / "metrics.json").read_text())
    config = metrics.get("config", {})
    partition = config.get("evaluation_partition")
    samples = int(config.get("evaluation_samples", metrics.get("n_total", 0)))
    if partition != "official_mnist_test":
        raise RuntimeError(
            f"frontier inference used {partition!r}, expected official_mnist_test"
        )
    if samples != EVAL_MAX_SAMPLES:
        raise RuntimeError(
            f"frontier inference evaluated {samples} samples, expected "
            f"{EVAL_MAX_SAMPLES}"
        )
    rates = metrics.get("rates_hz", {})
    return {
        "final_acc": float(metrics["best_acc"]),
        "rate_e": float(rates["hid"]),
        "evaluation_partition": partition,
        "evaluation_samples": samples,
        "checkpoint_role": CHECKPOINT_ROLE,
    }


# ── per-cycle participation p and gamma frequency f_γ vs rate target ────────
#
# Decomposes the rate floor into its two factors: rate ≈ p · f_γ. As
# rate target tightens, where does the optimiser act? The hypothesis is that p
# slides toward a viability minimum while f_γ stays put (biophysics
# fixed). When p hits the floor, accuracy collapses.

F_GAMMA_BAND_HZ: tuple[float, float] = (5.0, 150.0)
PFG_MAX_TRIALS: int = 20 if SMOKE else 256


def _f_gamma_from_population(
    pop_traces: list[np.ndarray], fs_hz: float,
) -> float:
    """Welch PSD per trial, averaged across trials, peak frequency in band
    via parabolic interpolation. Returns NaN if the spectrum is flat or
    the population is silent."""
    from scipy import signal as sp_signal

    if not pop_traces or pop_traces[0].size == 0:
        return float("nan")
    nperseg = pop_traces[0].size
    psds: list[np.ndarray] = []
    freqs: np.ndarray | None = None
    for tr in pop_traces:
        if tr.std() == 0:
            continue
        f, p = sp_signal.welch(
            tr - tr.mean(), fs=fs_hz, nperseg=nperseg,
            scaling="density", detrend=False,
        )
        psds.append(p)
        freqs = f
    if not psds or freqs is None:
        return float("nan")
    psd_mean = np.mean(np.stack(psds, axis=0), axis=0)
    band = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
    if not band.any() or psd_mean[band].max() <= 0:
        return float("nan")
    in_band = np.where(band)[0]
    peak_local = int(psd_mean[in_band].argmax())
    peak_idx = int(in_band[peak_local])
    if not (0 < peak_idx < len(psd_mean) - 1):
        return float(freqs[peak_idx])
    y0, y1, y2 = (
        float(psd_mean[peak_idx - 1]),
        float(psd_mean[peak_idx]),
        float(psd_mean[peak_idx + 1]),
    )
    denom = y0 - 2.0 * y1 + y2
    offset = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    offset = max(-0.5, min(0.5, offset))
    df = float(freqs[1] - freqs[0])
    return float(freqs[peak_idx] + offset * df)


def measure_p_fgamma(train_dir: Path, is_ping: bool) -> dict:
    """Report f_gamma, cycle-participation p, acc and mean E/I rate, via the CLI.

    Runs `sim --infer --outputs pop_traces rasters` (capped near PFG_MAX_TRIALS
    trials) and computes the metrics locally:
      - f_gamma via Welch PSD on the per-trial E-population traces (PING only)
      - p = fraction of (cell, cycle) pairs with >= 1 spike, from the sparse
        rasters reconstructed per trial (PING only — needs the I-burst peaks to
        delimit cycles)
    The CLI emits base data (population traces + sparse spike indices); all metric
    logic stays here.
    """
    cfg = json.loads((train_dir / "config.json").read_text())
    dt_ms = float(cfg["dt"])
    fs_hz = 1000.0 / dt_ms

    out_dir = _infer_cell(
        train_dir,
        ["--outputs", "pop_traces", "rasters"],
        out_name="pfg",
        max_samples=min(PFG_MAX_TRIALS * 5, EVAL_MAX_SAMPLES),
    )
    m = json.loads((out_dir / "metrics.json").read_text())
    rates = m.get("rates_hz", {})
    acc = float(m["best_acc"])
    e_rate = float(rates.get("hid", 0.0))
    i_rate = float(rates.get("inh", 0.0))
    if not is_ping:
        return {"acc": acc, "e_rate": e_rate, "i_rate": i_rate,
                "f_gamma": None, "p": None}

    pt = np.load(out_dir / "pop_traces.npz")
    pop_e_traces = list(pt["pop_e"])
    f_gamma_val = _f_gamma_from_population(pop_e_traces, fs_hz)
    f_gamma = float(f_gamma_val) if np.isfinite(f_gamma_val) else None

    # Reconstruct per-trial dense E/I rasters from the sparse indices and count
    # (cell, cycle) participation. Cycle boundaries come from the I-burst peaks.
    R = np.load(out_dir / "rasters.npz")
    T, n_e, n_i = int(R["T"]), int(R["n_e"]), int(R["n_i"])
    n_trials = min(int(R["n_trials"]), len(pop_e_traces))

    def _by_trial(prefix):
        tr = R[f"{prefix}_trial"]
        order = np.argsort(tr, kind="stable")
        return R[f"{prefix}_t"][order], R[f"{prefix}_cell"][order], \
            np.searchsorted(tr[order], np.arange(n_trials + 1))

    e_t, e_c, e_b = _by_trial("e")
    i_t, i_c, i_b = _by_trial("i")

    n_cycle_pairs = 0
    n_cycle_pairs_active = 0
    for b in range(n_trials):
        s_i_trial = np.zeros((T, n_i), dtype=np.int8)
        s_i_trial[i_t[i_b[b]:i_b[b + 1]], i_c[i_b[b]:i_b[b + 1]]] = 1
        f_gamma_batch = _f_gamma_from_population([pop_e_traces[b]], fs_hz)
        if not np.isfinite(f_gamma_batch) or f_gamma_batch <= 0:
            continue
        peaks = _detect_i_burst_peaks(s_i_trial, dt_ms, f_gamma_batch)
        if peaks.size == 0:
            continue
        s_e_trial = np.zeros((T, n_e), dtype=np.int8)
        s_e_trial[e_t[e_b[b]:e_b[b + 1]], e_c[e_b[b]:e_b[b + 1]]] = 1
        counts = _count_e_spikes_per_cycle(s_e_trial, peaks)  # (K, N_E)
        n_cycle_pairs += counts.size
        n_cycle_pairs_active += int((counts > 0).sum())

    p = (float(n_cycle_pairs_active) / float(n_cycle_pairs)
         if n_cycle_pairs > 0 else None)
    return {"acc": acc, "e_rate": e_rate, "i_rate": i_rate,
            "f_gamma": f_gamma, "p": p}


def _detect_i_burst_peaks(
    s_i_trial: np.ndarray, dt_ms: float, f_gamma_hz: float,
) -> np.ndarray:
    """Mirrors exp046.detect_i_burst_steps. Smooth I population rate with
    1-ms Gaussian, find peaks separated by at least 0.5 cycle."""
    from scipy.signal import find_peaks

    rate = s_i_trial.sum(axis=1).astype(np.float32)
    sigma_steps = max(1.0, 1.0 / dt_ms)
    L = int(np.ceil(4 * sigma_steps))
    k = np.arange(-L, L + 1)
    kernel = np.exp(-0.5 * (k / sigma_steps) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(rate, kernel, mode="same")
    cycle_steps = max(1.0, 1000.0 / max(f_gamma_hz, 1e-3) / dt_ms)
    height = 0.05 * float(smooth.max()) if smooth.max() > 0 else 0.0
    peaks, _ = find_peaks(
        smooth, distance=max(1, int(0.5 * cycle_steps)), height=height,
    )
    return peaks


def _count_e_spikes_per_cycle(
    s_e_trial: np.ndarray, peak_steps: np.ndarray,
) -> np.ndarray:
    """Mirrors exp046.count_e_spikes_per_cycle."""
    T, N_E = s_e_trial.shape
    K = len(peak_steps)
    if K == 0:
        return np.zeros((0, N_E), dtype=np.int32)
    edges = np.concatenate([
        [0],
        ((peak_steps[:-1] + peak_steps[1:]) // 2).astype(int),
        [T],
    ])
    counts = np.zeros((K, N_E), dtype=np.int32)
    for kk in range(K):
        a, b = edges[kk], edges[kk + 1]
        if b > a:
            counts[kk] = s_e_trial[a:b].sum(axis=0)
    return counts


def plot_rate_target_p_fgamma(
    rows: list[dict], out_path: Path, run_id: str,
) -> None:
    """4-panel decomposition vs rate target (Hz):
      (top-left)  p vs rate target (PING only) — the per-cycle participation gate
      (top-right) f_γ vs rate target (PING only) — biophysics, untouched by rate target
      (bottom-left)  E rate vs rate target (both architectures); overlay p · f_γ
      (bottom-right) accuracy vs rate target (both architectures)"""
    theme.apply()

    def by_model(model: str) -> list[dict]:
        sub = [r for r in rows if r["model"] == model and r["rate_target_hz"] is not None]
        sub.sort(key=lambda r: r["rate_target_hz"])
        return sub

    ping = by_model("ping")
    coba = by_model("coba")

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.018), dpi=150)
    (ax_p, ax_fg), (ax_r, ax_a) = axes

    if ping:
        ping_pf = [r for r in ping if r.get("p") is not None and r.get("f_gamma") is not None]
        xs = [r["rate_target_hz"] for r in ping_pf]
        ax_p.plot(xs, [r["p"] for r in ping_pf],
                  marker="o", color=theme.INK_BLACK, lw=1.5)
        ax_fg.plot(xs, [r["f_gamma"] for r in ping_pf],
                   marker="s", color=theme.DEEP_RED, lw=1.5)
        xs_all = [r["rate_target_hz"] for r in ping]
        ax_r.plot(xs_all, [r["e_rate"] for r in ping],
                  marker="o", color=theme.INK_BLACK, lw=1.5, label="PING E (measured)")
        # Predicted overlay stays greyscale-safe: near-black ink is reserved for the
        # measured trace, red is the single accent (COBA), so the prediction is grey
        # dashed with its own marker rather than a second chromatic hue (H13).
        ax_r.plot(xs, [r["p"] * r["f_gamma"] for r in ping_pf],
                  marker="^", color=theme.GREY_MID, lw=1.5, ls="--",
                  label="p × f_γ (predicted)")
        ax_a.plot(xs_all, [r["acc"] for r in ping],
                  marker="o", color=theme.INK_BLACK, lw=1.5, label="PING")

    if coba:
        xs = [r["rate_target_hz"] for r in coba]
        ax_r.plot(xs, [r["e_rate"] for r in coba],
                  marker="s", color=theme.DEEP_RED, lw=1.5, label="COBA E")
        ax_a.plot(xs, [r["acc"] for r in coba],
                  marker="s", color=theme.DEEP_RED, lw=1.5, label="COBA")

    for ax in (ax_p, ax_fg, ax_r, ax_a):
        ax.set_xlabel("rate target (Hz)", fontsize=theme.SIZE_LABEL)
        ax.invert_xaxis()  # tightest penalty on the right read left → right
    ax_p.set_ylabel("p (per-cycle participation)", fontsize=theme.SIZE_LABEL)
    ax_p.set_title("Participation gate vs rate target (PING)", fontsize=theme.SIZE_TITLE)
    ax_p.set_ylim(bottom=0)
    ax_fg.set_ylabel("f_γ (Hz)", fontsize=theme.SIZE_LABEL)
    ax_fg.set_title("Gamma frequency vs rate target (PING)", fontsize=theme.SIZE_TITLE)
    ax_fg.set_ylim(bottom=0)
    ax_r.set_ylabel("E firing rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax_r.set_title("E rate vs rate target — measured vs p · f_γ", fontsize=theme.SIZE_TITLE)
    ax_r.set_ylim(bottom=0)
    ax_r.legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="upper left")
    ax_a.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax_a.set_title("Accuracy vs rate target", fontsize=theme.SIZE_TITLE)
    ax_a.set_ylim(0, 100)
    ax_a.legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="lower left")

    # No baked-in suptitle: the caption carries the takeaway (HOUSESTYLE H17).
    fig.tight_layout()
    stamp_figure(fig, run_id)
    save_figure(fig, out_path)
    plt.close(fig)


# ── low-w_in alternate-schedule sweep (PING under heavy rate target) ────────
#
# Tests the path-dependent-barrier claim: if the network starts with
# W_in too small to recruit the I-loop and rate target is on from epoch 0, can
# training land in the sub-f* COBA-like basin instead of locking into
# PING? The w_in initializations straddle f*, with 0.9 as the shared standard.

def low_w_in_cell_dir(w_in: float, seed: int) -> Path:
    cell = training_run_cell("TR-07", w_in=w_in, seed=seed)
    return shared_cell_dir(cell["name"])


def aggregate_low_w_in_seed_rows(w_in: float, seed_rows: list[dict]) -> dict:
    """Aggregate one local control across independent training seeds."""
    if not seed_rows:
        raise ValueError("low-W_in aggregation requires at least one seed")

    def mean_sem(key: str) -> tuple[float, float]:
        values = np.asarray([row[key] for row in seed_rows], dtype=float)
        sem = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        return float(values.mean()), sem

    final_acc, final_acc_sem = mean_sem("final_acc")
    rate_e, rate_e_sem = mean_sem("rate_e")
    rate_i, rate_i_sem = mean_sem("rate_i")
    return {
        "w_in": float(w_in),
        "seeds": [int(row["seed"]) for row in seed_rows],
        "n_seeds": len(seed_rows),
        "statistic": "mean_across_independent_seeds",
        "uncertainty": "sem_across_independent_seeds",
        "final_acc": final_acc,
        "final_acc_sem": final_acc_sem,
        "rate_e": rate_e,
        "rate_e_sem": rate_e_sem,
        "rate_i": rate_i,
        "rate_i_sem": rate_i_sem,
        "per_seed": seed_rows,
    }


def plot_low_w_in(rows: list[dict], out_path: Path, run_id: str) -> None:
    """2 rows × 3 cols. One column per --w-in init. Top row: per-epoch
    accuracy. Bottom row: per-epoch firing rates with E (black) and I
    (red) overlaid. Reads per-epoch traces from each run's metrics.json."""
    theme.apply()
    n = len(rows)
    # Fixed 16:9 at column width (HOUSESTYLE H12) — one aspect across the writeup.
    # The earlier n-dependent height collapsed to a wide strip whose two stacked
    # y-axis labels overprinted each other; a taller frame gives each row room.
    fig, axes = plt.subplots(
        2, n, figsize=(6.9, 3.881), dpi=150, sharex=True,
    )
    rate_max = 0.0
    for col, row in enumerate(rows):
        histories = [
            load_metrics(low_w_in_cell_dir(row["w_in"], seed))["epochs"]
            for seed in LOW_W_IN_SEEDS
        ]
        epochs = np.arange(1, len(histories[0]) + 1)

        def series(key: str, fallback: str | None = None) -> np.ndarray:
            values = []
            for history in histories:
                values.append([
                    float(
                        e.get(key)
                        if e.get(key) is not None
                        else (e.get(fallback) or 0.0) if fallback else 0.0
                    )
                    for e in history
                ])
            return np.asarray(values, dtype=float)

        acc = series("acc")
        rate_e = series("test_rate_e", "rate_e")
        rate_i = series("test_rate_i", "rate_i")
        acc_mean, rate_e_mean, rate_i_mean = (
            acc.mean(axis=0), rate_e.mean(axis=0), rate_i.mean(axis=0)
        )
        rate_max = max(rate_max, float(rate_e.max()), float(rate_i.max()))

        ax_acc = axes[0, col]
        ax_rate = axes[1, col]
        ax_acc.plot(epochs, acc_mean, color=theme.INK_BLACK, lw=1.5)
        if len(histories) > 1:
            acc_sem = acc.std(axis=0, ddof=1) / np.sqrt(len(histories))
            ax_acc.fill_between(
                epochs, acc_mean - acc_sem, acc_mean + acc_sem,
                color=theme.INK_BLACK, alpha=0.16, linewidth=0,
            )
        ax_acc.axhline(10.0, color=theme.GREY_MID, lw=0.6, ls=":", alpha=0.5)
        ax_acc.set_ylim(0, 100)
        ax_acc.set_title(
            f"$W_\\text{{in}}$ = {row['w_in']:g}",
            fontsize=theme.SIZE_TITLE,
        )
        ax_rate.plot(epochs, rate_e_mean, color=theme.INK_BLACK,
                     lw=1.5, label="E")
        ax_rate.plot(epochs, rate_i_mean, color=theme.DEEP_RED,
                     lw=1.5, label="I")
        if len(histories) > 1:
            for mean, values, color in (
                (rate_e_mean, rate_e, theme.INK_BLACK),
                (rate_i_mean, rate_i, theme.DEEP_RED),
            ):
                sem = values.std(axis=0, ddof=1) / np.sqrt(len(histories))
                ax_rate.fill_between(
                    epochs, mean - sem, mean + sem,
                    color=color, alpha=0.14, linewidth=0,
                )
        ax_rate.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
        if col == 0:
            ax_acc.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
            ax_rate.set_ylabel("Firing rate (Hz)", fontsize=theme.SIZE_LABEL)
            ax_rate.legend(fontsize=theme.SIZE_LABEL, frameon=False,
                           loc="upper left")

    for col in range(n):
        axes[1, col].set_ylim(0, rate_max * 1.1 if rate_max > 0 else 1.0)

    # No baked-in suptitle: the caption carries the takeaway (HOUSESTYLE H17). The
    # per-column $W_"in"$ headers identify the panels.
    fig.tight_layout()
    stamp_figure(fig, run_id)
    save_figure(fig, out_path)
    plt.close(fig)


# ── End low-w_in ────────────────────────────────────────────────────


# ── W_in scale sweep (inference-only, trained PING and COBA) ──────
#
# Tests the bifurcation argument directly: scale each trained network's
# W_in by a multiplicative factor s and walk along the W_in axis at
# inference time. The bifurcation prediction is that PING shows a
# sharp loss feature as s crosses below f^* (loop disengages, readout
# sees mismatched activity), while COBA — which has no f^* — shows a
# smooth monotonic loss curve.

W_IN_SCALE_VALUES: list[float] = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00, 1.15, 1.30, 1.50,
    1.75, 2.00, 2.50, 3.00,
]
if SMOKE:
    # Preserve the trained point and the high-drive endpoint consumed by the
    # write-up while keeping the production sweep compact.
    W_IN_SCALE_VALUES = [0.5, 1.0, 3.0]


def run_w_in_scale_sweep(notebook_run_id: str) -> list[dict]:
    """Inference-only sweep multiplying each trained net's W_in by every value in
    W_IN_SCALE_VALUES, via the CLI, recording (cell, scale, loss, penalty, acc,
    rate_e, rate_i). Cells: PING and COBA at the 1 Hz target.

    Each point is one `sim --infer --scale-w-in s --outputs per_cell_rates`; acc /
    ce_loss / rates come from metrics.json and the firing-rate penalty is recomputed
    locally from per_cell_rates.npz (see _eval_scaled).
    """
    cells = [
        ("ping@rt1hz", cell_dir("ping", 1.0, REPRESENTATIVE_SEED), 1.0),
        ("coba@rt1hz", cell_dir("coba", 1.0, REPRESENTATIVE_SEED), 1.0),
    ]
    rows: list[dict] = []
    for label, train_dir, target_hz in cells:
        if not (train_dir / "weights_final.pth").exists():
            raise SystemExit(
                f"w_in-scale-sweep: missing weights for {label} at {train_dir}"
            )
        for scale in W_IN_SCALE_VALUES:
            acc, ce_loss, penalty, e_rate, i_rate = _eval_scaled(
                train_dir,
                scale_w_in=float(scale),
                fr_upper_target_hz=target_hz,
                fr_upper_strength=FR_STRENGTH_UPPER,
            )
            total_loss = ce_loss + penalty
            rows.append({
                "cell": label,
                "scale": float(scale),
                "loss": float(ce_loss),
                "penalty": float(penalty),
                "total_loss": float(total_loss),
                "acc": float(acc),
                "rate_e": float(e_rate),
                "rate_i": float(i_rate),
            })
            print(
                f"  {label:<11} s={scale:>5.2f}  "
                f"CE={ce_loss:6.3f}  pen={penalty:6.3f}  "
                f"tot={total_loss:6.3f}  acc={acc:5.2f}%  "
                f"E={e_rate:6.2f} Hz  I={i_rate:6.2f} Hz"
            )
    return rows


def plot_w_in_scale_sweep(rows: list[dict], out_path: Path, run_id: str) -> None:
    """Six-panel: CE loss, penalty, total loss, accuracy, E rate, I rate
    vs W_in scale. One curve per (model, rate_target_hz) cell."""
    theme.apply()
    fig, axes_2d = plt.subplots(2, 3, figsize=(6.9, 4.6), dpi=150)
    axes = axes_2d.flatten()
    styles = {
        "coba@rt1hz":  ("COBA (1 Hz target)",
                        theme.DEEP_RED, "s", "-"),
        "ping@rt1hz":  ("PING (1 Hz target)",
                        theme.INK_BLACK, "o", "-"),
    }
    # f* proxy on PING: the W_in scale where the I-population first
    # fires. Linear-interpolate between the largest s with I≈0 and the
    # smallest s with measurable I to estimate the crossing.
    ping_sorted = sorted(
        (r for r in rows if r["cell"] == "ping@rt1hz"),
        key=lambda r: r["scale"],
    )
    f_star_s = None
    for prev, curr in zip(ping_sorted, ping_sorted[1:]):
        if prev["rate_i"] < 0.05 <= curr["rate_i"]:
            f_star_s = 0.5 * (prev["scale"] + curr["scale"])
            break

    for ax in axes:
        ax.set_xlabel("$W_\\text{in}$ scale $s$", fontsize=theme.SIZE_LABEL)
        ax.axvline(1.0, color=theme.GREY_MID, lw=0.6, ls="--", alpha=0.7)
        if f_star_s is not None:
            ax.axvline(f_star_s, color=theme.INK_BLACK, lw=0.8, ls=":", alpha=0.7)
    for cell, (label, color, marker, ls) in styles.items():
        msel = [r for r in rows if r["cell"] == cell]
        if not msel:
            continue
        xs = [r["scale"] for r in msel]
        axes[0].plot(xs, [r["loss"] for r in msel], marker=marker,
                     color=color, lw=1.5, ls=ls, label=label)
        axes[1].plot(xs, [r["penalty"] for r in msel], marker=marker,
                     color=color, lw=1.5, ls=ls, label=label)
        axes[2].plot(xs, [r["total_loss"] for r in msel], marker=marker,
                     color=color, lw=1.5, ls=ls, label=label)
        axes[3].plot(xs, [r["acc"] for r in msel], marker=marker,
                     color=color, lw=1.5, ls=ls, label=label)
        axes[4].plot(xs, [r["rate_e"] for r in msel], marker=marker,
                     color=color, lw=1.5, ls=ls, label=label)
        axes[5].plot(xs, [r["rate_i"] for r in msel], marker=marker,
                     color=color, lw=1.5, ls=ls, label=label)
    axes[0].set_ylabel("Test cross-entropy", fontsize=theme.SIZE_LABEL)
    axes[0].set_title("CE loss", fontsize=theme.SIZE_TITLE)
    if f_star_s is not None:
        ylim = axes[0].get_ylim()
        axes[0].text(
            f_star_s, ylim[1] * 0.95, "$\\approx f^\\star$",
            ha="left", va="top", fontsize=theme.SIZE_ANNOTATION,
            color=theme.INK_BLACK,
        )
    axes[1].set_ylabel("Spike-budget penalty", fontsize=theme.SIZE_LABEL)
    axes[1].set_title("Penalty", fontsize=theme.SIZE_TITLE)
    axes[1].set_ylim(0, 4.0)
    axes[2].set_ylabel("CE + penalty", fontsize=theme.SIZE_LABEL)
    axes[2].set_title("Training-objective loss", fontsize=theme.SIZE_TITLE)
    axes[2].set_ylim(0, 4.0)
    axes[3].set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    axes[3].set_ylim(0, 100)
    axes[3].axhline(10.0, color=theme.GREY_MID, lw=0.6, ls=":", alpha=0.5)
    axes[3].set_title("Accuracy", fontsize=theme.SIZE_TITLE)
    axes[4].set_ylabel("E rate (Hz)", fontsize=theme.SIZE_LABEL)
    axes[4].set_title("E rate", fontsize=theme.SIZE_TITLE)
    axes[5].set_ylabel("I rate (Hz)", fontsize=theme.SIZE_LABEL)
    axes[5].set_title("I rate", fontsize=theme.SIZE_TITLE)
    axes[0].legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="upper right")
    # No baked-in suptitle: the caption carries the takeaway (HOUSESTYLE H17). The
    # s = 1 dashed line and ≈ f* dotted line are already annotated in-panel.
    fig.tight_layout()
    stamp_figure(fig, run_id)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_w_in_scale_sweep_vs_rate(
    rows: list[dict], out_path: Path, run_id: str
) -> None:
    """Same data as plot_w_in_scale_sweep, but x-axis is E rate instead
    of W_in scale s. Y-axes: CE | penalty | total loss | accuracy |
    I rate | s. Each cell's trained s=1 point marked with a filled star
    on every curve so the reader sees where on the rate axis training
    landed."""
    theme.apply()
    fig, axes_2d = plt.subplots(2, 3, figsize=(6.9, 4.6), dpi=150)
    axes = axes_2d.flatten()
    styles = {
        "coba@rt1hz":  ("COBA (1 Hz target)",
                        theme.DEEP_RED, "s", "-"),
        "ping@rt1hz":  ("PING (1 Hz target)",
                        theme.INK_BLACK, "o", "-"),
    }
    for ax in axes:
        ax.set_xlabel("Hidden E rate (Hz)", fontsize=theme.SIZE_LABEL)
    # Order each cell by E rate so lines don't backtrack.
    for cell, (label, color, marker, ls) in styles.items():
        msel = sorted(
            (r for r in rows if r["cell"] == cell),
            key=lambda r: r["rate_e"],
        )
        if not msel:
            continue
        xs = [r["rate_e"] for r in msel]
        axes[0].plot(xs, [r["loss"] for r in msel], marker=marker,
                     color=color, lw=1.5, ls=ls, label=label)
        axes[1].plot(xs, [r["penalty"] for r in msel], marker=marker,
                     color=color, lw=1.5, ls=ls, label=label)
        axes[2].plot(xs, [r["total_loss"] for r in msel], marker=marker,
                     color=color, lw=1.5, ls=ls, label=label)
        axes[3].plot(xs, [r["acc"] for r in msel], marker=marker,
                     color=color, lw=1.5, ls=ls, label=label)
        axes[4].plot(xs, [r["rate_i"] for r in msel], marker=marker,
                     color=color, lw=1.5, ls=ls, label=label)
        axes[5].plot(xs, [r["scale"] for r in msel], marker=marker,
                     color=color, lw=1.5, ls=ls, label=label)
        # Mark each cell's trained operating point (s = 1) with a star.
        trained = next((r for r in msel if abs(r["scale"] - 1.0) < 1e-6), None)
        if trained is not None:
            star_kwargs = dict(marker="*", color=color, markersize=16,
                               markeredgecolor=theme.INK_BLACK,
                               markeredgewidth=0.7, linestyle="None", zorder=5)
            axes[0].plot([trained["rate_e"]], [trained["loss"]], **star_kwargs)
            axes[1].plot([trained["rate_e"]], [trained["penalty"]], **star_kwargs)
            axes[2].plot([trained["rate_e"]], [trained["total_loss"]], **star_kwargs)
            axes[3].plot([trained["rate_e"]], [trained["acc"]], **star_kwargs)
            axes[4].plot([trained["rate_e"]], [trained["rate_i"]], **star_kwargs)
            axes[5].plot([trained["rate_e"]], [trained["scale"]], **star_kwargs)
    axes[0].set_ylabel("Test cross-entropy", fontsize=theme.SIZE_LABEL)
    axes[0].set_title("CE loss", fontsize=theme.SIZE_TITLE)
    axes[1].set_ylabel("Spike-budget penalty", fontsize=theme.SIZE_LABEL)
    axes[1].set_title("Penalty", fontsize=theme.SIZE_TITLE)
    axes[1].set_ylim(0, 4.0)
    axes[2].set_ylabel("CE + penalty", fontsize=theme.SIZE_LABEL)
    axes[2].set_title("Training-objective loss", fontsize=theme.SIZE_TITLE)
    axes[2].set_ylim(0, 4.0)
    axes[3].set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    axes[3].set_ylim(0, 100)
    axes[3].axhline(10.0, color=theme.GREY_MID, lw=0.6, ls=":", alpha=0.5)
    axes[3].set_title("Accuracy", fontsize=theme.SIZE_TITLE)
    axes[4].set_ylabel("I rate (Hz)", fontsize=theme.SIZE_LABEL)
    axes[4].set_title("I rate", fontsize=theme.SIZE_TITLE)
    axes[5].set_ylabel("$W_\\text{in}$ scale $s$", fontsize=theme.SIZE_LABEL)
    axes[5].set_title("$W_\\text{in}$ scale", fontsize=theme.SIZE_TITLE)
    axes[0].legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="upper right")
    # No baked-in suptitle: the caption carries the takeaway (HOUSESTYLE H17). The
    # trained-point stars are already annotated in-panel.
    fig.tight_layout()
    stamp_figure(fig, run_id)
    save_figure(fig, out_path)
    plt.close(fig)



# ── End W_in scale sweep ───────────────────────────────────────────


def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_results_compound(rows, npz_coba, npz_ping, out_path, run_id):
    """exp023-Figure-1-style super figure (replotted from cache, no retraining):
    top row two trained-baseline rasters (COBA | PING), bottom row four small
    plots — train loss, test accuracy, accuracy–rate frontier, accuracy/rate
    bars."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"  # keep the saved 16:9 exact
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(6.9, 3.881), dpi=150)  # 16:9
    gs = GridSpec(
        2, 2, figure=fig, height_ratios=[3.0, 2.6],
        hspace=0.45, wspace=0.2, top=0.93, bottom=0.1, left=0.07, right=0.96,
    )

    # --- top row: two rasters side by side (E black, I red above) ---
    for col, (npz_path, title) in enumerate([
        (npz_coba, "COBA — loop off"),
        (npz_ping, "PING — loop on"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        data = np.load(npz_path)
        spk_e, spk_i, dt = data["spk_e"], data["spk_i"], float(data["dt"])
        T, N_E = spk_e.shape[0], spk_e.shape[1]
        N_I = spk_i.shape[1] if spk_i.size > 0 and spk_i.ndim == 2 else 0
        t_ms = np.arange(T) * dt
        gap = max(8, N_E // 40)
        e_t, e_n = np.where(spk_e)
        ax.scatter(t_ms[e_t], e_n, s=0.8, c=theme.INK_BLACK, marker="|", linewidths=0.35)
        if N_I > 0 and spk_i.any():
            i_t, i_n = np.where(spk_i)
            ax.scatter(t_ms[i_t], i_n + N_E + gap, s=1.0, c=theme.DEEP_RED,
                       marker="|", linewidths=0.45)
            ax.set_ylim(-2, N_E + N_I + gap + 2)
            ax.set_yticks([N_E / 2, N_E + gap + N_I / 2])
            ax.set_yticklabels(["E", "I"])
        else:
            ax.set_ylim(-2, N_E + 2)
            ax.set_yticks([N_E / 2])
            ax.set_yticklabels(["E"])
            ax.text(T * dt * 0.985, N_E - 30, "I silent (loop off)",
                    ha="right", va="top", fontsize=theme.SIZE_LABEL - 1,
                    color=theme.MUTED, fontstyle="italic",
                    # Opaque backing so the note reads over the dense raster
                    # instead of smearing into it like a ghost watermark.
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="none", alpha=0.75))
        ax.set_xlim(0, T * dt)
        ax.set_xlabel("time (ms)")
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", fontweight="semibold")
        _despine(ax)

    # --- bottom-left: validation accuracy per epoch (checkpoint selection) ---
    ax_acc = fig.add_subplot(gs[1, 0])
    for m in MODELS:
        accs, eps = [], []
        for seed in SEEDS:
            if not (baseline_dir(m, seed) / "metrics.json").exists():
                continue
            md = load_metrics(baseline_dir(m, seed))
            if not eps:
                eps = [e["ep"] for e in md["epochs"]]
            accs.append([e["acc"] for e in md["epochs"]])
        if not accs:
            continue
        # Sparse markers keep the near-identical COBA/PING traces legible.
        ax_acc.plot(eps, np.asarray(accs).mean(0), marker=MODEL_MARKERS[m],
                    ms=3.2, markevery=10, lw=1.4,
                    color=MODEL_COLORS[m], label=m.upper())
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("validation accuracy (%)")
    ax_acc.set_ylim(0, 100)
    # No baked-in title: the caption carries the takeaway (HOUSESTYLE H17).
    ax_acc.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="lower right")
    _despine(ax_acc)

    # --- bottom-right: accuracy–rate frontier, operating points annotated ---
    ax_fr = fig.add_subplot(gs[1, 1])
    model_curves = {}
    xmax = 1.0
    frontier_stats = aggregate_frontier(rows)
    for m in MODELS:
        pts = [point for point in frontier_stats if point["model"] == m]
        pts.sort(key=lambda point: point["rate_mean"])
        base_point = next((point for point in pts if point["rate_target_hz"] is None), None)
        base = (
            (base_point["rate_mean"], base_point["acc_mean"])
            if base_point is not None else None
        )
        model_curves[m] = (pts, base)
        if pts:
            xmax = max(xmax, max(point["rate_mean"] for point in pts))
    ax_fr.set_xlim(-xmax * 0.03, xmax * 1.12)  # left margin so near-zero points read; right headroom for the COBA label
    ax_fr.set_ylim(40, 100)  # all points ≥ 54%; crop dead space to grow the frontier
    for m in MODELS:
        pts, base = model_curves[m]
        if pts:
            ax_fr.errorbar(
                [point["rate_mean"] for point in pts],
                [point["acc_mean"] for point in pts],
                xerr=[point["rate_sem"] for point in pts],
                yerr=[point["acc_sem"] for point in pts],
                marker=MODEL_MARKERS[m], ms=4, lw=1.4, capsize=2,
                color=MODEL_COLORS[m], label=m.upper(),
            )
    for m in MODELS:
        base = model_curves[m][1]
        if base is None:
            continue
        ax_fr.scatter([base[0]], [base[1]], s=130, marker="*",
                      color=MODEL_COLORS[m], edgecolor=theme.INK_BLACK,
                      linewidths=0.7, zorder=6)
        # PING star: label up-right into open plot space; COBA star sits top-
        # right, so label down-left to avoid clipping the axis and the title.
        if m == "ping":
            # Label up-and-right of the star, into the open space above the trace,
            # with enough offset that it never crowds the frontier line.
            dxdy, ha, va = (14, 5), "left", "bottom"
        else:
            dxdy, ha, va = (-6, -8), "right", "top"
        ax_fr.annotate(f"{m.upper()}\n{base[1]:.0f}% @ {base[0]:.0f} Hz",
                       (base[0], base[1]), xytext=dxdy,
                       textcoords="offset points", ha=ha, va=va,
                       fontsize=theme.SIZE_ANNOTATION, color=MODEL_COLORS[m])
    ax_fr.set_xlabel("hidden-E firing rate (Hz)")
    ax_fr.set_ylabel("test accuracy (%)")
    # No baked-in title: the caption carries the takeaway (HOUSESTYLE H17).
    ax_fr.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="lower right",
                 title="★ = spike budget off", title_fontsize=theme.SIZE_ANNOTATION)
    _despine(ax_fr)

    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense rasters: PNG, not SVG
    plt.close(fig)


def build_results_compound(figures: Path, run_id: str) -> None:
    """Redraw the compound from its published quantitative payload only."""
    numbers_path = figures / "numbers.json"
    if not numbers_path.exists():
        raise SystemExit(f"missing published frontier data: {numbers_path}")
    rows = json.loads(numbers_path.read_text()).get("results", [])
    if not rows or any(
        row.get("evaluation_partition") != "official_mnist_test" for row in rows
    ):
        raise SystemExit(
            "published frontier lacks the official-test evaluation contract; "
            "run exp025 once before plot-only"
        )
    npz_coba = baseline_dir("coba") / "infer" / "snapshot.npz"
    npz_ping = baseline_dir("ping") / "infer" / "snapshot.npz"
    for p in (npz_coba, npz_ping):
        if not p.exists():
            raise SystemExit(f"missing cached raster: {p} (run the notebook once first)")
    out = figures / "results_compound"
    fig_results_compound(rows, npz_coba, npz_ping, out, run_id)
    print(f"wrote {out}.{{png,pdf}}")


def main() -> None:
    meta = parse_meta(sys.argv)

    # Publication profile: every figure this notebook writes is a print-sized
    # vector, emitted as both SVG (docs) and PDF (manuscript) by save_figure;
    # dense rasters go to PNG + PDF. Set before any early-return path so the
    # `--plot-only compound` route gets paper mode too.
    theme.set_paper_mode(True)

    if meta.plot_fig == "compound":
        # Redraw the results compound from cache only — no training, no inference
        # reruns — and republish atomically alongside the other figures.
        run_id = next_run_id(SLUG)
        with published_run(
            SLUG, run_id, plot_only=True, scale=SCALE,
        ) as (_artifacts, figures):
            build_results_compound(figures, run_id)
        return

    t_start = time.monotonic()
    run_id = next_run_id(SLUG)
    n_cells = (
        len(MODELS) * len(RATE_TARGET_GRID_HZ) * len(SEEDS)
        + len(LOW_W_IN_VALUES) * len(LOW_W_IN_SEEDS)
    )
    print(
        f"notebook_run_id = {run_id} cells={n_cells}"
        + ("  [skip-training]" if meta.skip_training else "")
    )
    log_runner_event(SLUG, "started", run_id=run_id)

    with published_run(
        SLUG, run_id, skip_training=meta.skip_training, make_artifacts=False,
        scale=SCALE, plot_only=meta.plot_only,
    ) as (_artifacts, figures):
        _run(meta, run_id, figures, t_start)
    log_runner_event(SLUG, "completed", run_id=run_id)


def _run(meta, run_id: str, figures: Path, t_start: float) -> None:
    # All training is owned by exp022. exp025 only consumes TR-02 and TR-07.

    rows: list[dict] = []
    for model in MODELS:
        for rate_target_hz in RATE_TARGET_GRID_HZ:
            for seed in seeds_for(rate_target_hz):
                run_dir = cell_dir(model, rate_target_hz, seed)
                if not (run_dir / "metrics.json").exists():
                    raise SystemExit(f"missing metrics: {run_dir / 'metrics.json'}")
                metrics = load_metrics(run_dir)
                endpoint = evaluate_frontier_cell(run_dir)
                rows.append(
                    {
                        "cell_name": cell_name(model, rate_target_hz, seed),
                        "model": model,
                        "rate_target_display": rate_target_display(rate_target_hz),
                        "rate_target_hz": rate_target_hz,
                        "seed": seed,
                        "best_acc": float(metrics["best_acc"]),
                        "best_epoch": int(metrics["best_epoch"]),
                        **endpoint,
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
            f"acc(test)={r['final_acc']:6.2f}%  "
            f"best(validation)={r['best_acc']:6.2f}%  "
            f"rate_e(test)={r['rate_e']:6.1f} Hz"
        )

    # rate target vs (p, f_γ) decomposition — mechanism behind the frontier floor.
    print("[rate-target-pfg] measuring p and f_γ per (model, rate target) cell")
    pfg_rows: list[dict] = []
    for model in MODELS:
        is_ping = (model == "ping")
        for rate_target_hz in RATE_TARGET_GRID_HZ:
            seed = REPRESENTATIVE_SEED
            train_dir = cell_dir(model, rate_target_hz, seed)
            if not (train_dir / "weights_final.pth").exists():
                print(f"  skip {model} rate target={rate_target_label(rate_target_hz)} (no weights)")
                continue
            m = measure_p_fgamma(train_dir, is_ping=is_ping)
            row = {
                "model": model,
                "rate_target_hz": rate_target_hz,
                "seed": seed,
                "selection": "representative_seed",
                **m,
            }
            pfg_rows.append(row)
            theta_str = (
                f"rate target={rate_target_display(rate_target_hz):>4} ({rate_target_hz_value(rate_target_hz):>4.1f} Hz)"
                if rate_target_hz is not None else "rate target= off"
            )
            # f_γ and p are None for COBA (no loop → no gamma cycles to bin).
            fg = f"{m['f_gamma']:5.2f}" if m["f_gamma"] is not None else "   --"
            pp = f"{m['p']:.3f}" if m["p"] is not None else "  -- "
            print(
                f"  {model:<5}  {theta_str}  "
                f"acc={m['acc']:5.2f}%  E={m['e_rate']:5.2f} Hz  "
                f"f_γ={fg} Hz  p={pp}"
            )
    plot_rate_target_p_fgamma(
        pfg_rows, figures / "theta_p_fgamma", run_id,
    )
    print(f"wrote {figures / 'theta_p_fgamma'}.{{svg,pdf}}")

    for model in MODELS:
        out = figures / f"raster__{model}"
        generate_raster(model, out)
        print(f"wrote {out}.{{png,pdf}}")

    # Results compound (Figure 1): rasters + accuracy-per-epoch + the
    # accuracy–rate frontier in one frame (replaces the four standalones).
    npz_coba = baseline_dir("coba") / "infer" / "snapshot.npz"
    npz_ping = baseline_dir("ping") / "infer" / "snapshot.npz"
    if npz_coba.exists() and npz_ping.exists():
        out = figures / "results_compound"
        fig_results_compound(rows, npz_coba, npz_ping, out, run_id)
        print(f"wrote {out}.{{png,pdf}}")

    # Low-w_in alternate-schedule sweep — aggregate independent seeds and plot
    # mean trajectories with SEM bands.
    print("[low-w_in-sweep] reading metrics from dispatched trainings")
    low_w_in_rows: list[dict] = []
    for w_in in LOW_W_IN_VALUES:
        seed_rows: list[dict] = []
        for seed in LOW_W_IN_SEEDS:
            run_dir = low_w_in_cell_dir(w_in, seed)
            if not (run_dir / "metrics.json").exists():
                raise SystemExit(f"missing metrics: {run_dir / 'metrics.json'}")
            metrics = load_metrics(run_dir)
            last = metrics["epochs"][-1]
            seed_rows.append({
                "seed": seed,
                "best_acc": float(metrics["best_acc"]),
                "best_epoch": int(metrics["best_epoch"]),
                "final_acc": float(last["acc"]),
                "rate_e": float(last.get("test_rate_e") or last.get("rate_e") or 0.0),
                "rate_i": float(last.get("test_rate_i") or last.get("rate_i") or 0.0),
            })

        aggregate = aggregate_low_w_in_seed_rows(w_in, seed_rows)
        low_w_in_rows.append(aggregate)
        print(
            f"  w_in={w_in:>4}  "
            f"acc={aggregate['final_acc']:5.2f}±{aggregate['final_acc_sem']:.2f}%  "
            f"E={aggregate['rate_e']:6.2f}±{aggregate['rate_e_sem']:.2f} Hz  "
            f"I={aggregate['rate_i']:6.2f}±{aggregate['rate_i_sem']:.2f} Hz"
        )
    plot_low_w_in(low_w_in_rows, figures / "low_w_in_sweep", run_id)
    print(f"wrote {figures / 'low_w_in_sweep'}.{{svg,pdf}}")

    # W_in scale sweep — direct test of the bifurcation argument.
    print("[w_in-scale-sweep] inference-only on trained PING and COBA")
    w_in_scale_rows = run_w_in_scale_sweep(run_id)
    plot_w_in_scale_sweep(
        w_in_scale_rows, figures / "w_in_scale_sweep", run_id,
    )
    print(f"wrote {figures / 'w_in_scale_sweep'}.{{svg,pdf}}")
    plot_w_in_scale_sweep_vs_rate(
        w_in_scale_rows,
        figures / "w_in_scale_sweep_vs_rate",
        run_id,
    )
    print(f"wrote {figures / 'w_in_scale_sweep_vs_rate'}.{{svg,pdf}}")

    duration_s = time.monotonic() - t_start
    train_cfg = load_config(baseline_dir(MODELS[0]))
    tr07_cfg = load_config(low_w_in_cell_dir(LOW_W_IN_VALUES[0], LOW_W_IN_SEEDS[0]))
    frontier_statistics = aggregate_frontier(rows)
    shared_checkpoints = checkpoint_provenance(
        [cell_dir(model, theta, seed) for model in MODELS
         for theta in RATE_TARGET_GRID_HZ for seed in SEEDS],
        CHECKPOINT_ROLE,
    )
    tr07_checkpoints = checkpoint_provenance(
        [low_w_in_cell_dir(w_in, seed) for w_in in LOW_W_IN_VALUES
         for seed in LOW_W_IN_SEEDS],
        CHECKPOINT_ROLE,
    )
    write_numbers(
        figures, run_id=run_id, duration_s=duration_s,
        payload={
            "git_sha_train": train_cfg.get("git_sha"),
            "checkpoint_policy": CHECKPOINT_POLICY,
            "checkpoint_provenance": shared_checkpoints,
            "training_sources": {
                "shared_tr02": {
                    "owner": "exp022/TR-02",
                    "max_samples": int(train_cfg["max_samples"]),
                    "epochs": int(train_cfg["epochs"]),
                    "seeds": list(SEEDS),
                    "checkpoint_role": CHECKPOINT_ROLE,
                    "checkpoints": shared_checkpoints,
                },
                "low_w_in_controls": {
                    "owner": "exp022/TR-07",
                    "max_samples": int(tr07_cfg["max_samples"]),
                    "epochs": int(tr07_cfg["epochs"]),
                    "seeds": list(LOW_W_IN_SEEDS),
                    "checkpoint_role": CHECKPOINT_ROLE,
                    "checkpoints": tr07_checkpoints,
                },
            },
            "config": {
                "dataset": "mnist",
                "models": MODELS,
                "rate_target_grid_hz": [
                    rate_target_hz_value(t) for t in RATE_TARGET_GRID_HZ if t is not None
                ],
                "max_samples": MAX_SAMPLES,
                "evaluation_samples": EVAL_MAX_SAMPLES,
                "epochs": EPOCHS,
                "t_ms": T_MS,
                "dt": DT_TRAIN,
                "frontier_seeds": SEEDS,
                "representative_seed": REPRESENTATIVE_SEED,
                "fr_strength_upper": FR_STRENGTH_UPPER,
            },
            "results": rows,
            "frontier_statistics": frontier_statistics,
            "rate_target_p_fgamma": pfg_rows,
            "low_w_in_sweep": low_w_in_rows,
            "w_in_scale_sweep": w_in_scale_rows,
        },
    )
    print(f"wrote {figures / 'numbers.json'}")



if __name__ == "__main__":
    main()
