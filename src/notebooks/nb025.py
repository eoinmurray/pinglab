"""Notebook runner for entry 035 — why PING has a rate floor.

Standalone runner with no cross-notebook helpers. Inlines everything
it needs (constants, baseline training, plot fns, sweeps).

For each rung of the biophysical ladder (coba, ping), trains the
canonical recipe at six values of the upper-bound spike budget θ_u:
off (no penalty) plus θ_u ∈ {5, 2, 1, 0.5, 0.2} spikes/trial =
{25, 10, 5, 2.5, 1} Hz. Same recipe in every other respect — only the
regulariser flag changes — so each (model, θ_u) cell is one point on
that model's accuracy / rate Pareto frontier. The unpenalised "off"
cell of each model feeds the foundation figures (acc-vs-rate bar,
learning curves, post-training rasters).

Then runs:
- the low-W_in alternate-schedule sweep (four PING networks
  initialised across the recruitment cliff) showing all four converge
  to PING; and
- the inference-time W_in scale sweep on the heaviest-budget cells
  (ping@θ_u=0.2, coba@θ_u=0.2) projecting the cliff into the loss
  landscape.

All figures land in /figures/notebooks/nb025/ and the success-criteria
summary in nb025/numbers.json.

Notebook entry: src/docs/src/pages/notebooks/nb025.mdx
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from _modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402
from cli import theme  # noqa: E402

SLUG = "nb025"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "cli/cli.py"

TIER_CONFIG = {
    "extra small": dict(max_samples=100, epochs=2),
    "small": dict(max_samples=500, epochs=10),
    "medium": dict(max_samples=2000, epochs=100),
    "large": dict(max_samples=5000, epochs=100),
    "extra large": dict(max_samples=10000, epochs=100),
}
DEFAULT_TIER = "small"
T_MS = 200.0
DT_TRAIN = 0.1

# Baseline (θ_u = off) cells are trained at multiple seeds so the
# headline bar chart and learning curves can show mean ± SEM. The θ_u
# sweep cells stay single-seed — the frontier *shape* is dominated by
# the regulariser, not the seed.
SEEDS_BASELINE: list[int] = [42, 43, 44]
SEED_SWEEP: int = 42
# 100 epochs follows nb024's convergence audit: PING's E rate is a
# converged operating point at this horizon (slope ≤ 0.025 Hz/ep across
# seeds). COBA's rate is still drifting but closer to its long-run
# attractor than at 30 epochs.
BASELINE_EPOCHS: int = 100  # overrides TIER_CONFIG epochs for baselines

# Inference-time ei_strength sweep on the coba__off__seed42 baseline.
# Subsumes the now-retired nb019 — trains nothing new; just runs the
# already-trained coba weights forward through the test set with a
# fresh ping-arch I-loop at progressively higher ei_strength.
EI_SWEEP: list[float] = [round(0.1 * i, 1) for i in range(11)]  # 0.0–1.0
EI_RASTER: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
EI_RASTER_SAMPLE_IDX: int = 0
EI_RASTER_N_E_PLOT: int = 200
EI_RASTER_N_I_PLOT: int = 64

# θ_u sweep grid in spikes-per-trial. None = no penalty (baseline).
# At T = 200 ms, spikes/trial × 5 = Hz. The grid spans from no
# pressure (off → ~80 Hz coba baseline) down to 1 Hz —
# below ping's natural 5 Hz and into the regime where every model
# loses accuracy.
THETA_U_GRID: list[float | None] = [None, 5.0, 2.0, 1.0, 0.5, 0.2]
FR_STRENGTH_UPPER = 1e-3

MODELS = ["coba", "ping"]

MODEL_RECIPES: dict[str, dict] = {
    "coba": {
        "__build_as": "ping",
        "--ei-strength": "0",
        "--v-grad-dampen": "1000",
        "--w-in": "0.3",
        "--w-in-sparsity": "0.95",
        "--readout": "mem-mean",
        "--surrogate-slope": "1",
        "--readout-w-out-scale": "100",
        "--lr": "0.0004",
        "--batch-size": "256",
    },
    "ping": {
        "__build_as": "ping",
        "--ei-strength": "1",
        "--v-grad-dampen": "1000",
        "--w-in": "1.2",
        "--w-in-sparsity": "0.95",
        "--readout": "mem-mean",
        "--surrogate-slope": "1",
        "--readout-w-out-scale": "500",
        "--lr": "0.0004",
        "--batch-size": "256",
    },
}

MODEL_COLORS = {
    "coba": theme.DEEP_RED,
    "ping": theme.INK_BLACK,
}
MODEL_MARKERS = {"coba": "s", "ping": "D"}

MIN_ACC_BY_TIER = {
    "extra small": 15.0,
    "small": 30.0,
    "medium": 50.0,
    "large": 70.0,
    "extra large": 70.0,
}


def theta_label(theta_u: float | None) -> str:
    """Filesystem-safe label for an out-dir / video filename."""
    if theta_u is None:
        return "off"
    s = f"{theta_u:g}".replace(".", "p")
    return f"tu{s}"


def theta_display(theta_u: float | None) -> str:
    """Human label for plots / numbers.json."""
    if theta_u is None:
        return "off"
    return f"{theta_u:g}"


def theta_hz(theta_u: float | None) -> float | None:
    if theta_u is None:
        return None
    return theta_u * (1000.0 / T_MS)


def seeds_for(theta_u: float | None) -> list[int]:
    """Baseline cells run all seeds; sweep cells stay single-seed."""
    return list(SEEDS_BASELINE) if theta_u is None else [SEED_SWEEP]


def cell_dir(model: str, theta_u: float | None, seed: int) -> Path:
    """Per-cell artifact directory.

    Baseline cells get a `__seed{N}` suffix so multiple seeds coexist.
    Sweep cells run only at SEED_SWEEP and skip the suffix to keep
    paths short — they live alongside the baseline ones.
    """
    label = theta_label(theta_u)
    if theta_u is None:
        return ARTIFACTS / f"{model}__{label}__seed{seed}"
    return ARTIFACTS / f"{model}__{label}"


def baseline_dir(model: str, seed: int = SEEDS_BASELINE[0]) -> Path:
    return cell_dir(model, None, seed)


def build_train_args(
    model: str, theta_u: float | None, seed: int, tier: str, out_dir: Path
) -> list[str]:
    recipe = MODEL_RECIPES[model]
    args = [
        "train",
        "--model", recipe["__build_as"],
        "--dataset", "mnist",
        "--max-samples", str(TIER_CONFIG[tier]["max_samples"]),
        "--epochs", str(BASELINE_EPOCHS),
        "--t-ms", str(T_MS),
        "--dt", str(DT_TRAIN),
        "--seed", str(seed),
        "--observe", "video",
        "--frame-rate", "1",
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    for k, v in recipe.items():
        if k.startswith("__"):
            continue
        if v is True:
            args.append(k)
        elif v is not None:
            args += [k, v]
    if theta_u is not None:
        args += [
            "--fr-reg-upper-theta", str(theta_u),
            "--fr-reg-upper-strength", str(FR_STRENGTH_UPPER),
        ]
    return args


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def _stamp(fig, run_id: str) -> None:
    fig.text(
        0.995, 0.005, run_id,
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


def _baseline_seed_stats(rows: list[dict], model: str) -> tuple[float, float, float, float, int]:
    """Mean/SEM of final-epoch accuracy and rate_e across all seeds for
    one model's θ_u = off cells. Returns (acc_mean, acc_sem, rate_mean,
    rate_sem, n_seeds). With n=1, SEM is reported as 0 (no error bar)."""
    baseline_rows = [
        r for r in rows if r["model"] == model and r["theta_u"] is None
    ]
    accs = np.array([r["final_acc"] for r in baseline_rows], dtype=float)
    rates = np.array([r["rate_e"] for r in baseline_rows], dtype=float)
    n = len(accs)
    acc_sem = float(accs.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    rate_sem = float(rates.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return float(accs.mean()), acc_sem, float(rates.mean()), rate_sem, n


def plot_acc_rate_bars(rows: list[dict], out_path: Path, run_id: str) -> None:
    """Twin-y bar chart on the baseline (θ_u = off) cells: per model,
    side-by-side bars for accuracy (left y-axis) and mean hidden-E rate
    (right y-axis). Error bars are ±SEM across baseline seeds."""
    theme.apply()
    fig, ax_acc = plt.subplots(figsize=(8.0, 4.5))
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

    # Local black-and-grey scheme for the headline bar chart — keeps the
    # two-bars-per-model pairing readable on a single axes without leaning on
    # the model-color palette used elsewhere.
    bar_colors = {"coba": theme.GREY_MID, "ping": theme.DEEP_RED}
    ax_acc.bar(
        xs - width / 2, accs, width=width,
        color=[bar_colors[m] for m in MODELS],
        edgecolor=theme.INK_BLACK,
        yerr=acc_sems, ecolor=theme.INK_BLACK, capsize=4,
    )
    ax_rate.bar(
        xs + width / 2, rates, width=width,
        color=[bar_colors[m] for m in MODELS],
        edgecolor=theme.INK_BLACK, hatch="///",
        yerr=rate_sems, ecolor=theme.INK_BLACK, capsize=4,
    )

    for x, a, ae in zip(xs, accs, acc_sems):
        label = f"{a:.1f}%" if n_seeds <= 1 else f"{a:.1f}±{ae:.1f}%"
        ax_acc.text(
            x - width / 2, a + ae + 1.5, label,
            ha="center", va="bottom",
            fontsize=theme.SIZE_ANNOTATION, color=theme.INK_BLACK,
        )
    for x, r, re_ in zip(xs, rates, rate_sems):
        label = f"{r:.1f} Hz" if n_seeds <= 1 else f"{r:.1f}±{re_:.1f} Hz"
        ax_rate.text(
            x + width / 2, r + re_ + max(rates) * 0.02, label,
            ha="center", va="bottom",
            fontsize=theme.SIZE_ANNOTATION, color=theme.INK_BLACK,
        )

    ax_acc.set_xticks(xs)
    ax_acc.set_xticklabels(MODELS)
    ax_acc.set_ylabel("test accuracy (%, final epoch)")
    ax_rate.set_ylabel("hidden-E firing rate (Hz, final epoch)")
    title_suffix = (
        f" — accuracy vs firing rate (θ_u = off, n={n_seeds} seeds, mean ± SEM)"
        if n_seeds > 1
        else " — accuracy vs firing rate (θ_u = off)"
    )
    ax_acc.set_title("coba / ping" + title_suffix)
    ax_acc.set_ylim(0, max(100, max(accs) + 10))
    ax_rate.set_ylim(0, max(rates) * 1.2 if rates else 1.0)
    ax_acc.grid(True, axis="y", alpha=0.3)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=theme.PAPER, edgecolor=theme.INK_BLACK),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=theme.PAPER, edgecolor=theme.INK_BLACK,
            hatch="///",
        ),
    ]
    ax_acc.legend(
        handles, ["accuracy (left)", "rate (right)"],
        loc="upper right", fontsize=theme.SIZE_LEGEND,
    )

    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_learning_curves(out_path: Path, run_id: str) -> None:
    """Train loss + test accuracy per epoch, one curve per model
    (θ_u = off). With multiple seeds, plot mean and shade ±SEM."""
    theme.apply()
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10.0, 5.625))
    n_seeds = len(SEEDS_BASELINE)
    for m in MODELS:
        per_seed_loss: list[list[float]] = []
        per_seed_acc: list[list[float]] = []
        epochs_ref: list[int] = []
        for seed in SEEDS_BASELINE:
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
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
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
            2, 1, figsize=(8.0, 4.5), sharex=True,
            gridspec_kw={"height_ratios": [4, 1]},
        )
    else:
        fig, ax_e = plt.subplots(1, 1, figsize=(8.0, 4.5))
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
    fig.savefig(out_path, dpi=150)
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
        2, 1, figsize=(10.0, 6.2), dpi=150,
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
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_raster(model: str, out_path: Path) -> None:
    """Replay the trained baseline (θ_u = off) network on MNIST digit 0
    for 400 ms and render its raster figure."""
    infer_dir = baseline_dir(model) / "infer"
    npz_path = infer_dir / "snapshot.npz"
    if npz_path.exists():
        npz_path.unlink()
    argv = [
        "image",
        "--from-dir", str(baseline_dir(model)),
        "--input", "dataset",
        "--dataset", "mnist",
        "--digit", "0",
        "--sample", "0",
        # Longer than the 200 ms training window so PING's rhythm
        # has room to develop visibly.
        "--t-ms", "400",
    ]
    cmd = ["uv", "run", "python", str(OSCILLOSCOPE), *argv]
    print(f"[raster] {model}: {' '.join(argv)}")
    subprocess.run(cmd, cwd=REPO, check=True)
    if not npz_path.exists():
        raise SystemExit(f"oscilloscope did not produce {npz_path}")
    render_raster(npz_path, out_path, f"{model} — trained network, MNIST digit 0, 400 ms")



def plot_frontier(rows: list[dict], out_path: Path, run_id: str) -> None:
    """Pareto-style frontier: one line per model, baseline → tightest
    penalty. Both axes are final-epoch state. Baseline (θ_u = off)
    points show ±SEM error bars across baseline seeds; sweep points
    are single-seed and unbarred."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for model in MODELS:
        # Aggregate per (model, θ_u): for the baseline, mean ± SEM
        # across seeds. For sweep cells, just the single seed.
        agg_pts: list[dict] = []
        for theta_u in THETA_U_GRID:
            cell_rows = [
                r for r in rows
                if r["model"] == model and r["theta_u"] == theta_u
            ]
            if not cell_rows:
                continue
            accs = np.array([r["final_acc"] for r in cell_rows], dtype=float)
            rates = np.array([r["rate_e"] for r in cell_rows], dtype=float)
            n = len(cell_rows)
            agg_pts.append(
                {
                    "theta_display": cell_rows[0]["theta_display"],
                    "acc_mean": float(accs.mean()),
                    "rate_mean": float(rates.mean()),
                    "acc_sem": (
                        float(accs.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                    ),
                    "rate_sem": (
                        float(rates.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                    ),
                }
            )
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
                p["theta_display"],
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
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)




def _load_trained_full(train_dir: Path, device):
    """Load full state from a trained run (incl. W_ei/W_ie). Returns
    (net, cfg, X_te, y_te) ready for forward passes."""
    import torch

    import cli.config as C  # noqa: F401
    import models as M
    from cli.config import build_net, patch_dt
    from cli import load_dataset, seed_everything

    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", SEEDS_BASELINE[0])))
    M.T_ms = float(cfg["t_ms"])
    patch_dt(float(cfg["dt"]))
    hidden_sizes = cfg.get("hidden_sizes") or [int(cfg["n_hidden"])]
    M.N_HID = hidden_sizes[-1]
    M.N_INH = hidden_sizes[-1] // 4
    M.HIDDEN_SIZES = list(hidden_sizes)

    _, X_te, _, y_te = load_dataset(
        cfg["dataset"], max_samples=int(cfg["max_samples"]), split=True
    )
    M.N_IN = 784 if cfg["dataset"] == "mnist" else 64

    w_in_cfg = cfg.get("w_in")
    w_in_arg = (
        (float(w_in_cfg[0]), float(w_in_cfg[1]))
        if isinstance(w_in_cfg, list) and len(w_in_cfg) >= 2
        else None
    )
    net = build_net(
        cfg["model"],
        w_in=w_in_arg,
        w_in_sparsity=float(cfg.get("w_in_sparsity") or 0.0),
        ei_strength=float(cfg.get("ei_strength") or 1.0),
        ei_ratio=float(cfg.get("ei_ratio") or 2.0),
        sparsity=float(cfg.get("sparsity") or 0.0),
        device=device,
        randomize_init=not bool(cfg.get("kaiming_init", False)),
        dales_law=bool(cfg.get("dales_law", True)),
        hidden_sizes=hidden_sizes,
        readout_mode=cfg.get("readout_mode", "mem-mean"),
    )
    if hasattr(net, "readout_mode"):
        net.readout_mode = cfg.get("readout_mode", "mem-mean")

    state = torch.load(train_dir / "weights.pth", map_location=device)
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True
    return net, cfg, X_te, y_te


def _eval_net_on_test(net, cfg, X_te, y_te, device) -> tuple[float, float, float]:
    """Forward over test set; return (acc, hid_rate_hz, inh_rate_hz)."""
    acc, _ce, _pen, e_rate, i_rate = _eval_net_on_test_with_loss(
        net, cfg, X_te, y_te, device
    )
    return acc, e_rate, i_rate


def _eval_net_on_test_with_loss(
    net, cfg, X_te, y_te, device,
    fr_upper_theta: float = 0.0,
    fr_upper_strength: float = 0.0,
) -> tuple[float, float, float, float, float]:
    """Forward over test set; return
    (acc, ce_loss, penalty, hid_rate_hz, inh_rate_hz).

    `penalty` is the same firing-rate-upper regulariser the trainer applies:
    sum over hidden neurons of strength · ReLU(mean_per_neuron_spike_count -
    theta_u)^2, computed against the per-neuron spike-count mean over the
    full test set (one application, not per-batch averaging). When strength
    or theta_u is zero the penalty is zero."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    import models as M
    from cli import EVAL_SEED, encode_batch

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=64,
    )
    correct = total = 0
    ce_sum = 0.0
    e_spike_sum = i_spike_sum = 0.0
    # Per-neuron spike count accumulators (one tensor per hidden layer)
    # for computing the training-objective penalty term.
    sc_accums: list[torch.Tensor] = []
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
            logits = net(input_spikes=spk)
            ce_sum += float(
                F.cross_entropy(logits, y_b, reduction="sum").item()
            )
            correct += (logits.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
            e_spike_sum += float(net.spike_record["hid"].sum().item())
            if "inh" in net.spike_record:
                i_spike_sum += float(net.spike_record["inh"].sum().item())
            if fr_upper_strength > 0 and getattr(net, "last_spike_counts", None):
                if not sc_accums:
                    sc_accums = [
                        torch.zeros(sc.shape[1], device=device)
                        for sc in net.last_spike_counts
                    ]
                for acc_tensor, sc in zip(sc_accums, net.last_spike_counts):
                    acc_tensor += sc.sum(dim=0)
    n_e = M.N_HID
    n_i = M.N_INH or 1
    t_sec = float(cfg["t_ms"]) / 1000.0
    e_rate = e_spike_sum / (total * n_e * t_sec) if total else 0.0
    i_rate = i_spike_sum / (total * n_i * t_sec) if (total and i_spike_sum) else 0.0
    acc = 100.0 * correct / total if total else 0.0
    ce_loss = ce_sum / total if total else 0.0
    penalty = 0.0
    if sc_accums and total > 0 and fr_upper_strength > 0:
        for acc_tensor in sc_accums:
            mean_z = acc_tensor / total
            penalty += float(
                fr_upper_strength
                * (torch.relu(mean_z - fr_upper_theta) ** 2).sum().item()
            )
    return acc, ce_loss, penalty, e_rate, i_rate


# ── per-cycle participation p and gamma frequency f_γ vs θ_u ────────
#
# Decomposes the rate floor into its two factors: rate ≈ p · f_γ. As
# θ_u tightens, where does the optimiser act? The hypothesis is that p
# slides toward a viability minimum while f_γ stays put (biophysics
# fixed). When p hits the floor, accuracy collapses.

F_GAMMA_BAND_HZ: tuple[float, float] = (5.0, 150.0)
PFG_MAX_TRIALS: int = 256  # plenty for stable p and PSD peak


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


def measure_p_fgamma(train_dir: Path, device, is_ping: bool) -> dict:
    """Forward the test set; record E and I population spike trains.
    Compute:
      - f_γ via Welch PSD on E-population trace (PING only; NaN otherwise)
      - p = fraction of (cell, cycle) pairs with ≥ 1 spike
        (PING only — needs the I-burst peaks to delimit cycles)
      - acc, mean E rate, mean I rate as side measurements.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    import models as M
    from cli import EVAL_SEED, encode_batch

    net, cfg, X_te, y_te = _load_trained_full(train_dir, device)
    net.recording = True
    dt_ms = float(M.dt)
    t_ms = float(cfg["t_ms"])
    n_e = M.N_HID
    n_i = M.N_INH or 1
    fs_hz = 1000.0 / dt_ms
    t_sec = t_ms / 1000.0

    n_take = min(PFG_MAX_TRIALS, X_te.shape[0])
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_te[:n_take]), torch.from_numpy(y_te[:n_take])
        ),
        batch_size=64,
    )

    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    correct = total = 0
    e_spike_sum = i_spike_sum = 0.0
    pop_e_traces: list[np.ndarray] = []
    # Per-cycle (cell, cycle) tallies summed across trials.
    n_cycle_pairs = 0
    n_cycle_pairs_active = 0

    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
            logits = net(input_spikes=spk)
            correct += (logits.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
            s_e = net.spike_record["hid"]
            if s_e.ndim == 2:
                s_e = s_e.unsqueeze(1)
            e_spike_sum += float(s_e.sum().item())
            B = s_e.shape[1]
            # Population E trace per trial for PSD.
            pop = s_e.mean(dim=2).cpu().numpy()  # (T, B)
            for b in range(B):
                pop_e_traces.append(pop[:, b])
            if is_ping and "inh" in net.spike_record:
                s_i = net.spike_record["inh"]
                if s_i.ndim == 2:
                    s_i = s_i.unsqueeze(1)
                i_spike_sum += float(s_i.sum().item())
                s_e_np = s_e.cpu().numpy().astype(np.int8)  # (T, B, N_E)
                s_i_np = s_i.cpu().numpy().astype(np.int8)  # (T, B, N_I)
                # Quick f_γ guess per batch for cycle-peak detection. We
                # use the running mean f_γ estimate so peaks don't depend
                # on the final PSD — close enough; refined later.
                f_gamma_batch = _f_gamma_from_population(
                    [pop[:, b] for b in range(B)], fs_hz,
                )
                if not np.isfinite(f_gamma_batch) or f_gamma_batch <= 0:
                    continue
                for b in range(B):
                    peaks = _detect_i_burst_peaks(
                        s_i_np[:, b, :], dt_ms, f_gamma_batch,
                    )
                    if peaks.size == 0:
                        continue
                    counts = _count_e_spikes_per_cycle(
                        s_e_np[:, b, :], peaks,
                    )  # (K, N_E)
                    n_cycle_pairs += counts.size
                    n_cycle_pairs_active += int((counts > 0).sum())

    acc = 100.0 * correct / total if total else 0.0
    e_rate = e_spike_sum / (total * n_e * t_sec) if total else 0.0
    i_rate = (
        i_spike_sum / (total * n_i * t_sec) if (total and i_spike_sum) else 0.0
    )

    if is_ping:
        f_gamma_val = _f_gamma_from_population(pop_e_traces, fs_hz)
        f_gamma = (
            float(f_gamma_val)
            if np.isfinite(f_gamma_val) else None
        )
        p = (
            float(n_cycle_pairs_active) / float(n_cycle_pairs)
            if n_cycle_pairs > 0 else None
        )
    else:
        f_gamma = None
        p = None

    return {
        "acc": float(acc),
        "e_rate": float(e_rate),
        "i_rate": float(i_rate),
        "f_gamma": f_gamma,
        "p": p,
    }


def _detect_i_burst_peaks(
    s_i_trial: np.ndarray, dt_ms: float, f_gamma_hz: float,
) -> np.ndarray:
    """Mirrors nb046.detect_i_burst_steps. Smooth I population rate with
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
    """Mirrors nb046.count_e_spikes_per_cycle."""
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


def plot_theta_p_fgamma(
    rows: list[dict], out_path: Path, run_id: str,
) -> None:
    """4-panel decomposition vs θ_u (Hz):
      (top-left)  p vs θ_u (PING only) — the per-cycle participation gate
      (top-right) f_γ vs θ_u (PING only) — biophysics, untouched by θ_u
      (bottom-left)  E rate vs θ_u (both architectures); overlay p · f_γ
      (bottom-right) accuracy vs θ_u (both architectures)"""
    theme.apply()

    def by_model(model: str) -> list[dict]:
        sub = [r for r in rows if r["model"] == model and r["theta_u_hz"] is not None]
        sub.sort(key=lambda r: r["theta_u_hz"])
        return sub

    ping = by_model("ping")
    coba = by_model("coba")

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), dpi=150)
    (ax_p, ax_fg), (ax_r, ax_a) = axes

    if ping:
        ping_pf = [r for r in ping if r.get("p") is not None and r.get("f_gamma") is not None]
        xs = [r["theta_u_hz"] for r in ping_pf]
        ax_p.plot(xs, [r["p"] for r in ping_pf],
                  marker="o", color=theme.INK_BLACK, lw=1.5)
        ax_fg.plot(xs, [r["f_gamma"] for r in ping_pf],
                   marker="s", color=theme.DEEP_RED, lw=1.5)
        xs_all = [r["theta_u_hz"] for r in ping]
        ax_r.plot(xs_all, [r["e_rate"] for r in ping],
                  marker="o", color=theme.INK_BLACK, lw=1.5, label="PING E (measured)")
        ax_r.plot(xs, [r["p"] * r["f_gamma"] for r in ping_pf],
                  marker="^", color=theme.AMBER, lw=1.5, ls="--",
                  label="p × f_γ (predicted)")
        ax_a.plot(xs_all, [r["acc"] for r in ping],
                  marker="o", color=theme.INK_BLACK, lw=1.5, label="PING")

    if coba:
        xs = [r["theta_u_hz"] for r in coba]
        ax_r.plot(xs, [r["e_rate"] for r in coba],
                  marker="s", color=theme.DEEP_RED, lw=1.5, label="COBA E")
        ax_a.plot(xs, [r["acc"] for r in coba],
                  marker="s", color=theme.DEEP_RED, lw=1.5, label="COBA")

    for ax in (ax_p, ax_fg, ax_r, ax_a):
        ax.set_xlabel("θ_u (Hz)", fontsize=theme.SIZE_LABEL)
        ax.invert_xaxis()  # tightest penalty on the right read left → right
    ax_p.set_ylabel("p (per-cycle participation)", fontsize=theme.SIZE_LABEL)
    ax_p.set_title("Participation gate vs θ_u (PING)", fontsize=theme.SIZE_TITLE)
    ax_p.set_ylim(bottom=0)
    ax_fg.set_ylabel("f_γ (Hz)", fontsize=theme.SIZE_LABEL)
    ax_fg.set_title("Gamma frequency vs θ_u (PING)", fontsize=theme.SIZE_TITLE)
    ax_fg.set_ylim(bottom=0)
    ax_r.set_ylabel("E firing rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax_r.set_title("E rate vs θ_u — measured vs p · f_γ", fontsize=theme.SIZE_TITLE)
    ax_r.set_ylim(bottom=0)
    ax_r.legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="upper left")
    ax_a.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax_a.set_title("Accuracy vs θ_u", fontsize=theme.SIZE_TITLE)
    ax_a.set_ylim(0, 100)
    ax_a.legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="lower left")

    fig.suptitle(
        "Decomposing the rate floor under the spike penalty: "
        "θ_u presses on f_γ, p is architecturally protected, accuracy holds",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── low-w_in alternate-schedule sweep (PING under heavy θ_u) ────────
#
# Tests the path-dependent-barrier claim: if the network starts with
# W_in too small to recruit the I-loop and θ_u is on from epoch 0, can
# training land in the sub-f* COBA-like basin instead of locking into
# PING? Three w_in inits straddle f* (0.1 sub, 0.3 sub, 1.2 standard).

LOW_W_IN_VALUES: list[float] = [0.05, 0.1, 0.3, 1.2]  # 1.2 matches standard ping init
LOW_W_IN_THETA_U: float = 0.2                   # heaviest from frontier sweep
LOW_W_IN_SEED: int = SEED_SWEEP


def low_w_in_cell_dir(w_in: float) -> Path:
    label = f"{w_in:g}".replace(".", "p")
    return ARTIFACTS / f"ping__low_w_in__win{label}"


def build_low_w_in_args(w_in: float, tier: str, out_dir: Path) -> list[str]:
    """Train args for the low-w_in alternate-schedule sweep:
    PING recipe with --w-in overridden and θ_u = 0.2 on from epoch 0."""
    recipe = dict(MODEL_RECIPES["ping"])
    recipe["--w-in"] = f"{w_in:g}"
    args = [
        "train",
        "--model", recipe["__build_as"],
        "--dataset", "mnist",
        "--max-samples", str(TIER_CONFIG[tier]["max_samples"]),
        "--epochs", str(TIER_CONFIG[tier]["epochs"]),
        "--t-ms", str(T_MS),
        "--dt", str(DT_TRAIN),
        "--seed", str(LOW_W_IN_SEED),
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    for k, v in recipe.items():
        if k.startswith("__"):
            continue
        if v is True:
            args.append(k)
        elif v is not None:
            args += [k, v]
    args += [
        "--fr-reg-upper-theta", str(LOW_W_IN_THETA_U),
        "--fr-reg-upper-strength", str(FR_STRENGTH_UPPER),
    ]
    return args


def plot_low_w_in(rows: list[dict], out_path: Path, run_id: str) -> None:
    """2 rows × 3 cols. One column per --w-in init. Top row: per-epoch
    accuracy. Bottom row: per-epoch firing rates with E (black) and I
    (red) overlaid. Reads per-epoch traces from each run's metrics.json."""
    theme.apply()
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 5.5), dpi=150, sharex=True)
    rate_max = 0.0
    for col, row in enumerate(rows):
        metrics = load_metrics(low_w_in_cell_dir(row["w_in"]))
        epochs = list(range(1, len(metrics["epochs"]) + 1))
        accs = [float(e["acc"]) for e in metrics["epochs"]]
        # Prefer test-set rates (added to the trainer in train.py); fall
        # back to the single-trial observation rates for legacy runs.
        rate_e = [
            float(e.get("test_rate_e") if e.get("test_rate_e") is not None
                  else (e.get("rate_e") or 0.0))
            for e in metrics["epochs"]
        ]
        rate_i = [
            float(e.get("test_rate_i") if e.get("test_rate_i") is not None
                  else (e.get("rate_i") or 0.0))
            for e in metrics["epochs"]
        ]
        rate_max = max(rate_max, max(rate_e), max(rate_i))

        ax_acc = axes[0, col]
        ax_rate = axes[1, col]
        ax_acc.plot(epochs, accs, marker="o", color=theme.INK_BLACK, lw=1.5)
        ax_acc.axhline(10.0, color=theme.GREY_MID, lw=0.6, ls=":", alpha=0.5)
        ax_acc.set_ylim(0, 100)
        ax_acc.set_title(
            f"$W_\\text{{in}}$ = {row['w_in']:g}",
            fontsize=theme.SIZE_TITLE,
        )
        ax_rate.plot(epochs, rate_e, marker="o", color=theme.INK_BLACK,
                     lw=1.5, label="E")
        ax_rate.plot(epochs, rate_i, marker="s", color=theme.DEEP_RED,
                     lw=1.5, label="I")
        ax_rate.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
        if col == 0:
            ax_acc.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
            ax_rate.set_ylabel("Firing rate (Hz)", fontsize=theme.SIZE_LABEL)
            ax_rate.legend(fontsize=theme.SIZE_LABEL, frameon=False,
                           loc="upper left")

    for col in range(n):
        axes[1, col].set_ylim(0, rate_max * 1.1 if rate_max > 0 else 1.0)

    fig.suptitle(
        f"Per-epoch traces — PING, $\\theta_u = {LOW_W_IN_THETA_U:g}$, "
        "varying $W_\\text{in}$ init",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    fig.savefig(out_path, dpi=150)
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


def run_w_in_scale_sweep(notebook_run_id: str) -> list[dict]:
    """Inference-only sweep that multiplies each trained net's W_in
    (i.e. net.W_ff[0]) by every value in W_IN_SCALE_VALUES, evaluates on
    the test set, and records (cell, scale, loss, acc, rate_e, rate_i).
    Cells: PING at θ_u = off (baseline), COBA at θ_u = off (baseline),
    PING at θ_u = 0.2 (heaviest budget — the floor-pinned case).
    Restores the original W_in after each cell's sweep."""
    import torch
    from cli import _auto_device

    device = _auto_device()
    cells = [
        ("ping@tu0.2", cell_dir("ping", 0.2, SEED_SWEEP), 0.2),
        ("coba@tu0.2", cell_dir("coba", 0.2, SEED_SWEEP), 0.2),
    ]
    rows: list[dict] = []
    for label, train_dir, theta_u in cells:
        if not (train_dir / "weights.pth").exists():
            raise SystemExit(
                f"w_in-scale-sweep: missing weights for {label} at {train_dir}"
            )
        net, cfg, X_te, y_te = _load_trained_full(train_dir, device)
        w_in_original = net.W_ff[0].data.clone()
        try:
            for scale in W_IN_SCALE_VALUES:
                net.W_ff[0].data.copy_(w_in_original * float(scale))
                with torch.no_grad():
                    acc, ce_loss, penalty, e_rate, i_rate = (
                        _eval_net_on_test_with_loss(
                            net, cfg, X_te, y_te, device,
                            fr_upper_theta=theta_u,
                            fr_upper_strength=FR_STRENGTH_UPPER,
                        )
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
        finally:
            net.W_ff[0].data.copy_(w_in_original)
    return rows


def plot_w_in_scale_sweep(rows: list[dict], out_path: Path, run_id: str) -> None:
    """Six-panel: CE loss, penalty, total loss, accuracy, E rate, I rate
    vs W_in scale. One curve per (model, theta_u) cell."""
    theme.apply()
    fig, axes_2d = plt.subplots(2, 3, figsize=(12.0, 8.0), dpi=150)
    axes = axes_2d.flatten()
    styles = {
        "coba@tu0.2":  ("COBA ($\\theta_u = 0.2$)",
                        theme.DEEP_RED, "s", "-"),
        "ping@tu0.2":  ("PING ($\\theta_u = 0.2$)",
                        theme.INK_BLACK, "o", "-"),
    }
    # f* proxy on PING: the W_in scale where the I-population first
    # fires. Linear-interpolate between the largest s with I≈0 and the
    # smallest s with measurable I to estimate the crossing.
    ping_sorted = sorted(
        (r for r in rows if r["cell"] == "ping@tu0.2"),
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
    fig.suptitle(
        "Inference-time $W_\\text{in}$ scale sweep on trained networks "
        "(dashed line = trained $s = 1$)",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    fig.savefig(out_path, dpi=150)
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
    fig, axes_2d = plt.subplots(2, 3, figsize=(12.0, 8.0), dpi=150)
    axes = axes_2d.flatten()
    styles = {
        "coba@tu0.2":  ("COBA ($\\theta_u = 0.2$)",
                        theme.DEEP_RED, "s", "-"),
        "ping@tu0.2":  ("PING ($\\theta_u = 0.2$)",
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
    fig.suptitle(
        "Inference-time $W_\\text{in}$ scale sweep — replotted vs E rate "
        "(stars mark trained $s = 1$)",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)



# ── End W_in scale sweep ───────────────────────────────────────────

def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def copy_video(run_dir: Path, out_path: Path) -> None:
    src = run_dir / "training.mp4"
    if not src.exists():
        raise SystemExit(f"missing training video: {src}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_path)
    print(f"wrote {out_path}")


def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    skip_training = "--skip-training" in sys.argv
    wipe_dir = "--no-wipe-dir" not in sys.argv

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    n_cells = len(MODELS) * len(THETA_U_GRID)
    print(
        f"notebook_run_id = {notebook_run_id} tier={tier} cells={n_cells}"
        + ("  [skip-training]" if skip_training else "")
    )

    if wipe_dir:
        if skip_training:
            if FIGURES.exists():
                print(f"[wipe] {FIGURES.relative_to(REPO)}")
                shutil.rmtree(FIGURES)
        else:
            for d in (ARTIFACTS, FIGURES):
                if d.exists():
                    print(f"[wipe] {d.relative_to(REPO)}")
                    shutil.rmtree(d)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)

    only_missing = "--only-missing" in sys.argv
    if not skip_training:
        dispatcher = BatchDispatcher(modal_gpu, REPO, OSCILLOSCOPE)
        for model in MODELS:
            for theta_u in THETA_U_GRID:
                build_as = MODEL_RECIPES[model]["__build_as"]
                gpu_override = None
                if modal_gpu in ("T4", "L4", "A10G") and build_as == "ping":
                    gpu_override = "A100"
                for seed in seeds_for(theta_u):
                    out = cell_dir(model, theta_u, seed)
                    if only_missing and (out / "metrics.json").exists():
                        print(
                            f"[skip] {model}/θ_u={theta_display(theta_u)}/seed={seed} "
                            f"already trained → {out.relative_to(REPO)}"
                        )
                        continue
                    print(
                        f"[train] {model}/θ_u={theta_display(theta_u)}/seed={seed} → "
                        f"{out.relative_to(REPO)}"
                        + (f"  [modal:{modal_gpu}]" if modal_gpu else "")
                    )
                    dispatcher.submit(
                        build_train_args(model, theta_u, seed, tier, out),
                        out,
                        gpu_override=gpu_override,
                    )
        # Low-w_in alternate-schedule sweep dispatched in the same batch.
        gpu_override = None
        if modal_gpu in ("T4", "L4", "A10G"):
            gpu_override = "A100"
        for w_in in LOW_W_IN_VALUES:
            out = low_w_in_cell_dir(w_in)
            if only_missing and (out / "metrics.json").exists():
                print(
                    f"[skip] low_w_in/w_in={w_in} already trained → "
                    f"{out.relative_to(REPO)}"
                )
                continue
            print(
                f"[train] low_w_in/w_in={w_in} → {out.relative_to(REPO)}"
                + (f"  [modal:{modal_gpu}]" if modal_gpu else "")
            )
            dispatcher.submit(
                build_low_w_in_args(w_in, tier, out),
                out,
                gpu_override=gpu_override,
            )
        dispatcher.drain()

    rows: list[dict] = []
    for model in MODELS:
        for theta_u in THETA_U_GRID:
            for seed in seeds_for(theta_u):
                run_dir = cell_dir(model, theta_u, seed)
                if not (run_dir / "metrics.json").exists():
                    raise SystemExit(f"missing metrics: {run_dir / 'metrics.json'}")
                metrics = load_metrics(run_dir)
                last = metrics["epochs"][-1]
                rows.append(
                    {
                        "model": model,
                        "theta_u": theta_u,
                        "theta_display": theta_display(theta_u),
                        "theta_u_hz": theta_hz(theta_u),
                        "seed": seed,
                        "best_acc": float(metrics["best_acc"]),
                        "best_epoch": int(metrics["best_epoch"]),
                        "final_acc": float(last["acc"]),
                        "rate_e": float(last.get("rate_e") or 0.0),
                    }
                )
        # One training video per (model, θ_u) — use the canonical seed
        # (first of SEEDS_BASELINE for baselines, SEED_SWEEP for sweep).
        for theta_u in THETA_U_GRID:
            canonical_seed = seeds_for(theta_u)[0]
            copy_video(
                cell_dir(model, theta_u, canonical_seed),
                FIGURES / f"training__{model}__{theta_label(theta_u)}.mp4",
            )

    print("  results:")
    for r in rows:
        theta_str = (
            f"θ_u={r['theta_display']:>4} ({r['theta_u_hz']:>4.1f} Hz)"
            if r["theta_u"] is not None
            else "θ_u= off"
        )
        print(
            f"    {r['model']:<5}  {theta_str}  "
            f"acc(final)={r['final_acc']:6.2f}%  best={r['best_acc']:6.2f}%  "
            f"rate_e={r['rate_e']:6.1f} Hz"
        )

    plot_acc_rate_bars(rows, FIGURES / "acc_vs_rate.png", notebook_run_id)
    print(f"wrote {FIGURES / 'acc_vs_rate.png'}")
    plot_learning_curves(FIGURES / "learning_curves.png", notebook_run_id)
    print(f"wrote {FIGURES / 'learning_curves.png'}")
    plot_frontier(rows, FIGURES / "frontier.png", notebook_run_id)
    print(f"wrote {FIGURES / 'frontier.png'}")

    # θ_u vs (p, f_γ) decomposition — mechanism behind the frontier floor.
    print("[theta-pfg] measuring p and f_γ per (model, θ_u) cell")
    import torch
    pfg_device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    pfg_rows: list[dict] = []
    for model in MODELS:
        is_ping = (model == "ping")
        for theta_u in THETA_U_GRID:
            seed = seeds_for(theta_u)[0]
            train_dir = cell_dir(model, theta_u, seed)
            if not (train_dir / "weights.pth").exists():
                print(f"  skip {model} θ_u={theta_label(theta_u)} (no weights)")
                continue
            m = measure_p_fgamma(train_dir, pfg_device, is_ping=is_ping)
            row = {
                "model": model,
                "theta_u": theta_u,
                "theta_u_hz": theta_hz(theta_u),
                "seed": seed,
                **m,
            }
            pfg_rows.append(row)
            theta_str = (
                f"θ_u={theta_display(theta_u):>4} ({theta_hz(theta_u):>4.1f} Hz)"
                if theta_u is not None else "θ_u= off"
            )
            print(
                f"  {model:<5}  {theta_str}  "
                f"acc={m['acc']:5.2f}%  E={m['e_rate']:5.2f} Hz  "
                f"f_γ={m['f_gamma']:5.2f} Hz  p={m['p']:.3f}"
            )
    plot_theta_p_fgamma(
        pfg_rows, FIGURES / "theta_p_fgamma.png", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'theta_p_fgamma.png'}")

    for model in MODELS:
        out = FIGURES / f"raster__{model}.png"
        generate_raster(model, out)
        print(f"wrote {out}")

    # Publication-quality merged rasters figure — both baselines side by side.
    npz_coba = baseline_dir("coba") / "infer" / "snapshot.npz"
    npz_ping = baseline_dir("ping") / "infer" / "snapshot.npz"
    if npz_coba.exists() and npz_ping.exists():
        out = FIGURES / "baseline_rasters.png"
        render_baseline_rasters_combined(npz_coba, npz_ping, out)
        print(f"wrote {out}")

    # Low-w_in alternate-schedule sweep — reads metrics from the three
    # dispatched trainings and plots accuracy + E/I rates vs --w-in init.
    print("[low-w_in-sweep] reading metrics from dispatched trainings")
    low_w_in_rows: list[dict] = []
    for w_in in LOW_W_IN_VALUES:
        run_dir = low_w_in_cell_dir(w_in)
        if not (run_dir / "metrics.json").exists():
            raise SystemExit(f"missing metrics: {run_dir / 'metrics.json'}")
        metrics = load_metrics(run_dir)
        last = metrics["epochs"][-1]
        low_w_in_rows.append({
            "w_in": float(w_in),
            "best_acc": float(metrics["best_acc"]),
            "best_epoch": int(metrics["best_epoch"]),
            "final_acc": float(last["acc"]),
            "rate_e": float(last.get("rate_e") or 0.0),
            "rate_i": float(last.get("rate_i") or 0.0),
        })
        print(
            f"  w_in={w_in:>4}  acc={last['acc']:5.2f}%  "
            f"E={last.get('rate_e') or 0:6.2f} Hz  "
            f"I={last.get('rate_i') or 0:6.2f} Hz"
        )
    plot_low_w_in(low_w_in_rows, FIGURES / "low_w_in_sweep.png", notebook_run_id)
    print(f"wrote {FIGURES / 'low_w_in_sweep.png'}")

    # W_in scale sweep — direct test of the bifurcation argument.
    print("[w_in-scale-sweep] inference-only on trained PING and COBA")
    w_in_scale_rows = run_w_in_scale_sweep(notebook_run_id)
    plot_w_in_scale_sweep(
        w_in_scale_rows, FIGURES / "w_in_scale_sweep.png", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'w_in_scale_sweep.png'}")
    plot_w_in_scale_sweep_vs_rate(
        w_in_scale_rows,
        FIGURES / "w_in_scale_sweep_vs_rate.png",
        notebook_run_id,
    )
    print(f"wrote {FIGURES / 'w_in_scale_sweep_vs_rate.png'}")

    duration_s = time.monotonic() - t_start
    train_cfg = load_config(baseline_dir(MODELS[0]))
    summary = {
        "notebook_run_id": notebook_run_id,
        "git_sha": train_cfg.get("git_sha"),
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "tier": tier,
        "config": {
            "tier": tier,
            "dataset": "mnist",
            "models": MODELS,
            "theta_u_grid_spikes": [t for t in THETA_U_GRID if t is not None],
            "theta_u_grid_hz": [
                theta_hz(t) for t in THETA_U_GRID if t is not None
            ],
            "max_samples": TIER_CONFIG[tier]["max_samples"],
            "epochs": TIER_CONFIG[tier]["epochs"],
            "t_ms": T_MS,
            "dt": DT_TRAIN,
            "seeds_baseline": SEEDS_BASELINE,
            "seed_sweep": SEED_SWEEP,
            "fr_strength_upper": FR_STRENGTH_UPPER,
        },
        "results": rows,
        "theta_p_fgamma": pfg_rows,
        "low_w_in_sweep": low_w_in_rows,
        "w_in_scale_sweep": w_in_scale_rows,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")



if __name__ == "__main__":
    main()
