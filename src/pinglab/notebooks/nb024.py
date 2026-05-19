"""Notebook runner for entry 024 — coba / ping head-to-head on
test accuracy and mean firing rate, with the θ_u spike-budget sweep.

Subsumes the now-retired nb020. For each rung of the biophysical
ladder (coba, ping), trains the calibrated nb011 / nb012
recipe at six values of the upper-bound spike budget θ_u: off (no
penalty) plus θ_u ∈ {5, 2, 1, 0.5, 0.2} spikes/trial = {25, 10, 5,
2.5, 1} Hz. Same recipe in every other respect — only the regulariser
flag changes — so each (model, θ_u) cell is one point on that model's
accuracy / rate Pareto frontier.

The unpenalised "off" cell of each model also feeds the headline
figures: a twin-axis accuracy-vs-rate bar chart, learning curves
overlaid across the ladder, and per-model post-training spike
rasters.

Notebook entry: src/docs/src/pages/notebooks/nb024.mdx
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

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402
from pinglab import theme  # noqa: E402

SLUG = "nb024"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "cli/__main__.py"

TIER_CONFIG = {
    "extra small": dict(max_samples=100, epochs=1),
    "small": dict(max_samples=500, epochs=5),
    "medium": dict(max_samples=2000, epochs=10),
    "large": dict(max_samples=5000, epochs=40),
    "extra large": dict(max_samples=10000, epochs=40),
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
        "--epochs", str(TIER_CONFIG[tier]["epochs"]),
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


def generate_rate_sweep_video(model: str, out_path: Path) -> None:
    """Replay the trained baseline (θ_u = off) network on one MNIST digit
    while sweeping the input Poisson rate. The oscilloscope writes
    scan.mp4 into a per-call out-dir; we copy it to out_path."""
    artifact_dir = ARTIFACTS / f"rate_sweep__{model}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scan_mp4 = artifact_dir / "scan.mp4"
    if scan_mp4.exists():
        scan_mp4.unlink()
    argv = [
        "video",
        "--from-dir", str(baseline_dir(model)),
        "--input", "dataset",
        "--dataset", "mnist",
        "--digit", "0",
        "--sample", "0",
        "--scan-var", "spike_rate",
        "--scan-min", "0",
        "--scan-max", "100",
        "--frames", "40",
        "--frame-rate", "10",
        # 400 ms gives PING's loop room to settle at each rate.
        "--t-ms", "400",
        "--out-dir", str(artifact_dir),
    ]
    cmd = ["uv", "run", "python", str(OSCILLOSCOPE), *argv]
    print(f"[rate-sweep] {model}: {' '.join(argv)}")
    subprocess.run(cmd, cwd=REPO, check=True)
    if not scan_mp4.exists():
        raise SystemExit(f"oscilloscope did not produce {scan_mp4}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scan_mp4, out_path)


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


# ── COBA→PING ei_strength sweep (subsumes nb019) ──────────────────────


def _ei_sweep_dir() -> Path:
    return ARTIFACTS / "ei_sweep"


def run_inproc_infer(
    train_dir: Path, ei_strength: float, out_dir: Path
) -> dict:
    """Build a fresh ping net at the requested ei_strength, load only
    W_ff and W_ee from the trained coba checkpoint (skip W_ei/W_ie so
    the freshly-initialised I-loop survives), evaluate accuracy + mean
    E firing rate on the canonical test split.

    The CLI infer subcommand can't be used here because it always loads
    the full state dict; the trained-coba checkpoint has W_ei = W_ie = 0
    (init scaled by ei_strength=0, no gradient updates them) and loading
    those would nullify the I-loop at any inference ei_strength.
    """
    import torch

    import config as C  # noqa: F401
    import models as M
    from config import build_net, patch_dt
    from cli import (
        EVAL_SEED,
        _auto_device,
        encode_batch,
        load_dataset,
        seed_everything,
    )

    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", SEEDS_BASELINE[0])))
    M.T_ms = float(cfg["t_ms"])
    patch_dt(float(cfg["dt"]))

    hidden_sizes = cfg.get("hidden_sizes") or [int(cfg["n_hidden"])]
    M.N_HID = hidden_sizes[-1]
    M.N_INH = hidden_sizes[-1] // 4
    M.HIDDEN_SIZES = list(hidden_sizes)

    device = _auto_device()
    _, X_te, _, y_te = load_dataset(
        cfg["dataset"], max_samples=int(cfg["max_samples"]), split=True
    )
    M.N_IN = 784 if cfg["dataset"] == "mnist" else 64

    from torch.utils.data import DataLoader, TensorDataset

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=64,
    )

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
        ei_strength=float(ei_strength),
        ei_ratio=float(cfg.get("ei_ratio") or 2.0),
        sparsity=float(cfg.get("sparsity") or 0.0),
        device=device,
        randomize_init=not bool(cfg.get("kaiming_init", False)),
        kaiming_init=bool(cfg.get("kaiming_init", False)),
        dales_law=bool(cfg.get("dales_law", True)),
        hidden_sizes=hidden_sizes,
        w_rec=cfg.get("w_rec"),
        rec_layers=cfg.get("rec_layers"),
        ei_layers=cfg.get("ei_layers"),
    )
    if hasattr(net, "readout_mode"):
        net.readout_mode = cfg.get("readout_mode", "mem-mean")

    state = torch.load(train_dir / "weights.pth", map_location=device)
    skipped = {k: v for k, v in state.items() if k.startswith(("W_ei.", "W_ie."))}
    keep = {k: v for k, v in state.items() if k not in skipped}
    missing, unexpected = net.load_state_dict(keep, strict=False)
    print(
        f"  [transfer-load] loaded {len(keep)} keys, skipped "
        f"{sorted(skipped.keys())}; missing={list(missing)} "
        f"unexpected={list(unexpected)}"
    )

    net.eval()
    correct = total = 0
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    rate_sums: dict[str, float] = {}
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
            logits = net(input_spikes=spk)
            correct += (logits.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
            batch_rates = getattr(net, "rates", None) or {}
            B = y_b.size(0)
            for k, v in batch_rates.items():
                rate_sums[k] = rate_sums.get(k, 0.0) + float(v) * B

    acc = 100.0 * correct / total
    rates_hz = {k: v / total for k, v in rate_sums.items()} if total else {}
    hid_key = next((k for k in rates_hz if k.startswith("hid")), None)
    inh_key = next((k for k in rates_hz if k.startswith("inh")), None)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "mode": "infer",
        "ei_strength": ei_strength,
        "best_acc": acc,
        "n_correct": correct,
        "n_total": total,
        "rates_hz": rates_hz,
        "hid_rate_hz": rates_hz.get(hid_key) if hid_key else None,
        "inh_rate_hz": rates_hz.get(inh_key) if inh_key else None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"  ei={ei_strength:g}: acc={acc:.2f}%  "
        f"hid={metrics['hid_rate_hz']:.1f}Hz  "
        + (f"inh={metrics['inh_rate_hz']:.1f}Hz" if inh_key else "")
    )
    return metrics


def capture_ei_raster(train_dir: Path, ei_strength: float, sample_idx: int) -> dict:
    """Single-trial raster: build a fresh ping net at ei_strength, load
    the same selective state dict as run_inproc_infer, record one
    forward pass on a single test sample."""
    import torch

    import config as C  # noqa: F401
    import models as M
    from config import build_net, patch_dt
    from cli import (
        EVAL_SEED,
        _auto_device,
        encode_batch,
        load_dataset,
        seed_everything,
    )

    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", SEEDS_BASELINE[0])))
    M.T_ms = float(cfg["t_ms"])
    patch_dt(float(cfg["dt"]))
    hidden_sizes = cfg.get("hidden_sizes") or [int(cfg["n_hidden"])]
    M.N_HID = hidden_sizes[-1]
    M.N_INH = hidden_sizes[-1] // 4
    M.HIDDEN_SIZES = list(hidden_sizes)

    device = _auto_device()
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
        ei_strength=float(ei_strength),
        ei_ratio=float(cfg.get("ei_ratio") or 2.0),
        sparsity=float(cfg.get("sparsity") or 0.0),
        device=device,
        randomize_init=not bool(cfg.get("kaiming_init", False)),
        kaiming_init=bool(cfg.get("kaiming_init", False)),
        dales_law=bool(cfg.get("dales_law", True)),
        hidden_sizes=hidden_sizes,
    )
    if hasattr(net, "readout_mode"):
        net.readout_mode = cfg.get("readout_mode", "mem-mean")

    state = torch.load(train_dir / "weights.pth", map_location=device)
    keep = {k: v for k, v in state.items() if not k.startswith(("W_ei.", "W_ie."))}
    net.load_state_dict(keep, strict=False)
    net.eval()
    net.recording = True

    X_b = torch.from_numpy(X_te[sample_idx : sample_idx + 1]).to(device)
    y_b = int(y_te[sample_idx])
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    with torch.no_grad():
        spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
        _ = net(input_spikes=spk)

    e_full = net.spike_record["hid"].cpu().numpy()
    i_full = net.spike_record["inh"].cpu().numpy()
    rng = np.random.default_rng(0)
    e_idx = np.sort(rng.choice(e_full.shape[1], EI_RASTER_N_E_PLOT, replace=False))
    i_idx = np.sort(rng.choice(i_full.shape[1], EI_RASTER_N_I_PLOT, replace=False))
    return {
        "ei_strength": float(ei_strength),
        "label": y_b,
        "e": e_full[:, e_idx].astype(bool),
        "i": i_full[:, i_idx].astype(bool),
        "dt": float(cfg["dt"]),
        "t_ms": float(cfg["t_ms"]),
    }


def capture_rate_raster(train_dir: Path, spike_rate: float, sample_idx: int) -> dict:
    """Single-trial raster: load the trained ping baseline, override
    M.max_rate_hz, run one forward pass on a single test sample."""
    import torch

    import config as C  # noqa: F401
    import models as M
    from config import build_net, patch_dt
    from cli import (
        EVAL_SEED,
        _auto_device,
        encode_batch,
        load_dataset,
        seed_everything,
    )

    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", SEEDS_BASELINE[0])))
    M.T_ms = float(cfg["t_ms"])
    patch_dt(float(cfg["dt"]))
    hidden_sizes = cfg.get("hidden_sizes") or [int(cfg["n_hidden"])]
    M.N_HID = hidden_sizes[-1]
    M.N_INH = hidden_sizes[-1] // 4
    M.HIDDEN_SIZES = list(hidden_sizes)
    M.max_rate_hz = float(spike_rate)
    M.p_scale = M.max_rate_hz * M.dt / 1000.0

    device = _auto_device()
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
        kaiming_init=bool(cfg.get("kaiming_init", False)),
        dales_law=bool(cfg.get("dales_law", True)),
        hidden_sizes=hidden_sizes,
    )
    if hasattr(net, "readout_mode"):
        net.readout_mode = cfg.get("readout_mode", "mem-mean")

    state = torch.load(train_dir / "weights.pth", map_location=device)
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True

    X_b = torch.from_numpy(X_te[sample_idx : sample_idx + 1]).to(device)
    y_b = int(y_te[sample_idx])
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    with torch.no_grad():
        spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
        _ = net(input_spikes=spk)

    e_full = net.spike_record["hid"].cpu().numpy()
    i_full = net.spike_record["inh"].cpu().numpy()
    # Mean E firing rate over the trial, in Hz.
    e_rate_hz = float(e_full.sum() / (e_full.shape[1] * cfg["t_ms"] / 1000.0))
    rng = np.random.default_rng(0)
    e_idx = np.sort(rng.choice(e_full.shape[1], EI_RASTER_N_E_PLOT, replace=False))
    i_idx = np.sort(rng.choice(i_full.shape[1], EI_RASTER_N_I_PLOT, replace=False))
    return {
        "spike_rate": float(spike_rate),
        "e_rate_hz": e_rate_hz,
        "label": y_b,
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
        n, 1, figsize=(10.0, 5.625),
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
            1.012, 0.5,
            f"input = {s['spike_rate']:.1f} Hz\nE = {s['e_rate_hz']:.1f} Hz",
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
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


PERTURB_DROP_LEVELS: list[float] = [round(x * 0.1, 2) for x in range(11)]  # 0.0..1.0
PERTURB_ADD_LEVELS: list[float] = [float(x) for x in range(0, 41, 2)]  # 0..40 Hz, 2 Hz steps
PERTURB_RASTER_DROP_LEVELS: list[float] = [0.0, 0.3, 0.6, 0.8, 0.9, 1.0]
PERTURB_RASTER_ADD_LEVELS: list[float] = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0]


def capture_perturbation_raster(
    train_dir: Path, mode: str, level: float, sample_idx: int = 0
) -> dict:
    """Single-trial raster with the hidden-spike perturbation hook active.

    Same build / load / forward sequence as capture_rate_raster, with
    _hidden_perturb_fn installed for the duration of the forward pass.
    """
    import torch

    import config as C  # noqa: F401
    import models as M
    from config import build_net, patch_dt
    from cli import (
        EVAL_SEED,
        _auto_device,
        encode_batch,
        load_dataset,
        seed_everything,
    )

    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", SEEDS_BASELINE[0])))
    M.T_ms = float(cfg["t_ms"])
    patch_dt(float(cfg["dt"]))
    hidden_sizes = cfg.get("hidden_sizes") or [int(cfg["n_hidden"])]
    M.N_HID = hidden_sizes[-1]
    M.N_INH = hidden_sizes[-1] // 4
    M.HIDDEN_SIZES = list(hidden_sizes)

    device = _auto_device()
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
        kaiming_init=bool(cfg.get("kaiming_init", False)),
        dales_law=bool(cfg.get("dales_law", True)),
        hidden_sizes=hidden_sizes,
    )
    if hasattr(net, "readout_mode"):
        net.readout_mode = cfg.get("readout_mode", "mem-mean")

    state = torch.load(train_dir / "weights.pth", map_location=device)
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True

    perturb_gen = torch.Generator(device=device).manual_seed(EVAL_SEED + 1)
    net._hidden_perturb_fn = _make_perturb_fn(mode, level, M.dt, perturb_gen)

    X_b = torch.from_numpy(X_te[sample_idx : sample_idx + 1]).to(device)
    y_b = int(y_te[sample_idx])
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    with torch.no_grad():
        spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
        _ = net(input_spikes=spk)

    net._hidden_perturb_fn = None

    e_full = net.spike_record["hid"].cpu().numpy()
    i_full = net.spike_record["inh"].cpu().numpy()
    e_rate_hz = float(e_full.sum() / (e_full.shape[1] * cfg["t_ms"] / 1000.0))
    rng = np.random.default_rng(0)
    e_idx = np.sort(rng.choice(e_full.shape[1], EI_RASTER_N_E_PLOT, replace=False))
    i_idx = np.sort(rng.choice(i_full.shape[1], EI_RASTER_N_I_PLOT, replace=False))
    return {
        "mode": mode,
        "level": float(level),
        "e_rate_hz": e_rate_hz,
        "label": y_b,
        "e": e_full[:, e_idx].astype(bool),
        "i": i_full[:, i_idx].astype(bool),
        "dt": float(cfg["dt"]),
        "t_ms": float(cfg["t_ms"]),
    }


def plot_perturbation_rasters(
    samples: list[dict], out_path: Path, run_id: str, level_fmt: str, title: str
) -> None:
    """Stacked single-trial rasters across perturbation levels for one mode."""
    theme.apply()
    n = len(samples)
    n_e = EI_RASTER_N_E_PLOT
    n_i = EI_RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n, 1, figsize=(10.0, 5.625),
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
            1.012, 0.5,
            level_fmt.format(level=s["level"]) + f"\nE = {s['e_rate_hz']:.1f} Hz",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=theme.SIZE_LABEL,
        )
        if i == 0:
            ax.set_title(title)
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _make_perturb_fn(mode: str, level: float, dt_ms: float, generator):
    """Return a per-step callback (s_e, s_i, layer_idx) -> (s_e', s_i').

    Both populations get the same perturbation. The callback runs inside
    the COBANet step body right after spikes are emitted and before they
    feed into the readout / recurrence / I-loop, so the perturbation is
    dynamics-faithful: downstream state reacts to it within the trial.
    """
    import torch

    if mode == "drop":
        def fn(s_e, s_i, _layer):
            mask = (
                torch.rand(s_e.shape, generator=generator, device=s_e.device)
                >= level
            ).float()
            s_e = s_e * mask
            if s_i is not None:
                mask_i = (
                    torch.rand(s_i.shape, generator=generator, device=s_i.device)
                    >= level
                ).float()
                s_i = s_i * mask_i
            return s_e, s_i

        return fn

    if mode == "add":
        p = level * dt_ms / 1000.0
        def fn(s_e, s_i, _layer):
            extra_e = (
                torch.rand(s_e.shape, generator=generator, device=s_e.device) < p
            ).float()
            s_e = torch.clamp(s_e + extra_e, 0.0, 1.0)
            if s_i is not None:
                extra_i = (
                    torch.rand(s_i.shape, generator=generator, device=s_i.device) < p
                ).float()
                s_i = torch.clamp(s_i + extra_i, 0.0, 1.0)
            return s_e, s_i

        return fn

    raise ValueError(f"unknown perturbation mode {mode!r}")


def run_perturbation_sweep(
    train_dir: Path, mode: str, level: float
) -> dict:
    """Evaluate the test set under a hidden-spike perturbation.

    Builds the trained network, attaches the perturbation hook, runs one
    forward pass over the whole test set, and returns accuracy plus the
    achieved hidden E firing rate (perturbation can shift either).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    import config as C  # noqa: F401
    import models as M
    from config import build_net, patch_dt
    from cli import (
        EVAL_SEED,
        _auto_device,
        encode_batch,
        load_dataset,
        seed_everything,
    )

    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", SEEDS_BASELINE[0])))
    M.T_ms = float(cfg["t_ms"])
    patch_dt(float(cfg["dt"]))
    hidden_sizes = cfg.get("hidden_sizes") or [int(cfg["n_hidden"])]
    M.N_HID = hidden_sizes[-1]
    M.N_INH = hidden_sizes[-1] // 4
    M.HIDDEN_SIZES = list(hidden_sizes)

    device = _auto_device()
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
        kaiming_init=bool(cfg.get("kaiming_init", False)),
        dales_law=bool(cfg.get("dales_law", True)),
        hidden_sizes=hidden_sizes,
    )
    if hasattr(net, "readout_mode"):
        net.readout_mode = cfg.get("readout_mode", "mem-mean")

    state = torch.load(train_dir / "weights.pth", map_location=device)
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True

    perturb_gen = torch.Generator(device=device).manual_seed(EVAL_SEED + 1)
    net._hidden_perturb_fn = _make_perturb_fn(mode, level, M.dt, perturb_gen)

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=64,
    )

    correct = total = 0
    e_spike_sum = 0.0
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
            logits = net(input_spikes=spk)
            correct += (logits.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
            # Sum E spikes across the recorded hidden raster. Last hidden
            # layer feeds the readout; report its rate.
            last_key = f"hid_{net.n_layers}" if net.n_layers > 1 else "hid"
            hid_rec = net.spike_record.get(last_key)
            if hid_rec is None:
                hid_rec = net.spike_record["hid"]
            e_spike_sum += float(hid_rec.sum().item())

    net._hidden_perturb_fn = None  # unset so subsequent runs aren't poisoned

    n_e = hidden_sizes[-1]
    t_sec = float(cfg["t_ms"]) / 1000.0
    e_rate_hz = e_spike_sum / (total * n_e * t_sec) if total else 0.0
    acc = 100.0 * correct / total if total else 0.0
    return {
        "mode": mode,
        "level": level,
        "acc": acc,
        "e_rate_hz": e_rate_hz,
        "n_total": total,
    }


def plot_perturbation_curves(
    points: list[dict], out_path: Path, run_id: str
) -> None:
    """Two-panel accuracy plot: drop on the left, add on the right.

    One curve per model (coba, ping). The x-axis is the perturbation
    level in its native units (probability for drop, Hz for add).
    """
    theme.apply()
    # 16:9 styleguide ratio, sized for two side-by-side panels.
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.5), sharey=True, dpi=150)
    panel_specs = [
        ("drop", "Drop — Bernoulli spike mask",
         "p(drop) per spike", (-0.02, 1.02)),
        ("add", "Add — Poisson noise injection",
         "Poisson rate (Hz / neuron)", None),
    ]
    for ax, (mode, title, xlabel, xlim) in zip(axes, panel_specs):
        for model in MODELS:
            rows = [
                p for p in points if p["model"] == model and p["mode"] == mode
            ]
            rows.sort(key=lambda p: p["level"])
            xs = [p["level"] for p in rows]
            ys = [p["acc"] for p in rows]
            ax.plot(
                xs, ys,
                marker=MODEL_MARKERS[model],
                markersize=5,
                linewidth=1.4,
                color=MODEL_COLORS[model],
                label=model,
            )
        ax.axhline(10.0, ls="--", color=theme.MUTED, lw=0.7, alpha=0.6)
        # Light annotation of the chance line, only on the left panel
        # to avoid duplicate ink.
        if mode == "drop":
            ax.text(
                0.02, 12, "chance", transform=ax.get_yaxis_transform(),
                fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED,
                va="bottom",
            )
        ax.set_xlabel(xlabel, fontsize=theme.SIZE_LABEL)
        ax.set_title(title, fontsize=theme.SIZE_LABEL, loc="left", pad=4)
        ax.set_ylim(0, 100)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.tick_params(labelsize=theme.SIZE_TICK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(20))
        ax.grid(True, axis="y", alpha=0.15, linewidth=0.5)
    axes[0].set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    axes[1].legend(
        loc="upper right",
        fontsize=theme.SIZE_LEGEND,
        frameon=False,
    )
    fig.suptitle(
        "Hidden-spike perturbation — accuracy vs perturbation level",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ei_rasters(samples: list[dict], out_path: Path, run_id: str) -> None:
    """One row per ei value; I units stack over E units so the PING-style
    E-then-I cadence reads as alternating bursts when it appears."""
    theme.apply()
    n = len(samples)
    n_e = EI_RASTER_N_E_PLOT
    n_i = EI_RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n, 1, figsize=(10.0, 5.625),
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
        if i == 0:
            ax.set_title(
                "E (black) and I (red) spikes — single trial, MNIST test sample 0"
            )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ei_acc_sweep(points: list[dict], out_path: Path, run_id: str) -> None:
    theme.apply()
    eis = [p["ei_strength"] for p in points]
    accs = [p["acc"] for p in points]
    base_acc = points[0]["acc"]
    worst = min(points, key=lambda p: p["acc"])
    y_hi = min(max(accs + [base_acc]) + 6, 100)
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
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
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ei_rates_sweep(points: list[dict], out_path: Path, run_id: str) -> None:
    theme.apply()
    eis = [p["ei_strength"] for p in points]
    hid = [p.get("hid_rate_hz") or 0.0 for p in points]
    inh = [p.get("inh_rate_hz") or 0.0 for p in points]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(eis, hid, marker="o", color=theme.INK_BLACK, label="E (hidden)")
    ax.plot(eis, inh, marker="s", color=theme.DEEP_RED, label="I (inhibitory)")
    ax.set_xlabel("inference E→I strength")
    ax.set_ylabel("mean population rate (Hz)")
    ax.set_title("E and I population rates across the I-loop sweep")
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([round(0.1 * i, 1) for i in range(11)])
    ax.set_ylim(0, max(hid + inh) * 1.12 + 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _stamp(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_ei_sweep(notebook_run_id: str) -> list[dict]:
    """In-process inference sweep on the coba__off__seed42 baseline.
    Generates acc_sweep.png, rates_sweep.png, ei_rasters.png and returns
    the per-ei result rows."""
    train_dir = baseline_dir("coba")
    if not (train_dir / "weights.pth").exists():
        raise SystemExit(
            f"ei-sweep needs trained coba weights at {train_dir}; "
            "run training first or check baseline_dir naming."
        )
    sweep_root = _ei_sweep_dir()
    sweep_root.mkdir(parents=True, exist_ok=True)

    points: list[dict] = []
    for ei in EI_SWEEP:
        out = sweep_root / f"infer_ei{ei:g}"
        print(f"[ei-sweep] ei={ei} → {out.relative_to(REPO)}")
        m = run_inproc_infer(train_dir, ei, out)
        points.append(
            {
                "ei_strength": ei,
                "acc": m["best_acc"],
                "hid_rate_hz": m.get("hid_rate_hz"),
                "inh_rate_hz": m.get("inh_rate_hz"),
                "n_total": m.get("n_total"),
            }
        )

    print(f"[ei-sweep] capturing single-trial rasters for ei ∈ {EI_RASTER}")
    raster_samples = [
        capture_ei_raster(train_dir, ei, EI_RASTER_SAMPLE_IDX) for ei in EI_RASTER
    ]

    plot_ei_acc_sweep(points, FIGURES / "ei_acc_sweep.png", notebook_run_id)
    print(f"wrote {FIGURES / 'ei_acc_sweep.png'}")
    plot_ei_rates_sweep(points, FIGURES / "ei_rates_sweep.png", notebook_run_id)
    print(f"wrote {FIGURES / 'ei_rates_sweep.png'}")
    plot_ei_rasters(raster_samples, FIGURES / "ei_rasters.png", notebook_run_id)
    print(f"wrote {FIGURES / 'ei_rasters.png'}")
    return points


# ── End ei sweep ────────────────────────────────────────────────────


# ── tau_GABA sweep (inference-only, trained ping) ───────────────────

TAU_GABA_VALUES: list[float] = [4.5, 6.0, 9.0, 12.0, 18.0, 27.0]  # ms; default 9.0


def _load_trained_full(train_dir: Path, device):
    """Load full state from a trained run (incl. W_ei/W_ie). Returns
    (net, cfg, X_te, y_te) ready for forward passes."""
    import torch

    import config as C  # noqa: F401
    import models as M
    from config import build_net, patch_dt
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
        kaiming_init=bool(cfg.get("kaiming_init", False)),
        dales_law=bool(cfg.get("dales_law", True)),
        hidden_sizes=hidden_sizes,
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


def run_tau_gaba_sweep(notebook_run_id: str) -> list[dict]:
    """Inference-only τ_GABA sweep on trained ping. Tests the
    dynamics-bound floor claim: rate floor should track cycle period."""
    import numpy as np

    import models as M
    from cli import _auto_device

    train_dir = baseline_dir("ping")
    if not (train_dir / "weights.pth").exists():
        raise SystemExit(
            f"tau_gaba-sweep needs trained ping weights at {train_dir}"
        )
    device = _auto_device()
    rows: list[dict] = []
    # Snapshot module state so we can restore it after the sweep — otherwise
    # downstream code (or follow-on sweeps) inherits our mutations.
    tau_gaba_saved = M.tau_gaba
    decay_gaba_saved = M.decay_gaba
    try:
        for tau_gaba_ms in TAU_GABA_VALUES:
            net, cfg, X_te, y_te = _load_trained_full(train_dir, device)
            # _load_trained_full calls patch_dt → recomputes decay_gaba from
            # whatever M.tau_gaba currently is. Override AFTER load so our
            # value sticks for this iteration's forward pass.
            M.tau_gaba = float(tau_gaba_ms)
            M.decay_gaba = float(np.exp(-M.dt / tau_gaba_ms))
            acc, e_rate, i_rate = _eval_net_on_test(net, cfg, X_te, y_te, device)
            rows.append({
                "tau_gaba_ms": float(tau_gaba_ms),
                "acc": acc,
                "hid_rate_hz": e_rate,
                "inh_rate_hz": i_rate,
            })
            print(
                f"  τ_GABA={tau_gaba_ms:>5.1f} ms  "
                f"acc={acc:5.2f}%  E={e_rate:6.2f} Hz  I={i_rate:6.2f} Hz"
            )
    finally:
        M.tau_gaba = tau_gaba_saved
        M.decay_gaba = decay_gaba_saved
    return rows


def _plot_sweep_two_axes(
    rows: list[dict], x_key: str, x_label: str, x_vline: float | None,
    title: str, out_path: Path, run_id: str,
) -> None:
    """Shared plot: x-axis sweep var, left y E rate (red), right y accuracy (black)."""
    fig, ax_rate = plt.subplots(figsize=(8, 4.5), dpi=150)
    xs = [r[x_key] for r in rows]
    e_rate = [r["hid_rate_hz"] for r in rows]
    acc = [r["acc"] for r in rows]
    ax_rate.plot(xs, e_rate, color=theme.DEEP_RED, marker="o", lw=1.5)
    ax_rate.set_xlabel(x_label, fontsize=theme.SIZE_LABEL)
    ax_rate.set_ylabel("Hidden E rate (Hz)",
                       fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED)
    ax_rate.tick_params(axis="y", labelcolor=theme.DEEP_RED)
    if x_vline is not None:
        ax_rate.axvline(x_vline, color=theme.GREY_MID, lw=0.6, ls="--", alpha=0.7)
        ymax = ax_rate.get_ylim()[1]
        ax_rate.text(
            x_vline, ymax * 0.95, " training value",
            fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED, va="top",
        )
    ax_acc = ax_rate.twinx()
    ax_acc.plot(xs, acc, color=theme.INK_BLACK, marker="s", lw=1.5)
    ax_acc.axhline(10.0, color=theme.GREY_MID, lw=0.6, ls=":", alpha=0.5)
    ax_acc.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax_acc.set_ylim(0, 100)
    fig.suptitle(title, fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    _stamp(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── End tau_GABA ───────────────────────────────────────────────────


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


def evaluate_success(rows: list[dict], tier: str, figures: Path) -> list[dict]:
    floor = float(MIN_ACC_BY_TIER[tier])
    figs_root = figures.parents[2]

    def artifact(name: str, label: str) -> dict:
        path = figures / name
        ok = path.exists() and path.stat().st_size > 0
        href = "/" + str(path.relative_to(figs_root)) if ok else None
        return {
            "label": label,
            "passed": bool(ok),
            "detail": (
                f"{path.name} ({path.stat().st_size} bytes)"
                if ok else f"missing {path.name}"
            ),
            "detail_href": href,
        }

    crits: list[dict] = [
        artifact("acc_vs_rate.png", "bar chart rendered"),
        artifact("learning_curves.png", "learning curves rendered"),
        artifact("frontier.png", "frontier rendered"),
        artifact("ei_acc_sweep.png", "ei-sweep accuracy rendered"),
        artifact("ei_rates_sweep.png", "ei-sweep rates rendered"),
        artifact("ei_rasters.png", "ei-sweep rasters rendered"),
        artifact("tau_gaba_sweep.png", "tau_GABA sweep rendered"),
        artifact("low_w_in_sweep.png", "low-w_in sweep rendered"),
        artifact("w_in_scale_sweep.png", "W_in scale sweep rendered"),
        artifact(
            "w_in_scale_sweep_vs_rate.png",
            "W_in scale sweep vs E rate rendered",
        ),
    ]
    for model in MODELS:
        crits.append(artifact(f"raster__{model}.png", f"{model} raster rendered"))
    for model in MODELS:
        for theta_u in THETA_U_GRID:
            crits.append(
                artifact(
                    f"training__{model}__{theta_label(theta_u)}.mp4",
                    f"{model} θ_u={theta_display(theta_u)}: training video",
                )
            )

    for model in MODELS:
        base = next(
            r for r in rows if r["model"] == model and r["theta_u"] is None
        )
        crits.append(
            {
                "label": f"{model} baseline acc ≥ {floor:.0f}% ({tier} floor)",
                "passed": bool(base["best_acc"] >= floor),
                "detail": f"{model}={base['best_acc']:.2f}%",
            }
        )
    # Frontier monotonicity: tighter budget cannot give higher rate.
    for model in MODELS:
        ordered = [
            next(r for r in rows if r["model"] == model and r["theta_u"] == t)
            for t in [tu for tu in THETA_U_GRID if tu is not None]
        ]
        ordered.sort(key=lambda r: -r["theta_u"])
        rates = [r["rate_e"] for r in ordered]
        non_increasing = all(b <= a + 1.0 for a, b in zip(rates, rates[1:]))
        crits.append(
            {
                "label": f"{model} rate non-increasing as θ_u tightens (±1 Hz)",
                "passed": non_increasing,
                "detail": "rates(loose→tight): "
                + ", ".join(f"{r:.1f}" for r in rates),
            }
        )
    return crits


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

    for model in MODELS:
        out = FIGURES / f"raster__{model}.png"
        generate_raster(model, out)
        print(f"wrote {out}")

    # Input-rate sweep on the trained ping network — one digit, vary the
    # Poisson rate over a wide range, render as a scope-frame video.
    rate_sweep_out = FIGURES / "rate_sweep__ping.mp4"
    generate_rate_sweep_video("ping", rate_sweep_out)
    print(f"wrote {rate_sweep_out}")

    # Stacked raster snapshot at the first 10 frames of the rate sweep —
    # same panel style as the ei-sweep rasters so the two read as a pair.
    rate_grid = np.linspace(0.0, 100.0, 40)[:10]
    print(f"[rate-rasters] capturing rates {[round(r, 2) for r in rate_grid]}")
    rate_samples = [
        capture_rate_raster(baseline_dir("ping"), float(r), sample_idx=0)
        for r in rate_grid
    ]
    plot_rate_rasters(
        rate_samples, FIGURES / "rate_rasters__ping.png", notebook_run_id
    )
    print(f"wrote {FIGURES / 'rate_rasters__ping.png'}")

    # Hidden-layer perturbation sweep: drop spikes (Bernoulli mask) and
    # add Poisson noise spikes, applied inside the forward loop so the
    # I-population and readout both react.
    print("[perturb] hidden-spike drop + add sweep (coba, ping)")
    perturb_rows: list[dict] = []
    for model in MODELS:
        train_dir = baseline_dir(model)
        for mode, levels in (
            ("drop", PERTURB_DROP_LEVELS),
            ("add", PERTURB_ADD_LEVELS),
        ):
            for level in levels:
                res = run_perturbation_sweep(train_dir, mode, level)
                res["model"] = model
                perturb_rows.append(res)
                print(
                    f"  {model:<5} {mode:<4} level={level:>5.2f}  "
                    f"acc={res['acc']:5.2f}%  E={res['e_rate_hz']:6.2f} Hz"
                )
    plot_perturbation_curves(
        perturb_rows, FIGURES / "perturbation_curves.png", notebook_run_id
    )
    print(f"wrote {FIGURES / 'perturbation_curves.png'}")

    # Stacked-raster snapshots of each trained baseline under both
    # perturbation modes, six panels per (model, mode). Same MNIST digit 0
    # sample 0 as the other rasters so the panels read against the
    # unperturbed baselines (Figures 4-5).
    for model in MODELS:
        train_dir = baseline_dir(model)
        drop_samples = [
            capture_perturbation_raster(train_dir, "drop", lvl, 0)
            for lvl in PERTURB_RASTER_DROP_LEVELS
        ]
        plot_perturbation_rasters(
            drop_samples,
            FIGURES / f"perturb_rasters__drop__{model}.png",
            notebook_run_id,
            level_fmt="p(drop) = {level:.1f}",
            title=(
                f"E (black) and I (red) spikes — trained {model} with "
                "hidden-spike drop"
            ),
        )
        print(f"wrote {FIGURES / f'perturb_rasters__drop__{model}.png'}")
        add_samples = [
            capture_perturbation_raster(train_dir, "add", lvl, 0)
            for lvl in PERTURB_RASTER_ADD_LEVELS
        ]
        plot_perturbation_rasters(
            add_samples,
            FIGURES / f"perturb_rasters__add__{model}.png",
            notebook_run_id,
            level_fmt="r(add) = {level:g} Hz",
            title=(
                f"E (black) and I (red) spikes — trained {model} with "
                "hidden-spike Poisson noise added"
            ),
        )
        print(f"wrote {FIGURES / f'perturb_rasters__add__{model}.png'}")

    ei_points = run_ei_sweep(notebook_run_id)

    # τ_GABA sweep — tests the "loop sets the floor" claim.
    print("[tau-gaba-sweep] inference-only on trained ping")
    tau_gaba_rows = run_tau_gaba_sweep(notebook_run_id)
    _plot_sweep_two_axes(
        tau_gaba_rows, "tau_gaba_ms",
        "$\\tau_{\\mathrm{GABA}}$ (ms)", 9.0,
        "Trained ping — accuracy and E rate vs $\\tau_{\\mathrm{GABA}}$",
        FIGURES / "tau_gaba_sweep.png", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'tau_gaba_sweep.png'}")

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
    crits = evaluate_success(rows, tier, FIGURES)
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
        "ei_sweep": ei_points,
        "perturbation": perturb_rows,
        "tau_gaba_sweep": tau_gaba_rows,
        "low_w_in_sweep": low_w_in_rows,
        "w_in_scale_sweep": w_in_scale_rows,
        "success_criteria": crits,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")

    for c in crits:
        mark = "pass" if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['label']} — {c['detail']}")
    if any(not c["passed"] for c in crits):
        sys.exit(1)


if __name__ == "__main__":
    main()
