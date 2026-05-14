"""Notebook runner for entry 024 — cuba / coba / ping head-to-head on
test accuracy and mean firing rate, with the θ_u spike-budget sweep.

Subsumes the now-retired nb020. For each rung of the biophysical
ladder (cuba, coba, ping), trains the calibrated nb010 / nb011 / nb012
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
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope/__main__.py"

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
# pressure (off → ~80–90 Hz baselines for cuba/coba) down to 1 Hz —
# below ping's natural 5 Hz and into the regime where every model
# loses accuracy.
THETA_U_GRID: list[float | None] = [None, 5.0, 2.0, 1.0, 0.5, 0.2]
FR_STRENGTH_UPPER = 1e-3

MODELS = ["cuba", "coba", "ping"]

MODEL_RECIPES: dict[str, dict] = {
    "cuba": {
        "__build_as": "cuba",
        "--kaiming-init": True,
        "--readout": "mem-mean",
        "--surrogate-slope": "1",
        "--lr": "0.04",
        "--batch-size": "256",
    },
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
    "cuba": theme.DEEP_RED,
    "coba": theme.AMBER,
    "ping": theme.ELECTRIC_CYAN,
}
MODEL_MARKERS = {"cuba": "o", "coba": "s", "ping": "D"}

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


def baseline_dirs(model: str) -> list[Path]:
    return [cell_dir(model, None, s) for s in SEEDS_BASELINE]


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

    ax_acc.bar(
        xs - width / 2, accs, width=width,
        color=[MODEL_COLORS[m] for m in MODELS],
        edgecolor=theme.INK_BLACK,
        yerr=acc_sems, ecolor=theme.INK_BLACK, capsize=4,
    )
    ax_rate.bar(
        xs + width / 2, rates, width=width,
        color=[MODEL_COLORS[m] for m in MODELS],
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
    ax_acc.set_title("cuba / coba / ping" + title_suffix)
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
    fig.savefig(out_path)
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
    fig.savefig(out_path)
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
    fig.savefig(out_path, dpi=120)
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
    ax.set_title("Accuracy / rate frontier across the cuba → coba → ping ladder")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
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
    from oscilloscope import (
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
    from oscilloscope import (
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


def plot_ei_rasters(samples: list[dict], out_path: Path, run_id: str) -> None:
    """One row per ei value; I units stack over E units so the PING-style
    E-then-I cadence reads as alternating bursts when it appears."""
    theme.apply()
    n = len(samples)
    n_e = EI_RASTER_N_E_PLOT
    n_i = EI_RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n, 1, figsize=(10.0, 5.625 + 0.6 * max(n - 4, 0)),
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
    fig.savefig(out_path)
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
    fig.savefig(out_path)
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
    fig.savefig(out_path)
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

    ei_points = run_ei_sweep(notebook_run_id)

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
