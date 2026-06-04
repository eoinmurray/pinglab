"""Notebook runner for entry 024 — 100-epoch convergence audit.

Trains PING and COBA-no-loop networks at medium-tier sample count but
for 100 epochs (vs nb025's 30). The convergence diagnostics in
nb041/nb043/nb044 showed test accuracy plateaus within ~ 10–15 epochs
while E rate keeps drifting upward past epoch 30. This entry asks
whether both metrics stabilise with enough training, and — when the
rate does not — surfaces enough mechanism plots to diagnose *why*.

Six cells: {coba, ping} × seed {42, 43, 44}. Same baseline recipe as
nb025 except epochs = 100. Per-epoch trainable-parameter Frobenius
norms are captured via the `weight_norms` field added to train.py for
this audit.

Notebook entry: src/docs/src/pages/notebooks/nb024.mdx
"""

from __future__ import annotations

import json
import shutil
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
OSCILLOSCOPE = REPO / "src" / "pinglab" / "cli" / "__main__.py"

T_MS = 200.0
DT_TRAIN = 0.1

# Hardcoded recipe value — this is the point of the entry.
EPOCHS: int = 100

SEEDS: tuple[int, ...] = (42, 43, 44)

# Tier governs max_samples only; epochs is held at EPOCHS above.
TIER_CONFIG = {
    "extra small": dict(max_samples=100),
    "small": dict(max_samples=500),
    "medium": dict(max_samples=2000),
    "large": dict(max_samples=5000),
    "extra large": dict(max_samples=10000),
}
DEFAULT_TIER = "medium"

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

MODEL_COLORS = {"coba": theme.DEEP_RED, "ping": theme.INK_BLACK}
MODEL_MARKERS = {"coba": "s", "ping": "D"}
# Trainable parameters in the PING/COBA architecture under this recipe.
# Named by their nn.Module attribute path; both populations expose them
# the same way so the audit can compare on equal footing.
PARAM_LABELS = {"W_ff.0": "W_in", "W_ff.1": "W_out"}


def cell_dir(model: str, seed: int) -> Path:
    return ARTIFACTS / f"{model}__seed{seed}"


def build_train_args(model: str, seed: int, tier: str, out_dir: Path) -> list[str]:
    recipe = MODEL_RECIPES[model]
    args = [
        "train",
        "--model", recipe["__build_as"],
        "--dataset", "mnist",
        "--max-samples", str(TIER_CONFIG[tier]["max_samples"]),
        "--epochs", str(EPOCHS),
        "--t-ms", str(T_MS),
        "--dt", str(DT_TRAIN),
        "--seed", str(seed),
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    for k, v in recipe.items():
        if k.startswith("__"):
            continue
        args += [k, v]
    return args


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


# ─── plotting ───────────────────────────────────────────────────────


def _stamp(fig, run_id: str) -> None:
    fig.text(
        0.995, 0.005, run_id,
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


def _gather_cells() -> dict[tuple[str, int], dict]:
    """Load metrics for every (model, seed) cell. Returns a dict keyed
    by (model, seed) → parsed metrics.json. Missing cells skipped."""
    cells = {}
    for model in MODELS:
        for seed in SEEDS:
            mfile = cell_dir(model, seed) / "metrics.json"
            if mfile.exists():
                cells[(model, seed)] = json.loads(mfile.read_text())
    return cells


def plot_training_curves(out_path: Path, run_id: str) -> None:
    """Six panels — accuracy, loss (train+test), E rate, I rate,
    activity fraction, grad norm — vs epoch. One line per (model, seed)."""
    theme.apply()
    cells = _gather_cells()
    fig, axes = plt.subplots(
        3, 2, figsize=(13.0, 9.0), dpi=150, sharex=True,
        gridspec_kw={"hspace": 0.28, "wspace": 0.30,
                     "left": 0.08, "right": 0.985, "top": 0.93, "bottom": 0.07},
    )
    ax_acc, ax_loss = axes[0]
    ax_e, ax_i = axes[1]
    ax_act, ax_grad = axes[2]

    for (model, seed), m in cells.items():
        color = MODEL_COLORS[model]
        eps = np.array([e["ep"] for e in m["epochs"]])
        accs = [e.get("acc", 0) for e in m["epochs"]]
        train_loss = [e.get("loss", 0) for e in m["epochs"]]
        test_loss = [e.get("test_loss", 0) for e in m["epochs"]]
        e_rates = [e.get("test_rate_e", 0) for e in m["epochs"]]
        i_rates = [e.get("test_rate_i", 0) for e in m["epochs"]]
        acts = [e.get("act", 0) for e in m["epochs"]]
        grads = [e.get("grad_norm", np.nan) for e in m["epochs"]]
        label = model if seed == SEEDS[0] else None
        ax_acc.plot(eps, accs, color=color, lw=1.0, alpha=0.85, label=label)
        ax_loss.plot(eps, train_loss, color=color, lw=1.0, alpha=0.85, ls="-")
        ax_loss.plot(eps, test_loss, color=color, lw=1.0, alpha=0.85, ls="--")
        ax_e.plot(eps, e_rates, color=color, lw=1.0, alpha=0.85)
        ax_i.plot(eps, i_rates, color=color, lw=1.0, alpha=0.85)
        ax_act.plot(eps, acts, color=color, lw=1.0, alpha=0.85)
        ax_grad.plot(eps, grads, color=color, lw=1.0, alpha=0.85)

    ax_acc.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax_acc.set_ylim(0, 100)
    ax_loss.set_ylabel("CE loss (log)", fontsize=theme.SIZE_LABEL)
    ax_loss.set_yscale("log")
    # Differentiate train vs test in the legend rather than the y-label.
    from matplotlib.lines import Line2D
    loss_legend = [
        Line2D([0], [0], color="grey", lw=1.0, ls="-", label="train"),
        Line2D([0], [0], color="grey", lw=1.0, ls="--", label="test"),
    ]
    ax_loss.legend(handles=loss_legend, fontsize=theme.SIZE_LEGEND,
                   frameon=False, loc="upper right")
    ax_e.set_ylabel("Test E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax_i.set_ylabel("Test I rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax_act.set_ylabel("Activity fraction", fontsize=theme.SIZE_LABEL)
    ax_grad.set_ylabel("Grad norm (log)", fontsize=theme.SIZE_LABEL)
    ax_grad.set_yscale("log")
    axes[-1][0].set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    axes[-1][1].set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    ax_acc.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="lower right")
    for ax_row in axes:
        for ax in ax_row:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.15, lw=0.4)
            ax.tick_params(labelsize=theme.SIZE_TICK)
    fig.suptitle(
        "Training curves vs epoch",
        fontsize=theme.SIZE_TITLE,
    )
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_weight_dynamics(out_path: Path, run_id: str) -> None:
    """Per-parameter weight norm and grad ratio vs epoch.

    Two columns × N_params rows. Left column: ||W||_F per epoch
    (linear). Right column: grad/weight ratio per epoch (log). One line
    per cell, coloured by model. Tells us *which* trainable parameter
    is still moving when accuracy has plateaued."""
    theme.apply()
    cells = _gather_cells()
    params = list(PARAM_LABELS.keys())
    fig, axes = plt.subplots(
        len(params), 2, figsize=(10.0, 2.5 * len(params) + 1.5), dpi=150,
        sharex=True, gridspec_kw={"hspace": 0.18, "wspace": 0.22},
    )
    if len(params) == 1:
        axes = np.array([axes])
    for row, pname in enumerate(params):
        ax_norm = axes[row][0]
        ax_ratio = axes[row][1]
        for (model, seed), m in cells.items():
            color = MODEL_COLORS[model]
            eps = np.array([e["ep"] for e in m["epochs"]])
            norms = np.array([
                (e.get("weight_norms") or {}).get(pname, np.nan)
                for e in m["epochs"]
            ])
            ratios = np.array([
                (e.get("grad_ratios") or {}).get(pname, np.nan)
                for e in m["epochs"]
            ])
            label = model if (seed == SEEDS[0] and row == 0) else None
            ax_norm.plot(eps, norms, color=color, lw=1.0, alpha=0.85, label=label)
            ax_ratio.plot(eps, ratios, color=color, lw=1.0, alpha=0.85)
        ax_norm.set_ylabel(f"||{PARAM_LABELS[pname]}||_F", fontsize=theme.SIZE_LABEL)
        ax_ratio.set_ylabel(f"grad/{PARAM_LABELS[pname]} ratio",
                            fontsize=theme.SIZE_LABEL)
        ax_ratio.set_yscale("log")
        if row == 0:
            ax_norm.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="lower right")
        for ax in (ax_norm, ax_ratio):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.15, lw=0.4)
    axes[-1][0].set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    axes[-1][1].set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    fig.suptitle(
        "Per-parameter weight norm and grad/weight ratio vs epoch",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rate_acc_trajectory(out_path: Path, run_id: str) -> None:
    """Parametric (E rate, accuracy) trajectory per cell, dots
    per epoch, line connecting consecutive epochs. Reveals whether
    the network walks along an iso-accuracy manifold (acc flat, rate
    rising) — a 'rate climb is free in the loss landscape' signature."""
    theme.apply()
    cells = _gather_cells()
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=150)
    for (model, seed), m in cells.items():
        accs = [e.get("acc", 0) for e in m["epochs"]]
        rates = [e.get("test_rate_e", 0) for e in m["epochs"]]
        color = MODEL_COLORS[model]
        ax.plot(rates, accs, color=color, lw=0.7, alpha=0.5)
        ax.scatter(rates, accs, c=range(len(rates)), cmap="viridis",
                   s=8, marker=MODEL_MARKERS[model], edgecolors="none",
                   alpha=0.85)
        # Annotate start and end of each trajectory.
        ax.scatter(rates[0], accs[0], color=color, s=40,
                   marker="o", facecolors="none", lw=1.2, zorder=5)
        ax.scatter(rates[-1], accs[-1], color=color, s=60,
                   marker="*", zorder=5)
    ax.set_xlabel("Test E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, lw=0.4)
    # Manual legend.
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=MODEL_COLORS["coba"], marker="s",
               markersize=6, lw=1.0, label="coba"),
        Line2D([0], [0], color=MODEL_COLORS["ping"], marker="D",
               markersize=6, lw=1.0, label="ping"),
        Line2D([0], [0], marker="o", markersize=8, lw=0,
               markerfacecolor="none", markeredgecolor="grey",
               label="epoch 1"),
        Line2D([0], [0], marker="*", markersize=10, lw=0,
               markerfacecolor="grey", markeredgecolor="grey",
               label="final epoch"),
    ]
    ax.legend(handles=legend_elems, fontsize=theme.SIZE_LEGEND,
              frameon=False, loc="lower right")
    fig.suptitle(
        "Parametric (E rate, accuracy) trajectory — coloured by epoch",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── inference: per-cell rate distribution at final state ──────────


def _load_trained_full(train_dir: Path, device):
    """Borrowed from nb042/nb044 — load a trained PING/COBA checkpoint."""
    import torch

    import models as M
    from config import build_net, patch_dt
    from cli import load_dataset, seed_everything

    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", 42)))
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
    state = torch.load(train_dir / "weights.pth", map_location=device)
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True
    return net, cfg, X_te, y_te


def measure_per_cell_rates(train_dir: Path, device) -> np.ndarray:
    """Forward the test set under trained weights. Returns per-cell
    mean E spike rate (Hz) — array of shape (N_E,)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    import models as M
    from cli import EVAL_SEED, encode_batch

    net, cfg, X_te, y_te = _load_trained_full(train_dir, device)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=64,
    )
    per_cell_counts = None
    n_trials = 0
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b = X_b.to(device)
            spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
            _ = net(input_spikes=spk)
            hid = net.spike_record["hid"]
            # (T, B, N_E) or (T, N_E) at B=1 — reduce to per-cell counts.
            if hid.ndim == 3:
                cnt = hid.sum(dim=(0, 1)).cpu().numpy()
                n_trials += hid.shape[1]
            else:
                cnt = hid.sum(dim=0).cpu().numpy()
                n_trials += 1
            per_cell_counts = cnt if per_cell_counts is None else per_cell_counts + cnt
    t_sec = float(cfg["t_ms"]) / 1000.0
    return per_cell_counts / (n_trials * t_sec)


def plot_rate_distributions(rates_by_cell: dict, out_path: Path, run_id: str) -> None:
    """Per-cell rate histogram at final state, PING vs COBA overlaid.
    Three seeds per model shown as faint individual histograms; the
    cross-seed mean histogram is overlaid in solid colour."""
    theme.apply()
    fig, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.5), dpi=150, sharey=True,
        gridspec_kw={"wspace": 0.15},
    )
    # Determine a common bin range so PING and COBA are comparable.
    all_rates = np.concatenate(list(rates_by_cell.values()))
    finite = all_rates[np.isfinite(all_rates)]
    if finite.size == 0:
        plt.close(fig)
        return
    hi = float(np.quantile(finite, 0.995)) * 1.05
    bins = np.linspace(0, max(hi, 1.0), 60)

    for col, model in enumerate(MODELS):
        ax = axes[col]
        for seed in SEEDS:
            key = (model, seed)
            if key not in rates_by_cell:
                continue
            ax.hist(rates_by_cell[key], bins=bins,
                    color=MODEL_COLORS[model], alpha=0.25,
                    histtype="step", lw=1.0)
        ax.set_title(model, fontsize=theme.SIZE_LABEL)
        ax.set_xlabel("Per-cell E rate (Hz)", fontsize=theme.SIZE_LABEL)
        if col == 0:
            ax.set_ylabel("Cell count", fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Annotate mean / median / max as text.
        per_model = np.concatenate([
            rates_by_cell[(model, s)] for s in SEEDS
            if (model, s) in rates_by_cell
        ])
        ax.text(
            0.97, 0.95,
            f"mean = {per_model.mean():.2f} Hz\n"
            f"median = {np.median(per_model):.2f} Hz\n"
            f"max = {per_model.max():.2f} Hz",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED,
        )
    fig.suptitle(
        "Per-cell E rate distribution at final trained state — three seeds overlaid",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── convergence + weight-drift diagnostics ────────────────────────


def slope_last_n(values: list[float], n: int = 10) -> float:
    if len(values) < 2:
        return float("nan")
    tail = values[-min(n, len(values)):]
    if len(tail) < 2:
        return float("nan")
    return float((tail[-1] - tail[0]) / (len(tail) - 1))


def per_cell_diagnostics(rates_by_cell: dict) -> list[dict]:
    """Final + slope + weight-drift summary, one row per cell."""
    summary = []
    for model in MODELS:
        for seed in SEEDS:
            rd = cell_dir(model, seed)
            if not (rd / "metrics.json").exists():
                continue
            m = load_metrics(rd)
            accs = [e.get("acc", 0) for e in m["epochs"]]
            e_rates = [e.get("test_rate_e", 0) for e in m["epochs"]]
            i_rates = [e.get("test_rate_i", 0) for e in m["epochs"]]
            losses = [e.get("loss", 0) for e in m["epochs"]]
            grad_norms = [e.get("grad_norm", float("nan")) for e in m["epochs"]]
            # Weight-norm drift: ep1 → final.
            wn_first = m["epochs"][0].get("weight_norms") or {}
            wn_last = m["epochs"][-1].get("weight_norms") or {}
            drift = {
                pname: {
                    "init": wn_first.get(pname),
                    "final": wn_last.get(pname),
                    "ratio_final_over_init": (
                        (wn_last.get(pname) / wn_first.get(pname))
                        if wn_first.get(pname) else None
                    ),
                    "slope_last10": slope_last_n(
                        [(e.get("weight_norms") or {}).get(pname, 0) for e in m["epochs"]],
                        10,
                    ),
                }
                for pname in PARAM_LABELS
            }
            rates_at_cell = rates_by_cell.get((model, seed))
            cell_summary = {
                "model": model,
                "seed": seed,
                "epochs_completed": len(m["epochs"]),
                "final_acc": accs[-1] if accs else float("nan"),
                "final_e_rate_hz": e_rates[-1] if e_rates else float("nan"),
                "final_i_rate_hz": i_rates[-1] if i_rates else float("nan"),
                "final_loss": losses[-1] if losses else float("nan"),
                "final_grad_norm": grad_norms[-1] if grad_norms else float("nan"),
                "acc_slope_last10_pp_per_ep": slope_last_n(accs, 10),
                "e_rate_slope_last10_hz_per_ep": slope_last_n(e_rates, 10),
                "i_rate_slope_last10_hz_per_ep": slope_last_n(i_rates, 10),
                "loss_slope_last10_per_ep": slope_last_n(losses, 10),
                "weight_drift": drift,
            }
            if rates_at_cell is not None and rates_at_cell.size > 0:
                cell_summary["per_cell_rate_stats"] = {
                    "mean_hz": float(rates_at_cell.mean()),
                    "median_hz": float(np.median(rates_at_cell)),
                    "std_hz": float(rates_at_cell.std()),
                    "max_hz": float(rates_at_cell.max()),
                    "frac_silent": float((rates_at_cell == 0).mean()),
                }
            summary.append(cell_summary)
    return summary


def evaluate_success(summary: list[dict], figures: Path) -> list[dict]:
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

    crits = [
        artifact("training_curves.png", "training-curves figure rendered"),
        artifact("weight_dynamics.png", "weight-dynamics figure rendered"),
        artifact("rate_acc_trajectory.png", "rate/accuracy trajectory rendered"),
        artifact("rate_distributions.png", "rate-distribution figure rendered"),
    ]

    # Thresholds for declaring convergence at the per-cell level.
    ACC_SLOPE_OK = 0.1   # pp / epoch (≤ 1 pp drift over 10 epochs)
    RATE_SLOPE_OK = 0.05  # Hz / epoch (≤ 0.5 Hz drift over 10 epochs)
    for cell in summary:
        cell_label = f"{cell['model']} seed {cell['seed']}"
        crits.append({
            "label": f"{cell_label} accuracy converged (|slope| ≤ {ACC_SLOPE_OK} pp/ep)",
            "passed": bool(abs(cell["acc_slope_last10_pp_per_ep"]) <= ACC_SLOPE_OK),
            "detail": (
                f"acc = {cell['final_acc']:.2f}%, "
                f"slope = {cell['acc_slope_last10_pp_per_ep']:+.3f} pp/ep"
            ),
        })
        crits.append({
            "label": f"{cell_label} E rate converged (|slope| ≤ {RATE_SLOPE_OK} Hz/ep)",
            "passed": bool(abs(cell["e_rate_slope_last10_hz_per_ep"]) <= RATE_SLOPE_OK),
            "detail": (
                f"E = {cell['final_e_rate_hz']:.2f} Hz, "
                f"slope = {cell['e_rate_slope_last10_hz_per_ep']:+.3f} Hz/ep"
            ),
        })
    return crits


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    skip_training = "--skip-training" in sys.argv
    only_missing = "--only-missing" in sys.argv
    wipe_dir = "--no-wipe-dir" not in sys.argv

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    n_cells = len(MODELS) * len(SEEDS)
    print(
        f"notebook_run_id = {notebook_run_id} tier={tier} epochs={EPOCHS} "
        f"cells={n_cells}"
        + ("  [skip-training]" if skip_training else "")
        + (f"  [modal:{modal_gpu}]" if modal_gpu else "")
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

    if not skip_training:
        dispatcher = BatchDispatcher(modal_gpu, REPO, OSCILLOSCOPE)
        for model in MODELS:
            for seed in SEEDS:
                out = cell_dir(model, seed)
                if only_missing and (out / "metrics.json").exists():
                    print(
                        f"[skip] {model}/seed={seed} → already trained "
                        f"at {out.relative_to(REPO)}"
                    )
                    continue
                print(f"[train] {model}/seed={seed} → {out.relative_to(REPO)}")
                dispatcher.submit(
                    build_train_args(model, seed, tier, out),
                    out,
                )
        dispatcher.drain()

    # Inference: per-cell rate distribution at the final trained state.
    from cli import _auto_device

    device = _auto_device()
    print(f"device = {device}")
    rates_by_cell: dict[tuple[str, int], np.ndarray] = {}
    for model in MODELS:
        for seed in SEEDS:
            rd = cell_dir(model, seed)
            if not (rd / "weights.pth").exists():
                raise SystemExit(f"missing weights: {rd / 'weights.pth'}")
            t0 = time.monotonic()
            rates = measure_per_cell_rates(rd, device)
            rates_by_cell[(model, seed)] = rates
            print(
                f"  rates  {model:<5} seed={seed}  "
                f"mean={rates.mean():6.2f} Hz  median={np.median(rates):6.2f} Hz  "
                f"max={rates.max():6.2f} Hz  silent={int((rates == 0).sum())}"
                f" / {rates.size}  ({time.monotonic() - t0:.1f}s)"
            )

    plot_training_curves(FIGURES / "training_curves.png", notebook_run_id)
    print(f"wrote {FIGURES / 'training_curves.png'}")
    plot_weight_dynamics(FIGURES / "weight_dynamics.png", notebook_run_id)
    print(f"wrote {FIGURES / 'weight_dynamics.png'}")
    plot_rate_acc_trajectory(FIGURES / "rate_acc_trajectory.png", notebook_run_id)
    print(f"wrote {FIGURES / 'rate_acc_trajectory.png'}")
    plot_rate_distributions(rates_by_cell, FIGURES / "rate_distributions.png",
                            notebook_run_id)
    print(f"wrote {FIGURES / 'rate_distributions.png'}")

    summary = per_cell_diagnostics(rates_by_cell)
    print("  per-cell convergence + drift:")
    for cell in summary:
        wn = cell["weight_drift"]
        print(
            f"    {cell['model']:<5} seed={cell['seed']}  "
            f"acc={cell['final_acc']:5.2f}% (Δ{cell['acc_slope_last10_pp_per_ep']:+.3f}/ep)  "
            f"E={cell['final_e_rate_hz']:6.2f} Hz (Δ{cell['e_rate_slope_last10_hz_per_ep']:+.3f}/ep)  "
            f"||W_in||={wn['W_ff.0']['final']:6.2f} (×{wn['W_ff.0']['ratio_final_over_init']:5.3f})  "
            f"||W_out||={wn['W_ff.1']['final']:7.2f} (×{wn['W_ff.1']['ratio_final_over_init']:5.3f})"
        )

    duration_s = time.monotonic() - t_start
    train_cfg = load_config(cell_dir(MODELS[0], SEEDS[0]))
    crits = evaluate_success(summary, FIGURES)
    summary_doc = {
        "notebook_run_id": notebook_run_id,
        "git_sha": train_cfg.get("git_sha"),
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "tier": tier,
        "config": {
            "tier": tier,
            "dataset": "mnist",
            "models": MODELS,
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "max_samples": TIER_CONFIG[tier]["max_samples"],
            "t_ms": T_MS,
            "dt": DT_TRAIN,
        },
        "per_cell": summary,
        "success_criteria": crits,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary_doc, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary_doc['duration']}")

    for c in crits:
        mark = "pass" if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['label']} — {c['detail']}")
    if any(not c["passed"] for c in crits):
        sys.exit(1)


if __name__ == "__main__":
    main()
