"""Notebook runner for entry 024 — 100-epoch convergence audit.

Trains PING and COBA-no-loop networks at medium-tier sample count but
for 100 epochs (vs nb025's 30). The convergence diagnostics in
nb041/nb044 showed test accuracy plateaus within ~ 10–15 epochs
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
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from helpers.fmt import format_duration  # noqa: E402
from helpers.modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402
from cli import theme  # noqa: E402

SLUG = "nb024"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
OSCILLOSCOPE = REPO / "src" / "cli" / "cli.py"

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
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_acc_rate_vs_epoch(out_path: Path, run_id: str) -> None:
    """Two panels — test accuracy (top) and test E rate (bottom) vs epoch, one
    line per (model, seed). The headline: accuracy plateaus by ~15 epochs for both
    architectures, while PING's E rate locks to a tight band and COBA's keeps
    climbing through epoch 100."""
    theme.apply()
    cells = _gather_cells()
    fig, (ax_acc, ax_e) = plt.subplots(
        2, 1, figsize=(8.0, 6.0), dpi=150, sharex=True,
        gridspec_kw={"hspace": 0.12, "left": 0.11, "right": 0.97, "top": 0.93, "bottom": 0.09},
    )
    for (model, seed), m in cells.items():
        color = MODEL_COLORS[model]
        eps = np.array([e["ep"] for e in m["epochs"]])
        accs = [e.get("acc", 0) for e in m["epochs"]]
        e_rates = [e.get("test_rate_e", 0) for e in m["epochs"]]
        label = model.upper() if seed == SEEDS[0] else None
        ax_acc.plot(eps, accs, color=color, lw=1.2, alpha=0.85, label=label)
        ax_e.plot(eps, e_rates, color=color, lw=1.2, alpha=0.85, label=label)
    ax_acc.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax_acc.set_ylim(0, 100)
    ax_acc.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="lower right")
    ax_e.set_ylabel("Test E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax_e.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    for ax in (ax_acc, ax_e):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, lw=0.4)
        ax.tick_params(labelsize=theme.SIZE_TICK)
    fig.suptitle("Accuracy converges for both; only PING's rate settles",
                 fontsize=theme.SIZE_TITLE)
    stamp_figure(fig, run_id)
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
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rates_vs_epoch(out_path: Path, run_id: str) -> None:
    """Total network firing rate vs epoch — COBA E vs PING (E + I).

    COBA's I population is silent (ei_strength = 0 → no recurrent input
    to I), so its E rate IS its total network rate. PING's network total
    is the sum of E and I rates. Honest like-for-like comparison of
    the total spike budget the optimiser is spending per cell-second.
    """
    theme.apply()
    cells = _gather_cells()
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=150)
    seen_labels: set[str] = set()
    for (model, seed), m in cells.items():
        eps = np.array([e["ep"] for e in m["epochs"]])
        e_rates = np.array([e.get("test_rate_e", 0) for e in m["epochs"]])
        i_rates = np.array([e.get("test_rate_i", 0) for e in m["epochs"]])
        color = MODEL_COLORS[model]
        if model == "coba":
            label = ("COBA total (E only — I silent)"
                     if "COBA total" not in seen_labels else None)
            seen_labels.add("COBA total")
            ax.plot(eps, e_rates, color=color, lw=1.4, alpha=0.85, label=label)
        else:  # ping
            total = e_rates + i_rates
            label = ("PING total (E + I)"
                     if "PING total" not in seen_labels else None)
            seen_labels.add("PING total")
            ax.plot(eps, total, color=color, lw=1.4, alpha=0.85, label=label)
            # Faint dashed lines for the per-population components.
            if "PING E only" not in seen_labels:
                ax.plot(eps, e_rates, color=color, lw=0.8, ls=":",
                        alpha=0.5, label="PING E only")
                seen_labels.add("PING E only")
            else:
                ax.plot(eps, e_rates, color=color, lw=0.8, ls=":", alpha=0.5)
            if "PING I only" not in seen_labels:
                ax.plot(eps, i_rates, color=color, lw=0.8, ls="--",
                        alpha=0.5, label="PING I only")
                seen_labels.add("PING I only")
            else:
                ax.plot(eps, i_rates, color=color, lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Hidden firing rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.15, lw=0.4)
    ax.legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="center right")
    fig.suptitle(
        "Total network firing rate vs epoch — COBA E vs PING (E + I)",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_init_and_trained_state(train_dir: Path):
    """Reconstruct the init state for one cell (re-seed + re-build with
    the same cfg → state_dict before training) plus the trained state
    (state_dict from disk). Returns (init_state, trained_state)."""
    import torch
    import models as M
    from cli.config import build_net, patch_dt
    from cli import seed_everything

    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", 42)))
    M.T_ms = float(cfg["t_ms"])
    patch_dt(float(cfg["dt"]))
    hidden_sizes = cfg.get("hidden_sizes") or [int(cfg["n_hidden"])]
    M.N_HID = hidden_sizes[-1]
    M.N_INH = hidden_sizes[-1] // 4
    M.HIDDEN_SIZES = list(hidden_sizes)
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
        device="cpu",
        randomize_init=not bool(cfg.get("kaiming_init", False)),
        dales_law=bool(cfg.get("dales_law", True)),
        hidden_sizes=hidden_sizes,
        readout_mode=cfg.get("readout_mode", "mem-mean"),
    )
    init_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    trained_state = torch.load(train_dir / "weights.pth", map_location="cpu")
    return init_state, trained_state


def plot_weight_before_after(
    out_path: Path, run_id: str, seed: int = 42,
) -> None:
    """Before/after histograms of the two trainable parameter matrices
    (W_in and W_out) for one representative seed × (COBA, PING). Shows
    where weights start and where they end up, directly."""
    theme.apply()
    cells: list[tuple[str, dict, dict]] = []
    for model in MODELS:
        train_dir = cell_dir(model, seed)
        if not (train_dir / "weights.pth").exists():
            print(f"[warn] weight_before_after: missing {train_dir.name}, skipping")
            continue
        init, trained = _load_init_and_trained_state(train_dir)
        cells.append((model, init, trained))
    if not cells:
        print("[warn] no cells available for weight_before_after — skipping")
        return

    params = list(PARAM_LABELS.keys())  # ['W_ff.0', 'W_ff.1'] → W_in, W_out
    n_rows = len(params)
    n_cols = len(cells)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 3.0 * n_rows), dpi=150,
        sharex="row", gridspec_kw={"hspace": 0.32, "wspace": 0.22},
    )
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes[:, None]
    for col, (model, init, trained) in enumerate(cells):
        color = MODEL_COLORS[model]
        for row, pname in enumerate(params):
            ax = axes[row][col]
            init_w = init.get(pname)
            trained_w = trained.get(pname)
            if init_w is None or trained_w is None:
                ax.text(0.5, 0.5, f"{pname} missing",
                        ha="center", va="center", transform=ax.transAxes)
                continue
            init_arr = init_w.numpy().ravel()
            trained_arr = trained_w.numpy().ravel()
            # Symmetric bin range over the union of both.
            lo = float(min(init_arr.min(), trained_arr.min()))
            hi = float(max(init_arr.max(), trained_arr.max()))
            bins = np.linspace(lo, hi, 80)
            ax.hist(init_arr, bins=bins, color=theme.GREY_MID, alpha=0.55,
                    edgecolor="none", label=f"init  (||·||={np.linalg.norm(init_arr):.1f})")
            ax.hist(trained_arr, bins=bins, color=color, alpha=0.60,
                    edgecolor="none",
                    label=f"trained  (||·||={np.linalg.norm(trained_arr):.1f})")
            ax.set_yscale("log")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.15, lw=0.4)
            if row == 0:
                ax.set_title(f"{model.upper()} (seed {seed})",
                             fontsize=theme.SIZE_TITLE)
            if col == 0:
                ax.set_ylabel(f"count ({PARAM_LABELS[pname]} entries)",
                              fontsize=theme.SIZE_LABEL)
            if row == n_rows - 1:
                ax.set_xlabel("weight value", fontsize=theme.SIZE_LABEL)
            ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    fig.suptitle(
        f"Weight distributions before vs after training "
        f"(W_in 784×1024 sparse, W_out 1024×10)",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
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
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── inference: total rate vs (W^EI, W^IE) on trained PING ─────────


WEI_WIE_GRID_VALUES: tuple[float, ...] = (
    0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0,
)


def run_wei_wie_total_rate_sweep(
    train_dir: Path, device, grid: tuple[float, ...] = WEI_WIE_GRID_VALUES,
) -> list[dict]:
    """Inference-time 2D sweep on the trained PING baseline. For each
    (s_ei, s_ie) cell, multiply the loop matrices by the scaling factors
    and measure mean E and I firing rate on the test set. Records total
    network rate (E + I) as the headline quantity."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    import models as M
    from cli import EVAL_SEED, encode_batch

    net, cfg, X_te, y_te = _load_trained_full(train_dir, device)
    w_ei_original = {k: v.data.clone() for k, v in net.W_ei.items()}
    w_ie_original = {k: v.data.clone() for k, v in net.W_ie.items()}

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=64,
    )
    rows: list[dict] = []
    try:
        for s_ei in grid:
            for s_ie in grid:
                for k in net.W_ei.keys():
                    net.W_ei[k].data.copy_(w_ei_original[k] * float(s_ei))
                for k in net.W_ie.keys():
                    net.W_ie[k].data.copy_(w_ie_original[k] * float(s_ie))
                eval_gen = torch.Generator().manual_seed(EVAL_SEED)
                e_spike_sum = i_spike_sum = 0.0
                correct = total = 0
                with torch.no_grad():
                    for X_b, y_b in test_loader:
                        X_b, y_b = X_b.to(device), y_b.to(device)
                        spk = encode_batch(
                            X_b, M.dt, False, generator=eval_gen,
                        )
                        logits = net(input_spikes=spk)
                        correct += (logits.argmax(1) == y_b).sum().item()
                        total += y_b.size(0)
                        e_spike_sum += float(
                            net.spike_record["hid"].sum().item()
                        )
                        i_spike_sum += float(
                            net.spike_record["inh"].sum().item()
                        )
                t_sec = float(cfg["t_ms"]) / 1000.0
                n_e = M.N_HID
                n_i = M.N_INH or 1
                e_rate = e_spike_sum / (total * n_e * t_sec)
                i_rate = i_spike_sum / (total * n_i * t_sec)
                acc = 100.0 * correct / total
                rows.append({
                    "s_ei": float(s_ei),
                    "s_ie": float(s_ie),
                    "acc": acc,
                    "rate_e": e_rate,
                    "rate_i": i_rate,
                    "rate_total": e_rate + i_rate,
                })
                print(
                    f"  s_ei={s_ei:>4.2g}  s_ie={s_ie:>4.2g}  "
                    f"acc={acc:5.2f}%  E={e_rate:6.2f}Hz  I={i_rate:6.2f}Hz  "
                    f"total={e_rate + i_rate:6.2f}Hz"
                )
    finally:
        for k in net.W_ei.keys():
            net.W_ei[k].data.copy_(w_ei_original[k])
        for k in net.W_ie.keys():
            net.W_ie[k].data.copy_(w_ie_original[k])
    return rows


def plot_wei_wie_total_rate(
    rows: list[dict], out_path: Path, run_id: str,
) -> None:
    """Heatmap of total network firing rate (E + I) vs (s_ei, s_ie).
    Annotates each cell with the total rate. Star at (1, 1) marks the
    trained operating point."""
    theme.apply()
    s_ei_vals = sorted({r["s_ei"] for r in rows})
    s_ie_vals = sorted({r["s_ie"] for r in rows})
    rate_grid = np.full((len(s_ei_vals), len(s_ie_vals)), np.nan)
    e_grid = np.full_like(rate_grid, np.nan)
    i_grid = np.full_like(rate_grid, np.nan)
    for r in rows:
        ei = s_ei_vals.index(r["s_ei"])
        ie = s_ie_vals.index(r["s_ie"])
        rate_grid[ei, ie] = r["rate_total"]
        e_grid[ei, ie] = r["rate_e"]
        i_grid[ei, ie] = r["rate_i"]

    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=150)
    im = ax.imshow(
        rate_grid, origin="lower", aspect="equal",
        cmap="magma", vmin=0, vmax=150,
    )
    ax.set_xticks(range(len(s_ie_vals)))
    ax.set_yticks(range(len(s_ei_vals)))
    ax.set_xticklabels([f"{s:g}" for s in s_ie_vals])
    ax.set_yticklabels([f"{s:g}" for s in s_ei_vals])
    ax.set_xlabel("$W^{IE}$ scale  (× trained value)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("$W^{EI}$ scale  (× trained value)", fontsize=theme.SIZE_LABEL)
    for ei in range(len(s_ei_vals)):
        for ie in range(len(s_ie_vals)):
            v = rate_grid[ei, ie]
            if np.isnan(v):
                continue
            # Choose text colour based on cell value.
            txt_color = "white" if v < 0.55 * np.nanmax(rate_grid) else "black"
            ax.text(
                ie, ei,
                f"{v:.1f}\n({e_grid[ei,ie]:.1f}|{i_grid[ei,ie]:.1f})",
                ha="center", va="center", fontsize=theme.SIZE_ANNOTATION - 1,
                color=txt_color,
            )
    # Mark the trained operating point at (1, 1).
    if 1.0 in s_ei_vals and 1.0 in s_ie_vals:
        ei_idx = s_ei_vals.index(1.0)
        ie_idx = s_ie_vals.index(1.0)
        ax.scatter(ie_idx, ei_idx, marker="*", s=300,
                   facecolors="none", edgecolors="white", lw=1.8, zorder=5)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Total network rate E + I (Hz)",
                   fontsize=theme.SIZE_LABEL)
    fig.suptitle(
        "Inference-time (W^EI × W^IE) sweep on trained PING — "
        "total firing rate (E + I)\n"
        "cell labels: total Hz, with (E | I) breakdown.  "
        "Star marks the trained baseline.",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_wei_wie_diagonal_total_rate(
    rows: list[dict], out_path: Path, run_id: str,
) -> None:
    """Diagonal slice through the (W^EI × W^IE) sweep — total rate vs s
    where W^EI = W^IE = s. Total, E, and I rates on a single panel,
    plus accuracy on a twin axis. The bottom-left → top-right diagonal
    of the heatmap, redrawn as a line chart so the asymptote is visible.
    """
    theme.apply()
    diag = sorted(
        [r for r in rows if abs(r["s_ei"] - r["s_ie"]) < 1e-9],
        key=lambda r: r["s_ei"],
    )
    scales = [r["s_ei"] for r in diag]
    totals = [r["rate_total"] for r in diag]
    es = [r["rate_e"] for r in diag]
    is_ = [r["rate_i"] for r in diag]
    accs = [r["acc"] for r in diag]

    fig, ax_rate = plt.subplots(figsize=(9.0, 5.0), dpi=150)
    ax_rate.plot(scales, totals, marker="D", markersize=7, lw=1.6,
                 color=theme.INK_BLACK, label="total (E + I)")
    ax_rate.plot(scales, es, marker="o", markersize=5, lw=1.0, ls=":",
                 color=theme.INK_BLACK, alpha=0.7, label="E only")
    ax_rate.plot(scales, is_, marker="^", markersize=5, lw=1.0, ls="--",
                 color=theme.INK_BLACK, alpha=0.7, label="I only")
    ax_rate.set_xlabel(
        "Diagonal coupling scale  s  (with $W^{EI} = W^{IE} = s$ × trained)",
        fontsize=theme.SIZE_LABEL,
    )
    ax_rate.set_ylabel("Hidden firing rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax_rate.spines["top"].set_visible(False)
    ax_rate.grid(True, alpha=0.15, lw=0.4)
    ax_rate.legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="upper right")
    # Vertical marker at the trained baseline (s = 1).
    ax_rate.axvline(1.0, color=theme.GREY_MID, lw=0.7, ls=":", alpha=0.7)
    ax_rate.text(
        1.0, ax_rate.get_ylim()[1] * 0.95, " trained baseline (s = 1)",
        fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED,
        ha="left", va="top",
    )

    ax_acc = ax_rate.twinx()
    ax_acc.plot(scales, accs, marker="s", markersize=5, lw=1.0,
                color=theme.DEEP_RED, alpha=0.75)
    ax_acc.set_ylabel("Test accuracy (%)",
                      fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED)
    ax_acc.tick_params(axis="y", labelcolor=theme.DEEP_RED)
    ax_acc.set_ylim(0, 100)
    ax_acc.axhline(10.0, color=theme.DEEP_RED, lw=0.5, ls=":", alpha=0.4)
    ax_acc.spines["top"].set_visible(False)

    fig.suptitle(
        "Diagonal slice of the (W^EI × W^IE) sweep — "
        "total rate and accuracy vs coupling scale",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── inference: per-cell rate distribution at final state ──────────


def _load_trained_full(train_dir: Path, device):
    """Borrowed from nb042/nb044 — load a trained PING/COBA checkpoint."""
    import torch

    import models as M
    from cli.config import build_net, patch_dt
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
    stamp_figure(fig, run_id)
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

    prepare_run_dirs(
        SLUG, notebook_run_id, wipe=wipe_dir, skip_training=skip_training,
        make_artifacts=False,
    )

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

    plot_acc_rate_vs_epoch(FIGURES / "acc_rate_vs_epoch.png", notebook_run_id)
    print(f"wrote {FIGURES / 'acc_rate_vs_epoch.png'}")
    plot_training_curves(FIGURES / "training_curves.png", notebook_run_id)
    print(f"wrote {FIGURES / 'training_curves.png'}")
    plot_weight_dynamics(FIGURES / "weight_dynamics.png", notebook_run_id)
    print(f"wrote {FIGURES / 'weight_dynamics.png'}")
    plot_weight_before_after(
        FIGURES / "weights_before_after.png", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'weights_before_after.png'}")
    plot_rates_vs_epoch(FIGURES / "rates_vs_epoch.png", notebook_run_id)
    print(f"wrote {FIGURES / 'rates_vs_epoch.png'}")

    # Inference-time (W^EI × W^IE) sweep on the trained PING baseline.
    print("[wei-wie sweep] inference on trained PING (seed 42)")
    from cli import _auto_device
    device = _auto_device()
    wei_wie_rows = run_wei_wie_total_rate_sweep(cell_dir("ping", 42), device)
    plot_wei_wie_total_rate(
        wei_wie_rows, FIGURES / "wei_wie_total_rate.png", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'wei_wie_total_rate.png'}")
    plot_wei_wie_diagonal_total_rate(
        wei_wie_rows, FIGURES / "wei_wie_diagonal_total_rate.png",
        notebook_run_id,
    )
    print(f"wrote {FIGURES / 'wei_wie_diagonal_total_rate.png'}")
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
    summary_doc = {
        "notebook_run_id": notebook_run_id,
        "git_sha": train_cfg.get("git_sha"),
        "duration_s": round(duration_s, 1),
        "duration": format_duration(duration_s),
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
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary_doc, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary_doc['duration']}")



if __name__ == "__main__":
    main()
