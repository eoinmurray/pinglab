"""Goal 1 figure generators — one function per paper figure.

Reads trained calibrations from src/artifacts/mnist-dt-stability/calibrations/
and writes PNGs to src/artifacts/mnist-dt-stability/figures/.

Functions:
    calibration_accuracy(size) → Fig 1: bar chart of best_acc per model × dt
    dt_sweep(size)             → Fig 2: headline 5-curve dt-sweep plot
    training_curves(size)      → Fig 3: per-epoch test acc trajectories
    ablation_attribution(size) → Fig 4: feature-attribution bars at worst case
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pinglab.experiments.mnist_dt_stability.config import (
    CALIBS_ROOT,
    FIGURES_ROOT,
    HEADLINE_LADDER,
    MODELS,
    Size,
    calib_dir,
    suffix,
)


# ── Visual identity ─────────────────────────────────────────────────────

LABELS = {
    "snntorch":              "snnTorch",
    "snntorch-library":      "snnTorch-library",
    "cuba":                  "CUBA",
    "cuba-exp":              "CUBA+exp",
    "coba":                  "COBA",
    "ping":                  "PING",
}
COLORS = {
    "snntorch":              "#1f77b4",
    "snntorch-library":      "#6ca9d1",
    "cuba":                  "#ff7f0e",
    "cuba-exp":              "#ffd35c",
    "coba":                  "#2ca02c",
    "ping":                  "#d62728",
}


# ── Data loaders ────────────────────────────────────────────────────────

def _load_acc(model: str, dt: float, size: Size) -> float | None:
    path = calib_dir(model, dt, size) / "metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("best_acc")


def _load_sweep(model: str, dt: float, size: Size) -> list[dict] | None:
    path = calib_dir(model, dt, size) / "infer-frozen" / "results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("sweep")


def _load_curve(model: str, dt: float, size: Size) -> list[tuple[int, float]] | None:
    path = calib_dir(model, dt, size) / "metrics.jsonl"
    if not path.exists():
        return None
    pts = []
    for line in path.read_text().strip().split("\n"):
        if not line:
            continue
        rec = json.loads(line)
        ep, acc = rec.get("ep"), rec.get("acc")
        if ep is not None and acc is not None:
            pts.append((ep, acc))
    return pts


def _acc_at(sweep: list[dict], dt_target: float) -> float | None:
    for r in sweep:
        if abs(r["dt"] - dt_target) < 0.01:
            return r["acc"]
    return None


def _out_path(name: str, size: Size) -> Path:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    return FIGURES_ROOT / f"{name}{suffix(size)}.png"


# ── Fig 1: calibration accuracy ─────────────────────────────────────────

def calibration_accuracy(size: Size) -> Path:
    models = list(MODELS.keys())
    accs_01 = [_load_acc(m, 0.1, size) for m in models]
    accs_10 = [_load_acc(m, 1.0, size) for m in models]
    labels = [LABELS[m] for m in models]
    colors = [COLORS[m] for m in models]

    x = np.arange(len(models))
    w = 0.4
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    b1 = ax.bar(x - w / 2, [a or 0 for a in accs_01], w,
                label=r"trained $\Delta t = 0.1$ ms",
                color=colors, edgecolor="white", linewidth=1)
    b2 = ax.bar(x + w / 2, [a or 0 for a in accs_10], w,
                label=r"trained $\Delta t = 1.0$ ms",
                color=colors, edgecolor="black", linewidth=1.2,
                alpha=0.7, hatch="//")

    for bars, accs in [(b1, accs_01), (b2, accs_10)]:
        for bar, acc in zip(bars, accs):
            if acc is None:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8, f"{acc:.0f}",
                    ha="center", va="bottom", fontsize=8)

    ax.axhline(10, color="gray", ls="--", lw=0.8, alpha=0.6, zorder=0)
    ax.text(len(models) - 0.5, 11.5, "chance (10%)",
            fontsize=8, color="gray", ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Best test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.set_title(r"Calibration: all models train to $\geq$85% on MNIST at either $\Delta t$",
                 fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()

    out = _out_path("calibration_accuracy", size)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ── Fig 2: headline dt-sweep ────────────────────────────────────────────

def dt_sweep(size: Size) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.625), sharey=True)
    missing = []
    for col, dt_train in enumerate([0.1, 1.0]):
        ax = axes[col]
        for model in HEADLINE_LADDER:
            sweep = _load_sweep(model, dt_train, size)
            if sweep is None:
                missing.append(f"dt={dt_train} {model}")
                continue
            dts = [r["dt"] for r in sweep]
            accs = [r["acc"] for r in sweep]
            ax.plot(dts, accs, "o-", color=COLORS[model],
                    label=LABELS[model].replace("+", " + "),
                    markersize=4, linewidth=1.5)
        ax.axvline(dt_train, color="gray", ls="--", alpha=0.5,
                   label=f"train $\\Delta t={dt_train}$")
        ax.set_xlabel(r"Inference $\Delta t$ (ms)")
        ax.set_title(f"Trained at $\\Delta t={dt_train}$ ms")
        ax.set_xlim(0, 2.1)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Accuracy (%)")
    tag = f"  [{size.samples} × {size.epochs}]"
    fig.suptitle(f"dt Inference Stability (frozen inputs, OR-pool){tag}",
                 fontsize=14)
    fig.tight_layout()

    out = _out_path("dt_sweep_combined", size)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓ {out}")
    if missing:
        print(f"  missing: {', '.join(missing)}")
    return out


# ── Fig 3: training curves ──────────────────────────────────────────────

def training_curves(size: Size) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.5), sharey=True)
    for ax, dt_train, title in zip(
        axes, [0.1, 1.0],
        [r"trained $\Delta t = 0.1$ ms", r"trained $\Delta t = 1.0$ ms"],
    ):
        for model in MODELS:
            curve = _load_curve(model, dt_train, size)
            if curve is None:
                continue
            xs, ys = zip(*curve)
            ax.plot(xs, ys, "-", color=COLORS[model], label=LABELS[model],
                    linewidth=1.4, alpha=0.9)

        ax.axhline(10, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.text(1, 12, "chance (10%)", fontsize=7.5, color="gray")
        ax.set_xlabel("Epoch")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, size.epochs)
        ax.set_ylim(0, 100)
        n_ticks = min(5, size.epochs + 1)
        ax.set_xticks(np.linspace(0, size.epochs, n_ticks, dtype=int))
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(True, alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    axes[0].set_ylabel("Test accuracy (%)")
    axes[1].legend(loc="lower right", fontsize=7, framealpha=0.9, ncol=1)
    fig.suptitle(r"Training curves: every model climbs from chance to $\geq$85% within "
                 f"{size.epochs} epochs", fontsize=10.5, y=1.02)
    fig.tight_layout()

    out = _out_path("training_curves", size)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


# ── Fig 4: feature attribution ──────────────────────────────────────────

FEATURES_ADDED = {
    "cuba":      "(baseline)",
    "cuba-exp":  "+ exp synapse",
    "coba":      "+ conductance-V",
    "ping":      "+ E–I gamma",
}
# 4-step waterfall: cuba → cuba-exp → coba → ping. Each rung adds one
# feature. snntorch (the library-convention baseline) is deliberately not
# on this ladder — it differs from cuba by a discretisation convention, not
# by a biophysical feature.
ABLATION_LADDER = ["cuba", "cuba-exp", "coba", "ping"]
ABLATION_COLORS = ["#ff7f0e", "#ffd35c", "#2ca02c", "#d62728"]


def ablation_attribution(size: Size, train_dt: float = 0.1,
                          infer_dt: float = 2.0) -> Path:
    """Feature-attribution bars at the worst-case cell
    (train fine, infer coarse) where cuba collapses hardest.
    """
    accs = []
    for m in ABLATION_LADDER:
        sweep = _load_sweep(m, train_dt, size)
        accs.append(_acc_at(sweep, infer_dt) if sweep else None)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    x = np.arange(len(ABLATION_LADDER))
    bars = ax.bar(x, [a or 0 for a in accs], color=ABLATION_COLORS,
                  edgecolor="black", linewidth=0.8, width=0.65)

    for bar, acc in zip(bars, accs):
        if acc is None:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2, f"{acc:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    for i in range(1, len(accs)):
        if accs[i] is None or accs[i - 1] is None:
            continue
        delta = accs[i] - accs[i - 1]
        sign = "+" if delta >= 0 else ""
        midx = (i - 1 + i) / 2
        midy = max(accs[i], accs[i - 1]) + 8
        color = "#2ca02c" if delta >= 5 else ("#aaaaaa" if abs(delta) < 2 else "#d62728")
        ax.annotate(
            f"{sign}{delta:.0f}pp",
            xy=(midx, midy),
            ha="center", va="center",
            fontsize=11, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                     edgecolor=color, linewidth=1.2),
        )
        ax.annotate(
            "", xy=(i - 0.15, max(accs[i], accs[i - 1]) + 4),
            xytext=(i - 1 + 0.15, max(accs[i], accs[i - 1]) + 4),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.3),
        )

    labels = [f"{FEATURES_ADDED[m]}\n{m}" for m in ABLATION_LADDER]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(
        f"Accuracy at $\\Delta t_{{\\mathrm{{infer}}}} = {infer_dt}$ ms (%)"
    )
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(
        f"Feature attribution: trained at $\\Delta t = {train_dt}$ ms, "
        f"tested at $\\Delta t = {infer_dt}$ ms\n"
        "Exp synapse alone recovers most of the dt-stability gap",
        fontsize=10,
    )
    ax.axhline(10, color="gray", ls="--", lw=0.8, alpha=0.6, zorder=0)
    ax.text(len(ABLATION_LADDER) - 0.5, 11, "chance",
            fontsize=8, color="gray", ha="right", va="bottom")
    ax.grid(True, axis="y", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()

    out = _out_path("ablation_attribution", size)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  ✓ {out}")
    for m, acc in zip(ABLATION_LADDER, accs):
        print(f"    {FEATURES_ADDED[m]:20s}  {m:24s} {acc or 0:5.1f}%")
    return out


# ── Appendix: snnTorch library parity ───────────────────────────────────

def parity_sweep(size: Size, train_dt: float = 1.0) -> Path:
    """Overlay snntorch vs snntorch-library dt-sweep curves.

    At train-dt the two curves should be near-identical (matched architecture,
    matched input, matched optimizer). Differences elsewhere in the sweep
    isolate surrogate-gradient effects — snnTorch's fast_sigmoid(slope=25)
    produces dt-fragile weights relative to our slope-1 surrogate.
    """
    models = ["snntorch", "snntorch-library"]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    missing = []
    for m in models:
        sweep = _load_sweep(m, train_dt, size)
        if sweep is None:
            missing.append(m)
            continue
        dts = [r["dt"] for r in sweep]
        accs = [r["acc"] for r in sweep]
        ax.plot(dts, accs, "o-", color=COLORS[m], label=LABELS[m],
                markersize=6, linewidth=2.0)
    ax.axvline(train_dt, color="gray", ls="--", alpha=0.6,
               label=f"train $\\Delta t={train_dt}$ ms")
    ax.axhline(10, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"Inference $\Delta t$ (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(0, 2.1)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(
        f"Parity: our snntorch vs snnTorch library `snn.Leaky`\n"
        f"trained at $\\Delta t={train_dt}$ ms, {size.samples}x{size.epochs}",
        fontsize=11,
    )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()

    out = _out_path(f"parity_sweep_dt{train_dt}", size)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  ✓ {out}")
    if missing:
        print(f"  missing: {missing}")
    return out
