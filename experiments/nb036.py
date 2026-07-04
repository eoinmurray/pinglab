"""Notebook runner for entry 036 — coupling architecture of the
recruitment cliff.

Standalone runner with no cross-notebook helpers. Trains the
coba / ping × θ_u baseline sweep (needed for the frontier overlay
in Figure 2c), then runs the inference-time W^EI / W^IE coupling
sweep on the trained PING baseline, then trains the 25-cell
(W^EI, W^IE) grid (100 epochs each, seed 42), then trains the 30-cell
W^EI diagonal sweep (10 W^EI values × 3 seeds, 100 epochs each, with
W^IE = 2 W^EI). Figures land in /figures/notebooks/nb036/ and the
success-criteria summary in nb036/numbers.json.

Writing: writings/nb036.typ · figures + numbers.json: artifacts/data/nb036/
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
sys.path.insert(0, str(REPO / "src" / "notebooks"))
sys.path.insert(0, str(REPO / "src"))

from helpers import theme  # noqa: E402
from helpers.fmt import format_duration  # noqa: E402
from helpers.modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402
from nb022 import cell_dir as shared_cell_dir  # noqa: E402
from nb022 import cell_name  # noqa: E402

SLUG = "nb036"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
SNN_TOOL = REPO / "tools" / "snn" / "tool.py"

MAX_SAMPLES = 500
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


def theta_label(theta_u: float | None) -> str:
    """Filesystem-safe label for an out-dir."""
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
    """θ_u cell — now the shared nb022 cell (train-once / reuse-many). nb022
    owns the θ_u sweep; nb036 keeps its own coupling-grid cells locally."""
    return shared_cell_dir(cell_name(model, theta_u, seed))


def baseline_dir(model: str, seed: int = SEEDS_BASELINE[0]) -> Path:
    return cell_dir(model, None, seed)


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def _eval_scaled(
    train_dir: Path,
    scale_w_in: float = 1.0,
    scale_w_ei: float = 1.0,
    scale_w_ie: float = 1.0,
) -> tuple[float, float, float, float, float]:
    """Evaluate a trained cell with inference-time weight scaling, via the CLI.

    Shells out to `sim --infer` with the given weight-scale factors and reads
    metrics.json. Returns (acc, ce_loss, penalty, e_rate, i_rate). penalty is 0
    for these baseline (theta_u = off) cells — the trainer applied no firing-rate reg.
    """
    train_dir = train_dir.resolve()
    tag = f"win{scale_w_in:g}_wei{scale_w_ei:g}_wie{scale_w_ie:g}"
    out_dir = (ARTIFACTS / "coupling_sweep" / f"{train_dir.name}__{tag}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv", "run", "python", str(SNN_TOOL), "sim", "--infer",
            "--load-config", str(train_dir / "config.json"),
            "--load-weights", str(train_dir / "weights.pth"),
            "--scale-w-in", str(scale_w_in),
            "--scale-w-ei", str(scale_w_ei),
            "--scale-w-ie", str(scale_w_ie),
            "--out-dir", str(out_dir),
        ],
        cwd=REPO,
        check=True,
    )
    m = json.loads((out_dir / "metrics.json").read_text())
    rates = m.get("rates_hz", {})
    return (
        float(m["best_acc"]),
        float(m["ce_loss"]),
        0.0,
        float(rates.get("hid", 0.0)),
        float(rates.get("inh", 0.0)),
    )







# ── Coupling sweep (W^{EI} and W^{IE} scaling on top of W_in scaling) ──
#
# Validates the prof's first point: altering W_EI and W_IE should shift
# the critical recruitment fraction f*. For each coupling-scale value
# (multiplier on W_ei or W_ie), sweep W_in scale s and measure how the
# recruitment cliff moves.

COUPLING_SCALE_VALUES: list[float] = [0.25, 0.5, 1.0, 2.0]
# Inference-time W_in scale grid - same 24 points used to project the
# recruitment cliff in nb025, replayed here on top of each coupling
# scale to show how the cliff moves with E↔I coupling.
W_IN_SCALE_VALUES: list[float] = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00, 1.15, 1.30, 1.50,
    1.75, 2.00, 2.50, 3.00,
]


def run_coupling_sweep(notebook_run_id: str) -> list[dict]:
    """Inference-only on the trained PING baseline (θ_u = off, seed 42):
    for each (axis, coupling_scale, w_in_scale) point, multiply either
    W_ei or W_ie by the coupling scale, multiply W_in by s, evaluate.
    Records (axis, coupling_scale, scale, loss, penalty, total_loss,
    acc, rate_e, rate_i)."""
    train_dir = baseline_dir("ping")
    if not (train_dir / "weights.pth").exists():
        raise SystemExit(
            f"coupling-sweep needs trained PING weights at {train_dir}"
        )

    rows: list[dict] = []
    for axis in ("W_ei", "W_ie"):
        for coupling_scale in COUPLING_SCALE_VALUES:
            # Scale one coupling axis by this value (the other stays at 1.0).
            sc_ei = coupling_scale if axis == "W_ei" else 1.0
            sc_ie = coupling_scale if axis == "W_ie" else 1.0
            for scale in W_IN_SCALE_VALUES:
                acc, ce_loss, penalty, e_rate, i_rate = _eval_scaled(
                    train_dir,
                    scale_w_in=float(scale),
                    scale_w_ei=sc_ei,
                    scale_w_ie=sc_ie,
                )
                rows.append({
                    "axis": axis,
                    "coupling_scale": float(coupling_scale),
                    "scale": float(scale),
                    "loss": float(ce_loss),
                    "penalty": float(penalty),
                    "total_loss": float(ce_loss + penalty),
                    "acc": float(acc),
                    "rate_e": float(e_rate),
                    "rate_i": float(i_rate),
                })
                print(
                    f"  {axis} ×{coupling_scale:>4.2g}  s={scale:>5.2f}  "
                    f"acc={acc:5.2f}%  E={e_rate:6.2f} Hz  I={i_rate:6.2f} Hz"
                )
    return rows


def plot_coupling_sweep(rows: list[dict], out_path: Path, run_id: str) -> None:
    """Two-panel figure: left axis sweeps W^{EI} (W^{IE} held at 1), right
    sweeps W^{IE} (W^{EI} held at 1). Each panel is a stacked plot —
    accuracy (top) and I rate (bottom) vs W_in scale s, one curve per
    coupling scale value."""
    theme.apply()
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.0), dpi=150, sharex=True)
    cmap = plt.get_cmap("viridis")
    n = len(COUPLING_SCALE_VALUES)
    axes_labels = [("W_ei", "$W^{EI}$ scale"), ("W_ie", "$W^{IE}$ scale")]
    for col, (axis_key, axis_label) in enumerate(axes_labels):
        ax_acc = axes[0, col]
        ax_irate = axes[1, col]
        for j, c_scale in enumerate(COUPLING_SCALE_VALUES):
            color = cmap(j / max(n - 1, 1))
            msel = [
                r for r in rows
                if r["axis"] == axis_key and r["coupling_scale"] == c_scale
            ]
            xs = [r["scale"] for r in msel]
            ax_acc.plot(xs, [r["acc"] for r in msel], marker="o",
                        color=color, lw=1.5, label=f"× {c_scale:g}")
            ax_irate.plot(xs, [r["rate_i"] for r in msel], marker="o",
                          color=color, lw=1.5, label=f"× {c_scale:g}")
        ax_acc.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
        ax_acc.set_ylim(0, 100)
        ax_acc.axhline(10.0, color=theme.GREY_MID, lw=0.6, ls=":", alpha=0.5)
        ax_acc.set_title(axis_label, fontsize=theme.SIZE_TITLE)
        ax_acc.legend(
            fontsize=theme.SIZE_LABEL, frameon=False, loc="lower right",
            title=axis_label,
        )
        ax_irate.set_xlabel("$W_\\text{in}$ scale $s$", fontsize=theme.SIZE_LABEL)
        ax_irate.set_ylabel("I rate (Hz)", fontsize=theme.SIZE_LABEL)
        ax_irate.axvline(1.0, color=theme.GREY_MID, lw=0.6, ls="--", alpha=0.7)
        ax_acc.axvline(1.0, color=theme.GREY_MID, lw=0.6, ls="--", alpha=0.7)
    fig.suptitle(
        "Recruitment-cliff migration with coupling scale "
        "(trained PING, $\\theta_u =$ off)",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)



# ── End coupling sweep ────────────────────────────────────────────────


# ── W_ei × W_ie training grid sweep (GH #29) ─────────────────────────
#
# Train one PING-architecture network per cell of a 2D grid of (W_ei,
# W_ie) initialisations, all under heavy spike penalty (θ_u = 0.2).
# Read final test accuracy, E rate, I rate. Plot three heatmaps. The
# question: does the (acc, rate) outcome cluster into a PING-like
# region and a COBA-like region depending on coupling strength?

WEI_WIE_GRID_VALUES: list[float] = [0.0, 0.25, 0.5, 1.0, 2.0]  # absolute means
WEI_WIE_GRID_STD_FRAC: float = 0.1   # init std = STD_FRAC × mean
WEI_WIE_GRID_THETA_U: float = 0.2
WEI_WIE_GRID_SEED: int = SEED_SWEEP
WEI_WIE_GRID_EPOCHS: int = 100       # 100 epochs to match nb025/nb041/nb044 converged baselines


def wei_wie_grid_cell_dir(w_ei: float, w_ie: float) -> Path:
    """Per-cell artifact directory for the (W_ei, W_ie) grid."""
    a = f"{w_ei:g}".replace(".", "p")
    b = f"{w_ie:g}".replace(".", "p")
    return ARTIFACTS / f"ping__wei_wie_grid__wei{a}__wie{b}"


def build_wei_wie_grid_args(
    w_ei: float, w_ie: float, out_dir: Path,
) -> list[str]:
    """Train PING with explicit --w-ei and --w-ie overrides plus θ_u
    rate penalty. Uses the standard PING recipe in MODEL_RECIPES
    (--ei-strength 1 is left in so other fields keep their defaults,
    but --w-ei and --w-ie override its effect)."""
    recipe = dict(MODEL_RECIPES["ping"])
    args = [
        "train",
        "--model", recipe["__build_as"],
        "--dataset", "mnist",
        "--max-samples", str(MAX_SAMPLES),
        "--epochs", str(WEI_WIE_GRID_EPOCHS),
        "--t-ms", str(T_MS),
        "--dt", str(DT_TRAIN),
        "--seed", str(WEI_WIE_GRID_SEED),
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
    # Explicit overrides for W_ei, W_ie (these take precedence over
    # what --ei-strength would have set).
    args += [
        "--w-ei", str(w_ei), str(max(w_ei * WEI_WIE_GRID_STD_FRAC, 1e-6)),
        "--w-ie", str(w_ie), str(max(w_ie * WEI_WIE_GRID_STD_FRAC, 1e-6)),
        "--fr-reg-upper-theta", str(WEI_WIE_GRID_THETA_U),
        "--fr-reg-upper-strength", str(FR_STRENGTH_UPPER),
    ]
    return args


def plot_wei_wie_grid(rows: list[dict], out_path: Path, run_id: str) -> None:
    """Three heatmaps on the (W_ei, W_ie) grid: accuracy, E rate, I rate.
    I rate is the PING-vs-COBA cluster discriminator."""
    theme.apply()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5), dpi=150)
    g = WEI_WIE_GRID_VALUES
    n = len(g)
    val_to_idx = {v: i for i, v in enumerate(g)}
    A_acc = np.full((n, n), np.nan)
    A_e = np.full((n, n), np.nan)
    A_i = np.full((n, n), np.nan)
    for r in rows:
        i = val_to_idx[r["w_ei"]]
        j = val_to_idx[r["w_ie"]]
        A_acc[i, j] = r["best_acc"]
        A_e[i, j] = r["rate_e"]
        A_i[i, j] = r["rate_i"]
    for ax, A, title, cbar_label in [
        (axes[0], A_acc, "Test accuracy (%)", "%"),
        (axes[1], A_e, "Hidden E rate (Hz)", "Hz"),
        (axes[2], A_i, "Hidden I rate (Hz)", "Hz"),
    ]:
        im = ax.imshow(
            A, origin="lower", cmap="viridis",
            extent=[-0.5, n - 0.5, -0.5, n - 0.5], aspect="equal",
        )
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f"{v:g}" for v in g])
        ax.set_yticklabels([f"{v:g}" for v in g])
        ax.set_xlabel("$W^{IE}$ mean", fontsize=theme.SIZE_LABEL)
        ax.set_ylabel("$W^{EI}$ mean", fontsize=theme.SIZE_LABEL)
        ax.set_title(title, fontsize=theme.SIZE_TITLE)
        # annotate cells with values
        for i in range(n):
            for j in range(n):
                v = A[i, j]
                if not np.isnan(v):
                    ax.text(
                        j, i, f"{v:.1f}",
                        ha="center", va="center",
                        fontsize=theme.SIZE_ANNOTATION,
                        color="white" if v < (np.nanmax(A) + np.nanmin(A)) / 2 else "black",
                    )
        fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"5×5 (W_ei, W_ie) training grid at $\\theta_u = {WEI_WIE_GRID_THETA_U:g}$",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_wei_wie_acc_vs_e_with_frontier(
    grid_rows: list[dict],
    baseline_rows: list[dict],
    out_path: Path,
    run_id: str,
) -> None:
    """Figure 10b base (grid scatter coloured by I rate) with the COBA
    and PING θ_u-sweep frontiers from Figure 5 overlaid as connected
    curves. Lets the reader see where the (W^EI, W^IE)-grid solutions
    fall relative to the θ_u-driven accuracy/rate tradeoffs of the
    canonical COBA and PING baselines."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    # ── Base: Figure 10b scatter ──
    xs = np.array([r["rate_e"] for r in grid_rows])
    ys = np.array([r["best_acc"] for r in grid_rows])
    cs = np.array([r["rate_i"] for r in grid_rows])
    sc = ax.scatter(
        xs, ys, c=cs, cmap="viridis", s=70, edgecolor="k",
        linewidth=0.5, zorder=3, label=None,
    )
    cbar = fig.colorbar(sc, ax=ax, label="Hidden I rate (Hz)")
    cbar.ax.tick_params(labelsize=theme.SIZE_CAPTION)
    # ── Overlay: Figure 5 frontier per model ──
    for model in MODELS:
        pts: list[tuple[float, float, str]] = []
        for theta_u in THETA_U_GRID:
            cell = [
                r for r in baseline_rows
                if r["model"] == model and r["theta_u"] == theta_u
            ]
            if not cell:
                continue
            mean_acc = float(np.mean([r["final_acc"] for r in cell]))
            mean_rate = float(np.mean([r["rate_e"] for r in cell]))
            pts.append((mean_rate, mean_acc, theta_display(theta_u)))
        pts.sort()
        if not pts:
            continue
        rx = [p[0] for p in pts]
        ry = [p[1] for p in pts]
        ax.plot(
            rx, ry, color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model], lw=1.4, ms=6,
            label=f"{model} (θ_u sweep)", zorder=4,
        )
        for x, y, lab in pts:
            ax.annotate(
                lab, (x, y), xytext=(5, 5), textcoords="offset points",
                fontsize=theme.SIZE_CAPTION, color=theme.MUTED, alpha=0.85,
            )
    ax.set_xlabel("Hidden E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax.set_title(
        "Grid cells (dots) vs COBA/PING θ_u frontiers (lines)",
        fontsize=theme.SIZE_TITLE,
    )
    ax.set_ylim(70.0, 90.0)
    ax.legend(fontsize=theme.SIZE_CAPTION, frameon=False, loc="lower right")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_wei_wie_acc_vs_e(rows: list[dict], out_path: Path, run_id: str) -> None:
    """Accuracy vs hidden-E rate scatter, one dot per (W^EI, W^IE) grid
    cell from Figure 10, coloured by I rate. The cluster structure of
    the grid should appear as separated point clouds: I=0 cells (no-loop
    + stretched-COBA) at low I-rate colour, I>0 cells (PING) at high."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    xs = np.array([r["rate_e"] for r in rows])
    ys = np.array([r["best_acc"] for r in rows])
    cs = np.array([r["rate_i"] for r in rows])
    sc = ax.scatter(
        xs, ys, c=cs, cmap="viridis", s=70, edgecolor="k",
        linewidth=0.5, zorder=3,
    )
    for r in rows:
        ax.annotate(
            f"({r['w_ei']:g},{r['w_ie']:g})",
            (r["rate_e"], r["best_acc"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=theme.SIZE_CAPTION, color=theme.LABEL, alpha=0.75,
        )
    cbar = fig.colorbar(sc, ax=ax, label="Hidden I rate (Hz)")
    cbar.ax.tick_params(labelsize=theme.SIZE_CAPTION)
    ax.set_xlabel("Hidden E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax.set_title(
        "Grid cells in (E rate, accuracy) — coloured by I rate",
        fontsize=theme.SIZE_TITLE,
    )
    ax.set_ylim(70.0, 90.0)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── End W_ei × W_ie grid sweep ───────────────────────────────────────


# ── W_ei 1D sweep along the ei_ratio = 2 diagonal ─────────────────────
#
# 1D slice of the 5×5 grid above: hold W^IE = 2 × W^EI (the standard
# ei_ratio) and scan W^EI. Plots accuracy, E rate, I rate as smooth
# curves so the cluster transition is visible at finer resolution.

WEI_DIAGONAL_VALUES: list[float] = [
    0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0,
]
WEI_DIAGONAL_RATIO: float = 2.0       # W^IE / W^EI
WEI_DIAGONAL_THETA_U: float = 0.2     # same penalty as the grid
WEI_DIAGONAL_SEEDS: list[int] = list(SEEDS_BASELINE)  # 3-seed replicate
WEI_DIAGONAL_EPOCHS: int = 100        # 100 epochs to match nb025/nb041/nb044 converged baselines


def wei_diagonal_cell_dir(w_ei: float, seed: int) -> Path:
    a = f"{w_ei:g}".replace(".", "p")
    return ARTIFACTS / f"ping__wei_diagonal__wei{a}__seed{seed}"


def build_wei_diagonal_args(
    w_ei: float, seed: int, out_dir: Path,
) -> list[str]:
    w_ie = w_ei * WEI_DIAGONAL_RATIO
    recipe = dict(MODEL_RECIPES["ping"])
    recipe["--w-in"] = "0.6"   # match the W^EI×W^IE grid sweep
    args = [
        "train",
        "--model", recipe["__build_as"],
        "--dataset", "mnist",
        "--max-samples", str(MAX_SAMPLES),
        "--epochs", str(WEI_DIAGONAL_EPOCHS),
        "--t-ms", str(T_MS),
        "--dt", str(DT_TRAIN),
        "--seed", str(seed),
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    for k, v in recipe.items():
        if k.startswith("__"):
            continue
        if v is True:
            args.append(k)
        elif v is not None:
            if k == "--w-in":
                args += [k, v, "0.06"]
            else:
                args += [k, v]
    args += [
        "--w-ei", str(w_ei), str(max(w_ei * 0.1, 1e-6)),
        "--w-ie", str(w_ie), str(max(w_ie * 0.1, 1e-6)),
        "--fr-reg-upper-theta", str(WEI_DIAGONAL_THETA_U),
        "--fr-reg-upper-strength", str(FR_STRENGTH_UPPER),
    ]
    return args


def plot_wei_diagonal(rows: list[dict], out_path: Path, run_id: str) -> None:
    """Three panels: accuracy, E rate, I rate vs W^EI (with W^IE held
    at 2 × W^EI). Aggregates over the seeds present in rows and plots
    mean ± std as error bars. Individual seed points shown as faint
    markers."""
    theme.apply()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5), dpi=150)
    by_w: dict[float, list[dict]] = {}
    for r in rows:
        by_w.setdefault(float(r["w_ei"]), []).append(r)
    xs = sorted(by_w.keys())

    def agg(key: str) -> tuple[list[float], list[float], list[list[float]]]:
        means, stds, raw = [], [], []
        for w in xs:
            vals = [float(r[key]) for r in by_w[w]]
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=0)))
            raw.append(vals)
        return means, stds, raw

    panels = [
        ("best_acc", axes[0], "Test accuracy (%, best epoch)", theme.INK_BLACK, "Accuracy"),
        ("rate_e", axes[1], "E rate (Hz)", theme.INK_BLACK, "Hidden E rate"),
        ("rate_i", axes[2], "I rate (Hz)", theme.DEEP_RED, "Hidden I rate"),
    ]
    for key, ax, ylab, color, title in panels:
        means, stds, raw = agg(key)
        ax.errorbar(
            xs, means, yerr=stds, marker="o", color=color, lw=1.5,
            capsize=3, elinewidth=1.0, zorder=3,
        )
        # individual seeds as faint markers
        for w, vals in zip(xs, raw):
            ax.plot(
                [w] * len(vals), vals, marker=".", color=color,
                alpha=0.35, lw=0, ms=5, zorder=2,
            )
        ax.set_xlabel("$W^{EI}$ mean (init)", fontsize=theme.SIZE_LABEL)
        ax.set_ylabel(ylab, fontsize=theme.SIZE_LABEL)
        ax.axvline(1.0, color=theme.GREY_MID, lw=0.6, ls=":", alpha=0.7)
        ax.set_title(title, fontsize=theme.SIZE_TITLE)
    axes[0].set_ylim(0, 100)
    axes[0].axhline(10.0, color=theme.GREY_MID, lw=0.6, ls=":", alpha=0.5)
    n_seeds = max(len(v) for v in by_w.values()) if by_w else 0
    fig.suptitle(
        f"$W^{{EI}}$ sweep with $W^{{IE}} = {WEI_DIAGONAL_RATIO:g}\\,W^{{EI}}$ "
        f"(ei_ratio = {WEI_DIAGONAL_RATIO:g}), $\\theta_u = "
        f"{WEI_DIAGONAL_THETA_U:g}$, $n = {n_seeds}$ seeds (mean ± std)",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_wei_diagonal_acc_vs_e(
    rows: list[dict], out_path: Path, run_id: str,
) -> None:
    """Per-seed accuracy vs hidden-E rate scatter from the 30-epoch
    diagonal sweep, coloured by I rate. Same idea as Figure 10b but
    each point is one (W^EI, seed) cell; labels mark only the W^EI
    value (W^IE = 2 W^EI implicit)."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    xs = np.array([r["rate_e"] for r in rows])
    ys = np.array([r["best_acc"] for r in rows])
    cs = np.array([r["rate_i"] for r in rows])
    sc = ax.scatter(
        xs, ys, c=cs, cmap="viridis", s=70, edgecolor="k",
        linewidth=0.5, zorder=3,
    )
    # Label only the seed-42 point at each W^EI to avoid clutter.
    seen: set[float] = set()
    for r in rows:
        if r["w_ei"] in seen:
            continue
        if int(r.get("seed", -1)) != 42:
            continue
        seen.add(r["w_ei"])
        ax.annotate(
            f"{r['w_ei']:g}",
            (r["rate_e"], r["best_acc"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=theme.SIZE_CAPTION, color=theme.LABEL, alpha=0.75,
        )
    cbar = fig.colorbar(sc, ax=ax, label="Hidden I rate (Hz)")
    cbar.ax.tick_params(labelsize=theme.SIZE_CAPTION)
    ax.set_xlabel("Hidden E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Best-epoch test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax.set_title(
        "Diagonal cells in (E rate, accuracy) — coloured by I rate",
        fontsize=theme.SIZE_TITLE,
    )
    ax.set_ylim(70.0, 90.0)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── End W_ei diagonal sweep ──────────────────────────────────────────


# Run scale — stamped into the manifest by run_dirs.prepare and rendered as
# the Methods table via RunScale; the mdx never restates these numbers.
SCALE = {
    "dataset": "mnist",
    "max_samples": MAX_SAMPLES,
    "epochs": WEI_WIE_GRID_EPOCHS,
    "t_ms": T_MS,
    "dt_ms": DT_TRAIN,
    "batch_size": 256,
    "seeds": len(WEI_DIAGONAL_SEEDS),
    "cells": (
        len(WEI_WIE_GRID_VALUES) ** 2
        + len(WEI_DIAGONAL_VALUES) * len(WEI_DIAGONAL_SEEDS)
    ),
    "grid": "5×5 (W^EI,W^IE) grid + 10-point W^EI diagonal × 3 seeds",
}


def main() -> None:
    modal_gpu = parse_modal_gpu(sys.argv)
    skip_training = "--skip-training" in sys.argv
    wipe_dir = "--no-wipe-dir" not in sys.argv

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    n_cells = len(MODELS) * len(THETA_U_GRID)
    print(
        f"notebook_run_id = {notebook_run_id} cells={n_cells}"
        + ("  [skip-training]" if skip_training else "")
    )

    prepare_run_dirs(
        SLUG, notebook_run_id, wipe=wipe_dir, skip_training=skip_training,
        make_artifacts=False,
        scale=SCALE,
        host=f"modal:{modal_gpu}" if modal_gpu else "local",
    )

    only_missing = "--only-missing" in sys.argv
    if not skip_training:
        dispatcher = BatchDispatcher(modal_gpu, REPO, SNN_TOOL)
        # θ_u baseline/sweep cells (needed for the frontier overlay) are
        # trained in nb022 now (train-once / reuse-many); read via cell_dir.
        # nb036 trains only its own coupling cells below.
        gpu_override = None
        if modal_gpu in ("T4", "L4", "A10G"):
            gpu_override = "A100"
        # W_ei × W_ie 5×5 training grid (GH #29).
        for w_ei in WEI_WIE_GRID_VALUES:
            for w_ie in WEI_WIE_GRID_VALUES:
                out = wei_wie_grid_cell_dir(w_ei, w_ie)
                if only_missing and (out / "metrics.json").exists():
                    print(
                        f"[skip] wei_wie_grid/{w_ei}/{w_ie} already trained → "
                        f"{out.relative_to(REPO)}"
                    )
                    continue
                print(
                    f"[train] wei_wie_grid/wei={w_ei}/wie={w_ie} → "
                    f"{out.relative_to(REPO)}"
                    + (f"  [modal:{modal_gpu}]" if modal_gpu else "")
                )
                dispatcher.submit(
                    build_wei_wie_grid_args(w_ei, w_ie, out),
                    out,
                    gpu_override=gpu_override,
                )
        # W_ei 1D diagonal sweep with W^IE = ei_ratio × W^EI, 3 seeds.
        for w_ei in WEI_DIAGONAL_VALUES:
            for seed in WEI_DIAGONAL_SEEDS:
                out = wei_diagonal_cell_dir(w_ei, seed)
                if only_missing and (out / "metrics.json").exists():
                    print(
                        f"[skip] wei_diagonal/{w_ei}/seed={seed} already "
                        f"trained → {out.relative_to(REPO)}"
                    )
                    continue
                print(
                    f"[train] wei_diagonal/wei={w_ei}/seed={seed} → "
                    f"{out.relative_to(REPO)}"
                    + (f"  [modal:{modal_gpu}]" if modal_gpu else "")
                )
                dispatcher.submit(
                    build_wei_diagonal_args(w_ei, seed, out),
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

    print("  baseline results:")
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

    # Coupling sweep — validates the recruitment-fraction shift prediction.
    print("[coupling-sweep] inference-only on trained PING (θ_u = off)")
    coupling_rows = run_coupling_sweep(notebook_run_id)
    plot_coupling_sweep(
        coupling_rows, FIGURES / "coupling_sweep.png", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'coupling_sweep.png'}")

    # W_ei × W_ie 5×5 training grid (GH #29).
    print("[wei-wie-grid] reading metrics from 5×5 grid trainings")
    wei_wie_rows: list[dict] = []
    for w_ei in WEI_WIE_GRID_VALUES:
        for w_ie in WEI_WIE_GRID_VALUES:
            run_dir = wei_wie_grid_cell_dir(w_ei, w_ie)
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                print(f"  missing: {run_dir.name} (skipping in plot)")
                continue
            metrics = load_metrics(run_dir)
            last = metrics["epochs"][-1]
            wei_wie_rows.append({
                "w_ei": float(w_ei),
                "w_ie": float(w_ie),
                "best_acc": float(metrics["best_acc"]),
                "final_acc": float(last["acc"]),
                "rate_e": float(last.get("rate_e") or 0.0),
                "rate_i": float(last.get("rate_i") or 0.0),
            })
    if wei_wie_rows:
        plot_wei_wie_grid(
            wei_wie_rows, FIGURES / "wei_wie_grid.png", notebook_run_id,
        )
        print(f"wrote {FIGURES / 'wei_wie_grid.png'}")
        plot_wei_wie_acc_vs_e(
            wei_wie_rows, FIGURES / "wei_wie_acc_vs_e.png", notebook_run_id,
        )
        print(f"wrote {FIGURES / 'wei_wie_acc_vs_e.png'}")
        plot_wei_wie_acc_vs_e_with_frontier(
            wei_wie_rows, rows,
            FIGURES / "wei_wie_acc_vs_e_with_frontier.png",
            notebook_run_id,
        )
        print(f"wrote {FIGURES / 'wei_wie_acc_vs_e_with_frontier.png'}")

    # W_ei 1D diagonal sweep (W^IE = 2 W^EI), 3 seeds.
    print("[wei-diagonal] reading metrics from 1D diagonal trainings")
    wei_diagonal_rows: list[dict] = []
    for w_ei in WEI_DIAGONAL_VALUES:
        for seed in WEI_DIAGONAL_SEEDS:
            run_dir = wei_diagonal_cell_dir(w_ei, seed)
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                print(f"  missing: {run_dir.name} (skipping in plot)")
                continue
            metrics = load_metrics(run_dir)
            last = metrics["epochs"][-1]
            wei_diagonal_rows.append({
                "w_ei": float(w_ei),
                "w_ie": float(w_ei * WEI_DIAGONAL_RATIO),
                "seed": int(seed),
                "best_acc": float(metrics["best_acc"]),
                "final_acc": float(last["acc"]),
                "rate_e": float(last.get("rate_e") or 0.0),
                "rate_i": float(last.get("rate_i") or 0.0),
            })
            print(
                f"  W_ei={w_ei:>5}  seed={seed}  "
                f"acc={last['acc']:5.2f}%  "
                f"E={last.get('rate_e') or 0:6.2f} Hz  "
                f"I={last.get('rate_i') or 0:6.2f} Hz"
            )
    if wei_diagonal_rows:
        plot_wei_diagonal(
            wei_diagonal_rows, FIGURES / "wei_diagonal.png", notebook_run_id,
        )
        print(f"wrote {FIGURES / 'wei_diagonal.png'}")
        plot_wei_diagonal_acc_vs_e(
            wei_diagonal_rows,
            FIGURES / "wei_diagonal_acc_vs_e.png",
            notebook_run_id,
        )
        print(f"wrote {FIGURES / 'wei_diagonal_acc_vs_e.png'}")

    duration_s = time.monotonic() - t_start
    train_cfg = load_config(baseline_dir(MODELS[0]))
    summary = {
        "notebook_run_id": notebook_run_id,
        "git_sha": train_cfg.get("git_sha"),
        "duration_s": round(duration_s, 1),
        "duration": format_duration(duration_s),
        "config": {
            "dataset": "mnist",
            "models": MODELS,
            "theta_u_grid_spikes": [t for t in THETA_U_GRID if t is not None],
            "theta_u_grid_hz": [
                theta_hz(t) for t in THETA_U_GRID if t is not None
            ],
            "max_samples": MAX_SAMPLES,
            "epochs": 5,
            "t_ms": T_MS,
            "dt": DT_TRAIN,
            "seeds_baseline": SEEDS_BASELINE,
            "seed_sweep": SEED_SWEEP,
            "fr_strength_upper": FR_STRENGTH_UPPER,
        },
        "baseline_results": rows,
        "coupling_sweep": coupling_rows,
        "wei_wie_grid": wei_wie_rows,
        "wei_diagonal": wei_diagonal_rows,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")



if __name__ == "__main__":
    main()
