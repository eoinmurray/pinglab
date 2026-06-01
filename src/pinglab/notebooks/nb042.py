"""Notebook runner for entry 042 — CUBA-PING with a spike penalty.

θ_u sweep on both arms from [nb040](/notebooks/nb040/) — CUBA-PING and
CUBA-no-PING — adding the firing-rate regulariser term from [nb035](/notebooks/nb035/)
to slide each architecture along the rate–accuracy frontier. Tests
whether (a) the L_rate term tightens PING's already-sparse operating
point further, and (b) penalty alone can recover PING-style sparsity
on the no-PING control without the I-loop architecture.

Notebook entry: src/docs/src/pages/notebooks/nb042.mdx
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
from _tier import parse_tier  # noqa: E402
from pinglab import theme  # noqa: E402

SLUG = "nb042"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "cli" / "__main__.py"

# ── Architecture / recipe — inherits from nb040 ──────────────────
N_E: int = 1024
N_IN: int = 784
N_CLASSES: int = 10

T_MS: float = 200.0
DT: float = 1.0
TBPTT_WINDOW: int = max(1, round(10.0 / DT))

W_IN_MEAN: float = 0.0
W_IN_STD: float = 0.5
W_IN_SPARSITY: float = 0.95
W_EI_MEAN: float = 1.0
W_EI_STD: float = 0.1
W_IE_MEAN: float = 1.0
W_IE_STD: float = 0.1

SEED: int = 42
TAU_M_MS: float = 20.0
TAU_OUT_MS: float = 20.0
INPUT_RATE_HZ: float = 80.0
LR: float = 2e-3
BATCH_SIZE: int = 64

# ── Sweep config ─────────────────────────────────────────────────
TIER_CONFIG = {
    "extra small": dict(max_samples=200, epochs=2),
    "small": dict(max_samples=1000, epochs=5),
    "medium": dict(max_samples=5000, epochs=15),
    "large": dict(max_samples=10000, epochs=30),
    "extra large": dict(max_samples=20000, epochs=40),
}
DEFAULT_TIER = "medium"
# Per-cell epoch override — set to None to fall back to TIER_CONFIG, or
# an int to use that many epochs across all tiers. The under-converged
# tight-θ_u cells (see Figure 3) need substantially more than the
# medium-tier default of 15 epochs to resolve whether they recover
# toward the PING-off baseline.
EPOCHS_OVERRIDE: int | None = 40

# θ_u grid in spikes per trial. None = no penalty (baseline = nb040).
# Same grid as nb035/036 for cross-comparison.
THETA_U_GRID: list[float | None] = [None, 5.0, 2.0, 1.0, 0.5, 0.2]
FR_STRENGTH_UPPER: float = 1e-3
# Population-mean penalty (not per-neuron). The per-neuron formulation
# concentrates on the high-firing tail and crushes PING's selective
# code; the population formulation distributes pressure uniformly across
# all cells and lets the I-loop's architectural floor coexist with the
# explicit penalty. See nb042 mdx Findings.
FR_REG_MODE: str = "population"

ARMS = ("ping", "noping")
MODEL_FOR_ARM = {"ping": "cuba-ping", "noping": "cuba-noping"}
LABEL_FOR_ARM = {"ping": "CUBA-PING", "noping": "CUBA-no-PING"}
COLOR_FOR_ARM = {"ping": theme.INK_BLACK, "noping": theme.DEEP_RED}
MARKER_FOR_ARM = {"ping": "o", "noping": "s"}


def theta_tag(theta_u: float | None) -> str:
    return "off" if theta_u is None else f"{theta_u:g}".replace(".", "p")


def theta_display(theta_u: float | None) -> str:
    return "off" if theta_u is None else f"{theta_u:g}"


def cell_dir(arm: str, theta_u: float | None) -> Path:
    return ARTIFACTS / arm / theta_tag(theta_u)


def build_oscilloscope_args(
    arm: str, theta_u: float | None, tier: str, out_dir: Path,
) -> list[str]:
    args = [
        "train",
        "--model", MODEL_FOR_ARM[arm],
        "--dataset", "mnist",
        "--max-samples", str(TIER_CONFIG[tier]["max_samples"]),
        "--epochs", str(EPOCHS_OVERRIDE or TIER_CONFIG[tier]["epochs"]),
        "--t-ms", str(T_MS),
        "--dt", str(DT),
        "--input-rate", str(INPUT_RATE_HZ),
        "--seed", str(SEED),
        "--n-hidden", str(N_E),
        "--w-in", str(W_IN_MEAN), str(W_IN_STD),
        "--w-in-sparsity", str(W_IN_SPARSITY),
        "--readout", "mem-mean",
        "--surrogate-slope", "1",
        "--tau-mem", str(TAU_M_MS),
        "--readout-tau-out", str(TAU_OUT_MS),
        "--lr", str(LR),
        "--batch-size", str(BATCH_SIZE),
        "--grad-clip", "1e6",
        "--tbptt-window", str(TBPTT_WINDOW),
        "--no-dales-law",
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    if arm == "ping":
        args += [
            "--w-ei", str(W_EI_MEAN), str(W_EI_STD),
            "--w-ie", str(W_IE_MEAN), str(W_IE_STD),
        ]
    if theta_u is not None:
        args += [
            "--fr-reg-upper-theta", str(theta_u),
            "--fr-reg-upper-strength", str(FR_STRENGTH_UPPER),
            "--fr-reg-mode", FR_REG_MODE,
        ]
    return args


# ── Plotting ─────────────────────────────────────────────────────
def _stamp(fig) -> None:
    fig.text(
        0.995, 0.005, f"{SLUG}-{int(time.time())}",
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


def plot_frontier(rows: list[dict], out_path: Path) -> None:
    """Accuracy vs E rate, one trace per arm, one point per θ_u."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    for arm in ARMS:
        cells = [r for r in rows if r["arm"] == arm]
        cells.sort(key=lambda r: -1.0 if r["theta_u"] is None else r["theta_u"])
        xs = [r["rate_e_hz"] for r in cells]
        ys = [r["acc"] for r in cells]
        ax.scatter(
            xs, ys,
            marker=MARKER_FOR_ARM[arm], color=COLOR_FOR_ARM[arm],
            s=60, label=LABEL_FOR_ARM[arm],
            edgecolor=theme.INK_BLACK, linewidth=0.6,
        )
        for r in cells:
            label = "off" if r["theta_u"] is None else f"θ={r['theta_u']:g}"
            ax.annotate(
                label,
                xy=(r["rate_e_hz"], r["acc"]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=theme.SIZE_CAPTION,
                color=COLOR_FOR_ARM[arm], alpha=0.75,
            )
    ax.set_xlabel("Mean hidden-E firing rate (Hz)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Rate–accuracy frontier under θ_u sweep", fontsize=theme.SIZE_TITLE)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=theme.SIZE_CAPTION, frameon=False, loc="lower right")
    fig.tight_layout()
    _stamp(fig)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _build_net_for_arm(arm: str):
    """Construct the same CubaPingNet/CubaNoPingNet the train CLI builds,
    identically seeded. Used for raster replays."""
    import torch
    torch.manual_seed(SEED)
    import models as M
    M.HIDDEN_SIZES = [N_E]
    M.N_IN = N_IN
    M.N_OUT = N_CLASSES
    M.N_HID = N_E
    M.N_INH = N_E // 4
    M.T_steps = int(T_MS / DT)
    M.T_ms = T_MS
    M.dt = DT
    from config import build_net
    kw = dict(
        w_in=(W_IN_MEAN, W_IN_STD),
        w_in_sparsity=W_IN_SPARSITY,
        hidden_sizes=[N_E],
    )
    if arm == "ping":
        kw["w_ei"] = (W_EI_MEAN, W_EI_STD)
        kw["w_ie"] = (W_IE_MEAN, W_IE_STD)
    return build_net(MODEL_FOR_ARM[arm], **kw)


def _replay_digit0(arm: str, theta_u: float | None):
    """Load this cell's trained weights, replay a digit-0 MNIST sample, and
    return (spk_E, spk_I_or_None) at shape (T, n)."""
    import torch
    from torchvision import datasets, transforms
    cache = REPO / ".cache" / "mnist"
    test_ds = datasets.MNIST(
        cache, train=False, download=True, transform=transforms.ToTensor(),
    )
    # Pick the first digit-0 sample so the panel is comparable across cells.
    idx = next(i for i in range(len(test_ds)) if test_ds[i][1] == 0)
    img, _ = test_ds[idx]
    pixels = img.view(-1)
    T = int(T_MS / DT)
    p = pixels.unsqueeze(0).unsqueeze(0) * (INPUT_RATE_HZ * DT / 1000.0)
    p = p.expand(T, 1, N_IN).contiguous()
    gen = torch.Generator().manual_seed(SEED + 3)
    spk_in = torch.bernoulli(p, generator=gen)

    net = _build_net_for_arm(arm)
    weights = cell_dir(arm, theta_u) / "weights.pth"
    state = torch.load(weights, map_location="cpu")
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True
    with torch.no_grad():
        net(input_spikes=spk_in)
    rec = net.spike_record
    spk_e = rec["hid"].cpu().numpy()
    spk_i = rec["inh"].cpu().numpy() if "inh" in rec else None
    return spk_e, spk_i


def plot_raster_grid(arm: str, rows: list[dict], out_path: Path) -> None:
    """6-panel raster grid for one arm — one panel per θ_u value, all on a
    trained replay of MNIST digit 0. PING panels show a single composite
    raster: E (black) on top, I (red) below, sharing the same x-axis with
    no visible gap. no-PING panels show E only."""
    theme.apply()
    has_inh = (arm == "ping")
    fig = plt.figure(figsize=(13.0, 6.5), dpi=150)
    # Outer 2×3 layout; each PING panel becomes a 4:1 E/I stack via a nested
    # SubplotSpec so the two share an x-axis with no inter-panel gap.
    outer = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.18)
    panels = []
    for r in range(2):
        for c in range(3):
            if has_inh:
                inner = outer[r, c].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.0)
                ax_e = fig.add_subplot(inner[0])
                ax_i = fig.add_subplot(inner[1], sharex=ax_e)
                panels.append((ax_e, ax_i))
            else:
                panels.append(fig.add_subplot(outer[r, c]))

    by_theta = {r["theta_u"]: r for r in rows if r["arm"] == arm}
    for idx, theta_u in enumerate(THETA_U_GRID):
        spk_e, spk_i = _replay_digit0(arm, theta_u)
        r = by_theta[theta_u]
        title = (
            f"θ_u={theta_display(theta_u)}  "
            f"acc={r['acc']:.1f}%  E={r['rate_e_hz']:.1f}Hz"
            + (f"  I={r['rate_i_hz']:.0f}Hz" if has_inh else "")
        )
        t_ms = np.arange(spk_e.shape[0]) * DT
        if has_inh:
            ax_e, ax_i = panels[idx]
            e_idx, e_t = np.where(spk_e.T)
            ax_e.scatter(
                t_ms[e_t], e_idx, s=0.5, c=theme.INK_BLACK,
                marker="|", linewidths=0.35,
            )
            ax_e.set_ylim(0, spk_e.shape[1])
            ax_e.set_xlim(0, T_MS)
            ax_e.set_yticks([])
            ax_e.set_ylabel("E", fontsize=theme.SIZE_CAPTION)
            ax_e.set_title(title, fontsize=theme.SIZE_CAPTION, pad=4)
            ax_e.tick_params(labelbottom=False)
            i_idx, i_t = np.where(spk_i.T)
            ax_i.scatter(
                t_ms[i_t], i_idx, s=0.5, c=theme.DEEP_RED,
                marker="|", linewidths=0.35,
            )
            ax_i.set_ylim(0, spk_i.shape[1])
            ax_i.set_xlim(0, T_MS)
            ax_i.set_yticks([])
            ax_i.set_ylabel("I", fontsize=theme.SIZE_CAPTION)
            if idx >= 3:
                ax_i.set_xlabel("time (ms)", fontsize=theme.SIZE_CAPTION)
            else:
                ax_i.tick_params(labelbottom=False)
        else:
            ax = panels[idx]
            e_idx, e_t = np.where(spk_e.T)
            ax.scatter(
                t_ms[e_t], e_idx, s=0.7, c=theme.INK_BLACK,
                marker="|", linewidths=0.45,
            )
            ax.set_ylim(0, spk_e.shape[1])
            ax.set_xlim(0, T_MS)
            ax.set_ylabel("E neuron", fontsize=theme.SIZE_CAPTION)
            ax.set_title(title, fontsize=theme.SIZE_CAPTION, pad=4)
            if idx >= 3:
                ax.set_xlabel("time (ms)", fontsize=theme.SIZE_CAPTION)
            else:
                ax.tick_params(labelbottom=False)
    fig.suptitle(
        f"{LABEL_FOR_ARM[arm]} — trained rasters on MNIST digit 0",
        fontsize=theme.SIZE_TITLE,
    )
    _stamp(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(cells_by_arm: dict, out_path: Path) -> None:
    """Loss / acc / E rate over epochs, one row per arm, one curve per θ_u."""
    theme.apply()
    fig, axes = plt.subplots(
        2, 3, figsize=(13.0, 7.5), dpi=150, sharex=True,
    )
    # Color θ_u values from light → dark with the off baseline highlighted.
    n_theta = len(THETA_U_GRID)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(0.15 + 0.7 * i / max(n_theta - 1, 1)) for i in range(n_theta)]
    for row, arm in enumerate(ARMS):
        for col, key, ylab, title in (
            (0, "loss", "Cross-entropy loss", "Training loss"),
            (1, "acc", "Test accuracy (%)", "Test accuracy"),
            (2, "rate_e_hz", "Mean E rate (Hz)", "Hidden E firing rate"),
        ):
            ax = axes[row, col]
            for j, theta_u in enumerate(THETA_U_GRID):
                eps = cells_by_arm[(arm, theta_u)]
                xs = [r["ep"] for r in eps]
                if key == "rate_e_hz":
                    ys = [r["test_rate_e"] for r in eps]
                elif key == "acc":
                    ys = [r["acc"] for r in eps]
                else:
                    ys = [r["loss"] for r in eps]
                label = "θ=off" if theta_u is None else f"θ={theta_u:g}"
                ax.plot(
                    xs, ys, marker="o", ms=3,
                    color=colors[j], lw=1.4, label=label,
                )
            ax.set_ylabel(ylab)
            ax.set_title(
                f"{LABEL_FOR_ARM[arm]} — {title}", fontsize=theme.SIZE_TITLE,
            )
            ax.grid(True, alpha=0.3)
            if col == 1:
                ax.set_ylim(0, 100)
        axes[row, 0].legend(fontsize=theme.SIZE_CAPTION, frameon=False, ncol=2)
        axes[row, -1].set_xlabel("Epoch")
    for col in range(3):
        axes[-1, col].set_xlabel("Epoch")
    fig.tight_layout()
    _stamp(fig)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_headline_bars(rows: list[dict], out_path: Path) -> None:
    """Per-θ_u final accuracy and E rate, grouped by arm."""
    theme.apply()
    grid = [t for t in THETA_U_GRID]
    fig, (ax_acc, ax_rate) = plt.subplots(1, 2, figsize=(11.0, 4.5), dpi=150)
    x = np.arange(len(grid))
    width = 0.4
    for i, arm in enumerate(ARMS):
        cells = {r["theta_u"]: r for r in rows if r["arm"] == arm}
        accs = [cells[t]["acc"] for t in grid]
        rates = [cells[t]["rate_e_hz"] for t in grid]
        ax_acc.bar(
            x + (i - 0.5) * width, accs, width,
            color=COLOR_FOR_ARM[arm], edgecolor=theme.INK_BLACK,
            linewidth=0.8, label=LABEL_FOR_ARM[arm],
        )
        ax_rate.bar(
            x + (i - 0.5) * width, rates, width,
            color=COLOR_FOR_ARM[arm], edgecolor=theme.INK_BLACK,
            linewidth=0.8, label=LABEL_FOR_ARM[arm],
        )
    for ax, ylab, title in (
        (ax_acc, "Test accuracy (%)", "Final accuracy"),
        (ax_rate, "Mean hidden-E rate (Hz)", "Final E firing rate"),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels([theta_display(t) for t in grid])
        ax.set_xlabel(r"$\theta_u$ (spikes per trial)")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=theme.SIZE_TITLE)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=theme.SIZE_CAPTION, frameon=False)
    ax_acc.set_ylim(0, 100)
    fig.tight_layout()
    _stamp(fig)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────
def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    wipe_dir = "--no-wipe-dir" not in sys.argv

    if wipe_dir:
        for d in (ARTIFACTS, FIGURES):
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    FIGURES.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()
    cells = [(arm, t) for arm in ARMS for t in THETA_U_GRID]
    print(
        f"[{SLUG}] CUBA-PING + CUBA-no-PING × θ_u sweep, "
        f"tier={tier}, cells={len(cells)}"
    )

    dispatcher = BatchDispatcher(modal_gpu, REPO, OSCILLOSCOPE)
    for arm, theta_u in cells:
        out = cell_dir(arm, theta_u)
        print(
            f"  → {arm} θ_u={theta_display(theta_u)} → "
            f"{out.relative_to(REPO)}"
            + (f"  [modal:{modal_gpu}]" if modal_gpu else "")
        )
        dispatcher.submit(
            build_oscilloscope_args(arm, theta_u, tier, out), out
        )
    dispatcher.drain()

    # ── Read metrics + plot
    rows: list[dict] = []
    cells_by_arm: dict = {}
    for arm, theta_u in cells:
        out = cell_dir(arm, theta_u)
        m = json.loads((out / "metrics.json").read_text())
        cells_by_arm[(arm, theta_u)] = m["epochs"]
        last = m["epochs"][-1]
        rows.append({
            "arm": arm,
            "theta_u": theta_u,
            "acc": last["acc"],
            "rate_e_hz": last["test_rate_e"],
            "rate_i_hz": last.get("test_rate_i", 0.0),
            "loss": last["loss"],
        })
    plot_frontier(rows, FIGURES / "frontier.png")
    print(f"  wrote {FIGURES / 'frontier.png'}")
    plot_training_curves(cells_by_arm, FIGURES / "training_curves.png")
    print(f"  wrote {FIGURES / 'training_curves.png'}")
    plot_headline_bars(rows, FIGURES / "headline_bars.png")
    print(f"  wrote {FIGURES / 'headline_bars.png'}")
    for arm in ARMS:
        plot_raster_grid(arm, rows, FIGURES / f"rasters__{arm}.png")
        print(f"  wrote {FIGURES / f'rasters__{arm}.png'}")

    duration_s = time.monotonic() - t_start
    ping_off = next(
        r for r in rows if r["arm"] == "ping" and r["theta_u"] is None
    )
    nop_off = next(
        r for r in rows if r["arm"] == "noping" and r["theta_u"] is None
    )
    summary = {
        "slug": SLUG,
        "tier": tier,
        "duration_s": round(duration_s, 1),
        "config": {
            "n_e": N_E, "n_in": N_IN, "n_classes": N_CLASSES,
            "t_ms": T_MS, "dt": DT, "tau_m_ms": TAU_M_MS,
            "tau_out_ms": TAU_OUT_MS,
            "input_rate_hz": INPUT_RATE_HZ,
            "max_samples": TIER_CONFIG[tier]["max_samples"],
            "epochs": EPOCHS_OVERRIDE or TIER_CONFIG[tier]["epochs"],
            "batch_size": BATCH_SIZE, "lr": LR, "seed": SEED,
            "theta_u_grid": [t for t in THETA_U_GRID],
            "fr_strength_upper": FR_STRENGTH_UPPER,
            "fr_reg_mode": FR_REG_MODE,
            "tbptt_window": TBPTT_WINDOW,
        },
        "cells": rows,
        # Top-level shortcuts for the index page (use ping@θ=off baseline)
        "final_acc": ping_off["acc"],
        "final_rate_e_hz": ping_off["rate_e_hz"],
        "final_rate_i_hz": ping_off["rate_i_hz"],
        "success_criteria": [
            {
                "label": "frontier rendered",
                "passed": (FIGURES / "frontier.png").exists(),
                "detail": (
                    f"PING off → {ping_off['acc']:.1f}% @ "
                    f"{ping_off['rate_e_hz']:.1f} Hz"
                ),
            },
            {
                "label": "no-PING baseline above chance",
                "passed": nop_off["acc"] > 15.0,
                "detail": f"{nop_off['acc']:.2f}%",
            },
            {
                "label": "spike penalty lowers no-PING rate",
                "passed": any(
                    r["arm"] == "noping" and r["theta_u"] is not None
                    and r["rate_e_hz"] < nop_off["rate_e_hz"] * 0.5
                    for r in rows
                ),
                "detail": (
                    "≥1 θ_u cell with no-PING E rate < 50% of baseline"
                ),
            },
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {duration_s:.1f}s")


if __name__ == "__main__":
    main()
