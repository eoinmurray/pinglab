"""Repro for the 2026-04-17-1100 snntorch parity journal entry.

Run from the repository root:

    uv run python src/pinglab/journal/\\
        2026-04-17-1100-snntorch-parity-and-calibration.py

Regenerates every figure and every number the entry cites, from the
artifacts under src/artifacts/mnist-dt-stability/calibrations/modal/mnist/.
No arguments. Errors clearly if an input artifact is missing.

Journal entry: src/docs/src/pages/journal/
    2026-04-17-1100-snntorch-parity-and-calibration.md

Outputs (under src/docs/public/figures/journal/
    2026-04-17-1100-snntorch-parity-and-calibration/):
    - parity_and_calibration.png
    - rasters.png
    - numbers.json

Convention: entries whose inputs are already on disk do their work inline
with Python; entries that need to train / sweep / infer use the `sh`
library to shell out to oscilloscope.py and the sweep harness. This
entry's inputs exist, so no sh calls are made — but the import stays as
documentation of the pattern.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import sh  # noqa: F401 — convention: available for future entries that need to train/infer

import numpy as np
import torch

# ── Module-level state for the raster forward pass ──────────────────────
# config / models read module-level constants that must be set on the
# same module object build_net uses internally. Inserting src/pinglab on
# sys.path and importing by bare name gets us the right object.
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config as M  # noqa: E402
import models as MDL  # noqa: E402
from config import build_net  # noqa: E402
from oscilloscope import load_dataset, patch_dt, encode_batch  # noqa: E402


# ── Paths ───────────────────────────────────────────────────────────────
ART = REPO / "src/artifacts/mnist-dt-stability/calibrations/modal/mnist"
FIG_DIR = REPO / ("src/docs/public/figures/journal/"
                  "2026-04-17-1100-snntorch-parity-and-calibration")

SIZES = [(50, 1), (100, 3), (500, 10), (1000, 40), (15000, 40)]


# ── Loaders ─────────────────────────────────────────────────────────────
def load_sweep(model: str, train_dt: float, samples: int, epochs: int):
    p = ART / f"dt{train_dt}" / f"{model}.{samples}.{epochs}" / "infer-frozen" / "results.json"
    if not p.exists():
        return None
    payload = json.load(open(p))
    return payload.get("sweep", payload) if isinstance(payload, dict) else payload


def load_best(model: str, train_dt: float, samples: int, epochs: int) -> float | None:
    m = ART / f"dt{train_dt}" / f"{model}.{samples}.{epochs}" / "metrics.json"
    if not m.exists():
        return None
    return json.load(open(m)).get("best_acc")


# ── Figure 1: parity sweep + calibration ladder ─────────────────────────
def make_parity_and_calibration() -> Path:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    # Left: parity at 500.10 dt=1.0
    for model, colour, lbl in [("snntorch", "#1f77b4", "snntorch (ours, slope-1)"),
                               ("snntorch-library", "#d62728",
                                "snntorch-library (snn.Leaky, slope-25)")]:
        sweep = load_sweep(model, 1.0, 500, 10)
        if sweep is None:
            raise FileNotFoundError(
                f"Missing sweep: {ART}/dt1.0/{model}.500.10/infer-frozen/results.json")
        dts = [r["dt"] for r in sweep]
        accs = [r["acc"] for r in sweep]
        axL.plot(dts, accs, "o-", color=colour, label=lbl, lw=2.0, ms=6)
    axL.axvline(1.0, color="gray", ls="--", alpha=0.6, label=r"train $\Delta t$")
    axL.axhline(10, color="gray", ls=":", lw=0.8, alpha=0.5)
    axL.set_xlabel(r"Inference $\Delta t$ (ms)")
    axL.set_ylabel("Accuracy (%)")
    axL.set_xlim(0, 2.1)
    axL.set_ylim(0, 100)
    axL.grid(True, alpha=0.3)
    axL.legend(loc="lower right", fontsize=9)
    axL.set_title("Parity at train $\\Delta t{=}1.0$, 500 samples $\\times$ 10 epochs\n"
                  "on-diagonal: 85% vs 86%. off-diagonal: ours is less fragile.",
                  fontsize=10)
    for sp in ("top", "right"):
        axL.spines[sp].set_visible(False)

    # Right: snntorch calibration ladder
    for train_dt, colour, marker in [(0.1, "#1f77b4", "s"),
                                     (1.0, "#ff7f0e", "o")]:
        xs, ys = [], []
        for n, e in SIZES:
            acc = load_best("snntorch", train_dt, n, e)
            if acc is not None:
                xs.append(n)
                ys.append(acc)
        axR.plot(xs, ys, "-" + marker, color=colour,
                 label=f"train $\\Delta t{{=}}{train_dt}$ ms",
                 lw=2.0, ms=8)
        for x, y in zip(xs, ys):
            axR.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8, color=colour)
    axR.set_xscale("log")
    axR.set_xlabel("Training samples (log)")
    axR.set_ylabel("Best train-time accuracy (%)")
    axR.set_xlim(30, 30000)
    axR.set_ylim(20, 100)
    axR.grid(True, alpha=0.3, which="both")
    axR.legend(loc="lower right", fontsize=9)
    axR.set_title("snntorch calibration ladder\n"
                  "both dts scale cleanly; dt=1.0 leads dt=0.1 by ~3.5 pp at 15k.",
                  fontsize=10)
    for sp in ("top", "right"):
        axR.spines[sp].set_visible(False)

    fig.tight_layout()
    out = FIG_DIR / "parity_and_calibration.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ── Figure 2: 2x2 raster grid ───────────────────────────────────────────
def _run_forward(model_name: str, dt_infer: float, sample_idx: int = 0):
    """Load checkpoint, forward-pass one MNIST sample, return hid spikes."""
    M.T_ms = 200.0
    patch_dt(dt_infer)
    M.HIDDEN_SIZES = [1024]
    M.N_HID = 1024
    M.N_INH = 256
    M.N_IN = 784
    MDL.N_IN = 784
    MDL.N_HID = 1024
    MDL.N_INH = 256
    MDL.HIDDEN_SIZES = [1024]

    weights_path = ART / "dt1.0" / f"{model_name}.500.10" / "weights.pth"
    net = build_net(
        model_name, w_in=None, w_in_sparsity=0.0,
        ei_strength=0.0, ei_ratio=2.0, sparsity=0.0,
        device=torch.device("cpu"), randomize_init=False,
        kaiming_init=True, dales_law=False, hidden_sizes=[1024])

    state = torch.load(weights_path, map_location="cpu")
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True

    _, X_te, _, y_te = load_dataset("mnist", max_samples=200, split=True)
    x = torch.from_numpy(X_te[sample_idx:sample_idx + 1])
    y = int(y_te[sample_idx])

    gen = torch.Generator().manual_seed(7)
    spk = encode_batch(x, dt_infer, False, generator=gen)

    with torch.no_grad():
        logits = net(input_spikes=spk)
    pred = int(logits.argmax(1).item())

    hid = net.spike_record["hid"].numpy()
    return hid, y, pred


def make_rasters() -> Path:
    cells = [
        ("snntorch",         1.0, "snntorch (ours) — train $\\Delta t{=}1.0$, infer $\\Delta t{=}1.0$"),
        ("snntorch-library", 1.0, "snntorch-library (snn.Leaky) — infer $\\Delta t{=}1.0$"),
        ("snntorch",         0.1, "snntorch (ours) — infer $\\Delta t{=}0.1$ (off-diagonal)"),
        ("snntorch-library", 0.1, "snntorch-library — infer $\\Delta t{=}0.1$ (off-diagonal)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.2), sharey=True)

    for ax, (model, dt, title) in zip(axes.flat, cells):
        hid, y, pred = _run_forward(model, dt, sample_idx=0)
        T, N = hid.shape
        ts, ns = np.where(hid > 0)
        ax.scatter(ts * dt, ns, s=0.5, c="black", marker=".", alpha=0.7)
        ax.set_xlim(0, T * dt)
        ax.set_ylim(0, N)
        ax.set_title(title, fontsize=9)
        rate = hid.sum() / (N * T * dt / 1000.0)
        active_frac = (hid.sum(axis=0) > 0).mean()
        verdict = "✓" if pred == y else "✗"
        ax.text(0.02, 0.97,
                f"label={y}, pred={pred} {verdict}\n"
                f"rate={rate:.1f} Hz, active={active_frac*100:.0f}%",
                transform=ax.transAxes, fontsize=8,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="#ccc", alpha=0.85))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("Time (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Neuron index")

    fig.suptitle("Hidden-layer rasters: snntorch vs snntorch-library, "
                 "matched 500$\\times$10 checkpoints at train $\\Delta t{=}1.0$",
                 fontsize=11, y=0.995)
    fig.tight_layout()
    out = FIG_DIR / "rasters.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ── numbers.json: auditable trail of every stat cited ───────────────────
def make_numbers_json() -> Path:
    parity = {}
    for model in ("snntorch", "snntorch-library"):
        sweep = load_sweep(model, 1.0, 500, 10)
        parity[model] = {str(r["dt"]): round(r["acc"], 2) for r in sweep}

    calibration = {}
    for train_dt in (0.1, 1.0):
        row = {}
        for n, e in SIZES:
            acc = load_best("snntorch", train_dt, n, e)
            if acc is not None:
                row[str(n)] = round(acc, 2)
        calibration[f"train_dt_{train_dt}"] = row

    rasters = {}
    for model in ("snntorch", "snntorch-library"):
        for dt in (1.0, 0.1):
            hid, y, pred = _run_forward(model, dt, sample_idx=0)
            T, N = hid.shape
            rasters[f"{model}_dt{dt}"] = {
                "label": y,
                "pred": pred,
                "rate_hz": round(float(hid.sum() / (N * T * dt / 1000.0)), 2),
                "active_frac": round(float((hid.sum(axis=0) > 0).mean()), 4),
            }

    numbers = {
        "parity_sweep_train_dt_1.0_size_500x10": parity,
        "calibration_ladder_snntorch": calibration,
        "rasters_one_sample_sample_idx_0": rasters,
    }
    out = FIG_DIR / "numbers.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(numbers, indent=2) + "\n")
    return out


# ── Entry point ─────────────────────────────────────────────────────────
def main() -> None:
    print(f"→ {make_parity_and_calibration()}")
    print(f"→ {make_rasters()}")
    print(f"→ {make_numbers_json()}")


if __name__ == "__main__":
    main()
