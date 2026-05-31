"""Notebook runner for entry 040 — CUBA-PING via the oscilloscope.

Thin wrapper around the oscilloscope's `cuba-ping` model:
1. Dynamics raster: build CubaPingNet, forward at uniform Poisson input,
   render the E/I raster (no training).
2. Training: dispatch `oscilloscope train --model cuba-ping ...` with
   the recipe baked in below.
3. Plotting: learning curves from metrics.json, trained raster from
   weights.pth.

Notebook entry: src/docs/src/pages/notebooks/nb040.mdx
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from _tier import parse_tier  # noqa: E402
from pinglab import theme  # noqa: E402

SLUG = "nb040"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "cli" / "__main__.py"

# ── Architecture (matches nb040.mdx recipe) ───────────────────────
N_E: int = 1024
N_IN: int = 784
N_CLASSES: int = 10

T_MS: float = 200.0
DT: float = 1.0
N_STEPS: int = int(T_MS / DT)

TAU_M_MS: float = 20.0
TAU_OUT_MS: float = 20.0
V_TH: float = 1.0

W_IN_MEAN: float = 0.0
W_IN_STD: float = 0.5
W_IN_SPARSITY: float = 0.95
W_EI_MEAN: float = 1.0
W_EI_STD: float = 0.1
W_IE_MEAN: float = 1.0
W_IE_STD: float = 0.1

SEED: int = 42
DYNAMICS_INPUT_RATE_HZ: float = 80.0

# ── Training recipe (tier-scaled) ────────────────────────────────
TIER_CONFIG = {
    "extra small": dict(max_samples=200, epochs=2),
    "small": dict(max_samples=1000, epochs=5),
    "medium": dict(max_samples=5000, epochs=15),
    "large": dict(max_samples=10000, epochs=30),
    "extra large": dict(max_samples=20000, epochs=40),
}
DEFAULT_TIER = "medium"
INPUT_RATE_HZ: float = 80.0
LR: float = 2e-3
BATCH_SIZE: int = 64

MODEL: str = "cuba-ping"


def build_oscilloscope_args(tier: str, out_dir: Path) -> list[str]:
    return [
        "train",
        "--model", MODEL,
        "--dataset", "mnist",
        "--max-samples", str(TIER_CONFIG[tier]["max_samples"]),
        "--epochs", str(TIER_CONFIG[tier]["epochs"]),
        "--t-ms", str(T_MS),
        "--dt", str(DT),
        "--input-rate", str(INPUT_RATE_HZ),
        "--seed", str(SEED),
        "--n-hidden", str(N_E),
        "--w-in", str(W_IN_MEAN), str(W_IN_STD),
        "--w-ei", str(W_EI_MEAN), str(W_EI_STD),
        "--w-ie", str(W_IE_MEAN), str(W_IE_STD),
        "--w-in-sparsity", str(W_IN_SPARSITY),
        "--readout", "mem-mean",
        "--surrogate-slope", "1",
        "--tau-mem", str(TAU_M_MS),
        "--readout-tau-out", str(TAU_OUT_MS),
        "--lr", str(LR),
        "--batch-size", str(BATCH_SIZE),
        "--grad-clip", "1e6",
        "--no-dales-law",
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]


# ── Section 1: dynamics under uniform Poisson input ──────────────
def _build_cubanet():
    """Construct the same network the train CLI builds, identically seeded."""
    torch.manual_seed(SEED)
    import models as M
    M.HIDDEN_SIZES = [N_E]
    M.N_IN = N_IN
    M.N_OUT = N_CLASSES
    M.N_HID = N_E
    M.N_INH = N_E // 4
    M.T_steps = N_STEPS
    M.T_ms = T_MS
    M.dt = DT
    from config import build_net
    return build_net(
        MODEL,
        w_in=(W_IN_MEAN, W_IN_STD),
        w_in_sparsity=W_IN_SPARSITY,
        w_ei=(W_EI_MEAN, W_EI_STD),
        w_ie=(W_IE_MEAN, W_IE_STD),
        hidden_sizes=[N_E],
    )


def simulate_dynamics(input_rate_hz: float) -> dict[str, np.ndarray]:
    """One untrained forward pass at uniform Poisson input."""
    net = _build_cubanet()
    net.eval()
    gen = torch.Generator().manual_seed(SEED + 1)
    p = input_rate_hz * DT / 1000.0
    spk_in = (torch.rand(N_STEPS, 1, N_IN, generator=gen) < p).float()
    net.recording = True
    with torch.no_grad():
        net(input_spikes=spk_in)
    rec = net.spike_record
    spk_e = rec["hid"].cpu().numpy()  # (T, N_E)
    spk_i = rec["inh"].cpu().numpy()
    return {"spk_E": spk_e, "spk_I": spk_i}


def _stamp(fig) -> None:
    fig.text(
        0.995, 0.005, f"{SLUG}-{int(time.time())}",
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


def plot_dynamics_raster(sample: dict, title: str, out_path: Path) -> None:
    theme.apply()
    spk_e, spk_i = sample["spk_E"], sample["spk_I"]
    t_ms = np.arange(spk_e.shape[0]) * DT
    fig, (ax_e, ax_i) = plt.subplots(
        2, 1, figsize=(8.0, 4.5), sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )
    e_idx, e_t = np.where(spk_e.T)
    ax_e.scatter(t_ms[e_t], e_idx, s=1.0, c=theme.INK_BLACK, marker="|", linewidths=0.5)
    ax_e.set_ylabel("E neuron")
    ax_e.set_ylim(0, spk_e.shape[1])
    ax_e.set_xlim(0, T_MS)
    ax_e.set_title(title, fontsize=theme.SIZE_TITLE)
    i_idx, i_t = np.where(spk_i.T)
    ax_i.scatter(t_ms[i_t], i_idx, s=1.0, c=theme.DEEP_RED, marker="|", linewidths=0.5)
    ax_i.set_ylabel("I neuron")
    ax_i.set_ylim(0, spk_i.shape[1])
    ax_i.set_xlim(0, T_MS)
    ax_i.set_xlabel("time (ms)")
    fig.tight_layout()
    _stamp(fig)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_learning_curves(metrics: dict, out_path: Path) -> None:
    theme.apply()
    epochs = metrics["epochs"]
    eps = [e["ep"] for e in epochs]
    loss = [e["loss"] for e in epochs]
    acc = [e["acc"] for e in epochs]
    re = [e["test_rate_e"] for e in epochs]
    ri = [e["test_rate_i"] for e in epochs]
    fig, (ax_loss, ax_acc, ax_rate) = plt.subplots(1, 3, figsize=(13.0, 4.5), dpi=150)
    ax_loss.plot(eps, loss, marker="o", color=theme.INK_BLACK)
    ax_acc.plot(eps, acc, marker="o", color=theme.INK_BLACK)
    ax_rate.plot(eps, re, marker="o", color=theme.INK_BLACK, label="E")
    ax_rate.plot(eps, ri, marker="s", color=theme.DEEP_RED, label="I")
    for ax, ylab, title in (
        (ax_loss, "Cross-entropy", "Training loss"),
        (ax_acc, "Test accuracy (%)", "Test accuracy"),
        (ax_rate, "Mean firing rate (Hz)", "Hidden E / I rate"),
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=theme.SIZE_TITLE)
        ax.grid(True, alpha=0.3)
    ax_rate.legend(fontsize=theme.SIZE_CAPTION, frameon=False)
    ax_acc.set_ylim(0, 100)
    fig.tight_layout()
    _stamp(fig)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trained_raster(weights_path: Path, metrics: dict, out_path: Path) -> None:
    """Build a network, load trained weights, forward one MNIST test sample."""
    from torchvision import datasets, transforms
    cache = REPO / ".cache" / "mnist"
    test_ds = datasets.MNIST(cache, train=False, download=True,
                             transform=transforms.ToTensor())
    # Pick a sample (digit) and Poisson-encode it
    rng = np.random.default_rng(SEED + 2)
    idx = int(rng.integers(len(test_ds)))
    img, label = test_ds[idx]
    pixels = img.view(-1)
    p = pixels.unsqueeze(0).unsqueeze(0) * (INPUT_RATE_HZ * DT / 1000.0)
    p = p.expand(N_STEPS, 1, N_IN).contiguous()
    gen = torch.Generator().manual_seed(SEED + 3)
    spk_in = torch.bernoulli(p, generator=gen)

    net = _build_cubanet()
    state = torch.load(weights_path, map_location="cpu")
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True
    with torch.no_grad():
        net(input_spikes=spk_in)
    rec = net.spike_record
    sample = {
        "spk_E": rec["hid"].cpu().numpy(),
        "spk_I": rec["inh"].cpu().numpy(),
    }

    final = metrics["epochs"][-1]
    title = (
        f"Trained CUBA-PING — digit {label}, "
        f"acc={final['acc']:.1f}%, "
        f"E={final['test_rate_e']:.1f} Hz, "
        f"I={final['test_rate_i']:.1f} Hz"
    )
    plot_dynamics_raster(sample, title, out_path)


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
    print(f"[{SLUG}] CUBA-PING via oscilloscope, tier={tier}")

    # Section 1: dynamics (untrained).
    print(
        f"\n[{SLUG}:dynamics] uniform Poisson input @ "
        f"{DYNAMICS_INPUT_RATE_HZ} Hz/channel"
    )
    dyn = simulate_dynamics(DYNAMICS_INPUT_RATE_HZ)
    e_rate = float(dyn["spk_E"].sum()) / (N_E * T_MS / 1000.0)
    i_rate = float(dyn["spk_I"].sum()) / ((N_E // 4) * T_MS / 1000.0)
    print(f"  E rate = {e_rate:.2f} Hz, I rate = {i_rate:.2f} Hz")
    plot_dynamics_raster(
        dyn,
        f"CUBA-PING dynamics (untrained) — input {DYNAMICS_INPUT_RATE_HZ:g} Hz "
        f"(E = {e_rate:.1f} Hz, I = {i_rate:.1f} Hz)",
        FIGURES / "dynamics_raster.png",
    )
    print(f"  wrote {FIGURES / 'dynamics_raster.png'}")

    # Section 2: training via oscilloscope.
    train_dir = ARTIFACTS / "train"
    print(f"\n[{SLUG}:train] dispatching {MODEL} train run → "
          f"{train_dir.relative_to(REPO)}")
    dispatcher = BatchDispatcher(modal_gpu, REPO, OSCILLOSCOPE)
    dispatcher.submit(build_oscilloscope_args(tier, train_dir), train_dir)
    dispatcher.drain()

    metrics = json.loads((train_dir / "metrics.json").read_text())
    plot_learning_curves(metrics, FIGURES / "learning_curves.png")
    print(f"  wrote {FIGURES / 'learning_curves.png'}")
    plot_trained_raster(
        train_dir / "weights.pth", metrics, FIGURES / "trained_raster.png"
    )
    print(f"  wrote {FIGURES / 'trained_raster.png'}")

    final = metrics["epochs"][-1]
    duration_s = time.monotonic() - t_start
    summary = {
        "slug": SLUG,
        "tier": tier,
        "duration_s": round(duration_s, 1),
        "model": MODEL,
        "config": {
            "n_e": N_E, "n_in": N_IN, "n_classes": N_CLASSES,
            "t_ms": T_MS, "dt": DT, "tau_m_ms": TAU_M_MS,
            "tau_out_ms": TAU_OUT_MS, "v_th": V_TH,
            "input_rate_hz": INPUT_RATE_HZ,
            "dynamics_input_rate_hz": DYNAMICS_INPUT_RATE_HZ,
            "max_samples": TIER_CONFIG[tier]["max_samples"],
            "epochs": TIER_CONFIG[tier]["epochs"],
            "batch_size": BATCH_SIZE, "lr": LR, "seed": SEED,
        },
        "dynamics": {"rate_e_hz": e_rate, "rate_i_hz": i_rate},
        "final_acc": final["acc"],
        "final_rate_e_hz": final["test_rate_e"],
        "final_rate_i_hz": final["test_rate_i"],
        "epochs": metrics["epochs"],
        "success_criteria": [
            {
                "label": "dynamics raster rendered",
                "passed": (FIGURES / "dynamics_raster.png").exists(),
                "detail": f"E={e_rate:.1f} Hz, I={i_rate:.1f} Hz",
            },
            {
                "label": "training reaches above chance",
                "passed": final["acc"] > 15.0,
                "detail": f"{final['acc']:.2f}%",
            },
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {duration_s:.1f}s")


if __name__ == "__main__":
    main()
