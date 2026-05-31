"""Notebook runner for entry 041 — CUBA-no-PING control via oscilloscope.

Companion to nb040: same architecture *minus* the I-loop, dispatched
via the oscilloscope's `cuba-noping` model. Tests whether nb040's
sub-Hz firing comes from the I-loop (rate clamp) or the mem-mean
readout (which doesn't require spikes to classify).

Notebook entry: src/docs/src/pages/notebooks/nb041.mdx
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

SLUG = "nb041"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "cli" / "__main__.py"

# ── Architecture (matches nb040 exactly; no I population) ─────────
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

SEED: int = 42

# ── Training recipe (tier-scaled; mirror nb040) ──────────────────
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

MODEL: str = "cuba-noping"


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


def _build_cubanet():
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
        hidden_sizes=[N_E],
    )


def _stamp(fig) -> None:
    fig.text(
        0.995, 0.005, f"{SLUG}-{int(time.time())}",
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


def plot_learning_curves(metrics: dict, out_path: Path) -> None:
    theme.apply()
    epochs = metrics["epochs"]
    eps = [e["ep"] for e in epochs]
    loss = [e["loss"] for e in epochs]
    acc = [e["acc"] for e in epochs]
    re = [e["test_rate_e"] for e in epochs]
    fig, (ax_loss, ax_acc, ax_rate) = plt.subplots(1, 3, figsize=(13.0, 4.5), dpi=150)
    ax_loss.plot(eps, loss, marker="o", color=theme.INK_BLACK)
    ax_acc.plot(eps, acc, marker="o", color=theme.INK_BLACK)
    ax_rate.plot(eps, re, marker="o", color=theme.INK_BLACK)
    for ax, ylab, title in (
        (ax_loss, "Cross-entropy", "Training loss"),
        (ax_acc, "Test accuracy (%)", "Test accuracy"),
        (ax_rate, "Mean E firing rate (Hz)", "Hidden E rate"),
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=theme.SIZE_TITLE)
        ax.grid(True, alpha=0.3)
    ax_acc.set_ylim(0, 100)
    fig.tight_layout()
    _stamp(fig)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trained_raster(weights_path: Path, metrics: dict, out_path: Path) -> None:
    from torchvision import datasets, transforms
    cache = REPO / ".cache" / "mnist"
    test_ds = datasets.MNIST(cache, train=False, download=True,
                             transform=transforms.ToTensor())
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
    spk_e = rec["hid"].cpu().numpy()

    theme.apply()
    t_ms = np.arange(spk_e.shape[0]) * DT
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    e_idx, e_t = np.where(spk_e.T)
    ax.scatter(t_ms[e_t], e_idx, s=1.5, c=theme.INK_BLACK, marker="|", linewidths=0.6)
    ax.set_ylabel("E neuron")
    ax.set_ylim(0, N_E)
    ax.set_xlim(0, T_MS)
    ax.set_xlabel("time (ms)")
    final = metrics["epochs"][-1]
    ax.set_title(
        f"Trained CUBA-no-PING — digit {label}, "
        f"acc={final['acc']:.1f}%, E={final['test_rate_e']:.2f} Hz",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison(this_metrics: dict, ping_numbers: dict, out_path: Path) -> None:
    theme.apply()
    fig, (ax_acc, ax_rate) = plt.subplots(1, 2, figsize=(11.0, 4.5), dpi=150)
    this_eps = this_metrics["epochs"]
    ax_acc.plot(
        [e["ep"] for e in this_eps], [e["acc"] for e in this_eps],
        marker="o", color=theme.INK_BLACK, label="CUBA-no-PING",
    )
    ax_rate.plot(
        [e["ep"] for e in this_eps], [e["test_rate_e"] for e in this_eps],
        marker="o", color=theme.INK_BLACK, label="CUBA-no-PING",
    )
    if ping_numbers is not None:
        ping_eps = ping_numbers.get("epochs", [])
        if ping_eps:
            ax_acc.plot(
                [e["ep"] for e in ping_eps],
                [e["acc"] for e in ping_eps],
                marker="s", color=theme.DEEP_RED, label="CUBA-PING (nb040)",
            )
            ax_rate.plot(
                [e["ep"] for e in ping_eps],
                [e["test_rate_e"] for e in ping_eps],
                marker="s", color=theme.DEEP_RED, label="CUBA-PING (nb040)",
            )
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Test accuracy (%)")
    ax_acc.set_title("Test accuracy", fontsize=theme.SIZE_TITLE)
    ax_acc.set_ylim(0, 100)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(fontsize=theme.SIZE_CAPTION, frameon=False)
    ax_rate.set_xlabel("Epoch")
    ax_rate.set_ylabel("Mean hidden-E rate (Hz)")
    ax_rate.set_title("Hidden E firing rate", fontsize=theme.SIZE_TITLE)
    ax_rate.grid(True, alpha=0.3)
    ax_rate.legend(fontsize=theme.SIZE_CAPTION, frameon=False)
    fig.tight_layout()
    _stamp(fig)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    print(f"[{SLUG}] CUBA-no-PING via oscilloscope, tier={tier}")

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

    ping_path = REPO / "src/docs/public/figures/notebooks/nb040/numbers.json"
    ping_numbers = json.loads(ping_path.read_text()) if ping_path.exists() else None
    if ping_numbers is not None:
        plot_comparison(metrics, ping_numbers, FIGURES / "comparison.png")
        print(f"  wrote {FIGURES / 'comparison.png'}")

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
            "max_samples": TIER_CONFIG[tier]["max_samples"],
            "epochs": TIER_CONFIG[tier]["epochs"],
            "batch_size": BATCH_SIZE, "lr": LR, "seed": SEED,
        },
        "final_acc": final["acc"],
        "final_rate_e_hz": final["test_rate_e"],
        "epochs": metrics["epochs"],
        "success_criteria": [
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
