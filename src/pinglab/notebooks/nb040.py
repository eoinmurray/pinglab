"""Notebook runner for entry 040 — CUBA-PING + CUBA-no-PING ablation.

Trains the CUBA-PING network (with I-loop) and its no-PING control
(E-only) under matched recipe via the oscilloscope's `cuba-ping` and
`cuba-noping` models. Produces:
1. Dynamics raster: untrained CUBA-PING under uniform Poisson input.
2. Learning curves for each model.
3. Trained-network rasters for each model.
4. Side-by-side comparison plot.

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

# ── Architecture (shared by both arms) ────────────────────────────
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

# Two-arm experiment: PING with I-loop + no-PING E-only control.
ARMS = ("ping", "noping")
MODEL_FOR_ARM = {"ping": "cuba-ping", "noping": "cuba-noping"}
LABEL_FOR_ARM = {"ping": "CUBA-PING", "noping": "CUBA-no-PING"}


def build_oscilloscope_args(arm: str, tier: str, out_dir: Path) -> list[str]:
    args = [
        "train",
        "--model", MODEL_FOR_ARM[arm],
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
    if arm == "ping":
        args += [
            "--w-ei", str(W_EI_MEAN), str(W_EI_STD),
            "--w-ie", str(W_IE_MEAN), str(W_IE_STD),
        ]
    return args


def _build_net_for_arm(arm: str):
    """Construct the model the train CLI would build, identically seeded."""
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
    kw = dict(
        w_in=(W_IN_MEAN, W_IN_STD),
        w_in_sparsity=W_IN_SPARSITY,
        hidden_sizes=[N_E],
    )
    if arm == "ping":
        kw["w_ei"] = (W_EI_MEAN, W_EI_STD)
        kw["w_ie"] = (W_IE_MEAN, W_IE_STD)
    return build_net(MODEL_FOR_ARM[arm], **kw)


# ── Dynamics raster (PING only, untrained) ───────────────────────
def simulate_dynamics_ping(input_rate_hz: float) -> dict[str, np.ndarray]:
    net = _build_net_for_arm("ping")
    net.eval()
    gen = torch.Generator().manual_seed(SEED + 1)
    p = input_rate_hz * DT / 1000.0
    spk_in = (torch.rand(N_STEPS, 1, N_IN, generator=gen) < p).float()
    net.recording = True
    with torch.no_grad():
        net(input_spikes=spk_in)
    rec = net.spike_record
    return {
        "spk_E": rec["hid"].cpu().numpy(),
        "spk_I": rec["inh"].cpu().numpy(),
    }


# ── Plotting ─────────────────────────────────────────────────────
def _stamp(fig) -> None:
    fig.text(
        0.995, 0.005, f"{SLUG}-{int(time.time())}",
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


def plot_ping_raster(sample: dict, title: str, out_path: Path) -> None:
    """Two-panel raster (E above, I below)."""
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


def plot_noping_raster(spk_e: np.ndarray, title: str, out_path: Path) -> None:
    """Single-panel E raster for the no-PING control."""
    theme.apply()
    t_ms = np.arange(spk_e.shape[0]) * DT
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    e_idx, e_t = np.where(spk_e.T)
    ax.scatter(t_ms[e_t], e_idx, s=1.5, c=theme.INK_BLACK, marker="|", linewidths=0.6)
    ax.set_ylabel("E neuron")
    ax.set_ylim(0, spk_e.shape[1])
    ax.set_xlim(0, T_MS)
    ax.set_xlabel("time (ms)")
    ax.set_title(title, fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    _stamp(fig)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_learning_curves(metrics: dict, has_inh: bool, out_path: Path) -> None:
    theme.apply()
    epochs = metrics["epochs"]
    eps = [e["ep"] for e in epochs]
    loss = [e["loss"] for e in epochs]
    acc = [e["acc"] for e in epochs]
    re = [e["test_rate_e"] for e in epochs]
    fig, (ax_loss, ax_acc, ax_rate) = plt.subplots(1, 3, figsize=(13.0, 4.5), dpi=150)
    ax_loss.plot(eps, loss, marker="o", color=theme.INK_BLACK)
    ax_acc.plot(eps, acc, marker="o", color=theme.INK_BLACK)
    ax_rate.plot(eps, re, marker="o", color=theme.INK_BLACK, label="E")
    if has_inh:
        ri = [e["test_rate_i"] for e in epochs]
        ax_rate.plot(eps, ri, marker="s", color=theme.DEEP_RED, label="I")
    rate_title = "Hidden E / I rate" if has_inh else "Hidden E rate"
    for ax, ylab, title in (
        (ax_loss, "Cross-entropy", "Training loss"),
        (ax_acc, "Test accuracy (%)", "Test accuracy"),
        (ax_rate, "Mean firing rate (Hz)", rate_title),
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=theme.SIZE_TITLE)
        ax.grid(True, alpha=0.3)
    if has_inh:
        ax_rate.legend(fontsize=theme.SIZE_CAPTION, frameon=False)
    ax_acc.set_ylim(0, 100)
    fig.tight_layout()
    _stamp(fig)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _spk_for_test_sample(arm: str, weights_path: Path) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Replay a single MNIST test sample on the trained net of this arm."""
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

    net = _build_net_for_arm(arm)
    state = torch.load(weights_path, map_location="cpu")
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True
    with torch.no_grad():
        net(input_spikes=spk_in)
    rec = net.spike_record
    spk_e = rec["hid"].cpu().numpy()
    spk_i = rec["inh"].cpu().numpy() if "inh" in rec else None
    return spk_e, spk_i, int(label)


def plot_comparison(ping_metrics: dict, noping_metrics: dict, out_path: Path) -> None:
    theme.apply()
    fig, (ax_acc, ax_rate) = plt.subplots(1, 2, figsize=(11.0, 4.5), dpi=150)
    for name, m, color, marker in (
        ("CUBA-PING", ping_metrics, theme.INK_BLACK, "o"),
        ("CUBA-no-PING", noping_metrics, theme.DEEP_RED, "s"),
    ):
        eps = [e["ep"] for e in m["epochs"]]
        ax_acc.plot(eps, [e["acc"] for e in m["epochs"]],
                    marker=marker, color=color, label=name)
        ax_rate.plot(eps, [e["test_rate_e"] for e in m["epochs"]],
                     marker=marker, color=color, label=name)
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
    print(f"[{SLUG}] CUBA-PING + CUBA-no-PING via oscilloscope, tier={tier}")

    # ── 1: untrained CUBA-PING dynamics
    print(
        f"\n[{SLUG}:dynamics] uniform Poisson input @ "
        f"{DYNAMICS_INPUT_RATE_HZ} Hz/channel"
    )
    dyn = simulate_dynamics_ping(DYNAMICS_INPUT_RATE_HZ)
    e_rate = float(dyn["spk_E"].sum()) / (N_E * T_MS / 1000.0)
    i_rate = float(dyn["spk_I"].sum()) / ((N_E // 4) * T_MS / 1000.0)
    print(f"  E rate = {e_rate:.2f} Hz, I rate = {i_rate:.2f} Hz")
    plot_ping_raster(
        dyn,
        f"CUBA-PING dynamics (untrained) — input {DYNAMICS_INPUT_RATE_HZ:g} Hz "
        f"(E = {e_rate:.1f} Hz, I = {i_rate:.1f} Hz)",
        FIGURES / "dynamics_raster.png",
    )
    print(f"  wrote {FIGURES / 'dynamics_raster.png'}")

    # ── 2: train both arms in one batch (parallel on Modal, sequential local)
    print(f"\n[{SLUG}:train] dispatching {' + '.join(ARMS)} runs")
    dispatcher = BatchDispatcher(modal_gpu, REPO, OSCILLOSCOPE)
    train_dirs: dict[str, Path] = {}
    for arm in ARMS:
        out = ARTIFACTS / f"train__{arm}"
        train_dirs[arm] = out
        print(
            f"  → {arm} ({MODEL_FOR_ARM[arm]}) → "
            f"{out.relative_to(REPO)}"
            + (f"  [modal:{modal_gpu}]" if modal_gpu else "")
        )
        dispatcher.submit(build_oscilloscope_args(arm, tier, out), out)
    dispatcher.drain()

    # ── 3: per-arm learning curves + trained rasters
    metrics: dict[str, dict] = {}
    for arm in ARMS:
        m = json.loads((train_dirs[arm] / "metrics.json").read_text())
        metrics[arm] = m
        plot_learning_curves(
            m, has_inh=(arm == "ping"),
            out_path=FIGURES / f"learning_curves__{arm}.png",
        )
        print(f"  wrote {FIGURES / f'learning_curves__{arm}.png'}")

        spk_e, spk_i, label = _spk_for_test_sample(arm, train_dirs[arm] / "weights.pth")
        final = m["epochs"][-1]
        if arm == "ping":
            title = (
                f"Trained CUBA-PING — digit {label}, "
                f"acc={final['acc']:.1f}%, "
                f"E={final['test_rate_e']:.1f} Hz, "
                f"I={final['test_rate_i']:.1f} Hz"
            )
            plot_ping_raster(
                {"spk_E": spk_e, "spk_I": spk_i},
                title,
                FIGURES / f"trained_raster__{arm}.png",
            )
        else:
            title = (
                f"Trained CUBA-no-PING — digit {label}, "
                f"acc={final['acc']:.1f}%, "
                f"E={final['test_rate_e']:.1f} Hz"
            )
            plot_noping_raster(spk_e, title, FIGURES / f"trained_raster__{arm}.png")
        print(f"  wrote {FIGURES / f'trained_raster__{arm}.png'}")

    # ── 4: side-by-side comparison
    plot_comparison(metrics["ping"], metrics["noping"], FIGURES / "comparison.png")
    print(f"  wrote {FIGURES / 'comparison.png'}")

    # ── 5: summary
    ping_final = metrics["ping"]["epochs"][-1]
    nop_final = metrics["noping"]["epochs"][-1]
    duration_s = time.monotonic() - t_start
    summary = {
        "slug": SLUG,
        "tier": tier,
        "duration_s": round(duration_s, 1),
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
        "ping": {
            "model": MODEL_FOR_ARM["ping"],
            "final_acc": ping_final["acc"],
            "final_rate_e_hz": ping_final["test_rate_e"],
            "final_rate_i_hz": ping_final["test_rate_i"],
            "epochs": metrics["ping"]["epochs"],
        },
        "noping": {
            "model": MODEL_FOR_ARM["noping"],
            "final_acc": nop_final["acc"],
            "final_rate_e_hz": nop_final["test_rate_e"],
            "epochs": metrics["noping"]["epochs"],
        },
        # Top-level shortcuts used by the index page and NotebookHeader.
        "final_acc": ping_final["acc"],
        "final_rate_e_hz": ping_final["test_rate_e"],
        "final_rate_i_hz": ping_final["test_rate_i"],
        "success_criteria": [
            {
                "label": "dynamics raster rendered",
                "passed": (FIGURES / "dynamics_raster.png").exists(),
                "detail": f"E={e_rate:.1f} Hz, I={i_rate:.1f} Hz",
            },
            {
                "label": "CUBA-PING reaches above chance",
                "passed": ping_final["acc"] > 15.0,
                "detail": f"{ping_final['acc']:.2f}%",
            },
            {
                "label": "CUBA-no-PING reaches above chance",
                "passed": nop_final["acc"] > 15.0,
                "detail": f"{nop_final['acc']:.2f}%",
            },
            {
                "label": "no-PING E rate >> PING E rate",
                "passed": nop_final["test_rate_e"] > 5 * max(ping_final["test_rate_e"], 0.1),
                "detail": (
                    f"PING={ping_final['test_rate_e']:.1f} Hz, "
                    f"no-PING={nop_final['test_rate_e']:.1f} Hz"
                ),
            },
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {duration_s:.1f}s")


if __name__ == "__main__":
    main()
