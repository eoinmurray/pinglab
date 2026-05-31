"""Notebook runner for entry 041 — CUBA-no-PING baseline.

Standalone PyTorch runner. Companion to nb040: same architecture
*minus* the I-loop. Trains a current-based LIF E-only network with
the same mem-mean readout, same TBPTT recipe, same MNIST subset, same
hyperparameters. The point is to test whether nb040's sub-Hz firing
rates come from the I-loop (clamping the rate) or from the mem-mean
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
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from pinglab import theme  # noqa: E402

SLUG = "nb041"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG

# ── Architecture (same E params as nb040; no I population) ──────
N_E: int = 1024
N_IN: int = 784
N_CLASSES: int = 10

T_MS: float = 200.0
DT: float = 1.0
N_STEPS: int = int(T_MS / DT)

TAU_M_MS: float = 20.0
TAU_OUT_MS: float = 20.0  # output-LIF readout time constant (matches nb040)
V_REST: float = 0.0
V_TH: float = 1.0
V_RESET: float = 0.0
R_M: float = 1.0

W_IN_INIT_STD: float = 0.5
W_IN_SPARSITY: float = 0.95

SEED: int = 42

# ── Training (matches nb040 exactly) ─────────────────────────────
INPUT_RATE_HZ: float = 80.0

N_TRAIN: int = 10000
N_TEST: int = 1000
EPOCHS: int = 30
BATCH_SIZE: int = 64
LR: float = 5e-4
SURROGATE_SLOPE: float = 1.0
TBPTT_WINDOW: int = 10
GRAD_CLIP_VALUE: float = 1.0


# ── Surrogate gradient (same as nb040) ───────────────────────────
class SurrSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_minus_th: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(v_minus_th)
        ctx.slope = slope
        return (v_minus_th >= 0).float()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (v,) = ctx.saved_tensors
        s = ctx.slope
        denom = 1.0 + (s * np.pi * v).pow(2)
        return grad_out * s / denom, None


def spike(v: torch.Tensor, slope: float) -> torch.Tensor:
    return SurrSpike.apply(v - V_TH, slope)


class CubaNoPingNet(nn.Module):
    """LIF E-population only — no I-loop, no W_ei, no W_ie. Otherwise
    identical to the CubaPingNet in nb040."""

    def __init__(self, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w_in_dense = torch.randn(N_IN, N_E, generator=g) * W_IN_INIT_STD
        mask = (torch.rand(N_IN, N_E, generator=g) > W_IN_SPARSITY).float()
        self.W_in = nn.Parameter((w_in_dense * mask).abs())
        w_out = torch.randn(N_E, N_CLASSES, generator=g) * 0.05
        self.W_out = nn.Parameter(w_out)
        self.b_out = nn.Parameter(torch.zeros(N_CLASSES))

    def forward(self, input_spikes: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, T, _ = input_spikes.shape
        dev = input_spikes.device
        V_E = torch.zeros(B, N_E, device=dev)
        V_out = torch.zeros(B, N_CLASSES, device=dev)
        readout = torch.zeros(B, N_CLASSES, device=dev)
        e_spike_total = torch.zeros((), device=dev)
        alpha = DT / TAU_M_MS
        alpha_out = DT / TAU_OUT_MS
        for t in range(T):
            if self.training and t > 0 and t % TBPTT_WINDOW == 0:
                V_E = V_E.detach()
                V_out = V_out.detach()
            x = input_spikes[:, t, :]
            I_E = x @ self.W_in
            V_E = V_E + alpha * (-(V_E - V_REST) + R_M * I_E)
            s_E = spike(V_E, SURROGATE_SLOPE)
            V_E = V_E - s_E.detach() * (V_TH - V_RESET)
            # Output-LIF mem-mean readout (same as nb040). Hidden
            # spikes are now mandatory for the readout to carry signal.
            V_out = V_out + alpha_out * (-V_out + s_E @ self.W_out)
            readout = readout + V_out
            e_spike_total = e_spike_total + s_E.sum()
        logits = readout / T + self.b_out
        e_rate = float(e_spike_total.detach()) / (B * N_E * T * DT / 1000.0)
        return logits, {"rate_e_hz": e_rate}


# ── Data + training (mirrors nb040) ──────────────────────────────
def load_mnist() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from torchvision import datasets, transforms
    cache = REPO / ".cache" / "mnist"
    cache.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(cache, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(cache, train=False, download=True, transform=transform)
    rng = np.random.default_rng(SEED)
    train_idx = rng.choice(len(train_ds), size=N_TRAIN, replace=False)
    test_idx = rng.choice(len(test_ds), size=N_TEST, replace=False)
    X_tr = torch.stack([train_ds[int(i)][0].view(-1) for i in train_idx])
    y_tr = torch.tensor([train_ds[int(i)][1] for i in train_idx])
    X_te = torch.stack([test_ds[int(i)][0].view(-1) for i in test_idx])
    y_te = torch.tensor([test_ds[int(i)][1] for i in test_idx])
    return X_tr, y_tr, X_te, y_te


def encode_batch(X: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    B = X.shape[0]
    p = X.unsqueeze(1) * (INPUT_RATE_HZ * DT / 1000.0)
    p = p.expand(B, N_STEPS, N_IN).contiguous()
    return torch.bernoulli(p, generator=gen)


def evaluate(net, X, y, device, gen) -> dict:
    net.eval()
    correct = total = 0
    e_rate_sum = 0.0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = X[i:i + BATCH_SIZE].to(device)
            yb = y[i:i + BATCH_SIZE].to(device)
            spk = encode_batch(xb, gen).to(device)
            logits, info = net(spk)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
            e_rate_sum += info["rate_e_hz"]
            n_batches += 1
    return {"acc": 100.0 * correct / total,
            "rate_e_hz": e_rate_sum / max(n_batches, 1)}


def train(X_tr, y_tr, X_te, y_te, device) -> dict:
    print("\n[nb041] training CUBA-no-PING")
    net = CubaNoPingNet(seed=SEED).to(device)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  trainable parameters: {n_params:,}")
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    gen = torch.Generator(device="cpu").manual_seed(SEED + 7)
    epoch_rows: list[dict] = []
    for ep in range(1, EPOCHS + 1):
        net.train()
        perm = torch.randperm(len(X_tr))
        loss_sum = 0.0; n = 0
        t_ep = time.monotonic()
        for i in range(0, len(X_tr), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = X_tr[idx].to(device)
            yb = y_tr[idx].to(device)
            spk = encode_batch(xb, gen).to(device)
            logits, _ = net(spk)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), GRAD_CLIP_VALUE)
            opt.step()
            loss_sum += float(loss.item()) * yb.size(0)
            n += yb.size(0)
        train_loss = loss_sum / max(n, 1)
        ev = evaluate(net, X_te, y_te, device, gen)
        dt_ep = time.monotonic() - t_ep
        print(
            f"  ep {ep:2d}/{EPOCHS}  loss={train_loss:6.3f}  "
            f"acc={ev['acc']:5.2f}%  E={ev['rate_e_hz']:6.2f} Hz  [{dt_ep:5.1f}s]"
        )
        epoch_rows.append({"epoch": ep, "train_loss": train_loss, **ev, "duration_s": dt_ep})

    # Capture a test trial's E raster.
    net.eval()
    with torch.no_grad():
        scan_B = min(16, len(X_te))
        xb = X_te[:scan_B].to(device)
        spk = encode_batch(xb, gen).to(device)
        V_E = torch.zeros(scan_B, N_E, device=device)
        e_per = torch.zeros(scan_B, device=device)
        alpha = DT / TAU_M_MS
        for t in range(N_STEPS):
            x = spk[:, t, :]
            V_E = V_E + alpha * (-(V_E - V_REST) + R_M * (x @ net.W_in))
            s_E = (V_E >= V_TH).float()
            V_E = torch.where(s_E.bool(), torch.full_like(V_E, V_RESET), V_E)
            e_per = e_per + s_E.sum(dim=1)
        sample_idx = int(e_per.argmax().item())
        print(f"  raster sample: idx={sample_idx} digit={int(y_te[sample_idx].item())}")

        spk_one = encode_batch(X_te[sample_idx:sample_idx + 1].to(device), gen).to(device)
        V_E = torch.zeros(1, N_E, device=device)
        spk_E_log = torch.zeros(N_STEPS, N_E)
        for t in range(N_STEPS):
            x = spk_one[:, t, :]
            V_E = V_E + alpha * (-(V_E - V_REST) + R_M * (x @ net.W_in))
            s_E = (V_E >= V_TH).float()
            V_E = torch.where(s_E.bool(), torch.full_like(V_E, V_RESET), V_E)
            spk_E_log[t] = s_E[0].cpu()

    return {
        "epochs": epoch_rows,
        "final": epoch_rows[-1],
        "raster": {
            "spk_E": spk_E_log.numpy(),
            "label": int(y_te[sample_idx].item()),
        },
    }


# ── Plotting ─────────────────────────────────────────────────────
def _stamp(fig) -> None:
    fig.text(
        0.995, 0.005, f"nb041-{int(time.time())}",
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


def plot_learning_curves(run: dict, out_path: Path) -> None:
    theme.apply()
    fig, (ax_loss, ax_acc, ax_rate) = plt.subplots(1, 3, figsize=(13.0, 4.5), dpi=150)
    eps = [e["epoch"] for e in run["epochs"]]
    ax_loss.plot(eps, [e["train_loss"] for e in run["epochs"]],
                 marker="o", color=theme.INK_BLACK)
    ax_acc.plot(eps, [e["acc"] for e in run["epochs"]],
                marker="o", color=theme.INK_BLACK)
    ax_rate.plot(eps, [e["rate_e_hz"] for e in run["epochs"]],
                 marker="o", color=theme.INK_BLACK)
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


def plot_trained_raster(run: dict, out_path: Path) -> None:
    theme.apply()
    sample = run["raster"]
    spk_e = sample["spk_E"]
    t_ms = np.arange(spk_e.shape[0]) * DT
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    e_idx, e_t = np.where(spk_e.T)
    ax.scatter(t_ms[e_t], e_idx, s=1.5, c=theme.INK_BLACK, marker="|", linewidths=0.6)
    ax.set_ylabel("E neuron")
    ax.set_ylim(0, N_E)
    ax.set_xlim(0, T_MS)
    ax.set_xlabel("time (ms)")
    final = run["final"]
    ax.set_title(
        f"Trained CUBA-no-PING — digit {sample['label']}, "
        f"acc={final['acc']:.1f}%, E={final['rate_e_hz']:.2f} Hz",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison(this_run: dict, ping_numbers: dict, out_path: Path) -> None:
    """Side-by-side comparison of accuracy and E rate vs epoch — this
    notebook's no-PING run against nb040's PING run (loaded from
    nb040/numbers.json)."""
    theme.apply()
    fig, (ax_acc, ax_rate) = plt.subplots(1, 2, figsize=(11.0, 4.5), dpi=150)
    eps = [e["epoch"] for e in this_run["epochs"]]
    ax_acc.plot(eps, [e["acc"] for e in this_run["epochs"]],
                marker="o", color=theme.INK_BLACK, label="CUBA-no-PING")
    ax_rate.plot(eps, [e["rate_e_hz"] for e in this_run["epochs"]],
                 marker="o", color=theme.INK_BLACK, label="CUBA-no-PING")
    if ping_numbers is not None:
        ping_eps = ping_numbers.get("epochs", [])
        if ping_eps:
            ax_acc.plot(
                [e["epoch"] for e in ping_eps],
                [e["acc"] for e in ping_eps],
                marker="s", color=theme.DEEP_RED, label="CUBA-PING (nb040)",
            )
            ax_rate.plot(
                [e["epoch"] for e in ping_eps],
                [e["rate_e_hz"] for e in ping_eps],
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
    wipe_dir = "--no-wipe-dir" not in sys.argv
    if wipe_dir and FIGURES.exists():
        print(f"[wipe] {FIGURES.relative_to(REPO)}")
        shutil.rmtree(FIGURES)
    FIGURES.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    t_start = time.monotonic()
    print(f"[nb041] CUBA-no-PING (E only, mem-mean readout), device={device}")
    print(f"  N_E={N_E}, N_IN={N_IN}, T={T_MS}ms, dt={DT}ms, batch={BATCH_SIZE}")

    X_tr, y_tr, X_te, y_te = load_mnist()
    print(f"  data: {len(X_tr)} train, {len(X_te)} test")

    run = train(X_tr, y_tr, X_te, y_te, device)
    plot_learning_curves(run, FIGURES / "learning_curves.png")
    print(f"  wrote {FIGURES / 'learning_curves.png'}")
    plot_trained_raster(run, FIGURES / "trained_raster.png")
    print(f"  wrote {FIGURES / 'trained_raster.png'}")

    # Comparison with nb040 CUBA-PING.
    ping_path = REPO / "src/docs/public/figures/notebooks/nb040/numbers.json"
    ping_numbers = json.loads(ping_path.read_text()) if ping_path.exists() else None
    if ping_numbers is not None:
        plot_comparison(run, ping_numbers, FIGURES / "comparison.png")
        print(f"  wrote {FIGURES / 'comparison.png'}")

    duration_s = time.monotonic() - t_start
    summary = {
        "slug": SLUG,
        "duration_s": round(duration_s, 1),
        "device": str(device),
        "config": {
            "n_e": N_E, "n_in": N_IN, "n_classes": N_CLASSES,
            "t_ms": T_MS, "dt": DT, "tau_m_ms": TAU_M_MS,
            "v_th": V_TH, "v_reset": V_RESET, "v_rest": V_REST, "r_m": R_M,
            "input_rate_hz": INPUT_RATE_HZ,
            "n_train": N_TRAIN, "n_test": N_TEST,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "tbptt_window": TBPTT_WINDOW,
            "grad_clip_value": GRAD_CLIP_VALUE,
            "surrogate_slope": SURROGATE_SLOPE, "seed": SEED,
        },
        "final_acc": run["final"]["acc"],
        "final_rate_e_hz": run["final"]["rate_e_hz"],
        "epochs": [
            {k: v for k, v in e.items() if k != "raster"}
            for e in run["epochs"]
        ],
        "success_criteria": [
            {
                "label": "training reaches above chance",
                "passed": run["final"]["acc"] > 15.0,
                "detail": f"{run['final']['acc']:.2f}%",
            },
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {duration_s:.1f}s")


if __name__ == "__main__":
    main()
