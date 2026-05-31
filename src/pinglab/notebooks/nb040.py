"""Notebook runner for entry 040 — CUBA-PING with instant synapses:
dynamics, training, and why naive BPTT fails.

Standalone PyTorch runner. Three parts, produced by one main():

1. Build a current-based PING network from scratch — LIF E and I
   populations with random fixed feed-forward (W_in) and recurrent
   (W_ei, W_ie) weights and *instant* synapses. Run forward at uniform
   Poisson input and render the raster (no training).
2. Make W_in and W_out trainable. Train on Poisson-encoded MNIST with
   surrogate-gradient backprop through time, limited horizon = K = 10
   steps (TBPTT) and per-element gradient clip. Generate training
   curves and a trained-network raster.
3. The mdx writeup walks through why naive (full-horizon) BPTT
   diverges in float32 for this architecture; this runner just
   trains the working configuration so the figures match.

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
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from pinglab import theme  # noqa: E402

SLUG = "nb040"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG

# ── Architecture (shared by dynamics and training) ───────────────
N_E: int = 1024
N_I: int = 256
N_IN: int = 784
N_CLASSES: int = 10

T_MS: float = 200.0
DT: float = 1.0
N_STEPS: int = int(T_MS / DT)

TAU_M_MS: float = 20.0
TAU_OUT_MS: float = 20.0  # output-LIF time constant for the mem-mean readout
V_REST: float = 0.0
V_TH: float = 1.0
V_RESET: float = 0.0
R_M: float = 1.0

W_IN_INIT_STD: float = 0.5
W_EI_MEAN: float = 1.0
W_EI_STD: float = 0.1
W_IE_MEAN: float = 1.0   # halved from earlier runs to soften the
W_IE_STD: float = 0.1    # post-burst suppression (each I-spike's
                         # inhibitory kick is now ~half as large)
W_IN_SPARSITY: float = 0.95

SEED: int = 42

# ── Section 1: dynamics under Poisson input ───────────────────────
DYNAMICS_INPUT_RATE_HZ: float = 80.0


def build_fixed_weights(seed: int) -> dict[str, torch.Tensor]:
    """Returns W_in, W_ei, W_ie tensors with the same statistics the
    training section uses for its frozen recurrents, but with a
    larger W_in so the network fires under uniform Poisson input."""
    g = torch.Generator().manual_seed(seed)
    w_in_dense = (
        torch.randn(N_IN, N_E, generator=g) * 1.2 + 1.2  # mean 1.2, std 1.2
    )
    mask = (torch.rand(N_IN, N_E, generator=g) > W_IN_SPARSITY).float()
    w_in = (w_in_dense * mask).clamp_min(0.0)
    w_ei = (torch.randn(N_E, N_I, generator=g) * W_EI_STD + W_EI_MEAN).clamp_min(0.0)
    w_ie = (torch.randn(N_I, N_E, generator=g) * W_IE_STD + W_IE_MEAN).clamp_min(0.0)
    return {"W_in": w_in, "W_ei": w_ei, "W_ie": w_ie}


def simulate_dynamics(
    W: dict[str, torch.Tensor], input_rate_hz: float, seed: int,
) -> dict[str, np.ndarray]:
    """Forward pass with no training and no gradient. Uniform Poisson
    input on every channel at input_rate_hz for the full T_MS window."""
    g = torch.Generator().manual_seed(seed)
    p_in = input_rate_hz * DT / 1000.0
    V_E = torch.full((N_E,), V_REST)
    V_I = torch.full((N_I,), V_REST)
    s_E_prev = torch.zeros(N_E)
    s_I_prev = torch.zeros(N_I)
    spk_E = torch.zeros(N_STEPS, N_E)
    spk_I = torch.zeros(N_STEPS, N_I)
    alpha = DT / TAU_M_MS
    for t in range(N_STEPS):
        x = (torch.rand(N_IN, generator=g) < p_in).float()
        I_E = x @ W["W_in"] - s_I_prev @ W["W_ie"]
        I_I = s_E_prev @ W["W_ei"]
        V_E = V_E + alpha * (-(V_E - V_REST) + R_M * I_E)
        V_I = V_I + alpha * (-(V_I - V_REST) + R_M * I_I)
        V_E = torch.clamp(V_E, min=V_REST)
        V_I = torch.clamp(V_I, min=V_REST)
        s_E = (V_E >= V_TH).float()
        s_I = (V_I >= V_TH).float()
        spk_E[t] = s_E
        spk_I[t] = s_I
        V_E = torch.where(s_E.bool(), torch.full_like(V_E, V_RESET), V_E)
        V_I = torch.where(s_I.bool(), torch.full_like(V_I, V_RESET), V_I)
        s_E_prev, s_I_prev = s_E, s_I
    return {"spk_E": spk_E.numpy(), "spk_I": spk_I.numpy()}


# ── Section 2: training ──────────────────────────────────────────
INPUT_RATE_HZ: float = 80.0     # peak Poisson rate for a fully-on pixel

N_TRAIN: int = 10000
N_TEST: int = 1000
EPOCHS: int = 30
BATCH_SIZE: int = 64
LR: float = 5e-4
SURROGATE_SLOPE: float = 1.0
# TBPTT window — gradient compounds within K steps, not the full
# trial. See nb040.mdx Section 3 for the math.
TBPTT_WINDOW: int = 10
GRAD_CLIP_VALUE: float = 1.0


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


class CubaPingNet(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w_in_dense = torch.randn(N_IN, N_E, generator=g) * W_IN_INIT_STD
        mask = (torch.rand(N_IN, N_E, generator=g) > W_IN_SPARSITY).float()
        self.W_in = nn.Parameter((w_in_dense * mask).abs())
        w_out = torch.randn(N_E, N_CLASSES, generator=g) * 0.05
        self.W_out = nn.Parameter(w_out)
        self.b_out = nn.Parameter(torch.zeros(N_CLASSES))
        w_ei = (
            torch.randn(N_E, N_I, generator=g) * W_EI_STD + W_EI_MEAN
        ).clamp_min(0.0)
        w_ie = (
            torch.randn(N_I, N_E, generator=g) * W_IE_STD + W_IE_MEAN
        ).clamp_min(0.0)
        self.register_buffer("W_ei", w_ei)
        self.register_buffer("W_ie", w_ie)

    def forward(self, input_spikes: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, T, _ = input_spikes.shape
        dev = input_spikes.device
        V_E = torch.zeros(B, N_E, device=dev)
        V_I = torch.zeros(B, N_I, device=dev)
        s_E_prev = torch.zeros(B, N_E, device=dev)
        s_I_prev = torch.zeros(B, N_I, device=dev)
        V_out = torch.zeros(B, N_CLASSES, device=dev)
        readout = torch.zeros(B, N_CLASSES, device=dev)
        e_spike_total = torch.zeros((), device=dev)
        i_spike_total = torch.zeros((), device=dev)
        alpha = DT / TAU_M_MS
        alpha_out = DT / TAU_OUT_MS
        for t in range(T):
            # TBPTT: detach state every K steps so gradient horizon
            # is bounded.
            if self.training and t > 0 and t % TBPTT_WINDOW == 0:
                V_E = V_E.detach()
                V_I = V_I.detach()
                s_E_prev = s_E_prev.detach()
                s_I_prev = s_I_prev.detach()
                V_out = V_out.detach()
            x = input_spikes[:, t, :]
            I_E = x @ self.W_in - s_I_prev @ self.W_ie
            I_I = s_E_prev @ self.W_ei
            V_E = V_E + alpha * (-(V_E - V_REST) + R_M * I_E)
            V_I = V_I + alpha * (-(V_I - V_REST) + R_M * I_I)
            # Floor at V_REST: V can't go below the reversal of the
            # dominant ion (biological equivalent of E_K). Without
            # this floor, a single synchronous I-burst pushes V_E to
            # very negative values and takes tens of ms to recover.
            V_E = torch.clamp(V_E, min=V_REST)
            V_I = torch.clamp(V_I, min=V_REST)
            s_E = spike(V_E, SURROGATE_SLOPE)
            s_I = spike(V_I, SURROGATE_SLOPE)
            V_E = V_E - s_E.detach() * (V_TH - V_RESET)
            V_I = V_I - s_I.detach() * (V_TH - V_RESET)
            # Output-LIF mem-mean readout: an output LIF layer
            # integrates the hidden E spikes through W_out, and the
            # readout is the time-average of its membrane. The
            # surrogate gradient at s_E makes this trainable; hidden
            # spikes are now mandatory because the output layer only
            # accumulates when they fire.
            V_out = V_out + alpha_out * (-V_out + s_E @ self.W_out)
            readout = readout + V_out
            e_spike_total = e_spike_total + s_E.sum()
            i_spike_total = i_spike_total + s_I.sum()
            s_E_prev, s_I_prev = s_E, s_I
        logits = readout / T + self.b_out
        e_rate = float(e_spike_total.detach()) / (B * N_E * T * DT / 1000.0)
        i_rate = float(i_spike_total.detach()) / (B * N_I * T * DT / 1000.0)
        return logits, {"rate_e_hz": e_rate, "rate_i_hz": i_rate}


# ── Data ─────────────────────────────────────────────────────────
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


def evaluate(net: CubaPingNet, X: torch.Tensor, y: torch.Tensor,
             device: torch.device, gen: torch.Generator) -> dict:
    net.eval()
    correct = total = 0
    e_rate_sum = i_rate_sum = 0.0
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
            i_rate_sum += info["rate_i_hz"]
            n_batches += 1
    return {
        "acc": 100.0 * correct / total,
        "rate_e_hz": e_rate_sum / max(n_batches, 1),
        "rate_i_hz": i_rate_sum / max(n_batches, 1),
    }


def train(
    X_tr: torch.Tensor, y_tr: torch.Tensor,
    X_te: torch.Tensor, y_te: torch.Tensor, device: torch.device,
) -> dict:
    print("\n[nb040] training")
    net = CubaPingNet(seed=SEED).to(device)
    print(f"  trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad):,}")
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
            f"acc={ev['acc']:5.2f}%  "
            f"E={ev['rate_e_hz']:6.2f} Hz  I={ev['rate_i_hz']:6.2f} Hz  "
            f"[{dt_ep:5.1f}s]"
        )
        epoch_rows.append({
            "epoch": ep, "train_loss": train_loss, **ev, "duration_s": dt_ep,
        })

    # Capture a trained-network raster on the test sample with the
    # most E spikes (avoid an empty figure if sample 0 happens to be
    # silent).
    net.eval()
    with torch.no_grad():
        scan_B = min(16, len(X_te))
        xb = X_te[:scan_B].to(device)
        spk = encode_batch(xb, gen).to(device)
        V_E = torch.zeros(scan_B, N_E, device=device)
        V_I = torch.zeros(scan_B, N_I, device=device)
        s_E_prev = torch.zeros(scan_B, N_E, device=device)
        s_I_prev = torch.zeros(scan_B, N_I, device=device)
        e_per = torch.zeros(scan_B, device=device)
        alpha = DT / TAU_M_MS
        for t in range(N_STEPS):
            x = spk[:, t, :]
            I_E = x @ net.W_in - s_I_prev @ net.W_ie
            I_I = s_E_prev @ net.W_ei
            V_E = V_E + alpha * (-(V_E - V_REST) + R_M * I_E)
            V_I = V_I + alpha * (-(V_I - V_REST) + R_M * I_I)
            V_E = torch.clamp(V_E, min=V_REST)
            V_I = torch.clamp(V_I, min=V_REST)
            s_E = (V_E >= V_TH).float()
            s_I = (V_I >= V_TH).float()
            V_E = torch.where(s_E.bool(), torch.full_like(V_E, V_RESET), V_E)
            V_I = torch.where(s_I.bool(), torch.full_like(V_I, V_RESET), V_I)
            e_per = e_per + s_E.sum(dim=1)
            s_E_prev, s_I_prev = s_E, s_I
        sample_idx = int(e_per.argmax().item())
        print(f"  raster sample: idx={sample_idx} digit={int(y_te[sample_idx].item())}")

        # Resimulate that one sample logging full per-step spikes.
        spk_one = encode_batch(X_te[sample_idx:sample_idx + 1].to(device), gen).to(device)
        V_E = torch.zeros(1, N_E, device=device)
        V_I = torch.zeros(1, N_I, device=device)
        s_E_prev = torch.zeros(1, N_E, device=device)
        s_I_prev = torch.zeros(1, N_I, device=device)
        spk_E_log = torch.zeros(N_STEPS, N_E)
        spk_I_log = torch.zeros(N_STEPS, N_I)
        for t in range(N_STEPS):
            x = spk_one[:, t, :]
            I_E = x @ net.W_in - s_I_prev @ net.W_ie
            I_I = s_E_prev @ net.W_ei
            V_E = V_E + alpha * (-(V_E - V_REST) + R_M * I_E)
            V_I = V_I + alpha * (-(V_I - V_REST) + R_M * I_I)
            V_E = torch.clamp(V_E, min=V_REST)
            V_I = torch.clamp(V_I, min=V_REST)
            s_E = (V_E >= V_TH).float()
            s_I = (V_I >= V_TH).float()
            V_E = torch.where(s_E.bool(), torch.full_like(V_E, V_RESET), V_E)
            V_I = torch.where(s_I.bool(), torch.full_like(V_I, V_RESET), V_I)
            spk_E_log[t] = s_E[0].cpu()
            spk_I_log[t] = s_I[0].cpu()
            s_E_prev, s_I_prev = s_E, s_I

    return {
        "epochs": epoch_rows,
        "final": epoch_rows[-1],
        "raster": {
            "spk_E": spk_E_log.numpy(),
            "spk_I": spk_I_log.numpy(),
            "label": int(y_te[sample_idx].item()),
        },
    }


# ── Plotting ─────────────────────────────────────────────────────
def _stamp(fig) -> None:
    fig.text(
        0.995, 0.005, f"nb040-{int(time.time())}",
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
    ax_e.set_ylim(0, N_E)
    ax_e.set_xlim(0, T_MS)
    ax_e.set_title(title, fontsize=theme.SIZE_TITLE)
    i_idx, i_t = np.where(spk_i.T)
    ax_i.scatter(t_ms[i_t], i_idx, s=1.0, c=theme.DEEP_RED, marker="|", linewidths=0.5)
    ax_i.set_ylabel("I neuron")
    ax_i.set_ylim(0, N_I)
    ax_i.set_xlim(0, T_MS)
    ax_i.set_xlabel("time (ms)")
    fig.tight_layout()
    _stamp(fig)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_learning_curves(run: dict, out_path: Path) -> None:
    theme.apply()
    fig, (ax_loss, ax_acc, ax_rate) = plt.subplots(1, 3, figsize=(13.0, 4.5), dpi=150)
    eps = [e["epoch"] for e in run["epochs"]]
    loss = [e["train_loss"] for e in run["epochs"]]
    acc = [e["acc"] for e in run["epochs"]]
    re = [e["rate_e_hz"] for e in run["epochs"]]
    ri = [e["rate_i_hz"] for e in run["epochs"]]
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


def plot_trained_raster(run: dict, out_path: Path) -> None:
    sample = run["raster"]
    final = run["final"]
    title = (
        f"Trained CUBA-PING — digit {sample['label']}, "
        f"acc={final['acc']:.1f}%, E={final['rate_e_hz']:.1f} Hz, "
        f"I={final['rate_i_hz']:.1f} Hz"
    )
    plot_dynamics_raster(sample, title, out_path)


# ── Main ─────────────────────────────────────────────────────────
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
    print(f"[nb040] CUBA-PING (instant synapses), device={device}")

    # Section 1: dynamics — forward only, fixed weights.
    print(f"\n[nb040:dynamics] uniform Poisson input @ {DYNAMICS_INPUT_RATE_HZ} Hz/channel")
    W_fixed = build_fixed_weights(SEED)
    dyn = simulate_dynamics(W_fixed, DYNAMICS_INPUT_RATE_HZ, SEED + 1)
    e_rate = float(dyn["spk_E"].sum()) / (N_E * T_MS / 1000.0)
    i_rate = float(dyn["spk_I"].sum()) / (N_I * T_MS / 1000.0)
    print(f"  E rate = {e_rate:.2f} Hz, I rate = {i_rate:.2f} Hz")
    plot_dynamics_raster(
        dyn,
        f"CUBA-PING dynamics (no training) — input {DYNAMICS_INPUT_RATE_HZ:g} Hz "
        f"(E = {e_rate:.1f} Hz, I = {i_rate:.1f} Hz)",
        FIGURES / "dynamics_raster.png",
    )
    print(f"  wrote {FIGURES / 'dynamics_raster.png'}")

    # Section 2: training.
    X_tr, y_tr, X_te, y_te = load_mnist()
    print(f"  data: {len(X_tr)} train, {len(X_te)} test")
    run = train(X_tr, y_tr, X_te, y_te, device)
    plot_learning_curves(run, FIGURES / "learning_curves.png")
    print(f"  wrote {FIGURES / 'learning_curves.png'}")
    plot_trained_raster(run, FIGURES / "trained_raster.png")
    print(f"  wrote {FIGURES / 'trained_raster.png'}")

    duration_s = time.monotonic() - t_start
    summary = {
        "slug": SLUG,
        "duration_s": round(duration_s, 1),
        "device": str(device),
        "config": {
            "n_e": N_E, "n_i": N_I, "n_in": N_IN, "n_classes": N_CLASSES,
            "t_ms": T_MS, "dt": DT, "tau_m_ms": TAU_M_MS,
            "v_th": V_TH, "v_reset": V_RESET, "v_rest": V_REST, "r_m": R_M,
            "input_rate_hz": INPUT_RATE_HZ,
            "dynamics_input_rate_hz": DYNAMICS_INPUT_RATE_HZ,
            "n_train": N_TRAIN, "n_test": N_TEST,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "tbptt_window": TBPTT_WINDOW,
            "grad_clip_value": GRAD_CLIP_VALUE,
            "surrogate_slope": SURROGATE_SLOPE, "seed": SEED,
        },
        "dynamics": {"rate_e_hz": e_rate, "rate_i_hz": i_rate},
        "final_acc": run["final"]["acc"],
        "final_rate_e_hz": run["final"]["rate_e_hz"],
        "final_rate_i_hz": run["final"]["rate_i_hz"],
        "epochs": [
            {k: v for k, v in e.items() if k != "raster"}
            for e in run["epochs"]
        ],
        "success_criteria": [
            {
                "label": "dynamics raster rendered",
                "passed": (FIGURES / "dynamics_raster.png").exists(),
                "detail": f"E={e_rate:.1f} Hz, I={i_rate:.1f} Hz",
            },
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
