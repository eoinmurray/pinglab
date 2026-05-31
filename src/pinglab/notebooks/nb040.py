"""Notebook runner for entry 040 — training CUBA-PING on MNIST in
both synaptic modes.

Standalone PyTorch runner. Takes the network from nb039 (current-based
LIF PING with random fixed W_ei / W_ie) and makes W_in + W_out
trainable. Surrogate-gradient backprop through time. Trains two
networks side by side — instant synapses vs exponential synapses
(τ_AMPA = 2 ms, τ_GABA = 9 ms) — on Poisson-encoded MNIST.

No oscilloscope CLI, no Modal — pure PyTorch on whatever device is
available. Trial settings are intentionally modest so the run fits
on CPU in a few minutes per mode.

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

# ── Network architecture (matches nb039) ────────────────────────────
N_E: int = 1024
N_I: int = 256
N_IN: int = 784
N_CLASSES: int = 10

T_MS: float = 200.0
DT: float = 1.0   # Coarser than nb039's 0.1 ms — keeps the BPTT
                  # unroll tractable on CPU; 200 steps per trial.
N_STEPS: int = int(T_MS / DT)

TAU_M_MS: float = 20.0
V_REST: float = 0.0
V_TH: float = 1.0
V_RESET: float = 0.0
R_M: float = 1.0

# Initialisation: smaller weights than nb039 because the readout is
# now learned and the optimiser will scale things itself; we just need
# a sensible starting point.
W_IN_INIT_STD: float = 0.5
W_EI_MEAN: float = 1.0
W_EI_STD: float = 0.1
W_IE_MEAN: float = 2.0
W_IE_STD: float = 0.2
W_IN_SPARSITY: float = 0.95

INPUT_RATE_HZ: float = 80.0   # peak Poisson rate for a fully-on pixel

TAU_AMPA_MS: float = 2.0
TAU_GABA_MS: float = 9.0

# Training tier — intentionally small so it fits on CPU.
N_TRAIN: int = 2000
N_TEST: int = 400
EPOCHS: int = 50
BATCH_SIZE: int = 64
LR: float = 5e-4
SEED: int = 42
SURROGATE_SLOPE: float = 1.0


# ── Surrogate gradient ─────────────────────────────────────────────
class SurrSpike(torch.autograd.Function):
    """Forward: Heaviside. Backward: arctan surrogate — bounded, no
    overflow possible regardless of how far V drifts from threshold."""

    @staticmethod
    def forward(ctx, v_minus_th: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(v_minus_th)
        ctx.slope = slope
        return (v_minus_th >= 0).float()

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (v,) = ctx.saved_tensors
        s = ctx.slope
        # d/dv [1/pi * arctan(s*pi*v) + 1/2] = s / (1 + (s*pi*v)^2)
        denom = 1.0 + (s * np.pi * v).pow(2)
        return grad_out * s / denom, None


def spike(v: torch.Tensor, slope: float) -> torch.Tensor:
    return SurrSpike.apply(v - V_TH, slope)


# ── Model ──────────────────────────────────────────────────────────
class CubaPingNet(nn.Module):
    def __init__(self, synapse_mode: str, seed: int):
        super().__init__()
        if synapse_mode not in ("instant", "exp"):
            raise ValueError(f"unknown synapse_mode {synapse_mode!r}")
        self.synapse_mode = synapse_mode
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

        self.d_ampa = float(np.exp(-DT / TAU_AMPA_MS))
        self.d_gaba = float(np.exp(-DT / TAU_GABA_MS))
        self.s_ampa = 1.0 - self.d_ampa
        self.s_gaba = 1.0 - self.d_gaba

    def forward(self, input_spikes: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """input_spikes: (B, T, N_IN) binary tensor.
        Returns (logits, info-dict-with-rates)."""
        B, T, _ = input_spikes.shape
        dev = input_spikes.device
        V_E = torch.zeros(B, N_E, device=dev)
        V_I = torch.zeros(B, N_I, device=dev)
        s_E_prev = torch.zeros(B, N_E, device=dev)
        s_I_prev = torch.zeros(B, N_I, device=dev)
        if self.synapse_mode == "exp":
            I_E_exc = torch.zeros(B, N_E, device=dev)
            I_E_inh = torch.zeros(B, N_E, device=dev)
            I_I_exc = torch.zeros(B, N_I, device=dev)
        readout = torch.zeros(B, N_CLASSES, device=dev)
        e_spike_total = torch.zeros((), device=dev)
        i_spike_total = torch.zeros((), device=dev)
        alpha = DT / TAU_M_MS

        for t in range(T):
            x = input_spikes[:, t, :]
            # Detach the I-loop feedback so gradient doesn't compound
            # geometrically across cycles. W_ei and W_ie are frozen
            # buffers anyway — no gradient needed through them.
            if self.synapse_mode == "instant":
                I_E = x @ self.W_in - s_I_prev.detach() @ self.W_ie
                I_I = s_E_prev.detach() @ self.W_ei
            else:
                I_E_exc = self.d_ampa * I_E_exc + self.s_ampa * (x @ self.W_in)
                I_E_inh = self.d_gaba * I_E_inh + self.s_gaba * (s_I_prev.detach() @ self.W_ie)
                I_I_exc = self.d_ampa * I_I_exc + self.s_ampa * (s_E_prev.detach() @ self.W_ei)
                I_E = I_E_exc - I_E_inh
                I_I = I_I_exc
            V_E = V_E + alpha * (-(V_E - V_REST) + R_M * I_E)
            V_I = V_I + alpha * (-(V_I - V_REST) + R_M * I_I)
            s_E = spike(V_E, SURROGATE_SLOPE)
            s_I = spike(V_I, SURROGATE_SLOPE)
            # Hard reset via straight-through detach so spike is graded.
            V_E = V_E - s_E.detach() * (V_TH - V_RESET)
            V_I = V_I - s_I.detach() * (V_TH - V_RESET)
            readout = readout + s_E @ self.W_out
            e_spike_total = e_spike_total + s_E.sum()
            i_spike_total = i_spike_total + s_I.sum()
            s_E_prev, s_I_prev = s_E, s_I

        logits = readout / T + self.b_out
        e_rate = float(e_spike_total.detach()) / (B * N_E * T * DT / 1000.0)
        i_rate = float(i_spike_total.detach()) / (B * N_I * T * DT / 1000.0)
        return logits, {"rate_e_hz": e_rate, "rate_i_hz": i_rate}


# ── Data ───────────────────────────────────────────────────────────
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
    """Poisson-encode pixel intensities to spike trains of shape
    (B, T, N_IN). Each pixel × is the per-step spike probability after
    scaling by INPUT_RATE_HZ * DT / 1000."""
    B = X.shape[0]
    p = X.unsqueeze(1) * (INPUT_RATE_HZ * DT / 1000.0)
    p = p.expand(B, N_STEPS, N_IN).contiguous()
    spikes = torch.bernoulli(p, generator=gen)
    return spikes


# ── Training ───────────────────────────────────────────────────────
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


def train_one_mode(
    synapse_mode: str, X_tr: torch.Tensor, y_tr: torch.Tensor,
    X_te: torch.Tensor, y_te: torch.Tensor, device: torch.device,
) -> dict:
    print(f"\n[nb040:{synapse_mode}] training")
    net = CubaPingNet(synapse_mode=synapse_mode, seed=SEED).to(device)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  trainable parameters: {n_params:,}")
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    gen = torch.Generator(device="cpu").manual_seed(SEED + 7)
    epoch_rows: list[dict] = []
    for ep in range(1, EPOCHS + 1):
        net.train()
        perm = torch.randperm(len(X_tr))
        loss_sum = 0.0
        n = 0
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
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
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

    # Capture a single test trial's rasters for the figure.
    net.eval()
    with torch.no_grad():
        sample_idx = 0
        xb = X_te[sample_idx:sample_idx + 1].to(device)
        spk = encode_batch(xb, gen).to(device)
        B = 1
        V_E = torch.zeros(B, N_E, device=device)
        V_I = torch.zeros(B, N_I, device=device)
        s_E_prev = torch.zeros(B, N_E, device=device)
        s_I_prev = torch.zeros(B, N_I, device=device)
        if synapse_mode == "exp":
            I_E_exc = torch.zeros(B, N_E, device=device)
            I_E_inh = torch.zeros(B, N_E, device=device)
            I_I_exc = torch.zeros(B, N_I, device=device)
        spk_E_log = torch.zeros(N_STEPS, N_E)
        spk_I_log = torch.zeros(N_STEPS, N_I)
        alpha = DT / TAU_M_MS
        for t in range(N_STEPS):
            x = spk[:, t, :]
            if synapse_mode == "instant":
                I_E = x @ net.W_in - s_I_prev @ net.W_ie
                I_I = s_E_prev @ net.W_ei
            else:
                I_E_exc = net.d_ampa * I_E_exc + net.s_ampa * (x @ net.W_in)
                I_E_inh = net.d_gaba * I_E_inh + net.s_gaba * (s_I_prev @ net.W_ie)
                I_I_exc = net.d_ampa * I_I_exc + net.s_ampa * (s_E_prev @ net.W_ei)
                I_E = I_E_exc - I_E_inh
                I_I = I_I_exc
            V_E = V_E + alpha * (-(V_E - V_REST) + R_M * I_E)
            V_I = V_I + alpha * (-(V_I - V_REST) + R_M * I_I)
            s_E = (V_E >= V_TH).float()
            s_I = (V_I >= V_TH).float()
            V_E = torch.where(s_E.bool(), torch.full_like(V_E, V_RESET), V_E)
            V_I = torch.where(s_I.bool(), torch.full_like(V_I, V_RESET), V_I)
            spk_E_log[t] = s_E[0].cpu()
            spk_I_log[t] = s_I[0].cpu()
            s_E_prev, s_I_prev = s_E, s_I

    return {
        "synapse_mode": synapse_mode,
        "epochs": epoch_rows,
        "final": epoch_rows[-1],
        "raster": {
            "spk_E": spk_E_log.numpy(),
            "spk_I": spk_I_log.numpy(),
            "label": int(y_te[0].item()),
        },
    }


# ── Plotting ───────────────────────────────────────────────────────
def _stamp(fig) -> None:
    fig.text(
        0.995, 0.005, f"nb040-{int(time.time())}",
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


def plot_learning_curves(runs: list[dict], out_path: Path) -> None:
    theme.apply()
    fig, (ax_loss, ax_acc, ax_rate) = plt.subplots(1, 3, figsize=(13.0, 4.5), dpi=150)
    for run in runs:
        eps = [e["epoch"] for e in run["epochs"]]
        loss = [e["train_loss"] for e in run["epochs"]]
        acc = [e["acc"] for e in run["epochs"]]
        re = [e["rate_e_hz"] for e in run["epochs"]]
        ri = [e["rate_i_hz"] for e in run["epochs"]]
        color = theme.INK_BLACK if run["synapse_mode"] == "instant" else theme.DEEP_RED
        label = run["synapse_mode"]
        ax_loss.plot(eps, loss, marker="o", color=color, label=label)
        ax_acc.plot(eps, acc, marker="o", color=color, label=label)
        ax_rate.plot(eps, re, marker="o", color=color, label=f"{label} E")
        ax_rate.plot(eps, ri, marker="s", color=color, ls="--", label=f"{label} I")
    for ax, ylab, title in (
        (ax_loss, "Cross-entropy", "Training loss"),
        (ax_acc, "Test accuracy (%)", "Test accuracy"),
        (ax_rate, "Mean firing rate (Hz)", "Hidden E (solid) / I (dashed) rate"),
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=theme.SIZE_TITLE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=theme.SIZE_CAPTION, frameon=False)
    ax_acc.set_ylim(0, 100)
    fig.tight_layout()
    _stamp(fig)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trained_raster(run: dict, out_path: Path) -> None:
    theme.apply()
    sample = run["raster"]
    spk_e, spk_i = sample["spk_E"], sample["spk_I"]
    t_ms = np.arange(spk_e.shape[0]) * DT
    fig, (ax_e, ax_i) = plt.subplots(
        2, 1, figsize=(8.0, 4.5), sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )
    e_idx, e_t = np.where(spk_e.T)
    ax_e.scatter(t_ms[e_t], e_idx, s=1.5, c=theme.INK_BLACK, marker="|", linewidths=0.6)
    ax_e.set_ylabel("E neuron")
    ax_e.set_ylim(0, N_E)
    final = run["final"]
    ax_e.set_title(
        f"Trained CUBA-PING ({run['synapse_mode']} synapses) — "
        f"digit {sample['label']}, acc={final['acc']:.1f}%, "
        f"E={final['rate_e_hz']:.1f} Hz, I={final['rate_i_hz']:.1f} Hz",
        fontsize=theme.SIZE_TITLE,
    )
    i_idx, i_t = np.where(spk_i.T)
    ax_i.scatter(t_ms[i_t], i_idx, s=1.5, c=theme.DEEP_RED, marker="|", linewidths=0.6)
    ax_i.set_ylabel("I neuron")
    ax_i.set_ylim(0, N_I)
    ax_i.set_xlabel("time (ms)")
    fig.tight_layout()
    _stamp(fig)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────
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
    print(f"[nb040] training CUBA-PING on MNIST, device={device}")
    print(f"  N_E={N_E}, N_I={N_I}, T={T_MS}ms, dt={DT}ms, batch={BATCH_SIZE}")

    X_tr, y_tr, X_te, y_te = load_mnist()
    print(f"  data: {len(X_tr)} train, {len(X_te)} test")

    runs: list[dict] = []
    for mode in ("instant", "exp"):
        run = train_one_mode(mode, X_tr, y_tr, X_te, y_te, device)
        runs.append(run)
        plot_trained_raster(run, FIGURES / f"raster__{mode}.png")
        print(f"  wrote {FIGURES / f'raster__{mode}.png'}")

    plot_learning_curves(runs, FIGURES / "learning_curves.png")
    print(f"  wrote {FIGURES / 'learning_curves.png'}")

    duration_s = time.monotonic() - t_start
    summary = {
        "slug": SLUG,
        "duration_s": round(duration_s, 1),
        "device": str(device),
        "config": {
            "n_e": N_E, "n_i": N_I, "n_in": N_IN, "n_classes": N_CLASSES,
            "t_ms": T_MS, "dt": DT, "tau_m_ms": TAU_M_MS,
            "v_th": V_TH, "v_reset": V_RESET, "v_rest": V_REST, "r_m": R_M,
            "tau_ampa_ms": TAU_AMPA_MS, "tau_gaba_ms": TAU_GABA_MS,
            "input_rate_hz": INPUT_RATE_HZ,
            "n_train": N_TRAIN, "n_test": N_TEST,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "surrogate_slope": SURROGATE_SLOPE, "seed": SEED,
        },
        "results": [
            {
                "synapse_mode": r["synapse_mode"],
                "final_acc": r["final"]["acc"],
                "final_rate_e_hz": r["final"]["rate_e_hz"],
                "final_rate_i_hz": r["final"]["rate_i_hz"],
                "epochs": [
                    {k: v for k, v in e.items() if k != "raster"}
                    for e in r["epochs"]
                ],
            }
            for r in runs
        ],
        "success_criteria": [
            {
                "label": f"{r['synapse_mode']}: final acc above chance",
                "passed": r["final"]["acc"] > 15.0,
                "detail": f"{r['final']['acc']:.2f}%",
            }
            for r in runs
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {duration_s:.1f}s")


if __name__ == "__main__":
    main()
