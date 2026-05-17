"""Notebook runner for entry 032 — DMTS with trainable W_ee + SGCC + NMDA.

Pulls the design together from nb030 (structural bistability of coba),
nb031 (the lower edge of ping's sustained band), and the architectural
features available on COBANet:

  * ping config (ei_strength = 1) — inhibition provides the negative
    feedback that nb030 proved is structurally required.
  * trainable W_ee (--trainable-w-ee) — lets gradient descent find the
    recurrent-attractor regime instead of us hand-picking it.
  * slow synapse (--slow-syn) — NMDA-like long-decay excitatory channel,
    the substrate for cross-stimulus memory.
  * SGCC (--sgcc) — surgical gradient stabilizer that doesn't kill the
    cross-coupling signal the way --v-grad-dampen 1000 does.

Task: DMTS (delayed match-to-sample).
  *  sample phase [0, T_s):       MNIST digit A encoded as Poisson spikes.
  *  delay  phase [T_s, T_s+T_d): silence — network must hold sample state.
  *  probe  phase [T_s+T_d, T):   MNIST digit B encoded as Poisson spikes.
  *  target label: 1 if A == B (match), 0 otherwise.

Readout (custom, bypasses the built-in mem-mean): mean E spike count
over the probe window only → Linear(N_E → 2). This avoids the dilution
problem that diluted the per-trial gradient in nb028/29 — only what
happens during the probe window contributes to the loss.

Notebook entry: src/docs/src/pages/notebooks/nb032.mdx
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

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO / "src" / "pinglab" / "notebooks"))
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402
from pinglab import theme  # noqa: E402

SLUG = "nb032"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG

# ── Network setup ─────────────────────────────────────────────────────
DT = 1.0             # ms — keep BPTT graph shallow (300 ms / 1 ms = 300 steps)
N_E = 256
N_IN = 784           # MNIST
DATASET = "mnist"
EI_STRENGTH = 1.0    # ping
W_IN_MEAN = 1.2
W_IN_STD = 0.36
W_IN_SPARSITY = 0.95
W_EE_INIT_MEAN = 0.05  # lower edge — preserves sample pixel pattern via NMDA tail
W_EE_INIT_STD = 0.005  # rather than washing it through a gamma attractor
SLOW_SYN_GAIN = 0.5
SGCC_ALPHA = 0.5

# ── DMTS task ─────────────────────────────────────────────────────────
T_SAMPLE_MS = 100.0
T_DELAY_MS = 100.0
T_PROBE_MS = 100.0
T_TOTAL_MS = T_SAMPLE_MS + T_DELAY_MS + T_PROBE_MS
STIM_PEAK_RATE_HZ = 50.0  # peak Poisson rate per pixel during stim windows
N_OUT = 2                  # match / non-match
N_CLASSES_FOR_PAIRING = 10  # how many digit classes to sample from

# ── Training ──────────────────────────────────────────────────────────
LR = 1e-3
BATCH_SIZE = 32
SEED = 42

DEFAULT_TIER = "small"
TIER_CONFIG: dict[str, dict] = {
    "extra small": {"n_train": 256,  "n_test": 64,  "epochs": 2},
    "small":       {"n_train": 1024, "n_test": 256, "epochs": 10},
    "medium":      {"n_train": 4096, "n_test": 512, "epochs": 20},
    "large":       {"n_train": 8192, "n_test": 1024, "epochs": 40},
}


def _stamp(fig, run_id: str) -> None:
    fig.text(
        0.99, 0.005, run_id,
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


# ── DMTS sample generator ─────────────────────────────────────────────
def load_mnist_split():
    """Return (X_train, y_train, X_test, y_test) — pixel intensities in [0,1]."""
    from torchvision import datasets, transforms

    tr = datasets.MNIST(
        root="/tmp/mnist", train=True, download=True,
        transform=transforms.ToTensor(),
    )
    te = datasets.MNIST(
        root="/tmp/mnist", train=False, download=True,
        transform=transforms.ToTensor(),
    )
    X_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_tr = tr.targets.numpy()
    X_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_te = te.targets.numpy()
    return X_tr, y_tr, X_te, y_te


def _pair_indices(y: np.ndarray, n_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build n_pairs (idx_A, idx_B, label) trials with balanced match/non-match.

    Half the pairs are sample-class == probe-class (label = 1, match), the
    other half are sample-class != probe-class (label = 0, non-match).
    Within each class we pick distinct samples so the network doesn't see
    pixel-identical inputs at both phases.
    """
    rng = np.random.default_rng(seed)
    by_class = {c: np.where(y == c)[0] for c in range(N_CLASSES_FOR_PAIRING)}
    idx_a = np.empty(n_pairs, dtype=np.int64)
    idx_b = np.empty(n_pairs, dtype=np.int64)
    labels = np.empty(n_pairs, dtype=np.int64)
    for i in range(n_pairs):
        if i < n_pairs // 2:
            c = int(rng.integers(0, N_CLASSES_FOR_PAIRING))
            pool = by_class[c]
            a, b = rng.choice(pool, size=2, replace=False)
            idx_a[i], idx_b[i], labels[i] = a, b, 1
        else:
            ca, cb = rng.choice(N_CLASSES_FOR_PAIRING, size=2, replace=False)
            a = int(rng.choice(by_class[int(ca)]))
            b = int(rng.choice(by_class[int(cb)]))
            idx_a[i], idx_b[i], labels[i] = a, b, 0
    # Shuffle so match/non-match are interleaved
    order = rng.permutation(n_pairs)
    return idx_a[order], idx_b[order], labels[order]


def _encode_pair_spikes(
    pixel_a: np.ndarray,
    pixel_b: np.ndarray,
    T_sample_steps: int,
    T_delay_steps: int,
    T_probe_steps: int,
    seed: int,
) -> np.ndarray:
    """Build a (T_total, N_IN) Poisson-encoded spike pattern for one trial.

    Sample window: pixel_a * rate. Delay: zero. Probe: pixel_b * rate.
    """
    from cli.encoders import encode_image_spikes

    # Encode each window separately with the canonical encoder, then stitch.
    # base_rate=0 outside the window, stim_rate=STIM_PEAK_RATE_HZ inside.
    sample = encode_image_spikes(
        pixel_a, T_sample_steps, DT,
        base_rate=0.0, stim_rate=STIM_PEAK_RATE_HZ,
        step_on_ms=0.0, step_off_ms=T_SAMPLE_MS,
        seed=seed,
    ).numpy()
    delay = np.zeros((T_delay_steps, N_IN), dtype=np.float32)
    probe = encode_image_spikes(
        pixel_b, T_probe_steps, DT,
        base_rate=0.0, stim_rate=STIM_PEAK_RATE_HZ,
        step_on_ms=0.0, step_off_ms=T_PROBE_MS,
        seed=seed + 1000003,
    ).numpy()
    return np.concatenate([sample, delay, probe], axis=0)


def build_dmts_batch(
    X: np.ndarray, y: np.ndarray,
    idx_a: np.ndarray, idx_b: np.ndarray, labels: np.ndarray,
    seed: int, device: torch.device,
):
    """Materialise a batch of (input_spikes, labels) for one training step."""
    T_sample_steps = int(T_SAMPLE_MS / DT)
    T_delay_steps = int(T_DELAY_MS / DT)
    T_probe_steps = int(T_PROBE_MS / DT)
    T_total_steps = T_sample_steps + T_delay_steps + T_probe_steps
    B = len(idx_a)
    spk = np.zeros((T_total_steps, B, N_IN), dtype=np.float32)
    for j in range(B):
        spk[:, j, :] = _encode_pair_spikes(
            X[idx_a[j]], X[idx_b[j]],
            T_sample_steps, T_delay_steps, T_probe_steps,
            seed=seed + j,
        )
    return (
        torch.from_numpy(spk).to(device),
        torch.from_numpy(labels.astype(np.int64)).to(device),
    )


# ── Network + custom readout ──────────────────────────────────────────
def build_dmts_net(device: torch.device):
    """ping COBANet at the sustained-band W_ee with trainable_w_ee=True."""
    import models as M
    from config import build_net, patch_dt

    M.N_IN = N_IN
    patch_dt(DT)
    net = build_net(
        "ping",
        w_in=(W_IN_MEAN, W_IN_STD),
        w_in_sparsity=W_IN_SPARSITY,
        w_ee=(W_EE_INIT_MEAN, W_EE_INIT_STD),
        ei_strength=EI_STRENGTH,
        slow_synapse=True,
        slow_syn_gain=SLOW_SYN_GAIN,
        sgcc=True,
        sgcc_alpha=SGCC_ALPHA,
        trainable_w_ee=True,
        hidden_sizes=[N_E],
        device=device,
    )
    return net


class DMTSReadout(nn.Module):
    """Linear(N_E → 2) on probe-window mean E spike count.

    Bypasses the built-in mem-mean readout — we only want what happens
    during the probe window to drive the loss, otherwise the gradient on
    W_out is averaged over the silent delay phase and washes out.
    """

    def __init__(self, n_e: int, n_out: int):
        super().__init__()
        self.proj = nn.Linear(n_e, n_out)
        nn.init.kaiming_uniform_(self.proj.weight, a=5.0 ** 0.5)
        nn.init.zeros_(self.proj.bias)

    def forward(self, e_spikes_probe: torch.Tensor) -> torch.Tensor:
        # e_spikes_probe: (T_probe, B, N_E) → mean over time → (B, N_E)
        rate = e_spikes_probe.mean(dim=0)
        return self.proj(rate)


def forward_and_readout(
    net: nn.Module, readout: DMTSReadout,
    input_spikes: torch.Tensor,
) -> torch.Tensor:
    """Run network, pull probe-window E activity, apply readout → logits."""
    import models as M

    T_sample_steps = int(T_SAMPLE_MS / DT)
    T_delay_steps = int(T_DELAY_MS / DT)
    T_probe_steps = int(T_PROBE_MS / DT)
    T_total_steps = T_sample_steps + T_delay_steps + T_probe_steps
    M.T_steps = T_total_steps
    M.T_ms = T_TOTAL_MS

    net.recording = True
    _ = net(input_spikes=input_spikes)
    # spike_record["hid"] is (T, N_E) for B=1 or (T, B, N_E) for batched.
    e_record = net.spike_record["hid"]
    if e_record.dim() == 2:
        e_record = e_record.unsqueeze(1)
    e_probe = e_record[T_sample_steps + T_delay_steps:]  # (T_probe, B, N_E)
    return readout(e_probe)


# ── Training ──────────────────────────────────────────────────────────
def evaluate(net, readout, X, y, idx_a, idx_b, labels, device, batch_size=32):
    net.eval()
    readout.eval()
    n = len(idx_a)
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            spk, lbl = build_dmts_batch(
                X, y, idx_a[start:end], idx_b[start:end], labels[start:end],
                seed=999_000 + start, device=device,
            )
            logits = forward_and_readout(net, readout, spk)
            pred = logits.argmax(dim=-1)
            correct += (pred == lbl).sum().item()
            total += lbl.numel()
    return correct / total


def train_dmts(tier_cfg, device, run_id):
    import models as M
    from cli import seed_everything

    seed_everything(SEED)
    X_tr, y_tr, X_te, y_te = load_mnist_split()
    n_train = tier_cfg["n_train"]
    n_test = tier_cfg["n_test"]
    epochs = tier_cfg["epochs"]

    train_a, train_b, train_lbl = _pair_indices(y_tr, n_train, seed=SEED)
    test_a, test_b, test_lbl = _pair_indices(y_te, n_test, seed=SEED + 1)

    net = build_dmts_net(device)
    net = net.to(device)
    readout = DMTSReadout(N_E, N_OUT).to(device)
    optim = torch.optim.Adam(
        list(net.parameters()) + list(readout.parameters()),
        lr=LR,
    )

    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": []}
    n_batches = n_train // BATCH_SIZE

    for ep in range(epochs):
        net.train()
        readout.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        # Re-shuffle training pairs each epoch
        perm = np.random.default_rng(SEED + ep + 1).permutation(n_train)
        t_a_ep = train_a[perm]
        t_b_ep = train_b[perm]
        t_lbl_ep = train_lbl[perm]
        for bi in range(n_batches):
            start = bi * BATCH_SIZE
            end = start + BATCH_SIZE
            spk, lbl = build_dmts_batch(
                X_tr, y_tr,
                t_a_ep[start:end], t_b_ep[start:end], t_lbl_ep[start:end],
                seed=SEED + ep * 10_000 + bi, device=device,
            )
            optim.zero_grad()
            logits = forward_and_readout(net, readout, spk)
            loss = F.cross_entropy(logits, lbl)
            if not torch.isfinite(loss):
                print(f"  ep {ep} bi {bi}: non-finite loss, skipping")
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(net.parameters()) + list(readout.parameters()), 1.0,
            )
            if bi == 0:
                # Report W_ee and readout gradient magnitudes at the first batch
                # of each epoch so we can see whether signal is actually flowing.
                w_ee_grad = next(iter(net.W_ee.parameters())).grad
                w_ee_g = float(w_ee_grad.abs().mean().item()) if w_ee_grad is not None else 0.0
                readout_g = float(readout.proj.weight.grad.abs().mean().item())
                print(f"    [grad] |grad_norm|={float(grad_norm):.4f}  "
                      f"|W_ee.grad|={w_ee_g:.2e}  |W_out.grad|={readout_g:.2e}")
            optim.step()
            epoch_loss += float(loss.item()) * lbl.numel()
            epoch_correct += int((logits.argmax(dim=-1) == lbl).sum().item())
            epoch_total += int(lbl.numel())

        train_acc = epoch_correct / max(1, epoch_total)
        test_acc = evaluate(net, readout, X_te, y_te, test_a, test_b, test_lbl,
                            device, batch_size=BATCH_SIZE)
        avg_loss = epoch_loss / max(1, epoch_total)
        history["epoch"].append(ep + 1)
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        print(f"  ep {ep+1:>2}/{epochs}  loss={avg_loss:.4f}  "
              f"train_acc={train_acc*100:5.1f}%  test_acc={test_acc*100:5.1f}%")

    return net, readout, history, (X_te, y_te, test_a, test_b, test_lbl)


# ── Figures ───────────────────────────────────────────────────────────
def fig_training_curve(history: dict, run_id: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    ax_loss, ax_acc = axes
    ax_loss.plot(history["epoch"], history["train_loss"],
                 color=theme.DEEP_RED, lw=1.5, marker="o")
    ax_loss.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    ax_loss.set_ylabel("Train loss (BCE)", fontsize=theme.SIZE_LABEL)
    ax_loss.set_title("Training loss", fontsize=theme.SIZE_LABEL)
    ax_loss.grid(True, alpha=0.3)

    ax_acc.axhline(50, color=theme.MUTED, lw=0.8, ls="--", label="chance")
    ax_acc.plot(history["epoch"], [a * 100 for a in history["train_acc"]],
                color=theme.DEEP_RED, lw=1.5, marker="o", label="train")
    ax_acc.plot(history["epoch"], [a * 100 for a in history["test_acc"]],
                color=theme.INK_BLACK, lw=1.5, marker="s", label="test")
    ax_acc.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    ax_acc.set_ylabel("DMTS accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax_acc.set_ylim(40, 102)
    ax_acc.set_title("DMTS accuracy", fontsize=theme.SIZE_LABEL)
    ax_acc.legend(fontsize=theme.SIZE_LEGEND, loc="lower right", frameon=False)
    ax_acc.grid(True, alpha=0.3)

    fig.suptitle(
        f"DMTS with trainable $W_{{ee}}$ + SGCC + NMDA on ping  "
        f"(T_s={T_SAMPLE_MS:.0f} / T_d={T_DELAY_MS:.0f} / T_p={T_PROBE_MS:.0f} ms)",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    return fig


def fig_example_trial(
    net, readout, X_te, y_te, test_a, test_b, test_lbl, device, run_id,
) -> plt.Figure:
    """Show one match and one non-match trial: input raster + E activity."""
    import models as M

    T_sample_steps = int(T_SAMPLE_MS / DT)
    T_delay_steps = int(T_DELAY_MS / DT)
    T_probe_steps = int(T_PROBE_MS / DT)
    T_total_steps = T_sample_steps + T_delay_steps + T_probe_steps

    match_idx = next((i for i in range(len(test_lbl)) if test_lbl[i] == 1), 0)
    nm_idx = next((i for i in range(len(test_lbl)) if test_lbl[i] == 0), 0)

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), dpi=150, sharex=True)
    for ax, idx, title in [
        (axes[0], match_idx, "match (label = 1)"),
        (axes[1], nm_idx,    "non-match (label = 0)"),
    ]:
        spk, lbl = build_dmts_batch(
            X_te, y_te, test_a[idx:idx+1], test_b[idx:idx+1], test_lbl[idx:idx+1],
            seed=12345, device=device,
        )
        with torch.no_grad():
            logits = forward_and_readout(net, readout, spk)
            pred = int(logits.argmax(dim=-1).item())
        e = net.spike_record["hid"]
        if e.dim() == 3:
            e = e[:, 0, :]
        e = e.cpu().numpy()

        # Subsample E neurons for visibility
        rng = np.random.default_rng(0)
        e_pick = rng.choice(N_E, size=min(80, N_E), replace=False)
        e_pick.sort()
        e_sub = e[:, e_pick]
        t_idx, n_idx = np.where(e_sub > 0)
        ax.scatter(t_idx * DT, n_idx, s=2.0, c=theme.INK_BLACK, marker="|",
                   linewidths=0.5)
        ax.axvspan(0, T_SAMPLE_MS, color=theme.DEEP_RED, alpha=0.06,
                   label="sample")
        ax.axvspan(T_SAMPLE_MS, T_SAMPLE_MS + T_DELAY_MS,
                   color=theme.GREY_LIGHT, alpha=0.25, label="delay")
        ax.axvspan(T_SAMPLE_MS + T_DELAY_MS, T_TOTAL_MS,
                   color=theme.DEEP_RED, alpha=0.06, label="probe")
        digit_a = int(y_te[test_a[idx]])
        digit_b = int(y_te[test_b[idx]])
        correct = "✓" if pred == int(test_lbl[idx]) else "✗"
        ax.set_title(
            f"{title}  —  digit_A={digit_a}, digit_B={digit_b}  "
            f"→  pred={pred} ({correct})",
            fontsize=theme.SIZE_LABEL, loc="left",
        )
        ax.set_ylim(-1, len(e_pick))
        ax.set_yticks([])
        ax.set_ylabel("80 E cells", fontsize=theme.SIZE_ANNOTATION)
    axes[-1].set_xlabel("Time (ms)", fontsize=theme.SIZE_LABEL)
    axes[0].legend(fontsize=theme.SIZE_LEGEND, loc="upper right",
                   frameon=False, ncol=3)
    fig.suptitle(
        "Example test trials — input phases shaded",
        fontsize=theme.SIZE_TITLE, y=0.99,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    return fig


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    tier = parse_tier(sys.argv, choices=list(TIER_CONFIG), default=DEFAULT_TIER)
    tier_cfg = TIER_CONFIG[tier]

    if "--no-wipe-dir" not in sys.argv:
        if ARTIFACTS.exists():
            shutil.rmtree(ARTIFACTS)
        if FIGURES.exists():
            shutil.rmtree(FIGURES)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    run_id = next_run_id(SLUG)
    persist_run_id(SLUG, run_id)
    (FIGURES / "_run.txt").write_text(f"run_id: {run_id}\ntier: {tier}\n")

    # Force CPU. MPS hits dtype/device-mismatch in the recurrent recording
    # path; this notebook is small-enough that the CPU run is fast.
    device = torch.device("cpu")

    t0 = time.time()
    print(f"[{SLUG}] tier={tier}  device={device}  "
          f"n_train={tier_cfg['n_train']}  epochs={tier_cfg['epochs']}")
    print(f"  T_s/T_d/T_p = {T_SAMPLE_MS:.0f}/{T_DELAY_MS:.0f}/{T_PROBE_MS:.0f} ms, "
          f"dt = {DT} ms")
    print(f"  net: ping, trainable_w_ee, W_ee init=({W_EE_INIT_MEAN}, {W_EE_INIT_STD}), "
          f"slow-syn gain={SLOW_SYN_GAIN}, sgcc α={SGCC_ALPHA}")

    net, readout, history, test_pkg = train_dmts(tier_cfg, device, run_id)
    runtime = time.time() - t0

    fig1 = fig_training_curve(history, run_id)
    fig1.savefig(FIGURES / "training_curve.png", dpi=150)
    plt.close(fig1)

    fig2 = fig_example_trial(net, readout, *test_pkg, device, run_id)
    fig2.savefig(FIGURES / "example_trials.png", dpi=150)
    plt.close(fig2)

    best_test = max(history["test_acc"])
    final_test = history["test_acc"][-1]
    final_train = history["train_acc"][-1]

    numbers = {
        "run_id": run_id,
        "config": {
            "dt": DT,
            "t_sample_ms": T_SAMPLE_MS,
            "t_delay_ms": T_DELAY_MS,
            "t_probe_ms": T_PROBE_MS,
            "n_e": N_E,
            "n_in": N_IN,
            "ei_strength": EI_STRENGTH,
            "w_ee_init_mean": W_EE_INIT_MEAN,
            "w_ee_init_std": W_EE_INIT_STD,
            "w_in_mean": W_IN_MEAN,
            "slow_syn_gain": SLOW_SYN_GAIN,
            "sgcc_alpha": SGCC_ALPHA,
            "stim_peak_rate_hz": STIM_PEAK_RATE_HZ,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "n_train": tier_cfg["n_train"],
            "n_test": tier_cfg["n_test"],
            "epochs": tier_cfg["epochs"],
            "tier": tier,
        },
        "results": {
            "best_test_acc": best_test,
            "final_test_acc": final_test,
            "final_train_acc": final_train,
            "history": history,
        },
        "success_criteria": [
            {
                "label": "test acc above chance + margin (≥ 60%)",
                "passed": best_test >= 0.60,
                "detail": f"best test acc = {best_test*100:.1f}%",
            },
            {
                "label": "training acc above chance (≥ 55%) — loss has signal",
                "passed": final_train >= 0.55,
                "detail": f"final train acc = {final_train*100:.1f}%",
            },
        ],
        "runtime_s": runtime,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print(f"[{SLUG}] done in {runtime:.1f}s. "
          f"best_test={best_test*100:.1f}%  final_test={final_test*100:.1f}%  "
          f"final_train={final_train*100:.1f}%")

    if not all(c["passed"] for c in numbers["success_criteria"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
