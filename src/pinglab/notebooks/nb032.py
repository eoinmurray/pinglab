"""Notebook runner for entry 032 — N-way recall with trainable W_ee + SGCC + NMDA.

Reformulates the working-memory task to match the LSNN paper's actual
problem rather than the pathological binary-CE version we tried first.
Same architecture (ping COBANet, trainable W_ee, slow-syn, SGCC), but
task is now:

  *  sample phase [0, T_s):           MNIST digit A as Poisson spikes.
  *  delay  phase [T_s, T_s+T_d):     silence — network must hold class.
  *  recall phase [T_s+T_d, T):       silence — readout integrates E activity.
  *  target label: digit_A's class (10-way).

Loss is 10-class cross-entropy on the recall-window mean E spike rate
fed through a Linear(N_E → 10) head. The per-sample gradient on the
readout has directional variance across classes — no balanced-batch
cancellation, no chance-basin trap. This is the same loss structure that
let nb027 (regular MNIST classification with the same coba+SGCC stack)
train cleanly to 87% test accuracy.

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
DT = 1.0             # ms — 400-step BPTT graph at this trial length is the
                     # sweet spot. dt=2.0 (halving steps) and longer trials at
                     # dt=1.0 both hurt; baseline tested empirically optimal.
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

# ── N-way recall task ─────────────────────────────────────────────────
T_SAMPLE_MS = 100.0   # input on (digit_A's Poisson encoding)
T_DELAY_MS = 200.0    # input off — pure memory window
T_RECALL_MS = 100.0   # input off — readout integrates E activity here
T_TOTAL_MS = T_SAMPLE_MS + T_DELAY_MS + T_RECALL_MS  # 400 ms
# Both extending T_s (→ 200) and T_r (→ 300) were tried; each made
# accuracy worse roughly linearly with added timesteps (~5 pp per
# +100 ms of BPTT depth at the medium tier). BPTT depth is the
# dominant ceiling here, not signal/noise on either window.
STIM_PEAK_RATE_HZ = 50.0   # peak Poisson rate per pixel during sample window
N_CLASSES = 10              # MNIST digit classes
N_OUT = N_CLASSES

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


def _sample_indices(y: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Pick n random sample indices balanced across the N_CLASSES digit classes."""
    rng = np.random.default_rng(seed)
    by_class = {c: np.where(y == c)[0] for c in range(N_CLASSES)}
    per_class = n // N_CLASSES
    idx = []
    labels = []
    for c in range(N_CLASSES):
        chosen = rng.choice(by_class[c], size=per_class, replace=False)
        idx.append(chosen)
        labels.append(np.full(per_class, c, dtype=np.int64))
    idx_arr = np.concatenate(idx)
    lbl_arr = np.concatenate(labels)
    order = rng.permutation(len(idx_arr))
    return idx_arr[order], lbl_arr[order]


def _encode_recall_spikes(
    pixel: np.ndarray,
    T_sample_steps: int,
    T_silent_steps: int,
    seed: int,
) -> np.ndarray:
    """One trial's (T_total, N_IN) input: digit during sample, silence after."""
    from cli.encoders import encode_image_spikes

    sample = encode_image_spikes(
        pixel, T_sample_steps, DT,
        base_rate=0.0, stim_rate=STIM_PEAK_RATE_HZ,
        step_on_ms=0.0, step_off_ms=T_SAMPLE_MS,
        seed=seed,
    ).numpy()
    silent = np.zeros((T_silent_steps, N_IN), dtype=np.float32)
    return np.concatenate([sample, silent], axis=0)


def build_recall_batch(
    X: np.ndarray, idx: np.ndarray, labels: np.ndarray,
    seed: int, device: torch.device,
):
    """Materialise a batch of (input_spikes, labels) for one training step."""
    T_sample_steps = int(T_SAMPLE_MS / DT)
    T_silent_steps = int((T_DELAY_MS + T_RECALL_MS) / DT)
    T_total_steps = T_sample_steps + T_silent_steps
    B = len(idx)
    spk = np.zeros((T_total_steps, B, N_IN), dtype=np.float32)
    for j in range(B):
        spk[:, j, :] = _encode_recall_spikes(
            X[idx[j]], T_sample_steps, T_silent_steps, seed=seed + j,
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


class RecallReadout(nn.Module):
    """Linear(N_E → N_CLASSES) on recall-window mean E spike count.

    Bypasses the built-in mem-mean readout — we only want what's left in
    the network after the delay window to drive the loss. Integrating
    over the sample window would let the readout solve the task by
    looking at the input directly, which isn't the point.
    """

    def __init__(self, n_e: int, n_out: int):
        super().__init__()
        self.proj = nn.Linear(n_e, n_out)
        nn.init.kaiming_uniform_(self.proj.weight, a=5.0 ** 0.5)
        nn.init.zeros_(self.proj.bias)

    def forward(self, e_spikes_recall: torch.Tensor) -> torch.Tensor:
        rate = e_spikes_recall.mean(dim=0)
        return self.proj(rate)


def forward_and_readout(
    net: nn.Module, readout: RecallReadout,
    input_spikes: torch.Tensor,
) -> torch.Tensor:
    """Run network, pull recall-window E activity, apply readout → logits."""
    import models as M

    T_sample_steps = int(T_SAMPLE_MS / DT)
    T_delay_steps = int(T_DELAY_MS / DT)
    T_recall_steps = int(T_RECALL_MS / DT)
    T_total_steps = T_sample_steps + T_delay_steps + T_recall_steps
    M.T_steps = T_total_steps
    M.T_ms = T_TOTAL_MS

    net.recording = True
    _ = net(input_spikes=input_spikes)
    e_record = net.spike_record["hid"]
    if e_record.dim() == 2:
        e_record = e_record.unsqueeze(1)
    e_recall = e_record[T_sample_steps + T_delay_steps:]  # (T_recall, B, N_E)
    return readout(e_recall)


# ── Training ──────────────────────────────────────────────────────────
def evaluate(net, readout, X, idx, labels, device, batch_size=32):
    net.eval()
    readout.eval()
    n = len(idx)
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            spk, lbl = build_recall_batch(
                X, idx[start:end], labels[start:end],
                seed=999_000 + start, device=device,
            )
            logits = forward_and_readout(net, readout, spk)
            pred = logits.argmax(dim=-1)
            correct += (pred == lbl).sum().item()
            total += lbl.numel()
    return correct / total


def train_recall(tier_cfg, device, run_id):
    import models as M
    from cli import seed_everything

    seed_everything(SEED)
    X_tr, y_tr, X_te, y_te = load_mnist_split()
    n_train = tier_cfg["n_train"]
    n_test = tier_cfg["n_test"]
    epochs = tier_cfg["epochs"]

    train_idx, train_lbl = _sample_indices(y_tr, n_train, seed=SEED)
    test_idx, test_lbl = _sample_indices(y_te, n_test, seed=SEED + 1)

    net = build_dmts_net(device)
    net = net.to(device)
    readout = RecallReadout(N_E, N_OUT).to(device)
    optim = torch.optim.Adam(
        list(net.parameters()) + list(readout.parameters()),
        lr=LR,
    )

    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": []}
    # _sample_indices rounds down to N_CLASSES multiples, so use actual length
    n_train_actual = len(train_idx)
    n_batches = n_train_actual // BATCH_SIZE

    for ep in range(epochs):
        net.train()
        readout.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        perm = np.random.default_rng(SEED + ep + 1).permutation(n_train_actual)
        t_idx_ep = train_idx[perm]
        t_lbl_ep = train_lbl[perm]
        for bi in range(n_batches):
            start = bi * BATCH_SIZE
            end = start + BATCH_SIZE
            spk, lbl = build_recall_batch(
                X_tr, t_idx_ep[start:end], t_lbl_ep[start:end],
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
        test_acc = evaluate(net, readout, X_te, test_idx, test_lbl,
                            device, batch_size=BATCH_SIZE)
        avg_loss = epoch_loss / max(1, epoch_total)
        history["epoch"].append(ep + 1)
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        print(f"  ep {ep+1:>2}/{epochs}  loss={avg_loss:.4f}  "
              f"train_acc={train_acc*100:5.1f}%  test_acc={test_acc*100:5.1f}%")

    return net, readout, history, (X_te, y_te, test_idx, test_lbl)


# ── Figures ───────────────────────────────────────────────────────────
def fig_training_curve(history: dict, run_id: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    ax_loss, ax_acc = axes
    ax_loss.plot(history["epoch"], history["train_loss"],
                 color=theme.DEEP_RED, lw=1.5, marker="o")
    ax_loss.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    ax_loss.set_ylabel("Train loss (CE)", fontsize=theme.SIZE_LABEL)
    ax_loss.set_title("Training loss", fontsize=theme.SIZE_LABEL)
    ax_loss.grid(True, alpha=0.3)

    chance_pct = 100.0 / N_CLASSES
    ax_acc.axhline(chance_pct, color=theme.MUTED, lw=0.8, ls="--",
                   label=f"chance ({chance_pct:.0f}%)")
    ax_acc.plot(history["epoch"], [a * 100 for a in history["train_acc"]],
                color=theme.DEEP_RED, lw=1.5, marker="o", label="train")
    ax_acc.plot(history["epoch"], [a * 100 for a in history["test_acc"]],
                color=theme.INK_BLACK, lw=1.5, marker="s", label="test")
    ax_acc.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    ax_acc.set_ylabel(f"{N_CLASSES}-way recall accuracy (%)",
                      fontsize=theme.SIZE_LABEL)
    ax_acc.set_ylim(0, 102)
    ax_acc.set_title("Recall accuracy", fontsize=theme.SIZE_LABEL)
    ax_acc.legend(fontsize=theme.SIZE_LEGEND, loc="lower right", frameon=False)
    ax_acc.grid(True, alpha=0.3)

    fig.suptitle(
        f"N-way recall with trainable $W_{{ee}}$ + SGCC + NMDA on ping  "
        f"(T_s={T_SAMPLE_MS:.0f} / T_d={T_DELAY_MS:.0f} / T_r={T_RECALL_MS:.0f} ms)",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    return fig


def fig_example_trial(
    net, readout, X_te, y_te, test_idx, test_lbl, device, run_id,
) -> plt.Figure:
    """Show two random test trials: input phases + E activity."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), dpi=150, sharex=True)

    # Pick two trials from different classes
    chosen = []
    seen_classes = set()
    for i in range(len(test_lbl)):
        c = int(test_lbl[i])
        if c not in seen_classes:
            chosen.append(i)
            seen_classes.add(c)
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        chosen = list(range(min(2, len(test_lbl))))

    for ax, idx in zip(axes, chosen):
        spk, lbl = build_recall_batch(
            X_te, test_idx[idx:idx+1], test_lbl[idx:idx+1],
            seed=12345 + idx, device=device,
        )
        with torch.no_grad():
            logits = forward_and_readout(net, readout, spk)
            pred = int(logits.argmax(dim=-1).item())
        e = net.spike_record["hid"]
        if e.dim() == 3:
            e = e[:, 0, :]
        e = e.cpu().numpy()

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
                   color=theme.DEEP_RED, alpha=0.06, label="recall")
        true_c = int(test_lbl[idx])
        correct = "✓" if pred == true_c else "✗"
        ax.set_title(
            f"digit = {true_c}  →  pred = {pred} ({correct})",
            fontsize=theme.SIZE_LABEL, loc="left",
        )
        ax.set_ylim(-1, len(e_pick))
        ax.set_yticks([])
        ax.set_ylabel("80 E cells", fontsize=theme.SIZE_ANNOTATION)
    axes[-1].set_xlabel("Time (ms)", fontsize=theme.SIZE_LABEL)
    axes[0].legend(fontsize=theme.SIZE_LEGEND, loc="upper right",
                   frameon=False, ncol=3)
    fig.suptitle(
        "Example test trials — sample / delay / recall windows shaded",
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
    print(f"  T_s/T_d/T_r = {T_SAMPLE_MS:.0f}/{T_DELAY_MS:.0f}/{T_RECALL_MS:.0f} ms, "
          f"dt = {DT} ms")
    print(f"  net: ping, trainable_w_ee, W_ee init=({W_EE_INIT_MEAN}, {W_EE_INIT_STD}), "
          f"slow-syn gain={SLOW_SYN_GAIN}, sgcc α={SGCC_ALPHA}")

    net, readout, history, test_pkg = train_recall(tier_cfg, device, run_id)
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
            "t_recall_ms": T_RECALL_MS,
            "n_classes": N_CLASSES,
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
                "label": f"test acc above chance ({100/N_CLASSES:.0f}%) by ≥ 10 pp",
                "passed": best_test >= (1.0 / N_CLASSES + 0.10),
                "detail": f"best test acc = {best_test*100:.1f}%",
            },
            {
                "label": "training acc above chance — loss has signal",
                "passed": final_train >= (1.0 / N_CLASSES + 0.05),
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
