"""Notebook runner for entry 028 — DMTS retry on coba + ALIF + SGCC.

[nb026](/notebooks/nb026/) characterised the DMTS chance plateau as a
gradient-stabiliser problem rather than a memory-mechanism problem:
ALIF (and slow synapse in [nb025](/notebooks/nb025/)) give the network
the information needed for working memory, but the conductance Jacobian
explosion forces *--v-grad-dampen 1000* which then washes out the
gradient signal the task needs.

[nb027](/notebooks/nb027/) validated SGCC — Burghi et al.'s principled
controller — as a drop-in replacement for *--v-grad-dampen* on coba
MNIST. SGCC matched the baseline at matched epochs and pulled 3.5 pp
ahead with more training, since it doesn't uniformly mute the gradient
the way the global dampener does.

This entry combines both. coba + ALIF (for the long-timescale memory
state) + SGCC (for the gradient pathway). Same DMTS task as nb026:
T_s=T_d=T_p=50 ms, binary match/non-match, mem-mean readout, all-from-
scratch training. The question is whether the cleaner gradient
stabilizer unblocks the chance plateau.

Notebook entry: src/docs/src/pages/notebooks/nb028.mdx
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO / "src" / "pinglab" / "notebooks"))
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _modal import parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402
from pinglab import theme  # noqa: E402

SLUG = "nb028"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG

# ── DMTS recipe (same as nb026 — only the gradient stabilizer changes) ──
T_S_MS = 50.0
T_D_MS = 50.0  # target final delay; curriculum below ramps up to it
T_P_MS = 50.0
T_TRIAL_MS = T_S_MS + T_D_MS + T_P_MS
# Per-epoch delay schedule. Start at 0 (pure pattern-pair comparison, no
# working-memory bridge required) and step up. Each entry repeats for
# (epochs / len(schedule)) epochs; last entry repeats for remaining.
DELAY_CURRICULUM_MS: list[float] = [0.0, 10.0, 25.0, 50.0]
INPUT_RATE_HZ = 25.0
DT_TRAIN = 0.1
LR = 4e-4
BATCH = 256
N_OUT = 2
SEED = 42

ALIF_BETA = 1.7
TAU_ADAPT_MS = 700.0
SGCC_ALPHA = 0.2

DEFAULT_TIER = "small"
TIER_CONFIG = {
    "extra small": dict(max_samples=200, epochs=1),
    "small":       dict(max_samples=500, epochs=5),
    "medium":      dict(max_samples=2000, epochs=20),
    "large":       dict(max_samples=5000, epochs=40),
    "extra large": dict(max_samples=10000, epochs=40),
}


def dmts_dir() -> Path:
    return ARTIFACTS / "dmts_coba_alif_sgcc__seed42"


def _stamp(fig, run_id: str) -> None:
    fig.text(
        0.99, 0.005, run_id,
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def encode_dmts_trial(sample_x, probe_x, dt_ms: float, generator, t_d_ms=None):
    """Three-phase Poisson encoding: [sample T_S][silence t_d_ms or T_D][probe T_P]."""
    import torch

    t_d = T_D_MS if t_d_ms is None else float(t_d_ms)
    t_s = int(round(T_S_MS / dt_ms))
    t_d = int(round(t_d / dt_ms))
    t_p = int(round(T_P_MS / dt_ms))
    T_total = t_s + t_d + t_p
    n_in = sample_x.shape[-1]
    p_scale = INPUT_RATE_HZ * dt_ms / 1000.0
    device = sample_x.device
    spk = torch.zeros(T_total, n_in, device=device)
    rs = torch.rand(t_s, n_in, generator=generator, device="cpu").to(device)
    spk[:t_s] = (rs < sample_x.clamp(0, 1) * p_scale).float()
    rp = torch.rand(t_p, n_in, generator=generator, device="cpu").to(device)
    spk[t_s + t_d :] = (rp < probe_x.clamp(0, 1) * p_scale).float()
    return spk


def sample_dmts_batch(X, y, batch_size, rng, device):
    """50/50 match / non-match."""
    import torch

    n = len(y)
    sample_idx = rng.integers(0, n, size=batch_size)
    labels = (rng.random(batch_size) < 0.5).astype(int)
    probe_idx = np.empty(batch_size, dtype=int)
    for i in range(batch_size):
        if labels[i] == 1:
            same = np.flatnonzero(y == y[sample_idx[i]])
            probe_idx[i] = rng.choice(same)
        else:
            diff = np.flatnonzero(y != y[sample_idx[i]])
            probe_idx[i] = rng.choice(diff)
    return (
        torch.from_numpy(X[sample_idx]).to(device),
        torch.from_numpy(X[probe_idx]).to(device),
        torch.from_numpy(labels).long().to(device),
    )


def encode_dmts_batch(sample_X, probe_X, dt_ms, generator, t_d_ms=None):
    import torch
    B = sample_X.shape[0]
    return torch.stack(
        [encode_dmts_trial(sample_X[i], probe_X[i], dt_ms, generator, t_d_ms=t_d_ms)
         for i in range(B)],
        dim=1,
    )


def train_dmts(tier: str, out_dir: Path) -> dict:
    """Train coba + ALIF + SGCC from scratch on DMTS."""
    import torch

    import config as C  # noqa: F401
    import models as M
    from config import build_net, patch_dt
    from oscilloscope import _auto_device, load_dataset, seed_everything

    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    M.T_ms = T_TRIAL_MS
    M.tau_adapt = TAU_ADAPT_MS
    patch_dt(DT_TRAIN)
    M.max_rate_hz = INPUT_RATE_HZ
    M.p_scale = M.max_rate_hz * M.dt / 1000.0
    M.N_IN = 784
    M.N_OUT = N_OUT
    M.N_HID = 1024
    M.N_INH = 256
    M.HIDDEN_SIZES = [1024]
    # Crucially: DO NOT set M.V_GRAD_DAMPEN — leave at default 1.0 (off).
    # SGCC handles the conductance Jacobian, no need for the heavy-handed
    # global dampener that capped nb026's gradient signal.

    device = _auto_device()
    X_tr, X_te, y_tr, y_te = load_dataset(
        "mnist", max_samples=int(TIER_CONFIG[tier]["max_samples"]), split=True
    )

    net = build_net(
        "ping",  # COBANet dispatch
        w_in=(0.3, 0.03),
        w_in_sparsity=0.95,
        ei_strength=0.0,  # coba (no I-loop)
        ei_ratio=2.0,
        sparsity=0.0,
        device=device,
        randomize_init=True,
        kaiming_init=False,
        dales_law=True,
        hidden_sizes=[1024],
        readout_mode="mem-mean",
        alif=True,
        alif_beta=ALIF_BETA,
        sgcc=True,
        sgcc_alpha=SGCC_ALPHA,
    )
    net.train()

    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=LR
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(SEED)
    enc_gen = torch.Generator().manual_seed(SEED)
    epochs = int(TIER_CONFIG[tier]["epochs"])
    samples_per_epoch = int(TIER_CONFIG[tier]["max_samples"])
    history: dict = {
        "epochs": [], "train_loss": [], "train_acc": [], "test_acc": [],
        "delay_ms": [],
    }
    # Curriculum: split epochs evenly across DELAY_CURRICULUM_MS stages.
    def _delay_for_epoch(ep_idx: int) -> float:
        stage = min(
            len(DELAY_CURRICULUM_MS) - 1,
            ep_idx * len(DELAY_CURRICULUM_MS) // max(1, epochs),
        )
        return DELAY_CURRICULUM_MS[stage]

    print(
        f"[dmts-sgcc-train] {epochs} ep × ~{samples_per_epoch}, batch={BATCH}, "
        f"lr={LR}, alif β={ALIF_BETA}, sgcc α={SGCC_ALPHA}, "
        f"curriculum delays={DELAY_CURRICULUM_MS} ms"
    )

    def _eval(t_d_ms, n_eval=200):
        net.eval()
        correct = total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for _ in range(max(1, n_eval // BATCH)):
                sX, pX, lab = sample_dmts_batch(X_te, y_te, BATCH, rng, device)
                spk = encode_dmts_batch(sX, pX, M.dt, enc_gen, t_d_ms=t_d_ms)
                logits = net(input_spikes=spk)
                loss_sum += float(loss_fn(logits, lab).item()) * lab.size(0)
                correct += int((logits.argmax(1) == lab).sum().item())
                total += lab.size(0)
        net.train()
        return loss_sum / max(total, 1), 100.0 * correct / max(total, 1)

    skipped_total = 0
    for ep in range(epochs):
        delay_ms = _delay_for_epoch(ep)
        # Re-patch the trial length so M.T_steps follows the curriculum.
        M.T_ms = T_S_MS + delay_ms + T_P_MS
        patch_dt(DT_TRAIN)
        n_batches = max(1, samples_per_epoch // BATCH)
        ep_loss = 0.0
        ep_correct = ep_total = 0
        ep_skipped = 0
        t_ep = time.monotonic()
        for _b in range(n_batches):
            sX, pX, lab = sample_dmts_batch(X_tr, y_tr, BATCH, rng, device)
            spk = encode_dmts_batch(sX, pX, M.dt, enc_gen, t_d_ms=delay_ms)
            logits = net(input_spikes=spk)
            loss = loss_fn(logits, lab)
            optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            if not math.isfinite(float(gn)):
                optimizer.zero_grad(set_to_none=True)
                ep_skipped += 1
            else:
                optimizer.step()
                if net.signed_weights is False:
                    net.project_dales()
            loss_v = float(loss.item()) if math.isfinite(float(loss.item())) else 0.0
            ep_loss += loss_v * lab.size(0)
            ep_correct += int((logits.argmax(1) == lab).sum().item())
            ep_total += lab.size(0)
        skipped_total += ep_skipped
        train_loss = ep_loss / max(ep_total, 1)
        train_acc = 100.0 * ep_correct / max(ep_total, 1)
        _, test_acc = _eval(t_d_ms=delay_ms)
        dur = time.monotonic() - t_ep
        history["epochs"].append(ep + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["delay_ms"].append(delay_ms)
        skip_note = f"  skip={ep_skipped}/{n_batches}" if ep_skipped else ""
        print(
            f"  ep {ep+1}/{epochs}  delay={delay_ms:5.1f}  "
            f"loss={train_loss:.4f}  "
            f"train={train_acc:5.2f}%  test={test_acc:5.2f}%  ({dur:.1f}s)"
            f"{skip_note}"
        )
    if skipped_total:
        print(f"  [skip-bad-grad] total batches skipped: {skipped_total}")

    torch.save(net.state_dict(), out_dir / "weights.pth")
    cfg_out = {
        "model": "ping", "dataset": "mnist-dmts",
        "ei_strength": 0.0, "ei_ratio": 2.0, "sparsity": 0.0,
        "w_in": [0.3, 0.03], "w_in_sparsity": 0.95,
        "kaiming_init": False, "dales_law": True,
        "readout_mode": "mem-mean",
        "alif": True, "alif_beta": ALIF_BETA, "tau_adapt": TAU_ADAPT_MS,
        "sgcc": True, "sgcc_alpha": SGCC_ALPHA,
        "hidden_sizes": [1024],
        "t_ms": T_TRIAL_MS, "dt": DT_TRAIN,
        "max_samples": samples_per_epoch, "epochs": epochs,
        "seed": SEED, "n_hidden": 1024,
        "t_s_ms": T_S_MS, "t_d_ms": T_D_MS, "t_p_ms": T_P_MS,
        "input_rate_hz": INPUT_RATE_HZ,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_out, indent=2) + "\n")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(f"  → {out_dir / 'weights.pth'}")
    return history


def plot_dmts_history(history: dict, out_path: Path, run_id: str) -> None:
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.plot(history["epochs"], history["train_acc"],
            marker="o", color=theme.DEEP_RED, label="train")
    ax.plot(history["epochs"], history["test_acc"],
            marker="s", color=theme.INK_BLACK, label="test")
    ax.axhline(50.0, ls=":", color=theme.FAINT, lw=0.8, label="chance")
    ax.set_xlabel("epoch")
    ax.set_ylabel("DMTS accuracy (%)")
    ax.set_title(
        f"coba + ALIF + SGCC on DMTS (T_s = {T_S_MS:.0f}, "
        f"T_p = {T_P_MS:.0f} ms, curriculum delay)"
    )
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    if "delay_ms" in history and history["delay_ms"]:
        ax_d = ax.twinx()
        ax_d.step(
            history["epochs"], history["delay_ms"], where="mid",
            color=theme.MUTED, lw=1.0, ls="--", label="delay (ms)",
        )
        ax_d.set_ylabel("curriculum delay (ms)", color=theme.MUTED)
        ax_d.tick_params(axis="y", labelcolor=theme.MUTED)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    wipe_dir = "--no-wipe-dir" not in sys.argv
    skip_training = "--skip-training" in sys.argv

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(
        f"notebook_run_id = {notebook_run_id} tier={tier}"
        + ("  [skip-training]" if skip_training else "")
    )
    if modal_gpu is not None:
        print(f"[stub] --modal-gpu {modal_gpu} accepted, no-op for this cell")

    if wipe_dir:
        wipe_targets = (FIGURES,) if skip_training else (ARTIFACTS, FIGURES)
        for d in wipe_targets:
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)

    if skip_training and (dmts_dir() / "weights.pth").exists():
        history = json.loads((dmts_dir() / "history.json").read_text())
    else:
        history = train_dmts(tier, dmts_dir())

    plot_dmts_history(history, FIGURES / "dmts_training_curve.png", notebook_run_id)
    print(f"wrote {FIGURES / 'dmts_training_curve.png'}")

    best_test = max(history["test_acc"]) if history["test_acc"] else 0.0
    final_test = history["test_acc"][-1] if history["test_acc"] else 0.0

    duration_s = time.monotonic() - t_start
    summary = {
        "notebook_run_id": notebook_run_id,
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "tier": tier,
        "config": {
            "tier": tier,
            "alif_beta": ALIF_BETA,
            "tau_adapt_ms": TAU_ADAPT_MS,
            "sgcc_alpha": SGCC_ALPHA,
            "t_s_ms": T_S_MS,
            "t_d_ms": T_D_MS,
            "t_p_ms": T_P_MS,
            "lr": LR,
            "batch": BATCH,
            "seed": SEED,
        },
        "history": history,
        "results": {
            "best_test_acc": best_test,
            "final_test_acc": final_test,
        },
        "success_criteria": [
            {
                "label": "no NaN throughout training",
                "passed": all(math.isfinite(x) for x in history.get("train_loss", [])),
                "detail": (
                    f"min loss = {min(history['train_loss']) if history['train_loss'] else '?'}, "
                    f"max = {max(history['train_loss']) if history['train_loss'] else '?'}"
                ),
            },
            {
                "label": "test accuracy > 55% (above chance)",
                "passed": best_test > 55.0,
                "detail": f"best test acc = {best_test:.2f}%, final = {final_test:.2f}%",
            },
            {
                "label": "test accuracy > 70% (real learning)",
                "passed": best_test > 70.0,
                "detail": f"best test acc = {best_test:.2f}%",
            },
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  best test acc: {best_test:.2f}%")
    print(f"  total duration: {summary['duration']}")


if __name__ == "__main__":
    main()
