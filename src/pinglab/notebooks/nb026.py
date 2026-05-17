"""Notebook runner for entry 026 — adaptive LIF (ALIF) on coba/ping,
and a DMTS retry on top of it.

The slow-synapse work in nb025 gave the network a long-timescale memory
substrate on the input side, but coba+slow-syn still couldn't escape
the DMTS chance plateau — the gradient pathology was orthogonal to the
information pathology. ALIF is the LSNN-style alternative: a per-neuron
slow adaptation variable that rises with recent firing and raises that
cell's own threshold. Different memory mechanism (output side, not
input side), different gradient geometry, much stronger published
evidence of trainability on DMTS-like tasks.

Cells:
    1. Train coba and ping from scratch with --alif (recipes mirror
       nb024 / nb025 otherwise). Replay each at α-LIF off vs on.
    2. DMTS: train coba+ALIF from scratch on a sample/delay/probe
       task. The question is whether the LSNN-style memory makes the
       binary classification trainable where slow-syn couldn't.

Notebook entry: src/docs/src/pages/notebooks/nb026.mdx
"""

from __future__ import annotations

import json
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

SLUG = "nb026"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope/__main__.py"

# Pretrained baselines from nb024 (no ALIF at training time) — re-used
# for the parity raster comparison.
NB024_ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / "nb024"
PRETRAINED = {
    "coba": NB024_ARTIFACTS / "coba__off__seed42",
    "ping": NB024_ARTIFACTS / "ping__off__seed42",
}

# ── Recipe ────────────────────────────────────────────────────────────
T_MS_TRAIN = 200.0
DT_TRAIN = 0.1
T_MS_TRIAL = 400.0
SAMPLE_IDX = 0
ALIF_BETA = 1.7
TAU_ADAPT_MS = 700.0
SEED = 42
RASTER_N_E_PLOT = 200
RASTER_N_I_PLOT = 64

DEFAULT_TIER = "small"
TIER_CONFIG = {
    "extra small": dict(max_samples=200, epochs=1),
    "small":       dict(max_samples=500, epochs=5),
    "medium":      dict(max_samples=2000, epochs=20),
    "large":       dict(max_samples=5000, epochs=40),
    "extra large": dict(max_samples=10000, epochs=40),
}
MODELS = ["coba", "ping"]

ALIF_TRAIN_RECIPES: dict[str, dict] = {
    "coba": {
        "--ei-strength": "0",
        "--v-grad-dampen": "1000",
        "--w-in": "0.3",
        "--w-in-sparsity": "0.95",
        "--readout": "mem-mean",
        "--surrogate-slope": "1",
        "--readout-w-out-scale": "100",
        "--lr": "0.0004",
        "--batch-size": "256",
    },
    "ping": {
        "--ei-strength": "1",
        "--v-grad-dampen": "1000",
        "--w-in": "1.2",
        "--w-in-sparsity": "0.95",
        "--readout": "mem-mean",
        "--surrogate-slope": "1",
        "--readout-w-out-scale": "500",
        "--lr": "0.0004",
        "--batch-size": "256",
    },
}


def alif_train_dir(model: str) -> Path:
    return ARTIFACTS / f"{model}_alif__seed{SEED}"


# ── DMTS task constants (Cell 2) ─────────────────────────────────────
DMTS_T_S_MS = 50.0
DMTS_T_D_MS = 50.0
DMTS_T_P_MS = 50.0
DMTS_T_TRIAL_MS = DMTS_T_S_MS + DMTS_T_D_MS + DMTS_T_P_MS
DMTS_INPUT_RATE_HZ = 25.0
DMTS_LR = 1e-3
DMTS_BATCH = 256
DMTS_V_GRAD_DAMPEN = 1.0  # off — LSNN doesn't use this; rely on lower LR for stability
N_OUT_DMTS = 2


def dmts_dir() -> Path:
    return ARTIFACTS / "dmts_coba_alif__seed42"


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


def train_with_alif(model: str, tier: str) -> Path:
    """Train one coba or ping cell with --alif enabled."""
    import subprocess

    out_dir = alif_train_dir(model)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    recipe = ALIF_TRAIN_RECIPES[model]
    args = [
        "train",
        "--model", "ping",
        "--dataset", "mnist",
        "--max-samples", str(TIER_CONFIG[tier]["max_samples"]),
        "--epochs", str(TIER_CONFIG[tier]["epochs"]),
        "--t-ms", str(T_MS_TRAIN),
        "--dt", str(DT_TRAIN),
        "--seed", str(SEED),
        "--observe", "video",
        "--frame-rate", "1",
        "--out-dir", str(out_dir),
        "--wipe-dir",
        "--alif",
        "--alif-beta", str(ALIF_BETA),
        "--tau-adapt", str(TAU_ADAPT_MS),
    ]
    for k, v in recipe.items():
        if v is True:
            args.append(k)
        elif v is not None:
            args += [k, v]
    cmd = ["uv", "run", "python", str(OSCILLOSCOPE), *args]
    print(f"[train-alif] {model}: {' '.join(args)}")
    subprocess.run(cmd, cwd=REPO, check=True)
    return out_dir


def capture_raster(train_dir: Path, alif_beta: float) -> dict:
    """Forward pass on one MNIST trial with the chosen ALIF β."""
    import torch

    import config as C  # noqa: F401
    import models as M
    from config import build_net, patch_dt
    from oscilloscope import (
        EVAL_SEED,
        _auto_device,
        encode_batch,
        load_dataset,
        seed_everything,
    )

    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", SEED)))
    M.T_ms = float(T_MS_TRIAL)
    M.tau_adapt = TAU_ADAPT_MS
    patch_dt(float(cfg["dt"]))
    hidden_sizes = cfg.get("hidden_sizes") or [int(cfg["n_hidden"])]
    M.N_HID = hidden_sizes[-1]
    M.N_INH = hidden_sizes[-1] // 4
    M.HIDDEN_SIZES = list(hidden_sizes)

    device = _auto_device()
    _, X_te, _, y_te = load_dataset(
        cfg["dataset"], max_samples=int(cfg["max_samples"]), split=True
    )
    M.N_IN = 784 if cfg["dataset"] == "mnist" else 64

    w_in_cfg = cfg.get("w_in")
    w_in_arg = (
        (float(w_in_cfg[0]), float(w_in_cfg[1]))
        if isinstance(w_in_cfg, list) and len(w_in_cfg) >= 2
        else None
    )
    net = build_net(
        cfg["model"],
        w_in=w_in_arg,
        w_in_sparsity=float(cfg.get("w_in_sparsity") or 0.0),
        ei_strength=float(cfg.get("ei_strength") or 1.0),
        ei_ratio=float(cfg.get("ei_ratio") or 2.0),
        sparsity=float(cfg.get("sparsity") or 0.0),
        device=device,
        randomize_init=not bool(cfg.get("kaiming_init", False)),
        kaiming_init=bool(cfg.get("kaiming_init", False)),
        dales_law=bool(cfg.get("dales_law", True)),
        hidden_sizes=hidden_sizes,
        alif=True,
        alif_beta=alif_beta,
    )
    if hasattr(net, "readout_mode"):
        net.readout_mode = cfg.get("readout_mode", "mem-mean")

    state = torch.load(train_dir / "weights.pth", map_location=device)
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True

    X_b = torch.from_numpy(X_te[SAMPLE_IDX : SAMPLE_IDX + 1]).to(device)
    y_b = int(y_te[SAMPLE_IDX])
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    with torch.no_grad():
        spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
        _ = net(input_spikes=spk)

    e_full = net.spike_record["hid"].cpu().numpy()
    i_full = net.spike_record["inh"].cpu().numpy()
    dt_s = float(cfg["dt"]) / 1000.0
    e_rate = float(e_full.sum() / (e_full.shape[1] * e_full.shape[0] * dt_s))
    i_rate = float(i_full.sum() / (i_full.shape[1] * i_full.shape[0] * dt_s))
    return {
        "alif_beta": alif_beta,
        "label": y_b,
        "e": e_full,
        "i": i_full,
        "dt": float(cfg["dt"]),
        "t_total_ms": T_MS_TRIAL,
        "e_rate_hz": e_rate,
        "i_rate_hz": i_rate,
    }


def plot_alif_rasters(
    samples: dict[str, dict[float, dict]],
    out_path: Path,
    run_id: str,
) -> None:
    """2 cols (coba / ping) × 2 rows (ALIF off / on)."""
    theme.apply()
    n_e = RASTER_N_E_PLOT
    n_i = RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        2, 2, figsize=(10.0, 5.625), sharex=True,
        gridspec_kw={"hspace": 0.22, "wspace": 0.08,
                     "left": 0.14, "right": 0.97, "top": 0.92, "bottom": 0.12},
    )
    rng = np.random.default_rng(0)
    models = list(samples.keys())
    betas_in_order = [0.0, ALIF_BETA]
    row_labels = ["ALIF off", f"ALIF β = {ALIF_BETA:g}\n(τ = {TAU_ADAPT_MS:.0f} ms)"]

    for col, model in enumerate(models):
        for row, beta in enumerate(betas_in_order):
            ax = axes[row, col]
            s = samples[model][beta]
            T = s["e"].shape[0]
            e_full = s["e"][:, 0, :] if s["e"].ndim == 3 else s["e"]
            i_full = s["i"][:, 0, :] if s["i"].ndim == 3 else s["i"]
            e_idx = np.sort(rng.choice(e_full.shape[1], n_e, replace=False))
            i_idx = np.sort(rng.choice(i_full.shape[1], n_i, replace=False))
            t_axis = np.arange(T) * s["dt"]
            e_t, e_n = np.where(e_full[:, e_idx].astype(bool))
            i_t, i_n = np.where(i_full[:, i_idx].astype(bool))
            ax.scatter(t_axis[e_t], e_n, s=2.0, c=theme.INK_BLACK,
                       marker="|", linewidths=0.4)
            ax.scatter(t_axis[i_t], i_n + n_e + gap, s=2.0, c=theme.DEEP_RED,
                       marker="|", linewidths=0.4)
            ax.set_ylim(-2, n_e + n_i + gap + 2)
            if col == 0:
                ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
                ax.set_yticklabels(["E", "I"])
                ax.tick_params(axis="y", length=0)
            else:
                ax.yaxis.set_visible(False)
            ax.set_xlim(0, s["t_total_ms"])
            ax.text(
                0.985, 0.97,
                f"E = {s['e_rate_hz']:.1f} Hz   I = {s['i_rate_hz']:.1f} Hz",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=theme.SIZE_LABEL,
                bbox={"facecolor": theme.PAPER, "alpha": 0.85,
                      "edgecolor": "none", "pad": 2},
            )
            if row == 0:
                ax.set_title(model, fontsize=theme.SIZE_TITLE, pad=6)
            if col == 0:
                ax.text(
                    -0.08, 0.5, row_labels[row],
                    transform=ax.transAxes, ha="right", va="center",
                    fontsize=theme.SIZE_LABEL, color=theme.INK_BLACK,
                )
            if row == 1:
                ax.set_xlabel("time (ms)")
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── DMTS (Cell 2) ───────────────────────────────────────────────────


def encode_dmts_trial(sample_x, probe_x, dt_ms: float, generator):
    import torch

    t_s = int(round(DMTS_T_S_MS / dt_ms))
    t_d = int(round(DMTS_T_D_MS / dt_ms))
    t_p = int(round(DMTS_T_P_MS / dt_ms))
    T_total = t_s + t_d + t_p
    n_in = sample_x.shape[-1]
    p_scale = DMTS_INPUT_RATE_HZ * dt_ms / 1000.0
    device = sample_x.device
    spk = torch.zeros(T_total, n_in, device=device)
    rs = torch.rand(t_s, n_in, generator=generator, device="cpu").to(device)
    spk[:t_s] = (rs < sample_x.clamp(0, 1) * p_scale).float()
    rp = torch.rand(t_p, n_in, generator=generator, device="cpu").to(device)
    spk[t_s + t_d :] = (rp < probe_x.clamp(0, 1) * p_scale).float()
    return spk


def sample_dmts_batch(X, y, batch_size, rng, device):
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


def encode_dmts_batch(sample_X, probe_X, dt_ms, generator):
    import torch
    B = sample_X.shape[0]
    return torch.stack(
        [encode_dmts_trial(sample_X[i], probe_X[i], dt_ms, generator)
         for i in range(B)],
        dim=1,
    )


def train_dmts_alif(tier: str, out_dir: Path) -> dict:
    """Train coba+ALIF from scratch on DMTS."""
    import torch

    import config as C  # noqa: F401
    import models as M
    from config import build_net, patch_dt
    from oscilloscope import _auto_device, load_dataset, seed_everything

    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)
    M.T_ms = DMTS_T_TRIAL_MS
    M.tau_adapt = TAU_ADAPT_MS
    patch_dt(DT_TRAIN)
    M.max_rate_hz = DMTS_INPUT_RATE_HZ
    M.p_scale = M.max_rate_hz * M.dt / 1000.0
    M.N_IN = 784
    M.N_OUT = N_OUT_DMTS
    M.N_HID = 1024
    M.N_INH = 256
    M.HIDDEN_SIZES = [1024]
    # LSNN-style: no membrane gradient dampening. Bellec et al. don't
    # need it because their CUBA neurons don't have the conductance
    # Jacobian-explosion that nb024's recipe added the dampener for.
    # The question is whether the dampener was over-cautious for our
    # purposes — without it the gradient through the membrane is
    # ~1000× stronger.
    M.V_GRAD_DAMPEN = DMTS_V_GRAD_DAMPEN

    device = _auto_device()
    X_tr, X_te, y_tr, y_te = load_dataset(
        "mnist", max_samples=int(TIER_CONFIG[tier]["max_samples"]), split=True
    )
    net = build_net(
        "ping",
        w_in=(0.3, 0.03),
        w_in_sparsity=0.95,
        ei_strength=0.0,
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
    )
    net.train()

    optimizer = torch.optim.Adam(
        [p for p in net.parameters() if p.requires_grad], lr=DMTS_LR
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    rng = np.random.default_rng(SEED)
    enc_gen = torch.Generator().manual_seed(SEED)
    epochs = int(TIER_CONFIG[tier]["epochs"])
    samples_per_epoch = int(TIER_CONFIG[tier]["max_samples"])
    history: dict = {"epochs": [], "train_loss": [], "train_acc": [], "test_acc": []}
    print(
        f"[dmts-alif-train] {epochs} epochs × ~{samples_per_epoch} samples, "
        f"batch={DMTS_BATCH}, lr={DMTS_LR}, T={DMTS_T_TRIAL_MS:.0f}ms "
        f"(sample {DMTS_T_S_MS:.0f}+delay {DMTS_T_D_MS:.0f}+probe {DMTS_T_P_MS:.0f})"
    )

    def _eval(n_eval=200):
        net.eval()
        correct = total = 0
        loss_sum = 0.0
        with torch.no_grad():
            for _ in range(max(1, n_eval // DMTS_BATCH)):
                sX, pX, lab = sample_dmts_batch(X_te, y_te, DMTS_BATCH, rng, device)
                spk = encode_dmts_batch(sX, pX, M.dt, enc_gen)
                logits = net(input_spikes=spk)
                loss_sum += float(loss_fn(logits, lab).item()) * lab.size(0)
                correct += int((logits.argmax(1) == lab).sum().item())
                total += lab.size(0)
        net.train()
        return loss_sum / max(total, 1), 100.0 * correct / max(total, 1)

    import math
    skipped_total = 0
    for ep in range(epochs):
        n_batches = max(1, samples_per_epoch // DMTS_BATCH)
        ep_loss = 0.0
        ep_correct = ep_total = 0
        ep_skipped = 0
        t_ep = time.monotonic()
        for _b in range(n_batches):
            sX, pX, lab = sample_dmts_batch(X_tr, y_tr, DMTS_BATCH, rng, device)
            spk = encode_dmts_batch(sX, pX, M.dt, enc_gen)
            logits = net(input_spikes=spk)
            loss = loss_fn(logits, lab)
            optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            gn_f = float(gn)
            # Skip-bad-grad: drop the step if the batch's gradient is
            # non-finite or massively out of band. Prevents one exploded
            # batch from poisoning Adam's running moments and killing
            # the run — same pattern the oscilloscope's train() uses.
            bad_grad = not math.isfinite(gn_f)
            if bad_grad:
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
        _, test_acc = _eval()
        dur = time.monotonic() - t_ep
        history["epochs"].append(ep + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        print(
            f"  ep {ep+1}/{epochs}  loss={train_loss:.4f}  "
            f"train={train_acc:5.2f}%  test={test_acc:5.2f}%  "
            f"skip={ep_skipped}/{n_batches}  ({dur:.1f}s)"
        )
    if skipped_total > 0:
        print(f"  [skip-bad-grad] total batches skipped: {skipped_total}")

    torch.save(net.state_dict(), out_dir / "weights.pth")
    cfg_out = {
        "model": "ping", "dataset": "mnist-dmts",
        "ei_strength": 0.0, "ei_ratio": 2.0, "sparsity": 0.0,
        "w_in": [0.3, 0.03], "w_in_sparsity": 0.95,
        "kaiming_init": False, "dales_law": True,
        "readout_mode": "mem-mean",
        "alif": True, "alif_beta": ALIF_BETA, "tau_adapt": TAU_ADAPT_MS,
        "hidden_sizes": [1024],
        "t_ms": DMTS_T_TRIAL_MS, "dt": DT_TRAIN,
        "max_samples": samples_per_epoch, "epochs": epochs,
        "seed": SEED, "n_hidden": 1024,
        "t_s_ms": DMTS_T_S_MS, "t_d_ms": DMTS_T_D_MS, "t_p_ms": DMTS_T_P_MS,
        "input_rate_hz": DMTS_INPUT_RATE_HZ,
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
        f"coba + ALIF on DMTS (T_s = {DMTS_T_S_MS:.0f}, "
        f"T_d = {DMTS_T_D_MS:.0f}, T_p = {DMTS_T_P_MS:.0f} ms)"
    )
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
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

    # Cell 1: train coba and ping with ALIF enabled.
    if not skip_training:
        for model in MODELS:
            train_with_alif(model, tier)
    else:
        missing = [
            m for m in MODELS if not (alif_train_dir(m) / "weights.pth").exists()
        ]
        if missing:
            raise SystemExit(
                f"--skip-training but ALIF weights missing for {missing}"
            )

    print(
        f"[alif-rasters] coba + ping (trained with ALIF) at βs {{0.0, {ALIF_BETA}}}"
    )
    samples: dict[str, dict[float, dict]] = {}
    summary_rows: list[dict] = []
    for model in MODELS:
        samples[model] = {}
        for beta in (0.0, ALIF_BETA):
            s = capture_raster(alif_train_dir(model), beta)
            samples[model][beta] = s
            print(
                f"  {model:<4}  β={beta:.2f}  "
                f"E={s['e_rate_hz']:6.2f} Hz  I={s['i_rate_hz']:6.2f} Hz"
            )
            summary_rows.append({
                "model": model, "alif_beta": beta,
                "e_rate_hz": s["e_rate_hz"], "i_rate_hz": s["i_rate_hz"],
            })
    plot_alif_rasters(samples, FIGURES / "alif_rasters.png", notebook_run_id)
    print(f"wrote {FIGURES / 'alif_rasters.png'}")

    # Copy training videos
    for model in MODELS:
        src = alif_train_dir(model) / "training.mp4"
        dst = FIGURES / f"training__{model}_alif.mp4"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"wrote {dst}")

    # Cell 2: DMTS on coba+ALIF. Retrains whenever weights are missing.
    if skip_training and (dmts_dir() / "weights.pth").exists():
        dmts_history = json.loads((dmts_dir() / "history.json").read_text())
    else:
        dmts_history = train_dmts_alif(tier, dmts_dir())
    plot_dmts_history(
        dmts_history, FIGURES / "dmts_training_curve.png", notebook_run_id
    )
    print(f"wrote {FIGURES / 'dmts_training_curve.png'}")

    duration_s = time.monotonic() - t_start
    summary = {
        "notebook_run_id": notebook_run_id,
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "tier": tier,
        "config": {
            "tier": tier,
            "models": MODELS,
            "alif_beta": ALIF_BETA,
            "tau_adapt_ms": TAU_ADAPT_MS,
            "t_total_ms": T_MS_TRIAL,
            "dmts_t_s_ms": DMTS_T_S_MS,
            "dmts_t_d_ms": DMTS_T_D_MS,
            "dmts_t_p_ms": DMTS_T_P_MS,
            "dmts_lr": DMTS_LR,
            "sample_idx": SAMPLE_IDX,
            "seed": SEED,
        },
        "results": summary_rows,
        "dmts_history": dmts_history,
        "success_criteria": [
            {
                "label": "ALIF raster rendered",
                "passed": (FIGURES / "alif_rasters.png").exists(),
                "detail": (
                    f"coba E β0={samples['coba'][0.0]['e_rate_hz']:.1f} Hz; "
                    f"coba E β{ALIF_BETA}={samples['coba'][ALIF_BETA]['e_rate_hz']:.1f} Hz; "
                    f"ping E β0={samples['ping'][0.0]['e_rate_hz']:.1f} Hz; "
                    f"ping E β{ALIF_BETA}={samples['ping'][ALIF_BETA]['e_rate_hz']:.1f} Hz"
                ),
            },
            {
                "label": "DMTS test acc > chance (>55%)",
                "passed": max(dmts_history["test_acc"]) > 55.0,
                "detail": (
                    f"best test acc = {max(dmts_history['test_acc']):.1f}%; "
                    f"final = {dmts_history['test_acc'][-1]:.1f}%"
                ),
            },
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")


if __name__ == "__main__":
    main()
