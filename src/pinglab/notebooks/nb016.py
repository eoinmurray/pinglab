"""Notebook runner for entry 016 — sequential MNIST tracking + entry 015 latency.

Trains cuba/coba/ping on standard single-MNIST (recipe inherited from
nb010/nb011/nb012), then runs two *eval-only* probes against the
frozen weights:

  * Sequential MNIST tracking (entry 016) — each trial is four random
    MNIST digits packed into 200 ms; we apply the trained mem-mean
    readout per-window and ask whether the trial-level decoder still
    picks out the currently-shown digit.
  * Single-window latency + per-trial dynamics (entry 015) — one digit
    fills the full 200 ms; we score the running argmax through time
    and look at the per-trial rate / autocorrelation of the hidden
    layer to ask whether ping has a stimulus-locked rhythm.

The two probes share the trained weights, so this single runner emits
artifacts for both entries: nb016 figures + numbers.json for the
sequential sweep, nb015 figures + numbers.json for the latency curve
and per-trial dynamics. Run-time is dominated by training; splitting
the runner would just duplicate the training step.

Notebook entries:
  src/docs/src/pages/notebooks/nb016.mdx
  src/docs/src/pages/notebooks/nb015.mdx
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import sh

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _modal import append_modal_args, parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402
import theme  # noqa: E402

SLUG = "nb016"
SLUG_NB015 = "nb015"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
# nb015 reuses the same trained weights — its figures and numbers.json
# are emitted from this same runner (latency curve + per-trial dynamics).
FIGURES_NB015 = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG_NB015
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope" / "__main__.py"

MODELS = ("cuba", "coba", "ping")
MODEL_COLORS = {
    "cuba": theme.DEEP_RED,
    "coba": theme.AMBER,
    "ping": theme.ELECTRIC_CYAN,
}
DT = 0.1
T_MS = 200.0
SEED = 42
DEFAULT_TIER = "small"
TIER_CONFIG = {
    "extra small": dict(max_samples=100, epochs=1),
    "small": dict(max_samples=500, epochs=5),
    "medium": dict(max_samples=2000, epochs=10),
    "large": dict(max_samples=5000, epochs=40),
    "huge": dict(max_samples=10000, epochs=80),
}


def build_args(model: str, tier: str, out_dir: Path, modal_gpu: str | None) -> list[str]:
    """Recipes mirror nb010 (cuba), nb011 (coba), nb012 (ping) verbatim."""
    cfg = TIER_CONFIG[tier]
    common = [
        "run", "python", str(OSCILLOSCOPE), "train",
        "--dataset", "mnist",
        "--max-samples", str(cfg["max_samples"]),
        "--epochs", str(cfg["epochs"]),
        "--t-ms", str(T_MS),
        "--dt", str(DT),
        "--seed", str(SEED),
        "--observe", "video",
        "--frame-rate", "1",
        "--readout", "mem-mean",
        "--surrogate-slope", "1",
        "--batch-size", "256",
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    if model == "cuba":
        cell = ["--model", "cuba", "--kaiming-init", "--lr", "0.04"]
    elif model == "coba":
        cell = [
            "--model", "ping",  # COBANet with ei-strength 0
            "--ei-strength", "0",
            "--v-grad-dampen", "1000",
            "--w-in", "0.3",
            "--w-in-sparsity", "0.95",
            "--readout-w-out-scale", "100",
            "--lr", "0.0004",
        ]
    elif model == "ping":
        cell = [
            "--model", "ping",
            "--ei-strength", "1",
            "--v-grad-dampen", "1000",
            "--w-in", "1.2",
            "--w-in-sparsity", "0.95",
            "--readout-w-out-scale", "500",
            "--lr", "0.0004",
        ]
    else:
        raise ValueError(f"unknown model {model!r}")
    return append_modal_args(common + cell, modal_gpu)


def train_model(model: str, tier: str, modal_gpu: str | None,
                log_file: Path | None = None) -> Path:
    out_dir = ARTIFACTS / model / "train"
    print(f"[{model}] training → {out_dir.relative_to(REPO)}"
          + (f"  [modal:{modal_gpu}]" if modal_gpu else "")
          + (f"  log={log_file.relative_to(REPO)}" if log_file else ""),
          flush=True)
    args = build_args(model, tier, out_dir, modal_gpu)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w") as fh:
            sh.uv(*args, _cwd=str(REPO), _out=fh, _err=fh)
    else:
        sh.uv(*args, _cwd=str(REPO), _out=sys.stdout, _err=sys.stderr)
    if not (out_dir / "metrics.json").exists():
        raise SystemExit(f"training did not produce {out_dir / 'metrics.json'}")
    return out_dir


def copy_video(model: str, run_dir: Path) -> Path:
    src = run_dir / "training.mp4"
    if not src.exists():
        raise SystemExit(f"no training video at {src}")
    dst = FIGURES / f"training_{model}.mp4"
    FIGURES.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    print(f"  → {dst.relative_to(REPO)}")
    return dst


def write_numbers(run_dirs: dict[str, Path], notebook_run_id: str,
                  duration_s: float) -> dict:
    runs = {}
    for model, run_dir in run_dirs.items():
        m = json.loads((run_dir / "metrics.json").read_text())
        cfg = json.loads((run_dir / "config.json").read_text())
        runs[model] = {
            "best_acc": m.get("best_acc"),
            "final_acc": m["epochs"][-1]["acc"],
            "final_loss": m["epochs"][-1]["loss"],
            "rate_e": m["epochs"][-1].get("rate_e"),
            "rate_i": m["epochs"][-1].get("rate_i"),
            "git_sha": cfg.get("git_sha"),
            "run_id": cfg.get("run_id"),
        }
    return {
        "notebook_run_id": notebook_run_id,
        "duration_s": duration_s,
        "duration": f"{int(duration_s // 60)}m {int(duration_s % 60):02d}s",
        "tier": TIER,
        "config": {
            "dt": DT, "t_ms": T_MS, "seed": SEED,
            "max_samples": TIER_CONFIG[TIER]["max_samples"],
            "epochs": TIER_CONFIG[TIER]["epochs"],
            "dataset": "mnist",
        },
        "runs": runs,
        "run_finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


TIER = DEFAULT_TIER

SEQ_WINDOWS = 4
SEQ_WINDOW_MS = 50.0
SEQ_TRIALS = 800
# Sweep: each entry is the number of digits packed into the 200 ms trial.
# Window length = T_MS / n. Chosen so window_ms × DT_steps stays integer.
SEQ_SWEEP_NS = (1, 2, 4, 5, 8, 10, 20, 25, 40)
# Input-rate sweep at fixed window length (canonical 4×50 ms layout).
SEQ_RATE_SWEEP_HZS = (1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0)
SEQ_RATE_SWEEP_N = 4
SEQ_RATE_SWEEP_WINDOW_MS = 50.0
# Fractions of the 200 ms single-window trial at which to score the running
# argmax for the latency curve. Skewed toward early times where the curve moves.
SEQ_LATENCY_FRACS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30,
                     0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)


def eval_sequential(model_name: str, run_dir: Path,
                    n_windows: int = SEQ_WINDOWS,
                    window_ms: float = SEQ_WINDOW_MS,
                    rate_hz: float | None = None) -> dict:
    """Load trained weights, run sequential-MNIST trials, return per-window accuracy."""
    import numpy as np
    import torch
    from torchvision import datasets, transforms

    from config import build_net  # type: ignore[import]
    import models as M  # type: ignore[import]
    from oscilloscope.encoders import encode_images_poisson  # type: ignore[import]

    cfg = json.loads((run_dir / "config.json").read_text())
    device = torch.device("cpu")
    torch.manual_seed(SEED)

    M.N_IN = cfg.get("n_in", 784)
    M.N_INH = cfg.get("n_inh", 256)
    M.dt = DT
    M.T_ms = n_windows * window_ms
    M.T_steps = int(M.T_ms / M.dt)
    hidden_sizes = cfg.get("hidden_sizes") or [cfg["n_hidden"]]
    net = build_net(
        cfg["model"],
        w_in=cfg.get("w_in"),
        w_in_sparsity=cfg.get("w_in_sparsity", 0.0),
        ei_strength=cfg.get("ei_strength"),
        ei_ratio=cfg.get("ei_ratio", 2.0),
        sparsity=cfg.get("sparsity", 0.0),
        device=device,
        kaiming_init=cfg.get("kaiming_init", False),
        dales_law=cfg.get("dales_law", True),
        hidden_sizes=hidden_sizes,
        readout_mode=cfg.get("readout_mode", "mem-mean"),
    ).to(device)
    state = torch.load(run_dir / "weights.pth", map_location=device, weights_only=True)
    net.load_state_dict(state, strict=False)
    net.eval()

    mnist = datasets.MNIST(
        root=str(REPO / "src" / "artifacts" / "data"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(mnist), size=(SEQ_TRIALS, n_windows))
    images = torch.zeros(SEQ_TRIALS, n_windows, 784)
    labels = np.zeros((SEQ_TRIALS, n_windows), dtype=np.int64)
    for i in range(SEQ_TRIALS):
        for k in range(n_windows):
            x, y = mnist[int(indices[i, k])]
            images[i, k] = x.view(-1)
            labels[i, k] = y

    window_steps = int(round(window_ms / DT))
    T_steps = n_windows * window_steps
    rate_hz = rate_hz if rate_hz is not None else cfg.get("input_rate", 25.0)
    batch = 64
    # Per-window predictions: apply trained mem-mean readout to mem averaged
    # over each window. spike_record["out"][t] holds the running mem-mean
    # cumulative, i.e. (Σ_{s≤t} v_out_s) / T_steps. Window-k mean recovers
    # via finite difference of the cumulative endpoints.
    net.recording = True
    per_window_preds = np.zeros((SEQ_TRIALS, n_windows), dtype=np.int64)
    input_spike_total = 0.0
    hidden_spike_total = 0.0
    n_total_trials = 0
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    with torch.no_grad():
        for start in range(0, SEQ_TRIALS, batch):
            stop = min(start + batch, SEQ_TRIALS)
            blocks = []
            for k in range(n_windows):
                blocks.append(
                    encode_images_poisson(
                        images[start:stop, k], window_steps, DT, rate_hz, generator=gen
                    )
                )
            input_spikes = torch.cat(blocks, dim=0)
            net(input_spikes=input_spikes)
            input_spike_total += float(input_spikes.sum().item())
            for key, rec in net.spike_record.items():
                if key == "hid" or key.startswith("hid_"):
                    hidden_spike_total += float(rec.sum().item())
            n_total_trials += stop - start
            out_rec = net.spike_record["out"]
            if out_rec.dim() == 2:
                out_rec = out_rec.unsqueeze(1)
            cum = out_rec * float(T_steps)
            for k in range(n_windows):
                end_t = (k + 1) * window_steps - 1
                window_sum = cum[end_t] if k == 0 else cum[end_t] - cum[k * window_steps - 1]
                window_mean = window_sum / float(window_steps)
                per_window_preds[start:stop, k] = window_mean.argmax(dim=-1).cpu().numpy()

    per_window_acc: dict[str, float] = {}
    for k in range(n_windows):
        per_window_acc[f"window_{k}"] = float(
            (per_window_preds[:, k] == labels[:, k]).mean() * 100
        )
    mean_acc = float(sum(per_window_acc.values()) / n_windows)
    n_window_total = float(n_total_trials * n_windows)
    return {
        "per_window_acc_pct": per_window_acc,
        "mean_acc_pct": mean_acc,
        "n_trials": SEQ_TRIALS,
        "n_windows": n_windows,
        "window_ms": window_ms,
        "rate_hz": rate_hz,
        "mean_input_spikes_per_window": input_spike_total / n_window_total,
        "mean_hidden_spikes_per_window": hidden_spike_total / n_window_total,
    }


def eval_latency(model_name: str, run_dir: Path,
                 window_ms: float = T_MS,
                 rate_hz: float | None = None,
                 fracs: tuple[float, ...] = SEQ_LATENCY_FRACS) -> list[dict]:
    """Single-digit, full-window trials. Score running argmax + cumulative spikes
    at each fraction of the window.

    Returns a list of dicts (one per fraction) with keys:
        frac, time_ms, accuracy_pct, mean_input_spikes, mean_hidden_spikes.
    """
    import numpy as np
    import torch
    from torchvision import datasets, transforms

    from config import build_net  # type: ignore[import]
    import models as M  # type: ignore[import]
    from oscilloscope.encoders import encode_images_poisson  # type: ignore[import]

    cfg = json.loads((run_dir / "config.json").read_text())
    device = torch.device("cpu")
    torch.manual_seed(SEED)

    M.N_IN = cfg.get("n_in", 784)
    M.N_INH = cfg.get("n_inh", 256)
    M.dt = DT
    M.T_ms = window_ms
    M.T_steps = int(M.T_ms / M.dt)
    hidden_sizes = cfg.get("hidden_sizes") or [cfg["n_hidden"]]
    net = build_net(
        cfg["model"],
        w_in=cfg.get("w_in"),
        w_in_sparsity=cfg.get("w_in_sparsity", 0.0),
        ei_strength=cfg.get("ei_strength"),
        ei_ratio=cfg.get("ei_ratio", 2.0),
        sparsity=cfg.get("sparsity", 0.0),
        device=device,
        kaiming_init=cfg.get("kaiming_init", False),
        dales_law=cfg.get("dales_law", True),
        hidden_sizes=hidden_sizes,
        readout_mode=cfg.get("readout_mode", "mem-mean"),
    ).to(device)
    state = torch.load(run_dir / "weights.pth", map_location=device, weights_only=True)
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True

    mnist = datasets.MNIST(
        root=str(REPO / "src" / "artifacts" / "data"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(mnist), size=SEQ_TRIALS)
    images = torch.stack([mnist[int(i)][0].view(-1) for i in indices])
    labels = np.array([mnist[int(i)][1] for i in indices], dtype=np.int64)

    T_steps = int(round(window_ms / DT))
    rate_hz = rate_hz if rate_hz is not None else cfg.get("input_rate", 25.0)
    batch = 64
    step_indices = [max(0, min(T_steps - 1, int(round(f * T_steps)) - 1)) for f in fracs]

    n_fracs = len(fracs)
    correct = np.zeros(n_fracs, dtype=np.int64)
    input_spk_sum = np.zeros(n_fracs, dtype=np.float64)
    hidden_spk_sum = np.zeros(n_fracs, dtype=np.float64)
    n_total = 0
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    with torch.no_grad():
        for start in range(0, SEQ_TRIALS, batch):
            stop = min(start + batch, SEQ_TRIALS)
            input_spikes = encode_images_poisson(
                images[start:stop], T_steps, DT, rate_hz, generator=gen
            )
            net(input_spikes=input_spikes)
            out_rec = net.spike_record["out"]
            if out_rec.dim() == 2:
                out_rec = out_rec.unsqueeze(1)
            in_cum = input_spikes.sum(dim=-1).cumsum(dim=0)  # (T, B)
            hid_cum = None
            for key, rec in net.spike_record.items():
                if key == "hid" or key.startswith("hid_"):
                    contribution = rec.sum(dim=-1).cumsum(dim=0)
                    hid_cum = contribution if hid_cum is None else hid_cum + contribution
            if hid_cum is None:
                hid_cum = torch.zeros_like(in_cum)
            batch_labels = labels[start:stop]
            for fi, t_idx in enumerate(step_indices):
                preds = out_rec[t_idx].argmax(dim=-1).cpu().numpy()
                correct[fi] += int((preds == batch_labels).sum())
                input_spk_sum[fi] += float(in_cum[t_idx].sum().item())
                hidden_spk_sum[fi] += float(hid_cum[t_idx].sum().item())
            n_total += stop - start

    out: list[dict] = []
    for fi, frac in enumerate(fracs):
        out.append({
            "frac": float(frac),
            "time_ms": float(frac * window_ms),
            "accuracy_pct": float(correct[fi] / n_total * 100.0),
            "mean_input_spikes": float(input_spk_sum[fi] / n_total),
            "mean_hidden_spikes": float(hidden_spk_sum[fi] / n_total),
        })
    return out


def _build_seq_trial():
    """Build one canonical sequential trial. Returns (input_spikes, images, labels)."""
    import numpy as np
    import torch
    from torchvision import datasets, transforms

    sys.path.insert(0, str(REPO / "src" / "pinglab"))
    from oscilloscope.encoders import encode_images_poisson  # type: ignore[import]

    mnist = datasets.MNIST(
        root=str(REPO / "src" / "artifacts" / "data"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    rng = np.random.default_rng(SEED)
    chosen: list[int] = []
    seen: set[int] = set()
    while len(chosen) < SEQ_WINDOWS:
        i = int(rng.integers(0, len(mnist)))
        y = int(mnist[i][1])
        if y in seen:
            continue
        seen.add(y)
        chosen.append(i)
    images = torch.stack([mnist[i][0].view(-1) for i in chosen])
    labels = [int(mnist[i][1]) for i in chosen]

    window_steps = int(round(SEQ_WINDOW_MS / DT))
    rate_hz = 25.0
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    blocks = [
        encode_images_poisson(images[k : k + 1], window_steps, DT, rate_hz, generator=gen)
        for k in range(SEQ_WINDOWS)
    ]
    return torch.cat(blocks, dim=0), images, labels


def _record_response(model_name: str, run_dir: Path, input_spikes):
    """Run a single trial through a trained model with recording=True."""
    import json as _json

    import torch

    from config import build_net  # type: ignore[import]
    import models as M  # type: ignore[import]

    cfg = _json.loads((run_dir / "config.json").read_text())
    M.N_IN = cfg.get("n_in", 784)
    M.N_INH = cfg.get("n_inh", 256)
    M.dt = DT
    M.T_ms = SEQ_WINDOWS * SEQ_WINDOW_MS
    M.T_steps = int(M.T_ms / M.dt)
    hidden_sizes = cfg.get("hidden_sizes") or [cfg["n_hidden"]]

    net = build_net(
        cfg["model"],
        w_in=cfg.get("w_in"),
        w_in_sparsity=cfg.get("w_in_sparsity", 0.0),
        ei_strength=cfg.get("ei_strength"),
        ei_ratio=cfg.get("ei_ratio", 2.0),
        sparsity=cfg.get("sparsity", 0.0),
        device=torch.device("cpu"),
        kaiming_init=cfg.get("kaiming_init", False),
        dales_law=cfg.get("dales_law", True),
        hidden_sizes=hidden_sizes,
        readout_mode=cfg.get("readout_mode", "mem-mean"),
    )
    state = torch.load(run_dir / "weights.pth", map_location="cpu", weights_only=True)
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True
    with torch.no_grad():
        net(input_spikes=input_spikes)
    return net.spike_record


def plot_seq_response(run_dirs: dict[str, Path]) -> Path:
    """Render per-model hidden-layer rasters for the canonical sequential trial."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    theme.apply()
    input_spikes, images, labels = _build_seq_trial()
    T_total = input_spikes.shape[0]

    n_rows = 2 + len(MODELS)
    fig = plt.figure(figsize=(8, 1.0 + 1.6 * n_rows))
    gs = fig.add_gridspec(
        n_rows,
        SEQ_WINDOWS,
        height_ratios=[1] + [1.6] * (n_rows - 1),
        hspace=0.35,
        wspace=0.1,
    )

    for k in range(SEQ_WINDOWS):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(images[k].view(28, 28).numpy(), cmap="gray_r")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"w{k}: {labels[k]}", fontsize=theme.SIZE_ANNOTATION)

    inp = input_spikes.squeeze(1).numpy()
    ax_in = fig.add_subplot(gs[1, :])
    ti, ni = np.where(inp > 0)
    ax_in.scatter(ti * DT, ni, s=0.3, c=theme.INK_BLACK, alpha=0.4)
    for k in range(1, SEQ_WINDOWS):
        ax_in.axvline(k * SEQ_WINDOW_MS, color=theme.GREY_MID, linewidth=0.7, linestyle="--")
    ax_in.set_xlim(0, T_total * DT)
    ax_in.set_ylim(-1, inp.shape[1])
    ax_in.set_ylabel("input")
    ax_in.set_xticklabels([])
    ax_in.set_title("input spikes (Poisson encoding)")

    for row, model in enumerate(MODELS):
        rec = _record_response(model, run_dirs[model], input_spikes)
        hid_keys = sorted(k for k in rec.keys() if k == "hid" or k.startswith("hid_"))
        last_hid = rec[hid_keys[-1]].squeeze(1).numpy() if hid_keys else None
        ax = fig.add_subplot(gs[2 + row, :])
        if last_hid is not None:
            t_idx, n_idx = np.where(last_hid > 0)
            ax.scatter(t_idx * DT, n_idx, s=0.3, c=MODEL_COLORS[model], alpha=0.6)
            ax.set_ylim(-1, last_hid.shape[1])
        for k in range(1, SEQ_WINDOWS):
            ax.axvline(k * SEQ_WINDOW_MS, color=theme.GREY_MID, linewidth=0.7, linestyle="--")
        ax.set_xlim(0, T_total * DT)
        ax.set_ylabel(f"{model}\nhidden")
        if row == len(MODELS) - 1:
            ax.set_xlabel("time (ms)")
        else:
            ax.set_xticklabels([])

    fig.tight_layout()
    out = FIGURES / "sequential_response.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out.relative_to(REPO)}")
    return out


def plot_seq_results(seq_results: dict[str, dict]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme.apply()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in MODELS:
        points = sorted(seq_results[model].values(), key=lambda r: r["window_ms"])
        xs = [r["window_ms"] for r in points]
        ys = [r["mean_acc_pct"] for r in points]
        ax.plot(xs, ys, marker="o", color=MODEL_COLORS[model], label=model)
    ax.axhline(10, linestyle="--", linewidth=1.0, color=theme.DEEP_RED, label="chance (10%)")
    ax.set_xlabel("window length (ms)")
    ax.set_ylabel("mean per-window accuracy (%)")
    ax.set_title("sequential MNIST: accuracy vs window length (200 ms trial)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIGURES / "tracking_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out.relative_to(REPO)}")
    return out


def plot_seq_input_spikes(rate_results: dict[str, dict]) -> Path:
    """Accuracy vs mean input spikes per window (input-rate sweep at fixed 50 ms)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme.apply()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in MODELS:
        points = sorted(rate_results[model].values(), key=lambda r: r["rate_hz"])
        xs = [r["mean_input_spikes_per_window"] for r in points]
        ys = [r["mean_acc_pct"] for r in points]
        ax.plot(xs, ys, marker="o", color=MODEL_COLORS[model], label=model)
    ax.axhline(10, linestyle="--", linewidth=1.0, color=theme.DEEP_RED, label="chance (10%)")
    ax.set_xscale("log")
    ax.set_xlabel("mean input spikes per 50 ms window (log scale)")
    ax.set_ylabel("mean per-window accuracy (%)")
    ax.set_title("sequential MNIST: accuracy vs input spikes (4 × 50 ms, rate swept)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIGURES / "tracking_input_spikes.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out.relative_to(REPO)}")
    return out


def eval_per_trial_rate(model_name: str, run_dir: Path,
                        window_ms: float = T_MS,
                        rate_hz: float | None = None,
                        n_keep_trials: int = 50) -> dict:
    """Per-trial smoothed hidden-population rate for the single 200 ms digit.

    Returns:
        time_ms (T,), rates (K, T) for the first n_keep_trials trials,
        autocorr (T_lags,) — trial-averaged autocorrelation of the
        post-onset rate (computed on all SEQ_TRIALS trials).
    """
    import numpy as np
    import torch
    from torchvision import datasets, transforms

    from config import build_net  # type: ignore[import]
    import models as M  # type: ignore[import]
    from oscilloscope.encoders import encode_images_poisson  # type: ignore[import]

    cfg = json.loads((run_dir / "config.json").read_text())
    device = torch.device("cpu")
    torch.manual_seed(SEED)

    M.N_IN = cfg.get("n_in", 784)
    M.N_INH = cfg.get("n_inh", 256)
    M.dt = DT
    M.T_ms = window_ms
    M.T_steps = int(M.T_ms / M.dt)
    hidden_sizes = cfg.get("hidden_sizes") or [cfg["n_hidden"]]
    net = build_net(
        cfg["model"],
        w_in=cfg.get("w_in"),
        w_in_sparsity=cfg.get("w_in_sparsity", 0.0),
        ei_strength=cfg.get("ei_strength"),
        ei_ratio=cfg.get("ei_ratio", 2.0),
        sparsity=cfg.get("sparsity", 0.0),
        device=device,
        kaiming_init=cfg.get("kaiming_init", False),
        dales_law=cfg.get("dales_law", True),
        hidden_sizes=hidden_sizes,
        readout_mode=cfg.get("readout_mode", "mem-mean"),
    ).to(device)
    state = torch.load(run_dir / "weights.pth", map_location=device, weights_only=True)
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True

    mnist = datasets.MNIST(
        root=str(REPO / "src" / "artifacts" / "data"),
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(mnist), size=SEQ_TRIALS)
    images = torch.stack([mnist[int(i)][0].view(-1) for i in indices])

    T_steps = int(round(window_ms / DT))
    rate_hz_eff = rate_hz if rate_hz is not None else cfg.get("input_rate", 25.0)
    batch = 64

    per_trial_rate = np.zeros((SEQ_TRIALS, T_steps), dtype=np.float64)
    n_neurons = 0
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    with torch.no_grad():
        for start in range(0, SEQ_TRIALS, batch):
            stop = min(start + batch, SEQ_TRIALS)
            input_spikes = encode_images_poisson(
                images[start:stop], T_steps, DT, rate_hz_eff, generator=gen
            )
            net(input_spikes=input_spikes)
            hid_total = None
            n_neurons = 0
            for key, rec in net.spike_record.items():
                if key == "hid" or key.startswith("hid_"):
                    summed = rec.sum(dim=-1)  # (T, B)
                    hid_total = summed if hid_total is None else hid_total + summed
                    n_neurons += int(rec.shape[-1])
            # spikes per ms per neuron → Hz: divide by (n_neurons * dt_in_seconds)
            hid_hz = hid_total.transpose(0, 1).cpu().numpy() / (n_neurons * (DT / 1000.0))
            per_trial_rate[start:stop] = hid_hz

    # Smooth each trial with a 1 ms Gaussian
    smoothed = np.empty_like(per_trial_rate)
    for i in range(SEQ_TRIALS):
        smoothed[i] = _smooth_gaussian(per_trial_rate[i], sigma_ms=1.0, dt_ms=DT)

    # Average autocorrelation: drop first 20 ms (onset transient) so we look at
    # the sustained regime, then z-score per trial and autocorrelate.
    onset_steps = int(round(20.0 / DT))
    sustained = smoothed[:, onset_steps:]
    sustained = sustained - sustained.mean(axis=1, keepdims=True)
    norms = sustained.std(axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    sustained = sustained / norms
    n_lags_steps = int(round(60.0 / DT))  # 60 ms of lag is enough to span 2-3 cycles
    autocorr = np.zeros(n_lags_steps + 1, dtype=np.float64)
    for i in range(SEQ_TRIALS):
        x = sustained[i]
        # FFT-based autocorrelation, normalised by trial length
        n = x.shape[0]
        f = np.fft.rfft(x, n=2 * n)
        ac = np.fft.irfft(f * np.conj(f))[: n_lags_steps + 1] / n
        autocorr += ac
    autocorr /= SEQ_TRIALS

    time_ms = (np.arange(T_steps) + 1) * DT
    lag_ms = np.arange(n_lags_steps + 1) * DT
    return {
        "time_ms": time_ms.tolist(),
        "rates_subset": smoothed[:n_keep_trials].tolist(),
        "lag_ms": lag_ms.tolist(),
        "autocorr": autocorr.tolist(),
    }


def plot_per_trial_rate(per_trial: dict[str, dict]) -> tuple[Path, dict]:
    """Two figures-worth of analysis in one image:
       (top) ping single-trial heatmap; (bottom) trial-averaged autocorrelation
       for all three models with detected peaks."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.signal import find_peaks

    theme.apply()
    fig = plt.figure(figsize=(12, 6.75))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 1], hspace=0.4)

    # Top: ping heatmap
    ax_h = fig.add_subplot(gs[0, 0])
    ping_rates = np.asarray(per_trial["ping"]["rates_subset"])
    t = np.asarray(per_trial["ping"]["time_ms"])
    im = ax_h.imshow(ping_rates, aspect="auto",
                     extent=(float(t[0]), float(t[-1]),
                             ping_rates.shape[0], 0),
                     cmap="pinglab_brand")
    ax_h.set_xlabel("time (ms)")
    ax_h.set_ylabel("trial #")
    ax_h.set_title("ping: single-trial hidden firing rate (first 50 trials)")
    cbar = fig.colorbar(im, ax=ax_h, fraction=0.02, pad=0.01)
    cbar.set_label("rate (Hz)")

    # Bottom: trial-averaged autocorrelation per model
    ax_a = fig.add_subplot(gs[1, 0])
    peak_summary: dict[str, list[dict]] = {}
    for model in MODELS:
        lag = np.asarray(per_trial[model]["lag_ms"])
        ac = np.asarray(per_trial[model]["autocorr"])
        ac = ac / max(abs(ac[0]), 1e-9)  # normalise so lag-0 is 1
        ax_a.plot(lag, ac, color=MODEL_COLORS[model], label=model)
        # Skip lag-0; require positive peaks at least 5 ms apart, prominence 0.05
        skip = int(round(2.0 / DT))
        peaks, _ = find_peaks(ac[skip:], distance=int(round(5.0 / DT)),
                              prominence=0.05)
        peak_lags = lag[skip:][peaks]
        peak_vals = ac[skip:][peaks]
        peak_summary[model] = [
            {"lag_ms": float(pl), "value": float(pv)}
            for pl, pv in zip(peak_lags, peak_vals)
        ]
        for pl in peak_lags[:3]:
            ax_a.axvline(pl, color=MODEL_COLORS[model], linestyle=":", linewidth=0.6, alpha=0.6)
    ax_a.axhline(0, color=theme.GREY_MID, linewidth=0.5)
    ax_a.set_xlabel("lag (ms)")
    ax_a.set_ylabel("autocorr (normalised)")
    ax_a.set_title("trial-averaged autocorrelation of hidden rate (post-onset)")
    ax_a.legend(loc="upper right")

    fig.tight_layout()
    # Per-trial dynamics belong to entry 015 — emit under nb015's figures dir.
    out = FIGURES_NB015 / "per_trial_dynamics.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out.relative_to(REPO)}")
    return out, peak_summary


def _smooth_gaussian(x, sigma_ms: float, dt_ms: float):
    import numpy as np
    sigma_steps = max(sigma_ms / dt_ms, 0.5)
    radius = int(round(4 * sigma_steps))
    k = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (k / sigma_steps) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.asarray(x), kernel, mode="same")



def plot_seq_latency(latency_results: dict[str, list[dict]]) -> tuple[Path, Path, Path]:
    """Three latency plots from the single-window sweep:
       (1) accuracy vs time, (2) accuracy vs cumulative input spikes,
       (3) accuracy vs cumulative hidden spikes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme.apply()

    def _plot(xkey: str, xlabel: str, log_x: bool, title: str, fname: str) -> Path:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for model in MODELS:
            pts = latency_results[model]
            xs = [r[xkey] for r in pts]
            ys = [r["accuracy_pct"] for r in pts]
            ax.plot(xs, ys, marker="o", color=MODEL_COLORS[model], label=model)
        ax.axhline(10, linestyle="--", linewidth=1.0, color=theme.DEEP_RED, label="chance (10%)")
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("running accuracy (%)")
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.legend(loc="lower right")
        fig.tight_layout()
        # Latency plots belong to entry 015 — emit them under nb015's figures dir.
        out = FIGURES_NB015 / fname
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  → {out.relative_to(REPO)}")
        return out

    out_t = _plot("time_ms", "fraction of window seen (ms)", False,
                  "single-digit latency: accuracy vs time", "latency_time.png")
    out_in = _plot("mean_input_spikes",
                   "cumulative input spikes per trial (log scale)", True,
                   "single-digit latency: accuracy vs input spikes", "latency_input_spikes.png")
    out_hid = _plot("mean_hidden_spikes",
                    "cumulative hidden spikes per trial (log scale)", True,
                    "single-digit latency: accuracy vs hidden spikes", "latency_hidden_spikes.png")
    return out_t, out_in, out_hid


def plot_seq_hidden_spikes(seq_results: dict[str, dict]) -> Path:
    """Accuracy vs mean hidden spikes per window (window-length sweep at fixed input rate)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theme.apply()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in MODELS:
        points = sorted(seq_results[model].values(), key=lambda r: r["mean_hidden_spikes_per_window"])
        xs = [r["mean_hidden_spikes_per_window"] for r in points]
        ys = [r["mean_acc_pct"] for r in points]
        ax.plot(xs, ys, marker="o", color=MODEL_COLORS[model], label=model)
    ax.axhline(10, linestyle="--", linewidth=1.0, color=theme.DEEP_RED, label="chance (10%)")
    ax.set_xscale("log")
    ax.set_xlabel("mean hidden spikes per window (log scale)")
    ax.set_ylabel("mean per-window accuracy (%)")
    ax.set_title("sequential MNIST: accuracy vs hidden spikes (window length swept, 25 Hz input)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIGURES / "tracking_hidden_spikes.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out.relative_to(REPO)}")
    return out


def main() -> None:
    global TIER
    TIER = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    skip_training = "--skip-training" in sys.argv
    notebook_run_id = next_run_id(SLUG)
    print(
        f"notebook_run_id = {notebook_run_id} tier={TIER}"
        + ("  [skip-training]" if skip_training else "")
    )

    if "--no-wipe-dir" not in sys.argv:
        # When skipping training, only wipe figures (artifacts contain the weights).
        targets = (FIGURES, FIGURES_NB015) if skip_training else (ARTIFACTS, FIGURES, FIGURES_NB015)
        for path in targets:
            if path.exists():
                print(f"[wipe] {path.relative_to(REPO)}")
                shutil.rmtree(path)

    t_start = time.monotonic()
    run_dirs: dict[str, Path] = {}
    if skip_training:
        for model in MODELS:
            run_dir = ARTIFACTS / model / "train"
            if not (run_dir / "weights.pth").exists():
                raise SystemExit(
                    f"--skip-training requires existing weights at {run_dir}"
                )
            run_dirs[model] = run_dir
            copy_video(model, run_dir)
    elif modal_gpu:
        # Modal runs each cell in its own remote container — fan out the
        # three model trains and await them concurrently. Each model's
        # stdout/stderr goes to its own log file to avoid interleaving.
        log_files = {m: ARTIFACTS / m / "train.log" for m in MODELS}
        with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
            futures = {
                m: pool.submit(train_model, m, TIER, modal_gpu, log_files[m])
                for m in MODELS
            }
            for m in MODELS:
                run_dirs[m] = futures[m].result()
                print(f"[{m}] train done — log: {log_files[m].relative_to(REPO)}",
                      flush=True)
                copy_video(m, run_dirs[m])
    else:
        for model in MODELS:
            run_dirs[model] = train_model(model, TIER, modal_gpu)
            copy_video(model, run_dirs[model])

    seq_results: dict[str, dict] = {m: {} for m in MODELS}
    for n_windows in SEQ_SWEEP_NS:
        window_ms = T_MS / n_windows
        print(f"\n[seq-eval] {n_windows} × {window_ms:.1f} ms windows, {SEQ_TRIALS} trials")
        for model in MODELS:
            res = eval_sequential(model, run_dirs[model], n_windows, window_ms)
            seq_results[model][f"{n_windows}w"] = res
            print(f"  {model:5s}: mean={res['mean_acc_pct']:.1f}%  hid_spikes={res['mean_hidden_spikes_per_window']:.0f}")
    rate_results: dict[str, dict] = {m: {} for m in MODELS}
    for rate_hz in SEQ_RATE_SWEEP_HZS:
        print(f"\n[rate-eval] rate={rate_hz:.0f} Hz, {SEQ_RATE_SWEEP_N} × {SEQ_RATE_SWEEP_WINDOW_MS:.0f} ms")
        for model in MODELS:
            res = eval_sequential(model, run_dirs[model], SEQ_RATE_SWEEP_N,
                                  SEQ_RATE_SWEEP_WINDOW_MS, rate_hz=rate_hz)
            rate_results[model][f"{int(rate_hz)}hz"] = res
            print(f"  {model:5s}: mean={res['mean_acc_pct']:.1f}%  in_spikes={res['mean_input_spikes_per_window']:.0f}")
    print(f"\n[latency-eval] single 200 ms digit, {SEQ_TRIALS} trials")
    latency_results: dict[str, list[dict]] = {}
    for model in MODELS:
        latency_results[model] = eval_latency(model, run_dirs[model])
        last = latency_results[model][-1]
        print(f"  {model:5s}: final acc={last['accuracy_pct']:.1f}%  in_spk={last['mean_input_spikes']:.0f}  hid_spk={last['mean_hidden_spikes']:.0f}")
    plot_seq_response(run_dirs)
    plot_seq_results(seq_results)
    plot_seq_hidden_spikes(seq_results)
    plot_seq_input_spikes(rate_results)
    plot_seq_latency(latency_results)
    print("\n[per-trial] single-trial heatmap + trial-averaged autocorrelation")
    per_trial_results: dict[str, dict] = {}
    for model in MODELS:
        per_trial_results[model] = eval_per_trial_rate(model, run_dirs[model])
    _, autocorr_peaks = plot_per_trial_rate(per_trial_results)

    duration_s = time.monotonic() - t_start
    persist_run_id(SLUG, notebook_run_id)
    summary = write_numbers(run_dirs, notebook_run_id, duration_s)
    summary["seq_eval"] = seq_results
    summary["rate_eval"] = rate_results
    numbers_path = FIGURES / "numbers.json"
    numbers_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {numbers_path.relative_to(REPO)}")

    # Sidecar numbers.json for nb015 (Latency entry) — same trained
    # weights, separate artifact per styleguide §5 (figures namespaced
    # by notebook).
    nb015_run_id = next_run_id(SLUG_NB015)
    persist_run_id(SLUG_NB015, nb015_run_id)
    nb015_summary = {
        "notebook_run_id": nb015_run_id,
        "duration_s": duration_s,
        "duration": summary["duration"],
        "tier": TIER,
        "git_sha": next(iter(summary["runs"].values())).get("git_sha"),
        "config": summary["config"],
        "runs": summary["runs"],
        "latency_eval": latency_results,
        "autocorr_peaks": autocorr_peaks,
        "run_finished_at": summary["run_finished_at"],
    }
    nb015_numbers_path = FIGURES_NB015 / "numbers.json"
    nb015_numbers_path.parent.mkdir(parents=True, exist_ok=True)
    nb015_numbers_path.write_text(json.dumps(nb015_summary, indent=2) + "\n")
    print(f"wrote {nb015_numbers_path.relative_to(REPO)}")

    for model, info in summary["runs"].items():
        print(f"  {model:5s}: best={info['best_acc']}%  final={info['final_acc']}%"
              f"  rate_e={info['rate_e']:.1f}Hz")


if __name__ == "__main__":
    main()
