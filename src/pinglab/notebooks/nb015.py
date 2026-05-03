"""Notebook runner for entry 014 — sequential MNIST tracking.

Trains cuba/coba/ping on standard single-MNIST (recipe inherited from
nb010/10/11), then runs an *eval-only* probe in which each trial is a
sequence of four random MNIST digits, each shown for 50 ms inside a
200 ms trial. The trained network's mem-mean readout is queried
without retraining; for every target window position k ∈ {0,1,2,3} we
measure how often the network's argmax prediction equals the digit
shown in window k.

A network whose state turns over fast (small effective τ relative to
the 50 ms window) should peak at k = 3 — the most recent digit
dominates the readout. A network with slow recurrent persistence
(PING) should be flatter or biased toward earlier windows. This is a
direct probe of state turnover and temporal binding under each
architecture's natural dynamics.

Notebook entry: src/docs/src/pages/notebooks/nb015.mdx
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

SLUG = "nb015"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
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


def eval_sequential(model_name: str, run_dir: Path) -> dict:
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
    indices = rng.integers(0, len(mnist), size=(SEQ_TRIALS, SEQ_WINDOWS))
    images = torch.zeros(SEQ_TRIALS, SEQ_WINDOWS, 784)
    labels = np.zeros((SEQ_TRIALS, SEQ_WINDOWS), dtype=np.int64)
    for i in range(SEQ_TRIALS):
        for k in range(SEQ_WINDOWS):
            x, y = mnist[int(indices[i, k])]
            images[i, k] = x.view(-1)
            labels[i, k] = y

    window_steps = int(round(SEQ_WINDOW_MS / DT))
    rate_hz = cfg.get("input_rate", 25.0)
    batch = 64
    all_preds: list[int] = []
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    with torch.no_grad():
        for start in range(0, SEQ_TRIALS, batch):
            stop = min(start + batch, SEQ_TRIALS)
            B = stop - start
            blocks = []
            for k in range(SEQ_WINDOWS):
                blocks.append(
                    encode_images_poisson(
                        images[start:stop, k], window_steps, DT, rate_hz, generator=gen
                    )
                )
            input_spikes = torch.cat(blocks, dim=0)
            out = net(input_spikes=input_spikes)
            preds = out.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())

    preds_arr = np.array(all_preds)
    per_window_acc: dict[str, float] = {}
    for k in range(SEQ_WINDOWS):
        per_window_acc[f"window_{k}"] = float((preds_arr == labels[:, k]).mean() * 100)
    return {
        "per_window_acc_pct": per_window_acc,
        "n_trials": SEQ_TRIALS,
        "n_windows": SEQ_WINDOWS,
        "window_ms": SEQ_WINDOW_MS,
    }


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
    indices = rng.integers(0, len(mnist), size=SEQ_WINDOWS)
    images = torch.stack([mnist[int(i)][0].view(-1) for i in indices])
    labels = [int(mnist[int(i)][1]) for i in indices]

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


def plot_seq_input() -> Path:
    """Render input thumbnails + Poisson raster for the canonical trial."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    theme.apply()
    spikes_t, images, labels = _build_seq_trial()
    spikes = spikes_t.squeeze(1).numpy()
    T_total = spikes.shape[0]

    fig = plt.figure(figsize=(8, 4.5))
    gs = fig.add_gridspec(2, SEQ_WINDOWS, height_ratios=[1, 3], hspace=0.25, wspace=0.1)

    for k in range(SEQ_WINDOWS):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(images[k].view(28, 28).numpy(), cmap="gray_r")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"w{k}: {labels[k]}", fontsize=theme.SIZE_ANNOTATION)

    ax_raster = fig.add_subplot(gs[1, :])
    t_idx, n_idx = np.where(spikes > 0)
    ax_raster.scatter(t_idx * DT, n_idx, s=0.4, c=theme.INK_BLACK, alpha=0.5)
    for k in range(1, SEQ_WINDOWS):
        ax_raster.axvline(k * SEQ_WINDOW_MS, color=theme.GREY_MID, linewidth=0.8, linestyle="--")
    ax_raster.set_xlim(0, T_total * DT)
    ax_raster.set_ylim(-1, 784)
    ax_raster.set_xlabel("time (ms)")
    ax_raster.set_ylabel("input neuron")
    ax_raster.set_title("input spikes (Poisson encoding)")

    fig.tight_layout()
    out = FIGURES / "sequential_input.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out.relative_to(REPO)}")
    return out


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
    import numpy as np

    theme.apply()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    x = np.arange(SEQ_WINDOWS)
    for i, model in enumerate(MODELS):
        acc = [seq_results[model]["per_window_acc_pct"][f"window_{k}"] for k in range(SEQ_WINDOWS)]
        ax.bar(x + (i - 1) * width, acc, width, color=MODEL_COLORS[model], label=model)
    ax.axhline(10, linestyle="--", linewidth=1.0, color=theme.DEEP_RED, label="chance (10%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"window {k}\n({int(SEQ_WINDOW_MS * k)}-{int(SEQ_WINDOW_MS * (k + 1))} ms)" for k in range(SEQ_WINDOWS)])
    ax.set_ylabel("argmax matches digit at window k (%)")
    ax.set_title("sequential MNIST: which window dominates the readout?")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = FIGURES / "tracking_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out.relative_to(REPO)}")
    return out


def main() -> None:
    global TIER
    TIER = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id} tier={TIER}")

    if "--no-wipe-dir" not in sys.argv:
        for path in (ARTIFACTS, FIGURES):
            if path.exists():
                print(f"[wipe] {path.relative_to(REPO)}")
                shutil.rmtree(path)

    t_start = time.monotonic()
    run_dirs: dict[str, Path] = {}
    if modal_gpu:
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

    print(f"\n[seq-eval] {SEQ_WINDOWS} × {SEQ_WINDOW_MS:.0f} ms windows, {SEQ_TRIALS} trials")
    seq_results: dict[str, dict] = {}
    for model in MODELS:
        print(f"[{model}] sequential eval...")
        seq_results[model] = eval_sequential(model, run_dirs[model])
        accs = seq_results[model]["per_window_acc_pct"]
        per_w = "  ".join(f"w{k}={accs[f'window_{k}']:.1f}%" for k in range(SEQ_WINDOWS))
        print(f"  {model:5s}: {per_w}")
    plot_seq_input()
    plot_seq_response(run_dirs)
    plot_seq_results(seq_results)

    duration_s = time.monotonic() - t_start
    persist_run_id(SLUG, notebook_run_id)
    summary = write_numbers(run_dirs, notebook_run_id, duration_s)
    summary["seq_eval"] = seq_results
    numbers_path = FIGURES / "numbers.json"
    numbers_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {numbers_path.relative_to(REPO)}")
    for model, info in summary["runs"].items():
        print(f"  {model:5s}: best={info['best_acc']}%  final={info['final_acc']}%"
              f"  rate_e={info['rate_e']:.1f}Hz")


if __name__ == "__main__":
    main()
