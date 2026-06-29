"""Notebook runner for entry 044 — PING Δt audit.

Trains PING from scratch at five integration timesteps Δt ∈ {0.05, 0.1,
0.25, 0.5, 1.0} ms, holding total physical time T = 200 ms constant.
Measures post-training mean E rate (Hz) and accuracy per cell. Plots
the sweep on a log-Δt axis and grabs single-trial rasters at each Δt
to verify the gamma cycle stays at the same physical period in ms
(rather than the same step count).

Scaffolded by ar009 §Leg 1 item 4. Protects the ≈ 7 Hz headline rate
in ar008 against a discretisation-artefact reading.

Notebook entry: src/docs/src/pages/notebooks/nb044.mdx
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from helpers.figsave import save_figure  # noqa: E402
from helpers.fmt import format_duration  # noqa: E402
from helpers.modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402
from cli import theme  # noqa: E402

SLUG = "nb044"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
OSCILLOSCOPE = REPO / "src" / "cli" / "cli.py"

T_MS = 200.0

# Δt sweep — spans 20× across the integration timescale. nb025 trained
# at Δt = 0.1 ms; nb040 trained CUBA-PING at Δt = 1.0 ms. This audit
# trains the same recipe across the full range to verify the rate
# ceiling is a physical (Hz) feature rather than a step-count artefact.
DT_SWEEP_MS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0)
SEEDS: tuple[int, ...] = (42, 43, 44)

# Batch size is held at 64 across all Δt so per-step compute and memory
# stay comparable, and the Δt = 0.05 cells (4000 timesteps × N_E × N_I)
# fit in a single A100. nb025 used 256; smaller batches are slower but
# the recipe still trains.
BATCH_SIZE: int = 64

# Single-trial raster capture — same convention as nb025 / nb042.
RASTER_SAMPLE_IDX: int = 0
RASTER_N_E_PLOT: int = 200
RASTER_N_I_PLOT: int = 64
RASTER_T_WINDOW_MS: float = 100.0  # show first 100 ms so the cycle is visible

TIER_CONFIG = {
    "extra small": dict(max_samples=100, epochs=2),
    "small": dict(max_samples=500, epochs=10),
    # Medium tier upgraded 2026-06-03 to 100 epochs to match the
    # convergence audit in nb024. PING's rate is a converged operating
    # point at this horizon; the Δt-vs-rate scaling becomes a converged
    # measurement rather than a training-snapshot.
    "medium": dict(max_samples=2000, epochs=100),
    "large": dict(max_samples=5000, epochs=100),
    "extra large": dict(max_samples=10000, epochs=100),
}
DEFAULT_TIER = "small"

# Match nb025 PING recipe except for --batch-size (held at BATCH_SIZE
# here) and --dt (per cell).
PING_RECIPE: dict[str, str] = {
    "--ei-strength": "1",
    "--v-grad-dampen": "1000",
    "--w-in": "1.2",
    "--w-in-sparsity": "0.95",
    "--readout": "mem-mean",
    "--surrogate-slope": "1",
    "--readout-w-out-scale": "500",
    "--lr": "0.0004",
}


def dt_label(dt_ms: float) -> str:
    s = f"{dt_ms:g}".replace(".", "p")
    return f"dt{s}"


def cell_dir(dt_ms: float, seed: int) -> Path:
    """Trained cell — now the shared nb022 cell (train-once / reuse-many)."""
    from nb022 import cell_dir as shared_cell_dir
    return shared_cell_dir(f"ping__{dt_label(dt_ms)}__seed{seed}")


def build_train_args(dt_ms: float, seed: int, tier: str, out_dir: Path) -> list[str]:
    args = [
        "train",
        "--model", "ping",
        "--dataset", "mnist",
        "--max-samples", str(TIER_CONFIG[tier]["max_samples"]),
        "--epochs", str(TIER_CONFIG[tier]["epochs"]),
        "--t-ms", str(T_MS),
        "--dt", str(dt_ms),
        "--batch-size", str(BATCH_SIZE),
        "--seed", str(seed),
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    for k, v in PING_RECIPE.items():
        args += [k, v]
    return args


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


# ─── inference: rate, accuracy, raster ──────────────────────────────


def _load_trained_full(train_dir: Path, device):
    """Load trained PING checkpoint, restore the cell's Δt via patch_dt."""
    import torch

    import models as M
    from cli.config import build_net, patch_dt
    from cli import load_dataset, seed_everything

    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", 42)))
    M.T_ms = float(cfg["t_ms"])
    patch_dt(float(cfg["dt"]))

    hidden_sizes = cfg.get("hidden_sizes") or [int(cfg["n_hidden"])]
    M.N_HID = hidden_sizes[-1]
    M.N_INH = hidden_sizes[-1] // 4
    M.HIDDEN_SIZES = list(hidden_sizes)

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
        dales_law=bool(cfg.get("dales_law", True)),
        hidden_sizes=hidden_sizes,
        readout_mode=cfg.get("readout_mode", "mem-mean"),
    )
    state = torch.load(train_dir / "weights.pth", map_location=device, weights_only=False)
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True
    return net, cfg, X_te, y_te


def measure_rate_acc(train_dir: Path, device) -> dict:
    """Forward the test set; return (acc, mean E rate Hz, mean I rate Hz)."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    import models as M
    from cli import EVAL_SEED, encode_batch

    net, cfg, X_te, y_te = _load_trained_full(train_dir, device)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=64,
    )
    correct = total = 0
    e_spike_sum = i_spike_sum = 0.0
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    n_e = M.N_HID
    n_i = M.N_INH or 1
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
            logits = net(input_spikes=spk)
            correct += (logits.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
            e_spike_sum += float(net.spike_record["hid"].sum().item())
            if "inh" in net.spike_record:
                i_spike_sum += float(net.spike_record["inh"].sum().item())
    t_sec = float(cfg["t_ms"]) / 1000.0
    return {
        "dt_ms": float(cfg["dt"]),
        "t_ms": float(cfg["t_ms"]),
        "acc": 100.0 * correct / total,
        "e_rate_hz": e_spike_sum / (total * n_e * t_sec),
        "i_rate_hz": i_spike_sum / (total * n_i * t_sec),
        "n_total": total,
    }


def capture_raster(train_dir: Path, sample_idx: int, device) -> dict:
    """Single-trial raster from a trained cell."""
    import torch

    import models as M
    from cli import EVAL_SEED, encode_batch

    net, cfg, X_te, y_te = _load_trained_full(train_dir, device)
    X_b = torch.from_numpy(X_te[sample_idx : sample_idx + 1]).to(device)
    y_b = int(y_te[sample_idx])
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    with torch.no_grad():
        spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
        _ = net(input_spikes=spk)
    e_full = net.spike_record["hid"].cpu().numpy()
    i_full = net.spike_record["inh"].cpu().numpy()
    t_sec = float(cfg["t_ms"]) / 1000.0
    e_rate = float(e_full.sum() / (e_full.shape[1] * t_sec))
    i_rate = float(i_full.sum() / (i_full.shape[1] * t_sec))
    rng = np.random.default_rng(0)
    e_idx = np.sort(rng.choice(e_full.shape[1], RASTER_N_E_PLOT, replace=False))
    i_idx = np.sort(rng.choice(i_full.shape[1], RASTER_N_I_PLOT, replace=False))
    return {
        "dt_ms": float(cfg["dt"]),
        "label": y_b,
        "e": e_full[:, e_idx].astype(bool),
        "i": i_full[:, i_idx].astype(bool),
        "e_rate_hz": e_rate,
        "i_rate_hz": i_rate,
        "t_ms": float(cfg["t_ms"]),
    }


# ─── plotting ───────────────────────────────────────────────────────


def plot_dt_sweep(rows: list[dict], out_path: Path, run_id: str) -> None:
    """E rate (left axis) and accuracy (right axis) vs Δt (log)."""
    theme.apply()
    by_dt: dict[float, list[dict]] = {}
    for r in rows:
        by_dt.setdefault(r["dt_ms"], []).append(r)
    dts_sorted = sorted(by_dt.keys())
    e_means = [
        float(np.mean([r["e_rate_hz"] for r in by_dt[d]])) for d in dts_sorted
    ]
    e_sems = [
        float(np.std([r["e_rate_hz"] for r in by_dt[d]], ddof=1)
              / np.sqrt(max(1, len(by_dt[d]))))
        if len(by_dt[d]) > 1 else 0.0 for d in dts_sorted
    ]
    acc_means = [
        float(np.mean([r["acc"] for r in by_dt[d]])) for d in dts_sorted
    ]
    acc_sems = [
        float(np.std([r["acc"] for r in by_dt[d]], ddof=1)
              / np.sqrt(max(1, len(by_dt[d]))))
        if len(by_dt[d]) > 1 else 0.0 for d in dts_sorted
    ]

    fig, ax_rate = plt.subplots(figsize=(5.6, 3.5))
    ax_rate.errorbar(
        dts_sorted, e_means, yerr=e_sems,
        marker="D", markersize=6, lw=1.4, color=theme.INK_BLACK,
        capsize=3, label="E rate (Hz)",
    )
    ax_rate.set_xscale("log")
    ax_rate.set_xlabel("Δt (ms)", fontsize=theme.SIZE_LABEL)
    ax_rate.set_ylabel("Hidden E rate (Hz)",
                       fontsize=theme.SIZE_LABEL, color=theme.INK_BLACK)
    ax_rate.tick_params(axis="y", labelcolor=theme.INK_BLACK)
    ax_rate.set_ylim(0, 50)
    ax_rate.set_xticks(dts_sorted)
    ax_rate.set_xticklabels([f"{d:g}" for d in dts_sorted])
    ax_rate.spines["top"].set_visible(False)

    ax_acc = ax_rate.twinx()
    ax_acc.errorbar(
        dts_sorted, acc_means, yerr=acc_sems,
        marker="s", markersize=6, lw=1.4, color=theme.DEEP_RED,
        capsize=3, label="accuracy (%)",
    )
    ax_acc.set_ylabel("Test accuracy (%)",
                      fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED)
    ax_acc.tick_params(axis="y", labelcolor=theme.DEEP_RED)
    ax_acc.set_ylim(0, 100)
    ax_acc.spines["top"].set_visible(False)

    fig.suptitle(
        "PING Δt audit — post-training E rate and accuracy vs integration timestep",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_raster_strip(
    samples: list[dict], out_path: Path, run_id: str, t_window_ms: float,
) -> None:
    """Single-trial rasters across Δt, one panel per Δt. X-axis is
    physical time in ms (not steps), so gamma cycle alignment is read
    by eye — same physical period if dynamics survive Δt change."""
    theme.apply()
    n = len(samples)
    n_e = RASTER_N_E_PLOT
    n_i = RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n, 1, figsize=(6.9, 0.7 * n + 0.8),
        sharex=True, gridspec_kw={"hspace": 0.22},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples)):
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt_ms"]
        # Truncate display to the first t_window_ms ms so cycles are visible.
        mask = t_axis <= t_window_ms
        e_t, e_n = np.where(s["e"][mask])
        i_t, i_n = np.where(s["i"][mask])
        ax.scatter(t_axis[mask][e_t], e_n,
                   s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.4)
        ax.scatter(t_axis[mask][i_t], i_n + n_e + gap,
                   s=2.0, c=theme.DEEP_RED, marker="|", linewidths=0.4)
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, t_window_ms)
        ax.text(
            1.012, 0.5,
            f"Δt = {s['dt_ms']:g} ms\nE = {s['e_rate_hz']:.1f} Hz",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=theme.SIZE_LABEL,
        )
        if i == 0:
            ax.set_title(
                "Trained-PING rasters at each Δt (seed 42, MNIST digit 0 sample 0) — "
                "x-axis is physical time in ms"
            )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


# ─── success criteria ───────────────────────────────────────────────


def plot_training_curves(out_path: Path, run_id: str) -> None:
    """Per-cell training-trajectory curves. One line per (Δt, seed);
    colour by Δt (viridis on log Δt)."""
    theme.apply()
    cmap = plt.get_cmap("viridis")
    dts_sorted = list(DT_SWEEP_MS)
    fig, (ax_acc, ax_rate) = plt.subplots(
        2, 1, figsize=(5.6, 4.6), sharex=True,
        gridspec_kw={"hspace": 0.15},
    )
    for i, dt_ms in enumerate(dts_sorted):
        color = cmap(i / max(1, len(dts_sorted) - 1))
        for j, seed in enumerate(SEEDS):
            mfile = cell_dir(dt_ms, seed) / "metrics.json"
            if not mfile.exists():
                continue
            m = json.loads(mfile.read_text())
            eps = [e["ep"] for e in m["epochs"]]
            accs = [e.get("acc", 0) for e in m["epochs"]]
            rates = [e.get("test_rate_e", 0) for e in m["epochs"]]
            label = f"Δt = {dt_ms:g} ms" if j == 0 else None
            ax_acc.plot(eps, accs, color=color, lw=1.0, alpha=0.85, label=label)
            ax_rate.plot(eps, rates, color=color, lw=1.0, alpha=0.85)
    ax_acc.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax_rate.set_ylabel("Test E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax_rate.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    ax_acc.legend(fontsize=theme.SIZE_LEGEND, frameon=False, ncol=2, loc="lower right")
    ax_acc.set_ylim(0, 100)
    for ax in (ax_acc, ax_rate):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, lw=0.4)
    fig.suptitle(
        "Per-cell training curves — convergence check across Δt sweep",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)

def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    skip_training = "--skip-training" in sys.argv
    only_missing = "--only-missing" in sys.argv
    wipe_dir = "--no-wipe-dir" not in sys.argv

    # Publication profile: every figure this notebook writes is a print-sized
    # vector, emitted as both SVG (docs) and PDF (manuscript) by save_figure.
    theme.set_paper_mode(True)

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    n_cells = len(DT_SWEEP_MS) * len(SEEDS)
    print(
        f"notebook_run_id = {notebook_run_id} tier={tier} cells={n_cells}"
        + ("  [skip-training]" if skip_training else "")
        + (f"  [modal:{modal_gpu}]" if modal_gpu else "")
    )

    prepare_run_dirs(
        SLUG, notebook_run_id, wipe=wipe_dir, skip_training=skip_training,
        make_artifacts=False,
    )

    # Training lives in nb022 now (train-once / reuse-many): the dt sweep is a
    # registry family there (the documented dt exception). This notebook only
    # consumes the cells.

    from cli import _auto_device

    device = _auto_device()
    print(f"device = {device}")

    rows: list[dict] = []
    for dt_ms in DT_SWEEP_MS:
        for seed in SEEDS:
            run_dir = cell_dir(dt_ms, seed)
            if not (run_dir / "weights.pth").exists():
                raise SystemExit(f"missing weights: {run_dir / 'weights.pth'}")
            t0 = time.monotonic()
            res = measure_rate_acc(run_dir, device)
            res["seed"] = seed
            rows.append(res)
            # Cleanup between cells — without this, MPS memory pressure
            # causes the next torch.load to mis-dispatch to legacy tar
            # format and raise KeyError 'storages'.
            import gc
            import torch as _t
            gc.collect()
            if hasattr(_t, "mps") and _t.backends.mps.is_available():
                _t.mps.empty_cache()
            print(
                f"  Δt={dt_ms:>5.2f}ms seed={seed}  "
                f"acc={res['acc']:5.2f}%  E={res['e_rate_hz']:6.2f} Hz  "
                f"I={res['i_rate_hz']:6.2f} Hz  ({time.monotonic() - t0:.1f}s)"
            )

    plot_dt_sweep(rows, FIGURES / "dt_sweep", notebook_run_id)
    print(f"wrote {FIGURES / 'dt_sweep'}.{{svg,pdf}}")

    # Raster strip — one trial per Δt, all from seed 42.
    raster_seed = SEEDS[0]
    print(f"[raster] single-trial panels from seed {raster_seed}, "
          f"sample {RASTER_SAMPLE_IDX}")
    samples = []
    for dt_ms in DT_SWEEP_MS:
        run_dir = cell_dir(dt_ms, raster_seed)
        samples.append(capture_raster(run_dir, RASTER_SAMPLE_IDX, device))
    plot_raster_strip(
        samples, FIGURES / "raster_strip", notebook_run_id,
        t_window_ms=RASTER_T_WINDOW_MS,
    )
    print(f"wrote {FIGURES / 'raster_strip'}.{{svg,pdf}}")
    plot_training_curves(FIGURES / "training_curves", notebook_run_id)
    print(f"wrote {FIGURES / 'training_curves'}.{{svg,pdf}}")

    duration_s = time.monotonic() - t_start
    train_cfg = load_config(cell_dir(DT_SWEEP_MS[0], SEEDS[0]))
    summary = {
        "notebook_run_id": notebook_run_id,
        "git_sha": train_cfg.get("git_sha"),
        "duration_s": round(duration_s, 1),
        "duration": format_duration(duration_s),
        "tier": tier,
        "config": {
            "tier": tier,
            "dataset": "mnist",
            "dt_sweep_ms": list(DT_SWEEP_MS),
            "seeds": list(SEEDS),
            "batch_size": BATCH_SIZE,
            "max_samples": TIER_CONFIG[tier]["max_samples"],
            "epochs": TIER_CONFIG[tier]["epochs"],
            "t_ms": T_MS,
        },
        "results": rows,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")



if __name__ == "__main__":
    main()
