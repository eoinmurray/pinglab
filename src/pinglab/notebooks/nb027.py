"""Notebook runner for entry 027 — COBA on MNIST with SGCC instead of
the global v-grad-dampen.

nb024 showed that COBA can be BPTT-trained on MNIST at small tier
*only* when the conductance Jacobian is muted via --v-grad-dampen 1000
— a heavy-handed global scaling that nb026 then revealed kills the
gradient signal needed for any task longer than a couple hundred
milliseconds.

[Burghi, Pugliese Carratelli & Rule 2024](https://arxiv.org/) propose
Surrogate Gradients by Costate Control (SGCC) — a surgical
replacement that scales only the voltage↔conductance cross-coupling
gradient (the path that actually explodes), leaving diagonal /
parameter gradient flow alone. Their K^gat controller is what we
implement as the --sgcc / --sgcc-alpha flag pair.

This entry trains a fresh coba on MNIST with --sgcc-alpha 0.5 (no
v-grad-dampen) and compares accuracy + firing rate to nb024's
coba__off__seed42 baseline. Same recipe, only the gradient stabilizer
differs.

Notebook entry: src/docs/src/pages/notebooks/nb027.mdx
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

SLUG = "nb027"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope/__main__.py"

# nb024's coba baseline — trained with --v-grad-dampen 1000. The
# reference point we're comparing against.
NB024_COBA = REPO / "src" / "artifacts" / "notebooks" / "nb024" / "coba__off__seed42"

# ── Recipe ────────────────────────────────────────────────────────────
T_MS_TRAIN = 200.0
DT_TRAIN = 0.1
T_MS_TRIAL = 400.0  # for post-train raster
SGCC_ALPHA = 0.5
SAMPLE_IDX = 0
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

# Mirror nb024's coba recipe EXACTLY, then strip --v-grad-dampen and
# add --sgcc. Everything else identical so the only variable is the
# gradient stabilizer.
COBA_SGCC_RECIPE = {
    "--ei-strength": "0",
    "--w-in": "0.3",
    "--w-in-sparsity": "0.95",
    "--readout": "mem-mean",
    "--surrogate-slope": "1",
    "--readout-w-out-scale": "100",
    "--lr": "0.0004",
    "--batch-size": "256",
    # SGCC replaces v-grad-dampen
    "--sgcc": True,
    "--sgcc-alpha": str(SGCC_ALPHA),
    # No --v-grad-dampen → defaults to 1 (off)
}


def sgcc_train_dir() -> Path:
    return ARTIFACTS / f"coba_sgcc__seed{SEED}"


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


def train_coba_sgcc(tier: str) -> Path:
    """Train coba on MNIST with SGCC enabled. Same nb024 recipe
    otherwise — except v-grad-dampen stays at its default 1.0 (off)."""
    import subprocess

    out_dir = sgcc_train_dir()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "train",
        "--model", "ping",  # COBANet dispatch
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
    ]
    for k, v in COBA_SGCC_RECIPE.items():
        if v is True:
            args.append(k)
        elif v is not None:
            args += [k, v]
    cmd = ["uv", "run", "python", str(OSCILLOSCOPE), *args]
    print(f"[train-sgcc] coba+sgcc(α={SGCC_ALPHA}): {' '.join(args)}")
    subprocess.run(cmd, cwd=REPO, check=True)
    return out_dir


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_metrics_jsonl(run_dir: Path) -> list[dict]:
    """Per-epoch training metrics."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def capture_raster(train_dir: Path) -> dict:
    """Replay the trained network on MNIST digit 0 for 400 ms."""
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
    )
    if hasattr(net, "readout_mode"):
        net.readout_mode = cfg.get("readout_mode", "mem-mean")

    state = torch.load(train_dir / "weights.pth", map_location=device)
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True

    X_b = torch.from_numpy(X_te[SAMPLE_IDX : SAMPLE_IDX + 1]).to(device)
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
        "e": e_full, "i": i_full,
        "dt": float(cfg["dt"]), "t_total_ms": T_MS_TRIAL,
        "e_rate_hz": e_rate, "i_rate_hz": i_rate,
    }


def plot_compare_rasters(
    baseline: dict, sgcc: dict, out_path: Path, run_id: str
) -> None:
    """2-panel comparison: baseline (top, v_grad_dampen=1000) vs SGCC."""
    theme.apply()
    n_e = RASTER_N_E_PLOT
    n_i = RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        2, 1, figsize=(10.0, 5.625), sharex=True,
        gridspec_kw={"hspace": 0.22, "left": 0.15, "right": 0.97,
                     "top": 0.92, "bottom": 0.12},
    )
    rng = np.random.default_rng(0)
    panels = [
        ("baseline (v-grad-dampen 1000)", baseline),
        (f"SGCC (α = {SGCC_ALPHA:g})", sgcc),
    ]
    for ax, (label, s) in zip(axes, panels):
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
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, s["t_total_ms"])
        ax.text(
            0.985, 0.97,
            f"E = {s['e_rate_hz']:.1f} Hz   I = {s['i_rate_hz']:.1f} Hz",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=theme.SIZE_LABEL,
            bbox={"facecolor": theme.PAPER, "alpha": 0.85,
                  "edgecolor": "none", "pad": 2},
        )
        ax.text(
            -0.08, 0.5, label,
            transform=ax.transAxes, ha="right", va="center",
            fontsize=theme.SIZE_LABEL, color=theme.INK_BLACK,
        )
    axes[-1].set_xlabel("time (ms)")
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_learning_curves(
    baseline_metrics: list[dict], sgcc_metrics: list[dict],
    out_path: Path, run_id: str,
) -> None:
    """Train + test accuracy per epoch, both stabilizers overlaid."""
    theme.apply()
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10.0, 5.625))
    for name, color, ms in [
        ("baseline", theme.GREY_MID, baseline_metrics),
        (f"SGCC α={SGCC_ALPHA:g}", theme.DEEP_RED, sgcc_metrics),
    ]:
        if not ms:
            continue
        epochs = [m.get("ep", i + 1) for i, m in enumerate(ms)]
        loss = [m.get("loss", float("nan")) for m in ms]
        acc = [m.get("acc", float("nan")) for m in ms]
        ax_loss.plot(epochs, loss, marker="o", color=color, label=name)
        ax_acc.plot(epochs, acc, marker="o", color=color, label=name)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("train loss")
    ax_loss.set_title("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("test accuracy (%)")
    ax_acc.set_title("Test accuracy")
    ax_acc.set_ylim(0, 100)
    ax_acc.axhline(10.0, ls=":", color=theme.FAINT, lw=0.8)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="lower right")
    fig.suptitle("coba — gradient stabilizer comparison", fontsize=theme.SIZE_TITLE)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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

    if not (NB024_COBA / "weights.pth").exists():
        raise SystemExit(
            f"missing nb024 coba baseline at {NB024_COBA}; run nb024 first"
        )

    if not skip_training:
        train_coba_sgcc(tier)
    else:
        if not (sgcc_train_dir() / "weights.pth").exists():
            raise SystemExit(
                f"--skip-training but {sgcc_train_dir()} has no weights"
            )

    # Final metrics
    baseline_m = load_metrics(NB024_COBA)
    sgcc_m = load_metrics(sgcc_train_dir())
    print(
        f"[final] baseline (v-grad-dampen 1000): best_acc = "
        f"{baseline_m.get('best_acc', '?')}, rate_e = "
        f"{baseline_m.get('rate_e', '?')} Hz"
    )
    print(
        f"[final] SGCC (α = {SGCC_ALPHA}):          best_acc = "
        f"{sgcc_m.get('best_acc', '?')}, rate_e = "
        f"{sgcc_m.get('rate_e', '?')} Hz"
    )

    # Per-epoch curves
    baseline_jsonl = load_metrics_jsonl(NB024_COBA)
    sgcc_jsonl = load_metrics_jsonl(sgcc_train_dir())
    plot_learning_curves(
        baseline_jsonl, sgcc_jsonl,
        FIGURES / "learning_curves.png", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'learning_curves.png'}")

    # Rasters
    baseline_rast = capture_raster(NB024_COBA)
    sgcc_rast = capture_raster(sgcc_train_dir())
    print(
        f"  baseline raster: E = {baseline_rast['e_rate_hz']:.2f} Hz, "
        f"I = {baseline_rast['i_rate_hz']:.2f} Hz"
    )
    print(
        f"  sgcc     raster: E = {sgcc_rast['e_rate_hz']:.2f} Hz, "
        f"I = {sgcc_rast['i_rate_hz']:.2f} Hz"
    )
    plot_compare_rasters(
        baseline_rast, sgcc_rast,
        FIGURES / "compare_rasters.png", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'compare_rasters.png'}")

    # Copy training videos
    for label, src_dir in [("baseline", NB024_COBA), ("sgcc", sgcc_train_dir())]:
        src = src_dir / "training.mp4"
        dst = FIGURES / f"training__{label}.mp4"
        if src.exists():
            shutil.copy2(src, dst)
            print(f"wrote {dst}")

    duration_s = time.monotonic() - t_start
    summary = {
        "notebook_run_id": notebook_run_id,
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "tier": tier,
        "config": {
            "tier": tier,
            "sgcc_alpha": SGCC_ALPHA,
            "seed": SEED,
            "model": "coba",
        },
        "results": {
            "baseline": {
                "best_acc": baseline_m.get("best_acc"),
                "rate_e_hz": baseline_m.get("rate_e"),
                "rate_e_replay_hz": baseline_rast["e_rate_hz"],
                "rate_i_replay_hz": baseline_rast["i_rate_hz"],
            },
            "sgcc": {
                "best_acc": sgcc_m.get("best_acc"),
                "rate_e_hz": sgcc_m.get("rate_e"),
                "rate_e_replay_hz": sgcc_rast["e_rate_hz"],
                "rate_i_replay_hz": sgcc_rast["i_rate_hz"],
            },
        },
        "success_criteria": [
            {
                "label": "SGCC training completes without NaN",
                "passed": sgcc_m.get("best_acc") is not None
                          and isinstance(sgcc_m.get("best_acc"), (int, float))
                          and float(sgcc_m["best_acc"]) > 0,
                "detail": f"best_acc = {sgcc_m.get('best_acc', '?')}",
            },
            {
                "label": "SGCC accuracy within 10pp of baseline",
                "passed": (
                    isinstance(sgcc_m.get("best_acc"), (int, float))
                    and isinstance(baseline_m.get("best_acc"), (int, float))
                    and abs(float(sgcc_m["best_acc"]) - float(baseline_m["best_acc"])) <= 10
                ),
                "detail": (
                    f"baseline {baseline_m.get('best_acc', '?')}%, "
                    f"sgcc {sgcc_m.get('best_acc', '?')}%"
                ),
            },
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")


if __name__ == "__main__":
    main()
