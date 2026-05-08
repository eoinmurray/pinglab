"""Notebook runner for entry 018 — coba-to-ping weight transfer.

Trains a coba network (COBANet dispatched via ping arch with
ei_strength=0, so the E→I→E loop is disabled), then runs inference on
the held-out test set across an ei_strength sweep with the *same
trained W_ff and W_ee* — measuring how classification accuracy and
hidden firing rates degrade as inhibitory feedback the network never
saw during training is engaged at progressively higher strength.

The I-loop weights (W_ei, W_ie) are *not* loaded from the trained
state — they would arrive as zero matrices because ei_strength=0
during training scales their init by zero, and W_ei/W_ie carry no
gradient (requires_grad=False), so they stay zero in the checkpoint.
Loading them would silently nullify the inhibitory loop at any
ei_strength. Instead we re-initialise them at the inference-time
ei_strength so each sweep point actually has the corresponding
I-loop active.

Notebook entry: src/docs/src/pages/notebooks/nb019.mdx
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402
from pinglab import theme  # noqa: E402

SLUG = "nb019"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope/__main__.py"

TIER_CONFIG = {
    "extra small": dict(max_samples=100, epochs=1),
    "small": dict(max_samples=500, epochs=5),
    "medium": dict(max_samples=2000, epochs=10),
    "large": dict(max_samples=5000, epochs=40),
    "extra large": dict(max_samples=10000, epochs=40),
}
DEFAULT_TIER = "small"
T_MS = 200.0
DT_TRAIN = 0.1
SEED = 42

# Inference ei_strength sweep. 0.0 reproduces training conditions (coba
# native, baseline); 1.0 is the upper end of the canonical PING-on regime
# (nb013 ping recipe). 0.1 spacing is dense enough to resolve a degradation
# knee if there is one.
EI_SWEEP = [round(0.1 * i, 1) for i in range(11)]  # 0.0, 0.1, …, 1.0

# Subset of the sweep at which we also record E + I rasters from a single
# trial — picked to span the dynamical regimes: silent I (0.0), I-onset
# (0.2), crossover (0.4), saturation onset (0.6), full PING (0.8, 1.0).
EI_RASTER = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
RASTER_SAMPLE_IDX = 0  # which test sample to record
RASTER_N_E_PLOT = 200  # subsample of E units to plot
RASTER_N_I_PLOT = 64  # subsample of I units to plot

# Per-tier accuracy floor applied to the ei=0 baseline only. Sweep
# accuracies are reported and the lowest is gated to be above chance,
# but their absolute position is the experimental finding.
MIN_ACC_BY_TIER = {
    "extra small": 15.0,
    "small": 30.0,
    "medium": 50.0,
    "large": 70.0,
    "extra large": 70.0,
}


def build_train_args(tier: str, out_dir: Path) -> list[str]:
    # Same recipe as nb011 — coba via ping arch with ei_strength=0.
    return [
        "train",
        "--model",
        "ping",
        "--dataset",
        "mnist",
        "--max-samples",
        str(TIER_CONFIG[tier]["max_samples"]),
        "--epochs",
        str(TIER_CONFIG[tier]["epochs"]),
        "--t-ms",
        str(T_MS),
        "--dt",
        str(DT_TRAIN),
        "--seed",
        str(SEED),
        "--out-dir",
        str(out_dir),
        "--wipe-dir",
        "--ei-strength",
        "0",
        "--v-grad-dampen",
        "1000",
        "--w-in",
        "0.3",
        "--w-in-sparsity",
        "0.95",
        "--readout",
        "mem-mean",
        "--surrogate-slope",
        "1",
        "--readout-w-out-scale",
        "100",
        "--lr",
        "0.0004",
        "--batch-size",
        "256",
    ]


def run_inproc_infer(
    train_dir: Path, ei_strength: float, out_dir: Path
) -> dict:
    """Build a fresh ping net at the requested ei_strength, load *only*
    W_ff and W_ee from the trained checkpoint (skip W_ei/W_ie so the
    freshly-initialised I-loop survives), evaluate accuracy + mean E
    firing rate on the canonical test split.

    The CLI infer subcommand can't be used here because it always loads
    the full state dict; the trained-coba checkpoint has W_ei = W_ie = 0
    (init scaled by ei_strength=0, no gradient updates them) and loading
    those would nullify the I-loop at any inference ei_strength.
    """
    import torch  # noqa: E402

    # Match the per-process module setup oscilloscope.infer does, but
    # without the argparse / sys.exit envelope.
    import config as C  # noqa: E402
    import models as M  # noqa: E402
    from config import build_net, patch_dt  # noqa: E402
    from oscilloscope import (  # noqa: E402
        EVAL_SEED,
        _auto_device,
        encode_batch,
        load_dataset,
        seed_everything,
    )

    cfg = json.loads((train_dir / "config.json").read_text())

    seed_everything(int(cfg.get("seed", SEED)))
    M.T_ms = float(cfg["t_ms"])
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

    from torch.utils.data import DataLoader, TensorDataset

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=64,
    )

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
        ei_strength=float(ei_strength),
        ei_ratio=float(cfg.get("ei_ratio") or 2.0),
        sparsity=float(cfg.get("sparsity") or 0.0),
        device=device,
        randomize_init=not bool(cfg.get("kaiming_init", False)),
        kaiming_init=bool(cfg.get("kaiming_init", False)),
        dales_law=bool(cfg.get("dales_law", True)),
        hidden_sizes=hidden_sizes,
        w_rec=cfg.get("w_rec"),
        rec_layers=cfg.get("rec_layers"),
        ei_layers=cfg.get("ei_layers"),
    )
    if hasattr(net, "readout_mode"):
        net.readout_mode = cfg.get("readout_mode", "mem-mean")

    state = torch.load(train_dir / "weights.pth", map_location=device)
    # Selective transfer: keep input/hidden FF + E→E recurrence + readout
    # from the trained net, preserve the freshly-init I-loop weights.
    skipped = {
        k: v for k, v in state.items() if k.startswith(("W_ei.", "W_ie."))
    }
    keep = {k: v for k, v in state.items() if k not in skipped}
    missing, unexpected = net.load_state_dict(keep, strict=False)
    print(
        f"  [transfer-load] loaded {len(keep)} keys, skipped "
        f"{sorted(skipped.keys())}; missing={list(missing)} "
        f"unexpected={list(unexpected)}"
    )

    net.eval()
    correct = total = 0
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    rate_sums: dict[str, float] = {}
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
            logits = net(input_spikes=spk)
            correct += (logits.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
            batch_rates = getattr(net, "rates", None) or {}
            B = y_b.size(0)
            for k, v in batch_rates.items():
                rate_sums[k] = rate_sums.get(k, 0.0) + float(v) * B

    acc = 100.0 * correct / total
    rates_hz = {k: v / total for k, v in rate_sums.items()} if total else {}
    hid_key = next((k for k in rates_hz if k.startswith("hid")), None)
    inh_key = next((k for k in rates_hz if k.startswith("inh")), None)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "mode": "infer",
        "ei_strength": ei_strength,
        "best_acc": acc,
        "n_correct": correct,
        "n_total": total,
        "rates_hz": rates_hz,
        "hid_rate_hz": rates_hz.get(hid_key) if hid_key else None,
        "inh_rate_hz": rates_hz.get(inh_key) if inh_key else None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"  ei={ei_strength:g}: acc={acc:.2f}%  "
        f"hid={metrics['hid_rate_hz']:.1f}Hz  "
        + (f"inh={metrics['inh_rate_hz']:.1f}Hz" if inh_key else "")
    )
    return metrics


def capture_raster(train_dir: Path, ei_strength: float, sample_idx: int) -> dict:
    """Build a fresh ping net at ei_strength, load the same selective state
    dict as run_inproc_infer, then record one forward pass on a single
    test sample with spike-recording on. Returns subsampled E and I
    spike-time tensors for plotting."""
    import numpy as np  # noqa: E402
    import torch  # noqa: E402

    import config as C  # noqa: E402
    import models as M  # noqa: E402
    from config import build_net, patch_dt  # noqa: E402
    from oscilloscope import (  # noqa: E402
        EVAL_SEED,
        _auto_device,
        encode_batch,
        load_dataset,
        seed_everything,
    )

    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", SEED)))
    M.T_ms = float(cfg["t_ms"])
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
        ei_strength=float(ei_strength),
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
    keep = {k: v for k, v in state.items() if not k.startswith(("W_ei.", "W_ie."))}
    net.load_state_dict(keep, strict=False)
    net.eval()
    net.recording = True

    X_b = torch.from_numpy(X_te[sample_idx : sample_idx + 1]).to(device)
    y_b = int(y_te[sample_idx])
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    with torch.no_grad():
        spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
        _ = net(input_spikes=spk)

    e_full = net.spike_record["hid"].cpu().numpy()  # (T, N_E) — B=1 squeezed
    i_full = net.spike_record["inh"].cpu().numpy()  # (T, N_I)
    rng = np.random.default_rng(0)
    e_idx = np.sort(rng.choice(e_full.shape[1], RASTER_N_E_PLOT, replace=False))
    i_idx = np.sort(rng.choice(i_full.shape[1], RASTER_N_I_PLOT, replace=False))
    return {
        "ei_strength": float(ei_strength),
        "label": y_b,
        "e": e_full[:, e_idx].astype(bool),
        "i": i_full[:, i_idx].astype(bool),
        "dt": float(cfg["dt"]),
        "t_ms": float(cfg["t_ms"]),
    }


def _stamp(fig, run_id: str) -> None:
    fig.text(
        0.995,
        0.005,
        run_id,
        ha="right",
        va="bottom",
        fontsize=theme.SIZE_CAPTION,
        color=theme.LABEL,
        family="monospace",
    )


def plot_rasters(samples: list[dict], out_path: Path, run_id: str) -> None:
    """One row per ei value; I units stack over E units so the PING-style
    E-then-I cadence reads as alternating bursts when it appears. E in
    INK_BLACK, I in DEEP_RED — same categorical assignment as nb015."""
    import numpy as np  # noqa: E402

    theme.apply()
    n = len(samples)
    n_e = RASTER_N_E_PLOT
    n_i = RASTER_N_I_PLOT
    gap = 6  # blank rows between E and I for visual separation
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(10.0, 5.625 + 0.6 * max(n - 4, 0)),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples)):
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt"]
        e_t, e_n = np.where(s["e"])
        i_t, i_n = np.where(s["i"])
        ax.scatter(
            t_axis[e_t], e_n,
            s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.4,
        )
        ax.scatter(
            t_axis[i_t], i_n + n_e + gap,
            s=2.0, c=theme.DEEP_RED, marker="|", linewidths=0.4,
        )
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, s["t_ms"])
        # ei label on the right margin so the E/I ticks own the y axis.
        ax.text(
            1.012, 0.5, f"ei = {s['ei_strength']:g}",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=theme.SIZE_LABEL,
        )
        if i == 0:
            ax.set_title("E (black) and I (red) spikes — single trial, MNIST test sample 0")
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def load_metrics(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text())


def plot_acc_sweep(points: list[dict], out_path: Path, run_id: str) -> None:
    theme.apply()
    eis = [p["ei_strength"] for p in points]
    accs = [p["acc"] for p in points]
    base_acc = points[0]["acc"]
    worst = min(points, key=lambda p: p["acc"])
    # Anchor at zero so the magnitude of the drop reads honestly against
    # the chance floor; head room above baseline for the annotation.
    y_hi = min(max(accs + [base_acc]) + 6, 100)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(
        base_acc, color=theme.LABEL, lw=1.0, ls="--",
        label=f"baseline {base_acc:.1f}%",
    )
    ax.axhline(
        10.0, color=theme.FAINT, lw=1.0, ls=":",
        label="chance (10%)",
    )
    ax.plot(eis, accs, marker="o", color=theme.DEEP_RED, label="transfer")
    ax.annotate(
        f"{worst['acc']:.1f}%  (Δ {worst['acc'] - base_acc:+.1f} pp)",
        xy=(worst["ei_strength"], worst["acc"]),
        xytext=(8, -14), textcoords="offset points",
        fontsize=theme.SIZE_ANNOTATION,
    )
    ax.set_xlabel("inference E→I strength")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Transfer accuracy across the I-loop sweep")
    ax.set_ylim(0, y_hi)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([round(0.1 * i, 1) for i in range(11)])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_rates_sweep(points: list[dict], out_path: Path, run_id: str) -> None:
    theme.apply()
    eis = [p["ei_strength"] for p in points]
    hid = [p.get("hid_rate_hz") or 0.0 for p in points]
    inh = [p.get("inh_rate_hz") or 0.0 for p in points]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(eis, hid, marker="o", color=theme.INK_BLACK, label="E (hidden)")
    ax.plot(eis, inh, marker="s", color=theme.DEEP_RED, label="I (inhibitory)")
    ax.set_xlabel("inference E→I strength")
    ax.set_ylabel("mean population rate (Hz)")
    ax.set_title("E and I population rates across the I-loop sweep")
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([round(0.1 * i, 1) for i in range(11)])
    ax.set_ylim(0, max(hid + inh) * 1.12 + 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _stamp(fig, run_id)
    fig.savefig(out_path)
    plt.close(fig)


def evaluate_success(
    points: list[dict], tier: str, figures: Path
) -> list[dict]:
    floor = float(MIN_ACC_BY_TIER[tier])
    figs_root = figures.parents[2]

    def artifact(name: str, label: str) -> dict:
        path = figures / name
        ok = path.exists() and path.stat().st_size > 0
        href = "/" + str(path.relative_to(figs_root)) if ok else None
        return {
            "label": label,
            "passed": bool(ok),
            "detail": f"{path.name} ({path.stat().st_size} bytes)"
            if ok
            else f"missing {path.name}",
            "detail_href": href,
        }

    crits = [
        artifact("acc_sweep.png", "accuracy sweep rendered"),
        artifact("rates_sweep.png", "firing-rate sweep rendered"),
        artifact("rasters.png", "raster grid rendered"),
    ]
    base = points[0]
    crits.append(
        {
            "label": f"baseline (ei=0) acc ≥ {floor:.1f}% ({tier} tier floor)",
            "passed": bool(base["acc"] >= floor),
            "detail": f"baseline={base['acc']:.2f}%",
        }
    )
    worst = min(points, key=lambda p: p["acc"])
    crits.append(
        {
            "label": "all sweep points above chance (≥ 20%)",
            "passed": bool(worst["acc"] >= 20.0),
            "detail": f"worst@ei={worst['ei_strength']:g}: {worst['acc']:.2f}%",
        }
    )
    return crits


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


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

    train_dir = ARTIFACTS / "train"
    if wipe_dir:
        if skip_training:
            if FIGURES.exists():
                print(f"[wipe] {FIGURES.relative_to(REPO)}")
                shutil.rmtree(FIGURES)
        else:
            for d in (ARTIFACTS, FIGURES):
                if d.exists():
                    print(f"[wipe] {d.relative_to(REPO)}")
                    shutil.rmtree(d)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)

    if skip_training:
        if not (train_dir / "weights.pth").exists():
            raise SystemExit(f"--skip-training requires existing weights at {train_dir}")
    else:
        dispatcher = BatchDispatcher(modal_gpu, REPO, OSCILLOSCOPE)
        print(f"[train] coba (ping arch, ei=0) → {train_dir.relative_to(REPO)}")
        dispatcher.submit(build_train_args(tier, train_dir), train_dir)
        dispatcher.drain()
        if not (train_dir / "weights.pth").exists():
            raise SystemExit(f"training did not produce {train_dir / 'weights.pth'}")

    # In-process inference: one pass per sweep point, each building a
    # fresh ping net at its ei_strength and loading only W_ff + W_ee from
    # the trained checkpoint (see module docstring).
    points: list[dict] = []
    for ei in EI_SWEEP:
        out = ARTIFACTS / f"infer_ei{ei:g}"
        print(f"[infer] ei={ei} → {out.relative_to(REPO)}")
        m = run_inproc_infer(train_dir, ei, out)
        points.append(
            {
                "ei_strength": ei,
                "acc": m["best_acc"],
                "hid_rate_hz": m.get("hid_rate_hz"),
                "inh_rate_hz": m.get("inh_rate_hz"),
                "n_total": m.get("n_total"),
            }
        )

    base_acc = points[0]["acc"]
    print("  ei sweep:")
    for p in points:
        delta = p["acc"] - base_acc
        print(
            f"    ei={p['ei_strength']:g}  acc={p['acc']:.2f}%  "
            f"hid={p['hid_rate_hz']:.1f}Hz  inh={(p['inh_rate_hz'] or 0.0):.1f}Hz  "
            f"Δ={delta:+.2f}pp"
        )

    plot_acc_sweep(points, FIGURES / "acc_sweep.png", notebook_run_id)
    print(f"wrote {FIGURES / 'acc_sweep.png'}")
    plot_rates_sweep(points, FIGURES / "rates_sweep.png", notebook_run_id)
    print(f"wrote {FIGURES / 'rates_sweep.png'}")

    print(f"[raster] capturing single-trial rasters for ei ∈ {EI_RASTER}")
    raster_samples = [
        capture_raster(train_dir, ei, RASTER_SAMPLE_IDX) for ei in EI_RASTER
    ]
    plot_rasters(raster_samples, FIGURES / "rasters.png", notebook_run_id)
    print(f"wrote {FIGURES / 'rasters.png'}")

    duration_s = time.monotonic() - t_start
    train_cfg = json.loads((train_dir / "config.json").read_text())
    train_metrics = json.loads((train_dir / "metrics.json").read_text())
    crits = evaluate_success(points, tier, FIGURES)

    summary = {
        "notebook_run_id": notebook_run_id,
        "git_sha": train_cfg.get("git_sha"),
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "tier": tier,
        "config": {
            "tier": tier,
            "dataset": "mnist",
            "model": "coba",
            "max_samples": TIER_CONFIG[tier]["max_samples"],
            "epochs": TIER_CONFIG[tier]["epochs"],
            "t_ms": train_cfg["t_ms"],
            "dt": DT_TRAIN,
            "n_hidden": train_cfg["n_hidden"],
            "input_rate_hz": train_cfg.get("input_rate"),
            "batch_size": train_cfg["batch_size"],
            "seed": SEED,
            "ei_sweep": EI_SWEEP,
        },
        "train": {
            "best_acc": train_metrics["best_acc"],
            "best_epoch": train_metrics["best_epoch"],
            "final_acc": train_metrics["epochs"][-1]["acc"],
            "run_finished_at": train_metrics.get("run_finished_at"),
        },
        "sweep": points,
        "baseline_acc": base_acc,
        "success_criteria": crits,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")

    for c in crits:
        mark = "pass" if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['label']} — {c['detail']}")
    if any(not c["passed"] for c in crits):
        sys.exit(1)


if __name__ == "__main__":
    main()
