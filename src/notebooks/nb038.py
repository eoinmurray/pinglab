"""Notebook runner for entry 038 — functional probes of trained
PING vs COBA.

Standalone runner with no cross-notebook helpers. Trains coba / ping
baselines, then runs three inference-only probes on the trained
networks:
- input-rate sweep: per-cell f-I curve on MNIST digit 0 + uniform
  Poisson input;
- COBA → PING I-loop transfer: replay trained COBA at eval-time
  ei_strength ∈ [0, 1] and watch the gamma cycle self-assemble; and
- (readout latency removed)
  wall-clock time and cumulative spike count.

Figures land in /figures/notebooks/nb038/ and the success-criteria
summary in nb038/numbers.json.

Notebook entry: src/docs/src/pages/notebooks/nb038.mdx
"""

from __future__ import annotations

import json
import shutil
import subprocess
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
from nb022 import cell_dir as shared_cell_dir, cell_name  # noqa: E402

SLUG = "nb038"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
OSCILLOSCOPE = REPO / "src" / "cli/cli.py"

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

# Baseline (θ_u = off) cells are trained at multiple seeds so the
# headline bar chart and learning curves can show mean ± SEM. The θ_u
# sweep cells stay single-seed — the frontier *shape* is dominated by
# the regulariser, not the seed.
SEEDS_BASELINE: list[int] = [42, 43, 44]
SEED_SWEEP: int = 42
BASELINE_EPOCHS: int = 30  # overrides TIER_CONFIG epochs for baselines

# Inference-time ei_strength sweep on the coba__off__seed42 baseline.
# Subsumes the now-retired nb019 — trains nothing new; just runs the
# already-trained coba weights forward through the test set with a
# fresh ping-arch I-loop at progressively higher ei_strength.
EI_SWEEP: list[float] = [round(0.1 * i, 1) for i in range(11)]  # 0.0–1.0
EI_RASTER: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
EI_RASTER_SAMPLE_IDX: int = 0
EI_RASTER_N_E_PLOT: int = 200
EI_RASTER_N_I_PLOT: int = 64

# θ_u sweep grid in spikes-per-trial. None = no penalty (baseline).
# At T = 200 ms, spikes/trial × 5 = Hz. The grid spans from no
# pressure (off → ~80 Hz coba baseline) down to 1 Hz —
# below ping's natural 5 Hz and into the regime where every model
# loses accuracy.
THETA_U_GRID: list[float | None] = [None, 5.0, 2.0, 1.0, 0.5, 0.2]
FR_STRENGTH_UPPER = 1e-3

MODELS = ["coba", "ping"]

MODEL_RECIPES: dict[str, dict] = {
    "coba": {
        "__build_as": "ping",
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
        "__build_as": "ping",
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

MODEL_COLORS = {
    "coba": theme.DEEP_RED,
    "ping": theme.INK_BLACK,
}
MODEL_MARKERS = {"coba": "s", "ping": "D"}

MIN_ACC_BY_TIER = {
    "extra small": 15.0,
    "small": 30.0,
    "medium": 50.0,
    "large": 70.0,
    "extra large": 70.0,
}


def theta_label(theta_u: float | None) -> str:
    """Filesystem-safe label for an out-dir / video filename."""
    if theta_u is None:
        return "off"
    s = f"{theta_u:g}".replace(".", "p")
    return f"tu{s}"


def theta_display(theta_u: float | None) -> str:
    """Human label for plots / numbers.json."""
    if theta_u is None:
        return "off"
    return f"{theta_u:g}"


def theta_hz(theta_u: float | None) -> float | None:
    if theta_u is None:
        return None
    return theta_u * (1000.0 / T_MS)


def seeds_for(theta_u: float | None) -> list[int]:
    """Baseline cells run all seeds; sweep cells stay single-seed."""
    return list(SEEDS_BASELINE) if theta_u is None else [SEED_SWEEP]


def cell_dir(model: str, theta_u: float | None, seed: int) -> Path:
    """Trained cell — now the shared nb022 cell (train-once / reuse-many).
    nb022 owns the θ_u sweep; this notebook only consumes it."""
    return shared_cell_dir(cell_name(model, theta_u, seed))


def baseline_dir(model: str, seed: int = SEEDS_BASELINE[0]) -> Path:
    return cell_dir(model, None, seed)


def build_train_args(
    model: str, theta_u: float | None, seed: int, tier: str, out_dir: Path
) -> list[str]:
    recipe = MODEL_RECIPES[model]
    args = [
        "train",
        "--model", recipe["__build_as"],
        "--dataset", "mnist",
        "--max-samples", str(TIER_CONFIG[tier]["max_samples"]),
        "--epochs", str(BASELINE_EPOCHS),
        "--t-ms", str(T_MS),
        "--dt", str(DT_TRAIN),
        "--seed", str(seed),
        "--observe", "video",
        "--frame-rate", "1",
        "--out-dir", str(out_dir),
        "--wipe-dir",
    ]
    for k, v in recipe.items():
        if k.startswith("__"):
            continue
        if v is True:
            args.append(k)
        elif v is not None:
            args += [k, v]
    if theta_u is not None:
        args += [
            "--fr-reg-upper-theta", str(theta_u),
            "--fr-reg-upper-strength", str(FR_STRENGTH_UPPER),
        ]
    return args


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())





def generate_rate_sweep_video(model: str, out_path: Path) -> None:
    """Replay the trained baseline (θ_u = off) network on one MNIST digit
    while sweeping the input Poisson rate. The oscilloscope writes
    scan.mp4 into a per-call out-dir; we copy it to out_path."""
    artifact_dir = ARTIFACTS / f"rate_sweep__{model}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scan_mp4 = artifact_dir / "scan.mp4"
    if scan_mp4.exists():
        scan_mp4.unlink()
    argv = [
        "sim",
        "--video",
        "--from-dir", str(baseline_dir(model)),
        "--input", "dataset",
        "--dataset", "mnist",
        "--digit", "0",
        "--sample", "0",
        "--scan-var", "spike_rate",
        "--scan-min", "0",
        "--scan-max", "100",
        "--frames", "40",
        "--frame-rate", "10",
        # 400 ms gives PING's loop room to settle at each rate.
        "--t-ms", "400",
        "--out-dir", str(artifact_dir),
    ]
    cmd = ["uv", "run", "python", str(OSCILLOSCOPE), *argv]
    print(f"[rate-sweep] {model}: {' '.join(argv)}")
    subprocess.run(cmd, cwd=REPO, check=True)
    if not scan_mp4.exists():
        raise SystemExit(f"oscilloscope did not produce {scan_mp4}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(scan_mp4, out_path)




# ── COBA→PING ei_strength sweep (subsumes nb019) ──────────────────────


def _ei_sweep_dir() -> Path:
    return ARTIFACTS / "ei_sweep"


def run_inproc_infer(
    train_dir: Path, ei_strength: float, out_dir: Path
) -> dict:
    """Build a fresh ping net at the requested ei_strength, load only
    W_ff and W_ee from the trained coba checkpoint (skip W_ei/W_ie so
    the freshly-initialised I-loop survives), evaluate accuracy + mean
    E firing rate on the canonical test split.

    The CLI infer subcommand can't be used here because it always loads
    the full state dict; the trained-coba checkpoint has W_ei = W_ie = 0
    (init scaled by ei_strength=0, no gradient updates them) and loading
    those would nullify the I-loop at any inference ei_strength.
    """
    import torch

    import models as M
    from cli.config import build_net, patch_dt
    from cli import (
        EVAL_SEED,
        _auto_device,
        encode_batch,
        load_dataset,
        seed_everything,
    )

    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", SEEDS_BASELINE[0])))
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
        dales_law=bool(cfg.get("dales_law", True)),
        hidden_sizes=hidden_sizes,
        ei_layers=cfg.get("ei_layers"),
    )
    if hasattr(net, "readout_mode"):
        net.readout_mode = cfg.get("readout_mode", "mem-mean")

    state = torch.load(train_dir / "weights.pth", map_location=device)
    skipped = {k: v for k, v in state.items() if k.startswith(("W_ei.", "W_ie."))}
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


def capture_ei_raster(train_dir: Path, ei_strength: float, sample_idx: int) -> dict:
    """Single-trial raster: build a fresh ping net at ei_strength, load
    the same selective state dict as run_inproc_infer, record one
    forward pass on a single test sample."""
    import torch

    import models as M
    from cli.config import build_net, patch_dt
    from cli import (
        EVAL_SEED,
        _auto_device,
        encode_batch,
        load_dataset,
        seed_everything,
    )

    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", SEEDS_BASELINE[0])))
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

    e_full = net.spike_record["hid"].cpu().numpy()
    i_full = net.spike_record["inh"].cpu().numpy()
    rng = np.random.default_rng(0)
    e_idx = np.sort(rng.choice(e_full.shape[1], EI_RASTER_N_E_PLOT, replace=False))
    i_idx = np.sort(rng.choice(i_full.shape[1], EI_RASTER_N_I_PLOT, replace=False))
    return {
        "ei_strength": float(ei_strength),
        "label": y_b,
        "e": e_full[:, e_idx].astype(bool),
        "i": i_full[:, i_idx].astype(bool),
        "dt": float(cfg["dt"]),
        "t_ms": float(cfg["t_ms"]),
    }


def capture_rate_raster(train_dir: Path, spike_rate: float, sample_idx: int) -> dict:
    """Single-trial raster: load the trained ping baseline, override
    M.max_rate_hz, run one forward pass on a single test sample."""
    import torch

    import models as M
    from cli.config import build_net, patch_dt
    from cli import (
        EVAL_SEED,
        _auto_device,
        encode_batch,
        load_dataset,
        seed_everything,
    )

    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", SEEDS_BASELINE[0])))
    M.T_ms = float(cfg["t_ms"])
    patch_dt(float(cfg["dt"]))
    hidden_sizes = cfg.get("hidden_sizes") or [int(cfg["n_hidden"])]
    M.N_HID = hidden_sizes[-1]
    M.N_INH = hidden_sizes[-1] // 4
    M.HIDDEN_SIZES = list(hidden_sizes)
    M.max_rate_hz = float(spike_rate)
    M.p_scale = M.max_rate_hz * M.dt / 1000.0

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
        dales_law=bool(cfg.get("dales_law", True)),
        hidden_sizes=hidden_sizes,
    )
    if hasattr(net, "readout_mode"):
        net.readout_mode = cfg.get("readout_mode", "mem-mean")

    state = torch.load(train_dir / "weights.pth", map_location=device)
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True

    X_b = torch.from_numpy(X_te[sample_idx : sample_idx + 1]).to(device)
    y_b = int(y_te[sample_idx])
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    with torch.no_grad():
        spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
        _ = net(input_spikes=spk)

    e_full = net.spike_record["hid"].cpu().numpy()
    i_full = net.spike_record["inh"].cpu().numpy()
    # Mean per-cell firing rate over the trial, in Hz, for E and I.
    e_rate_hz = float(e_full.sum() / (e_full.shape[1] * cfg["t_ms"] / 1000.0))
    i_rate_hz = float(i_full.sum() / (i_full.shape[1] * cfg["t_ms"] / 1000.0))
    rng = np.random.default_rng(0)
    e_idx = np.sort(rng.choice(e_full.shape[1], EI_RASTER_N_E_PLOT, replace=False))
    i_idx = np.sort(rng.choice(i_full.shape[1], EI_RASTER_N_I_PLOT, replace=False))
    return {
        "spike_rate": float(spike_rate),
        "e_rate_hz": e_rate_hz,
        "i_rate_hz": i_rate_hz,
        "label": y_b,
        "e": e_full[:, e_idx].astype(bool),
        "i": i_full[:, i_idx].astype(bool),
        "dt": float(cfg["dt"]),
        "t_ms": float(cfg["t_ms"]),
    }


def plot_rate_rasters(samples: list[dict], out_path: Path, run_id: str) -> None:
    """One row per input-rate value; same E-over-I stacked layout as
    plot_ei_rasters so the two figures are visually comparable."""
    theme.apply()
    n = len(samples)
    n_e = EI_RASTER_N_E_PLOT
    n_i = EI_RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n, 1, figsize=(5.6, 3.15),
        sharex=True, gridspec_kw={"hspace": 0.18},
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
        i_rate_str = (
            f"\nI = {s['i_rate_hz']:.1f} Hz" if "i_rate_hz" in s else ""
        )
        ax.text(
            1.012, 0.5,
            f"input = {s['spike_rate']:.1f} Hz\nE = {s['e_rate_hz']:.1f} Hz" + i_rate_str,
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=theme.SIZE_ANNOTATION,
        )
        if i == 0:
            ax.set_title(
                "E (black) and I (red) spikes — trained ping, MNIST digit 0, "
                "input-rate sweep"
            )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


FI_UNIFORM_RATES_HZ: list[float] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    12.0, 14.0, 16.0, 18.0, 20.0, 25.0, 30.0, 35.0, 40.0,
    50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
]
FI_UNIFORM_ZOOM_RATES_HZ: list[float] = [round(r, 2) for r in np.linspace(0.0, 10.0, 101)]
FI_UNIFORM_BATCH: int = 32  # batch of uniform-1 inputs per rate; average over.


def run_fi_sweep_uniform(notebook_run_id: str, rates: list[float] | None = None) -> list[dict]:
    """Population f-I curves on trained PING and COBA baselines (θ_u =
    off, seed 42) with spatially uniform Poisson input — every input
    channel firing at the same rate, no MNIST structure. For each rate
    in `rates` (defaults to FI_UNIFORM_RATES_HZ), average per-cell E and
    I firing rates over FI_UNIFORM_BATCH trials."""
    import torch

    import models as M
    from cli import EVAL_SEED, _auto_device, encode_batch

    if rates is None:
        rates = FI_UNIFORM_RATES_HZ
    device = _auto_device()
    rows: list[dict] = []
    for model in MODELS:
        train_dir = baseline_dir(model)
        if not (train_dir / "weights.pth").exists():
            raise SystemExit(
                f"f-I sweep needs trained {model} weights at {train_dir}"
            )
        net, cfg, _X_te, _y_te = _load_trained_full(train_dir, device)
        net.eval()
        net.recording = True
        n_e = M.N_HID
        n_i = M.N_INH or 1
        t_sec = float(cfg["t_ms"]) / 1000.0
        x_uniform = torch.ones(FI_UNIFORM_BATCH, M.N_IN, device=device)
        original_max_rate = M.max_rate_hz
        try:
            for rate in rates:
                M.max_rate_hz = float(rate)
                M.p_scale = M.max_rate_hz * M.dt / 1000.0
                eval_gen = torch.Generator().manual_seed(EVAL_SEED)
                with torch.no_grad():
                    spk = encode_batch(x_uniform, M.dt, False, generator=eval_gen)
                    net(input_spikes=spk)
                    e_spk_sum = float(net.spike_record["hid"].sum().item())
                    i_spk_sum = float(net.spike_record["inh"].sum().item()) \
                        if "inh" in net.spike_record else 0.0
                e_rate = e_spk_sum / (FI_UNIFORM_BATCH * n_e * t_sec)
                i_rate = i_spk_sum / (FI_UNIFORM_BATCH * n_i * t_sec) \
                    if i_spk_sum else 0.0
                rows.append({
                    "model": model,
                    "input_rate_hz": float(rate),
                    "e_rate_hz": float(e_rate),
                    "i_rate_hz": float(i_rate),
                })
        finally:
            M.max_rate_hz = original_max_rate
            M.p_scale = M.max_rate_hz * M.dt / 1000.0
        print(
            f"  {model:<5} {len(rates)} rates done; "
            f"E range {min(r['e_rate_hz'] for r in rows if r['model']==model):.2f}–"
            f"{max(r['e_rate_hz'] for r in rows if r['model']==model):.2f} Hz"
        )
    return rows


def plot_fi_curve_uniform(
    rows: list[dict], out_path: Path, run_id: str,
    zoom_rows: list[dict] | None = None,
) -> None:
    """Two-panel f-I figure under spatially uniform Poisson input:
    COBA (E + I) on the left, PING (E + I) on the right. If `zoom_rows`
    is provided, a third panel below adds the 0-10 Hz zoom overlaying
    both models' E curves to expose the recruitment cliff."""
    theme.apply()
    if zoom_rows is None:
        fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.1))
        top_axes = list(axes)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.2))
        top_axes = list(axes[0])
    titles = {"ping": "PING (I-loop active)", "coba": "COBA (no I-loop)"}
    for ax, model in zip(top_axes, MODELS):
        msel = sorted(
            [r for r in rows if r["model"] == model],
            key=lambda r: r["input_rate_hz"],
        )
        xs = [r["input_rate_hz"] for r in msel]
        e_ys = [r["e_rate_hz"] for r in msel]
        i_ys = [r["i_rate_hz"] for r in msel]
        ax.plot(xs, e_ys, marker="o", color=theme.INK_BLACK, lw=1.5, label="E")
        ax.plot(xs, i_ys, marker="s", color=theme.DEEP_RED, lw=1.5, label="I")
        if model == "ping":
            ax.plot(xs, [e + i for e, i in zip(e_ys, i_ys)],
                    marker="^", color=theme.AMBER, lw=1.5, ls="--",
                    label="E + I")
        ax.set_xlabel("Input Poisson rate (Hz, per channel)",
                      fontsize=theme.SIZE_LABEL)
        ax.set_ylabel("Per-cell firing rate (Hz)", fontsize=theme.SIZE_LABEL)
        ax.set_title(titles[model], fontsize=theme.SIZE_TITLE)
        ax.legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="upper left")

    # share the y-axis across the two panels so COBA's saturation and PING's
    # compression are read on one scale (the whole point of the comparison)
    top_max = max(
        (max(r["e_rate_hz"], r["i_rate_hz"],
             r["e_rate_hz"] + r["i_rate_hz"] if r["model"] == "ping" else 0.0)
         for r in rows),
        default=1.0,
    )
    for ax in top_axes:
        ax.set_ylim(0, top_max * 1.05)

    if zoom_rows is not None:
        # Bottom row: zoom 0-10 Hz, one panel per model, same scheme
        # as the top row.
        for ax, model in zip(axes[1], MODELS):
            msel = sorted(
                [r for r in zoom_rows if r["model"] == model],
                key=lambda r: r["input_rate_hz"],
            )
            xs = [r["input_rate_hz"] for r in msel]
            e_ys = [r["e_rate_hz"] for r in msel]
            i_ys = [r["i_rate_hz"] for r in msel]
            ax.plot(xs, e_ys, color=theme.INK_BLACK, lw=1.5, label="E")
            ax.plot(xs, i_ys, color=theme.DEEP_RED, lw=1.5, label="I")
            if model == "ping":
                ax.plot(xs, [e + i for e, i in zip(e_ys, i_ys)],
                        color=theme.AMBER, lw=1.5, ls="--", label="E + I")
            ax.set_xlabel("Input Poisson rate (Hz, per channel)",
                          fontsize=theme.SIZE_LABEL)
            ax.set_ylabel("Per-cell firing rate (Hz)", fontsize=theme.SIZE_LABEL)
            ax.set_title(
                f"{titles[model]} — 0–10 Hz zoom",
                fontsize=theme.SIZE_TITLE,
            )
            ax.set_xlim(0, 10)
            ax.legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="upper left")

    fig.suptitle(
        "Population f-I curves: trained PING and COBA, uniform Poisson input",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    save_figure(fig, out_path)  # line/curve plot: SVG + PDF
    plt.close(fig)


def plot_fi_curve(samples: list[dict], out_path: Path, run_id: str) -> None:
    """f-I curve from the same data that plot_rate_rasters consumed.
    x-axis: input Poisson rate (Hz, per channel). y-axis: per-cell mean
    firing rate of E (black) and I (red) populations across the trial."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(5.6, 3.15))
    xs = [s["spike_rate"] for s in samples]
    e_ys = [s["e_rate_hz"] for s in samples]
    i_ys = [s["i_rate_hz"] for s in samples]
    ax.plot(xs, e_ys, marker="o", color=theme.INK_BLACK, lw=1.5, label="E")
    ax.plot(xs, i_ys, marker="s", color=theme.DEEP_RED, lw=1.5, label="I")
    ax.set_xlabel("Input Poisson rate (Hz, per channel)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Per-cell firing rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.legend(fontsize=theme.SIZE_LABEL, frameon=False)
    fig.suptitle(
        "Trained PING f-I curve (MNIST digit 0)",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    stamp_figure(fig, run_id)
    save_figure(fig, out_path)  # line/curve plot: SVG + PDF
    plt.close(fig)


PERTURB_DROP_LEVELS: list[float] = [round(x * 0.1, 2) for x in range(11)]  # 0.0..1.0
PERTURB_ADD_LEVELS: list[float] = [float(x) for x in range(0, 41, 2)]  # 0..40 Hz, 2 Hz steps
PERTURB_RASTER_DROP_LEVELS: list[float] = [0.0, 0.3, 0.6, 0.8, 0.9, 1.0]
PERTURB_RASTER_ADD_LEVELS: list[float] = [0.0, 5.0, 10.0, 20.0, 50.0, 100.0]


def plot_ei_rasters(samples: list[dict], out_path: Path, run_id: str) -> None:
    """One row per ei value; I units stack over E units so the PING-style
    E-then-I cadence reads as alternating bursts when it appears."""
    theme.apply()
    n = len(samples)
    n_e = EI_RASTER_N_E_PLOT
    n_i = EI_RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n, 1, figsize=(5.6, 3.15),
        sharex=True, gridspec_kw={"hspace": 0.18},
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
        ax.text(
            1.012, 0.5, f"ei = {s['ei_strength']:g}",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=theme.SIZE_LABEL,
        )
        if i == 0:
            ax.set_title(
                "E (black) and I (red) spikes — single trial, MNIST test sample 0"
            )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))  # dense raster: PNG, not SVG
    plt.close(fig)


def plot_ei_acc_sweep(points: list[dict], out_path: Path, run_id: str) -> None:
    theme.apply()
    eis = [p["ei_strength"] for p in points]
    accs = [p["acc"] for p in points]
    base_acc = points[0]["acc"]
    worst = min(points, key=lambda p: p["acc"])
    y_hi = min(max(accs + [base_acc]) + 6, 100)
    fig, ax = plt.subplots(figsize=(5.6, 3.15))
    ax.axhline(
        base_acc, color=theme.LABEL, lw=1.0, ls="--",
        label=f"baseline {base_acc:.1f}%",
    )
    ax.axhline(
        10.0, color=theme.FAINT, lw=1.0, ls=":", label="chance (10%)",
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
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)  # line/curve plot: SVG + PDF
    plt.close(fig)


def plot_ei_rates_sweep(points: list[dict], out_path: Path, run_id: str) -> None:
    theme.apply()
    eis = [p["ei_strength"] for p in points]
    hid = [p.get("hid_rate_hz") or 0.0 for p in points]
    inh = [p.get("inh_rate_hz") or 0.0 for p in points]
    fig, ax = plt.subplots(figsize=(5.6, 3.15))
    ax.plot(eis, hid, marker="o", color=theme.INK_BLACK, label="E (hidden)")
    ax.plot(eis, inh, marker="s", color=theme.DEEP_RED, label="I (inhibitory)")
    ax.set_xlabel("inference E→I strength")
    ax.set_ylabel("mean population rate (Hz)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    save_figure(fig, out_path)  # line/curve plot: SVG + PDF
    plt.close(fig)


def run_ei_sweep(notebook_run_id: str) -> list[dict]:
    """In-process inference sweep on the coba__off__seed42 baseline.
    Generates acc_sweep.png, rates_sweep.png, ei_rasters.png and returns
    the per-ei result rows."""
    train_dir = baseline_dir("coba")
    if not (train_dir / "weights.pth").exists():
        raise SystemExit(
            f"ei-sweep needs trained coba weights at {train_dir}; "
            "run training first or check baseline_dir naming."
        )
    sweep_root = _ei_sweep_dir()
    sweep_root.mkdir(parents=True, exist_ok=True)

    points: list[dict] = []
    for ei in EI_SWEEP:
        out = sweep_root / f"infer_ei{ei:g}"
        print(f"[ei-sweep] ei={ei} → {out.relative_to(REPO)}")
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

    print(f"[ei-sweep] capturing single-trial rasters for ei ∈ {EI_RASTER}")
    raster_samples = [
        capture_ei_raster(train_dir, ei, EI_RASTER_SAMPLE_IDX) for ei in EI_RASTER
    ]

    plot_ei_rasters(raster_samples, FIGURES / "ei_rasters", notebook_run_id)
    print(f"wrote {FIGURES / 'ei_rasters'}.{{png,pdf}}")
    # Compound (Figure 1): the rate/accuracy sweeps fold into it, so the two
    # standalone sweep plots are no longer emitted.
    fig_loop_transfer_compound(
        points, raster_samples[0], raster_samples[-1],
        FIGURES / "loop_transfer_compound", notebook_run_id)
    print(f"wrote {FIGURES / 'loop_transfer_compound'}.{{png,pdf}}")
    return points


# ── End ei sweep ────────────────────────────────────────────────────


# ── tau_GABA sweep (inference-only, trained ping) ───────────────────

TAU_GABA_VALUES: list[float] = [4.5, 6.0, 9.0, 12.0, 18.0, 27.0]  # ms; default 9.0


def _load_trained_full(train_dir: Path, device):
    """Load full state from a trained run (incl. W_ei/W_ie). Returns
    (net, cfg, X_te, y_te) ready for forward passes."""
    import torch

    import models as M
    from cli.config import build_net, patch_dt
    from cli import load_dataset, seed_everything

    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", SEEDS_BASELINE[0])))
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
    )
    if hasattr(net, "readout_mode"):
        net.readout_mode = cfg.get("readout_mode", "mem-mean")

    state = torch.load(train_dir / "weights.pth", map_location=device)
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True
    return net, cfg, X_te, y_te


def _eval_net_on_test(net, cfg, X_te, y_te, device) -> tuple[float, float, float]:
    """Forward over test set; return (acc, hid_rate_hz, inh_rate_hz)."""
    acc, _ce, _pen, e_rate, i_rate = _eval_net_on_test_with_loss(
        net, cfg, X_te, y_te, device
    )
    return acc, e_rate, i_rate


def _eval_net_on_test_with_loss(
    net, cfg, X_te, y_te, device,
    fr_upper_theta: float = 0.0,
    fr_upper_strength: float = 0.0,
) -> tuple[float, float, float, float, float]:
    """Forward over test set; return
    (acc, ce_loss, penalty, hid_rate_hz, inh_rate_hz).

    `penalty` is the same firing-rate-upper regulariser the trainer applies:
    sum over hidden neurons of strength · ReLU(mean_per_neuron_spike_count -
    theta_u)^2, computed against the per-neuron spike-count mean over the
    full test set (one application, not per-batch averaging). When strength
    or theta_u is zero the penalty is zero."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    import models as M
    from cli import EVAL_SEED, encode_batch

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=64,
    )
    correct = total = 0
    ce_sum = 0.0
    e_spike_sum = i_spike_sum = 0.0
    # Per-neuron spike count accumulators (one tensor per hidden layer)
    # for computing the training-objective penalty term.
    sc_accums: list[torch.Tensor] = []
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
            logits = net(input_spikes=spk)
            ce_sum += float(
                F.cross_entropy(logits, y_b, reduction="sum").item()
            )
            correct += (logits.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
            e_spike_sum += float(net.spike_record["hid"].sum().item())
            if "inh" in net.spike_record:
                i_spike_sum += float(net.spike_record["inh"].sum().item())
            if fr_upper_strength > 0 and getattr(net, "last_spike_counts", None):
                if not sc_accums:
                    sc_accums = [
                        torch.zeros(sc.shape[1], device=device)
                        for sc in net.last_spike_counts
                    ]
                for acc_tensor, sc in zip(sc_accums, net.last_spike_counts):
                    acc_tensor += sc.sum(dim=0)
    n_e = M.N_HID
    n_i = M.N_INH or 1
    t_sec = float(cfg["t_ms"]) / 1000.0
    e_rate = e_spike_sum / (total * n_e * t_sec) if total else 0.0
    i_rate = i_spike_sum / (total * n_i * t_sec) if (total and i_spike_sum) else 0.0
    acc = 100.0 * correct / total if total else 0.0
    ce_loss = ce_sum / total if total else 0.0
    penalty = 0.0
    if sc_accums and total > 0 and fr_upper_strength > 0:
        for acc_tensor in sc_accums:
            mean_z = acc_tensor / total
            penalty += float(
                fr_upper_strength
                * (torch.relu(mean_z - fr_upper_theta) ** 2).sum().item()
            )
    return acc, ce_loss, penalty, e_rate, i_rate

def copy_video(run_dir: Path, out_path: Path) -> None:
    src = run_dir / "training.mp4"
    if not src.exists():
        raise SystemExit(f"missing training video: {src}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_path)
    print(f"wrote {out_path}")


def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_loop_transfer_compound(points, raster_lo, raster_hi, out_path, run_id):
    """Claim-5 anchor: switching the I-loop on at inference (ei 0→1) on a
    trained COBA cuts the hidden-E rate ~10× at matched accuracy — the gating
    is architectural, not learned. Top: single-trial rasters at ei = 0 and
    ei = 1. Bottom: the ei sweep — E/I rate (left) and accuracy (right)."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"  # keep the saved 16:9 exact
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(6.9, 3.88))  # 16:9, full text width
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3.0, 2.6],
                  hspace=0.5, wspace=0.22, top=0.93, bottom=0.1, left=0.07, right=0.96)

    n_e, n_i, gap = EI_RASTER_N_E_PLOT, EI_RASTER_N_I_PLOT, 6
    for col, s in enumerate((raster_lo, raster_hi)):
        ax = fig.add_subplot(gs[0, col])
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt"]
        e_t, e_n = np.where(s["e"])
        i_t, i_n = np.where(s["i"])
        ax.scatter(t_axis[e_t], e_n, s=1.6, c=theme.INK_BLACK, marker="|", linewidths=0.4)
        ax.scatter(t_axis[i_t], i_n + n_e + gap, s=1.6, c=theme.DEEP_RED, marker="|", linewidths=0.4)
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, s["t_ms"])
        ax.set_xlabel("time (ms)")
        tag = "loop off (COBA)" if s["ei_strength"] == 0 else "loop on (PING)"
        ax.set_title(f"ei = {s['ei_strength']:g}  —  {tag}", loc="left", fontweight="semibold")
        _despine(ax)

    eis = [p["ei_strength"] for p in points]
    ax_r = fig.add_subplot(gs[1, 0])
    hid = [p.get("hid_rate_hz") or 0.0 for p in points]
    inh = [p.get("inh_rate_hz") or 0.0 for p in points]
    ax_r.plot(eis, hid, marker="o", ms=3, color=theme.INK_BLACK, label="E (hidden)")
    ax_r.plot(eis, inh, marker="s", ms=3, color=theme.DEEP_RED, label="I")
    ax_r.set_xlabel("inference E→I strength")
    ax_r.set_ylabel("rate (Hz)")
    ax_r.set_title("E rate falls ≈ 10× as the loop engages", loc="left", fontsize=theme.SIZE_LABEL)
    ax_r.legend(fontsize=theme.SIZE_LEGEND, frameon=False)
    _despine(ax_r)

    ax_a = fig.add_subplot(gs[1, 1])
    accs = [p["acc"] for p in points]
    base_acc = points[0]["acc"]
    ax_a.axhline(base_acc, color=theme.LABEL, lw=1.0, ls="--",
                 label=f"COBA baseline {base_acc:.0f}%")
    ax_a.plot(eis, accs, marker="o", ms=3, color=theme.DEEP_RED, label="transfer")
    ax_a.set_ylim(0, 100)
    ax_a.set_xlabel("inference E→I strength")
    ax_a.set_ylabel("test accuracy (%)")
    ax_a.set_title("Accuracy degrades without retraining", loc="left", fontsize=theme.SIZE_LABEL)
    ax_a.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="lower left")
    _despine(ax_a)

    stamp_figure(fig, run_id)
    # Compound contains dense single-trial raster panels: rasterise as PNG, not SVG.
    save_figure(fig, out_path, formats=("png", "pdf"))
    plt.close(fig)


def build_loop_transfer_compound(run_id: str = "replot") -> None:
    """Replot-only: inference sweep on the cached COBA weights, no training."""
    train_dir = baseline_dir("coba")
    if not (train_dir / "weights.pth").exists():
        raise SystemExit(f"need trained coba weights at {train_dir} (run the notebook once)")
    sweep_root = _ei_sweep_dir()
    sweep_root.mkdir(parents=True, exist_ok=True)
    points = []
    for ei in EI_SWEEP:
        m = run_inproc_infer(train_dir, ei, sweep_root / f"infer_ei{ei:g}")
        points.append({"ei_strength": ei, "acc": m["best_acc"],
                       "hid_rate_hz": m.get("hid_rate_hz"), "inh_rate_hz": m.get("inh_rate_hz")})
        print(f"[ei-sweep] ei={ei}: acc={m['best_acc']:.1f}% E={m.get('hid_rate_hz')} Hz")
    r_lo = capture_ei_raster(train_dir, 0.0, EI_RASTER_SAMPLE_IDX)
    r_hi = capture_ei_raster(train_dir, 1.0, EI_RASTER_SAMPLE_IDX)
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "loop_transfer_compound"
    fig_loop_transfer_compound(points, r_lo, r_hi, out, run_id)
    print(f"wrote {out}.{{png,pdf}}")


def main() -> None:
    # Publication profile: every figure this notebook writes is a print-sized
    # vector, emitted as both SVG (docs) and PDF (manuscript) by save_figure.
    theme.set_paper_mode(True)

    if "--compound-only" in sys.argv:
        build_loop_transfer_compound()
        return
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    skip_training = "--skip-training" in sys.argv
    wipe_dir = "--no-wipe-dir" not in sys.argv

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    n_cells = len(MODELS) * len(THETA_U_GRID)
    print(
        f"notebook_run_id = {notebook_run_id} tier={tier} cells={n_cells}"
        + ("  [skip-training]" if skip_training else "")
    )

    prepare_run_dirs(
        SLUG, notebook_run_id, wipe=wipe_dir, skip_training=skip_training,
        make_artifacts=False,
    )

    # Training lives in nb022 now (train-once / reuse-many). This notebook
    # consumes the shared cells via cell_dir → nb022.load_cell.

    rows: list[dict] = []
    for model in MODELS:
        for theta_u in THETA_U_GRID:
            for seed in seeds_for(theta_u):
                run_dir = cell_dir(model, theta_u, seed)
                if not (run_dir / "metrics.json").exists():
                    raise SystemExit(f"missing metrics: {run_dir / 'metrics.json'}")
                metrics = load_metrics(run_dir)
                last = metrics["epochs"][-1]
                rows.append(
                    {
                        "model": model,
                        "theta_u": theta_u,
                        "theta_display": theta_display(theta_u),
                        "theta_u_hz": theta_hz(theta_u),
                        "seed": seed,
                        "best_acc": float(metrics["best_acc"]),
                        "best_epoch": int(metrics["best_epoch"]),
                        "final_acc": float(last["acc"]),
                        "rate_e": float(last.get("rate_e") or 0.0),
                    }
                )
        # One training video per (model, θ_u) — use the canonical seed
        # (first of SEEDS_BASELINE for baselines, SEED_SWEEP for sweep).
        for theta_u in THETA_U_GRID:
            canonical_seed = seeds_for(theta_u)[0]
            copy_video(
                cell_dir(model, theta_u, canonical_seed),
                FIGURES / f"training__{model}__{theta_label(theta_u)}.mp4",
            )

    print("  results:")
    for r in rows:
        theta_str = (
            f"θ_u={r['theta_display']:>4} ({r['theta_u_hz']:>4.1f} Hz)"
            if r["theta_u"] is not None
            else "θ_u= off"
        )
        print(
            f"    {r['model']:<5}  {theta_str}  "
            f"acc(final)={r['final_acc']:6.2f}%  best={r['best_acc']:6.2f}%  "
            f"rate_e={r['rate_e']:6.1f} Hz"
        )

    # Input-rate sweep on the trained ping network — one digit, vary the
    # Poisson rate over a wide range, render as a scope-frame video.
    rate_sweep_out = FIGURES / "rate_sweep__ping.mp4"
    generate_rate_sweep_video("ping", rate_sweep_out)
    print(f"wrote {rate_sweep_out}")

    # Stacked raster snapshot at the first 10 frames of the rate sweep —
    # same panel style as the ei-sweep rasters so the two read as a pair.
    rate_grid = np.linspace(0.0, 100.0, 40)[:10]
    print(f"[rate-rasters] capturing rates {[round(r, 2) for r in rate_grid]}")
    rate_samples = [
        capture_rate_raster(baseline_dir("ping"), float(r), sample_idx=0)
        for r in rate_grid
    ]
    plot_rate_rasters(
        rate_samples, FIGURES / "rate_rasters__ping", notebook_run_id
    )
    print(f"wrote {FIGURES / 'rate_rasters__ping'}.{{png,pdf}}")
    plot_fi_curve(rate_samples, FIGURES / "fi_curve__ping", notebook_run_id)
    print(f"wrote {FIGURES / 'fi_curve__ping'}.{{svg,pdf}}")

    # Uniform-input f-I curves for PING and COBA — no MNIST structure.
    print("[fi-sweep] uniform Poisson input on trained PING and COBA (wide)")
    fi_rows = run_fi_sweep_uniform(notebook_run_id)
    plot_fi_curve_uniform(
        fi_rows, FIGURES / "fi_curve_uniform", notebook_run_id,
    )
    print(f"wrote {FIGURES / 'fi_curve_uniform'}.{{svg,pdf}}")

    # EI-strength sweep (subsumes nb019): replay COBA-trained weights
    # with progressively stronger I-loop.
    ei_points = run_ei_sweep(notebook_run_id)

    duration_s = time.monotonic() - t_start
    train_cfg = load_config(baseline_dir(MODELS[0]))
    summary = {
        "notebook_run_id": notebook_run_id,
        "git_sha": train_cfg.get("git_sha"),
        "duration_s": round(duration_s, 1),
        "duration": format_duration(duration_s),
        "tier": tier,
        "config": {
            "tier": tier,
            "dataset": "mnist",
            "models": MODELS,
            "theta_u_grid_spikes": [t for t in THETA_U_GRID if t is not None],
            "theta_u_grid_hz": [
                theta_hz(t) for t in THETA_U_GRID if t is not None
            ],
            "max_samples": TIER_CONFIG[tier]["max_samples"],
            "epochs": TIER_CONFIG[tier]["epochs"],
            "t_ms": T_MS,
            "dt": DT_TRAIN,
            "seeds_baseline": SEEDS_BASELINE,
            "seed_sweep": SEED_SWEEP,
            "fr_strength_upper": FR_STRENGTH_UPPER,
        },
        "baseline_results": rows,
        "ei_sweep": ei_points,
        "fi_sweep_uniform": fi_rows,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")



if __name__ == "__main__":
    main()
