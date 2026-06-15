"""Notebook runner for entry 037 — spike-stream perturbations of
trained PING.

Standalone runner with no cross-notebook helpers. Trains coba / ping
baseline cells (θ_u = off, three seeds), then runs:
- hidden-spike perturbation sweep (drop + Poisson add) against the
  trained baselines; and
- τ_GABA sweep (inference-time mutation of the inhibitory decay
  constant on trained PING).

Figures land in /figures/notebooks/nb037/ and the success-criteria
summary in nb037/numbers.json.

Notebook entry: src/docs/src/pages/notebooks/nb037.mdx
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

from helpers.modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from helpers.run_id import next_run_id, persist as persist_run_id  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402
from cli import theme  # noqa: E402

SLUG = "nb037"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
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
    """Per-cell artifact directory.

    Baseline cells get a `__seed{N}` suffix so multiple seeds coexist.
    Sweep cells run only at SEED_SWEEP and skip the suffix to keep
    paths short — they live alongside the baseline ones.
    """
    label = theta_label(theta_u)
    if theta_u is None:
        return ARTIFACTS / f"{model}__{label}__seed{seed}"
    return ARTIFACTS / f"{model}__{label}"


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


def _stamp(fig, run_id: str) -> None:
    fig.text(
        0.995, 0.005, run_id,
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )



def capture_perturbation_raster(
    train_dir: Path, mode: str, level: float, sample_idx: int = 0
) -> dict:
    """Single-trial raster with the hidden-spike perturbation hook active.

    Same build / load / forward sequence as capture_rate_raster, with
    _hidden_perturb_fn installed for the duration of the forward pass.
    """
    import torch

    import cli.config as C  # noqa: F401
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

    perturb_gen = torch.Generator(device=device).manual_seed(EVAL_SEED + 1)
    net._hidden_perturb_fn = _make_perturb_fn(mode, level, M.dt, perturb_gen)

    X_b = torch.from_numpy(X_te[sample_idx : sample_idx + 1]).to(device)
    y_b = int(y_te[sample_idx])
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    with torch.no_grad():
        spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
        _ = net(input_spikes=spk)

    net._hidden_perturb_fn = None

    e_full = net.spike_record["hid"].cpu().numpy()
    i_full = net.spike_record["inh"].cpu().numpy()
    e_rate_hz = float(e_full.sum() / (e_full.shape[1] * cfg["t_ms"] / 1000.0))
    rng = np.random.default_rng(0)
    e_idx = np.sort(rng.choice(e_full.shape[1], EI_RASTER_N_E_PLOT, replace=False))
    i_idx = np.sort(rng.choice(i_full.shape[1], EI_RASTER_N_I_PLOT, replace=False))
    return {
        "mode": mode,
        "level": (
            list(float(x) for x in level)
            if isinstance(level, (list, tuple))
            else float(level)
        ),
        "e_rate_hz": e_rate_hz,
        "label": y_b,
        "e": e_full[:, e_idx].astype(bool),
        "i": i_full[:, i_idx].astype(bool),
        "dt": float(cfg["dt"]),
        "t_ms": float(cfg["t_ms"]),
    }


def plot_perturbation_rasters(
    samples: list[dict], out_path: Path, run_id: str, level_fmt: str, title: str
) -> None:
    """Stacked single-trial rasters across perturbation levels for one mode."""
    theme.apply()
    n = len(samples)
    n_e = EI_RASTER_N_E_PLOT
    n_i = EI_RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n, 1, figsize=(10.0, 5.625),
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
            1.012, 0.5,
            level_fmt.format(level=s["level"]) + f"\nE = {s['e_rate_hz']:.1f} Hz",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=theme.SIZE_LABEL,
        )
        if i == 0:
            ax.set_title(title)
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _make_perturb_fn(mode: str, level: float, dt_ms: float, generator):
    """Return a per-step callback (s_e, s_i, layer_idx) -> (s_e', s_i').

    Both populations get the same perturbation. The callback runs inside
    the COBANet step body right after spikes are emitted and before they
    feed into the readout / recurrence / I-loop, so the perturbation is
    dynamics-faithful: downstream state reacts to it within the trial.
    """
    import torch

    if mode == "drop":
        def fn(s_e, s_i, _layer):
            mask = (
                torch.rand(s_e.shape, generator=generator, device=s_e.device)
                >= level
            ).float()
            s_e = s_e * mask
            if s_i is not None:
                mask_i = (
                    torch.rand(s_i.shape, generator=generator, device=s_i.device)
                    >= level
                ).float()
                s_i = s_i * mask_i
            return s_e, s_i

        return fn

    if mode == "add":
        p = level * dt_ms / 1000.0
        def fn(s_e, s_i, _layer):
            extra_e = (
                torch.rand(s_e.shape, generator=generator, device=s_e.device) < p
            ).float()
            s_e = torch.clamp(s_e + extra_e, 0.0, 1.0)
            if s_i is not None:
                extra_i = (
                    torch.rand(s_i.shape, generator=generator, device=s_i.device) < p
                ).float()
                s_i = torch.clamp(s_i + extra_i, 0.0, 1.0)
            return s_e, s_i

        return fn

    if mode == "add_split":
        # level is a (r_e_hz, r_i_hz) tuple — different per-population Poisson
        # rates so we can express noise as a fraction of each model's own
        # baseline activity. Keeps E and I perturbations independent.
        r_e_hz, r_i_hz = float(level[0]), float(level[1])
        p_e = r_e_hz * dt_ms / 1000.0
        p_i = r_i_hz * dt_ms / 1000.0

        def fn(s_e, s_i, _layer):
            if p_e > 0:
                extra_e = (
                    torch.rand(s_e.shape, generator=generator, device=s_e.device)
                    < p_e
                ).float()
                s_e = torch.clamp(s_e + extra_e, 0.0, 1.0)
            if s_i is not None and p_i > 0:
                extra_i = (
                    torch.rand(s_i.shape, generator=generator, device=s_i.device)
                    < p_i
                ).float()
                s_i = torch.clamp(s_i + extra_i, 0.0, 1.0)
            return s_e, s_i

        return fn

    raise ValueError(f"unknown perturbation mode {mode!r}")


def run_perturbation_sweep(
    train_dir: Path, mode: str, level: float
) -> dict:
    """Evaluate the test set under a hidden-spike perturbation.

    Builds the trained network, attaches the perturbation hook, runs one
    forward pass over the whole test set, and returns accuracy plus the
    achieved hidden E firing rate (perturbation can shift either).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    import cli.config as C  # noqa: F401
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

    perturb_gen = torch.Generator(device=device).manual_seed(EVAL_SEED + 1)
    net._hidden_perturb_fn = _make_perturb_fn(mode, level, M.dt, perturb_gen)

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=64,
    )

    correct = total = 0
    e_spike_sum = 0.0
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
            logits = net(input_spikes=spk)
            correct += (logits.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
            # Sum E spikes across the recorded hidden raster. Last hidden
            # layer feeds the readout; report its rate.
            last_key = f"hid_{net.n_layers}" if net.n_layers > 1 else "hid"
            hid_rec = net.spike_record.get(last_key)
            if hid_rec is None:
                hid_rec = net.spike_record["hid"]
            e_spike_sum += float(hid_rec.sum().item())

    net._hidden_perturb_fn = None  # unset so subsequent runs aren't poisoned

    n_e = hidden_sizes[-1]
    t_sec = float(cfg["t_ms"]) / 1000.0
    e_rate_hz = e_spike_sum / (total * n_e * t_sec) if total else 0.0
    acc = 100.0 * correct / total if total else 0.0
    return {
        "mode": mode,
        "level": level,
        "acc": acc,
        "e_rate_hz": e_rate_hz,
        "n_total": total,
    }


def plot_perturbation_curves(
    points: list[dict], out_path: Path, run_id: str,
    add_pct_rows: list[dict] | None = None,
) -> None:
    """Two-panel accuracy plot: drop on the left, add on the right.

    Left panel: drop probability (Bernoulli spike mask).
    Right panel: Poisson add. If `add_pct_rows` is provided, the right
    panel uses percentage-of-baseline-rate from those rows (the fair
    comparison). Otherwise it falls back to absolute Hz from `points`.
    """
    theme.apply()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.5), sharey=True, dpi=150)
    use_pct = add_pct_rows is not None and len(add_pct_rows) > 0

    # Left panel: drop (as % of spikes dropped)
    ax_drop = axes[0]
    for model in MODELS:
        rows = [
            p for p in points if p["model"] == model and p["mode"] == "drop"
        ]
        rows.sort(key=lambda p: p["level"])
        ax_drop.plot(
            [p["level"] * 100 for p in rows], [p["acc"] for p in rows],
            marker=MODEL_MARKERS[model], markersize=5, linewidth=1.4,
            color=MODEL_COLORS[model], label=model,
        )
    ax_drop.set_xlabel("Spikes dropped (% of emitted)",
                       fontsize=theme.SIZE_LABEL)
    ax_drop.set_title("Drop — Bernoulli spike mask",
                      fontsize=theme.SIZE_LABEL, loc="left", pad=4)
    ax_drop.set_xlim(-2, 102)
    ax_drop.axhline(10.0, ls="--", color=theme.MUTED, lw=0.7, alpha=0.6)
    ax_drop.text(
        0.02, 12, "chance", transform=ax_drop.get_yaxis_transform(),
        fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED, va="bottom",
    )

    # Right panel: add
    ax_add = axes[1]
    if use_pct:
        for model in MODELS:
            rows = sorted(
                [r for r in add_pct_rows if r["model"] == model],
                key=lambda r: r["pct"],
            )
            ax_add.plot(
                [r["pct"] * 100 for r in rows], [r["acc"] for r in rows],
                marker=MODEL_MARKERS[model], markersize=5, linewidth=1.4,
                color=MODEL_COLORS[model], label=model,
            )
        ax_add.set_xlabel(
            "Added Poisson noise (% of baseline rate)",
            fontsize=theme.SIZE_LABEL,
        )
        ax_add.set_title(
            "Add — Poisson noise as % of baseline",
            fontsize=theme.SIZE_LABEL, loc="left", pad=4,
        )
        ax_add.set_xlim(-2, 102)
    else:
        for model in MODELS:
            rows = sorted(
                [p for p in points
                 if p["model"] == model and p["mode"] == "add"],
                key=lambda p: p["level"],
            )
            ax_add.plot(
                [p["level"] for p in rows], [p["acc"] for p in rows],
                marker=MODEL_MARKERS[model], markersize=5, linewidth=1.4,
                color=MODEL_COLORS[model], label=model,
            )
        ax_add.set_xlabel("Poisson rate (Hz / neuron)",
                          fontsize=theme.SIZE_LABEL)
        ax_add.set_title("Add — Poisson noise injection",
                          fontsize=theme.SIZE_LABEL, loc="left", pad=4)
    ax_add.axhline(10.0, ls="--", color=theme.MUTED, lw=0.7, alpha=0.6)

    for ax in axes:
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=theme.SIZE_TICK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_locator(plt.matplotlib.ticker.MultipleLocator(20))
        ax.grid(True, axis="y", alpha=0.15, linewidth=0.5)
    ax_drop.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax_add.legend(
        loc="upper right", fontsize=theme.SIZE_LEGEND, frameon=False,
    )
    fig.suptitle(
        "Hidden-spike perturbation — accuracy vs perturbation level",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── inference helpers used by perturbation sweep ────────────────────


def _load_trained_full(train_dir: Path, device):
    """Load full state from a trained run (incl. W_ei/W_ie). Returns
    (net, cfg, X_te, y_te) ready for forward passes."""
    import torch

    import cli.config as C  # noqa: F401
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

def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def copy_video(run_dir: Path, out_path: Path) -> None:
    src = run_dir / "training.mp4"
    if not src.exists():
        raise SystemExit(f"missing training video: {src}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_path)
    print(f"wrote {out_path}")


def main() -> None:
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

    only_missing = "--only-missing" in sys.argv
    if not skip_training:
        dispatcher = BatchDispatcher(modal_gpu, REPO, OSCILLOSCOPE)
        for model in MODELS:
            for theta_u in THETA_U_GRID:
                build_as = MODEL_RECIPES[model]["__build_as"]
                gpu_override = None
                if modal_gpu in ("T4", "L4", "A10G") and build_as == "ping":
                    gpu_override = "A100"
                for seed in seeds_for(theta_u):
                    out = cell_dir(model, theta_u, seed)
                    if only_missing and (out / "metrics.json").exists():
                        print(
                            f"[skip] {model}/θ_u={theta_display(theta_u)}/seed={seed} "
                            f"already trained → {out.relative_to(REPO)}"
                        )
                        continue
                    print(
                        f"[train] {model}/θ_u={theta_display(theta_u)}/seed={seed} → "
                        f"{out.relative_to(REPO)}"
                        + (f"  [modal:{modal_gpu}]" if modal_gpu else "")
                    )
                    dispatcher.submit(
                        build_train_args(model, theta_u, seed, tier, out),
                        out,
                        gpu_override=gpu_override,
                    )
        dispatcher.drain()

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

    # Hidden-layer perturbation sweep: drop spikes (Bernoulli mask) and
    # add Poisson noise spikes, applied inside the forward loop so the
    # I-population and readout both react.
    print("[perturb] hidden-spike drop + add sweep (coba, ping)")
    perturb_rows: list[dict] = []
    for model in MODELS:
        train_dir = baseline_dir(model)
        for mode, levels in (
            ("drop", PERTURB_DROP_LEVELS),
            ("add", PERTURB_ADD_LEVELS),
        ):
            for level in levels:
                res = run_perturbation_sweep(train_dir, mode, level)
                res["model"] = model
                perturb_rows.append(res)
                print(
                    f"  {model:<5} {mode:<4} level={level:>5.2f}  "
                    f"acc={res['acc']:5.2f}%  E={res['e_rate_hz']:6.2f} Hz"
                )
    plot_perturbation_curves(
        perturb_rows, FIGURES / "perturbation_curves.png", notebook_run_id
    )
    print(f"wrote {FIGURES / 'perturbation_curves.png'}")

    # Stacked-raster snapshots of each trained baseline under both
    # perturbation modes, six panels per (model, mode). Same MNIST digit 0
    # sample 0 as the other rasters so the panels read against the
    # unperturbed baselines (Figures 4-5).
    for model in MODELS:
        train_dir = baseline_dir(model)
        drop_samples = [
            capture_perturbation_raster(train_dir, "drop", lvl, 0)
            for lvl in PERTURB_RASTER_DROP_LEVELS
        ]
        plot_perturbation_rasters(
            drop_samples,
            FIGURES / f"perturb_rasters__drop__{model}.png",
            notebook_run_id,
            level_fmt="p(drop) = {level:.1f}",
            title=(
                f"E (black) and I (red) spikes — trained {model} with "
                "hidden-spike drop"
            ),
        )
        print(f"wrote {FIGURES / f'perturb_rasters__drop__{model}.png'}")
        add_samples = [
            capture_perturbation_raster(train_dir, "add", lvl, 0)
            for lvl in PERTURB_RASTER_ADD_LEVELS
        ]
        plot_perturbation_rasters(
            add_samples,
            FIGURES / f"perturb_rasters__add__{model}.png",
            notebook_run_id,
            level_fmt="r(add) = {level:g} Hz",
            title=(
                f"E (black) and I (red) spikes — trained {model} with "
                "hidden-spike Poisson noise added"
            ),
        )
        print(f"wrote {FIGURES / f'perturb_rasters__add__{model}.png'}")

    duration_s = time.monotonic() - t_start
    train_cfg = load_config(baseline_dir(MODELS[0]))
    summary = {
        "notebook_run_id": notebook_run_id,
        "git_sha": train_cfg.get("git_sha"),
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
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
        "perturbation": perturb_rows,
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")



if __name__ == "__main__":
    main()
