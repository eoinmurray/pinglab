"""Notebook runner for entry 049 — Does gradient descent discover PING?

Trains four conditions × three seeds. Each condition is a different
choice of *initial* W_EI / W_IE and whether they are gradient-carrying:

  - frozen_ping            (baseline) — canonical nb025 PING; W_ei/W_ie frozen
  - trainable_ping_init     PING init, W_ei/W_ie trainable from canonical
  - trainable_zero_init     COBA init (ei_strength=0), W_ei/W_ie trainable
  - trainable_small_init    ei_strength=0.1, W_ei/W_ie trainable

Question: does the network discover a non-trivial PING loop from a
zero-init or small-seed start, and does the PING optimum drift if it's
allowed to move?

Per-epoch metrics include weight Frobenius norms for every trainable
parameter (already tracked by the train loop), so we can read off the
W_ei / W_ie trajectories without extra hooks.

Post-training: PSD on the trained-cell E population to read off the
gamma peak; final-state raster strip; (acc, E rate, gamma peak,
W_ei norm, W_ie norm) summary table.

Notebook entry: src/docs/src/pages/notebooks/nb049.mdx
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

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402
from pinglab import theme  # noqa: E402

SLUG = "nb049"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "cli/__main__.py"

TIER_CONFIG = {
    "extra small": dict(max_samples=100, epochs=2),
    "small":       dict(max_samples=500, epochs=10),
    "medium":      dict(max_samples=2000, epochs=100),
    "large":       dict(max_samples=5000, epochs=100),
    "extra large": dict(max_samples=10000, epochs=100),
}
DEFAULT_TIER = "small"
T_MS = 200.0
DT_TRAIN = 0.1

SEEDS: list[int] = [42, 43, 44]

# Per-condition recipes — what's different from the canonical nb025 PING
# init is the (ei_strength, trainable flags) trio. Everything else
# (Wfeed-forward init, lr, batch size, readout, surrogate slope) matches
# nb025's PING baseline exactly so the comparisons stay honest.
CONDITIONS: dict[str, dict] = {
    "frozen_ping": {
        "ei_strength": "1",
        "trainable_w_ei": False,
        "trainable_w_ie": False,
        "label": "Frozen PING (control)",
    },
    "trainable_ping_init": {
        "ei_strength": "1",
        "trainable_w_ei": True,
        "trainable_w_ie": True,
        "label": "Trainable, PING init",
    },
    "trainable_zero_init": {
        "ei_strength": "0",
        "trainable_w_ei": True,
        "trainable_w_ie": True,
        "label": "Trainable, zero init",
    },
    "trainable_small_init": {
        "ei_strength": "0.1",
        "trainable_w_ei": True,
        "trainable_w_ie": True,
        "label": "Trainable, small seed init",
    },
}
COND_ORDER = list(CONDITIONS.keys())

# Drive parameters match nb025's PING baseline so the comparison only
# tests the structural change (trainability of the recurrent loop).
COMMON_RECIPE: dict[str, str] = {
    "--v-grad-dampen": "1000",
    "--w-in": "1.2",
    "--w-in-sparsity": "0.95",
    "--readout": "mem-mean",
    "--surrogate-slope": "1",
    "--readout-w-out-scale": "500",
    "--lr": "0.0004",
    "--batch-size": "256",
}

COND_COLOURS: dict[str, str] = {
    "frozen_ping":           theme.MUTED,
    "trainable_ping_init":   theme.INK_BLACK,
    "trainable_zero_init":   theme.DEEP_RED,
    "trainable_small_init":  theme.AMBER,
}
COND_MARKERS: dict[str, str] = {
    "frozen_ping":           "o",
    "trainable_ping_init":   "s",
    "trainable_zero_init":   "^",
    "trainable_small_init":  "D",
}


# ── Paths ───────────────────────────────────────────────────────────
def cell_dir(condition: str, seed: int) -> Path:
    return ARTIFACTS / f"{condition}__seed{seed}"


def build_train_args(
    condition: str, seed: int, tier: str, out_dir: Path,
) -> list[str]:
    recipe = CONDITIONS[condition]
    args = [
        "train",
        "--model", "ping",
        "--dataset", "mnist",
        "--max-samples", str(TIER_CONFIG[tier]["max_samples"]),
        "--epochs", str(TIER_CONFIG[tier]["epochs"]),
        "--t-ms", str(T_MS),
        "--dt", str(DT_TRAIN),
        "--seed", str(seed),
        "--out-dir", str(out_dir),
        "--wipe-dir",
        "--ei-strength", str(recipe["ei_strength"]),
    ]
    if recipe["trainable_w_ei"]:
        args.append("--trainable-w-ei")
    if recipe["trainable_w_ie"]:
        args.append("--trainable-w-ie")
    for k, v in COMMON_RECIPE.items():
        if v is True:
            args.append(k)
        elif v is not None:
            args += [k, v]
    return args


def load_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


# ── Post-hoc inference ─────────────────────────────────────────────
def _load_trained_full(train_dir: Path, device):
    """Mirror nb025/nb041 loader pattern but read the per-condition
    trained config so trainable W_ei/W_ie get loaded correctly."""
    import torch

    import models as M
    from config import build_net, patch_dt
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
        ei_strength=float(cfg.get("ei_strength") or 0.0),
        ei_ratio=float(cfg.get("ei_ratio") or 2.0),
        sparsity=float(cfg.get("sparsity") or 0.0),
        device=device,
        randomize_init=not bool(cfg.get("kaiming_init", False)),
        dales_law=bool(cfg.get("dales_law", True)),
        hidden_sizes=hidden_sizes,
        readout_mode=cfg.get("readout_mode", "mem-mean"),
        trainable_w_ei=bool(cfg.get("trainable_w_ei", False)),
        trainable_w_ie=bool(cfg.get("trainable_w_ie", False)),
    )
    if hasattr(net, "readout_mode"):
        net.readout_mode = cfg.get("readout_mode", "mem-mean")
    state = torch.load(train_dir / "weights.pth", map_location=device)
    net.load_state_dict(state, strict=False)
    net.eval()
    net.recording = True
    return net, cfg, X_te, y_te


F_GAMMA_BAND_HZ: tuple[float, float] = (5.0, 150.0)


def load_init_and_trained_weights(train_dir: Path, device):
    """Return (W_ei_init, W_ei_trained, W_ie_init, W_ie_trained) as CPU tensors.

    Re-runs build_net under the cell's seed to recover the deterministic
    init weights, then loads the saved state_dict for the trained weights.
    """
    import torch

    import models as M
    from config import build_net, patch_dt
    from cli import seed_everything

    cfg = json.loads((train_dir / "config.json").read_text())
    seed_everything(int(cfg.get("seed", 42)))
    M.T_ms = float(cfg["t_ms"])
    patch_dt(float(cfg["dt"]))
    hidden_sizes = cfg.get("hidden_sizes") or [int(cfg["n_hidden"])]
    M.N_HID = hidden_sizes[-1]
    M.N_INH = hidden_sizes[-1] // 4
    M.HIDDEN_SIZES = list(hidden_sizes)
    M.N_IN = 784 if cfg["dataset"] == "mnist" else 64
    w_in_cfg = cfg.get("w_in")
    w_in_arg = (
        (float(w_in_cfg[0]), float(w_in_cfg[1]))
        if isinstance(w_in_cfg, list) and len(w_in_cfg) >= 2 else None
    )
    net_init = build_net(
        cfg["model"],
        w_in=w_in_arg,
        w_in_sparsity=float(cfg.get("w_in_sparsity") or 0.0),
        ei_strength=float(cfg.get("ei_strength") or 0.0),
        ei_ratio=float(cfg.get("ei_ratio") or 2.0),
        sparsity=float(cfg.get("sparsity") or 0.0),
        device=device,
        randomize_init=not bool(cfg.get("kaiming_init", False)),
        dales_law=bool(cfg.get("dales_law", True)),
        hidden_sizes=hidden_sizes,
        readout_mode=cfg.get("readout_mode", "mem-mean"),
        trainable_w_ei=bool(cfg.get("trainable_w_ei", False)),
        trainable_w_ie=bool(cfg.get("trainable_w_ie", False)),
    )
    w_ei_init = net_init.W_ei["1"].detach().cpu().numpy()
    w_ie_init = net_init.W_ie["1"].detach().cpu().numpy()
    state = torch.load(train_dir / "weights.pth", map_location="cpu")
    w_ei_trained = state["W_ei.1"].numpy()
    w_ie_trained = state["W_ie.1"].numpy()
    return w_ei_init, w_ei_trained, w_ie_init, w_ie_trained


def plot_weight_matrices(
    cond: str,
    seed_to_dir: dict[int, Path],
    device,
    out_path: Path, run_id: str,
) -> None:
    """Per-condition weight-matrix card: W_ei and W_ie, init vs trained, seed 42."""
    theme.apply()
    label = CONDITIONS[cond]["label"]
    seed = sorted(seed_to_dir.keys())[0]
    w_ei_init, w_ei_trained, w_ie_init, w_ie_trained = load_init_and_trained_weights(
        seed_to_dir[seed], device,
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.5), dpi=150,
        gridspec_kw={"wspace": 0.25},
    )

    def _hist_panel(ax, init_arr, trained_arr, title, color):
        # Plot the *effective* weight — Dale's law clamps stored w < 0 to 0
        # in the forward pass, so that's what the network actually sees.
        flat_init = np.maximum(init_arr.ravel(), 0.0)
        flat_trained = np.maximum(trained_arr.ravel(), 0.0)
        v_hi = float(max(flat_init.max(), flat_trained.max(), 1e-12))
        bins = np.linspace(0.0, v_hi * 1.05, 60)

        eff_init = flat_init.mean()
        eff_trained = flat_trained.mean()
        # Fraction of entries that the forward pass sees as exactly zero
        # (init: structural sparsity; trained: Dale's-law pruning).
        frac_pruned_init = float((flat_init <= 0).mean())
        frac_pruned_trained = float((flat_trained <= 0).mean())

        ax.hist(
            flat_init, bins=bins, histtype="step", color=color, lw=1.4,
            label=(f"init   (mean = {eff_init:.4f}, "
                   f"pruned = {frac_pruned_init:.0%})"),
        )
        ax.hist(
            flat_trained, bins=bins, histtype="stepfilled", color=color,
            alpha=0.35, edgecolor=color, lw=1.0,
            label=(f"trained (mean = {eff_trained:.4f}, "
                   f"pruned = {frac_pruned_trained:.0%})"),
        )
        ax.set_title(title, fontsize=theme.SIZE_LABEL)
        ax.set_xlabel("effective weight (forward-pass value)",
                      fontsize=theme.SIZE_LABEL)
        ax.set_ylabel("count", fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")

    _hist_panel(axes[0], w_ei_init, w_ei_trained, r"$W^{EI}$ distribution", theme.INK_BLACK)
    _hist_panel(axes[1], w_ie_init, w_ie_trained, r"$W^{IE}$ distribution", theme.DEEP_RED)
    fig.suptitle(f"{label} — recurrent weight distributions, seed {seed}",
                 fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def measure_trained_state(train_dir: Path, device) -> dict:
    """Forward the test set; report acc, mean E rate, mean I rate,
    f_γ (via Welch PSD peak), and the W_ei / W_ie final means."""
    import torch
    from scipy import signal as sp_signal
    from torch.utils.data import DataLoader, TensorDataset

    import models as M
    from cli import EVAL_SEED, encode_batch

    net, cfg, X_te, y_te = _load_trained_full(train_dir, device)

    # Final W_ei / W_ie statistics (across the single ei-layer).
    w_ei_vals = [w.detach().abs().mean().item() for w in net.W_ei.values()]
    w_ie_vals = [w.detach().abs().mean().item() for w in net.W_ie.values()]

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=64,
    )
    correct = total = 0
    e_spike_sum = i_spike_sum = 0.0
    pop_e_traces: list[np.ndarray] = []
    n_e = M.N_HID
    n_i = M.N_INH or 1
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
            logits = net(input_spikes=spk)
            correct += (logits.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
            hid = net.spike_record["hid"]
            if hid.ndim == 2:
                hid = hid.unsqueeze(1)
            e_spike_sum += float(hid.sum().item())
            if "inh" in net.spike_record:
                inh = net.spike_record["inh"]
                if inh.ndim == 2:
                    inh = inh.unsqueeze(1)
                i_spike_sum += float(inh.sum().item())
            pop = hid.mean(dim=2).cpu().numpy()  # (T, B)
            for b in range(pop.shape[1]):
                pop_e_traces.append(pop[:, b])

    t_sec = float(cfg["t_ms"]) / 1000.0
    e_rate = e_spike_sum / (total * n_e * t_sec) if total else 0.0
    i_rate = (
        i_spike_sum / (total * n_i * t_sec) if (total and i_spike_sum)
        else 0.0
    )

    # f_γ via Welch PSD on the per-trial E-population trace.
    fs_hz = 1000.0 / float(cfg["dt"])
    nperseg = pop_e_traces[0].size
    psds: list[np.ndarray] = []
    freqs: np.ndarray | None = None
    for tr in pop_e_traces:
        if tr.std() == 0:
            continue
        f, p = sp_signal.welch(
            tr - tr.mean(), fs=fs_hz, nperseg=nperseg,
            scaling="density", detrend=False,
        )
        psds.append(p)
        freqs = f
    if psds and freqs is not None:
        psd_mean = np.mean(np.stack(psds, axis=0), axis=0)
        band = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
        if band.any() and psd_mean[band].max() > 0:
            peak_local = int(psd_mean[band].argmax())
            peak_idx = int(np.where(band)[0][peak_local])
            f_gamma = float(freqs[peak_idx])
            psd_used = psd_mean
            freqs_used = freqs
        else:
            f_gamma = float("nan")
            psd_used = psd_mean
            freqs_used = freqs
    else:
        f_gamma = float("nan")
        psd_used = np.zeros(1)
        freqs_used = np.zeros(1)

    return {
        "acc": float(100.0 * correct / total) if total else float("nan"),
        "e_rate_hz": float(e_rate),
        "i_rate_hz": float(i_rate),
        "f_gamma_hz": f_gamma,
        "w_ei_mean": float(np.mean(w_ei_vals)) if w_ei_vals else 0.0,
        "w_ie_mean": float(np.mean(w_ie_vals)) if w_ie_vals else 0.0,
        "psd": psd_used.tolist(),
        "freqs_hz": freqs_used.tolist(),
    }


def capture_raster(train_dir: Path, device, sample_idx: int = 0) -> dict:
    """Single-trial raster for the trained network."""
    import torch
    import models as M
    from cli import EVAL_SEED, encode_batch

    net, cfg, X_te, y_te = _load_trained_full(train_dir, device)
    X_b = torch.from_numpy(X_te[sample_idx:sample_idx + 1]).to(device)
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    with torch.no_grad():
        spk = encode_batch(X_b, M.dt, False, generator=eval_gen)
        _ = net(input_spikes=spk)
    e = net.spike_record["hid"].cpu().numpy()
    i = net.spike_record.get("inh")
    i = i.cpu().numpy() if i is not None else np.zeros((e.shape[0], 0))
    if e.ndim == 3:
        e = e[:, 0, :]
        if i.ndim == 3:
            i = i[:, 0, :]
    return {
        "e": e.astype(bool),
        "i": i.astype(bool),
        "dt_ms": float(cfg["dt"]),
        "t_ms": float(cfg["t_ms"]),
        "label": int(y_te[sample_idx]),
    }


# ── Plotting ────────────────────────────────────────────────────────
def _stamp(fig, run_id: str) -> None:
    fig.text(
        0.995, 0.005, run_id,
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


def plot_training_trajectories(
    metrics_by_cond: dict[str, list[dict]],
    out_path: Path, run_id: str,
) -> None:
    """2×2 panel: W_ei norm, W_ie norm, E rate, accuracy — vs epoch.
    Each line is the mean of three seeds; light shading is mean ± SEM."""
    theme.apply()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), dpi=150, sharex=True)
    (ax_wei, ax_wie), (ax_re, ax_acc) = axes
    # (axis, metrics_key, weight_sub_key, ylabel, linestyle, label_in_legend)
    panel_specs = [
        (ax_wei, "weight_norms", "W_ei.1", r"$\|W^{EI}\|_F$",       "-",  True),
        (ax_wie, "weight_norms", "W_ie.1", r"$\|W^{IE}\|_F$",       "-",  False),
        (ax_re,  "rate_e",       None,    "Firing rate (Hz)",       "-",  False),
        (ax_re,  "rate_i",       None,    None,                     "--", False),
        (ax_acc, "acc",          None,    "Test accuracy (%)",      "-",  False),
    ]
    for ax, key, sub_key, label, linestyle, show_cond_legend in panel_specs:
        for cond in COND_ORDER:
            metrics_list = metrics_by_cond.get(cond, [])
            if not metrics_list:
                continue
            per_seed_curves = []
            for m in metrics_list:
                eps = m.get("epochs") or []
                curve = []
                for e in eps:
                    val = e.get(key)
                    if sub_key is not None:
                        val = (val or {}).get(sub_key)
                    curve.append(float(val) if val is not None else float("nan"))
                if curve:
                    per_seed_curves.append(curve)
            if not per_seed_curves:
                continue
            n_eps = min(len(c) for c in per_seed_curves)
            arr = np.array([c[:n_eps] for c in per_seed_curves], dtype=np.float64)
            mean = np.nanmean(arr, axis=0)
            sem = (
                np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
                if arr.shape[0] > 1 else np.zeros_like(mean)
            )
            xs = np.arange(1, n_eps + 1)
            ax.plot(
                xs, mean,
                color=COND_COLOURS[cond],
                marker=COND_MARKERS[cond] if linestyle == "-" else None,
                markersize=4,
                lw=1.4,
                ls=linestyle,
                label=CONDITIONS[cond]["label"] if show_cond_legend else None,
            )
            ax.fill_between(
                xs, mean - sem, mean + sem,
                color=COND_COLOURS[cond], alpha=0.12, linewidth=0,
            )
        if label is not None:
            ax.set_ylabel(label, fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if key == "acc":
            ax.set_ylim(0, 100)
            ax.axhline(10.0, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.6)
    # Tiny inline legend on the rates panel to disambiguate E (solid) vs I (dashed).
    from matplotlib.lines import Line2D
    ax_re.legend(
        handles=[
            Line2D([0], [0], color=theme.INK_BLACK, lw=1.4, ls="-",  label="E"),
            Line2D([0], [0], color=theme.INK_BLACK, lw=1.4, ls="--", label="I"),
        ],
        fontsize=theme.SIZE_LEGEND, frameon=False, loc="best",
    )
    axes[1, 0].set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    axes[1, 1].set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    axes[0, 0].legend(
        fontsize=theme.SIZE_LEGEND, frameon=False, loc="best",
    )
    fig.suptitle(
        "Trainable-W^EI/W^IE training trajectories — "
        r"does gradient descent discover PING on MNIST?",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_condition_card(
    cond: str,
    metrics_list: list[dict],
    final_cells: list[dict],
    raster: dict | None,
    out_path: Path, run_id: str,
) -> None:
    """Per-condition diagnostic card: trajectory strip on top, PSD + raster below."""
    theme.apply()
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    color = COND_COLOURS[cond]
    label = CONDITIONS[cond]["label"]

    fig = plt.figure(figsize=(14.0, 7.0), dpi=150)
    gs = GridSpec(
        2, 4, figure=fig,
        height_ratios=[1.0, 1.4], hspace=0.45, wspace=0.32,
    )
    ax_wei  = fig.add_subplot(gs[0, 0])
    ax_wie  = fig.add_subplot(gs[0, 1])
    ax_rate = fig.add_subplot(gs[0, 2])
    ax_acc  = fig.add_subplot(gs[0, 3])
    ax_psd  = fig.add_subplot(gs[1, 0:2])
    ax_rast = fig.add_subplot(gs[1, 2:4])

    # --- trajectory strip (mean + per-seed faint lines) ---
    traj_specs = [
        (ax_wei,  "weight_norms", "W_ei.1", r"$\|W^{EI}\|_F$",  "-"),
        (ax_wie,  "weight_norms", "W_ie.1", r"$\|W^{IE}\|_F$",  "-"),
        (ax_rate, "rate_e",       None,     "Firing rate (Hz)", "-"),
        (ax_rate, "rate_i",       None,     None,               "--"),
        (ax_acc,  "acc",          None,     "Test accuracy (%)","-"),
    ]
    for ax, key, sub_key, ylabel, linestyle in traj_specs:
        per_seed = []
        for m in metrics_list:
            curve = []
            for e in m.get("epochs", []) or []:
                val = e.get(key)
                if sub_key is not None:
                    val = (val or {}).get(sub_key)
                curve.append(float(val) if val is not None else float("nan"))
            if curve:
                per_seed.append(curve)
        if not per_seed:
            continue
        n_eps = min(len(c) for c in per_seed)
        arr = np.array([c[:n_eps] for c in per_seed], dtype=np.float64)
        xs = np.arange(1, n_eps + 1)
        for row in arr:
            ax.plot(xs, row, color=color, lw=0.6, ls=linestyle, alpha=0.35)
        ax.plot(xs, np.nanmean(arr, axis=0), color=color, lw=1.6, ls=linestyle)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=theme.SIZE_LABEL)
        ax.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if key == "acc":
            ax.set_ylim(0, 100)
            ax.axhline(10.0, color=theme.GREY_MID, lw=0.5, ls=":", alpha=0.6)
    ax_rate.legend(
        handles=[
            Line2D([0], [0], color=color, lw=1.6, ls="-",  label="E"),
            Line2D([0], [0], color=color, lw=1.6, ls="--", label="I"),
        ],
        fontsize=theme.SIZE_LEGEND, frameon=False, loc="best",
    )

    # --- final PSD (per-seed faint + mean bold) ---
    if final_cells:
        freqs = np.array(final_cells[0]["freqs_hz"])
        band = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
        psds = np.stack([np.array(c["psd"]) for c in final_cells], axis=0)
        for row in psds:
            ax_psd.plot(freqs[band], row[band], color=color, lw=0.6, alpha=0.35)
        ax_psd.plot(
            freqs[band], psds.mean(axis=0)[band], color=color, lw=1.6,
        )
    ax_psd.set_xlabel("Frequency (Hz)", fontsize=theme.SIZE_LABEL)
    ax_psd.set_ylabel("Population E PSD (a.u.)", fontsize=theme.SIZE_LABEL)
    ax_psd.set_xlim(F_GAMMA_BAND_HZ)
    ax_psd.spines["top"].set_visible(False)
    ax_psd.spines["right"].set_visible(False)
    ax_psd.set_title("Trained-network E PSD", fontsize=theme.SIZE_LABEL)

    # --- single-trial raster ---
    if raster is not None:
        T = raster["e"].shape[0]
        t_axis = np.arange(T) * raster["dt_ms"]
        n_e = raster["e"].shape[1]
        n_i = raster["i"].shape[1] if raster["i"].ndim == 2 else 0
        rng = np.random.default_rng(42)
        n_e_plot = min(200, n_e)
        e_idx = np.sort(rng.choice(n_e, n_e_plot, replace=False))
        e_t, e_n = np.where(raster["e"][:, e_idx])
        ax_rast.scatter(
            t_axis[e_t], e_n, s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.4,
        )
        if n_i > 0:
            n_i_plot = min(50, n_i)
            i_idx = np.sort(rng.choice(n_i, n_i_plot, replace=False))
            i_t, i_n = np.where(raster["i"][:, i_idx])
            ax_rast.scatter(
                t_axis[i_t], i_n + n_e_plot + 6,
                s=2.0, c=theme.DEEP_RED, marker="|", linewidths=0.4,
            )
            ax_rast.set_ylim(-2, n_e_plot + n_i_plot + 8)
            ax_rast.set_yticks([n_e_plot / 2, n_e_plot + 6 + n_i_plot / 2])
            ax_rast.set_yticklabels(["E", "I"])
        else:
            ax_rast.set_ylim(-2, n_e_plot + 2)
            ax_rast.set_yticks([n_e_plot / 2])
            ax_rast.set_yticklabels(["E"])
    ax_rast.set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    ax_rast.spines["top"].set_visible(False)
    ax_rast.spines["right"].set_visible(False)
    ax_rast.set_title("Single-trial raster (seed 42)", fontsize=theme.SIZE_LABEL)

    fig.suptitle(f"{label}", fontsize=theme.SIZE_TITLE)
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_final_psds(
    final_by_cond: dict[str, list[dict]],
    out_path: Path, run_id: str,
) -> None:
    """One PSD curve per condition (mean across seeds) — peak marks f_γ."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    for cond in COND_ORDER:
        cells = final_by_cond.get(cond, [])
        if not cells:
            continue
        freqs = np.array(cells[0]["freqs_hz"])
        psd_mean = np.mean(
            np.stack([np.array(c["psd"]) for c in cells], axis=0), axis=0,
        )
        band = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
        ax.plot(
            freqs[band], psd_mean[band],
            color=COND_COLOURS[cond], lw=1.4,
            label=CONDITIONS[cond]["label"],
        )
    ax.set_xlabel("Frequency (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Population E PSD (a.u.)", fontsize=theme.SIZE_LABEL)
    ax.set_xlim(F_GAMMA_BAND_HZ)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(
        "Trained-network E-population PSDs — gamma peak indicates "
        "discovered PING dynamics",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_raster_strip(
    rasters_by_cond: dict[str, dict],
    out_path: Path, run_id: str,
) -> None:
    theme.apply()
    n = len(COND_ORDER)
    fig, axes = plt.subplots(
        n, 1, figsize=(10.0, 1.5 * n + 1.0),
        sharex=True, gridspec_kw={"hspace": 0.22},
    )
    if n == 1:
        axes = [axes]
    rng = np.random.default_rng(42)
    for ax, cond in zip(axes, COND_ORDER):
        s = rasters_by_cond.get(cond)
        if s is None:
            continue
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt_ms"]
        n_e = s["e"].shape[1]
        n_i = s["i"].shape[1] if s["i"].ndim == 2 else 0
        n_e_plot = min(200, n_e)
        e_idx = np.sort(rng.choice(n_e, n_e_plot, replace=False))
        e_raster = s["e"][:, e_idx]
        e_t, e_n = np.where(e_raster)
        ax.scatter(t_axis[e_t], e_n,
                   s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.4)
        if n_i > 0:
            n_i_plot = min(50, n_i)
            i_idx = np.sort(rng.choice(n_i, n_i_plot, replace=False))
            i_raster = s["i"][:, i_idx]
            i_t, i_n = np.where(i_raster)
            ax.scatter(
                t_axis[i_t], i_n + n_e_plot + 6,
                s=2.0, c=theme.DEEP_RED, marker="|", linewidths=0.4,
            )
            ax.set_ylim(-2, n_e_plot + n_i_plot + 8)
            ax.set_yticks([n_e_plot / 2, n_e_plot + 6 + n_i_plot / 2])
            ax.set_yticklabels(["E", "I"])
        else:
            ax.set_ylim(-2, n_e_plot + 2)
            ax.set_yticks([n_e_plot / 2])
            ax.set_yticklabels(["E"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, s["t_ms"])
        ax.text(
            1.012, 0.5, CONDITIONS[cond]["label"],
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=theme.SIZE_LABEL,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    fig.suptitle(
        "Trained-network single-trial rasters per condition",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────
def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    notebook_run_id = next_run_id(SLUG)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)
    t_start = time.monotonic()

    only_missing = "--only-missing" in sys.argv
    skip_training = "--no-train" in sys.argv

    if not skip_training:
        dispatcher = BatchDispatcher(modal_gpu, REPO, OSCILLOSCOPE)
        for cond in COND_ORDER:
            for seed in SEEDS:
                out = cell_dir(cond, seed)
                if only_missing and (out / "metrics.json").exists():
                    print(f"[skip] {cond}/seed={seed} → exists")
                    continue
                print(f"[train] {cond}/seed={seed} → {out.relative_to(REPO)}")
                dispatcher.submit(build_train_args(cond, seed, tier, out), out)
        dispatcher.drain()

    # ── Per-condition aggregation
    print("[aggregate] reading metrics from trained cells")
    metrics_by_cond: dict[str, list[dict]] = {}
    final_by_cond: dict[str, list[dict]] = {}
    rasters_by_cond: dict[str, dict] = {}
    summary_rows: list[dict] = []

    from cli import _auto_device
    device = _auto_device()
    for cond in COND_ORDER:
        metrics_list: list[dict] = []
        final_list: list[dict] = []
        for seed in SEEDS:
            d = cell_dir(cond, seed)
            if not (d / "metrics.json").exists():
                print(f"  missing {cond}/seed={seed} → skipping")
                continue
            m = load_metrics(d)
            metrics_list.append(m)
            f = measure_trained_state(d, device)
            final_list.append(f)
            summary_rows.append({
                "condition": cond,
                "seed": int(seed),
                **{k: v for k, v in f.items() if k not in ("psd", "freqs_hz")},
            })
            print(
                f"  {cond} seed={seed}  acc={f['acc']:5.2f}%  "
                f"E={f['e_rate_hz']:5.2f} Hz  I={f['i_rate_hz']:5.2f} Hz  "
                f"f_γ={f['f_gamma_hz']:5.2f} Hz  "
                f"|W_ei|={f['w_ei_mean']:.4f}  |W_ie|={f['w_ie_mean']:.4f}"
            )
        metrics_by_cond[cond] = metrics_list
        final_by_cond[cond] = final_list
        if final_list:
            # First seed's raster as representative.
            rasters_by_cond[cond] = capture_raster(
                cell_dir(cond, SEEDS[0]), device,
            )

    # Per-condition diagnostic cards + matching weight-matrix cards.
    for cond in COND_ORDER:
        out = FIGURES / f"card__{cond}.png"
        plot_condition_card(
            cond,
            metrics_by_cond.get(cond, []),
            final_by_cond.get(cond, []),
            rasters_by_cond.get(cond),
            out, notebook_run_id,
        )
        print(f"wrote {out}")
        seed_to_dir = {
            s: cell_dir(cond, s) for s in SEEDS
            if (cell_dir(cond, s) / "weights.pth").exists()
        }
        if seed_to_dir:
            w_out = FIGURES / f"weights__{cond}.png"
            plot_weight_matrices(cond, seed_to_dir, device, w_out, notebook_run_id)
            print(f"wrote {w_out}")

    duration_s = time.monotonic() - t_start
    summary = {
        "notebook_run_id": notebook_run_id,
        "duration_s": round(duration_s, 1),
        "config": {
            "tier": tier,
            "epochs": TIER_CONFIG[tier]["epochs"],
            "max_samples": TIER_CONFIG[tier]["max_samples"],
            "seeds": SEEDS,
            "conditions": CONDITIONS,
            "common_recipe": COMMON_RECIPE,
        },
        "summary": summary_rows,
    }
    def _clean(o):
        if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
            return None
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        return o
    (FIGURES / "numbers.json").write_text(
        json.dumps(_clean(summary), indent=2, allow_nan=False) + "\n"
    )
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  duration: {duration_s:.1f}s")


if __name__ == "__main__":
    main()
