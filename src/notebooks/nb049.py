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
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from helpers.figsave import save_figure  # noqa: E402
from helpers.modal import BatchDispatcher, parse_modal_gpu  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402
from helpers import theme  # noqa: E402

SLUG = "nb049"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
OSCILLOSCOPE = REPO / "src" / "cli/cli.py"

TIER_CONFIG = {
    "extra small": dict(max_samples=100, epochs=2),
    "small":       dict(max_samples=500, epochs=10),
    "medium":      dict(max_samples=3500, epochs=100),  # 5% of the 70k MNIST corpus
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
    """Trained cell — now the shared nb022 cell (train-once / reuse-many)."""
    from nb022 import cell_dir as shared_cell_dir
    return shared_cell_dir(f"{condition}__seed{seed}")


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
def _infer_cell(train_dir: Path, extra_args: list[str], out_name: str) -> Path:
    """Shell out to the CLI's `sim --infer` for one trained cell; return the out dir.

    Network construction, weight loading and the forward pass all happen in the CLI —
    this notebook only runs it and reads the artifacts. extra_args adds mode flags
    (e.g. --emit-pop-traces for PSD, --sample-index for a snapshot raster).
    """
    train_dir = train_dir.resolve()
    out_dir = (ARTIFACTS / out_name / train_dir.name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv", "run", "python", str(OSCILLOSCOPE), "sim", "--infer",
            "--load-config", str(train_dir / "config.json"),
            "--load-weights", str(train_dir / "weights.pth"),
            "--out-dir", str(out_dir),
            *extra_args,
        ],
        cwd=REPO,
        check=True,
    )
    return out_dir


F_GAMMA_BAND_HZ: tuple[float, float] = (5.0, 150.0)


def load_init_and_trained_weights(train_dir: Path):
    """Return (W_ei_init, W_ei_trained, W_ie_init, W_ie_trained) as numpy arrays.

    Shells out to the CLI's dump-weights mode, which rebuilds the net under the
    cell's seed to recover the deterministic init weights and reads the trained
    values from the saved state_dict, then loads the emitted weights_dump.npz.
    Network construction stays inside the CLI — the notebook only runs it.
    """
    # Resolve to absolute paths: the subprocess runs with cwd=REPO, so any
    # relative train_dir would otherwise be interpreted against the wrong root.
    train_dir = train_dir.resolve()
    out_dir = (ARTIFACTS / "weights_dump" / train_dir.name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv", "run", "python", str(OSCILLOSCOPE), "dump-weights",
            "--load-config", str(train_dir / "config.json"),
            "--load-weights", str(train_dir / "weights.pth"),
            "--out-dir", str(out_dir),
        ],
        cwd=REPO,
        check=True,
    )
    d = np.load(out_dir / "weights_dump.npz")
    return (
        d["W_ei_1_init"],
        d["W_ei_1_trained"],
        d["W_ie_1_init"],
        d["W_ie_1_trained"],
    )


def plot_weight_matrices(
    cond: str,
    seed_to_dir: dict[int, Path],
    out_path: Path, run_id: str,
) -> None:
    """Per-condition weight-distribution card: W^EI and W^IE, init vs trained.

    Two panels side-by-side. Each panel shows the *surviving* (>0) weights as
    overlaid histograms (init outline, trained fill). The fraction of entries
    pruned to zero by Dale's-law clamping is shown as a small bar chart inset
    so the spike-at-zero doesn't dominate the histogram visually.
    """
    theme.apply()
    from matplotlib.gridspec import GridSpec
    label = CONDITIONS[cond]["label"]
    seeds_sorted = sorted(seed_to_dir.keys())

    # Pool across all available seeds — the legend stats then describe what
    # happens on average, not what happens to any one initialisation.
    ei_init_chunks, ei_trained_chunks = [], []
    ie_init_chunks, ie_trained_chunks = [], []
    for seed in seeds_sorted:
        a, b, c, d = load_init_and_trained_weights(seed_to_dir[seed])
        ei_init_chunks.append(a)
        ei_trained_chunks.append(b)
        ie_init_chunks.append(c)
        ie_trained_chunks.append(d)
    w_ei_init    = np.concatenate([x.ravel() for x in ei_init_chunks])
    w_ei_trained = np.concatenate([x.ravel() for x in ei_trained_chunks])
    w_ie_init    = np.concatenate([x.ravel() for x in ie_init_chunks])
    w_ie_trained = np.concatenate([x.ravel() for x in ie_trained_chunks])

    # Canonical biophysical means (from N(1.0, 0.1) / 1024 and N(2.0, 0.2) / 256)
    canon_ei = 1.0 / 1024.0
    canon_ie = 2.0 / 256.0

    fig = plt.figure(figsize=(6.9, 2.55), dpi=150)
    gs = GridSpec(
        2, 2, figure=fig,
        width_ratios=[1.0, 1.0],
        height_ratios=[0.10, 1.0],
        hspace=0.45, wspace=0.30,
        top=0.92, bottom=0.16, left=0.06, right=0.97,
    )
    ax_hdr = fig.add_subplot(gs[0, :])
    ax_ei  = fig.add_subplot(gs[1, 0])
    ax_ie  = fig.add_subplot(gs[1, 1])

    # --- header ---
    ax_hdr.set_axis_off()
    ax_hdr.text(
        0.0, 0.5, label,
        transform=ax_hdr.transAxes, ha="left", va="center",
        fontsize=theme.SIZE_TITLE + 1, fontweight="semibold",
        color=COND_COLOURS[cond],
    )
    ax_hdr.text(
        1.0, 0.5,
        f"Recurrent-weight distributions  ·  pooled across {len(seeds_sorted)} seeds  ·  effective (post-clamp) values",
        transform=ax_hdr.transAxes, ha="right", va="center",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, fontfamily="monospace",
    )

    def _panel(ax, init_arr, trained_arr, title, color, canon_mean):
        flat_init = np.maximum(init_arr.ravel(), 0.0)
        flat_trained = np.maximum(trained_arr.ravel(), 0.0)
        nonzero_init    = flat_init[flat_init > 0]
        nonzero_trained = flat_trained[flat_trained > 0]
        v_hi = float(max(
            nonzero_init.max() if nonzero_init.size else 0.0,
            nonzero_trained.max() if nonzero_trained.size else 0.0,
            canon_mean * 1.2,
            1e-12,
        ))
        bins = np.linspace(0.0, v_hi * 1.05, 50)

        eff_init = flat_init.mean()
        eff_trained = flat_trained.mean()
        frac_pruned_init = float((flat_init <= 0).mean())
        frac_pruned_trained = float((flat_trained <= 0).mean())

        # Surviving (>0) distributions only — keeps the histogram readable.
        if nonzero_init.size:
            ax.hist(
                nonzero_init, bins=bins, histtype="step",
                color=color, lw=1.6, label="init",
            )
        if nonzero_trained.size:
            ax.hist(
                nonzero_trained, bins=bins, histtype="stepfilled",
                color=color, alpha=0.32, edgecolor=color, lw=0.8,
                label="trained",
            )

        # Reference vertical at the canonical biophysical mean.
        ax.axvline(canon_mean, color=theme.GREY_MID, lw=0.9, ls=":")
        y_lo, y_hi = ax.get_ylim()
        ax.text(
            canon_mean, y_hi * 0.5, "  canonical",
            ha="left", va="center", fontsize=theme.SIZE_CAPTION,
            color=theme.GREY_MID, rotation=90,
        )

        ax.set_title(title, fontsize=theme.SIZE_LABEL, loc="left", pad=4)
        ax.set_xlabel("effective weight  (Dale-clamped, surviving entries shown)",
                      fontsize=theme.SIZE_LABEL)
        ax.set_ylabel("entry count", fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=theme.SIZE_LABEL - 1, direction="out", length=3)
        ax.set_xlim(0, v_hi * 1.05)
        ax.legend(
            fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right",
        )

        # Stats box (upper-right): the two key numbers per row.
        stat_text = (
            f"init     mean = {eff_init:.4f}    pruned = {frac_pruned_init:5.1%}\n"
            f"trained  mean = {eff_trained:.4f}    pruned = {frac_pruned_trained:5.1%}"
        )
        ax.text(
            0.99, 0.78, stat_text,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=theme.SIZE_CAPTION, color=theme.LABEL,
            fontfamily="monospace",
            bbox=dict(facecolor="white", edgecolor=theme.GREY_MID,
                      lw=0.5, boxstyle="round,pad=0.4", alpha=0.95),
        )

    _panel(ax_ei, w_ei_init, w_ei_trained,
           r"$W^{EI}$  (1024 × 256)", theme.INK_BLACK, canon_ei)
    _panel(ax_ie, w_ie_init, w_ie_trained,
           r"$W^{IE}$  (256 × 1024)", theme.DEEP_RED, canon_ie)

    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


def measure_trained_state(train_dir: Path) -> dict:
    """Report acc, mean E/I rate, f_γ (Welch PSD peak), and W_ei/W_ie means.

    Runs `sim --infer --emit-pop-traces` (acc + rates in metrics.json, population
    traces in pop_traces.npz) and dump-weights (trained W_ei/W_ie), then computes
    the PSD and gamma peak locally. Metric logic stays in the notebook; the CLI
    only emits the base data.
    """
    from scipy import signal as sp_signal

    cfg = json.loads((train_dir / "config.json").read_text())
    out_dir = _infer_cell(train_dir, ["--emit-pop-traces"], "infer")
    m = json.loads((out_dir / "metrics.json").read_text())
    rates = m.get("rates_hz", {})

    # Trained W_ei / W_ie means via dump-weights (mean abs of the trained matrices).
    _, w_ei_trained, _, w_ie_trained = load_init_and_trained_weights(train_dir)
    w_ei_mean = float(np.abs(w_ei_trained).mean())
    w_ie_mean = float(np.abs(w_ie_trained).mean())

    # f_γ via Welch PSD on the per-trial E-population trace (from pop_traces.npz).
    pt = np.load(out_dir / "pop_traces.npz")
    pop_e_traces = list(pt["pop_e"])
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
        "acc": float(m["best_acc"]),
        "e_rate_hz": float(rates.get("hid", 0.0)),
        "i_rate_hz": float(rates.get("inh", 0.0)),
        "f_gamma_hz": f_gamma,
        "w_ei_mean": w_ei_mean,
        "w_ie_mean": w_ie_mean,
        "psd": psd_used.tolist(),
        "freqs_hz": freqs_used.tolist(),
    }


def capture_raster(train_dir: Path, sample_idx: int = 0) -> dict:
    """Single-trial raster for the trained network, via the CLI snapshot.

    Runs `sim --infer --sample-index N` and reads E/I rasters + the sample's
    class label from snapshot.npz.
    """
    cfg = json.loads((train_dir / "config.json").read_text())
    out_dir = _infer_cell(train_dir, ["--sample-index", str(sample_idx)], "snapshot")
    d = np.load(out_dir / "snapshot.npz")
    e = d["spk_e"]
    i = d["spk_i"]
    if e.ndim == 3:
        e = e[:, 0, :]
    if i.ndim == 3:
        i = i[:, 0, :]
    return {
        "e": e.astype(bool),
        "i": i.astype(bool),
        "dt_ms": float(cfg["dt"]),
        "t_ms": float(cfg["t_ms"]),
        "label": int(d["label"]),
    }


# ── Plotting ────────────────────────────────────────────────────────


def plot_training_trajectories(
    metrics_by_cond: dict[str, list[dict]],
    out_path: Path, run_id: str,
) -> None:
    """2×2 panel: W_ei norm, W_ie norm, E rate, accuracy — vs epoch.
    Each line is the mean of three seeds; light shading is mean ± SEM."""
    theme.apply()
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.02), dpi=150, sharex=True)
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
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
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

    # --- summary stats for header ---
    def _agg(key, sub=None):
        vals = []
        for c in final_cells:
            v = c.get(key)
            if isinstance(v, dict) and sub is not None:
                v = v.get(sub)
            if v is not None and not (isinstance(v, float) and v != v):
                vals.append(float(v))
        return float(np.mean(vals)) if vals else float("nan")

    acc_final  = _agg("acc")
    e_rate_f   = _agg("e_rate_hz")
    i_rate_f   = _agg("i_rate_hz")
    f_gamma_f  = _agg("f_gamma_hz")

    fig = plt.figure(figsize=(6.9, 4.34), dpi=150)
    gs = GridSpec(
        3, 4, figure=fig,
        height_ratios=[0.32, 1.0, 1.4],
        hspace=0.55, wspace=0.40,
        top=0.96, bottom=0.07, left=0.06, right=0.97,
    )
    ax_hdr  = fig.add_subplot(gs[0, :])
    ax_wei  = fig.add_subplot(gs[1, 0])
    ax_wie  = fig.add_subplot(gs[1, 1])
    ax_rate = fig.add_subplot(gs[1, 2])
    ax_acc  = fig.add_subplot(gs[1, 3])
    ax_psd  = fig.add_subplot(gs[2, 0:2])
    ax_rast = fig.add_subplot(gs[2, 2:4])

    # --- header strip ---
    ax_hdr.set_axis_off()
    ax_hdr.text(
        0.0, 1.0, label,
        transform=ax_hdr.transAxes, ha="left", va="top",
        fontsize=theme.SIZE_TITLE + 2, fontweight="semibold", color=color,
    )
    stat_pieces = [
        f"acc = {acc_final:5.2f}%",
        f"E = {e_rate_f:5.1f} Hz" if e_rate_f == e_rate_f else "E = —",
        f"I = {i_rate_f:5.1f} Hz" if i_rate_f == i_rate_f else "I = —",
        f"f$_\\gamma$ = {f_gamma_f:4.1f} Hz" if f_gamma_f == f_gamma_f else "f$_\\gamma$ = —",
    ]
    ax_hdr.text(
        0.0, 0.0, "    ".join(stat_pieces),
        transform=ax_hdr.transAxes, ha="left", va="bottom",
        fontsize=theme.SIZE_LABEL + 1, color=theme.LABEL, fontfamily="monospace",
    )

    # --- trajectory strip (per-seed alpha + mean) ---
    def _per_seed_curves(key, sub_key=None):
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
            return None, None
        n_eps = min(len(c) for c in per_seed)
        arr = np.array([c[:n_eps] for c in per_seed], dtype=np.float64)
        xs = np.arange(1, n_eps + 1)
        return xs, arr

    def _plot_traj(ax, xs, arr, *, ls="-", lw_mean=1.8):
        for row in arr:
            ax.plot(xs, row, color=color, lw=0.7, ls=ls, alpha=0.28)
        ax.plot(xs, np.nanmean(arr, axis=0), color=color, lw=lw_mean, ls=ls,
                solid_capstyle="round")

    def _style_panel(ax, *, ylabel, last_xlabel=False, last_val=None,
                     last_val_fmt="{:.4f}"):
        ax.set_ylabel(ylabel, fontsize=theme.SIZE_LABEL)
        ax.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=theme.SIZE_LABEL - 1, direction="out", length=3)
        if last_val is not None and last_val == last_val:
            ax.text(
                0.99, 0.97, last_val_fmt.format(last_val),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=theme.SIZE_LABEL - 1, color=color, fontweight="semibold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.0),
            )

    def _empty_panel(ax, ylabel, message):
        ax.set_ylabel(ylabel, fontsize=theme.SIZE_LABEL)
        ax.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=theme.SIZE_LABEL - 1, direction="out", length=3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.text(
            0.5, 0.5, message,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=theme.SIZE_LABEL, color=theme.GREY_MID, fontstyle="italic",
        )

    # ‖W^EI‖
    xs, arr = _per_seed_curves("weight_norms", "W_ei.1")
    if xs is not None and np.isfinite(arr).any() and arr.max() > 0:
        _plot_traj(ax_wei, xs, arr)
        _style_panel(
            ax_wei, ylabel=r"$\|W^{EI}\|_F$",
            last_val=float(np.nanmean(arr[:, -1])), last_val_fmt="end={:.3f}",
        )
    else:
        msg = "frozen" if cond == "frozen_ping" else "no gradient"
        _empty_panel(ax_wei, ylabel=r"$\|W^{EI}\|_F$", message=msg)
    # ‖W^IE‖
    xs, arr = _per_seed_curves("weight_norms", "W_ie.1")
    if xs is not None and np.isfinite(arr).any() and arr.max() > 0:
        _plot_traj(ax_wie, xs, arr)
        _style_panel(
            ax_wie, ylabel=r"$\|W^{IE}\|_F$",
            last_val=float(np.nanmean(arr[:, -1])), last_val_fmt="end={:.3f}",
        )
    else:
        msg = "frozen" if cond == "frozen_ping" else "no gradient"
        _empty_panel(ax_wie, ylabel=r"$\|W^{IE}\|_F$", message=msg)
    # Firing rates: E (solid) + I (dashed)
    xs_e, arr_e = _per_seed_curves("rate_e")
    xs_i, arr_i = _per_seed_curves("rate_i")
    if xs_e is not None:
        _plot_traj(ax_rate, xs_e, arr_e, ls="-")
    if xs_i is not None:
        _plot_traj(ax_rate, xs_i, arr_i, ls="--")
    _style_panel(ax_rate, ylabel="Rate (Hz)")
    ax_rate.legend(
        handles=[
            Line2D([0], [0], color=color, lw=1.8, ls="-",  label="E"),
            Line2D([0], [0], color=color, lw=1.8, ls="--", label="I"),
        ],
        fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left",
    )
    # Accuracy
    xs, arr = _per_seed_curves("acc")
    if xs is not None:
        _plot_traj(ax_acc, xs, arr)
        _style_panel(
            ax_acc, ylabel="Test accuracy (%)",
            last_val=float(np.nanmean(arr[:, -1])), last_val_fmt="end={:.1f}%",
        )
        ax_acc.set_ylim(0, 100)
        ax_acc.axhline(10.0, color=theme.GREY_MID, lw=0.6, ls=":", alpha=0.6)
        ax_acc.text(
            0.02, 0.13, "chance", transform=ax_acc.transAxes,
            fontsize=theme.SIZE_CAPTION, color=theme.GREY_MID,
        )

    # --- final PSD ---
    if final_cells:
        freqs = np.array(final_cells[0]["freqs_hz"])
        band = (freqs >= F_GAMMA_BAND_HZ[0]) & (freqs <= F_GAMMA_BAND_HZ[1])
        psds = np.stack([np.array(c["psd"]) for c in final_cells], axis=0)
        for row in psds:
            ax_psd.plot(freqs[band], row[band], color=color, lw=0.7, alpha=0.3)
        psd_mean = psds.mean(axis=0)
        ax_psd.plot(freqs[band], psd_mean[band], color=color, lw=1.8)
        # Mark f_γ if defined
        if f_gamma_f == f_gamma_f and F_GAMMA_BAND_HZ[0] <= f_gamma_f <= F_GAMMA_BAND_HZ[1]:
            ax_psd.axvline(f_gamma_f, color=color, lw=0.9, ls="--", alpha=0.55)
            ax_psd.text(
                f_gamma_f, ax_psd.get_ylim()[1] * 0.95,
                f"  $f_\\gamma$ = {f_gamma_f:.1f} Hz",
                ha="left", va="top", fontsize=theme.SIZE_LABEL - 1,
                color=color, fontweight="semibold",
            )
    ax_psd.set_xlabel("Frequency (Hz)", fontsize=theme.SIZE_LABEL)
    ax_psd.set_ylabel("Population E PSD (a.u.)", fontsize=theme.SIZE_LABEL)
    ax_psd.set_xlim(F_GAMMA_BAND_HZ)
    ax_psd.spines["top"].set_visible(False)
    ax_psd.spines["right"].set_visible(False)
    ax_psd.tick_params(labelsize=theme.SIZE_LABEL - 1, direction="out", length=3)
    ax_psd.set_title("Trained-network E-population PSD",
                     fontsize=theme.SIZE_LABEL, loc="left", pad=4)

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
            t_axis[e_t], e_n, s=1.6, c=theme.INK_BLACK, marker="|", linewidths=0.35,
        )
        if n_i > 0:
            n_i_plot = min(50, n_i)
            i_idx = np.sort(rng.choice(n_i, n_i_plot, replace=False))
            i_t, i_n = np.where(raster["i"][:, i_idx])
            divider_y = n_e_plot + 4
            ax_rast.axhline(divider_y, color=theme.GREY_MID, lw=0.5, alpha=0.5)
            ax_rast.scatter(
                t_axis[i_t], i_n + divider_y + 4,
                s=1.6, c=theme.DEEP_RED, marker="|", linewidths=0.35,
            )
            ax_rast.set_ylim(-2, divider_y + 4 + n_i_plot + 2)
            ax_rast.set_yticks([n_e_plot / 2, divider_y + 4 + n_i_plot / 2])
            ax_rast.set_yticklabels(
                [f"E ({n_e})", f"I ({n_i})"], fontsize=theme.SIZE_LABEL - 1,
            )
        else:
            ax_rast.set_ylim(-2, n_e_plot + 2)
            ax_rast.set_yticks([n_e_plot / 2])
            ax_rast.set_yticklabels([f"E ({n_e})"], fontsize=theme.SIZE_LABEL - 1)
    ax_rast.set_xlabel("Time (ms)", fontsize=theme.SIZE_LABEL)
    ax_rast.spines["top"].set_visible(False)
    ax_rast.spines["right"].set_visible(False)
    ax_rast.tick_params(labelsize=theme.SIZE_LABEL - 1, direction="out", length=3)
    ax_rast.set_title("Single-trial raster (seed 42)",
                      fontsize=theme.SIZE_LABEL, loc="left", pad=4)

    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))
    plt.close(fig)


def plot_final_psds(
    final_by_cond: dict[str, list[dict]],
    out_path: Path, run_id: str,
) -> None:
    """One PSD curve per condition (mean across seeds) — peak marks f_γ."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(6.9, 3.88), dpi=150)
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
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


def plot_raster_strip(
    rasters_by_cond: dict[str, dict],
    out_path: Path, run_id: str,
) -> None:
    theme.apply()
    n = len(COND_ORDER)
    fig, axes = plt.subplots(
        n, 1, figsize=(6.9, 1.035 * n + 0.69),
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
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("png", "pdf"))
    plt.close(fig)


def fig_attractor(summary_rows, out_path, run_id):
    """One-glance summary in the (E rate, I rate) plane. The frozen control sits in
    the PING corner (low E, high I); every trainable condition collapses to the
    dense-E / silent-I (COBA) corner regardless of its initialisation, at ≈ equal
    accuracy — so gradient descent never reaches PING when the loop can train."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(5.6, 3.5), dpi=150)
    ax.axhline(0.0, color=theme.FAINT, lw=0.8, ls=":")
    for cond in COND_ORDER:
        rows = [r for r in summary_rows if r["condition"] == cond]
        if not rows:
            continue
        E = [r["e_rate_hz"] for r in rows]
        I = [r["i_rate_hz"] for r in rows]
        acc = float(np.nanmean([r["acc"] for r in rows]))
        ax.scatter(E, I, s=95, color=COND_COLOURS[cond], marker=COND_MARKERS[cond],
                   edgecolor="white", linewidths=0.7, zorder=5,
                   label=f"{CONDITIONS[cond]['label']}  ·  {acc:.0f}%")
    ax.annotate("PING\nloop on", xy=(9, 38), xytext=(17, 33),
                fontsize=theme.SIZE_LABEL, color=theme.MUTED, va="center",
                arrowprops=dict(arrowstyle="->", color=theme.MUTED, lw=1.0))
    ax.annotate("loop pruned → COBA\n(I silent)", xy=(45, 0.6), xytext=(28, 13),
                fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED, va="center",
                arrowprops=dict(arrowstyle="->", color=theme.DEEP_RED, lw=1.0))
    ax.set_xlabel("E firing rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("I firing rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_title("Trainable W_EI/W_IE never reach PING — every init collapses to the loop-free corner",
                 fontsize=theme.SIZE_LABEL, color=theme.INK)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right",
              title="condition · test acc", title_fontsize=theme.SIZE_LEGEND)
    ax.margins(0.13)
    stamp_figure(fig, run_id)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────
def _load_epoch_curves() -> dict:
    """Read per-epoch metrics.jsonl for every nb049 condition from the shared
    training root (nb022 train-once / reuse-many), not nb049's own dir."""
    runs = {}
    for cond in COND_ORDER:
        for seed in SEEDS:
            d = cell_dir(cond, seed)
            f = d / "metrics.jsonl"
            if not f.exists():
                continue
            rows = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
            runs[d.name] = {
                "cond": cond,
                "ep": [r["ep"] for r in rows],
                "acc": [r["acc"] for r in rows],
                "rate_e": [r.get("test_rate_e", r.get("rate_e")) for r in rows],
                "rate_i": [r.get("test_rate_i", r.get("rate_i")) for r in rows],
                "contrast": [r.get("contrast") for r in rows],
            }
    return runs


def fig_training_curves(out_path: Path, run_id: str) -> None:
    """Real per-epoch training curves from the logs, 2×2: accuracy, E rate,
    I rate, and pingness (nb054 lobe–trough contrast). Trainable inits (black)
    vs the frozen-PING control (red dashed). The I-rate collapse and the
    pingness gap are the loop being pruned."""
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"  # keep the saved 16:9 exact

    runs = _load_epoch_curves()

    def _smooth(y, w=5):
        y = np.asarray(y, dtype=float)
        if w <= 1 or y.size < w:
            return y
        yp = np.pad(y, w // 2, mode="edge")
        return np.convolve(yp, np.ones(w) / w, mode="valid")[: y.size]

    # Aggregate the three seeds per condition into mean + min/max envelope.
    by_cond: dict[str, list[dict]] = {}
    for d in runs.values():
        by_cond.setdefault(d["cond"], []).append(d)

    panels = [
        ("A", "test accuracy (%)", "acc", (0, 100)),
        ("B", "E firing rate (Hz)", "rate_e", (0, 60)),
        ("C", "I firing rate (Hz)", "rate_i", (0, 55)),
        ("D", "pingness  (lobe–trough contrast)", "contrast", (0, 1.0)),
    ]
    xmax = max(max(d["ep"]) for d in runs.values())
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 3.15), dpi=200,
                             gridspec_kw={"hspace": 0.34, "wspace": 0.2})
    axes = axes.ravel()
    for k, (letter, ylabel, key, ylim) in enumerate(panels):
        ax = axes[k]
        for cond in COND_ORDER:
            seeds = [d for d in by_cond.get(cond, [])
                     if not any(v is None for v in d[key])]
            if not seeds:
                continue
            ep = np.asarray(seeds[0]["ep"], dtype=float)
            stack = np.array([np.asarray(d[key], dtype=float) for d in seeds])
            mean = _smooth(stack.mean(axis=0))
            lo = _smooth(stack.min(axis=0))
            hi = _smooth(stack.max(axis=0))
            color = COND_COLOURS.get(cond, theme.INK_BLACK)
            frozen = cond == "frozen_ping"
            ax.fill_between(ep, lo, hi, color=color, alpha=0.13, lw=0)
            ax.plot(ep, mean, color=color, lw=2.0,
                    ls="--" if frozen else "-")
        ax.set_ylabel(ylabel, fontsize=theme.SIZE_LABEL)
        ax.set_ylim(*ylim)
        ax.set_xlim(0, xmax)
        ax.tick_params(labelsize=theme.SIZE_TICK)
        if k >= 2:
            ax.set_xlabel("training epoch", fontsize=theme.SIZE_LABEL)
        ax.text(0.012, 0.97, letter, transform=ax.transAxes,
                fontsize=theme.SIZE_TITLE + 1, fontweight="bold", va="top")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    # One legend entry per condition, mean-line style, in the accuracy panel.
    for cond in COND_ORDER:
        if cond in by_cond:
            axes[0].plot([], [], color=COND_COLOURS.get(cond, theme.INK_BLACK),
                         lw=2.0, ls="--" if cond == "frozen_ping" else "-",
                         label=CONDITIONS[cond]["label"])
    axes[0].legend(frameon=False, fontsize=theme.SIZE_LEGEND - 1, loc="lower right")
    fig.tight_layout(pad=0.4)
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Crop surrounding whitespace — this is a standalone publication figure, so
    # trim to content rather than holding the fixed 16:9 frame. save_figure takes
    # no kwargs, so the crop is applied via rcParams here.
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.04
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


def fig_phase_portrait(out_path: Path, run_id: str) -> None:
    """(E rate × pingness) phase portrait of training.

    One trajectory per trainable condition (mean across seeds), faded-to-saturated
    along epoch via LineCollection so the flow direction reads off the line itself.
    Open marker = epoch 0, filled marker = final epoch. The frozen-PING control
    sits as a single dot cluster in the PING basin — it can't move because the
    loop gradients are zero, not because it's an attractor of the same dynamics.

    Story: gradient descent has one preferred direction in this plane, and the
    PING corner only exists under architectural enforcement.
    """
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"

    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba

    runs = _load_epoch_curves()
    by_cond: dict[str, list[dict]] = {}
    for d in runs.values():
        by_cond.setdefault(d["cond"], []).append(d)

    fig, ax = plt.subplots(figsize=(5.6, 3.15), dpi=200)

    e_max = 0.0
    final_accs: dict[str, float] = {}

    for cond in COND_ORDER:
        seeds = [
            d for d in by_cond.get(cond, [])
            if not any(v is None for v in d["rate_e"])
            and not any(v is None for v in d["contrast"])
            and not any(v is None for v in d["acc"])
        ]
        if not seeds:
            continue
        e_stack = np.array([np.asarray(d["rate_e"], dtype=float) for d in seeds])
        p_stack = np.array([np.asarray(d["contrast"], dtype=float) for d in seeds])
        a_stack = np.array([np.asarray(d["acc"], dtype=float) for d in seeds])
        e_mean = e_stack.mean(axis=0)
        p_mean = p_stack.mean(axis=0)
        final_accs[cond] = float(a_stack[:, -1].mean())
        e_max = max(e_max, float(e_mean.max()))

        color = COND_COLOURS.get(cond, theme.INK_BLACK)
        marker = COND_MARKERS.get(cond, "o")
        frozen = cond == "frozen_ping"
        label = f"{CONDITIONS[cond]['label']}  ·  {final_accs[cond]:.1f}%"

        if frozen:
            # Single dot cluster — frozen control sits in the PING basin and
            # only drifts a little as the feedforward weights train.
            ax.scatter(
                e_stack[:, -1], p_stack[:, -1],
                s=55, color=color, marker=marker,
                alpha=0.40, edgecolor="none", zorder=6,
            )
            ax.scatter(
                e_mean[-1], p_mean[-1],
                s=170, color=color, marker=marker,
                edgecolor="white", linewidths=1.3, zorder=10, label=label,
            )
        else:
            # Faded-to-saturated trajectory: alpha encodes epoch progress.
            points = np.array([e_mean, p_mean]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            alphas = np.linspace(0.22, 1.0, len(segments))
            seg_colors = [to_rgba(color, alpha=a) for a in alphas]
            lc = LineCollection(segments, colors=seg_colors, linewidths=1.6, zorder=4)
            ax.add_collection(lc)
            # Start marker — hollow
            ax.scatter(
                e_mean[0], p_mean[0],
                s=70, facecolor="white", edgecolor=color, marker=marker,
                linewidths=1.4, zorder=10,
            )
            # End marker — filled
            ax.scatter(
                e_mean[-1], p_mean[-1],
                s=70, color=color, marker=marker,
                edgecolor="white", linewidths=0.9, zorder=10, label=label,
            )

    # Basin annotations — orient the reader without claiming a separatrix.
    ax.text(
        0.02, 0.97, "PING basin\n(loop active)",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=theme.SIZE_LABEL, color=theme.MUTED, fontstyle="italic",
    )
    ax.text(
        0.98, 0.04, "COBA basin\n(loop pruned)",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED, fontstyle="italic",
    )

    ax.set_xlabel("E firing rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("pingness  (lobe–trough contrast)", fontsize=theme.SIZE_LABEL)
    ax.set_title(
        "Training trajectories collapse to COBA — PING persists only under freeze",
        fontsize=theme.SIZE_LABEL - 1, color=theme.INK,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, max(e_max * 1.08, 60.0))
    ax.tick_params(labelsize=theme.SIZE_TICK)
    ax.legend(
        fontsize=theme.SIZE_LEGEND - 1, frameon=False, loc="center right",
        title="condition · final acc", title_fontsize=theme.SIZE_LEGEND - 1,
    )
    ax.text(
        1.0, -0.16, "○ epoch 0   ● final epoch",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=theme.SIZE_CAPTION, color=theme.GREY_MID,
    )

    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.04
    fig.tight_layout(pad=0.4)
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


def fig_acc_rate_trajectory(out_path: Path, run_id: str) -> None:
    """(E rate × accuracy) trajectories with pingness as colour.

    The accuracy–rate frontier from nb025 (Figure 3 of the manuscript) given a
    time axis: every condition climbs in accuracy as training proceeds, but the
    *route* differs. Trainable inits sprint rightward in E rate; the frozen
    control barely moves in E. Per-segment colour is the pingness at that
    epoch, on a shared viridis colormap: only the frozen trajectory lights up
    bright (high pingness) — every trainable one is dark (low pingness) end-to-end.

    Story: same accuracy, very different spike economy, and only one of them
    is actually doing PING.
    """
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"

    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    runs = _load_epoch_curves()
    by_cond: dict[str, list[dict]] = {}
    for d in runs.values():
        by_cond.setdefault(d["cond"], []).append(d)

    fig, ax = plt.subplots(figsize=(5.6, 3.15), dpi=200)

    # Basin divider in E rate (same boundary as the Figure 3 phase portrait):
    # left = the low-rate PING corner, right = the dense-E COBA attractor.
    DIVIDER = 17.0
    ax.axvspan(0, DIVIDER, color=theme.INK_BLACK, alpha=0.04, zorder=0)
    ax.axvline(DIVIDER, color=theme.MUTED, lw=0.8, ls="--", zorder=1)

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=1.0)

    e_max = 0.0
    final_pings: dict[str, float] = {}

    for cond in COND_ORDER:
        seeds = [
            d for d in by_cond.get(cond, [])
            if not any(v is None for v in d["rate_e"])
            and not any(v is None for v in d["contrast"])
            and not any(v is None for v in d["acc"])
        ]
        if not seeds:
            continue
        e_stack = np.array([np.asarray(d["rate_e"], dtype=float) for d in seeds])
        p_stack = np.array([np.asarray(d["contrast"], dtype=float) for d in seeds])
        a_stack = np.array([np.asarray(d["acc"], dtype=float) for d in seeds])
        e_mean = e_stack.mean(axis=0)
        p_mean = p_stack.mean(axis=0)
        a_mean = a_stack.mean(axis=0)
        final_pings[cond] = float(p_mean[-1])
        e_max = max(e_max, float(e_mean.max()))

        marker = COND_MARKERS.get(cond, "o")
        # Per-segment pingness colour: average of the segment's two endpoint pingnesses.
        points = np.array([e_mean, a_mean]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_pings = 0.5 * (p_mean[:-1] + p_mean[1:])
        lc = LineCollection(
            segments, cmap=cmap, norm=norm,
            array=seg_pings, linewidths=2.2, zorder=4,
        )
        ax.add_collection(lc)

        # Start (epoch 1) and end (final epoch) markers, coloured by their pingness.
        start_color = cmap(norm(p_mean[0]))
        end_color = cmap(norm(p_mean[-1]))
        ax.scatter(
            e_mean[0], a_mean[0],
            s=70, facecolor="white", edgecolor=start_color, marker=marker,
            linewidths=1.6, zorder=10,
        )
        label = f"{CONDITIONS[cond]['label']}  ·  final pingness {final_pings[cond]:.2f}"
        ax.scatter(
            e_mean[-1], a_mean[-1],
            s=85, color=end_color, marker=marker,
            edgecolor="white", linewidths=0.9, zorder=10, label=label,
        )

    # Colorbar for pingness — the third dimension.
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.045, aspect=22)
    cbar.set_label("pingness  (lobe–trough contrast)", fontsize=theme.SIZE_LABEL)
    cbar.ax.tick_params(labelsize=theme.SIZE_TICK)

    ax.set_xlabel("E firing rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax.set_title(
        "Same accuracy reached at very different spike economies — colour shows who still has the rhythm",
        fontsize=theme.SIZE_LABEL - 1, color=theme.INK,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(60, 100)
    ax.set_xlim(0, max(e_max * 1.08, 60.0))
    ax.tick_params(labelsize=theme.SIZE_TICK)
    # Basin labels in the empty top band, clear of the trajectories and legend.
    x_right = ax.get_xlim()[1]
    ax.text(DIVIDER * 0.5, 96, "PING basin", ha="center", va="center",
            fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED, zorder=2)
    ax.text(DIVIDER + (x_right - DIVIDER) * 0.5, 96, "COBA attractor",
            ha="center", va="center", fontsize=theme.SIZE_ANNOTATION,
            color=theme.DEEP_RED, zorder=2)
    ax.legend(
        fontsize=theme.SIZE_LEGEND - 1, frameon=False, loc="lower right",
        title="condition · end pingness", title_fontsize=theme.SIZE_LEGEND - 1,
    )
    ax.text(
        1.0, -0.16, "○ epoch 1   ● final epoch",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=theme.SIZE_CAPTION, color=theme.GREY_MID,
    )

    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["savefig.pad_inches"] = 0.04
    fig.tight_layout(pad=0.4)
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


def main() -> None:
    # Publication profile: every figure this notebook writes is a print-sized
    # vector, emitted as both SVG (docs) and PDF (manuscript) by save_figure.
    theme.set_paper_mode(True)

    if "--curves-only" in sys.argv:
        fig_training_curves(FIGURES / "training_curves", "nb049-curves")
        print(f"wrote {FIGURES / 'training_curves'}.{{svg,pdf}}")
        return

    if "--portrait-only" in sys.argv:
        fig_phase_portrait(FIGURES / "phase_portrait", "nb049-portrait")
        print(f"wrote {FIGURES / 'phase_portrait'}.{{svg,pdf}}")
        return

    if "--accrate-only" in sys.argv:
        fig_acc_rate_trajectory(FIGURES / "acc_rate_trajectory", "nb049-accrate")
        print(f"wrote {FIGURES / 'acc_rate_trajectory'}.{{svg,pdf}}")
        return

    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    notebook_run_id = next_run_id(SLUG)
    prepare_run_dirs(SLUG, notebook_run_id, wipe=False, make_artifacts=True)
    t_start = time.monotonic()

    only_missing = "--only-missing" in sys.argv
    skip_training = "--no-train" in sys.argv

    # Training lives in nb022 now (train-once / reuse-many): the init-variant
    # conditions are a registry family there. This notebook only consumes them.

    # ── Per-condition aggregation
    print("[aggregate] reading metrics from trained cells")
    metrics_by_cond: dict[str, list[dict]] = {}
    final_by_cond: dict[str, list[dict]] = {}
    rasters_by_cond: dict[str, dict] = {}
    summary_rows: list[dict] = []

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
            f = measure_trained_state(d)
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
                cell_dir(cond, SEEDS[0]),
            )

    # Per-condition diagnostic cards + matching weight-matrix cards.
    for cond in COND_ORDER:
        out = FIGURES / f"card__{cond}"
        plot_condition_card(
            cond,
            metrics_by_cond.get(cond, []),
            final_by_cond.get(cond, []),
            rasters_by_cond.get(cond),
            out, notebook_run_id,
        )
        print(f"wrote {out}.{{png,pdf}}")
        seed_to_dir = {
            s: cell_dir(cond, s) for s in SEEDS
            if (cell_dir(cond, s) / "weights.pth").exists()
        }
        if seed_to_dir:
            w_out = FIGURES / f"weights__{cond}"
            plot_weight_matrices(cond, seed_to_dir, w_out, notebook_run_id)
            print(f"wrote {w_out}.{{svg,pdf}}")


    fig_attractor(summary_rows, FIGURES / "attractor_ei", notebook_run_id)
    print(f"wrote {FIGURES / 'attractor_ei'}.{{svg,pdf}}")

    fig_training_curves(FIGURES / "training_curves", notebook_run_id)
    print(f"wrote {FIGURES / 'training_curves'}.{{svg,pdf}}")

    fig_phase_portrait(FIGURES / "phase_portrait", notebook_run_id)
    print(f"wrote {FIGURES / 'phase_portrait'}.{{svg,pdf}}")

    fig_acc_rate_trajectory(FIGURES / "acc_rate_trajectory", notebook_run_id)
    print(f"wrote {FIGURES / 'acc_rate_trajectory'}.{{svg,pdf}}")

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
