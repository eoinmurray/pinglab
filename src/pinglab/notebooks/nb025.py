"""Notebook runner for entry 025 — coba / ping rasters with and
without the slow synaptic channel.

Takes the trained nb024 baselines (coba__off__seed42, ping__off__seed42)
and replays each on one MNIST trial twice: once at slow-syn gain 0.0
(parity with nb024) and once at slow-syn gain 0.5. Same weights, same
input, same trial — only the slow-NMDA channel toggles.

The slow channel decays with tau_nmda (default 100 ms), an order of
magnitude longer than the existing AMPA (2 ms) and GABA (9 ms). Sample-
period spikes leave a residue that's still readable long after the
input cuts out — the substrate working-memory tasks need.

Notebook entry: src/docs/src/pages/notebooks/nb025.mdx
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

SLUG = "nb025"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG

# Pretrained baselines from nb024 (no slow-syn at training time).
NB024_ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / "nb024"
PRETRAINED = {
    "coba": NB024_ARTIFACTS / "coba__off__seed42",
    "ping": NB024_ARTIFACTS / "ping__off__seed42",
}

# ── Recipe (CLI accepts only --tier and --modal-gpu) ──────────────────
T_MS_TRIAL = 400.0
SAMPLE_IDX = 0
SLOW_SYN_GAIN = 0.5
TAU_NMDA_MS = 100.0
SEED = 42
RASTER_N_E_PLOT = 200
RASTER_N_I_PLOT = 64

DEFAULT_TIER = "small"
TIER_CONFIG = {
    "extra small": dict(max_samples=200),
    "small":       dict(max_samples=500),
    "medium":      dict(max_samples=2000),
    "large":       dict(max_samples=5000),
    "extra large": dict(max_samples=10000),
}
MODELS = ["coba", "ping"]


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


def capture_raster(train_dir: Path, slow_syn_gain: float) -> dict:
    """Forward pass on one MNIST trial with the chosen slow-syn gain.

    Loads the nb024-trained weights, rebuilds the net with the slow
    channel enabled at the given gain (0.0 = behaviour identical to
    nb024 baseline), runs 400 ms forward with recording on, returns
    E + I rasters.
    """
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
    M.tau_nmda = TAU_NMDA_MS
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
    # Build with slow_synapse=True so the net allocates the extra slow
    # conductance buffer; gain=0 keeps behaviour identical to the
    # nb024-trained baseline.
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
        slow_synapse=True,
        slow_syn_gain=slow_syn_gain,
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
        "slow_syn_gain": slow_syn_gain,
        "label": y_b,
        "e": e_full,
        "i": i_full,
        "dt": float(cfg["dt"]),
        "t_total_ms": T_MS_TRIAL,
        "e_rate_hz": e_rate,
        "i_rate_hz": i_rate,
    }


def plot_slow_syn_rasters(
    samples: dict[str, dict[float, dict]],
    out_path: Path,
    run_id: str,
) -> None:
    """2 cols (coba / ping) × 2 rows (slow-syn off / on) — same weights,
    same input, only the slow channel toggles.

    Layout:
      • Model name (coba / ping) as column header (top row only).
      • Condition (slow-syn off vs gain 0.5) as a row label on the left.
      • E / I rate per panel as right-side annotation.
    """
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
    gains_in_order = [0.0, SLOW_SYN_GAIN]
    row_labels = [
        "slow-syn off",
        f"slow-syn gain {SLOW_SYN_GAIN:g}\n(τ = {TAU_NMDA_MS:.0f} ms)",
    ]

    for col, model in enumerate(models):
        for row, gain in enumerate(gains_in_order):
            ax = axes[row, col]
            s = samples[model][gain]
            T = s["e"].shape[0]
            e_full = s["e"][:, 0, :] if s["e"].ndim == 3 else s["e"]
            i_full = s["i"][:, 0, :] if s["i"].ndim == 3 else s["i"]
            e_idx = np.sort(rng.choice(e_full.shape[1], n_e, replace=False))
            i_idx = np.sort(rng.choice(i_full.shape[1], n_i, replace=False))
            t_axis = np.arange(T) * s["dt"]
            e_t, e_n = np.where(e_full[:, e_idx].astype(bool))
            i_t, i_n = np.where(i_full[:, i_idx].astype(bool))
            ax.scatter(
                t_axis[e_t], e_n,
                s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.4,
            )
            ax.scatter(
                t_axis[i_t], i_n + n_e + gap,
                s=2.0, c=theme.DEEP_RED, marker="|", linewidths=0.4,
            )
            ax.set_ylim(-2, n_e + n_i + gap + 2)
            if col == 0:
                ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
                ax.set_yticklabels(["E", "I"])
                ax.tick_params(axis="y", length=0)
            else:
                ax.yaxis.set_visible(False)
            ax.set_xlim(0, s["t_total_ms"])
            # In-panel rate annotation (top-right inside the axes so it
            # doesn't compete with the panel next door for gutter space).
            ax.text(
                0.985, 0.97,
                f"E = {s['e_rate_hz']:.1f} Hz   I = {s['i_rate_hz']:.1f} Hz",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=theme.SIZE_LABEL,
                bbox={
                    "facecolor": theme.PAPER, "alpha": 0.85,
                    "edgecolor": "none", "pad": 2,
                },
            )
            # Column title (top row only).
            if row == 0:
                ax.set_title(model, fontsize=theme.SIZE_TITLE, pad=6)
            # Row label (left column only).
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


def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    wipe_dir = "--no-wipe-dir" not in sys.argv

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id} tier={tier}")
    if modal_gpu is not None:
        print(f"[stub] --modal-gpu {modal_gpu} accepted, no-op for this cell")

    if wipe_dir:
        for d in (ARTIFACTS, FIGURES):
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)

    missing = [m for m, d in PRETRAINED.items() if not (d / "weights.pth").exists()]
    if missing:
        raise SystemExit(
            f"pretrained baselines missing for {missing}; run nb024 first"
        )

    print(
        f"[slow-syn-rasters] coba + ping at gains {{0.0, {SLOW_SYN_GAIN}}}, "
        f"τ_nmda = {TAU_NMDA_MS} ms, trial = {T_MS_TRIAL:.0f} ms"
    )
    samples: dict[str, dict[float, dict]] = {}
    summary_rows = []
    for model in MODELS:
        samples[model] = {}
        for gain in (0.0, SLOW_SYN_GAIN):
            s = capture_raster(PRETRAINED[model], gain)
            samples[model][gain] = s
            print(
                f"  {model:<4}  gain={gain:.2f}  "
                f"E={s['e_rate_hz']:6.2f} Hz  I={s['i_rate_hz']:6.2f} Hz"
            )
            summary_rows.append(
                {
                    "model": model,
                    "slow_syn_gain": gain,
                    "e_rate_hz": s["e_rate_hz"],
                    "i_rate_hz": s["i_rate_hz"],
                }
            )
    plot_slow_syn_rasters(
        samples, FIGURES / "slow_syn_rasters.png", notebook_run_id
    )
    print(f"wrote {FIGURES / 'slow_syn_rasters.png'}")

    duration_s = time.monotonic() - t_start
    summary = {
        "notebook_run_id": notebook_run_id,
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "tier": tier,
        "config": {
            "tier": tier,
            "models": MODELS,
            "slow_syn_gain": SLOW_SYN_GAIN,
            "tau_nmda_ms": TAU_NMDA_MS,
            "t_total_ms": T_MS_TRIAL,
            "sample_idx": SAMPLE_IDX,
            "seed": SEED,
        },
        "results": summary_rows,
        "success_criteria": [
            {
                "label": "slow-syn raster rendered",
                "passed": (FIGURES / "slow_syn_rasters.png").exists(),
                "detail": (
                    f"coba E gain0={samples['coba'][0.0]['e_rate_hz']:.1f} Hz; "
                    f"coba E gain{SLOW_SYN_GAIN}="
                    f"{samples['coba'][SLOW_SYN_GAIN]['e_rate_hz']:.1f} Hz; "
                    f"ping E gain0={samples['ping'][0.0]['e_rate_hz']:.1f} Hz; "
                    f"ping E gain{SLOW_SYN_GAIN}="
                    f"{samples['ping'][SLOW_SYN_GAIN]['e_rate_hz']:.1f} Hz"
                ),
            },
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {summary['duration']}")


if __name__ == "__main__":
    main()
