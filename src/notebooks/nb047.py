"""Notebook runner for entry 047 — I-pool size sweep on untrained PING.

For a fixed N_E and a fixed uniform Poisson input rate, sweep N_I
across a range and measure per-cell E rate, per-cell I rate, and
population-weighted total rate
    r_tot = (N_E · r_E + N_I · r_I) / (N_E + N_I).
Inference only — no training. Untrained PING with the canonical
biophysical W_EI, W_IE init at each N_I.

The output is one plot: E (black), I (red), total (amber dashed)
firing rate vs N_I on a log x-axis.

Notebook entry: src/docs/src/pages/notebooks/nb047.mdx
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402
import theme  # noqa: E402

SLUG = "nb047"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG

# ── Architecture constants ──────────────────────────────────────────
N_E: int = 1024
N_IN: int = 784
T_MS: float = 200.0
DT: float = 0.1
N_STEPS: int = int(T_MS / DT)

# Untrained-PING init (matches nb023's biophysical-scale defaults so the
# loop self-sustains at init without any training).
W_IN_MEAN: float = 1.2
W_IN_STD: float = 0.12
W_IN_SPARSITY: float = 0.95
W_EI_MEAN: float = 1.0
W_EI_STD: float = 0.1
W_IE_MEAN: float = 2.0
W_IE_STD: float = 0.2

INPUT_RATE_HZ: float = 25.0  # uniform Poisson, matches nb025's baseline
SEED: int = 42

RASTER_N_E_PLOT: int = 200  # E cells subsampled for raster strip
RASTER_T_WINDOW_MS: float = 200.0  # full trial window

# Reference I-pool size used to anchor the normalizations. At
# N_I = N_I_REF every regime gives the same W^IE; the regimes only
# differ for N_I ≠ N_I_REF.
N_I_REF: int = 256  # canonical 1:4 init


def w_ie_scale(regime: str, n_inh: int) -> float:
    """Per-edge W^IE multiplier vs the canonical biophysical value."""
    if regime == "constant":
        return 1.0
    if regime == "synaptic":
        # W^IE per edge ∝ 1/N_I → summed I→E drive into each E cell is
        # N_I-invariant by construction.
        return N_I_REF / float(n_inh) if n_inh > 0 else 1.0
    if regime == "critical":
        # W^IE per edge ∝ 1/√N_I → variance of summed input is
        # N_I-invariant (mean still grows as √N_I).
        import math
        return math.sqrt(N_I_REF / float(n_inh)) if n_inh > 0 else 1.0
    raise ValueError(f"unknown regime {regime!r}")


REGIMES: list[str] = ["constant", "synaptic", "critical"]
REGIME_LABEL: dict[str, str] = {
    "constant": "Constant per-edge (W^IE fixed)",
    "synaptic": "Synaptic normalization (W^IE ∝ 1/N_I)",
    "critical": "Critical-balance (W^IE ∝ 1/√N_I)",
}

# I-pool sizes to sweep: 0%, 5%, 10%, 15%, 20%, 25% of N_E.
N_I_PCT: list[float] = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
N_I_VALUES: list[int] = [max(1, int(round(p * 1024))) for p in N_I_PCT]

TIER_CONFIG: dict[str, dict] = {
    "extra small": dict(n_batch=1),
    "small": dict(n_batch=4),
    "medium": dict(n_batch=16),
    "large": dict(n_batch=64),
    "extra large": dict(n_batch=256),
}
DEFAULT_TIER: str = "small"


# ── Net build ───────────────────────────────────────────────────────
def _build_untrained_ping(n_inh: int):
    """Build an untrained PING net at the given inhibitory pool size."""
    torch.manual_seed(SEED)
    import models as M
    from config import build_net

    M.HIDDEN_SIZES = [N_E]
    M.N_HID = N_E
    M.N_INH = int(n_inh)
    M.N_IN = N_IN
    M.T_steps = N_STEPS
    M.T_ms = T_MS
    M.dt = DT

    return build_net(
        "ping",
        w_in=(W_IN_MEAN, W_IN_STD),
        w_in_sparsity=W_IN_SPARSITY,
        w_ei=(W_EI_MEAN, W_EI_STD),
        w_ie=(W_IE_MEAN, W_IE_STD),
        hidden_sizes=[N_E],
        n_inh_per_layer={1: int(n_inh)},
    )


# ── Sweep ───────────────────────────────────────────────────────────
def measure_one(n_inh: int, n_batch: int, regime: str = "constant") -> dict:
    """One forward pass at a given N_I and normalization regime. Returns
    per-cell E rate, per-cell I rate, and a single-trial raster sample."""
    net = _build_untrained_ping(n_inh)
    # Apply per-regime W^IE rescaling. constant: x1. synaptic: x N_REF/N_I.
    # critical: x √(N_REF/N_I).
    scale = w_ie_scale(regime, n_inh)
    if scale != 1.0:
        with torch.no_grad():
            for k, w in net.W_ie.items():
                w.mul_(scale)
    net.eval()
    net.recording = True

    gen = torch.Generator().manual_seed(SEED + 1)
    p_step = INPUT_RATE_HZ * DT / 1000.0
    spk_in = (torch.rand(N_STEPS, n_batch, N_IN, generator=gen) < p_step).float()

    with torch.no_grad():
        net(input_spikes=spk_in)

    rec = net.spike_record
    spk_e = rec["hid"]  # (T, B, N_E)
    spk_i = rec["inh"]  # (T, B, N_I)
    t_sec = T_MS / 1000.0
    r_e = float(spk_e.sum().item()) / (n_batch * N_E * t_sec)
    r_i = float(spk_i.sum().item()) / (n_batch * n_inh * t_sec) if n_inh > 0 else 0.0
    r_total = (N_E * r_e + n_inh * r_i) / (N_E + n_inh)

    # Per-cell raster from trial 0, subsample E cells to a fixed plotting
    # count so panels look comparable across N_I.
    e_full = spk_e[:, 0, :].cpu().numpy().astype(bool)  # (T, N_E)
    i_full = spk_i[:, 0, :].cpu().numpy().astype(bool)  # (T, N_I)
    rng = np.random.default_rng(SEED)
    n_e_plot = min(RASTER_N_E_PLOT, e_full.shape[1])
    e_idx = np.sort(rng.choice(e_full.shape[1], n_e_plot, replace=False))
    e_raster = e_full[:, e_idx]
    i_raster = i_full  # plot all I cells (varies per panel)

    return {
        "n_inh": int(n_inh),
        "n_e": int(N_E),
        "regime": regime,
        "w_ie_scale": float(scale),
        "ei_ratio": N_E / float(n_inh) if n_inh else float("inf"),
        "r_e_hz": r_e,
        "r_i_hz": r_i,
        "r_total_hz": r_total,
        "raster_e": e_raster,
        "raster_i": i_raster,
    }


# ── Plotting ────────────────────────────────────────────────────────
def _stamp(fig, run_id: str) -> None:
    fig.text(
        0.995, 0.005, run_id,
        ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


def plot_rate_vs_n_inh(
    rows: list[dict], out_path: Path, run_id: str,
    regime: str = "constant",
) -> None:
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    xs = [100.0 * r["n_inh"] / N_E for r in rows]
    r_e = [r["r_e_hz"] for r in rows]
    r_i = [r["r_i_hz"] for r in rows]
    ax.plot(xs, r_e, marker="o", color=theme.INK_BLACK, lw=1.5,
            label="E per-cell rate")
    ax.plot(xs, r_i, marker="s", color=theme.DEEP_RED, lw=1.5,
            label="I per-cell rate")
    ax.set_xlabel("$N_I$ (% of $N_E$)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Firing rate (Hz / cell)", fontsize=theme.SIZE_LABEL)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    fig.suptitle(
        f"Untrained PING — {REGIME_LABEL[regime]}\n"
        f"firing rates vs $N_I$ ({INPUT_RATE_HZ:g} Hz Poisson, $N_E$ = {N_E})",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_raster_strip(
    rows: list[dict], out_path: Path, run_id: str,
    regime: str = "constant",
) -> None:
    """Stacked single-trial rasters, one per N_I level.

    E cells (subsampled to RASTER_N_E_PLOT) plotted in black on the lower
    band; I cells (all of them — count varies per panel) plotted in red on
    the upper band. The label panel reports N_I (as % of N_E) and rates.
    """
    theme.apply()
    rows = sorted(rows, key=lambda r: r["n_inh"])
    n = len(rows)
    n_e_plot = min(RASTER_N_E_PLOT, N_E)
    fig, axes = plt.subplots(
        n, 1, figsize=(10.0, 1.0 * n + 1.0),
        sharex=True, gridspec_kw={"hspace": 0.22},
    )
    if n == 1:
        axes = [axes]
    # Maximum I cells we'll display, reserved at the top of every panel
    # so that all rows have the same y-axis extent and the E band always
    # occupies the same visual fraction. At canonical 25% (1:4 ratio) we
    # plot n_e_plot/4 I cells; smaller N_I just under-fills that band.
    n_i_band = max(1, int(round(n_e_plot * 0.25)))
    gap = 6
    y_max = n_e_plot + gap + n_i_band

    for i, (ax, r) in enumerate(zip(axes, rows)):
        e_raster = r["raster_e"]  # (T, n_e_plot)
        i_raster_full = r["raster_i"]  # (T, n_inh)
        T = e_raster.shape[0]
        t_axis = np.arange(T) * DT
        mask = t_axis <= RASTER_T_WINDOW_MS
        n_inh = i_raster_full.shape[1]
        # Subsample so visual band height = true N_I/N_E fraction of the
        # E band. n_i_plot is at most n_i_band; smaller N_I → sparser
        # display within the same reserved I band.
        n_i_plot = max(1, min(n_inh, int(round(n_e_plot * n_inh / N_E))))
        if n_i_plot < n_inh:
            rng = np.random.default_rng(SEED + r["n_inh"])
            i_idx = np.sort(rng.choice(n_inh, n_i_plot, replace=False))
            i_raster = i_raster_full[:, i_idx]
        else:
            i_raster = i_raster_full
        e_t, e_n = np.where(e_raster[mask])
        i_t, i_n = np.where(i_raster[mask])
        ax.scatter(t_axis[mask][e_t], e_n,
                   s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.4)
        ax.scatter(t_axis[mask][i_t], i_n + n_e_plot + gap,
                   s=2.0, c=theme.DEEP_RED, marker="|", linewidths=0.4)
        ax.set_ylim(-2, y_max + 2)
        ax.set_yticks([n_e_plot / 2, n_e_plot + gap + n_i_band / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, RASTER_T_WINDOW_MS)
        pct = 100.0 * r["n_inh"] / N_E
        ax.text(
            1.012, 0.5,
            f"$N_I$ = {r['n_inh']} ({pct:.0f}%)\n"
            f"E = {r['r_e_hz']:.1f} Hz\n"
            f"I = {r['r_i_hz']:.1f} Hz",
            transform=ax.transAxes,
            ha="left", va="center",
            fontsize=theme.SIZE_LABEL,
        )
        if i == 0:
            ax.set_title(
                f"Untrained PING rasters — {REGIME_LABEL[regime]}\n"
                f"(uniform {INPUT_RATE_HZ:g} Hz Poisson input, $N_E$ = {N_E})",
                fontsize=theme.SIZE_TITLE,
            )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    fig.tight_layout()
    _stamp(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────
def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    cfg = TIER_CONFIG[tier]
    n_batch = int(cfg["n_batch"])
    notebook_run_id = next_run_id(SLUG)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)

    t_start = time.monotonic()
    print(f"[n_inh-sweep] N_E={N_E}  input={INPUT_RATE_HZ:g} Hz  "
          f"batch={n_batch}  tier={tier}")
    rows_by_regime: dict[str, list[dict]] = {}
    for regime in REGIMES:
        print(f"--- regime: {regime} ---")
        regime_rows: list[dict] = []
        for n_inh in N_I_VALUES:
            r = measure_one(n_inh, n_batch, regime=regime)
            regime_rows.append(r)
            print(
                f"  N_I={r['n_inh']:>4}  scale={r['w_ie_scale']:>6.3f}  "
                f"E={r['r_e_hz']:6.2f} Hz  I={r['r_i_hz']:6.2f} Hz"
            )
        rows_by_regime[regime] = regime_rows
        # File suffix: leave the "constant" outputs unsuffixed so existing
        # mdx links to rate_vs_n_inh.png / raster_strip.png keep working.
        suffix = "" if regime == "constant" else f"__{regime}"
        rate_out = FIGURES / f"rate_vs_n_inh{suffix}.png"
        raster_out = FIGURES / f"raster_strip{suffix}.png"
        plot_rate_vs_n_inh(regime_rows, rate_out, notebook_run_id, regime=regime)
        plot_raster_strip(regime_rows, raster_out, notebook_run_id, regime=regime)
        print(f"wrote {rate_out}")
        print(f"wrote {raster_out}")

    duration_s = time.monotonic() - t_start
    summary = {
        "notebook_run_id": notebook_run_id,
        "duration_s": round(duration_s, 1),
        "config": {
            "tier": tier,
            "n_e": N_E,
            "n_in": N_IN,
            "t_ms": T_MS,
            "dt": DT,
            "input_rate_hz": INPUT_RATE_HZ,
            "n_batch": n_batch,
            "seed": SEED,
            "n_i_values": N_I_VALUES,
        },
        "rows_by_regime": {
            regime: [
                {k: v for k, v in r.items() if k not in ("raster_e", "raster_i")}
                for r in rows
            ]
            for regime, rows in rows_by_regime.items()
        },
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  duration: {duration_s:.1f}s")


if __name__ == "__main__":
    main()
