"""Notebook runner for entry 047 — what sets the PING rate.

Independently sweep the per-synapse I->E weight W^IE and the inhibitory
pool size N_I on an untrained PING network, and measure per-cell E and I
rates. The claim is anchor-free: the rate is a function of the per-event
shunt W^IE (relative to the fixed release level g_i*), NOT of N_I. So
when r is plotted against W^IE the curves for different N_I collapse onto
one another — pool size does not move the rate, the per-synapse weight
does. The three "scalings" (constant per-synapse, synaptic 1/N, critical
1/sqrt(N)) are then just rules for choosing W^IE given N_I, i.e. paths
across this single master curve; they need an arbitrary anchor and are
not plotted.

Inference only — no training. Untrained PING with the canonical
biophysical init at each (N_I, W^IE).

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

from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402
from cli import theme  # noqa: E402

SLUG = "nb047"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

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

INPUT_RATE_HZ: float = 25.0  # uniform Poisson, matches nb025's baseline
SEED: int = 42

# The per-synapse I->E weight is the control variable. Its biophysical
# default (the 1:4-init value) is W_IE_DEFAULT; the figure marks it but
# does not anchor anything to it.
W_IE_DEFAULT: float = 2.0
W_IE_REL_STD: float = 0.1  # per-synapse spread held at 10% of the mean

# Independent sweeps: per-synapse weight (x-axis) and pool size (lines).
W_IE_VALUES: list[float] = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
N_I_SWEEP: list[int] = [16, 64, 256]

TIER_CONFIG: dict[str, dict] = {
    "extra small": dict(n_batch=1),
    "small": dict(n_batch=4),
    "medium": dict(n_batch=16),
    "large": dict(n_batch=64),
    "extra large": dict(n_batch=256),
}
DEFAULT_TIER: str = "small"


# ── Net build ───────────────────────────────────────────────────────
def _build_untrained_ping(n_inh: int, w_ie_mean: float):
    """Build an untrained PING net at the given pool size and per-synapse
    I->E weight."""
    torch.manual_seed(SEED)
    import models as M
    from cli.config import build_net

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
        w_ie=(w_ie_mean, W_IE_REL_STD * w_ie_mean),
        hidden_sizes=[N_E],
        n_inh_per_layer={1: int(n_inh)},
    )


# ── Sweep ───────────────────────────────────────────────────────────
def measure_one(n_inh: int, w_ie_mean: float, n_batch: int) -> dict:
    """One forward pass at a given (N_I, W^IE). Returns per-cell E and I
    rates and the population-weighted total."""
    net = _build_untrained_ping(n_inh, w_ie_mean)
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

    return {
        "n_inh": int(n_inh),
        "n_e": int(N_E),
        "w_ie": float(w_ie_mean),
        "r_e_hz": r_e,
        "r_i_hz": r_i,
        "r_total_hz": r_total,
    }


# ── Plotting ────────────────────────────────────────────────────────
def plot_summary(rows_by_ni: dict, out_path: Path, run_id: str) -> None:
    """One summary figure: E (left) and I (right) per-cell rate vs the
    per-synapse weight W^IE, one line per pool size N_I. The lines
    collapse onto one another — the rate is set by W^IE, not N_I."""
    theme.apply()
    fig, (ax_e, ax_i) = plt.subplots(1, 2, figsize=(10.0, 4.3), dpi=150)
    palette = [theme.DEEP_RED, theme.AMBER, theme.INK_BLACK]
    markers = ["s", "^", "o"]
    n_is = sorted(rows_by_ni.keys())
    for ax, key in ((ax_e, "r_e_hz"), (ax_i, "r_i_hz")):
        for n_inh, col, mk in zip(n_is, palette, markers):
            rows = sorted(rows_by_ni[n_inh], key=lambda r: r["w_ie"])
            xs = [r["w_ie"] for r in rows]
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, mk + "-", color=col, lw=1.6, ms=6,
                    label=f"$N_I = {n_inh}$")
        ax.axvline(W_IE_DEFAULT, ls=":", color=theme.GREY_MID, lw=0.8)
        ax.set_xlabel("per-synapse $W^{IE}$  (μS)", fontsize=theme.SIZE_LABEL)
        ax.set_xlim(0, 16.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax_e.set_ylabel("E rate  (Hz / cell)", fontsize=theme.SIZE_LABEL)
    ax_i.set_ylabel("I rate  (Hz / cell)", fontsize=theme.SIZE_LABEL)
    ax_e.set_ylim(bottom=0)
    ax_i.set_ylim(bottom=0)
    ax_e.text(W_IE_DEFAULT + 0.4, ax_e.get_ylim()[1] * 0.06, "default",
              fontsize=theme.SIZE_ANNOTATION - 1, color=theme.GREY_DARK,
              ha="left", va="bottom")
    ax_e.annotate(
        "all $N_I$ overlap:\nrate set by $W^{IE}$, not pool size",
        xy=(8.0, [r["r_e_hz"] for r in
                  sorted(rows_by_ni[n_is[-1]], key=lambda r: r["w_ie"])][4]),
        xytext=(6.0, ax_e.get_ylim()[1] * 0.78),
        fontsize=theme.SIZE_ANNOTATION - 1, color=theme.GREY_DARK,
        ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=theme.GREY_MID, lw=0.8),
    )
    ax_e.legend(fontsize=theme.SIZE_LEGEND - 1, frameon=False, loc="upper right")
    fig.suptitle("Rate is set by the per-synapse weight, not the pool size",
                 fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────
def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    cfg = TIER_CONFIG[tier]
    n_batch = int(cfg["n_batch"])
    wipe_dir = "--no-wipe-dir" not in sys.argv
    notebook_run_id = next_run_id(SLUG)
    prepare_run_dirs(SLUG, notebook_run_id, wipe=wipe_dir, make_artifacts=True)

    t_start = time.monotonic()
    print(f"[w_ie x n_inh sweep] N_E={N_E}  input={INPUT_RATE_HZ:g} Hz  "
          f"batch={n_batch}  tier={tier}")
    rows_by_ni: dict[int, list[dict]] = {}
    for n_inh in N_I_SWEEP:
        print(f"--- N_I = {n_inh} ---")
        ni_rows: list[dict] = []
        for w_ie in W_IE_VALUES:
            r = measure_one(n_inh, w_ie, n_batch)
            ni_rows.append(r)
            print(f"  W^IE={r['w_ie']:>5.2f} μS  "
                  f"E={r['r_e_hz']:6.2f} Hz  I={r['r_i_hz']:6.2f} Hz")
        rows_by_ni[n_inh] = ni_rows

    summary_out = FIGURES / "rate_vs_w_ie.png"
    plot_summary(rows_by_ni, summary_out, notebook_run_id)
    print(f"wrote {summary_out}")

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
            "w_ie_values": W_IE_VALUES,
            "n_i_sweep": N_I_SWEEP,
            "w_ie_default": W_IE_DEFAULT,
        },
        "rows_by_n_i": {str(k): v for k, v in rows_by_ni.items()},
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  duration: {duration_s:.1f}s")


if __name__ == "__main__":
    main()
