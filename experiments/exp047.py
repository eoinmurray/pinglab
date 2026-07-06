"""Experiment runner for entry exp047 — what sets the PING rate.

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

Writing: writings/exp047.typ · figures + numbers.json: artifacts/data/exp047/
Reaches the pinglab engine through the snn tool (tools/snn/tool.py), never
importing it — the tool forwards the flags to cli.py and stamps provenance.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
# Helpers + sibling runners live alongside this file under experiments/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import theme  # noqa: E402
from helpers.figsave import save_figure  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402

SLUG = "exp047"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)
SNN_TOOL = REPO / "tools" / "snn" / "tool.py"

# ── Architecture constants ──────────────────────────────────────────
N_E: int = 1024
N_IN: int = 784
T_MS: float = 200.0
DT: float = 0.1
N_STEPS: int = int(T_MS / DT)

# Untrained-PING init (matches exp023's biophysical-scale defaults so the
# loop self-sustains at init without any training).
W_IN_MEAN: float = 1.2
W_IN_STD: float = 0.12
W_IN_SPARSITY: float = 0.95
W_EI_MEAN: float = 1.0
W_EI_STD: float = 0.1

INPUT_RATE_HZ: float = 25.0  # uniform Poisson, matches exp025's baseline
SEED: int = 42

# The per-synapse I->E weight is the control variable. Its biophysical
# default (the 1:4-init value) is W_IE_DEFAULT; the figure marks it but
# does not anchor anything to it.
W_IE_DEFAULT: float = 2.0
W_IE_REL_STD: float = 0.1  # per-synapse spread held at 10% of the mean

# Independent sweeps: per-synapse weight (x-axis) and pool size (lines).
W_IE_VALUES: list[float] = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
N_I_SWEEP: list[int] = [16, 64, 256]

# Baked "small"-tier run scale (retired --tier system): batch of 4 trials.
N_BATCH: int = 4

# Run scale — stamped into the manifest by run_dirs.prepare and rendered as
# the Methods table via RunScale; the mdx never restates these numbers.
SCALE: dict = {
    "input": "synthetic-spikes",
    "max_samples": N_BATCH,
    "t_ms": T_MS,
    "dt_ms": DT,
    "input_rate_hz": INPUT_RATE_HZ,
    "hidden": N_E,
    "batch_size": N_BATCH,
    "seeds": 1,
    "cells": len(W_IE_VALUES) * len(N_I_SWEEP),
    "grid": f"{len(W_IE_VALUES)} W_ie × {len(N_I_SWEEP)} N_I",
}


# ── Sweep ───────────────────────────────────────────────────────────
def measure_one(n_inh: int, w_ie_mean: float, n_batch: int) -> dict:
    """One (N_I, W^IE) point via the snn tool: untrained PING driven by uniform
    Poisson input, returning population E/I rates and the pop-weighted total.

    Shells out to `tools/snn/tool.py sim`, which forwards to `cli.py sim` (which
    builds the untrained net and runs it) — the experiment only runs it and reads
    metrics.json back. The recurrent means map to ei_strength=1 (→ w_ei=(1, 0.1))
    and ei_ratio=W^IE (→ w_ie=(W^IE, 0.1·W^IE)).
    """
    out_dir = (ARTIFACTS / "probe" / f"nI{int(n_inh)}_wie{w_ie_mean:g}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "uv", "run", "python", str(SNN_TOOL), "sim",
            "--input", "synthetic-spikes",
            "--model", "ping",
            "--n-hidden", str(N_E),
            "--n-in", str(N_IN),
            "--n-inh", str(int(n_inh)),
            "--ei-strength", "1",
            "--ei-ratio", str(w_ie_mean),
            "--w-in", str(W_IN_MEAN),
            "--w-in-sparsity", str(W_IN_SPARSITY),
            "--input-rate", str(INPUT_RATE_HZ),
            "--n-batch", str(int(n_batch)),
            "--t-ms", str(T_MS),
            "--dt", str(DT),
            "--seed", str(SEED),
            "--out-dir", str(out_dir),
        ],
        cwd=REPO,
        check=True,
    )
    m = json.loads((out_dir / "metrics.json").read_text())
    r_e = float(m["rate_e_hz"])
    r_i = float(m["rate_i_hz"])
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
    fig, (ax_e, ax_i) = plt.subplots(1, 2, figsize=(5.6, 2.41))
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
    ax_e.legend(fontsize=theme.SIZE_LEGEND - 1, frameon=False, loc="upper right")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────
def main() -> None:
    n_batch = N_BATCH
    wipe_dir = "--no-wipe-dir" not in sys.argv

    # Publication profile: every figure this notebook writes is a print-sized
    # vector, emitted as both SVG (docs) and PDF (manuscript) by save_figure.
    theme.set_paper_mode(True)

    notebook_run_id = next_run_id(SLUG)
    prepare_run_dirs(SLUG, notebook_run_id, wipe=wipe_dir, make_artifacts=True,
                     scale=SCALE, host="local")

    t_start = time.monotonic()
    print(f"[w_ie x n_inh sweep] N_E={N_E}  input={INPUT_RATE_HZ:g} Hz  "
          f"batch={n_batch}")
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

    summary_out = FIGURES / "rate_vs_w_ie"
    plot_summary(rows_by_ni, summary_out, notebook_run_id)
    print(f"wrote {summary_out}.{{svg,pdf}}")

    duration_s = time.monotonic() - t_start
    summary = {
        "notebook_run_id": notebook_run_id,
        "duration_s": round(duration_s, 1),
        "config": {
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
