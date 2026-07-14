"""Experiment 047 — disentangle inhibitory pool size from I→E coupling.

The engine fan-in-normalises recurrent matrices, so the nominal I→E parameter
G_IE is the expected summed weight onto one E cell, while the realised mean
synaptic weight is j_IE = G_IE / N_I.  This experiment varies N_I under two
controls in one plotset:

1. fixed G_IE (the engine's default convention; j_IE falls as 1/N_I), and
2. fixed j_IE (G_IE grows as N_I).

Inference only.  The runner uses the public snn CLI and does not modify it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import theme  # noqa: E402
from helpers.cli import parse_meta  # noqa: E402
from helpers.figsave import save_figure  # noqa: E402
from helpers.numbers import write_numbers  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_cli import run_cli  # noqa: E402
from helpers.run_dirs import published_run  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import stamp_figure  # noqa: E402

SLUG = "exp047"
ARTIFACTS, FIGURES = artifacts_and_figures(SLUG)

N_E = 1024
N_IN = 784
N_I_SWEEP = [16, 64, 256]
N_I_REFERENCE = 256
T_MS = 500.0
DT = 0.1
N_BATCH = 8
SEEDS = [40, 41, 42]

W_IN_MEAN = 1.2
W_IN_SPARSITY = 0.95
W_EI_TOTAL = 1.0
INPUT_RATE_HZ = 25.0

# At the reference pool these are both nominal total couplings G_IE and,
# divided by N_I_REFERENCE, the realised per-synapse strengths j_IE.
REFERENCE_G_IE = [1.0, 2.0, 4.0]
REFERENCE_J_IE = [g / N_I_REFERENCE for g in REFERENCE_G_IE]

SCALE = {
    "input": "synthetic-spikes",
    "max_samples": N_BATCH,
    "t_ms": T_MS,
    "dt_ms": DT,
    "input_rate_hz": INPUT_RATE_HZ,
    "hidden": N_E,
    "batch_size": N_BATCH,
    "seeds": len(SEEDS),
    "cells": len(N_I_SWEEP) * len(REFERENCE_G_IE) * 2,
    "grid": "fixed total coupling + fixed realised synapse",
}


def measure_one(n_inh: int, g_ie: float, seed: int) -> dict:
    """Measure one network; G_IE is the pre-normalisation CLI parameter."""
    tag = f"nI{n_inh}_g{g_ie:.8g}_s{seed}"
    out_dir = (ARTIFACTS / "probe" / tag).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cli([
        "sim",
        "--input", "synthetic-spikes",
        "--model", "ping",
        "--n-hidden", str(N_E),
        "--n-in", str(N_IN),
        "--n-inh", str(n_inh),
        "--ei-strength", str(W_EI_TOTAL),
        "--ei-ratio", str(g_ie / W_EI_TOTAL),
        "--w-in", str(W_IN_MEAN),
        "--w-in-sparsity", str(W_IN_SPARSITY),
        "--input-rate", str(INPUT_RATE_HZ),
        "--n-batch", str(N_BATCH),
        "--t-ms", str(T_MS),
        "--dt", str(DT),
        "--seed", str(seed),
        "--out-dir", str(out_dir),
    ])
    metrics = json.loads((out_dir / "metrics.json").read_text())
    return {
        "n_i": n_inh,
        "g_ie_total": g_ie,
        "j_ie_synapse": g_ie / n_inh,
        "seed": seed,
        "r_e_hz": float(metrics["rate_e_hz"]),
        "r_i_hz": float(metrics["rate_i_hz"]),
    }


def summarise(rows: list[dict]) -> dict:
    result = {}
    for key in ("r_e_hz", "r_i_hz"):
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_sd"] = float(values.std(ddof=1))
    return result


def plot_controls(summary: dict, out_path: Path, run_id: str) -> None:
    theme.apply()
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.0), sharex=True)
    colors = [theme.GREY_MID, theme.DEEP_RED, theme.INK_BLACK]
    markers = ["s", "^", "o"]

    controls = [
        ("fixed_total", "(a) Fixed summed coupling $G_{IE}$", REFERENCE_G_IE),
        ("fixed_synapse", "(b) Fixed realised synapse $j_{IE}$", REFERENCE_J_IE),
    ]
    for col, (control, title, levels) in enumerate(controls):
        axes[0, col].set_title(title, fontsize=theme.SIZE_TITLE)
        for level, color, marker in zip(levels, colors, markers):
            cells = [summary[control][f"{level:.12g}"][str(n)] for n in N_I_SWEEP]
            if control == "fixed_total":
                label = f"$G_{{IE}}={level:g}$ μS"
            else:
                label = f"$j_{{IE}}={level * 1000:.2f}$ nS"
            for row, metric in enumerate(("r_e_hz", "r_i_hz")):
                means = [cell[f"{metric}_mean"] for cell in cells]
                sds = [cell[f"{metric}_sd"] for cell in cells]
                axes[row, col].errorbar(
                    N_I_SWEEP, means, yerr=sds, marker=marker, color=color,
                    lw=1.5, ms=5.5, capsize=2, label=label,
                )

    for row, ylabel in enumerate(("E rate  (Hz / cell)", "I rate  (Hz / cell)")):
        axes[row, 0].set_ylabel(ylabel, fontsize=theme.SIZE_LABEL)
    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks(N_I_SWEEP, labels=[str(n) for n in N_I_SWEEP])
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes[1, :]:
        ax.set_xlabel("inhibitory pool size  $N_I$", fontsize=theme.SIZE_LABEL)
    axes[0, 0].legend(frameon=False, fontsize=theme.SIZE_LEGEND - 1)
    axes[0, 1].legend(frameon=False, fontsize=theme.SIZE_LEGEND - 1)

    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


def main() -> None:
    meta = parse_meta(sys.argv)
    theme.set_paper_mode(True)
    run_id = next_run_id(SLUG)

    with published_run(SLUG, run_id, scale=SCALE, plot_only=meta.plot_only) as (
        _artifacts, figures,
    ):
        started = time.monotonic()
        cache: dict[tuple[int, float, int], dict] = {}

        def get(n_i: int, g_ie: float, seed: int) -> dict:
            key = (n_i, round(g_ie, 12), seed)
            if key not in cache:
                print(f"N_I={n_i:3d}  G_IE={g_ie:.6g} μS  seed={seed}")
                cache[key] = measure_one(n_i, g_ie, seed)
            return cache[key]

        raw = {"fixed_total": {}, "fixed_synapse": {}}
        summary = {"fixed_total": {}, "fixed_synapse": {}}

        for g_ie in REFERENCE_G_IE:
            level = f"{g_ie:.12g}"
            raw["fixed_total"][level] = {}
            summary["fixed_total"][level] = {}
            for n_i in N_I_SWEEP:
                rows = [get(n_i, g_ie, seed) for seed in SEEDS]
                raw["fixed_total"][level][str(n_i)] = rows
                summary["fixed_total"][level][str(n_i)] = summarise(rows)

        for j_ie in REFERENCE_J_IE:
            level = f"{j_ie:.12g}"
            raw["fixed_synapse"][level] = {}
            summary["fixed_synapse"][level] = {}
            for n_i in N_I_SWEEP:
                g_ie = n_i * j_ie
                rows = [get(n_i, g_ie, seed) for seed in SEEDS]
                raw["fixed_synapse"][level][str(n_i)] = rows
                summary["fixed_synapse"][level][str(n_i)] = summarise(rows)

        plot_controls(summary, figures / "pool_size_controls", run_id)
        write_numbers(
            figures,
            run_id=run_id,
            duration_s=time.monotonic() - started,
            payload={
                "config": {
                    "n_e": N_E,
                    "n_in": N_IN,
                    "n_i_sweep": N_I_SWEEP,
                    "n_i_reference": N_I_REFERENCE,
                    "reference_g_ie": REFERENCE_G_IE,
                    "reference_j_ie": REFERENCE_J_IE,
                    "g_ei_total": W_EI_TOTAL,
                    "input_rate_hz": INPUT_RATE_HZ,
                    "t_ms": T_MS,
                    "dt_ms": DT,
                    "n_batch": N_BATCH,
                    "seeds": SEEDS,
                },
                "definition": "j_ie_synapse = g_ie_total / n_i",
                "raw": raw,
                "summary": summary,
            },
        )
        print(f"wrote {figures / 'pool_size_controls'}.{{svg,pdf}}")
        print(f"wrote {figures / 'numbers.json'}")


if __name__ == "__main__":
    main()
