"""Render saved analysis only; no simulation, measurement or publication.

Invoke this stage explicitly from the repository root. Scientific outputs
and input provenance belong to its independently completed Pingstore run.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp044.custom import inputs
from experiments.exp044.custom.network_artifacts import copy_circuits
from pingstore.library import PingstoreError, load_json, write_json_atomic

DEFAULT_FORMATS = ("svg", "pdf")


def save_figure(fig, stem, formats=DEFAULT_FORMATS) -> list[Path]:
    """Write *fig* to <stem>.<ext> for each ext in *formats*.

    *stem* is a path without an extension (a trailing .png/.pdf/.svg is
    stripped, so existing call sites that pass "foo.png" keep working).
    Format is inferred from the extension; rcParams (dpi, bbox, fonttype)
    come from the active theme profile.
    """
    stem = Path(stem)
    if stem.suffix:
        stem = stem.with_suffix("")
    out: list[Path] = []
    for ext in formats:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p)
        out.append(p)
    return out


def format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def present(identity: str, *, run_id: str | None = None) -> str:

    analysis = inputs.source(REPO, identity, "analyse")
    if set(analysis.record["inputs"]) != {"compute", "bank"}:
        raise PingstoreError(
            "exp044 analysis must pin its computation and training bank"
        )
    ref = analysis.record["inputs"]["compute"]
    compute = inputs.source(REPO, ref["run_id"], "compute", reference=ref)
    cfg = inputs.configuration(compute)
    if compute.record["inputs"]["bank"] != analysis.record["inputs"]["bank"]:
        raise PingstoreError("analysis and compute disagree on the training bank")
    result = load_json(analysis.export / "results.json")
    if (
        result.get("schema") != "exp044.analysis/v1"
        or result.get("recipe") != cfg
        or result.get("measurement") != analysis.record["execution"]["configuration"]
    ):
        raise PingstoreError("unsupported or inconsistent exp044 analysis payload")
    samples = []
    with np.load(analysis.export / "rasters.npz", allow_pickle=False) as data:
        for row in result["rasters"]:
            samples.append(
                {
                    **row,
                    "e": np.array(data[row["cell_name"] + "__e"]),
                    "i": np.array(data[row["cell_name"] + "__i"]),
                }
            )
    started = time.monotonic()
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": analysis},
        run_id=run_id,
        configuration={
            "schema": "exp044.presentation/v1",
            "paper_mode": True,
            "raster_window_ms": cfg["raster"]["window_ms"],
        },
    ) as run:
        theme.set_paper_mode(True)
        plot_dt_sweep(result["aggregate"], run.export / "dt_sweep")
        plot_raster_strip(
            samples, run.export / "raster_strip", cfg["raster"]["window_ms"]
        )
        plot_training_curves(
            result["curves"],
            cfg["dt_sweep_ms"],
            cfg["seeds"],
            run.export / "training_curves",
        )
        run.record["presentation_lineage"] = {
            "analysis": analysis.reference,
            "operation": "render saved measurements and selected raster samples",
            "historical_training_curves": analysis.record["inputs"]["bank"],
            "new_inference": analysis.record["inputs"]["compute"],
        }
        write_json_atomic(
            run.export / "numbers.json",
            {
                **result,
                "run_id": run.run_id,
                "duration_s": time.monotonic() - started,
                "duration": format_duration(time.monotonic() - started),
                "git_sha": run.record["provenance"]["git_commit"],
                "circuit_files": copy_circuits(run),
            },
        )
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="completed exp044 analyse ID")
    parser.add_argument("--run-id", help="unused v4 reservation")
    args = parser.parse_args()
    try:
        present(args.source, run_id=args.run_id)
    except PingstoreError as exc:
        parser.exit(1, f"exp044 present: {exc}\n")


def plot_dt_sweep(rows: list[dict], out_path: Path) -> None:
    """E rate (left axis) and accuracy (right axis) vs Δt (log)."""
    theme.apply()
    dts_sorted = [row["dt_ms"] for row in rows]
    e_means = [row["e_rate_hz"]["mean"] for row in rows]
    e_sems = [row["e_rate_hz"]["sem"] for row in rows]
    acc_means = [row["acc"]["mean"] for row in rows]
    acc_sems = [row["acc"]["sem"] for row in rows]

    fig, ax_rate = plt.subplots(figsize=(5.6, 3.5))
    ax_rate.errorbar(
        dts_sorted,
        e_means,
        yerr=e_sems,
        marker="D",
        markersize=6,
        lw=1.4,
        color=theme.INK_BLACK,
        capsize=3,
        label="E rate (Hz)",
    )
    ax_rate.set_xscale("log")
    ax_rate.set_xlabel("Δt (ms)", fontsize=theme.SIZE_LABEL)
    ax_rate.set_ylabel(
        "Hidden E rate (Hz)", fontsize=theme.SIZE_LABEL, color=theme.INK_BLACK
    )
    ax_rate.tick_params(axis="y", labelcolor=theme.INK_BLACK)
    ax_rate.set_ylim(0, 50)
    ax_rate.set_xticks(dts_sorted)
    ax_rate.set_xticklabels([f"{d:g}" for d in dts_sorted])
    ax_rate.spines["top"].set_visible(False)

    ax_acc = ax_rate.twinx()
    ax_acc.errorbar(
        dts_sorted,
        acc_means,
        yerr=acc_sems,
        marker="s",
        markersize=6,
        lw=1.4,
        color=theme.DEEP_RED,
        capsize=3,
        label="accuracy (%)",
    )
    ax_acc.set_ylabel(
        "Test accuracy (%)", fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED
    )
    ax_acc.tick_params(axis="y", labelcolor=theme.DEEP_RED)
    ax_acc.set_ylim(0, 100)
    ax_acc.spines["top"].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


def plot_raster_strip(
    samples: list[dict],
    out_path: Path,
    t_window_ms: float,
) -> None:
    """Single-trial rasters across Δt, one panel per Δt. X-axis is
    physical time in ms (not steps), so gamma cycle alignment is read
    by eye — same physical period if dynamics survive Δt change."""
    theme.apply()
    n = len(samples)
    n_e = samples[0]["e"].shape[1]
    n_i = samples[0]["i"].shape[1]
    gap = 6
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(6.9, 0.7 * n + 0.8),
        sharex=True,
        gridspec_kw={"hspace": 0.22},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples, strict=True)):
        T = s["e"].shape[0]
        t_axis = np.arange(T) * s["dt_ms"]
        # Truncate display to the first t_window_ms ms so cycles are visible.
        mask = t_axis <= t_window_ms
        e_t, e_n = np.where(s["e"][mask])
        i_t, i_n = np.where(s["i"][mask])
        ax.scatter(
            t_axis[mask][e_t], e_n, s=2.0, c=theme.INK_BLACK, marker="|", linewidths=0.4
        )
        ax.scatter(
            t_axis[mask][i_t],
            i_n + n_e + gap,
            s=2.0,
            c=theme.DEEP_RED,
            marker="|",
            linewidths=0.4,
        )
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, t_window_ms)
        ax.text(
            1.012,
            0.5,
            f"Δt = {s['dt_ms']:g} ms\nE = {s['e_rate_hz']:.1f} Hz",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=theme.SIZE_LABEL,
        )
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    theme.label_panels(axes)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


# Training trajectories are retained validation observations.


def plot_training_curves(
    curves: list[dict], dts_sorted: list[float], seeds: list[int], out_path: Path
) -> None:
    """Per-cell training-trajectory curves. One line per (Δt, seed);
    colour by Δt (viridis on log Δt)."""
    theme.apply()
    cmap = plt.get_cmap("viridis")
    fig, (ax_acc, ax_rate) = plt.subplots(
        2,
        1,
        figsize=(5.6, 4.6),
        sharex=True,
        gridspec_kw={"hspace": 0.15},
    )
    for i, dt_ms in enumerate(dts_sorted):
        color = cmap(i / max(1, len(dts_sorted) - 1))
        for j, seed in enumerate(seeds):
            m = next(
                row for row in curves if row["dt_ms"] == dt_ms and row["seed"] == seed
            )
            eps = [e["ep"] for e in m["epochs"]]
            accs = [e["acc"] for e in m["epochs"]]
            rates = [e["test_rate_e"] for e in m["epochs"]]
            label = f"Δt = {dt_ms:g} ms" if j == 0 else None
            ax_acc.plot(eps, accs, color=color, lw=1.0, alpha=0.85, label=label)
            ax_rate.plot(eps, rates, color=color, lw=1.0, alpha=0.85)
    ax_acc.set_ylabel("Validation accuracy (%)", fontsize=theme.SIZE_LABEL)
    ax_rate.set_ylabel("Validation E rate (Hz)", fontsize=theme.SIZE_LABEL)
    ax_rate.set_xlabel("Epoch", fontsize=theme.SIZE_LABEL)
    ax_acc.legend(fontsize=theme.SIZE_LEGEND, frameon=False, ncol=2, loc="lower right")
    ax_acc.set_ylim(0, 100)
    for ax in (ax_acc, ax_rate):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.15, lw=0.4)
    theme.label_panels((ax_acc, ax_rate))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
