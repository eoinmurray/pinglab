"""Render the retained exp112 factorial comparison; never train or infer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import matplotlib.pyplot as plt
import numpy as np
from experiments.exp112 import recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import source_run, stage_run

COLORS = {"coba": "#a33a2b", "ping": "#171717"}
POPULATION_COLORS = {"e": "#171717", "i": "#a33a2b"}
LINESTYLES = {1.0: "-", 1000.0: "--"}


def _training_figure(result: dict, destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for case in recipe.CASES:
        history = result["histories"][case["id"]]
        epochs = [row["ep"] for row in history]
        label = f"{case['architecture'].upper()}, d={case['v_grad_dampen']:g}"
        style = {
            "color": COLORS[case["architecture"]],
            "linestyle": LINESTYLES[case["v_grad_dampen"]],
            "linewidth": 1.8,
            "label": label,
        }
        axes[0].plot(epochs, [row["acc"] for row in history], **style)
        axes[1].plot(epochs, [row["grad_norm"] for row in history], **style)
    axes[0].set(xlabel="Epoch", ylabel="Validation accuracy (%)")
    axes[1].set(xlabel="Epoch", ylabel="Mean pre-clip gradient norm", yscale="log")
    axes[0].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.18)
    fig.savefig(destination, dpi=220)
    plt.close(fig)


def _test_figure(result: dict, destination: Path) -> None:
    rows = result["conditions"]
    labels = [
        f"{row['architecture'].upper()}\nd={row['v_grad_dampen']:g}" for row in rows
    ]
    values = [row["official_test_accuracy_pct"] for row in rows]
    colors = [COLORS[row["architecture"]] for row in rows]
    hatches = ["" if row["v_grad_dampen"] == 1.0 else "//" for row in rows]
    fig, axis = plt.subplots(figsize=(5.8, 3.5), constrained_layout=True)
    bars = axis.bar(np.arange(len(rows)), values, color=colors, width=0.7)
    for bar, hatch in zip(bars, hatches, strict=True):
        bar.set_hatch(hatch)
    axis.set(
        xticks=np.arange(len(rows)),
        xticklabels=labels,
        ylabel="Official MNIST test accuracy (%)",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.savefig(destination, dpi=220)
    plt.close(fig)


def _raster_figure(result: dict, rasters: Path, destination: Path) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(7.2, 5.4), sharex=True, sharey=True, constrained_layout=True
    )
    with np.load(rasters, allow_pickle=False) as arrays:
        for axis, case, panel in zip(axes.flat, recipe.CASES, "ABCD", strict=True):
            case_id = case["id"]
            metadata = result["rasters"][case_id]
            dt = metadata["dt_ms"]
            n_e = metadata["n_e"]
            e_t = arrays[f"{case_id}__e_t"] * dt
            e_cell = arrays[f"{case_id}__e_cell"]
            i_t = arrays[f"{case_id}__i_t"] * dt
            i_cell = arrays[f"{case_id}__i_cell"] + n_e
            axis.scatter(
                e_t,
                e_cell,
                marker="|",
                s=1.2,
                linewidths=0.35,
                color=POPULATION_COLORS["e"],
                rasterized=True,
            )
            axis.scatter(
                i_t,
                i_cell,
                marker="|",
                s=1.4,
                linewidths=0.4,
                color=POPULATION_COLORS["i"],
                rasterized=True,
            )
            axis.axhline(n_e - 0.5, color="#777777", linewidth=0.6)
            axis.set_title(
                f"{panel} · {case['architecture'].upper()}, d={case['v_grad_dampen']:g}",
                loc="left",
                fontsize=9,
            )
            axis.set_xlim(0, metadata["duration_ms"])
            axis.set_ylim(-0.5, metadata["n_e"] + metadata["n_i"] - 0.5)
            axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[:, 0]:
        axis.set_ylabel("Neuron index (E then I)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (ms)")
    fig.savefig(destination, dpi=240)
    plt.close(fig)


def present(identity: str, *, run_id: str | None = None) -> str:
    source = source_run(
        REPO / ".pingstore", identity, stage="analyse", experiment=recipe.SLUG
    )
    result = load_json(source.export / "results.json")
    if result.get("schema") != "exp112.results/v1":
        raise PingstoreError("analysis result schema differs from exp112")
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs={"analysis": source},
        run_id=run_id,
        configuration={"schema": "exp112.presentation/v1"},
    ) as run:
        _training_figure(result, run.export / "training-comparison.png")
        _test_figure(result, run.export / "test-accuracy.png")
        _raster_figure(
            result,
            source.export / "rasters.npz",
            run.export / "final-epoch-rasters.png",
        )
        write_json_atomic(
            run.export / "numbers.json",
            {
                "schema": "exp112.presentation-numbers/v1",
                "conditions": result["conditions"],
                "dataset_identity": result["dataset_identity"],
                "rasters": result["rasters"],
            },
        )
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    try:
        present(args.source, run_id=args.run_id)
    except (OSError, KeyError, TypeError, ValueError, PingstoreError) as exc:
        parser.exit(1, f"exp112 present: {exc}\n")


if __name__ == "__main__":
    main()
