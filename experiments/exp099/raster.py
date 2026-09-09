"""Add a SNNViz spike raster to an explicit completed presentation."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp099 import inputs, recipe
from experiments.exp099.present import resolved
from pingstore.stages import stage_run
from tools.snnviz import FigureGrid, Recording, Theme  # noqa: TID251


def render_raster(recording, cfg, path):
    theme = Theme()
    theme.apply()
    grid = FigureGrid(
        rows=1, columns=(1.7, 1), column_gap=0.10, bounds=(0.065, 0.12, 0.90, 0.77)
    )
    grid.place("overview", row=0, column=0)
    grid.place("detail", row=0, column=1)
    left = grid.subgrid("overview", rows=(1, 3.5), columns=1, row_gap=0.12)
    right = grid.subgrid("detail", rows=(2.5, 2), columns=1, row_gap=0.12)
    left.place("drive", row=0, column=0)
    left.place("raster", row=1, column=0)
    right.place("zoom", row=0, column=0)
    right.place("counts", row=1, column=0)
    fig = grid.figure(figsize=(15, 6), dpi=240)
    drive = left.add_axes(fig, "drive")
    raster = left.add_axes(fig, "raster")
    zoom = right.add_axes(fig, "zoom")
    counts = right.add_axes(fig, "counts")
    # Centre a fixed 100 ms display window on the plateau midpoint.
    origin = cfg.get("burn_in_ms", 0.0)
    view_start = cfg["view_start_ms"] - origin
    view_end = cfg["view_end_ms"] - origin
    centre = (cfg["peak_ms"] + cfg["plateau_end_ms"]) / 2 - origin
    start, stop = centre - 50, centre + 50
    edges = np.arange(start, stop + 1, 1.0)
    t = np.arange(len(recording.signals["spk_e"])) * recording.dt_ms
    e, i = recipe.source_rates(t, cfg)
    t = t - origin
    drive.plot(t, e, color=theme.ink, lw=1.6, label="E-targeted")
    drive.plot(t, i, color=theme.accent, lw=1.6, ls="--", label="I-targeted")
    drive.set(
        ylabel="Hz / source",
        ylim=(min(e.min(), i.min()) - 0.1, max(e.max(), i.max()) * 1.15),
    )
    drive.set_title("A · EXTERNAL AFFERENT RATE", loc="left", pad=10)
    drive.legend(
        loc="upper left", bbox_to_anchor=(0.42, 1.52), ncol=2, frameon=False, fontsize=9
    )
    for pop, offset, colour in (("e", 0, theme.ink), ("i", cfg["n_e"], theme.accent)):
        step, cell = np.nonzero(recording.signals[f"spk_{pop}"])
        spike_times = t[step]
        for ax, mask in (
            (raster, (spike_times >= view_start) & (spike_times < view_end)),
            (zoom, (spike_times >= start) & (spike_times < stop)),
        ):
            ax.scatter(
                spike_times[mask],
                cell[mask] + offset,
                marker="|",
                s=1 if ax is raster else 2,
                linewidths=0.4,
                color=colour,
                rasterized=True,
            )
        selected = spike_times[(spike_times >= start) & (spike_times < stop)]
        bin_counts, _ = np.histogram(selected, bins=edges)
        assert bin_counts.sum() == len(selected)
        counts.stairs(bin_counts, edges, color=colour, lw=1.3, label=pop.upper())
    zoom.axhline(cfg["n_e"] - 0.5, color=theme.muted, lw=0.6)
    zoom.set(
        ylim=(-0.5, cfg["n_e"] + cfg["n_i"] - 0.5),
        ylabel="Neuron index",
        yticks=[0, 800, 1600, 2000],
    )
    zoom.set_title(f"C · CLOSE-UP: {start:.0f}–{stop:.0f} ms", loc="left", pad=10)
    counts.set_title("D · SPIKE COUNTS · 1 ms BINS", loc="left", pad=10)
    counts.set(ylabel="Spikes / bin", xlabel="Time (ms)", ylim=(0, None))
    counts.legend(frameon=False, loc="upper left", fontsize=9)
    for ax in (zoom, counts):
        ax.set_xlim(start, stop)
        ax.set_xticks(np.arange(start, stop + 1, 20))
    raster.axvspan(start, stop, color=theme.muted, alpha=0.12, lw=0)
    raster.axhline(cfg["n_e"] - 0.5, color=theme.muted, lw=0.6)
    raster.set(
        ylim=(-0.5, cfg["n_e"] + cfg["n_i"] - 0.5),
        ylabel="Neuron index",
        xlabel="Time (ms)",
        yticks=[0, 400, 800, 1200, 1600, 2000],
    )
    raster.set_title("B · SPIKE RASTER", loc="left", pad=10)
    raster.text(1.005, 0.4, "E", transform=raster.transAxes, color=theme.ink)
    raster.text(1.005, 0.9, "I", transform=raster.transAxes, color=theme.accent)
    for ax in (drive, raster):
        ax.set_xlim(view_start, view_end)
        ax.set_xticks(np.linspace(view_start, view_end, 7))
        for boundary in (cfg["onset_ms"], cfg["offset_ms"]):
            ax.axvline(boundary - origin, color=theme.muted, ls=":", lw=0.7)
    drive.tick_params(labelbottom=False)
    fig.text(
        0.065,
        0.035,
        f"All {cfg['n_e']:,} E and {cfg['n_i']:,} I neurons · seed {cfg['seed']} · one mark per spike · counts are population totals, unsmoothed",
        fontsize=9,
        color=theme.muted,
    )
    fig.savefig(path, dpi=240)
    plt.close(fig)


def present_raster(identity, *, from_analysis=False):
    previous = None if from_analysis else inputs.source(REPO, identity, "present")
    analysis, compute, cfg, _ = resolved(
        identity if from_analysis else previous.record["inputs"]["analysis"]["run_id"]
    )
    sources = {"analysis": analysis, "compute": compute}
    if previous is not None:
        sources["presentation"] = previous
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs=sources,
        configuration=cfg,
    ) as run:
        if previous is not None:
            for path in previous.export.iterdir():
                shutil.copy2(path, run.export / path.name)
        render_raster(
            Recording(cfg["dt_ms"], inputs.recording(compute)),
            cfg,
            run.export / "spike-raster.png",
        )
        run.record["execution"]["raster"] = {
            "neurons": "all E and I",
            "interval_ms": [cfg["view_start_ms"], cfg["view_end_ms"]],
            "display_time_origin_ms": cfg.get("burn_in_ms", 0.0),
            "source_video": previous.reference if previous is not None else None,
            "closeup_ms": [
                (cfg["peak_ms"] + cfg["plateau_end_ms"]) / 2 - 50,
                (cfg["peak_ms"] + cfg["plateau_end_ms"]) / 2 + 50,
            ],
            "count_bin_ms": 1.0,
            "count_smoothing": "none",
        }
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--from-analysis", action="store_true")
    args = parser.parse_args()
    present_raster(args.source, from_analysis=args.from_analysis)
