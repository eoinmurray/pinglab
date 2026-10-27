"""Render an explicit analysis and pinned compute source; never publish.

The CLI requires explicit completed analysis evidence. Figures and browser
replays are written to a new presentation run; upstream execution is never
launched automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import argparse
import hashlib
import subprocess
from importlib.metadata import version

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp099 import dataset, recipe
from experiments.exp099.custom import inputs
from pingstore.library import PingstoreError, load_json, stage_run, write_json_atomic
from snnlab import lang as snn
from snnlab.viz import FigureGrid, Recording, Theme

REFERENCE_PDF = REPO / "papers/Susin-Destexhe-2021.pdf"


def render_reference(source: Path, destination: Path) -> dict:
    """Crop the comparison panels from the locally retained CC BY reference."""
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    subprocess.run(
        [
            "uv",
            "run",
            "--with",
            "pymupdf",
            "python",
            "-c",
            """
import pymupdf as fitz
import sys
source = fitz.open(sys.argv[1])
scale = 300 / 72
x, y, width, height = 375, 555, 710, 540
out = fitz.open()
page = out.new_page(width=width / scale, height=height / scale)
for left, top, right, bottom in [(0, 0, 100, 45), (0, 45, width, height)]:
    target = fitz.Rect(left / scale, top / scale, right / scale, bottom / scale)
    clip = fitz.Rect((x + left) / scale, (y + top) / scale,
                    (x + right) / scale, (y + bottom) / scale)
    page.show_pdf_page(target, source, 12, clip=clip)
page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(sys.argv[2])
""",
            str(source),
            str(destination),
        ],
        check=True,
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError("reference PDF changed during extraction")
    return {
        "doi": "10.1371/journal.pcbi.1009416",
        "pdf_sha256": digest,
        "page": 13,
        "figure": "4, PING column B–C",
        "dpi": 300,
        "crop_xywh_pixels": [375, 555, 710, 540],
        "license": "CC BY 4.0",
        "modification": "cropped B/C; upper-right adjacent A fragment excluded with PDF clipping",
    }


def resolved(identity):
    analysis = inputs.source(REPO, identity, "analyse")
    cfg = inputs.configuration(analysis)
    refs = analysis.record["inputs"]
    if set(refs) != {"compute"}:
        raise PingstoreError("exp099 analysis must pin one compute input")
    compute = inputs.source(
        REPO, refs["compute"]["run_id"], "compute", reference=refs["compute"]
    )
    results = load_json(analysis.export / "results.json")
    if (
        inputs.configuration(compute) != cfg
        or results.get("parameters") != cfg
        or results.get("schema") != "exp099.analysis/v2"
    ):
        raise PingstoreError("exp099 analysis configuration disagrees with compute")
    return analysis, compute, cfg, results


def render_outputs(analysis, compute, cfg, results, output, *, scratch, view=None):
    recording = Recording(cfg["dt_ms"], inputs.recording(compute))
    with np.load(analysis.export / "measurements.npz", allow_pickle=False) as arrays:
        measurements = dict(arrays)
    with np.load(compute.export / "weights.npz", allow_pickle=False) as arrays:
        weights = dict(arrays)
    write_json_atomic(
        output / "circuit.json",
        circuit_data(
            recording.signals,
            weights,
            cfg,
            view
            or {
                "start_ms": cfg["view_start_ms"],
                "end_ms": cfg["view_end_ms"],
            },
            measurements=measurements,
        ),
    )
    bundle_path = (
        compute.restore_bundle("network.bundle", scratch / "network.bundle")
        if compute.record.get("scientific_artifacts", {}).get("native_paths")
        else compute.unit("network.bundle")
    )
    diagram_metadata = network_diagram(
        output / "network.svg",
        bundle=snn.load_bundle(bundle_path),
    )
    display_cfg = dict(cfg)
    if view is not None:
        display_cfg.update(view_start_ms=view["start_ms"], view_end_ms=view["end_ms"])
    render_raster(recording, display_cfg, output / "spike-raster.svg")
    reference = render_reference(REFERENCE_PDF, output / "susin-destexhe-2021-ping.png")
    write_json_atomic(output / "numbers.json", results)
    centre = (cfg["peak_ms"] + cfg["plateau_end_ms"]) / 2
    return {
        "network_diagram": diagram_metadata,
        "raster": {
            "neurons": "all E and I",
            "interval_ms": [
                display_cfg["view_start_ms"],
                display_cfg["view_end_ms"],
            ],
            "display_time_origin_ms": cfg.get("burn_in_ms", 0.0),
            "closeup_ms": [centre - 50, centre + 50],
            "count_bin_ms": 1.0,
            "count_smoothing": "none",
        },
        "reference_crop": reference,
    }


def present(identity, *, run_id=None):
    analysis, compute, cfg, results = resolved(identity)
    view = {
        "start_ms": cfg["view_start_ms"],
        "end_ms": cfg["view_end_ms"],
    }
    if not 0 <= view["start_ms"] < view["end_ms"] <= cfg["t_ms"]:
        raise ValueError("presentation interval must lie within the recording")
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs={"analysis": analysis, "compute": compute},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        run.record["execution"]["presentation"] = view
        run.record["execution"].update(
            render_outputs(
                analysis,
                compute,
                cfg,
                results,
                run.export,
                scratch=run.scratch,
                view=view,
            )
        )
    return run.run_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    a = p.parse_args()
    present(a.source, run_id=a.run_id)


def circuit_data(recording, weights, cfg, view, *, measurements):
    dt = cfg["dt_ms"]
    first = round(view["start_ms"] / dt)
    stop = round(view["end_ms"] / dt)
    selections = {
        pop: np.linspace(0, cfg[f"n_{pop}"] - 1, count, dtype=int)
        for pop, count in (("e", min(160, cfg["n_e"])), ("i", min(40, cfg["n_i"])))
    }
    groups = {}
    for pop, indices in selections.items():
        for name, key in ((pop, f"spk_{pop}"), (f"private_{pop}", f"private_{pop}")):
            values = recording[key][first:stop, indices]
            steps, cells = np.nonzero(values)
            groups[name] = {
                "indices": indices.tolist(),
                # Integer steps preserve exact event timing and simultaneous counts.
                "events": np.column_stack((steps, cells, values[steps, cells]))
                .astype(int)
                .tolist(),
            }
    edges = []
    for src in ("e", "i"):
        for dst in ("e", "i"):
            matrix = weights[f"{src.upper()}_to_{dst.upper()}.weight"]
            if matrix.shape != (cfg[f"n_{src}"], cfg[f"n_{dst}"]):
                raise ValueError("connection matrix must have source then target axes")
            selected = matrix[np.ix_(selections[src], selections[dst])]
            source, target = np.nonzero(selected)
            # A deterministic display subset; every displayed edge actually exists.
            take = np.linspace(0, len(source) - 1, min(80, len(source)), dtype=int)
            edges.extend([src, int(source[k]), dst, int(target[k])] for k in take)
    means = {}
    for key in ("mean_g_e", "mean_g_i", "mean_v_e", "mean_v_i"):
        values = np.asarray(measurements[key])
        if values.shape != (round(cfg["t_ms"] / dt),) or not np.isfinite(values).all():
            raise ValueError(f"invalid recorded population mean: {key}")
        means[key] = values[first:stop].tolist()
    conductance_max = max(
        0.001, max(max(means[key]) for key in ("mean_g_e", "mean_g_i")) * 1.15
    )
    rates = {}
    for pop in ("e", "i"):
        values = np.asarray(measurements[f"rate_private_{pop}"])
        if (
            values.shape != (round(cfg["t_ms"] / dt),)
            or not np.isfinite(values).all()
            or (values < 0).any()
        ):
            raise ValueError(f"invalid recorded afferent-rate protocol: {pop}")
        rates[pop] = values[first:stop].tolist()
    return {
        "schema": "exp099.circuit/v3",
        "dt_ms": dt,
        "duration_ms": (stop - first) * dt,
        "start_ms": view["start_ms"] - cfg.get("burn_in_ms", 0),
        "delay_ms": cfg["delay_ms"],
        "population_sizes": {pop: cfg[f"n_{pop}"] for pop in ("e", "i")},
        "afferents_per_neuron": cfg["external_afferents_per_neuron"],
        "groups": groups,
        "edges": edges,
        "means": means,
        "piston_scales": {
            "conductance_us": [0, conductance_max],
            "voltage_mv": [-80, -50],
        },
        "afferent_rates_hz": rates,
    }


def network_diagram(path, *, bundle):
    """Render the unmodified expanded bundle diagram with SNNLab defaults."""
    bundle.visualise(path, view="circuit")
    return {
        "library": "snnlab",
        "version": version("snnlab"),
        "construction": "snnlab.lang.Bundle.visualise",
        "renderer": "snnlab.lang.Bundle.visualise",
        "view": "circuit",
        "annotations": "library defaults",
        "layout": "library defaults",
    }


def render_raster(recording, cfg, path):
    """Render the publication raster from the retained simulation recording."""
    raster_theme = Theme()
    raster_theme.apply()
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

    origin = cfg.get("burn_in_ms", 0.0)
    view_start = cfg["view_start_ms"] - origin
    view_end = cfg["view_end_ms"] - origin
    centre = (cfg["peak_ms"] + cfg["plateau_end_ms"]) / 2 - origin
    start, stop = centre - 50, centre + 50
    edges = np.arange(start, stop + 1, 1.0)
    t = np.arange(len(recording.signals["spk_e"])) * recording.dt_ms
    e, i = dataset.source_rates(t, cfg)
    t = t - origin
    drive.plot(t, e, color=raster_theme.ink, lw=1.6, label="E-targeted")
    drive.plot(t, i, color=raster_theme.accent, lw=1.6, ls="--", label="I-targeted")
    drive.set(
        ylabel="Hz / source",
        ylim=(min(e.min(), i.min()) - 0.1, max(e.max(), i.max()) * 1.15),
    )
    drive.set_title("A · EXTERNAL AFFERENT RATE", loc="left", pad=10)
    drive.legend(
        loc="upper left", bbox_to_anchor=(0.42, 1.52), ncol=2, frameon=False, fontsize=9
    )
    for pop, offset, colour in (
        ("e", 0, raster_theme.ink),
        ("i", cfg["n_e"], raster_theme.accent),
    ):
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
        if bin_counts.sum() != len(selected):
            raise ValueError("raster histogram does not preserve spike counts")
        counts.stairs(bin_counts, edges, color=colour, lw=1.3, label=pop.upper())
    zoom.axhline(cfg["n_e"] - 0.5, color=raster_theme.muted, lw=0.6)
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
    raster.axvspan(start, stop, color=raster_theme.muted, alpha=0.12, lw=0)
    raster.axhline(cfg["n_e"] - 0.5, color=raster_theme.muted, lw=0.6)
    raster.set(
        ylim=(-0.5, cfg["n_e"] + cfg["n_i"] - 0.5),
        ylabel="Neuron index",
        xlabel="Time (ms)",
        yticks=[0, 400, 800, 1200, 1600, 2000],
    )
    raster.set_title("B · SPIKE RASTER", loc="left", pad=10)
    raster.text(1.005, 0.4, "E", transform=raster.transAxes, color=raster_theme.ink)
    raster.text(1.005, 0.9, "I", transform=raster.transAxes, color=raster_theme.accent)
    for ax in (drive, raster):
        ax.set_xlim(view_start, view_end)
        ax.set_xticks(np.linspace(view_start, view_end, 7))
        for boundary in (cfg["onset_ms"], cfg["offset_ms"]):
            ax.axvline(boundary - origin, color=raster_theme.muted, ls=":", lw=0.7)
    drive.tick_params(labelbottom=False)
    fig.text(
        0.065,
        0.035,
        f"All {cfg['n_e']:,} E and {cfg['n_i']:,} I neurons · seed {cfg['seed']} · one mark per spike · counts are population totals, unsmoothed",
        fontsize=9,
        color=raster_theme.muted,
    )
    fig.savefig(path, dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
