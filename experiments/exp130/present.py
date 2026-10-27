"""Render Figures 1–3 from explicit paired compute and analysis evidence.

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
import json
from importlib.metadata import version

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp130 import dataset, recipe
from experiments.exp130.custom import inputs
from pingstore.library import PingstoreError, load_json, stage_run, write_json_atomic
from snnlab import lang
from snnlab.viz import Theme


def present(identity):
    analysis = inputs.source(REPO, identity, "analyse")
    cfg = inputs.configuration(analysis)
    refs = analysis.record["inputs"]
    if set(refs) != {"compute"}:
        raise PingstoreError("analysis must pin exactly one paired compute")
    compute = inputs.source(
        REPO, refs["compute"]["run_id"], "compute", reference=refs["compute"]
    )
    results = load_json(analysis.export / "results.json")
    if (
        inputs.configuration(compute) != cfg
        or results.get("parameters") != cfg
        or results.get("schema") != "exp130.analysis/v1"
    ):
        raise PingstoreError("compute and analysis disagree")
    recordings = {
        condition: inputs.recording(compute, condition)
        for condition in recipe.CONDITIONS
    }
    with np.load(analysis.export / "measurements.npz", allow_pickle=False) as raw:
        measurements = {
            condition: {
                key.split("__", 1)[1]: raw[key]
                for key in raw.files
                if key.startswith(condition + "__")
            }
            for condition in recipe.CONDITIONS
        }
    view = dict(start_ms=cfg["view_start_ms"], end_ms=cfg["view_end_ms"])
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs={"analysis": analysis, "compute": compute},
        configuration=cfg,
    ) as run:
        replays = {}
        for condition in recipe.CONDITIONS:
            with np.load(
                compute.unit(condition) / "weights.npz", allow_pickle=False
            ) as raw:
                weights = dict(raw)
            replays[condition] = circuit_data(
                recordings[condition],
                weights,
                cfg,
                view,
                measurements=measurements[condition],
            )
        shared_max = max(
            data["piston_scales"]["conductance_us"][1] for data in replays.values()
        )
        for condition, data in replays.items():
            data["piston_scales"]["conductance_us"][1] = shared_max
            data["condition"] = condition
        (run.export / "circuit.json").write_text(
            json.dumps(
                {"schema": "exp130.comparison/v1", "conditions": replays},
                separators=(",", ":"),
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        bundle_path = (
            compute.restore_bundle(
                "theta/network.bundle", run.scratch / "network.bundle"
            )
            if compute.record.get("scientific_artifacts", {}).get("native_paths")
            else compute.unit("theta", "network.bundle")
        )
        run.record["execution"]["network_diagram"] = network_diagram(
            run.export / "network.svg",
            lang.load_bundle(bundle_path),
        )
        render_raster(recordings, cfg, run.export / "spike-raster.png")
        write_json_atomic(run.export / "numbers.json", results)
        run.record["execution"]["presentation"] = dict(
            figures="fresh structural diagram, selectable recorded circuit replay, paired all-neuron raster",
            view=view,
            shared_conductance_scale_us=[0, shared_max],
            raster_interval_ms=[cfg["view_start_ms"], cfg["view_end_ms"]],
            raster_panels=["A", "B", "D", "E", "F", "H"],
            raster_counts="1 ms population counts over the complete recorded interval; no smoothing",
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    print(present(args.source))


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
        "schema": "exp130.circuit/v3",
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


def network_diagram(path, bundle):
    bundle.visualise(path, view="circuit")
    return dict(
        library="snnlab",
        version=version("snnlab"),
        view="circuit",
        construction="lang.Bundle.visualise",
        renderer="lang.Bundle.visualise",
    )


def render_raster(recordings, cfg, path):
    theme = Theme()
    theme.apply()
    fig, axes = plt.subplots(
        3, 2, figsize=(16, 8), dpi=180, gridspec_kw={"height_ratios": [1, 3, 1.5]}
    )
    origin = cfg["burn_in_ms"]
    start, stop = cfg["view_start_ms"] - origin, cfg["view_end_ms"] - origin
    t = np.arange(round(cfg["t_ms"] / cfg["dt_ms"])) * cfg["dt_ms"] - origin
    colours = dict(e=theme.ink, i=theme.accent)
    for column, condition in enumerate(recipe.CONDITIONS):
        drive, raster, counts_ax = axes[:, column]
        rates = dataset.source_rates(
            t + origin, recipe.condition_configuration(cfg, condition)
        )
        for pop, rate, linestyle in zip(("e", "i"), rates, ("-", "--"), strict=True):
            drive.plot(
                t,
                rate,
                color=colours[pop],
                ls=linestyle,
                lw=1.5,
                label=f"{pop.upper()}-targeted",
            )
        title = (
            f"THETA · {cfg['theta_hz']:g} Hz"
            if condition == "theta"
            else f"CONSTANT · {cfg['baseline_e_hz']:g} Hz"
        )
        drive.set_title(f"{'A' if column == 0 else 'E'} · {title}", loc="left")
        drive.set(
            ylabel="Hz / afferent",
            ylim=(
                cfg["baseline_e_hz"] - cfg["modulation_hz"] - 0.07,
                cfg["baseline_e_hz"] + cfg["modulation_hz"] + 0.07,
            ),
        )
        drive.legend(
            loc="lower right",
            bbox_to_anchor=(1, 1.02),
            frameon=False,
            fontsize=8,
            ncol=2,
        )
        edges = np.arange(start, stop, 1.0)
        edges = np.append(edges, stop)
        for pop, offset in (("e", 0), ("i", cfg["n_e"])):
            step, cell = np.nonzero(recordings[condition][f"spk_{pop}"])
            times = t[step]
            selected = (times >= start) & (times < stop)
            raster.scatter(
                times[selected],
                cell[selected] + offset,
                marker="|",
                s=1,
                linewidths=0.35,
                color=colours[pop],
                rasterized=True,
            )
            selected = times[(times >= start) & (times < stop)]
            counts, _ = np.histogram(selected, bins=edges)
            if counts.sum() != len(selected):
                raise ValueError("histogram lost spikes")
            # Final partial bin is divided by its width, preserving comparable units.
            counts_ax.stairs(
                counts / np.diff(edges),
                edges,
                color=colours[pop],
                lw=1.0,
                label=pop.upper(),
            )
        for ax in (raster,):
            ax.axhline(cfg["n_e"] - 0.5, color=theme.muted, lw=0.5)
            ax.set(
                ylim=(-0.5, cfg["n_e"] + cfg["n_i"] - 0.5),
                ylabel="Neuron index",
                yticks=[0, 800, 1600, 2000],
            )
        raster.set_title(
            f"{'B' if column == 0 else 'F'} · ALL-NEURON RASTER", loc="left"
        )
        counts_ax.set_title(
            f"{'D' if column == 0 else 'H'} · POPULATION COUNTS", loc="left"
        )
        for ax in (drive, raster, counts_ax):
            ax.set_xlim(start, stop)
            ax.set_xticks(np.linspace(start, stop, 6))
            ax.set_xlabel("Time (ms)")
        counts_ax.set(xlabel="Time (ms)", ylabel="Spikes / ms", ylim=(0, None))
        counts_ax.legend(frameon=False, fontsize=8)
    for row in (0, 1):
        limits = [ax.get_ylim() for ax in axes[row]]
        for ax in axes[row]:
            ax.set_ylim(min(x[0] for x in limits), max(x[1] for x in limits))
    ceiling = max(ax.get_ylim()[1] for ax in axes[2])
    for ax in axes[2]:
        ax.set_ylim(0, ceiling)
    fig.subplots_adjust(
        left=0.065, right=0.98, top=0.94, bottom=0.09, hspace=0.65, wspace=0.24
    )
    fig.text(
        0.065,
        0.025,
        f"1,600 E (black) / 400 I (red) · seed {cfg['seed']} · identical circuit · private Poisson inputs · counts unsmoothed; E has four times as many cells",
        fontsize=9,
        color=theme.muted,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
