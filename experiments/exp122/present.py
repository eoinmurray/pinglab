"""Render retained measurements and rasters in the house style."""

import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[2] / "tools"),
]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp122 import recipe
from experiments.helpers import theme
from pingstore.stages import source_run, stage_run


def save(fig, destination):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(destination.with_suffix("." + ext), dpi=300)
    plt.close(fig)


def plot_compound(raw, compute, destination, sweep):
    cfg = raw["design"]
    conditions = [c for c in cfg["conditions"] if c["sweep"] == sweep]
    values = [c["axis_value"] for c in conditions]
    summaries = [r for r in raw["summaries"] if r["sweep"] == sweep]
    fig, all_axes = plt.subplots(
        2, 4, figsize=(180 / 25.4, 105 / 25.4), layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0.08, h_pad=0.06)
    axes = all_axes[:, :2]
    specs = [
        ("psd_frequency_hz", "PSD frequency", "Spectral peak (Hz)"),
        ("participation_e_pct", "E participation", "Participation (%)"),
        ("e_spikes_per_second", "E spike rate", "E spikes/s"),
        ("rhythmicity_contrast", "Rhythmicity", "Lobe–trough contrast"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, specs):
        ax.plot(
            values,
            [r[key] for r in summaries],
            "o-",
            color=theme.INK_BLACK,
            linewidth=1.3,
            markersize=3,
            label="Peak of mean PSD" if key == "psd_frequency_hz" else "Encoding mean",
        )
        ax.axvline(6.0 if sweep == "tau" else 1.0, color=theme.FAINT, ls="--", lw=0.7)
        ax.set(
            xscale="log",
            xlabel={
                "tau": "GABA decay (ms)",
                "leak": "Leak multiplier",
                "cap": "Capacitance multiplier",
            }[sweep],
            ylabel=ylabel,
            title=title,
        )
        ax.set_xticks(
            [3, 6, 10, 20, 30] if sweep == "tau" else [0.5, 1, 2],
            labels=["3", "6", "10", "20", "30"]
            if sweep == "tau"
            else ["0.5", "1", "2"],
        )
        ax.minorticks_off()
        if key == "participation_e_pct":
            ax.set_ylim(0, 30)
        if key == "rhythmicity_contrast":
            ax.set_ylim(0, 1.05)
        if key == "e_spikes_per_second":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
    theme.label_panels(axes.flat)
    axes = all_axes[:, 2:]
    for ax, j in zip(axes.flat, [0, 3, 6, 9]):
        with np.load(
            compute.file(f"{conditions[j]['condition_id']}--spikes.npz"),
            allow_pickle=False,
        ) as data:
            for key, colour, offset in [
                ("spk_e", theme.INK_BLACK, 0),
                ("spk_i", theme.DEEP_RED, 1024),
            ]:
                k, neuron = np.nonzero(data[key][:, 0])
                ax.scatter(
                    k * cfg["dt_ms"],
                    neuron + offset,
                    s=0.22,
                    color=colour,
                    linewidths=0,
                    rasterized=True,
                )
        ax.axhline(1023.5, color=theme.FAINT, lw=0.5)
        ax.set(
            xlim=(0, 200),
            ylim=(-1, 1280),
            xlabel="Time (ms)",
            ylabel="Neuron",
            title=f"τ = {values[j]:.2f} ms"
            if sweep == "tau"
            else f"{'gL' if sweep == 'leak' else 'C'} × {values[j]:.2f}",
        )
        ax.set_yticks([0, 1024, 1280])
        ax.set_xticks([0, 100, 200])
    theme.label_panels(axes.flat, labels=["E", "F", "G", "H"])
    save(fig, destination)


def present(identity):
    source = source_run(
        recipe.REPO / ".pingstore", identity, stage="analyse", experiment=recipe.SLUG
    )
    raw = json.loads(source.file("metrics.json").read_text())
    if raw.get("schema") != "exp122.analysis/v2":
        raise ValueError("unsupported analysis schema")
    cfg, rows = raw["design"], raw["rows"]
    ref = source.record["inputs"]["compute"]
    compute = source_run(
        recipe.REPO / ".pingstore", ref["run_id"], stage="compute", reference=ref
    )
    theme.set_paper_mode(True)
    theme.apply()
    plt.rcParams["svg.fonttype"] = "none"
    with stage_run(
        recipe.REPO,
        recipe.SLUG,
        "present",
        inputs={"analyse": source, "compute": compute},
        configuration={
            "schema": "exp122.presentation/v1",
            "layout": "metrics_left_rasters_right",
            "paper_mode": True,
            "style": "shared paper theme; black primary metrics, black E and red I rasters",
            "show_encoding_traces": False,
            "raster_tau_indices": [0, 3, 6, 9],
            "raster_encoding_seed": 42,
        },
    ) as run:
        plot_compound(raw, compute, run.export / "calibration_compound", "tau")
        plot_compound(raw, compute, run.export / "leak_compound", "leak")
        plot_compound(raw, compute, run.export / "capacitance_compound", "cap")
        with np.load(compute.file("input.npz"), allow_pickle=False) as data:
            plt.imsave(
                run.export / "input_image.png",
                data["pixels"],
                cmap="gray",
                vmin=0,
                vmax=1,
            )
        (run.export / "numbers.json").write_text(
            json.dumps(raw, indent=2, allow_nan=False) + "\n"
        )
        identity = run.run_id
    print(identity, flush=True)
    return identity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    present(parser.parse_args().source)
