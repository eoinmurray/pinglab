"""Render the A-cued and B-cued digit figures from a pinned analysis."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import shutil
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp164 import recipe
from pingstore.library import source_run, stage_run, write_json_atomic

COLORS = {"A": "#168047", "B": "#5367c4", "C": "#222222"}
PATHS = recipe.PATHS


def figure(cfg, time, result, scales, *, cued, end_ms, window_ms, closeup_ms):
    fig = plt.figure(figsize=(12, 11))
    grid = fig.add_gridspec(4, 3, height_ratios=[1.0, 0.85, 1.0, 1.25])
    axes = [fig.add_subplot(grid[i, :]) for i in range(3)]
    shown = time <= end_ms
    for c in "ABC":
        axes[0].plot(
            time[shown],
            result["rates"][c][shown],
            color=COLORS[c],
            lw=0.9,
            label=c + " E",
        )
    axes[0].set(
        ylabel="population rate (Hz)",
        ylim=(0, scales["rate_max_hz"]),
        title="a. Population activity",
    )
    axes[0].legend(ncol=3, loc="upper right", frameon=False)
    for c in "AB":
        axes[1].plot(
            time[shown],
            result["differences"][c][shown],
            color=COLORS[c],
            lw=1.0,
            label=c + " − C",
        )
    axes[1].set(
        ylabel="phase difference (cycles)",
        ylim=scales["phase_limits_cycles"],
        title="b. Gamma phase difference · flat indicates locking",
    )
    axes[1].legend(ncol=2, loc="lower left", frameon=False)
    for ax in axes[:2]:
        ax.set(xlim=(0, end_ms), xlabel="time (ms)")
        ax.axvspan(*window_ms, color="#888888", alpha=0.10, zorder=-2)
    v = (time >= closeup_ms[0]) & (time <= closeup_ms[1])
    gi = result["conductances"][PATHS[2]]
    axes[2].fill_between(
        time[v],
        0,
        1,
        where=gi[v] <= scales["low_inhibition_threshold_us"],
        transform=axes[2].get_xaxis_transform(),
        color="#e5bc48",
        alpha=0.25,
        label="Low C inhibition (proxy)",
    )
    for c in "AB":
        axes[2].plot(
            time[v],
            result["conductances"][c + "_E_to_C_E"][v] / scales["excitation_peak_us"],
            color=COLORS[c],
            lw=1.1,
            label=c + " arriving excitation",
        )
    axes[2].plot(
        time[v],
        gi[v] / scales["inhibition_peak_us"],
        color="#bc2942",
        lw=1.1,
        label="C inhibition",
    )
    axes[2].set(
        xlim=closeup_ms,
        ylim=(0, 1.08),
        xlabel="time (ms)",
        ylabel="conductance / peak",
        title="c. Conductance close-up · reconstructed from saved spikes",
    )
    axes[2].legend(ncol=2, loc="upper right", fontsize=8, frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.title.set_ha("left")
        ax.title.set_position((0.0, 1.0))
    for i, c in enumerate("ABC"):
        ax = fig.add_subplot(grid[3, i])
        im = ax.imshow(
            result["maps"][c],
            cmap="magma",
            vmin=0,
            vmax=scales["map_max_hz"][c],
            interpolation="nearest",
        )
        ax.set_title(
            f"{chr(100 + i)}. {c} E · "
            + (f"digit {cfg['labels'][i]}" if i < 2 else "receiver")
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8, label="Hz", pad=0.025)
    plv = result["plvs"]
    fig.suptitle(
        f"Digit competition · {cued} cued throughout the displayed interval\n"
        f"A: digit {cfg['labels'][0]}    B: digit {cfg['labels'][1]}    |    PLV A–C {plv['A']:.3f}, B–C {plv['B']:.3f}",
        fontsize=13,
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.09, right=0.96, bottom=0.11, top=0.90, hspace=0.65, wspace=0.48
    )
    caption = (
        f"Both sensory inputs stay present. Grey bands and heatmaps: {window_ms[0]:g}–{window_ms[1]:g} ms. "
        "Rates/phase use 1 ms SD temporal smoothing; heatmaps are raw E-cell rates, with no spatial smoothing.\n"
        "Excitation shares one peak scale; inhibition has its own. Scales are shared between cue figures for each quantity/population. "
        "Shading is an inhibition proxy, not measured gain.\n"
        "Conductances are linear-synapse reconstructions using native weights, decay and delays; native conductance recordings were not retained. "
        "Only pre-switch data are shown."
    )
    fig.text(0.09, 0.025, caption, fontsize=7.3, va="bottom", linespacing=1.7)
    return fig


def present(*, source, run_id=None):
    started = perf_counter()
    analysed = source_run(
        REPO / ".pingstore", source, experiment=recipe.SLUG, stage="analyse"
    )
    pin = analysed.record["inputs"]["compute"]
    computed = source_run(
        REPO / ".pingstore",
        pin["run_id"],
        reference=pin,
        experiment=recipe.SLUG,
        stage="compute",
    )
    settings = analysed.record["execution"]["configuration"]
    numbers = json.loads(analysed.file("numbers.json").read_text())
    if numbers["schema"] != recipe.SCHEMA or numbers["configuration"] != settings:
        raise ValueError("analysis settings or schema differ")
    cfg = computed.record["execution"]["configuration"]
    with np.load(analysed.file("traces.npz")) as data:
        time = data["time_ms"]
        results = {
            c: {
                kind: {name: data[f"{c}_{kind}_{name}"] for name in names}
                for kind, names in dict(
                    rates="ABC", differences="AB", conductances=PATHS, maps="ABC"
                ).items()
            }
            for c in "AB"
        }
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs={"analyse": analysed, "compute": computed},
        configuration=settings,
        run_id=run_id,
    ) as run:
        for c, result in results.items():
            result["plvs"] = numbers["plvs"][c]
            fig = figure(
                cfg,
                time,
                result,
                numbers["scales"],
                cued=c,
                end_ms=settings["display_end_ms"],
                window_ms=settings["window_ms"],
                closeup_ms=settings["closeup_ms"],
            )
            for ext in ["png", "svg"]:
                fig.savefig(run.export / f"cue-{c}.{ext}", dpi=170)
            plt.close(fig)
        shutil.copyfile(computed.file("circuit.svg"), run.export / "circuit.svg")
        write_json_atomic(run.export / "numbers.json", numbers)
        run.record["execution"].update(
            elapsed_seconds=perf_counter() - started, no_simulation=True
        )
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    present(**vars(p.parse_args()))
