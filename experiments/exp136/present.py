"""Render synchrony references and optional signal-routing measurements."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.exp136 import recipe
from experiments.helpers import theme
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run
from snnlab.lang import load_bundle


def draw(spikes, plv, cfg, frequencies):
    """Equal raster sizes; R centred vertically between the two sender rasters."""
    theme.apply()
    senders = ["S1", "S2"]
    fig = plt.figure(figsize=(7.1, 6.0))
    grid = fig.add_gridspec(
        5,
        2,
        height_ratios=[1, 1, 1, 1, 2.3],
        left=0.10,
        right=0.98,
        top=0.88,
        bottom=0.09,
        hspace=1.35,
        wspace=0.35,
    )
    axes = {
        "S1": fig.add_subplot(grid[0:2, 0]),
        "S2": fig.add_subplot(grid[2:4, 0]),
        "R": fig.add_subplot(grid[1:3, 1]),
    }
    bottom = fig.add_subplot(grid[4, :])
    for name in (*senders, "R"):
        ax = axes[name]
        for pop, offset, color in (
            ("E", 0, theme.INK_BLACK),
            ("I", recipe.N_E, theme.DEEP_RED),
        ):
            t, cell = np.nonzero(spikes[f"{name}_{pop}_spikes"][:, 0])
            ax.scatter(
                t * cfg["dt_ms"],
                cell + offset + 1,
                marker="|",
                s=4.0,
                linewidths=0.45,
                color=color,
                rasterized=True,
            )
        ax.axhline(recipe.N_E + 0.5, color=theme.GREY_LIGHT, lw=0.6)
        ax.set(
            ylim=(0, recipe.N_E + recipe.N_I + 1), yticks=[1, 80, 100], ylabel="neuron"
        )
        for window in frequencies[name]:
            value = window["frequency_hz"]
            label = "—" if value is None else f"{value:.1f} Hz"
            ax.annotate(
                label,
                xy=(window["end_ms"], 1),
                xycoords=("data", "axes fraction"),
                xytext=(-2, 4),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=theme.SIZE_ANNOTATION,
                color=theme.INK_BLACK,
                clip_on=False,
            )
    for key, color in [
        (f"{name}R", theme.INK_BLACK if name == "S1" else theme.DEEP_RED)
        for name in senders
    ]:
        bottom.plot(
            plv["time_ms"],
            plv[key],
            color=color,
            lw=1.1,
            label="S1 − R" if key == "S1R" else "S2 − R",
        )
    bottom.set(ylabel="PLV", ylim=(0, 1), yticks=[0, 0.5, 1])
    bottom.set_title("Phase locking (200 ms window)")
    bottom.legend(
        loc="lower right",
        bbox_to_anchor=(1, 1.12),
        ncol=2,
        fontsize=theme.SIZE_ANNOTATION,
        frameon=False,
    )
    for ax in [*axes.values(), bottom]:
        boundaries = sorted(
            {v for _, start, end in cfg["cue_segments"] for v in (start, end)}
        )
        for boundary in boundaries:
            ax.axvline(boundary, color=theme.DEEP_RED, lw=1.2, ls="--")
        ax.set(
            xlim=(0, cfg["duration_ms"]),
            xticks=np.arange(0, cfg["duration_ms"] + 1, cfg.get("segment_ms", 500)),
            xlabel="time (ms)",
        )
    labels = theme.label_panels(
        [axes[n] for n in (*senders, "R")] + [bottom], labels=[*senders, "R", "D"]
    )
    labels[-1].set_x(-0.025)
    return fig


def present(identity, *, run_id=None):
    analysis = source_run(
        REPO / ".pingstore", identity, stage="analyse", experiment=recipe.SLUG
    )
    result = load_json(analysis.file("results.json"))
    if result.get("schema") == "exp136.receiver-analysis/v1":
        from experiments.exp136.refinement_present import present_refinement

        return present_refinement(REPO, identity, run_id=run_id)
    if result.get("schema") == "exp136.routing-search-analysis/v1":
        from experiments.exp136.routing_present import present_search

        return present_search(REPO, identity, run_id=run_id)
    if result.get("schema") == "exp136.attention-analysis/v1":
        from experiments.exp136.attention_present import present_attention

        return present_attention(REPO, identity, run_id=run_id)
    if result.get("schema") != "exp136.fixed-analysis/v1":
        raise ValueError("requires a fixed-setting analysis")
    pin = analysis.record["inputs"]["compute"]
    compute = source_run(
        REPO / ".pingstore",
        pin["run_id"],
        stage="compute",
        experiment=recipe.SLUG,
        reference=pin,
    )
    cfg = result["configuration"]
    history = source_run(
        REPO / ".pingstore",
        recipe.SELECTION_EVIDENCE,
        stage="present",
        experiment=recipe.SLUG,
    )
    result["selection_evidence"] = history.reference
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs={"analysis": analysis, "selection_evidence": history},
        configuration=dict(kind="fixed coupled/zero comparison", plv_window_ms=200),
        run_id=run_id,
    ) as run:
        bundle = load_bundle(
            compute.restore_bundle("network.bundle", run.scratch / "network.bundle")
        )
        if any(p.get("modulation") for p in bundle.graph["projections"]):
            raise ValueError("weights must remain constant")
        for suffix in ("svg", "png"):
            bundle.visualise(
                run.export / f"network-circuit.{suffix}", view="circuit", scale=2
            )
        for condition in cfg["conditions"]:
            with np.load(
                compute.file(f"{condition}--spikes.npz"), allow_pickle=False
            ) as raw:
                spikes = dict(raw)
            with np.load(
                analysis.file(f"{condition}--plv.npz"), allow_pickle=False
            ) as raw:
                plv = dict(raw)
            fig = draw(spikes, plv, cfg, result["conditions"][condition]["frequencies"])
            fig.suptitle(
                f"{'With coupling' if condition == 'coupled' else 'Without coupling'}\nNo cue → cue S1 → cue S2 → no cue (1 s each)",
                fontsize=9,
                fontweight="bold",
            )
            for suffix in ("svg", "png"):
                fig.savefig(run.export / f"continuous-{condition}.{suffix}")
            plt.close(fig)
        write_json_atomic(run.export / "numbers.json", result)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="Fixed paired-trial analysis run."
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)
