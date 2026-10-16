"""Show validation search and independent routing confirmation in house style."""

import shutil

import matplotlib.pyplot as plt
import numpy as np
from experiments.exp136 import recipe
from experiments.helpers import theme
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def read_analysis(repo, identity, reference=None):
    source = source_run(
        repo / ".pingstore",
        identity,
        stage="analyse",
        experiment=recipe.SLUG,
        reference=reference,
    )
    result = load_json(source.file("results.json"))
    if result.get("schema") != "exp136.routing-search-analysis/v1":
        raise ValueError("requires a routing search analysis")
    return source, result


def draw_search(coarse, fine, confirmation, predictions):
    theme.apply()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.6))
    fig.subplots_adjust(
        left=0.1, right=0.98, bottom=0.1, top=0.91, wspace=0.36, hspace=0.65
    )
    for result, marker, color, label in (
        (coarse, "o", theme.GREY_MID, "Coarse"),
        (fine, "^", theme.DEEP_RED, "Fine"),
    ):
        local = [
            min(v["mean_r2"] for v in row["result"]["local_encoding"].values())
            for row in result["candidates"]
        ]
        gain = [
            min(row["result"]["conditions"]["coupled"]["mean_cue_gain"])
            for row in result["candidates"]
        ]
        axes[0, 0].scatter(
            local, gain, marker=marker, c=color, s=19, alpha=0.8, label=label
        )
    axes[0, 0].axvline(0.1, c=theme.GREY_LIGHT, ls="--", lw=0.8)
    axes[0, 0].axhline(0, c=theme.GREY_LIGHT, lw=0.8)
    axes[0, 0].set(
        xlabel="Weaker sender decoding (R²)",
        ylabel="Smaller cue benefit (ΔR²)",
        title="Validation search",
    )
    axes[0, 0].legend(frameon=False, fontsize=7)
    chosen = fine["selected"]["result"]
    axes[0, 0].scatter(
        [min(v["mean_r2"] for v in chosen["local_encoding"].values())],
        [min(chosen["conditions"]["coupled"]["mean_cue_gain"])],
        marker="*",
        s=105,
        facecolors="none",
        edgecolors=theme.INK_BLACK,
        linewidths=1.2,
        zorder=5,
    )
    result = confirmation["selected"]["result"]
    names = ["Coupled", "Zero", "Scrambled"]
    for index, condition in enumerate(("coupled", "zero", "jittered")):
        scores = np.asarray(result["conditions"][condition]["r2"])
        for sender, color in enumerate((theme.INK_BLACK, theme.DEEP_RED)):
            x = index * 3 + sender
            for cue, marker, offset in ((sender, "o", -0.13), (1 - sender, "x", 0.13)):
                values = scores[:, cue, sender]
                axes[0, 1].scatter(
                    x + offset + np.linspace(-0.035, 0.035, len(values)),
                    values,
                    c=color,
                    marker=marker,
                    s=15,
                    alpha=0.6,
                )
                axes[0, 1].plot(
                    [x + offset - 0.08, x + offset + 0.08],
                    [values.mean()] * 2,
                    c=color,
                    lw=1.6,
                )
    axes[0, 1].axhline(0, c=theme.GREY_LIGHT, lw=0.8)
    axes[0, 1].set(
        xticks=[0.5, 3.5, 6.5],
        xticklabels=names,
        ylabel="Receiver reconstruction (R²)",
        title="Fresh trials: ● cued, × uncued",
    )
    for sender, color in enumerate((theme.INK_BLACK, theme.DEEP_RED)):
        ax = axes[1, sender]
        keep = (
            (predictions["split"] == 1)
            & (predictions["trial"] == 0)
            & (predictions["order"] == 0)
        )
        order = np.argsort(predictions["time_ms"][keep])
        time = predictions["time_ms"][keep][order] / 1000
        target = predictions["targets"][keep, sender][order]
        decoded = predictions["coupled"][keep, sender][order]
        # Leave excluded cue-transition samples blank rather than bridge the gap.
        for segment, (start, end) in enumerate(((1, 2), (2, 3))):
            mask = (time >= start) & (time < end)
            ax.plot(
                time[mask],
                target[mask],
                c=theme.GREY_LIGHT,
                lw=1.3,
                label="Input signal" if segment == 0 else None,
            )
            ax.plot(
                time[mask],
                decoded[mask],
                c=color,
                lw=1,
                label="Decoded from R" if segment == 0 else None,
            )
        ax.axvline(2, c=theme.DEEP_RED, ls="--", lw=0.8)
        ax.set(
            xlim=(1, 3),
            xlabel="Observation time (s)",
            ylabel="Signal value",
            title=f"S{sender + 1}: cue S1 → cue S2",
        )
        ax.legend(frameon=False, fontsize=7, loc="lower left")
    theme.label_panels(axes.ravel())
    fig.suptitle(
        "Signal selection after coarse and fine searches",
        fontsize=10,
        fontweight="bold",
    )
    return fig


def present_search(repo, identity, *, run_id=None):
    analysis, confirmation = read_analysis(repo, identity)
    if confirmation["level"] != "confirm":
        raise ValueError(
            "present requires fresh-trial confirmation, not validation-selected examples"
        )
    cp = analysis.record["inputs"]["compute"]
    compute = source_run(
        repo / ".pingstore",
        cp["run_id"],
        stage="compute",
        experiment=recipe.SLUG,
        reference=cp,
    )
    fp = compute.record["inputs"]["selection"]
    fine_source, fine = read_analysis(repo, fp["run_id"], fp)
    fcp = fine_source.record["inputs"]["compute"]
    fine_compute = source_run(
        repo / ".pingstore",
        fcp["run_id"],
        stage="compute",
        experiment=recipe.SLUG,
        reference=fcp,
    )
    coarse_pin = fine_compute.record["inputs"]["selection"]
    coarse_source, coarse = read_analysis(repo, coarse_pin["run_id"], coarse_pin)
    baseline = source_run(
        repo / ".pingstore",
        confirmation["baseline_presentation"],
        stage="present",
        experiment=recipe.SLUG,
    )
    with stage_run(
        repo,
        recipe.SLUG,
        "present",
        inputs=dict(
            analysis=analysis,
            fine=fine_source,
            coarse=coarse_source,
            synchrony_reference=baseline,
        ),
        configuration=dict(
            kind="coarse/fine validation and fresh signal-routing confirmation"
        ),
        run_id=run_id,
    ) as run:
        for condition in ("coupled", "zero"):
            for suffix in ("svg", "png"):
                name = f"continuous-{condition}.{suffix}"
                shutil.copy2(baseline.file(name), run.export / name)
        for suffix in ("svg", "png"):
            name = f"network-circuit.{suffix}"
            shutil.copy2(
                compute.file(f"network.bundle/reports/circuit.{suffix}"),
                run.export / name,
            )
        with np.load(
            analysis.file("confirmation--predictions.npz"), allow_pickle=False
        ) as raw:
            figure = draw_search(coarse, fine, confirmation, dict(raw))
        for suffix in ("svg", "png"):
            figure.savefig(run.export / f"routing-search.{suffix}", dpi=180)
        plt.close(figure)
        write_json_atomic(
            run.export / "numbers.json",
            dict(
                coarse=coarse,
                fine=fine,
                confirmation=confirmation,
                synchrony=load_json(baseline.file("numbers.json")),
            ),
        )
    return run.run_id
