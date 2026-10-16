"""Readout audit, targeted receiver search and untouched-trial confirmation."""

import shutil

import matplotlib.pyplot as plt
import numpy as np
from experiments.exp136 import recipe
from experiments.helpers import theme
from matplotlib.patches import Rectangle
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run


def draw_refinement(audit, search, final):
    theme.apply()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.7))
    fig.subplots_adjust(
        left=0.12, right=0.98, bottom=0.1, top=0.9, wspace=0.4, hspace=0.65
    )
    for mode, color, marker in [
        ("population", theme.INK_BLACK, "o"),
        ("individual", theme.DEEP_RED, "^"),
    ]:
        selected = []
        for width in (25, 50, 100, 200):
            rows = [
                r
                for r in audit["candidates"]
                if r["readout"]["feature_mode"] == mode
                and r["readout"]["bin_ms"] == width
            ]
            selected.append(max(rows, key=lambda r: tuple(r["quality"])))
        axes[0, 0].plot(
            [r["readout"]["bin_ms"] for r in selected],
            [r["quality"][2] for r in selected],
            c=color,
            marker=marker,
            ms=4,
            lw=1,
            label="Population" if mode == "population" else "Individual neurons",
        )
    chosen = audit["selected"]
    axes[0, 0].scatter(
        [chosen["readout"]["bin_ms"]],
        [chosen["quality"][2]],
        marker="*",
        s=100,
        facecolors="none",
        edgecolors=theme.INK_BLACK,
        zorder=5,
    )
    axes[0, 0].axhline(0, c=theme.GREY_LIGHT, lw=0.7)
    axes[0, 0].set(
        xlabel="Readout window (ms)",
        ylabel="Weaker cued R²",
        title="Readout validation",
        xticks=[25, 50, 100, 200],
    )
    axes[0, 0].legend(frameon=False, fontsize=7)
    drives = sorted(
        {r["network"]["receiver_background_hz"] for r in search["candidates"]}
    )
    coupling = sorted(
        {r["network"]["forward_strength_us"] for r in search["candidates"]}
    )
    grid = np.empty((len(drives), len(coupling)))
    for row in search["candidates"]:
        grid[
            drives.index(row["network"]["receiver_background_hz"]),
            coupling.index(row["network"]["forward_strength_us"]),
        ] = row["quality"][2]
    axes[0, 1].imshow(
        grid,
        cmap="Greys",
        aspect="auto",
        vmin=min(0, grid.min()),
        vmax=max(0.001, grid.max()),
    )
    for i in range(3):
        for j in range(3):
            shade = (grid[i, j] - min(0, grid.min())) / (
                max(0.001, grid.max()) - min(0, grid.min())
            )
            axes[0, 1].text(
                j,
                i,
                f"{grid[i, j]:+.3f}",
                ha="center",
                va="center",
                fontsize=8,
                c=theme.PAPER if shade > 0.55 else theme.INK_BLACK,
            )
    n = search["selected"]["network"]
    axes[0, 1].add_patch(
        Rectangle(
            (
                coupling.index(n["forward_strength_us"]) - 0.48,
                drives.index(n["receiver_background_hz"]) - 0.48,
            ),
            0.96,
            0.96,
            fill=False,
            edgecolor=theme.DEEP_RED,
            lw=2,
        )
    )
    axes[0, 1].set(
        xticks=range(3),
        xticklabels=[f"{k:.3f}" for k in coupling],
        yticks=range(3),
        yticklabels=drives,
        xlabel="Coupling (μS per pathway)",
        ylabel="R input (Hz/channel)",
        title="Receiver validation: weaker R²",
    )
    result = final["selected"]["result"]
    for ax, kind, title in [
        (axes[1, 0], "score", "Fresh cued reconstruction"),
        (axes[1, 1], "gain", "Fresh cue benefit"),
    ]:
        for index, condition in enumerate(("coupled", "zero", "jittered")):
            for sender, color in enumerate((theme.INK_BLACK, theme.DEEP_RED)):
                if kind == "score":
                    values = np.asarray(result["conditions"][condition]["r2"])[
                        :, sender, sender
                    ]
                else:
                    values = np.asarray(result["conditions"][condition]["cue_gain"])[
                        :, sender
                    ]
                x = index * 3 + sender
                ax.scatter(
                    x + np.linspace(-0.08, 0.08, len(values)),
                    values,
                    c=color,
                    s=15,
                    alpha=0.65,
                )
                ax.plot([x - 0.18, x + 0.18], [values.mean()] * 2, c=color, lw=1.5)
        ax.axhline(0, c=theme.GREY_LIGHT, lw=0.7)
        ax.set(
            xticks=[0.5, 3.5, 6.5],
            xticklabels=["Coupled", "Zero", "Scrambled"],
            ylabel="R²" if kind == "score" else "ΔR²",
            title=title,
        )
    theme.label_panels(axes.ravel())
    fig.suptitle(
        "Refining signal readout and receiver drive", fontsize=10, fontweight="bold"
    )
    return fig


def present_refinement(repo, identity, *, run_id=None):
    def analysis(identity, reference=None):
        src = source_run(
            repo / ".pingstore",
            identity,
            stage="analyse",
            experiment=recipe.SLUG,
            reference=reference,
        )
        return src, load_json(src.file("results.json"))

    final_source, final = analysis(identity)
    if (
        final.get("schema") != "exp136.receiver-analysis/v1"
        or final["level"] != "confirm"
    ):
        raise ValueError("presentation requires untouched-trial receiver confirmation")

    def upstream_compute(src):
        pin = src.record["inputs"]["compute"]
        return source_run(
            repo / ".pingstore",
            pin["run_id"],
            stage="compute",
            experiment=recipe.SLUG,
            reference=pin,
        )

    final_compute = upstream_compute(final_source)
    pin = final_compute.record["inputs"]["selection"]
    search_source, search = analysis(pin["run_id"], pin)
    search_compute = upstream_compute(search_source)
    pin = search_compute.record["inputs"]["selection"]
    audit_source, audit = analysis(pin["run_id"], pin)
    baseline = source_run(
        repo / ".pingstore",
        final["baseline_presentation"],
        stage="present",
        experiment=recipe.SLUG,
    )
    with stage_run(
        repo,
        recipe.SLUG,
        "present",
        inputs=dict(
            analysis=final_source,
            receiver_search=search_source,
            readout_audit=audit_source,
            reference=baseline,
        ),
        configuration=dict(kind="readout and receiver refinement with fresh tests"),
        run_id=run_id,
    ) as run:
        for name in ("continuous-coupled", "continuous-zero", "routing-search"):
            for suffix in ("svg", "png"):
                shutil.copy2(
                    baseline.file(f"{name}.{suffix}"), run.export / f"{name}.{suffix}"
                )
        for suffix in ("svg", "png"):
            shutil.copy2(
                final_compute.file(f"network.bundle/reports/circuit.{suffix}"),
                run.export / f"network-circuit.{suffix}",
            )
        figure = draw_refinement(audit, search, final)
        for suffix in ("svg", "png"):
            figure.savefig(run.export / f"receiver-refinement.{suffix}", dpi=180)
        plt.close(figure)
        write_json_atomic(
            run.export / "numbers.json",
            dict(
                readout_audit=audit,
                receiver_search=search,
                confirmation=final,
                reference=load_json(baseline.file("numbers.json")),
            ),
        )
    return run.run_id
