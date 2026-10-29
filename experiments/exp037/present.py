"""Render saved exp037 analysis; never simulate, aggregate or publish.

Invoke this stage explicitly from the repository root. Scientific outputs
and input provenance belong to its independently completed Pingstore run.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

import argparse
import time

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from experiments import theme
from experiments.exp037 import recipe
from experiments.exp037.analyse import measurement
from experiments.exp037.custom import evidence, inputs
from experiments.exp037.custom.network_artifacts import copy_circuits
from experiments.exp037.recipe import EI_RASTER_N_E_PLOT, EI_RASTER_N_I_PLOT, MODELS
from matplotlib.patches import Patch
from pingstore.library import PingstoreError, load_json, write_json_atomic
from snnlab import viz

DEFAULT_FORMATS = ("svg", "pdf")


def schematic_diagrams():
    """Lay out the documented model; these are illustrations, not execution evidence."""
    specifications = [
        {
            "name": "coba",
            "title": "COBA classifier",
            "nodes": [
                ("X", "Poisson spikes", "input", (-360, 0)),
                ("E", "Excitatory neurons", "population", (0, 0)),
                ("R", "Class readout", "population", (360, 0)),
            ],
            "edges": [("X", "E", "excitatory"), ("E", "R", "signal")],
        },
        {
            "name": "ping",
            "title": "PING classifier",
            "nodes": [
                ("X", "Poisson spikes", "input", (-360, 0)),
                ("E", "Excitatory neurons", "population", (0, 0)),
                ("I", "Inhibitory neurons", "population", (0, 300)),
                ("R", "Class readout", "population", (360, 0)),
            ],
            "edges": [
                ("X", "E", "excitatory"),
                ("E", "I", "excitatory"),
                ("I", "E", "inhibitory"),
                ("E", "R", "signal"),
            ],
        },
    ]
    diagrams = {}
    ports = dict(
        top_left=(0.25, 0),
        top_right=(0.75, 0),
        bottom_left=(0.25, 1),
        bottom_right=(0.75, 1),
    )
    for spec in specifications:
        visual = viz.Diagram(
            name="exp037_" + spec["name"],
            title=spec["title"],
            nodes=tuple(
                viz.DiagramNode(identifier, identifier, detail, "", kind=kind)
                for identifier, detail, kind, _ in spec["nodes"]
            ),
            edges=tuple(
                viz.DiagramEdge(
                    source,
                    target,
                    id=source + "_to_" + target,
                    role="signal" if role == "average" else role,
                )
                for source, target, role in spec["edges"]
            ),
            metadata=dict(
                view="illustrative documented model; no historical bundle reconstructed"
            ),
        )
        positions = {
            identifier: position for identifier, _, _, position in spec["nodes"]
        }
        routes = {}
        for source, target, role in spec["edges"]:
            label = (
                r"$\langle V-E_L\rangle_t$"
                if role == "average"
                else rf"$W_{{{target},{source}}}$"
            )
            if (source, target) == ("E", "I"):
                route = dict(source_port="bottom_left", target_port="top_left")
            elif (source, target) == ("I", "E"):
                route = dict(source_port="top_right", target_port="bottom_right")
            elif source == "J":
                route = dict(source_port="top", target_port="bottom")
            elif positions[source][0] == positions[target][0]:
                route = dict(source_port="bottom", target_port="top")
            elif positions[source][0] > positions[target][0]:
                route = dict(source_port="left", target_port="right")
            else:
                route = dict(source_port="right", target_port="left")
            routes[source + "_to_" + target] = viz.EdgeLayout(
                label=viz.DiagramLabel(label), **route
            )
        layout = viz.DiagramLayout(
            nodes={
                identifier: viz.NodeLayout(
                    position=position,
                    label=f"${identifier}$",
                    min_size=(240, 160),
                    ports=ports,
                )
                for identifier, position in positions.items()
            },
            edges=routes,
            defaults=viz.LayoutDefaults(
                node=viz.NodeLayout(content_align="center", padding=16), gap=30
            ),
        )
        with matplotlib.rc_context({"savefig.bbox": None}):
            diagrams[spec["name"]] = viz.resolve_diagram(
                visual, layout=layout, theme=viz.DiagramTheme(secondary_size=13)
            )
    return diagrams


def draw_schematics():
    """Regenerate report-local SVG illustrations without executing scientific stages."""
    destination = Path(__file__).parent / "report" / "images"
    destination.mkdir(exist_ok=True)
    with matplotlib.rc_context({"savefig.bbox": None}):
        for name, diagram in schematic_diagrams().items():
            viz.render_diagram(diagram, destination / (name + "-network.svg"))


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


def present(identity, *, run_id=None):

    source = inputs.source(REPO, identity, "analyse")
    refs = source.record["inputs"]
    if set(refs) != {"compute", "bank"}:
        raise PingstoreError("analysis must pin compute and bank")
    compute = inputs.source(
        REPO, refs["compute"]["run_id"], "compute", reference=refs["compute"]
    )
    cfg, bank, contract = inputs.compute_evidence(REPO, compute)
    result = load_json(source.export / "results.json")
    if (
        refs["bank"] != bank.reference
        or result.get("schema")
        != ("exp037.analysis/v2" if recipe.relative(cfg) else "exp037.analysis/v1")
        or result.get("recipe") != cfg
        or result.get("measurement") != measurement(cfg)
        or source.record["execution"].get("configuration") != measurement(cfg)
        or result.get("checkpoint_policy") != recipe.CHECKPOINT_POLICY
        or result.get("checkpoint_provenance") != contract["checkpoints"]
    ):
        raise PingstoreError("analysis evidence or bank pin differs")
    resolved = evidence.resolved_jobs(
        cfg, contract, lambda j: compute.file(j["path"], "metrics.json")
    )
    jobs = [j for j in resolved if "sample_index" in j]
    if result.get("rasters") != [
        {"job": j, "file": f"raster-{i:03d}.npz"} for i, j in enumerate(jobs)
    ]:
        raise PingstoreError("analysis raster grid differs")
    expected_grid = [
        {k: j[k] for k in ("model", "seed", "mode", "level")}
        for j in recipe.jobs(cfg)
        if j["kind"] == "sweep"
    ]
    if [
        {k: r.get(k) for k in ("model", "seed", "mode", "level")}
        for r in result.get("perturbation", [])
    ] != expected_grid:
        raise PingstoreError("analysis perturbation grid incomplete")
    if (
        len(result.get("baseline_results", [])) != 36
        or len(result.get("frontier_summary", [])) != 12
        or len(result.get("perturbation_summary", []))
        != len(expected_grid) // len(cfg["seeds"])
    ):
        raise PingstoreError("analysis summary grid incomplete")
    rasters = []
    for entry in result["rasters"]:
        with np.load(source.export / entry["file"], allow_pickle=False) as raw:
            rasters.append(
                {
                    k: np.array(raw[k]) if raw[k].ndim else raw[k].item()
                    for k in raw.files
                }
            )
    started = time.monotonic()
    with inputs.execution(
        REPO,
        "present",
        sources={"analysis": source},
        run_id=run_id,
        configuration={
            "schema": "exp037.presentation/v4",
            "labels": result.get("addition_axis", "reference_image_percent"),
            "range": "complete saved perturbation grid",
            "figure_run_stamps": False,
        },
    ) as run:
        theme.set_paper_mode(True)
        out, rid = run.export, run.run_id
        plot_perturbation_curves(result["plot_data"], out / "perturbation_curves", rid)
        for model in recipe.MODELS:
            for mode in ("drop", "add"):
                samples = [
                    r for r in rasters if r["model"] == model and r["mode"] == mode
                ]
                plot_perturbation_rasters(
                    samples,
                    out / f"perturb_rasters__{mode}__{model}",
                    rid,
                    level_fmt="p(drop) = {level:.1f}"
                    if mode == "drop"
                    else "add = {level:g}%"
                    if recipe.relative(cfg)
                    else "r(add) = {level:g} Hz",
                    title="",
                )
        plot_combined_rasters(
            rasters, out / "perturbation_rasters", relative=recipe.relative(cfg)
        )
        duration = time.monotonic() - started
        write_json_atomic(
            out / "numbers.json",
            {
                **{key: value for key, value in result.items() if key != "rasters"},
                "illustrative_labels": [r["label"] for r in rasters],
                "notebook_run_id": rid,
                "run_id": rid,
                "duration_s": duration,
                "duration": format_duration(duration),
                "git_sha": run.record["provenance"]["git_commit"],
                "circuit_files": copy_circuits(run),
            },
        )
    return run.run_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    a = p.parse_args()
    try:
        present(a.source, run_id=a.run_id)
    except (PingstoreError, OSError, KeyError, ValueError) as exc:
        p.exit(1, f"exp037 present: {exc}\n")


MODEL_COLORS = {"coba": theme.DEEP_RED, "ping": theme.INK_BLACK}
MODEL_MARKERS = {"coba": "s", "ping": "D"}


def plot_perturbation_rasters(
    samples: list[dict], out_path: Path, run_id: str, level_fmt: str, title: str
) -> None:
    """Stacked single-trial rasters across perturbation levels for one mode."""
    theme.apply()
    n = len(samples)
    n_e = EI_RASTER_N_E_PLOT
    n_i = EI_RASTER_N_I_PLOT
    gap = 6
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(6.9, 3.88),
        sharex=True,
        gridspec_kw={"hspace": 0.18},
    )
    if n == 1:
        axes = [axes]
    for i, (ax, s) in enumerate(zip(axes, samples, strict=True)):
        ax.scatter(
            s["e_t"],
            s["e_n"],
            s=2.0,
            c=theme.INK_BLACK,
            marker="|",
            linewidths=0.4,
        )
        ax.scatter(
            s["i_t"],
            s["i_n"],
            s=2.0,
            c=theme.DEEP_RED,
            marker="|",
            linewidths=0.4,
        )
        ax.set_ylim(-2, n_e + n_i + gap + 2)
        ax.set_yticks([n_e / 2, n_e + gap + n_i / 2])
        ax.set_yticklabels(["E", "I"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, s["t_ms"])
        ax.text(
            1.012,
            0.5,
            level_fmt.format(level=s["level"]) + f"\nE = {s['e_rate_hz']:.1f} Hz",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=theme.SIZE_LABEL,
        )
        # The writeup caption carries the takeaway rather than a figure title.
        if i < n - 1:
            ax.tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("time (ms)")
    theme.label_panels(axes)
    fig.subplots_adjust(left=0.07, right=0.78, bottom=0.12, top=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


def plot_perturbation_curves(
    data: dict,
    out_path: Path,
    run_id: str,
    *,
    axes=None,
) -> None:
    """Plot saved across-seed means and sample-SD envelopes."""
    theme.apply()
    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3), sharey=True)
    fig = axes[0].figure
    use_pct = data["use_pct"]

    # Left panel: drop (as % of spikes dropped)
    ax_drop = axes[0]
    for model in MODELS:
        row = data["panels"]["drop"][model]
        xs, means, lo, hi = (row[k] for k in ("x", "mean", "lo", "hi"))
        ax_drop.plot(
            xs,
            means,
            marker=MODEL_MARKERS[model],
            markersize=5,
            linewidth=1.4,
            color=MODEL_COLORS[model],
            label=model.upper(),
            zorder=3 if model == "coba" else 2,
        )
        ax_drop.fill_between(
            xs,
            lo,
            hi,
            color=MODEL_COLORS[model],
            alpha=0.15,
            linewidth=0,
        )
    ax_drop.set_xlabel("Spike-deletion probability (%)", fontsize=theme.SIZE_LABEL)
    ax_drop.set_title(
        "Drop — Bernoulli deletion", fontsize=theme.SIZE_LABEL, loc="left", pad=4
    )
    ax_drop.set_xlim(-2, 102)
    ax_drop.axhline(10.0, ls="--", color=theme.MUTED, lw=0.7, alpha=0.6)
    ax_drop.text(
        0.02,
        12,
        "chance",
        transform=ax_drop.get_yaxis_transform(),
        fontsize=theme.SIZE_ANNOTATION,
        color=theme.MUTED,
        va="bottom",
    )

    # Right panel: add
    ax_add = axes[1]
    if use_pct:
        for model in MODELS:
            row = data["panels"]["add"][model]
            xs, means, lo, hi = (row[k] for k in ("x", "mean", "lo", "hi"))
            ax_add.plot(
                xs,
                means,
                marker=MODEL_MARKERS[model],
                markersize=5,
                linewidth=1.4,
                color=MODEL_COLORS[model],
                label=model.upper(),
                zorder=3 if model == "coba" else 2,
            )
            ax_add.fill_between(
                xs,
                lo,
                hi,
                color=MODEL_COLORS[model],
                alpha=0.15,
                linewidth=0,
            )
        ax_add.set_xlabel(
            "Nominal added rate / baseline E rate (%)"
            if data.get("relative_test_baseline")
            else "Nominal added rate / reference E rate (%)",
            fontsize=theme.SIZE_LABEL,
        )
        ax_add.set_title(
            "Add — Bernoulli insertion",
            fontsize=theme.SIZE_LABEL,
            loc="left",
            pad=4,
        )
        maximum = max(max(row["x"]) for row in data["panels"]["add"].values())
        margin = max(1.0, 0.03 * maximum)
        ax_add.set_xlim(-margin, maximum + margin)
    else:
        for model in MODELS:
            row = data["panels"]["add"][model]
            xs, means, lo, hi = (row[k] for k in ("x", "mean", "lo", "hi"))
            ax_add.plot(
                xs,
                means,
                marker=MODEL_MARKERS[model],
                markersize=5,
                linewidth=1.4,
                color=MODEL_COLORS[model],
                label=model.upper(),
                zorder=3 if model == "coba" else 2,
            )
            ax_add.fill_between(
                xs,
                lo,
                hi,
                color=MODEL_COLORS[model],
                alpha=0.15,
                linewidth=0,
            )
        ax_add.set_xlabel("Nominal added rate (Hz / neuron)", fontsize=theme.SIZE_LABEL)
        ax_add.set_title(
            "Add — Bernoulli insertion",
            fontsize=theme.SIZE_LABEL,
            loc="left",
            pad=4,
        )
    ax_add.axhline(10.0, ls="--", color=theme.MUTED, lw=0.7, alpha=0.6)

    for ax in axes:
        ax.set_ylim(0, 100)
        ax.tick_params(labelsize=theme.SIZE_TICK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.grid(True, axis="y", alpha=0.15, linewidth=0.5)
    ax_drop.set_ylabel("Test accuracy (%)", fontsize=theme.SIZE_LABEL)
    legend_handles, legend_labels = ax_add.get_legend_handles_labels()
    legend_handles.append(
        Patch(
            facecolor=theme.INK_BLACK,
            edgecolor="none",
            alpha=0.15,
            label="±1 SD band",
        )
    )
    legend_labels.append("±1 SD band")
    ax_add.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        fontsize=theme.SIZE_LEGEND,
        frameon=False,
    )
    # The writeup caption carries the takeaway rather than a figure title.
    if standalone:
        theme.label_panels(axes)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_figure(fig, out_path)  # line/curve plot: SVG + PDF
        plt.close(fig)


def plot_combined_rasters(
    samples: list[dict], out_path: Path, *, relative: bool
) -> None:
    """Four model/intervention panels, each containing three saved trials."""
    theme.apply()
    fig = plt.figure(figsize=(7.0, 4.8))
    outer = fig.add_gridspec(
        2, 2, left=0.065, right=0.98, bottom=0.065, top=0.95, wspace=0.19, hspace=0.35
    )
    for panel, (model, mode) in enumerate(
        (("ping", "drop"), ("ping", "add"), ("coba", "drop"), ("coba", "add"))
    ):
        rows = sorted(
            (s for s in samples if s["model"] == model and s["mode"] == mode),
            key=lambda s: s["level"],
        )
        grid = outer[panel // 2, panel % 2].subgridspec(len(rows), 1, hspace=0.42)
        for index, s in enumerate(rows):
            ax = fig.add_subplot(grid[index])
            for population, colour in (("e", theme.INK_BLACK), ("i", theme.DEEP_RED)):
                ax.scatter(
                    s[f"{population}_t"],
                    s[f"{population}_n"],
                    s=1.5,
                    c=colour,
                    marker="|",
                    linewidths=0.35,
                    rasterized=True,
                )
            ax.set_xlim(0, s["t_ms"])
            ax.set_ylim(-2, EI_RASTER_N_E_PLOT + EI_RASTER_N_I_PLOT + 8)
            ax.set_yticks(
                [
                    EI_RASTER_N_E_PLOT / 2,
                    EI_RASTER_N_E_PLOT + 6 + EI_RASTER_N_I_PLOT / 2,
                ],
                ["E", "I"],
            )
            ax.tick_params(axis="y", length=0, labelsize=8)
            ax.tick_params(axis="x", labelsize=8, labelbottom=index == len(rows) - 1)
            dose = (
                f"Delete {100 * s['level']:g}%"
                if mode == "drop"
                else f"Add {s['level']:g}{'%' if relative else ' Hz'}"
            )
            ax.set_title(
                f"{dose}   ·   E = {s['e_rate_hz']:.1f} Hz",
                loc="left",
                fontsize=8,
                pad=3,
            )
            if index == 0:
                ax.text(
                    0,
                    1.45,
                    f"{'ABCD'[panel]}  {model.upper()} — {'deletion' if mode == 'drop' else 'addition'}",
                    transform=ax.transAxes,
                    fontsize=10,
                    fontweight="bold",
                    va="bottom",
                )
            if index == len(rows) - 1 and panel >= 2:
                ax.set_xlabel("Time (ms)", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_path, formats=("svg", "pdf"))
    plt.close(fig)


if __name__ == "__main__":
    if "--draw-schematics" in sys.argv[1:]:
        parser = argparse.ArgumentParser(
            description="Render illustrative model diagrams only."
        )
        parser.add_argument("--draw-schematics", action="store_true", required=True)
        parser.parse_args()
        draw_schematics()
    else:
        main()
