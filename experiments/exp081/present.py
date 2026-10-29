"""Render completed exp081 analysis; never simulate, analyse or materialize.

Invoke this stage explicitly from the repository root. Scientific outputs
and input provenance belong to its independently completed Pingstore run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
from snnlab import viz

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "libraries")]


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp081 import recipe
from experiments.exp081.custom import inputs
from pingstore.library import PingstoreError, load_json, stage_run, write_json_atomic


def schematic_diagrams():
    """Lay out the documented model; these are illustrations, not execution evidence."""
    specifications = [
        {
            "name": "membrane",
            "title": "Nonspiking membrane filter",
            "nodes": [
                ("X", "Pixel spike events", "input", (0, 0)),
                ("M", "Nonspiking membrane", "population", (420, 0)),
            ],
            "edges": [("X", "M", "excitatory")],
        }
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
            name="exp081_" + spec["name"],
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


def plot_moments(
    empirical_mean: np.ndarray,
    empirical_sd: np.ndarray,
    cfg: dict,
    output: Path,
) -> None:
    theme.apply()
    colors = (theme.INK_BLACK, theme.DEEP_RED, theme.ELECTRIC_CYAN)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25), constrained_layout=True)
    for index, (probe, color) in enumerate(zip(cfg["probes_uS"], colors, strict=True)):
        axes[0].plot(
            cfg["input_rate_grid_hz"],
            empirical_mean[index],
            color=color,
            label=f"{probe:g} μS",
        )
        axes[1].plot(cfg["input_rate_grid_hz"], empirical_sd[index], color=color)
    axes[0].set(title="Mean feature", ylabel="Mean feature z (mV)")
    axes[1].set(title="Feature SD", ylabel="Feature SD (mV)")
    for axis in axes:
        axis.set(xlabel="Input rate (Hz)", xlim=(0, 25))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.14)
    axes[0].legend(frameon=False)
    theme.label_panels(axes)
    fig.savefig(output / "empirical_moments.svg", metadata={"Date": None})
    plt.close(fig)


def plot_distributions(histograms: dict, cfg: dict, output: Path) -> None:
    theme.apply()
    fig, axes = plt.subplots(
        1, 3, figsize=(6.5, 2.75), constrained_layout=True, sharex=True, sharey=True
    )
    bins = histograms["edges_mV"]
    for axis, rate, probability in zip(
        axes, cfg["distribution_rates_hz"], histograms["probability"], strict=True
    ):
        axis.bar(
            bins[:-1],
            probability,
            width=np.diff(bins),
            align="edge",
            color=theme.INK_BLACK,
            alpha=0.72,
        )
        axis.set_title(f"{rate:g} Hz")
        axis.set_xlabel("Feature z (mV)")
        axis.set_yscale("log")
        axis.set_ylim(0.5 / cfg["distribution_draws"], 1.0)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Probability per bin (log scale)")
    theme.label_panels(axes)
    fig.savefig(output / "response_distributions.svg", metadata={"Date": None})
    plt.close(fig)


def plot_frequency_response(responses: dict, cfg: dict, output: Path) -> None:
    theme.apply()
    colors = (theme.INK_BLACK, theme.DEEP_RED, theme.ELECTRIC_CYAN)
    fig, axes = plt.subplots(
        1, 2, figsize=(6.5, 3.25), constrained_layout=True, sharey=True
    )
    for index, (rate, color) in enumerate(
        zip(cfg["frequency_response_rates_hz"], colors, strict=True)
    ):
        axes[0].semilogx(
            responses["frequency_hz"],
            responses["unaveraged_db"][index],
            color=color,
            label=f"{rate:g} Hz drive",
        )
        axes[1].semilogx(
            responses["frequency_hz"], responses["averaged_db"][index], color=color
        )
    axes[0].set(
        title="Synapse + membrane", ylabel="Magnitude relative to low-drive DC (dB)"
    )
    axes[1].set(title=f"After {cfg['presentation_ms']:g} ms averaging")
    for axis in axes:
        axis.set(xlabel="Frequency (Hz)", ylim=(-90, 4))
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=7)
    theme.label_panels(axes)
    fig.savefig(output / "frequency_response.svg", metadata={"Date": None})
    plt.close(fig)


def plot_comparison(
    empirical_mean: np.ndarray,
    empirical_sd: np.ndarray,
    analytical_mean: np.ndarray,
    analytical_sd: np.ndarray,
    cfg: dict,
    output: Path,
) -> None:
    theme.apply()
    colors = (theme.INK_BLACK, theme.DEEP_RED, theme.ELECTRIC_CYAN)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25), constrained_layout=True)
    for index, (probe, color) in enumerate(zip(cfg["probes_uS"], colors, strict=True)):
        axes[0].plot(
            cfg["input_rate_grid_hz"],
            analytical_mean[index],
            color=color,
            label=f"{probe:g} μS",
        )
        axes[0].scatter(
            cfg["input_rate_grid_hz"],
            empirical_mean[index],
            s=7,
            color=color,
            alpha=0.28,
            edgecolors="none",
        )
        axes[1].plot(cfg["input_rate_grid_hz"], analytical_sd[index], color=color)
        axes[1].scatter(
            cfg["input_rate_grid_hz"],
            empirical_sd[index],
            s=7,
            color=color,
            alpha=0.28,
            edgecolors="none",
        )
    axes[0].set(title="Mean feature", ylabel="Mean feature z (mV)")
    axes[1].set(title="Feature SD", ylabel="Feature SD (mV)")
    for axis in axes:
        axis.set(xlabel="Input rate (Hz)", xlim=(0, 25))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.14)
    axes[0].legend(frameon=False)
    theme.label_panels(axes)
    fig.savefig(output / "analytical_empirical.svg", metadata={"Date": None})
    plt.close(fig)


def load_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as retained:
        return {key: retained[key] for key in retained.files}


def present(identity: str, *, run_id: str | None = None) -> str:
    analysis = inputs.source(REPO, identity, "analyse")
    cfg = inputs.configuration(analysis)
    refs = analysis.record["inputs"]
    if set(refs) != {"compute"}:
        raise PingstoreError("exp081 analysis must pin exactly one compute input")
    compute = inputs.source(
        REPO, refs["compute"]["run_id"], "compute", reference=refs["compute"]
    )
    if inputs.configuration(compute) != cfg or compute.record["inputs"]:
        raise PingstoreError("analysis recipe or compute lineage disagrees with source")
    results = load_json(analysis.export / "results.json")
    if (
        results.get("schema") != "exp081.analysis/v1"
        or results.get("parameters") != cfg
    ):
        raise PingstoreError("unsupported or inconsistent exp081 analysis payload")
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs={"analysis": analysis, "compute": compute},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        moments = load_arrays(analysis.export / "moments.npz")
        histograms = load_arrays(analysis.export / "histograms.npz")
        responses = load_arrays(analysis.export / "frequency_response.npz")
        mean, sd = moments["empirical_mean_mV"], moments["empirical_sd_mV"]
        plot_moments(mean, sd, cfg, run.export)
        plot_distributions(histograms, cfg, run.export)
        plot_frequency_response(responses, cfg, run.export)
        plot_comparison(
            mean,
            sd,
            moments["analytical_mean_mV"],
            moments["analytical_sd_mV"],
            cfg,
            run.export,
        )
        write_json_atomic(run.export / "numbers.json", results)
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed exp081 v4 analyse run ID"
    )
    parser.add_argument("--run-id", help="unused v4 identity reserved before dispatch")
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)


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
