"""Present full-window and 200 ms rasters and four AI diagnostics independently.

Run present.py --source ANALYSE_ID. The analysis run identifies its validated
compute input, whose retained spikes supply the raster after burn-in removal.
Four diagnostic panels summarize the same measurement window.
"""

import sys
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

import argparse
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp145.custom.recording import read_recording
from pingstore.library import source_run, stage_run
from snnlab import lang, viz


def circuit_diagram(bundle):
    """Lay out native weighted projections; omit recording and scoring aliases."""
    import matplotlib

    native = lang.diagram(bundle, view="circuit")
    known = {node.id for node in native.nodes if node.kind in ("input", "population")}
    symbols = dict(X="X", E="E", I="I")
    positions = dict(X=(-400, 150), E=(0, 0), I=(0, 300))
    routes = dict(
        X_to_E=viz.EdgeLayout(source_port="upper_right", target_port="left"),
        X_to_I=viz.EdgeLayout(source_port="lower_right", target_port="left"),
        E_to_I=viz.EdgeLayout(source_port="bottom_left", target_port="top_left"),
        I_to_E=viz.EdgeLayout(source_port="top_right", target_port="bottom_right"),
        E_to_E=viz.EdgeLayout(
            source_port="top_right", target_port="right", waypoints=((180, -130),)
        ),
        I_to_I=viz.EdgeLayout(
            source_port="bottom_right", target_port="right", waypoints=((180, 430),)
        ),
    )
    title = "Shared input and recurrent E/I pathways"
    diagram = replace(
        native,
        title=title,
        groups=(),
        nodes=tuple(
            replace(node, detail=node.detail.splitlines()[0])
            for node in native.nodes
            if node.id in known
        ),
        edges=tuple(edge for edge in native.edges if edge.id is not None),
    )
    ports = dict(
        upper_left=(0, 0.3),
        lower_left=(0, 0.7),
        upper_right=(1, 0.3),
        lower_right=(1, 0.7),
        top_left=(0.25, 0),
        top_right=(0.75, 0),
        bottom_left=(0.25, 1),
        bottom_right=(0.75, 1),
    )
    layout = viz.DiagramLayout(
        nodes={
            identifier: viz.NodeLayout(
                position=position,
                label=f"${symbols[identifier]}$",
                min_size=(240, 160),
                ports=ports,
            )
            for identifier, position in positions.items()
            if identifier in known
        },
        edges={
            edge.id: replace(
                routes.get(edge.id, viz.EdgeLayout()),
                label=viz.DiagramLabel(
                    rf"$W_{{{symbols[edge.target]},{symbols[edge.source]}}}$"
                ),
            )
            for edge in diagram.edges
        },
        defaults=viz.LayoutDefaults(
            node=viz.NodeLayout(content_align="center", padding=16), gap=30
        ),
    )
    with matplotlib.rc_context({"savefig.bbox": None}):
        return viz.resolve_diagram(
            diagram, layout=layout, theme=viz.DiagramTheme(secondary_size=13)
        )


def save_circuit(bundle, path):
    """Export the resolved native architecture without inheriting plot cropping."""
    import matplotlib

    circuit = circuit_diagram(bundle)
    with matplotlib.rc_context({"savefig.bbox": None}):
        for suffix in ("svg", "png"):
            viz.render_diagram(circuit, Path(path).with_suffix("." + suffix))


def redraw_networks(presentation, *, run_id=None):
    """Preserve an explicit presentation's evidence and replace only its diagrams."""
    previous = source_run(
        REPO / ".pingstore", presentation, experiment="exp145", stage="present"
    )
    sources = {
        role: source_run(REPO / ".pingstore", pin["run_id"], reference=pin)
        for role, pin in previous.record["inputs"].items()
    }
    specs = [(sources["compute"], "simulation/network.bundle", "circuit")]
    with stage_run(
        REPO,
        previous.record["experiment"],
        "present",
        inputs={**sources, "figure_source": previous},
        configuration=previous.record["execution"]["configuration"],
        run_id=run_id,
    ) as run:
        shutil.copytree(previous.export, run.export, dirs_exist_ok=True)
        for source, prefix, stem in specs:
            bundle_path = (
                source.restore_bundle(prefix, run.scratch / (stem + ".bundle"))
                if source.record.get("scientific_artifacts", {}).get("native_paths")
                else source.unit(*prefix.split("/"))
            )
            save_circuit(lang.load_bundle(bundle_path), run.export / (stem + ".svg"))
        run.record["execution"]["snnlab_version"] = version("snnlab")
        previous.check_unchanged()
    return run.run_id


def present(args):
    theme.apply()
    plt.rcParams["svg.fonttype"] = "none"
    source = source_run(
        REPO / ".pingstore", args.source, experiment="exp145", stage="analyse"
    )
    # Resolve the recorded ancestry explicitly; no latest-run selection.
    inputs = source.record["inputs"]
    compute_id = inputs["compute"]["run_id"]
    compute = source_run(
        REPO / ".pingstore", compute_id, experiment="exp145", stage="compute"
    )
    cfg = source.record["execution"]["configuration"]
    summary = json.loads(source.file("results.json").read_text())
    start = round(cfg["burn_in_ms"] / cfg["dt_ms"])
    window_ms = cfg["duration_ms"] - cfg["burn_in_ms"]
    with (
        np.load(source.export / "measurements.npz") as data,
        stage_run(
            REPO,
            "exp145",
            "present",
            inputs={"analyse": source, "compute": compute},
            configuration=cfg,
            run_id=args.run_id,
        ) as run,
    ):
        recording = read_recording(compute.unit("recording"), run.scratch / "reader")
        save_circuit(
            lang.load_bundle(
                compute.restore_bundle(
                    "simulation/network.bundle", run.scratch / "network.bundle"
                )
            ),
            run.export / "circuit.svg",
        )
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
        for population, color, offset in (
            ("E", theme.INK_BLACK, 0),
            ("I", theme.DEEP_RED, cfg["n_e"]),
        ):
            times, cells = np.nonzero(recording.signals[f"{population}.spikes"][start:])
            times_ms = times * cfg["dt_ms"]
            for axis, selected, marker_size in (
                (axes[0], slice(None), 1),
                (axes[1], times_ms < min(200.0, window_ms), 3),
            ):
                axis.scatter(
                    times_ms[selected],
                    cells[selected] + offset,
                    s=marker_size,
                    c=color,
                    linewidths=0,
                    rasterized=True,
                )
        axes[0].set(
            xlim=(0, window_ms),
            xlabel="Time after burn-in (ms)",
            ylabel="Neuron index",
            title="Shared-input conductance network",
        )
        axes[1].set(
            xlim=(0, min(200.0, window_ms)),
            xlabel="Time after burn-in (ms)",
            ylabel="Neuron index",
            title="First 200 ms after burn-in",
        )
        for axis in axes:
            axis.set_ylim(0, cfg["n_e"] + cfg["n_i"])
        theme.label_panels(axes)
        fig.savefig(run.export / "onset.png")
        fig.savefig(run.export / "onset.svg")
        plt.close(fig)
        diagnostics = diagnostic_figure(data, summary, cfg)
        diagnostics.savefig(run.export / "ai-diagnostics.png")
        diagnostics.savefig(run.export / "ai-diagnostics.svg")
        plt.close(diagnostics)
        shutil.copyfile(source.file("results.json"), run.export / "numbers.json")
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    print(present(parser.parse_args()))


COLORS = {"E": theme.INK_BLACK, "I": theme.DEEP_RED}


def diagnostic_figure(data, summary, cfg):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    cvax, corrax, scaleax, currentax = axes.ravel()
    finite_cv = np.concatenate(
        [data[f"{p}_cv"][np.isfinite(data[f"{p}_cv"])] for p in ("E", "I")]
    )
    bins = np.linspace(
        0, max(2.5, float(finite_cv.max()) * 1.05 if finite_cv.size else 2.5), 26
    )
    for index, p in enumerate(("E", "I")):
        values = data[f"{p}_cv"]
        values = values[np.isfinite(values)]
        info = summary["populations"][p]
        if values.size:
            cvax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                color=COLORS[p],
                label=p,
            )
        median = (
            f"{info['median_cv']:.2f}"
            if info["median_cv"] is not None
            else "unresolved"
        )
        cvax.text(
            0.98,
            0.95 - index * 0.12,
            f"{p}: median {median}; {info['cv_eligible']}/{info['cells']} eligible",
            transform=cvax.transAxes,
            ha="right",
            va="top",
            color=COLORS[p],
            fontsize=theme.SIZE_ANNOTATION,
        )
    cvax.axvline(1, color=theme.GREY_LIGHT, ls="--", lw=1)
    cvax.set(
        title="Spike irregularity",
        xlabel="Interspike-interval CV",
        ylabel="Density",
        xlim=(bins[0], bins[-1]),
    )
    if not finite_cv.size:
        cvax.text(
            0.5,
            0.45,
            "Insufficient spikes for ISI CV",
            ha="center",
            transform=cvax.transAxes,
        )

    widths = np.asarray(cfg["analysis"]["correlation_bins_ms"])
    for pair, color in (
        ("EE", COLORS["E"]),
        ("II", COLORS["I"]),
        ("EI", theme.ELECTRIC_CYAN),
    ):
        rows = [summary["correlations"][str(float(width))][pair] for width in widths]
        means = np.asarray(
            [np.nan if row["mean"] is None else row["mean"] for row in rows]
        )
        stds = np.asarray(
            [np.nan if row["std"] is None else row["std"] for row in rows]
        )
        corrax.plot(widths, means, "o-", color=color, label=pair[0] + "–" + pair[1])
        corrax.fill_between(widths, means - stds, means + stds, color=color, alpha=0.12)
    corrax.axhline(0, color=theme.GREY_LIGHT, lw=1, ls="--")
    corrax.set(
        title="Pairwise spike-count correlation",
        xlabel="Sliding counting window (ms)",
        ylabel="Mean Pearson r ± pairwise SD",
    )
    corrax.legend()

    for p in ("E", "I"):
        sizes = data[f"{p}_group_sizes"]
        med = data[f"{p}_q_median"]
        slope = summary["populations"][p]["scaling"]["loglog_slope"]
        label = f"{p}: slope {slope:.2f}" if slope is not None else f"{p}: unresolved"
        scaleax.loglog(sizes, med, "o-", color=COLORS[p], label=label, markersize=4)
        scaleax.fill_between(
            sizes, data[f"{p}_q_low"], data[f"{p}_q_high"], color=COLORS[p], alpha=0.15
        )
    reference = np.asarray(
        [min(cfg["analysis"]["group_sizes"]), max(cfg["n_e"], cfg["n_i"])]
    )
    scaleax.loglog(
        reference,
        1 / reference,
        "--",
        color=theme.GREY_MID,
        label="Independent reference: 1/K",
    )
    scaleax.set(
        title="Population-fluctuation scaling",
        xlabel="Neurons averaged, K",
        ylabel="Normalised variance Q(K)",
    )
    scaleax.legend()
    scaleax.text(
        0.03,
        0.04,
        "Shading: 5–95% across random groups",
        transform=scaleax.transAxes,
        fontsize=theme.SIZE_ANNOTATION,
    )

    lag = data["current_lags_ms"]
    for name, color, style in (
        ("EE", theme.INK_BLACK, "--"),
        ("II", theme.DEEP_RED, "-"),
        ("EI", theme.ELECTRIC_CYAN, "-"),
        ("IE", theme.AMBER, "--"),
    ):
        currentax.plot(
            lag, data[f"current_{name}"], color=color, ls=style, lw=1.3, label=name
        )
    currentax.plot(
        lag, data["current_total"], color=theme.INK_BLACK, lw=2, label="Total"
    )
    currentax.plot(
        lag,
        data["current_filtered_total"],
        color=theme.GREY_MID,
        ls=":",
        lw=2,
        label=f"Total, {cfg['analysis']['current_filter_ms']:g} ms filter",
    )
    currentax.axhline(0, color=theme.GREY_LIGHT, lw=0.7)
    currentax.set(
        title="Current covariance cancellation",
        xlabel="Lag τ (ms)",
        ylabel="Mean cross-covariance (nA²)",
    )
    currentax.legend(ncol=3, fontsize=theme.SIZE_ANNOTATION)
    # Leave headroom so the framed legend does not hide the covariance peaks.
    low, high = currentax.get_ylim()
    currentax.set_ylim(low, high + 0.25 * (high - low))
    ratio = summary["current_cancellation"]["total_to_same_type_zero_lag"]
    ratio_text = f"{ratio:.3f}" if ratio is not None else "unresolved"
    currentax.text(
        0.02,
        0.04,
        f"Zero-lag total / (EE + II): {ratio_text}",
        transform=currentax.transAxes,
        fontsize=theme.SIZE_ANNOTATION,
    )
    theme.label_panels(axes.ravel())
    fig.suptitle(
        "Shared-input conductance network · AI diagnostics",
        fontsize=theme.SIZE_TITLE,
        fontweight="bold",
    )
    fig.supxlabel(
        f"{cfg['burn_in_ms']:g} ms burn-in · {summary['window_ms'] / 1000:g} s measured · seed {cfg['seed']} · CV requires ≥{cfg['analysis']['min_cv_spikes']} spikes",
        fontsize=theme.SIZE_CAPTION,
    )
    return fig


if __name__ == "__main__":
    if any(arg.startswith("--redraw-from") for arg in sys.argv[1:]):
        parser = argparse.ArgumentParser(
            description="Refresh only the circuit diagrams of an explicit presentation."
        )
        parser.add_argument("--redraw-from", dest="presentation", required=True)
        parser.add_argument("--run-id")
        print(redraw_networks(**vars(parser.parse_args())), flush=True)
    else:
        main()
