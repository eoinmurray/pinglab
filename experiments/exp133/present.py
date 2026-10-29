"""Render the map and coupling-onset figures used by the article.

Require explicit map and onset analysis IDs. Save SVG figures, retained circuit
diagrams and measured numerical summaries in one Pingstore presentation run."""

import argparse
import shutil
import sys
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp133 import recipe
from experiments.exp133.custom import evidence, inputs, sender_receiver
from pingstore.library import (
    PingstoreError,
    load_json,
    source_run,
    stage_run,
    write_json_atomic,
)
from snnlab import lang, viz


def circuit_diagram(bundle):
    """Align each saved E/I pair in a column with explicitly routed projections."""
    import matplotlib

    native = lang.diagram(bundle, view="circuit")
    known = {node.id for node in native.nodes if node.kind in ("input", "population")}
    pair = ("S", "R") if "drive_S" in known else ("A", "B")
    a, b = pair
    symbols = {f"drive_{c}": f"X_{c}" for c in pair}
    symbols.update({f"PING_{c}_{p}": f"{p}_{c}" for c in pair for p in ("E", "I")})
    positions = {
        f"drive_{a}": (-360, 0),
        f"drive_{b}": (1010, 0),
        f"PING_{a}_E": (0, 0),
        f"PING_{b}_E": (650, 0),
        f"PING_{a}_I": (0, 360),
        f"PING_{b}_I": (650, 360),
    }
    routes = {
        f"PING_{a}_input_to_E": viz.EdgeLayout(source_port="right", target_port="left"),
        f"PING_{b}_input_to_E": viz.EdgeLayout(source_port="left", target_port="right"),
        f"PING_{a}_E_to_I": viz.EdgeLayout(
            source_port="bottom_left", target_port="top_left"
        ),
        f"PING_{a}_I_to_E": viz.EdgeLayout(
            source_port="top_right", target_port="bottom_right"
        ),
        f"PING_{b}_E_to_I": viz.EdgeLayout(
            source_port="bottom_right", target_port="top_right"
        ),
        f"PING_{b}_I_to_E": viz.EdgeLayout(
            source_port="top_left", target_port="bottom_left"
        ),
        f"PING_{a}_E_to_PING_{b}_E_K_EE": viz.EdgeLayout(
            source_port="upper_right", target_port="upper_left"
        ),
        f"PING_{b}_E_to_PING_{a}_E_K_EE": viz.EdgeLayout(
            source_port="lower_left", target_port="lower_right"
        ),
        f"PING_{a}_E_to_PING_{b}_I_K_EI": viz.EdgeLayout(
            source_port="cross_right",
            target_port="upper_left",
            routing="polyline",
            waypoints=((325, 180),),
        ),
        f"PING_{b}_E_to_PING_{a}_I_K_EI": viz.EdgeLayout(
            source_port="cross_left",
            target_port="upper_right",
            routing="polyline",
            waypoints=((325, 180),),
        ),
    }
    if f"attention_{a}" in known:
        # Historical v2 bundles retain their originally recorded input.
        symbols[f"attention_{a}"] = f"U_{a}"
        positions[f"attention_{a}"] = (-360, -300)
        routes[f"PING_{a}_attention_to_E"] = viz.EdgeLayout(
            source_port="right",
            target_port="upper_left",
            routing="polyline",
            waypoints=((-180, -300), (-180, -32)),
        )
    for circuit, x in ((a, 0), (b, 650)):
        # Compact half-ellipses meet each card vertically, clear of its inputs.
        angles = np.linspace(0, np.pi, 33)
        routes[f"PING_{circuit}_E_to_E"] = viz.EdgeLayout(
            source_port="top_right",
            target_port="top_left",
            routing="polyline",
            waypoints=tuple(
                (x + 60 * np.cos(angle), -92 - 88 * np.sin(angle)) for angle in angles
            ),
        )
        routes[f"PING_{circuit}_I_to_I"] = viz.EdgeLayout(
            source_port="bottom_left",
            target_port="bottom_right",
            routing="polyline",
            waypoints=tuple(
                (x - 60 * np.cos(angle), 452 + 88 * np.sin(angle)) for angle in angles
            ),
        )
    for key, route in tuple(routes.items()):
        if "_K_E" in key:
            routes[key.rsplit("_K_", 1)[0]] = route
    title = "Reciprocally coupled PING circuits"
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
        cross_left=(0, 0.9),
        cross_right=(1, 0.9),
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
                routes[edge.id],
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
    """Regenerate combined figures from analysis without retaining older presentations."""
    previous = source_run(
        REPO / ".pingstore", presentation, experiment="exp133", stage="present"
    )
    sources = {
        role: source_run(REPO / ".pingstore", pin["run_id"], reference=pin)
        for role, pin in previous.record["inputs"].items()
    }
    analysis = sources["analysis"]
    if (
        analysis.record["execution"]["configuration"].get("schema")
        == "exp133.analysis/v2"
    ):
        identity = present(analysis.record["run_id"], run_id=run_id)
        previous.check_unchanged()
        return identity
    specs = [(sources["compute"], "paired-ping.bundle", "network-circuit")]
    if "sender_receiver_compute" in sources:
        specs.append(
            (
                sources["sender_receiver_compute"],
                "sender-receiver.bundle",
                "sender-receiver-circuit",
            )
        )
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


def edges(centres):
    centres = np.asarray(centres)
    if (
        len(centres) < 2
        or not np.isfinite(centres).all()
        or np.any(np.diff(centres) <= 0)
    ):
        raise ValueError("map needs at least two distinct finite ordered detunings")
    middle = (centres[:-1] + centres[1:]) / 2
    return np.r_[
        centres[0] - (middle[0] - centres[0]),
        middle,
        centres[-1] + (centres[-1] - middle[-1]),
    ]


def plot_map(data, output):
    """Render the original Fig. 1A concentration map in the shared house style."""
    theme.apply()
    order = np.argsort(data["detuning_hz"])
    x = edges(data["detuning_hz"][order])
    y = edges(data["coupling_us"])
    fig, ax = plt.subplots(figsize=(7.1, 5.3), layout="constrained")
    values = np.ma.masked_where(
        ~data["valid"][:, order], data["concentration"][:, order]
    )
    artist = ax.pcolormesh(
        x,
        y,
        values,
        cmap="pinglab_brand",
        vmin=0,
        vmax=1,
        shading="flat",
        rasterized=True,
    )
    ax.set_facecolor(theme.GREY_LIGHT)
    ax.set(
        xlabel="uncoupled detuning fA − fB (Hz)",
        ylabel="equal reciprocal coupling (μS)",
        ylim=(0, data["coupling_us"].max()),
    )
    drive_axis = ax.secondary_xaxis("top")
    drive_axis.set_xticks(
        data["detuning_hz"][order],
        labels=[f"{value:g}" for value in data["input_difference_hz"][order]],
    )
    drive_axis.set_xlabel("input drive difference A − B (Hz)")
    fig.colorbar(artist, ax=ax, label="PLV (dimensionless)")
    fig.savefig(output)
    plt.close(fig)


def plot_sender_receiver(
    spikes, phase, frequencies, output, *, display_names=("S", "R")
):
    theme.apply()
    fig, axes = plt.subplots(
        3, 1, figsize=(7.1, 6.5), sharex=True, layout="constrained"
    )
    for ax, name, display_name in zip(axes[:2], ("S", "R"), display_names, strict=True):
        for pop, offset, color in (
            ("E", 0, theme.INK_BLACK),
            ("I", recipe.N_E, theme.DEEP_RED),
        ):
            step, neuron = np.nonzero(spikes[f"{name}_{pop}"][:, 0])
            ax.scatter(
                step * recipe.DT_MS,
                neuron + offset + 1,
                marker="|",
                s=4,
                linewidths=0.45,
                color=color,
                rasterized=True,
            )
        ax.set(
            ylim=(0, recipe.N_E + recipe.N_I + 1),
            yticks=[1, recipe.N_E, recipe.N_E + recipe.N_I],
            ylabel=f"{display_name} neuron",
        )
        ax.axhline(recipe.N_E + 0.5, color=theme.GREY_LIGHT, lw=0.5)
        for label, x, align in (("first", 0.02, "left"), ("last", 0.98, "right")):
            measurement = frequencies[name][label]
            frequency = measurement["frequency_hz"]
            text = "—" if not measurement["valid"] else f"{frequency:.1f} Hz"
            ax.text(
                x,
                0.97,
                text,
                transform=ax.transAxes,
                ha=align,
                va="top",
                fontsize=theme.SIZE_ANNOTATION,
                bbox=dict(facecolor=theme.PAPER, edgecolor="none", alpha=0.92, pad=2),
            )
    axes[2].plot(phase["time_ms"], phase["phase_cycles"], color=theme.INK_BLACK, lw=1)
    a, b = display_names
    axes[2].set(ylabel=f"relative phase\n{a} − {b} (cycles)", xlabel="time (ms)")
    for ax in axes:
        ax.axvline(sender_receiver.ONSET_MS, color=theme.DEEP_RED, ls="--", lw=1.2)
        ax.set_xlim(0, sender_receiver.DURATION_MS)
        ax.set_xticks(np.arange(0, sender_receiver.DURATION_MS + 1, 500))
    axes[0].set_title(
        f"Sender {a}: {sender_receiver.DRIVE_HZ['S']:g} Hz input per channel"
    )
    axes[1].set_title(
        f"Receiver {b}: {sender_receiver.DRIVE_HZ['R']:g} Hz input per channel"
    )
    axes[0].annotate(
        "coupling on",
        xy=(sender_receiver.ONSET_MS, 1),
        xycoords=("data", "axes fraction"),
        xytext=(5, 4),
        textcoords="offset points",
        fontsize=theme.SIZE_ANNOTATION,
        color=theme.DEEP_RED,
    )
    theme.label_panels(axes)
    fig.savefig(output)
    plt.close(fig)


def sender_receiver_sources(identity):
    analysis = inputs.source(REPO, identity, "analyse")
    if set(analysis.record["inputs"]) != {"compute"}:
        raise PingstoreError("expected sender–receiver analysis with one compute input")
    pin = analysis.record["inputs"]["compute"]
    compute = inputs.source(REPO, pin["run_id"], "compute", reference=pin)
    cfg = sender_receiver.compute_contract(compute)
    if analysis.record["execution"].get("configuration") != cfg:
        raise PingstoreError("sender–receiver analysis configuration mismatch")
    result = load_json(analysis.export / "results.json")
    if (
        result.get("schema") != "exp133.sender-receiver-analysis/v1"
        or result.get("configuration") != cfg
    ):
        raise PingstoreError("sender–receiver analysis payload mismatch")
    return analysis, compute, result


def preview_sender_receiver(identity, output):
    """Render a local Fig. 3 preview without replacing the full report runset."""
    analysis, compute, result = sender_receiver_sources(identity)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with np.load(analysis.export / "relative-phase.npz", allow_pickle=False) as raw:
        phase = dict(raw)
    spikes = sender_receiver.recording(compute)
    for extension in ("svg", "png"):
        plot_sender_receiver(
            spikes,
            phase,
            result["raster_frequencies"],
            output / f"fig3.{extension}",
            display_names=("A", "B"),
        )
    write_json_atomic(
        output / "source.json",
        dict(
            compute=compute.reference,
            analysis=analysis.reference,
            labels="A/B in the figure correspond to S/R in the native diagnostic",
            figure="local onset preview; full report presentation unchanged",
        ),
    )
    return output


def present(identity, *, sender_receiver_analysis=None, run_id=None):
    analysis = inputs.source(REPO, identity, "analyse")
    if set(analysis.record["inputs"]) != {"compute"}:
        raise PingstoreError("analysis must pin exactly one grid compute run")
    pin = analysis.record["inputs"]["compute"]
    compute = inputs.source(REPO, pin["run_id"], "compute", reference=pin)
    cfg = evidence.compute_contract(compute)
    result = load_json(analysis.export / "results.json")
    if (
        result.get("schema") not in {"exp133.analysis/v1", "exp133.analysis/v2"}
        or result.get("configuration") != cfg
    ):
        raise PingstoreError("analysis payload and compute recipe disagree")
    with np.load(analysis.export / "metrics.npz", allow_pickle=False) as raw:
        data = dict(raw)
    sources = {"analysis": analysis, "compute": compute}
    sr_compute = None
    if result["schema"] == "exp133.analysis/v2":
        if sender_receiver_analysis is not None:
            raise PingstoreError(
                "combined analysis already contains the onset diagnostic"
            )
        expected = recipe.analysis_configuration(cfg)
        if analysis.record["execution"]["configuration"] != expected:
            raise PingstoreError("combined analysis settings differ from the grid")
        sr_analysis = analysis
        sr_result = result["sender_receiver"]
        if sr_result["configuration"] != expected["onset"]:
            raise PingstoreError(
                "onset extraction settings differ from recorded analysis"
            )
        spikes = evidence.onset_recording(compute, expected["onset"])
    else:
        if sender_receiver_analysis is None:
            raise PingstoreError("historical split analysis requires --sender-receiver")
        if analysis.record["execution"]["configuration"] != cfg:
            raise PingstoreError("map analysis settings differ from the grid")
        sr_analysis, sr_compute, sr_result = sender_receiver_sources(
            sender_receiver_analysis
        )
        if sr_result["configuration"].get("standalone"):
            raise PingstoreError("standalone onset requires --fig3-only")
        expected_grid_schema = (
            "exp133.recipe/v1"
            if sr_result["configuration"]["schema"] == sender_receiver.LEGACY_SCHEMA
            else "exp133.recipe/v2"
            if sr_result["configuration"]["schema"] == sender_receiver.HISTORICAL_SCHEMA
            else "exp133.recipe/v3"
        )
        if cfg["schema"] != expected_grid_schema:
            raise PingstoreError("map and onset must use the same network topology")
        sources.update(
            sender_receiver_analysis=sr_analysis, sender_receiver_compute=sr_compute
        )
        spikes = sender_receiver.recording(sr_compute)
    result["sender_receiver"] = sr_result
    with np.load(sr_analysis.export / "relative-phase.npz", allow_pickle=False) as raw:
        phase = dict(raw)
    sr = (
        spikes,
        phase,
        sr_result["raster_frequencies"],
    )
    with inputs.execution(
        REPO,
        "present",
        sources=sources,
        run_id=run_id,
        configuration=cfg,
    ) as run:
        plot_map(data, run.export / "arnold-tongue.svg")
        bundle = lang.load_bundle(
            compute.restore_bundle("paired-ping.bundle", run.scratch / "paired.bundle")
        )
        save_circuit(bundle, run.export / "network-circuit.svg")
        plot_sender_receiver(
            *sr, run.export / "sender-receiver.svg", display_names=("A", "B")
        )
        if sr_compute is not None:
            bundle = lang.load_bundle(
                sr_compute.restore_bundle(
                    "sender-receiver.bundle", run.scratch / "sender-receiver.bundle"
                )
            )
            save_circuit(bundle, run.export / "sender-receiver-circuit.svg")
        write_json_atomic(run.export / "numbers.json", result)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--sender-receiver", help="Onset analysis for historical split runsets only"
    )
    args = parser.parse_args()
    present(
        args.source, run_id=args.run_id, sender_receiver_analysis=args.sender_receiver
    )


if __name__ == "__main__":
    if "--fig3-only" in sys.argv[1:]:
        parser = argparse.ArgumentParser(description="Render only an onset preview.")
        parser.add_argument("--fig3-only", action="store_true", required=True)
        parser.add_argument("--sender-receiver", required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        print(preview_sender_receiver(args.sender_receiver, args.output), flush=True)
    elif any(arg.startswith("--redraw-from") for arg in sys.argv[1:]):
        parser = argparse.ArgumentParser(
            description="Refresh only the circuit diagrams of an explicit presentation."
        )
        parser.add_argument("--redraw-from", dest="presentation", required=True)
        parser.add_argument("--run-id")
        print(redraw_networks(**vars(parser.parse_args())), flush=True)
    else:
        main()
