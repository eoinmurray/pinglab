"""Render a matched Arnold-tongue map and E/I coupling-onset figure.

The selected map trajectory supplies both the rate and phase panels. Validated
analysis controls masking, calibration and frequency annotations; the source
network diagram and native model remain available to the report.
"""

import argparse
import shutil
import sys
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp153 import recipe
from experiments.exp153.compute import ancestry
from pingstore.library import load_json, source_run, stage_run, write_json_atomic
from snnlab import lang, viz


def network_diagram(bundle):
    """Align each saved E/I pair in a column with explicitly routed projections."""
    native = lang.diagram(bundle, view="circuit")
    known = {node.id for node in native.nodes if node.kind in ("input", "population")}
    pair = ("A", "B")
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


def edges(centres):
    """Cell edges delimit samples, without estimating a locking boundary."""
    centres = np.asarray(centres)
    if (
        len(centres) < 2
        or not np.isfinite(centres).all()
        or np.any(np.diff(centres) <= 0)
    ):
        raise ValueError("map needs distinct finite ordered centres")
    middle = (centres[:-1] + centres[1:]) / 2
    return np.r_[2 * centres[0] - middle[0], middle, 2 * centres[-1] - middle[-1]]


def map_figure(data):
    """Match the recorded PLV map; unresolved calibrations cannot invent coordinates."""
    detuning = data["detuning_hz"]
    order = np.argsort(detuning)
    calibrated = np.isfinite(detuning).all() and np.all(np.diff(detuning[order]) > 0)
    if calibrated:
        centres = detuning[order]
    else:
        # Retain every invalid column and explicitly disclose non-metric spacing.
        order = np.argsort(data["input_difference_hz"])
        centres = data["input_difference_hz"][order]
    fig, axis = plt.subplots(figsize=(7.1, 5.3), layout="constrained")
    values = np.ma.masked_where(
        ~data["valid"][:, order], data["concentration"][:, order]
    )
    artist = axis.pcolormesh(
        edges(centres),
        edges(data["coupling_us"]),
        values,
        cmap="pinglab_brand",
        vmin=0,
        vmax=1,
        shading="flat",
        rasterized=True,
    )
    axis.set_facecolor(theme.GREY_LIGHT)
    axis.set(
        xlabel="uncoupled detuning fA − fB (Hz)",
        ylabel="equal reciprocal coupling (μS)",
        ylim=(0, data["coupling_us"].max()),
    )
    if not calibrated:
        axis.set_xticks(
            centres,
            labels=[
                f"{value:.3g}" if np.isfinite(value) else "invalid"
                for value in detuning[order]
            ],
        )
        axis.set_xlabel("uncoupled detuning (Hz; columns spaced by input)")
    drive_axis = axis.secondary_xaxis("top")
    drive_axis.set_xticks(
        centres, labels=[f"{value:g}" for value in data["input_difference_hz"][order]]
    )
    drive_axis.set_xlabel("input drive difference A − B (Hz)")
    fig.colorbar(artist, ax=axis, label="PLV (dimensionless)")
    return fig


def onset_figure(rates, trace, onset, cfg):
    """Match the recorded three-panel onset layout, replacing spikes with E/I rates."""
    fig, axes = plt.subplots(
        3, 1, figsize=(7.1, 6.5), sharex=True, layout="constrained"
    )
    maximum = max(np.max(rates[f"{c}_{p}"]) for c in ("A", "B") for p in ("E", "I"))
    for axis, circuit, role, sign in zip(
        axes[:2], ("A", "B"), ("Sender", "Receiver"), (1, -1), strict=True
    ):
        for population, color in (("E", theme.INK_BLACK), ("I", theme.DEEP_RED)):
            axis.plot(
                rates["time_ms"],
                rates[f"{circuit}_{population}"],
                color=color,
                lw=0.9,
                label=population,
            )
        axis.set(ylabel=f"{circuit} rate (Hz)", ylim=(0, maximum * 1.15))
        for label, x, align in (("first", 0.02, "left"), ("last", 0.98, "right")):
            summary = onset["frequencies"][circuit][label]
            frequency = summary["frequency_hz"] if summary["valid"] else None
            axis.text(
                x,
                0.97,
                "—" if frequency is None else f"{frequency:.1f} Hz",
                transform=axis.transAxes,
                ha=align,
                va="top",
                fontsize=theme.SIZE_ANNOTATION,
                bbox=dict(facecolor=theme.PAPER, edgecolor="none", alpha=0.92, pad=2),
            )
        drive = cfg["network"]["mean_input_hz"] + sign * onset["difference_hz"] / 2
        axis.set_title(f"{role} {circuit}: {drive:g} Hz input per channel")
    axes[0].legend(
        loc="upper center", ncols=2, frameon=False, fontsize=theme.SIZE_ANNOTATION
    )
    axes[2].plot(
        trace["time_ms"], trace["relative_phase_cycles"], color=theme.INK_BLACK, lw=1
    )
    if not trace["time_ms"]:
        axes[2].text(
            0.5,
            0.5,
            "Phase unavailable: insufficient observed peaks",
            transform=axes[2].transAxes,
            ha="center",
        )
    axes[2].set(ylabel="relative phase\nA − B (cycles)", xlabel="time (ms)")
    for axis in axes:
        axis.axvline(cfg["network"]["onset_ms"], color=theme.DEEP_RED, ls="--", lw=1.2)
        axis.set_xlim(onset["display_window_ms"])
        axis.set_xticks(np.arange(0, onset["display_window_ms"][1] + 1, 500))
    axes[0].annotate(
        "coupling on",
        xy=(cfg["network"]["onset_ms"], 1),
        xycoords=("data", "axes fraction"),
        xytext=(5, 4),
        textcoords="offset points",
        fontsize=theme.SIZE_ANNOTATION,
        color=theme.DEEP_RED,
    )
    theme.label_panels(axes)
    return fig


def presentation_numbers(numbers, cfg, map_source):
    """Keep the preserved map settings separate from a fresh onset calculation."""
    onset_only = cfg["protocol"] == "onset-only"
    preserved = None
    if onset_only:
        if map_source is None:
            raise ValueError("onset-only presentation requires an explicit map source")
        preserved = source_run(
            REPO / ".pingstore", map_source, experiment=recipe.SLUG, stage="present"
        )
        previous = load_json(preserved.file("numbers.json"))
        numbers = dict(
            previous,
            schema="exp153.presentation/v3",
            configuration=cfg,
            onset=numbers["onset"],
            map_configuration=previous.get(
                "map_configuration", previous["configuration"]
            ),
            preserved_map=preserved.reference,
        )
    elif map_source is not None:
        raise ValueError("map source applies only to onset-only presentation")
    return numbers, preserved


def preview_onset(source, output):
    """Render only Figure 2 from pinned analysis without changing the report runset."""
    ancestors = ancestry(REPO, source)
    analysed = ancestors[source]
    if analysed.record["stage"] != "analyse":
        raise ValueError("Figure 2 preview requires an exp153 analysis")
    cfg = analysed.record["execution"]["configuration"]
    computed = ancestors[analysed.record["inputs"]["compute"]["run_id"]]
    numbers = load_json(analysed.file("results.json"))
    if (
        numbers["configuration"] != cfg
        or recipe.analysis_configuration(computed.record["execution"]["configuration"])
        != cfg
    ):
        raise ValueError("preview lineage settings differ")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    theme.apply()
    with np.load(analysed.file("onset-rates.npz"), allow_pickle=False) as raw:
        rates = dict(raw)
    figure = onset_figure(
        rates, load_json(analysed.file("onset-phase.json")), numbers["onset"], cfg
    )
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output / f"coupling-onset.{suffix}")
    plt.close(figure)
    write_json_atomic(
        output / "source.json",
        dict(compute=computed.reference, analyse=analysed.reference),
    )
    for item in ancestors.values():
        item.check_unchanged()
    return output


def present(*, source, run_id=None, map_source=None):
    ancestors = ancestry(REPO, source)
    analysed = ancestors[source]
    if (
        analysed.record["experiment"] != recipe.SLUG
        or analysed.record["stage"] != "analyse"
    ):
        raise ValueError("requires an exp153 analysis")
    computed = ancestors[analysed.record["inputs"]["compute"]["run_id"]]
    cfg = analysed.record["execution"]["configuration"]
    numbers = load_json(analysed.file("results.json"))
    if (
        numbers["configuration"] != cfg
        or recipe.analysis_configuration(computed.record["execution"]["configuration"])
        != cfg
    ):
        raise ValueError("presentation lineage settings differ")
    onset_only = cfg["protocol"] == "onset-only"
    numbers, preserved = presentation_numbers(numbers, cfg, map_source)
    if (
        cfg["schema"] != recipe.SCHEMA
        or cfg["protocol"] not in ("matched-grid", "onset-only")
        or (not onset_only and numbers.get("schema") != "exp153.analysis/v2")
    ):
        raise ValueError("requires matched-grid analysis")
    theme.apply()
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs=dict(
            analyse=analysed,
            compute=computed,
            **({"map": preserved} if preserved else {}),
        ),
        configuration=cfg,
        run_id=run_id,
    ) as run:
        if preserved:
            for suffix in ("png", "svg", "pdf"):
                shutil.copyfile(
                    preserved.file(f"arnold-tongue.{suffix}"),
                    run.export / f"arnold-tongue.{suffix}",
                )
        else:
            with np.load(analysed.file("metrics.npz"), allow_pickle=False) as raw:
                data = dict(raw)
        with np.load(analysed.file("onset-rates.npz"), allow_pickle=False) as raw:
            rates = dict(raw)
        figures = dict(
            coupling_onset=onset_figure(
                rates,
                load_json(analysed.file("onset-phase.json")),
                numbers["onset"],
                cfg,
            ),
        )
        if not preserved:
            figures["arnold_tongue"] = map_figure(data)
        for name, figure in figures.items():
            for suffix in ("png", "svg", "pdf"):
                figure.savefig(run.export / f"{name.replace('_', '-')}.{suffix}")
            plt.close(figure)
        for name in ("onset-phase.json", "onset-rates.npz"):
            shutil.copyfile(analysed.file(name), run.export / name)
        shutil.copyfile(
            (preserved if preserved else analysed).file("metrics.npz"),
            run.export / "metrics.npz",
        )
        shutil.copyfile(
            computed.file(numbers["onset"]["label"] + "-model.json"),
            run.export / "meanfield-model.json",
        )
        bundle = lang.load_bundle(
            computed.restore_bundle("network.bundle", run.scratch / "diagram.bundle")
        )
        circuit = network_diagram(bundle)
        with matplotlib.rc_context({"savefig.bbox": None}):
            for extension in ("svg", "png"):
                viz.render_diagram(circuit, run.export / f"circuit.{extension}")
        run.record["execution"]["snnlab_version"] = version("snnlab")
        write_json_atomic(run.export / "numbers.json", numbers)
        if preserved:
            preserved.check_unchanged()
        for item in ancestors.values():
            item.check_unchanged()
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--map-source")
    parser.add_argument("--fig2-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.fig2_only:
        if args.output is None or args.run_id or args.map_source:
            parser.error("--fig2-only requires --output and accepts no run/map ID")
        preview_onset(args.source, args.output)
    else:
        if args.output is not None:
            parser.error("--output requires --fig2-only")
        present(source=args.source, run_id=args.run_id, map_source=args.map_source)
