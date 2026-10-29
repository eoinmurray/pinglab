"""Render completed exp023 measurements and pinned rasters; never simulate.

Invoke this stage explicitly from the repository root. Scientific outputs
and input provenance belong to its independently completed Pingstore run.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

import argparse
import shutil
import time
from dataclasses import replace
from importlib.metadata import version
from tempfile import TemporaryDirectory

import matplotlib
from snnlab import lang, viz

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp023 import recipe
from experiments.exp023.custom import inputs
from pingstore.library import (
    PingstoreError,
    load_json,
    source_run,
    stage_run,
    write_json_atomic,
)

DEFAULT_FORMATS = ("svg", "pdf")


def circuit_diagram(bundle):
    """Lay out native weighted projections; omit recording and scoring aliases."""
    import matplotlib

    native = lang.diagram(bundle, view="circuit")
    known = {node.id for node in native.nodes if node.kind in ("input", "population")}
    symbols = dict(drive="X", image="X", E="E", I="I", readout="R")
    positions = dict(
        drive=(-360, 0), image=(-360, 0), E=(0, 0), I=(0, 300), readout=(360, 0)
    )
    routes = dict(
        E_to_I=viz.EdgeLayout(source_port="bottom_left", target_port="top_left"),
        I_to_E=viz.EdgeLayout(source_port="top_right", target_port="bottom_right"),
    )
    zero_feedback = all(
        row["initializer"].get("mean") == 0 and row["initializer"].get("std") == 0
        for row in bundle.graph["parameters"]
        if row["id"] in ("E_to_I.weight", "I_to_E.weight")
    )
    title = (
        "COBA: reciprocal loop disabled"
        if zero_feedback
        else "PING: reciprocal loop active"
    )
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
                    rf"$W_{{{symbols[edge.target]},{symbols[edge.source]}}}"
                    + (
                        "=0$"
                        if zero_feedback and edge.id in ("E_to_I", "I_to_E")
                        else "$"
                    )
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


def render_networks(sources, export, scratch):
    """Render both executed loop conditions, including zero feedback in COBA."""
    compute = sources["compute"]
    files = []
    for mode in ("coba", "ping"):
        prefix = mode + "-network"
        bundle = (
            compute.restore_bundle(prefix, scratch / prefix)
            if compute.record.get("scientific_artifacts", {}).get("native_paths")
            else compute.unit(prefix)
        )
        diagram = circuit_diagram(lang.load_bundle(bundle))
        with matplotlib.rc_context({"savefig.bbox": None}):
            for suffix in ("svg", "png"):
                viz.render_diagram(diagram, export / (prefix + "." + suffix))
        files.append(prefix + ".svg")
    return files


def redraw_networks(presentation, *, run_id=None):
    """Copy an explicit presentation, replacing only its saved-bundle diagrams."""
    previous = source_run(
        REPO / ".pingstore", presentation, experiment="exp023", stage="present"
    )
    sources = {
        role: source_run(REPO / ".pingstore", pin["run_id"], reference=pin)
        for role, pin in previous.record["inputs"].items()
    }
    with stage_run(
        REPO,
        "exp023",
        "present",
        inputs={**sources, "figure_source": previous},
        configuration=previous.record["execution"]["configuration"],
        run_id=run_id,
    ) as run:
        shutil.copytree(previous.export, run.export, dirs_exist_ok=True)
        render_networks(sources, run.export, run.scratch)
        redraw_overview(sources["analysis"], sources["compute"], run.export)
        run.record["execution"]["snnlab_version"] = version("snnlab")
        previous.check_unchanged()
    return run.run_id


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


def present(identity: str, *, run_id: str | None = None) -> str:

    analysis = inputs.source(REPO, identity, "analyse")
    if set(analysis.record["inputs"]) != {"compute"}:
        raise PingstoreError("exp023 analysis must pin exactly its computation")
    ref = analysis.record["inputs"]["compute"]
    compute = inputs.source(REPO, ref["run_id"], "compute", reference=ref)
    cfg = inputs.configuration(compute)
    results = load_json(analysis.export / "results.json")
    if (
        results.get("schema") != "exp023.analysis/v1"
        or results.get("config") != cfg
        or results.get("measurement") != analysis.record["execution"]["configuration"]
    ):
        raise PingstoreError("unsupported or inconsistent exp023 analysis payload")
    with np.load(analysis.export / "spectra.npz", allow_pickle=False) as data:
        spectra = {
            cell: {
                key: np.array(data[f"{cell}__{key}"])
                for key in ("frequency_hz", "density")
            }
            for cell in cfg["cells"]
        }
    with np.load(analysis.export / "traces.npz", allow_pickle=False) as data:
        traces = {
            cell: {
                key.removeprefix(cell + "__"): np.array(data[key])
                for key in data.files
                if key.startswith(cell + "__")
            }
            for cell in cfg["cells"]
        }
    snaps = {}
    for cell in cfg["cells"]:
        with np.load(
            compute.file("scope", cell, "recording.npz"), allow_pickle=False
        ) as data:
            snaps[cell] = {key: np.array(data[key]) for key in ("spk_e", "spk_i", "dt")}
    started = time.monotonic()
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs={"analysis": analysis, "compute": compute},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        theme.set_paper_mode(True)
        plot_architecture(run.export / "architecture")
        for cell in cfg["cells"]:
            plot_traces(
                traces[cell],
                results["raster"][cell],
                cfg["biophysics"],
                run.export / f"traces__{cell}",
                cell.upper(),
            )
        titles = {
            "coba": "COBA — recurrent loop off",
            "ping": "PING — recurrent loop active",
        }
        for name, include_arch in (
            ("raster_compound", False),
            ("overview_compound", True),
        ):
            plot_raster_compound(
                snaps,
                results["fi_curves"],
                run.export / name,
                titles,
                spectra,
                results["f_gamma_hz"],
                results["measurement"]["frequency_band_hz"],
                include_arch=include_arch,
            )
        run.record["presentation_lineage"] = {
            "measurements": analysis.reference,
            "rasters": compute.reference,
            "operation": "render retained measurements and spikes; no simulation or remeasurement",
        }
        write_json_atomic(
            run.export / "numbers.json",
            {
                **results,
                "config": cfg,
                "run_id": run.run_id,
                "duration_s": time.monotonic() - started,
                "git_sha": run.record["provenance"]["git_commit"],
                "circuit_files": render_networks(
                    {"compute": compute}, run.export, run.scratch
                ),
            },
        )
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed exp023 analyse run ID"
    )
    parser.add_argument("--run-id", help="unused v4 identity reserved before dispatch")
    args = parser.parse_args()
    present(args.source, run_id=args.run_id)


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_raster_compound(
    snaps: dict,
    fi: dict,
    out_path: Path,
    titles: dict,
    spectra: dict,
    peaks: dict,
    band_hz: tuple[float, float],
    include_arch: bool = False,
) -> None:
    """Super figure: COBA vs PING side by side.

    Each condition is a column-pair: a single raster (I stacked above E, no gap)
    spans the pair; below it the population-E Welch PSD sits next to the
    free-running f–I curve. Silent I populations are omitted from the raster.

    With include_arch=True a top row carries each column's architecture
    schematic (COBA / PING) directly above its plots.
    """
    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"  # preserve the planned canvas
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(180 / 25.4, 4.7 if include_arch else 3.88))
    if include_arch:
        # Nested gridspecs so the (small) schematic→plots gap is independent of
        # the (larger) raster→PSD gap that has to clear the time-axis label.
        outer = GridSpec(
            2,
            1,
            figure=fig,
            height_ratios=[2.5, 7.4],
            hspace=0.10,
            top=0.92,
            bottom=0.13,
            left=0.08,
            right=0.955,
        )
        arch_gs = outer[0].subgridspec(1, 4, wspace=0.5)
        plot_gs = outer[1].subgridspec(
            2,
            4,
            height_ratios=[4.4, 2.6],
            hspace=0.85,
            wspace=0.5,
        )
    else:
        arch_gs = None
        plot_gs = GridSpec(
            2,
            4,
            figure=fig,
            height_ratios=[4.4, 2.6],
            hspace=0.85,
            wspace=0.5,
            top=0.92,
            bottom=0.13,
            left=0.08,
            right=0.955,
        )

    arch_axes = []
    raster_axes = []
    lower_axes = []
    psd_axes = []
    for col, cell in enumerate(("coba", "ping")):
        c0 = 2 * col
        s = snaps[cell]
        spk_e, spk_i, dt = s["spk_e"], s["spk_i"], s["dt"]
        T = spk_e.shape[0]
        t_ms = np.arange(T) * dt
        has_i = spk_i.size > 0 and spk_i.shape[0] == T and spk_i.any()

        if include_arch:
            # arch_gs is only built (non-None) in the include_arch branch above.
            assert arch_gs is not None
            ax_arch = fig.add_subplot(arch_gs[0, c0 : c0 + 2])
            arch_axes.append(ax_arch)
            _draw_schematic(ax_arch, cell)
            ax_arch.set_title(
                "COBA — loop off" if cell == "coba" else "PING — loop on",
                loc="left",
                fontweight="semibold",
                fontsize=theme.SIZE_LABEL,
            )

        ax_r = fig.add_subplot(plot_gs[0, c0 : c0 + 2])  # one raster, I above E
        ax_psd = fig.add_subplot(
            plot_gs[1, c0], sharey=psd_axes[0] if psd_axes else None
        )  # PSD next to f–I
        psd_axes.append(ax_psd)
        ax_fi = fig.add_subplot(plot_gs[1, c0 + 1])
        raster_axes.append(ax_r)
        lower_axes.extend((ax_psd, ax_fi))

        # Combined raster: E (black) at the bottom, I (red) stacked directly
        # above it in the same axes — no vertical gap between the populations.
        n_e = spk_e.shape[1]
        e_idx, e_t = np.where(spk_e.T)
        ax_r.scatter(
            t_ms[e_t], e_idx, s=1.0, c=theme.INK_BLACK, marker="|", linewidths=0.5
        )
        if has_i:
            n_i = spk_i.shape[1]
            i_idx, i_t = np.where(spk_i.T)
            ax_r.scatter(
                t_ms[i_t],
                n_e + i_idx,
                s=1.0,
                c=theme.DEEP_RED,
                marker="|",
                linewidths=0.5,
            )
            ax_r.axhline(n_e, color=theme.GREY_MID, lw=0.6, alpha=0.6)
            total = n_e + n_i
            ax_r.set_yticks([n_e / 2, n_e + n_i / 2])
            ax_r.set_yticklabels(["E", "I"])
        else:
            total = n_e
            ax_r.set_yticks([n_e / 2])
            ax_r.set_yticklabels(["E"])
            ax_r.set_title(
                "I silent",
                loc="right",
                fontsize=theme.SIZE_ANNOTATION,
                color=theme.MUTED,
            )
        ax_r.set_ylim(0, total)
        ax_r.set_xlim(0, T * dt)
        ax_r.set_xlabel("time (ms)")
        if not include_arch:
            ax_r.set_title(titles[cell], loc="left", fontweight="semibold")
        _despine(ax_r)

        # Numerical measurements are retained analysis outputs, never recomputed here.
        freqs, psd = spectra[cell]["frequency_hz"], spectra[cell]["density"]
        f_peak = peaks[cell]
        band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
        ax_psd.plot(freqs[band], psd[band], color=theme.INK_BLACK, lw=1.4)
        ax_psd.set_xlim(band_hz)
        ax_psd.set_xlabel("frequency (Hz)")
        ax_psd.set_ylabel("E PSD (a.u.)")
        _despine(ax_psd)
        # Only the loop-on condition has a recurrent gamma peak to mark; the
        # loop-off control has scattered input-driven power, not a rhythm.
        if has_i and f_peak is not None:
            ax_psd.axvline(f_peak, color=theme.DEEP_RED, lw=0.9, ls="--", alpha=0.8)
            ax_psd.text(
                0.97,
                0.94,
                f"  $f_\\gamma$ = {f_peak:.1f} Hz",
                transform=ax_psd.transAxes,
                ha="right",
                va="top",
                fontsize=theme.SIZE_LABEL - 1,
                color=theme.DEEP_RED,
                fontweight="semibold",
            )
        elif has_i:
            # Loop on but no clean peak found. COBA (loop off) gets no label —
            # the flat spectrum speaks for itself.
            ax_psd.text(
                0.97,
                0.95,
                "no clear peak",
                transform=ax_psd.transAxes,
                ha="right",
                va="top",
                fontsize=theme.SIZE_LABEL - 1,
                color=theme.GREY_MID,
                fontstyle="italic",
            )

        # f–I curve (bottom-right of the pair). Per-condition y-scale: COBA runs
        # to its ~400+ Hz ceiling; PING is clamped by the loop, so it gets its
        # own smaller y-axis where the E-clamp and climbing I are legible (a
        # shared axis buried PING's curves at the bottom).
        f = fi[cell]
        ax_fi.plot(
            f["in"], f["e"], color=theme.INK_BLACK, marker="o", ms=3, lw=1.3, label="E"
        )
        ax_fi.plot(
            f["in"], f["i"], color=theme.DEEP_RED, marker="s", ms=3, lw=1.3, label="I"
        )
        cell_max = max(f["e"] + f["i"])
        ax_fi.set_ylim(0, max(cell_max * 1.08, 1.0))
        ax_fi.set_xlim(0, max(f["in"]))
        ax_fi.set_xlabel("input rate (Hz)")
        ax_fi.set_ylabel("rate (Hz)")
        _despine(ax_fi)
        if col == 1:
            ax_fi.legend(frameon=False, fontsize=theme.SIZE_LABEL - 2, loc="upper left")

    psd_max = max(
        float(
            np.max(
                values["density"][
                    (values["frequency_hz"] >= band_hz[0])
                    & (values["frequency_hz"] <= band_hz[1])
                ]
            )
        )
        for values in spectra.values()
    )
    psd_axes[0].set_ylim(0, psd_max * 1.08 if psd_max > 0 else 1.0)

    # Label every row from left to right.
    theme.label_panels((*arch_axes, *raster_axes, *lower_axes))
    save_figure(
        fig, out_path, formats=("svg", "pdf", "png") if include_arch else ("svg", "pdf")
    )
    plt.close(fig)


def plot_traces(
    data: dict, selected: dict, biophysics: dict, out_path: Path, title: str
) -> None:
    """Draw analysis-selected voltage, conductance and current traces unchanged."""
    theme.apply()
    for population, index in (("e", selected["e_index"]), ("i", selected["i_index"])):
        if index is None:
            continue
        color = theme.INK_BLACK if population == "e" else theme.DEEP_RED
        time_ms = data["time_ms"]
        for panel in ("v", "g", "i"):
            fig, ax = plt.subplots(figsize=(4.0, 2.25))
            if panel == "v":
                ax.plot(
                    time_ms,
                    data[f"v_{population}"],
                    color=color,
                    lw=0.8,
                    label=f"V_{population.upper()}",
                )
                ax.axhline(
                    biophysics["threshold_mV"],
                    color=theme.FAINT,
                    lw=0.5,
                    ls="--",
                    label="V_th",
                )
                label, unit = "voltage", "V [mV]"
            elif panel == "g":
                ax.plot(
                    time_ms,
                    data[f"ge_{population}"],
                    color=theme.INK_BLACK,
                    lw=0.9,
                    label="g_E (exc)",
                )
                if population == "e" and selected["has_gi_e"]:
                    ax.plot(
                        time_ms,
                        data["gi_e"],
                        color=theme.DEEP_RED,
                        lw=0.9,
                        label="g_I (inh)",
                    )
                ax.axhline(
                    biophysics[f"g_L_{population.upper()}_uS"],
                    color=theme.FAINT,
                    lw=0.7,
                    ls=":",
                    label="g_L (leak)",
                )
                label, unit = "conductances", "g [µS]"
            else:
                ax.axhline(0, color=theme.FAINT, lw=0.5)
                ax.plot(
                    time_ms,
                    data[f"ie_{population}"],
                    color=theme.INK_BLACK,
                    lw=0.9,
                    label="I_E in",
                )
                if population == "e" and selected["has_gi_e"]:
                    ax.plot(
                        time_ms,
                        data["ii_e"],
                        color=theme.DEEP_RED,
                        lw=0.9,
                        label="I_I in",
                    )
                ax.plot(
                    time_ms,
                    data[f"il_{population}"],
                    color=theme.FAINT,
                    lw=0.7,
                    ls=":",
                    label="I_L in",
                )
                label, unit = "currents", "inward current [nA]"
            ax.set_xlim(0, time_ms[-1] + (time_ms[1] - time_ms[0]))
            ax.set_xlabel("time (ms)")
            ax.set_ylabel(unit)
            ax.set_title(
                f"{title} · {population.upper()} {label} (cell {index})",
                loc="left",
                fontsize=theme.SIZE_LABEL,
            )
            ax.legend(loc="upper right", fontsize=theme.SIZE_LEGEND)
            fig.tight_layout()
            save_figure(
                fig, out_path.with_name(f"{out_path.name}__{panel}_{population}")
            )
            plt.close(fig)


def overview_diagram(kind):
    """Illustrate the loop condition without claiming sampled historical weights."""
    nodes = (
        viz.DiagramNode("X", "X", "", "", kind="input"),
        viz.DiagramNode("E", "E", "", "", kind="population"),
        viz.DiagramNode("I", "I", "", "", kind="population"),
    )
    edges = (
        viz.DiagramEdge("X", "E", id="X_to_E", role="excitatory"),
        viz.DiagramEdge("E", "I", id="E_to_I", role="excitatory"),
        viz.DiagramEdge("I", "E", id="I_to_E", role="inhibitory"),
    )
    diagram = viz.Diagram(
        name="exp023_" + kind,
        title=" ",
        nodes=nodes,
        edges=edges,
        metadata=dict(view="illustrative loop conditions"),
    )
    layout = viz.DiagramLayout(
        nodes={
            n.id: viz.NodeLayout(
                position=dict(X=(-360, 0), E=(0, 0), I=(360, 0))[n.id],
                label=f"${n.id}$",
                min_size=(180, 110),
                ports=dict(a=(0, 0.25), b=(0, 0.75), c=(1, 0.25), d=(1, 0.75)),
            )
            for n in nodes
        },
        edges={
            e.id: viz.EdgeLayout(
                label=viz.DiagramLabel(
                    rf"$W_{{{e.target},{e.source}}}"
                    + ("=0$" if kind == "coba" and e.source != "X" else "$")
                ),
                **(
                    dict(source_port="c", target_port="a")
                    if e.id == "E_to_I"
                    else dict(source_port="b", target_port="d")
                    if e.id == "I_to_E"
                    else dict(source_port="right", target_port="left")
                ),
            )
            for e in edges
        },
        defaults=viz.LayoutDefaults(
            node=viz.NodeLayout(content_align="center", padding=16), gap=30
        ),
    )
    with matplotlib.rc_context({"savefig.bbox": None}):
        return viz.resolve_diagram(
            diagram, layout=layout, theme=viz.DiagramTheme(secondary_size=13)
        )


def _draw_schematic(ax, kind):
    """Place SNNLab's annotated illustration above the unchanged scientific panels."""
    with TemporaryDirectory(prefix="exp023-diagram-") as temporary:
        path = Path(temporary) / "circuit.png"
        with matplotlib.rc_context({"savefig.bbox": None}):
            viz.render_diagram(overview_diagram(kind), path, scale=2)
        ax.imshow(plt.imread(path))
    ax.axis("off")


def plot_architecture(out_path):
    """Render both illustrative loop-condition diagrams with SNNLab."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for kind, axis in zip(("coba", "ping"), axes, strict=True):
        _draw_schematic(axis, kind)
        axis.set_title(
            "COBA: loop off" if kind == "coba" else "PING: loop on", loc="left"
        )
    save_figure(figure, out_path)
    plt.close(figure)


def redraw_overview(analysis, compute, export):
    """Reuse recorded spikes and saved measurements while replacing diagram panels."""
    cfg = inputs.configuration(compute)
    results = load_json(analysis.file("results.json"))
    with np.load(analysis.file("spectra.npz"), allow_pickle=False) as arrays:
        spectra = {
            cell: {
                key: np.array(arrays[f"{cell}__{key}"])
                for key in ("frequency_hz", "density")
            }
            for cell in cfg["cells"]
        }
    recordings = {}
    for cell in cfg["cells"]:
        with np.load(
            compute.file("scope", cell, "recording.npz"), allow_pickle=False
        ) as arrays:
            recordings[cell] = {
                key: np.array(arrays[key]) for key in ("spk_e", "spk_i", "dt")
            }
    titles = {
        "coba": "COBA — recurrent loop off",
        "ping": "PING — recurrent loop active",
    }
    theme.set_paper_mode(True)
    plot_raster_compound(
        recordings,
        results["fi_curves"],
        export / "overview_compound",
        titles,
        spectra,
        results["f_gamma_hz"],
        results["measurement"]["frequency_band_hz"],
        include_arch=True,
    )
    plot_architecture(export / "architecture")


if __name__ == "__main__":
    if any(arg.startswith("--redraw-from") for arg in sys.argv[1:]):
        parser = argparse.ArgumentParser(
            description="Refresh only diagrams of an explicit presentation."
        )
        parser.add_argument("--redraw-from", dest="presentation", required=True)
        parser.add_argument("--run-id")
        print(redraw_networks(**vars(parser.parse_args())))
    else:
        main()
