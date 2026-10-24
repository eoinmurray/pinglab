"""Run the independent present stage for exp022.

Invoke this CLI from the repository root. The experiment-local implementation
owns argument validation, explicit source pins and Pingstore stage completion.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import argparse
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import theme
from experiments.exp022 import recipe
from experiments.exp022.custom.network_artifacts import copy_circuits
from pingstore.contracts import (
    PingstoreError,
    file_sha256,
    load_json,
    write_json_atomic,
)
from pingstore.stages import SourceRun, source_run, stage_run


def draw_family_curves(family: str, cells: list[dict], ax) -> int:
    """One figure for one family: each cell's validation-accuracy learning curve,
    coloured by the swept value. Returns the number of cells actually drawn."""
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D

    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"
    tags = list(dict.fromkeys(c["tag"] for c in cells))  # ordered unique
    tag_labels = {
        "frozen_ping": "fixed weights",
        "trainable_ping_init": "trained: standard init",
        "trainable_zero_init": "trained: zero init",
        "trainable_small_init": "trained: small init",
    }
    # cm.viridis exists at runtime; the matplotlib stub omits it (false positive).
    colours = {
        t: cm.viridis(i / max(1, len(tags) - 1))  # ty: ignore[unresolved-attribute]
        for i, t in enumerate(tags)
    }
    # ping (and ping-init) solid, coba dashed — distinguishes the two models
    # in families that train both (rate target, canonical).
    linestyle = {"coba": "--", "ping": "-", "ping_init": "-"}
    models = list(dict.fromkeys(c["model"] for c in cells))

    n = 0
    for c in cells:
        eps, accs = c["epochs"], c["accuracy_pct"]
        if eps:
            ax.plot(
                eps,
                accs,
                lw=1.1,
                color=colours[c["tag"]],
                ls=linestyle.get(c["model"], "-"),
                alpha=0.85,
            )
            n += 1
    handles = [
        Line2D(
            [0],
            [0],
            color=colours[t],
            lw=2.4,
            label=f"{t} ms" if family == "dt" else tag_labels.get(t, t),
        )
        for t in tags
    ]
    leg1 = ax.legend(
        handles=handles,
        frameon=False,
        fontsize=theme.SIZE_LEGEND,
        ncol=2,
        loc="lower right",
        title="swept value",
    )
    ax.add_artist(leg1)
    if len(models) > 1:
        mh = [
            Line2D(
                [0],
                [0],
                color=theme.MUTED,
                lw=2.0,
                ls=linestyle.get(m, "-"),
                label="ping" if m == "ping_init" else m,
            )
            for m in models
        ]
        ax.legend(
            handles=mh,
            frameon=False,
            fontsize=theme.SIZE_LEGEND,
            loc="lower center",
            title="model",
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_ylim(0, 100)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # H11: no plot title — the Typst caption carries the family + takeaway.
    return n


def plot_family_curves(family: str, cells: list[dict], out_path: Path) -> int:
    theme.apply()
    fig, ax = plt.subplots(figsize=(6.5, 3.66))
    n = draw_family_curves(family, cells, ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return n


TRAINING_OVERVIEW_RASTERS = {
    "canonical": ("ping__canonical__seed42", "PING · full training pool"),
    "activity_frontier": ("ping__rt1hz__seed42", "PING · 1 Hz activity ceiling"),
    "tau_gaba": ("ping__tg6__seed42", "PING · inhibitory decay 6 ms"),
    "dt": ("ping__dt0p1__seed42", "PING · timestep 0.1 ms"),
    "init": ("frozen_ping__seed42", "PING · fixed recurrent weights"),
    "variable_rate": ("ping__variable_rate__seed42", "PING · variable-rate training"),
    "low_w_in": ("ping__low_w_in__win0p05__seed42", "PING · input coupling 0.05"),
}


def plot_training_overview(
    family: str, cells: list[dict], raster: Path, destination: Path
) -> int:
    """Draw saved validation histories beside one explicitly selected fixed probe."""
    from matplotlib.lines import Line2D

    theme.apply()
    fig, (curves, spikes) = plt.subplots(1, 2, figsize=(9, 5.06))
    count = draw_family_curves(family, cells, curves)
    with np.load(raster, allow_pickle=False) as data:
        ne, ni = int(data["ne"]), int(data["ni"])
        spikes.scatter(
            data["e_times"],
            data["e_cells"],
            s=1,
            c=theme.INK_BLACK,
            marker="|",
            linewidths=0.35,
        )
        spikes.scatter(
            data["i_times"],
            data["i_cells"] + ne,
            s=1,
            c=theme.DEEP_RED,
            marker="|",
            linewidths=0.35,
        )
        spikes.axhline(ne, color=theme.MUTED, lw=0.5)
        spikes.set_xlim(0, float(data["duration_ms"]))
        spikes.set_ylim(0, ne + ni)
    spikes.set_xlabel("time (ms)")
    spikes.set_ylabel("neuron")
    spikes.set_yticks([0, 512, 1024, 1280])
    spikes.spines[["top", "right"]].set_visible(False)
    spikes.legend(
        handles=[
            Line2D(
                [0], [0], color=theme.INK_BLACK, marker="|", ls="", label="excitatory"
            ),
            Line2D(
                [0], [0], color=theme.DEEP_RED, marker="|", ls="", label="inhibitory"
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.29),
        ncol=2,
        frameon=False,
        fontsize=theme.SIZE_LEGEND,
    )
    spikes.text(
        0,
        1.04,
        TRAINING_OVERVIEW_RASTERS[family][1],
        transform=spikes.transAxes,
        fontsize=theme.SIZE_ANNOTATION,
        color=theme.MUTED,
    )
    # Put condition legends below the curves so they do not cover early epochs.
    legends = dict.fromkeys(
        child
        for child in curves.get_children()
        if isinstance(child, matplotlib.legend.Legend)
    )
    for legend in legends:
        if legend.get_title().get_text() == "swept value":
            legend.set_bbox_to_anchor((0.5, -0.17))
            legend.set_loc("upper center")
            legend.set_title(None)
        else:
            legend.set_bbox_to_anchor((0.5, 1.03))
            legend.set_loc("lower center")
            legend.set_title(None)
            legend.set_ncols(2)
    theme.label_panels((curves, spikes))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.30, wspace=0.34)
    fig.savefig(destination)
    plt.close(fig)
    return count


def _plot_snapshot_raster(snap_path: Path, out_path: Path) -> None:
    """Raster + population rate + I-population PSD (γ peak labelled) for a single
    fixed-image snapshot. The PSD describes the exact raster shown."""
    import numpy as np

    d = np.load(snap_path, allow_pickle=False)
    et, ec, it, ic = d["e_times"], d["e_cells"], d["i_times"], d["i_cells"]
    ne, ni, tms = int(d["ne"]), int(d["ni"]), float(d["duration_ms"])
    e_hz, i_hz = float(d["e_hz"]), float(d["i_hz"])
    fgam = float(d["gamma_hz"])
    fgam = None if np.isnan(fgam) else fgam
    fr, P = d["frequencies"], d["normalized_power"]

    theme.apply()
    # H12: stacked multi-panel, column width, height capped so it fits a page.
    fig = plt.figure(figsize=(6.5, 5.0))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[3, 1.15], width_ratios=[3, 1], hspace=0.32, wspace=0.20
    )
    ax = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax.scatter(et, ec, s=1.0, c=theme.INK_BLACK, marker="|", linewidths=0.35)
    ax.scatter(it, ic + ne, s=1.0, c=theme.DEEP_RED, marker="|", linewidths=0.35)
    ax.axhline(ne, color="k", lw=0.4, alpha=0.4)
    ax.set_ylim(0, ne + ni)
    ax.set_xlim(0, tms)
    ax.set_ylabel("neuron  (E below · I above)")
    # H11: no plot title (config descriptor lives in the caption). The measured
    # values stay as a compact data annotation — they're read from the explicit analysis run, not recomputed by presentation.
    gtxt = f"spectral peak ≈ {fgam:.0f} Hz" if fgam else "no resolved spectral peak"
    ax.annotate(
        f"digit 0 · E {e_hz:.0f} Hz · I {i_hz:.0f} Hz · {gtxt}",
        xy=(0, 1.02),
        xycoords="axes fraction",
        fontsize=theme.SIZE_ANNOTATION,
        color=theme.MUTED,
        ha="left",
        va="bottom",
    )

    ax2.plot(d["bin_centres"], d["e_rate"], c=theme.INK_BLACK, lw=0.7, label="E")
    ax2.plot(d["bin_centres"], d["i_rate"], c=theme.DEEP_RED, lw=0.7, label="I")
    ax2.set_xlabel("time (ms)")
    ax2.set_ylabel("Hz/cell")
    ax2.set_xlim(0, tms)
    ax2.legend(loc="upper right", frameon=False, ncol=2, fontsize=8)

    if len(fr):
        ax3.plot(fr, P, c=theme.DEEP_RED, lw=1.0)
        if fgam:
            ax3.axvline(fgam, color=theme.INK_BLACK, ls="--", lw=0.9)
            ax3.annotate(
                f"{fgam:.0f} Hz",
                xy=(fgam, 1.0),
                xytext=(5, -3),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )
        ax3.set_ylim(0, 1.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "I silent" if i_hz == 0 else "too few\nI spikes",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=9,
            color=theme.MUTED,
        )
    ax3.set_xlim(0, 120)
    ax3.set_xlabel("freq (Hz)")
    ax3.set_ylabel("I PSD (norm)")

    for a in (ax, ax2, ax3):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    theme.label_panels((ax, ax2, ax3))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    d.close()


def comparison_rasters(rasters: dict[str, Path], destination: Path) -> None:
    """Draw four already-measured fixed probes; no inference or rate calculation."""
    grid = [
        ("COBA", "coba__canonical__seed42", "coba__off__seed42"),
        ("PING", "ping__canonical__seed42", "ping__off__seed42"),
    ]
    theme.apply()
    fig, axes = plt.subplots(
        2, 2, figsize=(9, 5.06), gridspec_kw={"hspace": 0.30, "wspace": 0.14}
    )
    for row, (label, full, reduced) in enumerate(grid):
        for column, name in enumerate((full, reduced)):
            ax = axes[row][column]
            with np.load(rasters[name], allow_pickle=False) as data:
                ne, ni = int(data["ne"]), int(data["ni"])
                ax.scatter(
                    data["e_times"],
                    data["e_cells"],
                    s=0.5,
                    c=theme.INK_BLACK,
                    marker="|",
                    linewidths=0.25,
                )
                ax.scatter(
                    data["i_times"],
                    data["i_cells"] + ne,
                    s=0.5,
                    c=theme.DEEP_RED,
                    marker="|",
                    linewidths=0.25,
                )
                ax.axhline(ne, color="k", lw=0.4, alpha=0.4)
                ax.set_ylim(0, ne + ni)
                ax.set_xlim(0, float(data["duration_ms"]))
                ax.annotate(
                    f"E {float(data['e_hz']):.0f} Hz · I {float(data['i_hz']):.0f} Hz",
                    xy=(0, 1.01),
                    xycoords="axes fraction",
                    fontsize=theme.SIZE_ANNOTATION,
                    color=theme.MUTED,
                )
            if row == 0:
                ax.set_title(
                    ("Full training pool", "Reduced training pool")[column],
                    fontsize=theme.SIZE_LABEL,
                )
            if column == 0:
                ax.set_ylabel(f"{label}\nneuron (E · I)")
            if row == 1:
                ax.set_xlabel("time (ms)")
            ax.spines[["top", "right"]].set_visible(False)
    theme.label_panels(axes.flat)
    fig.savefig(destination)
    plt.close(fig)


def carry_historical(retained: SourceRun, filename: str, destination: Path) -> dict:
    """Copy a retained image, preserving its lineage through prior presentations."""
    source = retained.presentation / filename
    if not source.is_file() and source.suffix == ".svg":
        source = source.with_suffix(".png")
        filename = source.name
        destination = destination.with_suffix(".png")
    if not source.is_file():
        raise PingstoreError(f"historical image is missing: {filename}")
    checksum = file_sha256(source)
    lineage = {
        "file": filename,
        "operation": "carry-historical",
        "source_run": retained.record["run_id"],
        "source_path": source.relative_to(retained.directory).as_posix(),
        "sha256": checksum,
    }
    if retained.record.get("stage") == "present":
        records = [
            item
            for item in retained.record.get("presentation_lineage", [])
            if item["file"] == filename
        ]
        if (
            len(records) != 1
            or records[0].get("operation") != "carry-historical"
            or records[0].get("sha256") != checksum
        ):
            raise PingstoreError(f"historical image lineage does not match: {filename}")
        lineage["source_lineage"] = records[0]
    shutil.copy2(source, destination)
    return lineage


def present(
    identity: str,
    *,
    retained_presentation: str | None = None,
    run_id: str | None = None,
    training_overview: bool = False,
) -> str:
    root = REPO / ".pingstore"
    analysis = source_run(root, identity, stage="analyse", experiment=recipe.SLUG)
    results = load_json(analysis.export / "results.json")
    curves = load_json(analysis.export / "curves.json")["cells"]
    raster_paths = {
        name: analysis.file(results["raster_files"][name])
        if "raster_files" in results
        else analysis.file("rasters", f"{name}.npz")
        for name in results["rasters"]
    }
    analysis_path = analysis.export.relative_to(analysis.directory).as_posix()
    inputs = {"analysis": analysis}
    retained = None
    if retained_presentation:
        retained = source_run(
            root, retained_presentation, stage="present", experiment=recipe.SLUG
        )
        # Accept the original import or a prior presentation of the exact same
        # analysis and bank. An experiment-name match alone is insufficient.
        compute_ref = (
            analysis.record["inputs"].get("bank")
            or analysis.record["inputs"]["compute"]
        )
        bank = source_run(
            root,
            compute_ref["run_id"],
            stage="compute",
            experiment=recipe.SLUG,
            reference=compute_ref,
        )
        same_evidence = (
            retained.record.get("stage") == "present"
            and retained.record.get("inputs", {}).get("analysis") == analysis.reference
            and retained.record.get("inputs", {}).get("bank") == bank.reference
        )
        if (
            bank.record["inputs"].get("import") != retained.reference
            and not same_evidence
        ):
            raise PingstoreError(
                "retained presentation must belong to this bank's exact import source"
            )
        inputs["bank"] = bank
        inputs["retained_presentation"] = retained
    missing = results["missing_raw_rasters"]
    if missing and retained is None and not training_overview:
        raise PingstoreError(
            "raw raster evidence is missing; explicitly provide --retained-presentation "
            "for the historical import, or run compute.py --source BANK --diagnostics "
            "and then analyse that new compute run"
        )
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs=inputs,
        run_id=run_id,
        configuration={
            "analysis": identity,
            "training_overview": training_overview,
            "representative_rasters": {
                family: value[0] for family, value in TRAINING_OVERVIEW_RASTERS.items()
            }
            if training_overview
            else None,
        },
    ) as run:
        lineage = []
        training_figures = {}
        if training_overview:
            for family in recipe.FAMILY_ORDER:
                name, _ = TRAINING_OVERVIEW_RASTERS[family]
                if name not in raster_paths:
                    raise PingstoreError(
                        f"training overview requires measured raster: {name}"
                    )
                cells = [cell for cell in curves if cell["family"] == family]
                training_id = results["training_run_ids"][family]
                filename = f"training__{training_id}.svg"
                count = plot_training_overview(
                    family, cells, raster_paths[name], run.export / filename
                )
                training_figures[family] = {
                    "file": filename,
                    "raster_cell": name,
                    "curves": count,
                }
                lineage.append(
                    {
                        "file": filename,
                        "operation": "render",
                        "source_run": identity,
                        "source_paths": [
                            f"{analysis_path}/curves.json",
                            raster_paths[name]
                            .relative_to(analysis.directory)
                            .as_posix(),
                        ],
                        "raster_cell": name,
                    }
                )
        else:
            for family in recipe.FAMILY_ORDER:
                cells = [cell for cell in curves if cell["family"] == family]
                filename = (
                    f"curves__{recipe.FAMILY_ARTIFACT_SLUGS.get(family, family)}.svg"
                )
                plot_family_curves(family, cells, run.export / filename)
                lineage.append(
                    {
                        "file": filename,
                        "operation": "render",
                        "source_run": identity,
                        "source_path": f"{analysis_path}/curves.json",
                    }
                )
            for cell in recipe.cells_for_names(row["name"] for row in results["cells"]):
                if cell["seed"] != 42:
                    continue
                name = cell["name"]
                filename = f"rasters__{name}.svg"
                if name in results["rasters"]:
                    _plot_snapshot_raster(raster_paths[name], run.export / filename)
                    lineage.append(
                        {
                            "file": filename,
                            "operation": "render",
                            "source_run": identity,
                            "source_path": raster_paths[name]
                            .relative_to(analysis.directory)
                            .as_posix(),
                        }
                    )
                else:
                    if retained is None:
                        raise PingstoreError(f"missing raster source for {name}")
                    lineage.append(
                        carry_historical(retained, filename, run.export / filename)
                    )
            comparison_cells = {
                "coba__canonical__seed42",
                "coba__off__seed42",
                "ping__canonical__seed42",
                "ping__off__seed42",
            }
            filename = "comparison__data_fraction.svg"
            if comparison_cells <= set(results["rasters"]):
                comparison_rasters(raster_paths, run.export / filename)
                lineage.append(
                    {"file": filename, "operation": "render", "source_run": identity}
                )
            elif retained is not None:
                lineage.append(
                    carry_historical(retained, filename, run.export / filename)
                )
        numbers = dict(results)
        if training_overview:
            numbers["training_figures"] = training_figures
        numbers["analysis_run_id"] = identity
        numbers["notebook_run_id"] = run.run_id
        numbers["presentation_lineage"] = lineage
        write_json_atomic(
            run.export / "numbers.json",
            {**(numbers), "circuit_files": copy_circuits(run)},
        )
        run.record["presentation_lineage"] = lineage
        with (run.directory / "README.md").open("a") as readme:
            readme.write(
                "\n## Presentation\n\n"
                f"Rendered analysis `{identity}`. Select `{run.run_id}` in Demolab preview.\n\n"
                "Curves and numbers are produced from saved analysis. Any "
                "`carry-historical` raster in run.json was copied unchanged from the "
                "explicit source because its raw snapshot was not retained. "
                "When reusing a prior presentation, source_lineage retains the earlier image record. "
                "It is not a newly simulated or remeasured result.\n\n"
                "No scientific execution, materialization or publication occurs here.\n"
            )
    return run.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, help="completed exp022 analyse run ID"
    )
    parser.add_argument(
        "--retained-presentation",
        help="original import or prior presentation of the same analysis and bank",
    )
    parser.add_argument("--run-id", help="already reserved present identity")
    parser.add_argument(
        "--training-overview",
        action="store_true",
        help="render seven paired learning-curve and raster figures",
    )
    args = parser.parse_args()
    present(
        args.source,
        retained_presentation=args.retained_presentation,
        run_id=args.run_id,
        training_overview=args.training_overview,
    )


if __name__ == "__main__":
    main()
