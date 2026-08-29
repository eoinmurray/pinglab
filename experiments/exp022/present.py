"""Exp022 presentation: draw saved analysis; never train, simulate or measure."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "experiments"), str(REPO / "tools")]

from experiments.exp022 import recipe
from helpers import theme
from helpers.stamp import stamp_figure
from pingstore.contracts import PingstoreError, file_sha256, load_json, write_json_atomic
from pingstore.stages import SourceRun, source_run, stage_run

def plot_family_curves(family: str, cells: list[dict],
                       out_path: Path, run_id: str) -> int:
    """One figure for one family: each cell's validation-accuracy learning curve,
    coloured by the swept value. Returns the number of cells actually drawn."""
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D

    theme.apply()
    plt.rcParams["savefig.bbox"] = "standard"
    tags = list(dict.fromkeys(c["tag"] for c in cells))  # ordered unique
    # cm.viridis exists at runtime; the matplotlib stub omits it (false positive).
    colours = {t: cm.viridis(i / max(1, len(tags) - 1))  # ty: ignore[unresolved-attribute]
               for i, t in enumerate(tags)}
    # ping (and ping-init) solid, coba dashed — distinguishes the two models
    # in families that train both (rate target, canonical).
    linestyle = {"coba": "--", "ping": "-", "ping_init": "-"}
    models = list(dict.fromkeys(c["model"] for c in cells))

    fig, ax = plt.subplots(figsize=(6.5, 3.66))   # H11–H12: column width, 16:9
    n = 0
    for c in cells:
        eps, accs = c["epochs"], c["accuracy_pct"]
        if eps:
            ax.plot(eps, accs, lw=1.1, color=colours[c["tag"]],
                    ls=linestyle.get(c["model"], "-"), alpha=0.85)
            n += 1
    handles = [Line2D([0], [0], color=colours[t], lw=2.4, label=t) for t in tags]
    leg1 = ax.legend(handles=handles, frameon=False, fontsize=theme.SIZE_LEGEND,
                     ncol=2, loc="lower right", title="swept value")
    ax.add_artist(leg1)
    if len(models) > 1:
        mh = [Line2D([0], [0], color=theme.MUTED, lw=2.0,
                     ls=linestyle.get(m, "-"), label="ping" if m == "ping_init" else m)
              for m in models]
        ax.legend(handles=mh, frameon=False, fontsize=theme.SIZE_LEGEND,
                  loc="lower center", title="model")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy (%)")
    ax.set_ylim(0, 100)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # H11: no plot title — the Typst caption carries the family + takeaway.
    fig.tight_layout()
    stamp_figure(fig, run_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)   # H10: line plot → SVG (caller passes .svg); dpi from theme
    plt.close(fig)
    return n


def _plot_snapshot_raster(snap_path: Path, out_png: Path) -> None:
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
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.15], width_ratios=[3, 1],
                          hspace=0.32, wspace=0.20)
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
    gtxt = f"f_γ ≈ {fgam:.0f} Hz" if fgam else "asynchronous (no γ)"
    ax.annotate(f"digit 0 · E {e_hz:.0f} Hz · I {i_hz:.0f} Hz · {gtxt}",
                xy=(0, 1.02), xycoords="axes fraction", fontsize=theme.SIZE_ANNOTATION,
                color=theme.MUTED, ha="left", va="bottom")

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
            ax3.annotate(f"{fgam:.0f} Hz", xy=(fgam, 1.0), xytext=(5, -3),
                         textcoords="offset points", fontsize=9, fontweight="bold")
        ax3.set_ylim(0, 1.3)
    else:
        ax3.text(0.5, 0.5, "I silent\n(no γ loop)", ha="center", va="center",
                 transform=ax3.transAxes, fontsize=9, color=theme.MUTED)
    ax3.set_xlim(0, 120)
    ax3.set_xlabel("freq (Hz)")
    ax3.set_ylabel("I PSD (norm)")

    for a in (ax, ax2, ax3):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)   # PNG (dense raster, H10); dpi 240 from theme (H11)
    plt.close(fig)
    d.close()



def comparison_rasters(rasters: Path, destination: Path) -> None:
    """Draw four already-measured fixed probes; no inference or rate calculation."""
    grid = [
        ("COBA", "coba__canonical__seed42", "coba__off__seed42"),
        ("PING", "ping__canonical__seed42", "ping__off__seed42"),
    ]
    theme.apply()
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.06),
                             gridspec_kw={"hspace": 0.30, "wspace": 0.14})
    for row, (label, full, reduced) in enumerate(grid):
        for column, name in enumerate((full, reduced)):
            ax = axes[row][column]
            with np.load(rasters / f"{name}.npz", allow_pickle=False) as data:
                ne, ni = int(data["ne"]), int(data["ni"])
                ax.scatter(data["e_times"], data["e_cells"], s=0.5,
                           c=theme.INK_BLACK, marker="|", linewidths=0.25)
                ax.scatter(data["i_times"], data["i_cells"] + ne, s=0.5,
                           c=theme.DEEP_RED, marker="|", linewidths=0.25)
                ax.axhline(ne, color="k", lw=0.4, alpha=0.4)
                ax.set_ylim(0, ne + ni)
                ax.set_xlim(0, float(data["duration_ms"]))
                ax.annotate(f"E {float(data['e_hz']):.0f} Hz · I {float(data['i_hz']):.0f} Hz",
                            xy=(0, 1.01), xycoords="axes fraction",
                            fontsize=theme.SIZE_ANNOTATION, color=theme.MUTED)
            if row == 0:
                ax.set_title(("Full training pool", "Reduced training pool")[column],
                             fontsize=theme.SIZE_LABEL)
            if column == 0:
                ax.set_ylabel(f"{label}\nneuron (E · I)")
            if row == 1:
                ax.set_xlabel("time (ms)")
            ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(destination)
    plt.close(fig)


def carry_historical(retained: SourceRun, filename: str, destination: Path) -> dict:
    """Copy a retained image, preserving its lineage through prior presentations."""
    source = retained.presentation / filename
    if not source.is_file():
        raise PingstoreError(f"historical image is missing: {filename}")
    checksum = file_sha256(source)
    lineage = {"file": filename, "operation": "carry-historical",
               "source_run": retained.record["run_id"],
               "source_path": source.relative_to(retained.directory).as_posix(),
               "sha256": checksum}
    if retained.record.get("stage") == "present":
        records = [item for item in retained.record.get("presentation_lineage", [])
                   if item["file"] == filename]
        if (len(records) != 1 or records[0].get("operation") != "carry-historical"
                or records[0].get("sha256") != checksum):
            raise PingstoreError(f"historical image lineage does not match: {filename}")
        lineage["source_lineage"] = records[0]
    shutil.copy2(source, destination)
    return lineage


def present(identity: str, *, retained_presentation: str | None = None,
            run_id: str | None = None) -> str:
    root = REPO / ".pingstore"
    analysis = source_run(root, identity, stage="analyse", experiment=recipe.SLUG)
    results = load_json(analysis.export / "results.json")
    curves = load_json(analysis.export / "curves.json")["cells"]
    analysis_path = analysis.export.relative_to(analysis.directory).as_posix()
    inputs = {"analysis": analysis}
    retained = None
    if retained_presentation:
        retained = source_run(root, retained_presentation, stage="present", experiment=recipe.SLUG)
        # Accept the original import or a prior presentation of the exact same
        # analysis and bank. An experiment-name match alone is insufficient.
        compute_ref = (analysis.record["inputs"].get("bank")
                       or analysis.record["inputs"]["compute"])
        bank = source_run(root, compute_ref["run_id"], stage="compute",
                          experiment=recipe.SLUG, reference=compute_ref)
        same_evidence = (
            retained.record.get("stage") == "present"
            and retained.record.get("inputs", {}).get("analysis") == analysis.reference
            and retained.record.get("inputs", {}).get("bank") == bank.reference
        )
        if bank.record["inputs"].get("import") != retained.reference and not same_evidence:
            raise PingstoreError("retained presentation must belong to this bank's exact import source")
        inputs["bank"] = bank
        inputs["retained_presentation"] = retained
    missing = results["missing_raw_rasters"]
    if missing and retained is None:
        raise PingstoreError(
            "raw raster evidence is missing; explicitly provide --retained-presentation "
            "for the historical import, or run compute.py --source BANK --diagnostics "
            "and then analyse that new compute run"
        )
    with stage_run(REPO, recipe.SLUG, "present", inputs=inputs, run_id=run_id,
                   configuration={"analysis": identity}) as run:
        lineage = []
        for family in recipe.FAMILY_ORDER:
            cells = [cell for cell in curves if cell["family"] == family]
            filename = f"curves__{recipe.FAMILY_ARTIFACT_SLUGS.get(family, family)}.svg"
            plot_family_curves(family, cells, run.export / filename, run.run_id)
            lineage.append({"file": filename, "operation": "render",
                            "source_run": identity, "source_path": f"{analysis_path}/curves.json"})
        for cell in recipe.CANONICAL_CELLS:
            if cell["seed"] != 42:
                continue
            name = cell["name"]
            filename = f"rasters__{name}.png"
            if name in results["rasters"]:
                _plot_snapshot_raster(analysis.export / "rasters" / f"{name}.npz",
                                      run.export / filename)
                lineage.append({"file": filename, "operation": "render",
                                "source_run": identity,
                                "source_path": f"{analysis_path}/rasters/{name}.npz"})
            else:
                if retained is None:
                    raise PingstoreError(f"missing raster source for {name}")
                lineage.append(carry_historical(retained, filename, run.export / filename))
        comparison_cells = {
            "coba__canonical__seed42", "coba__off__seed42",
            "ping__canonical__seed42", "ping__off__seed42",
        }
        filename = "comparison__data_fraction.png"
        if comparison_cells <= set(results["rasters"]):
            comparison_rasters(analysis.export / "rasters", run.export / filename)
            lineage.append({"file": filename, "operation": "render", "source_run": identity})
        elif retained is not None:
            lineage.append(carry_historical(retained, filename, run.export / filename))
        numbers = dict(results)
        numbers["analysis_run_id"] = identity
        numbers["notebook_run_id"] = run.run_id
        numbers["presentation_lineage"] = lineage
        write_json_atomic(run.export / "numbers.json", numbers)
        run.record["presentation_lineage"] = lineage
        (run.directory / "README.md").write_text(
            "# Exp022 presentation\n\n"
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
    parser.add_argument("--source", required=True, help="completed exp022 analyse run ID")
    parser.add_argument("--retained-presentation",
                        help="original import or prior presentation of the same analysis and bank")
    parser.add_argument("--run-id", help="already reserved present identity")
    args = parser.parse_args()
    present(args.source, retained_presentation=args.retained_presentation, run_id=args.run_id)


if __name__ == "__main__":
    main()
