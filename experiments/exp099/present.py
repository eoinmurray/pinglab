"""Render an explicit analysis and pinned compute source; never publish."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
import numpy as np
from experiments.exp099 import inputs, recipe
from experiments.exp099.render import network_diagram, render, render_raster
from pingstore.contracts import PingstoreError, load_json, write_json_atomic
from pingstore.stages import stage_run
from tools import snnlang as snn  # noqa: TID251
from tools.snnviz import Recording  # noqa: TID251

REFERENCE_PDF = REPO / "papers/Susin-Destexhe-2021.pdf"


def render_reference(source: Path, destination: Path) -> dict:
    """Crop the comparison panels from the locally retained CC BY reference."""
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    subprocess.run(
        [
            "uv",
            "run",
            "--with",
            "pymupdf",
            "python",
            "-c",
            """
import pymupdf as fitz
import sys
source = fitz.open(sys.argv[1])
scale = 300 / 72
x, y, width, height = 375, 555, 710, 540
out = fitz.open()
page = out.new_page(width=width / scale, height=height / scale)
for left, top, right, bottom in [(0, 0, 100, 45), (0, 45, width, height)]:
    target = fitz.Rect(left / scale, top / scale, right / scale, bottom / scale)
    clip = fitz.Rect((x + left) / scale, (y + top) / scale,
                    (x + right) / scale, (y + bottom) / scale)
    page.show_pdf_page(target, source, 12, clip=clip)
page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(sys.argv[2])
""",
            str(source),
            str(destination),
        ],
        check=True,
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError("reference PDF changed during extraction")
    return {
        "doi": "10.1371/journal.pcbi.1009416",
        "pdf_sha256": digest,
        "page": 13,
        "figure": "4, PING column B–C",
        "dpi": 300,
        "crop_xywh_pixels": [375, 555, 710, 540],
        "license": "CC BY 4.0",
        "modification": "cropped B/C; upper-right adjacent A fragment excluded with PDF clipping",
    }


def resolved(identity):
    analysis = inputs.source(REPO, identity, "analyse")
    cfg = inputs.configuration(analysis)
    refs = analysis.record["inputs"]
    if set(refs) != {"compute"}:
        raise PingstoreError("exp099 analysis must pin one compute input")
    compute = inputs.source(
        REPO, refs["compute"]["run_id"], "compute", reference=refs["compute"]
    )
    results = load_json(analysis.export / "results.json")
    if (
        inputs.configuration(compute) != cfg
        or results.get("parameters") != cfg
        or results.get("schema") != "exp099.analysis/v2"
    ):
        raise PingstoreError("exp099 analysis configuration disagrees with compute")
    return analysis, compute, cfg, results


def render_outputs(
    analysis, compute, cfg, results, output, *, preview_only=False, view=None
):
    recording = Recording(cfg["dt_ms"], inputs.recording(compute))
    with np.load(analysis.export / "measurements.npz", allow_pickle=False) as arrays:
        measurements = dict(arrays)
    with np.load(compute.export / "weights.npz", allow_pickle=False) as arrays:
        weights = dict(arrays)
    render(
        recording,
        weights,
        measurements,
        cfg,
        output,
        preview_only=preview_only,
        view=view,
    )
    network_diagram(
        cfg,
        weights,
        output / "network.svg",
        bundle=snn.load_bundle(compute.unit("network.bundle")),
    )
    display_cfg = dict(cfg)
    if view is not None:
        display_cfg.update(
            view_start_ms=view["start_ms"], view_end_ms=view["end_ms"]
        )
    render_raster(recording, display_cfg, output / "spike-raster.png")
    reference = render_reference(
        REFERENCE_PDF, output / "susin-destexhe-2021-ping.png"
    )
    write_json_atomic(output / "numbers.json", results)
    centre = (cfg["peak_ms"] + cfg["plateau_end_ms"]) / 2
    return {
        "raster": {
            "neurons": "all E and I",
            "interval_ms": [
                display_cfg["view_start_ms"],
                display_cfg["view_end_ms"],
            ],
            "display_time_origin_ms": cfg.get("burn_in_ms", 0.0),
            "source_video": recipe.VIDEO,
            "closeup_ms": [centre - 50, centre + 50],
            "count_bin_ms": 1.0,
            "count_smoothing": "none",
        },
        "reference_crop": reference,
    }


def present(identity, *, run_id=None):
    analysis, compute, cfg, results = resolved(identity)
    view = {
        "start_ms": cfg["view_start_ms"],
        "end_ms": cfg["view_end_ms"],
        "frames": 625,
    }
    if not 0 <= view["start_ms"] < view["end_ms"] <= cfg["t_ms"]:
        raise ValueError("presentation interval must lie within the recording")
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs={"analysis": analysis, "compute": compute},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        run.record["execution"]["presentation"] = view
        run.record["execution"].update(
            render_outputs(analysis, compute, cfg, results, run.export, view=view)
        )
    return run.run_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    a = p.parse_args()
    present(a.source, run_id=a.run_id)


if __name__ == "__main__":
    main()
