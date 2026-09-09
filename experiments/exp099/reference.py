"""Extract the published PING column into an explicit presentation."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp099 import inputs, recipe
from pingstore.stages import stage_run


def present_reference(identity):
    previous = inputs.source(REPO, identity, "present")
    pdf = REPO / "papers/Susin-Destexhe-2021.pdf"
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    with stage_run(
        REPO,
        recipe.SLUG,
        "present",
        inputs={"presentation": previous},
        configuration=inputs.configuration(previous),
    ) as run:
        for path in previous.export.iterdir():
            shutil.copy2(path, run.export / path.name)
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
# Assemble a PDF clipping region retaining B/C and excluding adjacent artwork.
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
                str(pdf),
                str(run.export / "susin-destexhe-2021-ping.png"),
            ],
            check=True,
        )
        if hashlib.sha256(pdf.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Reference PDF changed during extraction")
        run.record["execution"]["reference_crop"] = {
            "doi": "10.1371/journal.pcbi.1009416",
            "pdf_sha256": digest,
            "page": 13,
            "figure": "4, PING column B–C",
            "dpi": 300,
            "crop_xywh_pixels": [375, 555, 710, 540],
            "license": "CC BY 4.0",
            "modification": "cropped B/C; upper-right adjacent A fragment excluded with PDF clipping",
        }
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    present_reference(parser.parse_args().source)
