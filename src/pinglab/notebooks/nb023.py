"""Notebook runner for entry 023 — PING fundamentals.

Runs the oscilloscope in *image* mode against a canonical PING recipe
(no training, synthetic-conductance drive) and captures the resulting
oscilloscope snapshot — rasters, rates, conductance, PSD, weight
histograms — into the figures directory. The point of this entry is
to show the fundamental dynamics in one frame: gamma rhythm, E-then-I
phase, the PSD peak in the gamma band.

Later iterations of this notebook will sweep --ei-strength and the
synaptic time constants for a richer characterisation; for now we
ship one canonical frame.

Notebook entry: src/docs/src/pages/notebooks/nb023.mdx
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _modal import parse_modal_gpu  # noqa: E402
from _run_id import next_run_id, persist as persist_run_id  # noqa: E402
from _tier import parse_tier  # noqa: E402

SLUG = "nb023"
ARTIFACTS = REPO / "src" / "artifacts" / "notebooks" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
OSCILLOSCOPE = REPO / "src" / "pinglab" / "oscilloscope/__main__.py"
SCOPE_OUT = REPO / "src" / "artifacts" / "oscilloscope" / "snapshot.png"

# Two cells driven by MNIST digit 0 sample 0 with constant Poisson
# encoding across the whole 400 ms integration window (no step-on /
# step-off stim). Only the recurrent E-I loop differs:
#
#   ping  — --ei-strength 1.5: gamma rhythm active.
#   coba  — --ei-strength 0  : no recurrent inhibition. Feedforward-only
#                              conductance-based LIF. Sometimes called
#                              "PING off" in the article series.
#
# Same recipe in every other respect so the two snapshots can be
# compared side-by-side.
COMMON_ARGS = [
    "image",
    "--model", "ping",
    "--input", "dataset",
    "--dataset", "mnist",
    "--digit", "0",
    "--sample", "0",
    "--w-in", "1.5", "0.3",
    "--input-rate", "50",
    "--t-ms", "400",
]
CELLS: dict[str, list[str]] = {
    "coba": ["--ei-strength", "0"],
    "ping": ["--ei-strength", "1.5"],
}

TIER_CONFIG = {
    "extra small": {},
    "small": {},
    "medium": {},
    "large": {},
    "extra large": {},
}
DEFAULT_TIER = "small"


def main() -> None:
    tier = parse_tier(sys.argv, choices=TIER_CONFIG.keys(), default=DEFAULT_TIER)
    modal_gpu = parse_modal_gpu(sys.argv)
    wipe_dir = "--no-wipe-dir" not in sys.argv

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id} tier={tier}")

    if wipe_dir:
        for d in (ARTIFACTS, FIGURES):
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)

    snapshots: dict[str, Path] = {}
    for cell, extra in CELLS.items():
        if SCOPE_OUT.exists():
            SCOPE_OUT.unlink()
        scope_argv = [*COMMON_ARGS, *extra]
        cmd = ["uv", "run", "python", str(OSCILLOSCOPE), *scope_argv]
        print(f"[scope] {cell}: {' '.join(scope_argv)}")
        subprocess.run(cmd, cwd=REPO, check=True)
        if not SCOPE_OUT.exists():
            raise SystemExit(f"oscilloscope did not produce {SCOPE_OUT}")
        dst = FIGURES / f"snapshot__{cell}.png"
        shutil.copy2(SCOPE_OUT, dst)
        snapshots[cell] = dst
        print(f"wrote {dst}")

    duration_s = time.monotonic() - t_start
    figs_root = FIGURES.parents[2]
    summary = {
        "notebook_run_id": notebook_run_id,
        "duration_s": round(duration_s, 1),
        "tier": tier,
        "config": {
            "tier": tier,
            "model": "ping",
            "input": "synthetic-conductance",
            "cells": list(CELLS),
            "t_ms": 400,
            "modal_gpu": modal_gpu,
        },
        "results": [],
        "success_criteria": [
            {
                "label": f"{cell} snapshot rendered",
                "passed": dst.exists() and dst.stat().st_size > 0,
                "detail": f"{dst.name} ({dst.stat().st_size} bytes)" if dst.exists() else "missing",
                "detail_href": "/" + str(dst.relative_to(figs_root)) if dst.exists() else None,
            }
            for cell, dst in snapshots.items()
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {FIGURES / 'numbers.json'}")
    print(f"  total duration: {duration_s:.1f}s")


if __name__ == "__main__":
    main()
