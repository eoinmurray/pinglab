"""Notebook runner for entry 010 — PING integration-step sweep.

Sweeps the integration timestep *dt* from 0.05 → 2 ms with the input
overdrive held well past the PING threshold (fixed 10×), isolating dt
as the only knob that can break the rhythm. Fine dt is stable; coarse
dt distorts or saturates. Split out from the original nb002 basic-PING
notebook; companion to nb002 (stim-overdrive) and nb011 (ei-strength).

Notebook entry: src/docs/src/pages/notebooks/nb010.mdx
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _ping_scan import (  # noqa: E402
    INPUT_RATE_HZ, W_IN_OVERDRIVE,
    ScanSpec, run_scan,
)

SLUG = "nb010"
DT_SCAN_OVERDRIVE = 10.0  # pinned above PING threshold so dt is the only variable
DT_SCAN_MIN = 0.05
DT_SCAN_MAX = 2.0


if __name__ == "__main__":
    run_scan(ScanSpec(
        slug=SLUG,
        scan_var="dt",
        scan_min=DT_SCAN_MIN,
        scan_max=DT_SCAN_MAX,
        video_name="scan_dt.mp4",
        extra_osc_args=[
            "--input-rate", str(INPUT_RATE_HZ),
            "--w-in-overdrive", str(W_IN_OVERDRIVE),
            "--stim-overdrive", str(DT_SCAN_OVERDRIVE),
        ],
        config_payload={"fixed_overdrive": DT_SCAN_OVERDRIVE},
    ))
    sys.exit(0)
