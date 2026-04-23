"""Notebook runner for entry 011 — PING E→I coupling sweep.

Sweeps the E→I coupling strength from 0 → 1 with *no* stim-window
overdrive (input rate flat through the trial), walking the network
from the async baseline (E and I effectively decoupled) through the
emergence of gamma as the E→I→E feedback loop closes. Input rate and
W_in are bumped relative to the other scans so E has enough baseline
drive to recruit I at all. Split out from the original nb002
basic-PING notebook; companion to nb002 (stim-overdrive) and nb010
(dt).

Notebook entry: src/docs/src/pages/notebooks/nb011.mdx
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

from _ping_scan import (  # noqa: E402
    DT_MS, INPUT_RATE_HZ,
    ScanSpec, run_scan,
)

SLUG = "nb011"
EI_SCAN_INPUT_RATE_HZ = 2 * INPUT_RATE_HZ
EI_SCAN_W_IN_OVERDRIVE = 3.0
EI_SCAN_MIN = 0.0
EI_SCAN_MAX = 1.0   # past 1 the E rate is already saturated-low


if __name__ == "__main__":
    run_scan(ScanSpec(
        slug=SLUG,
        scan_var="ei_strength",
        scan_min=EI_SCAN_MIN,
        scan_max=EI_SCAN_MAX,
        video_name="scan_ei.mp4",
        extra_osc_args=[
            "--input-rate", str(EI_SCAN_INPUT_RATE_HZ),
            "--w-in-overdrive", str(EI_SCAN_W_IN_OVERDRIVE),
            "--stim-overdrive", "1.0",
            "--dt", str(DT_MS),
        ],
        config_payload={
            "fixed_overdrive": 1.0,
            "input_rate_hz": EI_SCAN_INPUT_RATE_HZ,
            "w_in_overdrive": EI_SCAN_W_IN_OVERDRIVE,
        },
    ))
    sys.exit(0)
