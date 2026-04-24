"""Regression test: `_ping_scan._render_video` must forward `--t-ms` to the
oscilloscope CLI. Without it, the video runs at the CLI default (200 ms),
which ends before the 200–300 ms stim window fires — every frame lands in
flat baseline and the raster / PSD / I-population panels look identical
across the whole sweep.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parents[1]
NOTEBOOKS = TESTS.parent / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))


def test_render_video_forwards_t_ms(monkeypatch):
    import _ping_scan
    from _ping_scan import REPO, ScanSpec, SIM_MS, STEP_OFF_MS, _render_video

    captured: dict = {}

    def fake_uv(*args, **kwargs):
        captured["args"] = list(args)

    monkeypatch.setattr(_ping_scan.sh, "uv", fake_uv)

    # out_dir must sit under REPO because _render_video logs with .relative_to(REPO)
    out_dir = REPO / "src" / "artifacts" / "_ping_scan_test"

    spec = ScanSpec(slug="nbTEST", scan_var="stim-overdrive",
                    scan_min=1.0, scan_max=10.0, video_name="scan.mp4")

    with pytest.raises(SystemExit):
        _render_video(spec, out_dir, frames=2, modal_gpu=None)

    args = captured["args"]
    assert "--t-ms" in args, (
        "oscilloscope video call is missing --t-ms; it will default to 200 ms "
        "and the 200–300 ms stim window never fires — every scan frame will "
        "be flat baseline"
    )
    t_ms = float(args[args.index("--t-ms") + 1])
    assert t_ms >= STEP_OFF_MS, (
        f"--t-ms={t_ms} ends before STEP_OFF_MS={STEP_OFF_MS}; stim window "
        f"never completes"
    )
    assert t_ms == SIM_MS
