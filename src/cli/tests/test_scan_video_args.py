"""Regression test: every PING-video scan runner (nb003–nb006) must forward
`--t-ms` to the oscilloscope `sim --video` call. Without it the video runs at
the CLI default (200 ms), which ends before the 200–300 ms stim window fires —
every scan frame lands in flat baseline and the raster / PSD / I-population
panels look identical across the whole sweep.

The scan-video orchestration used to live in a shared `_ping_scan` helper; it
is now inlined into each runner, so this guard checks each runner's source for
the forwarded flag set to the full sim length (SIM_MS = 600).
"""

from __future__ import annotations

from pathlib import Path

import pytest

NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"  # src/notebooks/
SCAN_RUNNERS = ["nb003.py", "nb004.py", "nb005.py", "nb006.py"]


@pytest.mark.parametrize("runner", SCAN_RUNNERS)
def test_scan_runner_forwards_t_ms(runner):
    src = (NOTEBOOKS / runner).read_text()
    assert '"--t-ms"' in src, (
        f"{runner}: oscilloscope video call is missing --t-ms; it will default "
        "to 200 ms and the 200–300 ms stim window never fires — every scan "
        "frame will be flat baseline"
    )
    # The flag must carry the full sim length, not the 200 ms CLI default.
    assert "str(SIM_MS)" in src, (
        f"{runner}: --t-ms must forward SIM_MS (600 ms) so the stim window "
        "completes"
    )
