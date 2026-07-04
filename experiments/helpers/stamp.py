"""Run-id stamping for notebook figures and videos.

`stamp_figure` writes the monotonic run id (e.g. "r007") into the bottom-right
corner of a matplotlib figure using the shared pinglab-cli theme.
`render_stamp_png` + `overlay_stamp_video` produce the same stamp as a
transparent PNG and burn it into the corner of an mp4 via ffmpeg, for the
video runners that can't annotate a figure directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .paths import REPO


def stamp_figure(fig, run_id: str) -> None:
    """Stamp run_id into the bottom-right corner of a matplotlib figure."""
    from . import theme

    fig.text(
        0.995,
        0.005,
        run_id,
        ha="right",
        va="bottom",
        fontsize=theme.SIZE_CAPTION,
        color=theme.LABEL,
        family="monospace",
    )


def render_stamp_png(run_id: str, stamp_path: Path) -> None:
    """Render run_id as a small transparent PNG for overlaying onto a video."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(2.8, 0.28), dpi=150)
    fig.patch.set_alpha(0.0)
    fig.text(
        0.97,
        0.5,
        run_id,
        ha="right",
        va="center",
        fontsize=10,
        color="white",
        family="monospace",
        bbox=dict(facecolor="black", alpha=0.55, pad=3, edgecolor="none"),
    )
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stamp_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def overlay_stamp_video(src: Path, dst: Path, stamp: Path) -> None:
    """Burn the stamp PNG into the bottom-right corner of src, write to dst."""
    import sh

    dst.parent.mkdir(parents=True, exist_ok=True)
    sh.ffmpeg(  # ty: ignore[unresolved-attribute]  # sh generates command attrs at runtime
        "-y",
        "-i",
        str(src),
        "-i",
        str(stamp),
        "-filter_complex",
        "[0:v][1:v]overlay=W-w-10:H-h-10",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-movflags",
        "+faststart",
        str(dst),
        _out=sys.stdout,
        _err=sys.stderr,
    )
    print(f"wrote {dst.relative_to(REPO)}")
