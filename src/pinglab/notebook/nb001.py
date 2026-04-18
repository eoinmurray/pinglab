"""Notebook runner for entry 001 — scope-frame.

Renders a single SCOPE_FRAME (see src/pinglab/plot.py) for aesthetic
iteration. No sweep, no video — one forward pass at a canonical working
point, saved as a PNG plus a stamped copy into the entry's figures dir.

Invokes `oscilloscope.py image` once with MNIST d0 s0, ping model, at the
same working point that lights up PING in nb002 (overdrive 5×, w_in 1.8×),
then copies the produced snapshot.png into the figures dir with the
notebook_run_id stamped in the corner.

Notebook entry: src/docs/src/pages/notebook/nb001.mdx
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import sh

REPO = Path(__file__).resolve().parents[3]
PINGLAB = REPO / "src" / "pinglab"
sys.path.insert(0, str(PINGLAB))

import matplotlib.pyplot as plt  # noqa: E402

from _run_id import next_run_id, persist as persist_run_id  # noqa: E402

SLUG = "nb001"
ARTIFACTS = REPO / "src" / "artifacts" / "notebook" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebook" / SLUG
OSCILLOSCOPE = PINGLAB / "oscilloscope.py"

# ── Frame config — matches nb002's canonical PING working point ──────────
SEED            = 42
TIER            = "tiny"    # single forward pass — no sweep cost
N_HIDDEN        = 512
DT_MS           = 0.1
SIM_MS          = 600.0
STEP_ON_MS      = 200.0
STEP_OFF_MS     = 300.0
STIM_OVERDRIVE  = 5.0
INPUT_RATE_HZ   = 50.0
W_IN_OVERDRIVE  = 1.8
EI_STRENGTH     = 0.5
DATASET         = "mnist"
DIGIT_CLASS     = 0
SAMPLE_IDX      = 0


def _render_stamp_png(notebook_run_id: str, stamp_path: Path) -> None:
    fig = plt.figure(figsize=(2.8, 0.28), dpi=150)
    fig.patch.set_alpha(0.0)
    fig.text(0.97, 0.5, notebook_run_id, ha="right", va="center",
             fontsize=10, color="white", family="monospace",
             bbox=dict(facecolor="black", alpha=0.55, pad=3, edgecolor="none"))
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stamp_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _overlay_stamp_image(src: Path, dst: Path, stamp: Path) -> None:
    """Copy src→dst, overlaying the notebook_run_id PNG in the bottom-right."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    sh.ffmpeg(
        "-y", "-i", str(src), "-i", str(stamp),
        "-filter_complex", "[0:v][1:v]overlay=W-w-10:H-h-10",
        "-frames:v", "1",
        str(dst),
        _out=sys.stdout, _err=sys.stderr,
    )
    print(f"wrote {dst.relative_to(REPO)}")


def render_frame(out_dir: Path) -> Path:
    """One forward pass, one SCOPE_FRAME rendered to snapshot.png."""
    print(f"[frame] → {out_dir.relative_to(REPO)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sh.uv(
        "run", "python", str(OSCILLOSCOPE), "image",
        "--model", "ping",
        "--n-hidden", str(N_HIDDEN),
        "--input", "dataset",
        "--dataset", DATASET,
        "--digit", str(DIGIT_CLASS),
        "--sample", str(SAMPLE_IDX),
        "--input-rate", str(INPUT_RATE_HZ),
        "--w-in-overdrive", str(W_IN_OVERDRIVE),
        "--stim-overdrive", str(STIM_OVERDRIVE),
        "--ei-strength", str(EI_STRENGTH),
        "--dt", str(DT_MS),
        "--out-dir", str(out_dir),
        "--wipe-dir",
        _cwd=str(REPO),
        _out=sys.stdout, _err=sys.stderr,
    )
    png = out_dir / "snapshot.png"
    if not png.exists():
        raise SystemExit(f"image run did not produce {png}")
    return png


def _format_run_datetime(dt: datetime) -> str:
    day = dt.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return dt.strftime(f"%A, {day}{suffix} %B %y at %H:%M")


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def write_numbers(out_path: Path, notebook_run_id: str,
                  duration_s: float) -> dict:
    summary = {
        "notebook_run_id": notebook_run_id,
        "run_datetime": _format_run_datetime(datetime.now().astimezone()),
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "config": {
            "tier": TIER,
            "model": "ping",
            "n_e": N_HIDDEN, "n_i": N_HIDDEN // 4,
            "dt_ms": DT_MS, "sim_ms": SIM_MS,
            "step_on_ms": STEP_ON_MS, "step_off_ms": STEP_OFF_MS,
            "stim_overdrive": STIM_OVERDRIVE,
            "input_rate_hz": INPUT_RATE_HZ,
            "w_in_overdrive": W_IN_OVERDRIVE,
            "ei_strength": EI_STRENGTH,
            "input": {"dataset": DATASET,
                      "digit": DIGIT_CLASS, "sample": SAMPLE_IDX},
            "seed": SEED,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out_path.relative_to(REPO)}")
    return summary


def main() -> None:
    wipe_dir = "--no-wipe-dir" not in sys.argv

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(f"notebook_run_id = {notebook_run_id}")

    if wipe_dir:
        for d in (ARTIFACTS, FIGURES):
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)
    stamp = FIGURES / "_stamp.png"
    _render_stamp_png(notebook_run_id, stamp)

    frame_dir = ARTIFACTS / "frame"
    frame_src = render_frame(frame_dir)
    _overlay_stamp_image(frame_src, FIGURES / "scope_frame.png", stamp)

    stamp.unlink(missing_ok=True)

    duration_s = time.monotonic() - t_start
    summary = write_numbers(FIGURES / "numbers.json", notebook_run_id, duration_s)
    print(f"  duration: {summary['duration']}")


if __name__ == "__main__":
    main()
    sys.exit(0)
