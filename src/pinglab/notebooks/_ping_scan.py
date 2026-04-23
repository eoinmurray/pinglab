"""Shared orchestration for single-scan PING-video notebooks.

Each scan runner (nb002 stim-overdrive, nb010 dt, nb011 ei-strength)
hands `run_scan` a ScanSpec and gets back the full notebook pipeline:
per-tier frame counts, oscilloscope-video dispatch, run-id stamping,
numbers.json, duration bookkeeping. Canonical-overdrive population-rate
replay stays in nb002 (only relevant to the overdrive scan).
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import sh

from _modal import append_modal_args, parse_modal_gpu
from _run_id import next_run_id, persist as persist_run_id
from _tier import parse_tier

REPO = Path(__file__).resolve().parents[3]
PINGLAB = REPO / "src" / "pinglab"
OSCILLOSCOPE = PINGLAB / "oscilloscope.py"

# Simulation config shared by every PING-video scan. Defaults in config.py:
# sim_ms=600, step_on_ms=200, step_off_ms=300. Input is MNIST d0s0,
# Poisson-encoded with per-pixel rate ∝ pixel intensity.
SEED           = 42
DEFAULT_TIER   = "small"  # see src/docs/src/pages/styleguide.md § 10
TIER_FRAMES    = {"extra small": 5, "small": 15, "medium": 100, "large": 300, "extra large": 600}
N_HIDDEN       = 512     # → N_E=512, N_I=128 (n_i = n_e//4)
DT_MS          = 0.1
SIM_MS         = 600.0
STEP_ON_MS     = 200.0
STEP_OFF_MS    = 300.0
INPUT_RATE_HZ  = 50.0    # max per-pixel Poisson rate (fully-on pixel)
W_IN_OVERDRIVE = 1.8     # W_in multiplier pushing net toward PING threshold
DATASET        = "mnist"
DIGIT_CLASS    = 0
SAMPLE_IDX     = 0
SCAN_FPS       = 30


def dataset_args() -> tuple[str, ...]:
    return (
        "--input", "dataset",
        "--dataset", DATASET,
        "--digit", str(DIGIT_CLASS),
        "--sample", str(SAMPLE_IDX),
    )


@dataclass
class ScanSpec:
    slug: str            # e.g. "nb002"
    scan_var: str        # oscilloscope --scan-var value
    scan_min: float
    scan_max: float
    video_name: str      # e.g. "scan_overdrive.mp4"
    # Extra flags appended to the oscilloscope video invocation (e.g.
    # --stim-overdrive, overridden --input-rate, --w-in-overdrive).
    extra_osc_args: list[str] = field(default_factory=list)
    # Optional scan-level config payload copied into numbers.json['config'].
    config_payload: dict = field(default_factory=dict)
    # Optional hook: (tier, notebook_run_id) -> dict merged into numbers.json.
    extras_fn: Callable[[str, str], dict] | None = None


def _render_stamp_png(notebook_run_id: str, stamp_path: Path) -> None:
    fig = plt.figure(figsize=(2.8, 0.28), dpi=150)
    fig.patch.set_alpha(0.0)
    fig.text(0.97, 0.5, notebook_run_id, ha="right", va="center",
             fontsize=10, color="white", family="monospace",
             bbox=dict(facecolor="black", alpha=0.55, pad=3, edgecolor="none"))
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stamp_path, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _overlay_stamp_video(src: Path, dst: Path, stamp: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    sh.ffmpeg(
        "-y", "-i", str(src), "-i", str(stamp),
        "-filter_complex", "[0:v][1:v]overlay=W-w-10:H-h-10",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "veryfast", "-crf", "20",
        "-movflags", "+faststart",
        str(dst),
        _out=sys.stdout, _err=sys.stderr,
    )
    print(f"wrote {dst.relative_to(REPO)}")


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


def _render_video(spec: ScanSpec, out_dir: Path, frames: int,
                  modal_gpu: str | None) -> Path:
    print(f"[{spec.scan_var}] → {out_dir.relative_to(REPO)}"
          + (f"  [modal:{modal_gpu}]" if modal_gpu else ""))
    args = [
        "run", "python", str(OSCILLOSCOPE), "video",
        "--model", "ping",
        "--n-hidden", str(N_HIDDEN),
        *dataset_args(),
        "--scan-var", spec.scan_var,
        "--scan-min", str(spec.scan_min),
        "--scan-max", str(spec.scan_max),
        "--frames", str(frames),
        "--frame-rate", str(SCAN_FPS),
        "--out-dir", str(out_dir),
        "--wipe-dir",
        *spec.extra_osc_args,
    ]
    args = append_modal_args(args, modal_gpu)
    sh.uv(*args, _cwd=str(REPO), _out=sys.stdout, _err=sys.stderr)
    mp4 = out_dir / "scan.mp4"
    if not mp4.exists():
        raise SystemExit(f"video run did not produce {mp4}")
    return mp4


def run_scan(spec: ScanSpec) -> dict:
    """Run the scan end-to-end: parse CLI, dispatch oscilloscope video,
    overlay stamp, write numbers.json. Returns the summary dict."""
    artifacts = REPO / "src" / "artifacts" / "notebooks" / spec.slug
    figures = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / spec.slug

    wipe_dir = "--no-wipe-dir" not in sys.argv
    skip_training = "--skip-training" in sys.argv
    modal_gpu = parse_modal_gpu(sys.argv)
    tier = parse_tier(sys.argv, choices=TIER_FRAMES.keys(), default=DEFAULT_TIER)
    frames = TIER_FRAMES[tier]

    t_start = time.monotonic()
    notebook_run_id = next_run_id(spec.slug)
    print(f"notebook_run_id = {notebook_run_id} tier={tier}"
          + ("  [skip-training]" if skip_training else "")
          + (f"  [modal:{modal_gpu}]" if modal_gpu else ""))

    if wipe_dir:
        wipe_targets = (figures,) if skip_training else (artifacts, figures)
        for d in wipe_targets:
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    artifacts.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    persist_run_id(spec.slug, notebook_run_id)
    stamp = figures / "_stamp.png"
    _render_stamp_png(notebook_run_id, stamp)

    mp4_src = artifacts / "scan" / "scan.mp4"
    if skip_training:
        if not mp4_src.exists():
            raise SystemExit(f"--skip-training requires existing {mp4_src}")
    else:
        _render_video(spec, artifacts / "scan", frames, modal_gpu)
    _overlay_stamp_video(mp4_src, figures / spec.video_name, stamp)
    stamp.unlink(missing_ok=True)

    duration_s = time.monotonic() - t_start
    summary = {
        "notebook_run_id": notebook_run_id,
        "run_datetime": _format_run_datetime(datetime.now().astimezone()),
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "tier": tier,
        "config": {
            "tier": tier,
            "model": "ping",
            "n_e": N_HIDDEN, "n_i": N_HIDDEN // 4,
            "dt_ms": DT_MS, "sim_ms": SIM_MS,
            "step_on_ms": STEP_ON_MS, "step_off_ms": STEP_OFF_MS,
            "input": {"mode": "dataset", "dataset": DATASET,
                      "digit": DIGIT_CLASS, "sample": SAMPLE_IDX,
                      "base_rate_hz": INPUT_RATE_HZ,
                      "w_in_overdrive": W_IN_OVERDRIVE},
            "seed": SEED,
            "scan": {"var": spec.scan_var, "min": spec.scan_min,
                     "max": spec.scan_max, "frames": frames, "fps": SCAN_FPS,
                     **spec.config_payload},
        },
    }
    if spec.extras_fn is not None:
        summary.update(spec.extras_fn(tier, notebook_run_id))

    out_path = figures / "numbers.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out_path.relative_to(REPO)}")
    print(f"  total duration: {summary['duration']}")
    return summary
