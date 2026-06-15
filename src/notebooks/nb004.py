"""Notebook runner for entry 003 — PING input-rate video.

Sweeps the Poisson input rate (max per-pixel rate, applied uniformly
over the whole sim — no stim window, no overdrive) from 25 Hz → 1000 Hz
with everything else held fixed, and produces the *scan_input_rate.mp4*
frame-by-frame video. Companion to nb003 which sweeps the in-window
overdrive factor at fixed base rate; this entry walks the absolute drive
itself with a flat envelope. Input is MNIST digit 0 sample 0.

Also writes numbers.json with mean E and I population rates from an
in-Python replay of the canonical input-rate run so the MDX can
interpolate exact values.

Notebook entry: src/docs/src/pages/notebooks/nb004.mdx
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import sh  # noqa: E402

from helpers.modal import append_modal_args, parse_modal_gpu  # noqa: E402
from helpers.run_id import next_run_id, persist as persist_run_id  # noqa: E402
from helpers.tier import parse_tier  # noqa: E402

OSCILLOSCOPE = REPO / "src" / "cli" / "cli.py"

# --- PING-video scan recipe (inlined; shared defaults across nb003–nb006) ---
# config.py defaults: sim_ms=600, step_on_ms=200, step_off_ms=300. Input is
# MNIST d0s0, Poisson-encoded with per-pixel rate ∝ pixel intensity.
SEED = 42
DEFAULT_TIER = "small"
TIER_FRAMES = {
    "extra small": 5,
    "small": 15,
    "medium": 100,
    "large": 300,
    "extra large": 600,
}
N_HIDDEN = 512  # → N_E=512, N_I=128 (n_i = n_e//4)
DT_MS = 0.1
SIM_MS = 600.0
STEP_ON_MS = 200.0
STEP_OFF_MS = 300.0
INPUT_RATE_HZ = 50.0  # max per-pixel Poisson rate (fully-on pixel)
W_IN_MEAN = 0.54  # W_in init mean — pushes net toward PING threshold
W_IN_STD = 0.108  # W_in init std (proportional to mean)
DATASET = "mnist"
DIGIT_CLASS = 0
SAMPLE_IDX = 0
SCAN_FPS = 30

# --- This entry's scan ------------------------------------------------------
SLUG = "nb004"
SCAN_VAR = "spike_rate"
SCAN_MIN = 25.0  # 25 Hz → barely-driven control
SCAN_MAX = 1000.0  # 1 kHz → strong drive
VIDEO_NAME = "scan_input_rate.mp4"
CANON_INPUT_RATE = (
    250.0  # canonical input rate for the replay (suprathreshold for flat envelope)
)


def _render_stamp_png(notebook_run_id: str, stamp_path: Path) -> None:
    fig = plt.figure(figsize=(2.8, 0.28), dpi=150)
    fig.patch.set_alpha(0.0)
    fig.text(
        0.97,
        0.5,
        notebook_run_id,
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


def _overlay_stamp_video(src: Path, dst: Path, stamp: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    sh.ffmpeg(
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


def _format_run_datetime(dt: datetime) -> str:
    day = dt.day
    suffix = (
        "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    )
    return dt.strftime(f"%A, {day}{suffix} %B %y at %H:%M")


def _format_duration(seconds: float) -> str:
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def compute_summary_rates() -> dict:
    """Replay one canonical-input-rate forward pass in-process with MNIST
    d0s0 spike input, then extract mean E/I population rates."""
    import torch  # noqa: E402
    import cli.config as C  # noqa: E402
    import models as M  # noqa: E402
    from cli.config import make_net, patch_dt  # noqa: E402
    from cli import (
        _extract_records,
        _load_dataset_image,
        encode_image_spikes,
        primary_hid_key,
    )  # noqa: E402

    C.cfg.n_e = N_HIDDEN
    C.cfg.n_i = N_HIDDEN // 4
    C.cfg.sim_ms = SIM_MS
    C.cfg.step_on_ms = STEP_ON_MS
    C.cfg.step_off_ms = STEP_OFF_MS
    C.cfg.seed = SEED
    C._sync_globals_from_cfg(C.cfg)

    pixel_vec, _ = _load_dataset_image(DATASET, DIGIT_CLASS, SAMPLE_IDX)
    M.N_IN = len(pixel_vec)
    M.N_HID = C.N_E
    M.N_INH = C.N_I
    M.max_rate_hz = CANON_INPUT_RATE
    patch_dt(DT_MS)

    rate = M.max_rate_hz
    input_spikes = encode_image_spikes(
        pixel_vec,
        M.T_steps,
        DT_MS,
        rate,
        rate,
        C.STEP_ON_MS,
        C.STEP_OFF_MS,
        C.SEED,
    ).to(C.DEVICE)

    net = make_net(
        C.cfg, w_in=(W_IN_MEAN, W_IN_STD, "normal", C.W_IN_SPARSITY), model_name="ping"
    )
    net.recording = True
    with torch.no_grad():
        net.forward(input_spikes=input_spikes)
    rec = _extract_records(net)

    spk_e = rec[primary_hid_key(rec)]
    spk_i = rec["inh"]

    def mean_rate(spk):
        return float(spk.mean()) * 1000.0 / DT_MS

    return {
        "mean": {"e": mean_rate(spk_e), "i": mean_rate(spk_i)},
        "input_rate_hz": float(rate),
    }


def extras(tier: str, notebook_run_id: str) -> dict:
    rates = compute_summary_rates()
    print(f"  mean rate  E={rates['mean']['e']:.1f} Hz  I={rates['mean']['i']:.1f} Hz")
    return {
        "rates_hz": rates,
        "canonical_input_rate_hz": CANON_INPUT_RATE,
    }


if __name__ == "__main__":
    artifacts = REPO / "src" / "artifacts" / "notebooks" / SLUG
    figures = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG

    wipe_dir = "--no-wipe-dir" not in sys.argv
    skip_training = "--skip-training" in sys.argv
    modal_gpu = parse_modal_gpu(sys.argv)
    tier = parse_tier(sys.argv, choices=TIER_FRAMES.keys(), default=DEFAULT_TIER)
    frames = TIER_FRAMES[tier]

    t_start = time.monotonic()
    notebook_run_id = next_run_id(SLUG)
    print(
        f"notebook_run_id = {notebook_run_id} tier={tier}"
        + ("  [skip-training]" if skip_training else "")
        + (f"  [modal:{modal_gpu}]" if modal_gpu else "")
    )

    if wipe_dir:
        wipe_targets = (figures,) if skip_training else (artifacts, figures)
        for d in wipe_targets:
            if d.exists():
                print(f"[wipe] {d.relative_to(REPO)}")
                shutil.rmtree(d)
    artifacts.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    persist_run_id(SLUG, notebook_run_id)
    stamp = figures / "_stamp.png"
    _render_stamp_png(notebook_run_id, stamp)

    scan_dir = artifacts / "scan"
    mp4_src = scan_dir / "scan.mp4"
    if skip_training:
        if not mp4_src.exists():
            raise SystemExit(f"--skip-training requires existing {mp4_src}")
    else:
        print(
            f"[{SCAN_VAR}] → {scan_dir.relative_to(REPO)}"
            + (f"  [modal:{modal_gpu}]" if modal_gpu else "")
        )
        args = [
            "run",
            "python",
            str(OSCILLOSCOPE),
            "sim",
            "--video",
            "--model",
            "ping",
            "--n-hidden",
            str(N_HIDDEN),
            "--t-ms",
            str(SIM_MS),
            "--input",
            "dataset",
            "--dataset",
            DATASET,
            "--digit",
            str(DIGIT_CLASS),
            "--sample",
            str(SAMPLE_IDX),
            "--scan-var",
            SCAN_VAR,
            "--scan-min",
            str(SCAN_MIN),
            "--scan-max",
            str(SCAN_MAX),
            "--frames",
            str(frames),
            "--frame-rate",
            str(SCAN_FPS),
            "--out-dir",
            str(scan_dir),
            "--wipe-dir",
            # --- this entry's extra oscilloscope flags ---
            "--stim-overdrive",
            "1.0",
            "--w-in",
            str(W_IN_MEAN),
            str(W_IN_STD),
            "--dt",
            str(DT_MS),
        ]
        args = append_modal_args(args, modal_gpu)
        sh.uv(*args, _cwd=str(REPO), _out=sys.stdout, _err=sys.stderr)
        if not mp4_src.exists():
            raise SystemExit(f"video run did not produce {mp4_src}")

    _overlay_stamp_video(mp4_src, figures / VIDEO_NAME, stamp)
    stamp.unlink(missing_ok=True)

    duration_s = time.monotonic() - t_start
    summary: dict = {
        "notebook_run_id": notebook_run_id,
        "run_datetime": _format_run_datetime(datetime.now().astimezone()),
        "duration_s": round(duration_s, 1),
        "duration": _format_duration(duration_s),
        "tier": tier,
        "config": {
            "tier": tier,
            "model": "ping",
            "n_e": N_HIDDEN,
            "n_i": N_HIDDEN // 4,
            "dt_ms": DT_MS,
            "sim_ms": SIM_MS,
            "step_on_ms": STEP_ON_MS,
            "step_off_ms": STEP_OFF_MS,
            "input": {
                "mode": "dataset",
                "dataset": DATASET,
                "digit": DIGIT_CLASS,
                "sample": SAMPLE_IDX,
                "base_rate_hz": INPUT_RATE_HZ,
                "w_in_mean": W_IN_MEAN,
                "w_in_std": W_IN_STD,
            },
            "seed": SEED,
            "scan": {
                "var": SCAN_VAR,
                "min": SCAN_MIN,
                "max": SCAN_MAX,
                "frames": frames,
                "fps": SCAN_FPS,
            },
        },
    }
    summary.update(extras(tier, notebook_run_id))

    out_path = figures / "numbers.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out_path.relative_to(REPO)}")
    print(f"  total duration: {summary['duration']}")
    sys.exit(0)
