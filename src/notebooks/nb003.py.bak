"""Notebook runner for entry 002 — PING stim-overdrive video.

Sweeps the Poisson-input overdrive factor (in-window rate multiplier)
from 1× → 10× while everything else is held fixed, and produces the
canonical *scan_overdrive.mp4* frame-by-frame video. Input is MNIST
digit 0 sample 0. Companion dt and ei-strength scans live in nb005
and nb006 respectively.

Also writes numbers.json with pre/stim/post E and I population rates
from an in-Python replay of the canonical overdrive=5× run so the MDX
can interpolate exact values.

Notebook entry: src/docs/src/pages/notebooks/nb003.mdx
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import sh  # noqa: E402

from helpers.fmt import format_duration, format_run_datetime  # noqa: E402
from helpers.modal import append_modal_args, parse_modal_gpu  # noqa: E402
from helpers.paths import artifacts_and_figures  # noqa: E402
from helpers.run_dirs import prepare as prepare_run_dirs  # noqa: E402
from helpers.run_id import next_run_id  # noqa: E402
from helpers.stamp import overlay_stamp_video, render_stamp_png  # noqa: E402
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
SLUG = "nb003"
SCAN_VAR = "stim-overdrive"
SCAN_MIN = 1.0  # overdrive=1 → no elevation (control)
SCAN_MAX = 10.0  # overdrive=10 → strong elevation
VIDEO_NAME = "scan_overdrive.mp4"
CANON_OVERDRIVE = 5.0  # canonical in-window rate multiplier for the replay


def compute_summary_rates() -> dict:
    """Replay one canonical-overdrive forward pass in-process with MNIST
    d0s0 spike input, then extract pre/stim/post E/I population rates."""
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
    M.max_rate_hz = INPUT_RATE_HZ
    patch_dt(DT_MS)

    base_rate = M.max_rate_hz
    stim_rate = base_rate * CANON_OVERDRIVE
    input_spikes = encode_image_spikes(
        pixel_vec,
        M.T_steps,
        DT_MS,
        base_rate,
        stim_rate,
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
    T_steps = spk_e.shape[0]
    t_ms = np.arange(T_steps) * DT_MS

    pre = (t_ms >= 0) & (t_ms < STEP_ON_MS)
    stim = (t_ms >= STEP_ON_MS) & (t_ms < STEP_OFF_MS)
    post = (t_ms >= STEP_OFF_MS) & (t_ms <= SIM_MS)

    def mean_rate(spk, mask):
        return float(spk[mask].mean()) * 1000.0 / DT_MS

    return {
        "pre": {"e": mean_rate(spk_e, pre), "i": mean_rate(spk_i, pre)},
        "stim": {"e": mean_rate(spk_e, stim), "i": mean_rate(spk_i, stim)},
        "post": {"e": mean_rate(spk_e, post), "i": mean_rate(spk_i, post)},
        "base_rate_hz": float(base_rate),
        "stim_rate_hz": float(stim_rate),
    }


def extras(tier: str, notebook_run_id: str) -> dict:
    rates = compute_summary_rates()
    r = rates
    print(
        f"  E rate  pre={r['pre']['e']:.1f} Hz  "
        f"stim={r['stim']['e']:.1f} Hz  post={r['post']['e']:.1f} Hz"
    )
    print(
        f"  I rate  pre={r['pre']['i']:.1f} Hz  "
        f"stim={r['stim']['i']:.1f} Hz  post={r['post']['i']:.1f} Hz"
    )
    return {"rates_hz": rates, "canonical_overdrive": CANON_OVERDRIVE}


if __name__ == "__main__":
    artifacts, figures = artifacts_and_figures(SLUG)

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

    prepare_run_dirs(
        SLUG, notebook_run_id, wipe=wipe_dir, skip_training=skip_training, make_artifacts=True
    )
    stamp = figures / "_stamp.png"
    render_stamp_png(notebook_run_id, stamp)

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
            "--input-rate",
            str(INPUT_RATE_HZ),
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

    overlay_stamp_video(mp4_src, figures / VIDEO_NAME, stamp)
    stamp.unlink(missing_ok=True)

    duration_s = time.monotonic() - t_start
    summary: dict = {
        "notebook_run_id": notebook_run_id,
        "run_datetime": format_run_datetime(datetime.now().astimezone()),
        "duration_s": round(duration_s, 1),
        "duration": format_duration(duration_s),
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
