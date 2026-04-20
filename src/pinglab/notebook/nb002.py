"""Notebook runner for entry 002 — basic PING videos.

Invokes `oscilloscope.py video` three times to produce:
  * scan_overdrive.mp4 — stim-overdrive sweep 1×–10× (fixed dt, fixed ei_strength)
  * scan_dt.mp4 — dt sweep 0.05–2 ms (fixed overdrive=5×, fixed ei_strength)
  * scan_ei.mp4 — ei_strength sweep 0→1.5 (fixed dt, fixed overdrive=5×) —
                  walks from async baseline (no E→I coupling) to PING

Input is MNIST digit 0 sample 0, Poisson-encoded into the input layer
with per-pixel rate ∝ pixel intensity. During the stim window the rate
is multiplied by the overdrive factor; outside the window it's the
baseline.

Also writes numbers.json with the notebook_run_id, config snapshot, and
pre/stim/post population rates from an in-Python replay of the canonical
overdrive=5 run (so the MDX can interpolate exact values).

Notebook entry: src/docs/src/pages/notebook/nb002.mdx
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

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from _run_id import next_run_id, persist as persist_run_id  # noqa: E402

SLUG = "nb002"
ARTIFACTS = REPO / "src" / "artifacts" / "notebook" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebook" / SLUG
OSCILLOSCOPE = PINGLAB / "oscilloscope.py"

# ── Simulation config ─────────────────────────────────────────────────────
# Defaults in config.py: sim_ms=600, step_on_ms=200, step_off_ms=300.
# Input is MNIST digit 0 sample 0, Poisson-encoded. base_rate comes from
# M.max_rate_hz; during the stim window rate = base_rate * overdrive.
SEED           = 42
TIER           = "medium"  # see src/docs/src/pages/styleguide.md § 8
# Per-scan frame count by tier. nb002 runs 3 scans, so wall-clock ≈ 3 × frames × ~1.3 s.
TIER_FRAMES    = {"extra small": 5, "small": 15, "medium": 100, "large": 300, "extra large": 600}
N_HIDDEN       = 512     # → N_E=512, N_I=128 (n_i = n_e//4)
DT_MS          = 0.1
SIM_MS         = 600.0
STEP_ON_MS     = 200.0
STEP_OFF_MS    = 300.0
CANON_OVERDRIVE = 5.0    # canonical in-window rate multiplier
INPUT_RATE_HZ  = 50.0    # max per-pixel Poisson rate (fully-on pixel, baseline)
W_IN_OVERDRIVE = 1.8     # multiplier on W_in weights — pushes net closer to PING threshold
DATASET        = "mnist"
DIGIT_CLASS    = 0
SAMPLE_IDX     = 0

# ── Scan config ───────────────────────────────────────────────────────────
SCAN_MIN       = 1.0     # overdrive=1 → no elevation (control)
SCAN_MAX       = 10.0    # overdrive=10 → strong elevation
SCAN_FRAMES    = TIER_FRAMES[TIER]
SCAN_FPS       = 30

# dt-scan: fix overdrive high so PING is robustly on, sweep integration
# time step. We want any rhythm distortion to come from dt alone, not
# from being near the drive threshold — so drive higher than the canonical 5×.
DT_SCAN_OVERDRIVE = 10.0
DT_SCAN_MIN    = 0.05    # ms — fine temporal resolution
DT_SCAN_MAX    = 2.0     # ms — coarse (likely unstable)
DT_SCAN_FRAMES = TIER_FRAMES[TIER]
DT_SCAN_FPS    = 30

# ei-scan: overdrive pinned at 1× (no stim-window boost), sweep E→I coupling
# strength. At 0, I-cells don't hear E-cells → no feedback → async baseline.
# As strength grows, the E→I→E feedback loop closes and gamma emerges.
# Because there's no overdrive to push E firing, input rate and W_in are
# both boosted relative to the other scans so E has enough baseline drive
# to recruit I at all.
EI_SCAN_INPUT_RATE_HZ = 2 * INPUT_RATE_HZ
EI_SCAN_W_IN_OVERDRIVE = 3.0
EI_SCAN_MIN    = 0.0
EI_SCAN_MAX    = 1.0     # sweep up to the default ei_strength — past 1 the E rate is already saturated-low
EI_SCAN_FRAMES = TIER_FRAMES[TIER]
EI_SCAN_FPS    = 30


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
    """Copy src→dst, overlaying the notebook_run_id PNG in the bottom-right."""
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


_DATASET_ARGS = (
    "--input", "dataset",
    "--dataset", DATASET,
    "--digit", str(DIGIT_CLASS),
    "--sample", str(SAMPLE_IDX),
)


def render_scan(out_dir: Path) -> Path:
    """Sweep stim-overdrive 1×→10× with MNIST d0s0 input. Each frame
    re-runs the sim with a different in-window rate multiplier."""
    print(f"[scan] → {out_dir.relative_to(REPO)}")
    sh.uv(
        "run", "python", str(OSCILLOSCOPE), "video",
        "--model", "ping",
        "--n-hidden", str(N_HIDDEN),
        *_DATASET_ARGS,
        "--input-rate", str(INPUT_RATE_HZ),
        "--w-in-overdrive", str(W_IN_OVERDRIVE),
        "--scan-var", "stim-overdrive",
        "--scan-min", str(SCAN_MIN),
        "--scan-max", str(SCAN_MAX),
        "--frames", str(SCAN_FRAMES),
        "--frame-rate", str(SCAN_FPS),
        "--dt", str(DT_MS),
        "--out-dir", str(out_dir),
        "--wipe-dir",
        _cwd=str(REPO),
        _out=sys.stdout, _err=sys.stderr,
    )
    mp4 = out_dir / "scan.mp4"
    if not mp4.exists():
        raise SystemExit(f"video run did not produce {mp4}")
    return mp4


def render_dt_scan(out_dir: Path) -> Path:
    """Sweep integration dt at high fixed overdrive with MNIST d0s0 input.
    Fine dt is stable, coarse dt distorts or diverges. Overdrive is held
    well past the PING threshold so dt is the only variable breaking
    the rhythm."""
    print(f"[dt-scan] → {out_dir.relative_to(REPO)}")
    sh.uv(
        "run", "python", str(OSCILLOSCOPE), "video",
        "--model", "ping",
        "--n-hidden", str(N_HIDDEN),
        *_DATASET_ARGS,
        "--input-rate", str(INPUT_RATE_HZ),
        "--w-in-overdrive", str(W_IN_OVERDRIVE),
        "--stim-overdrive", str(DT_SCAN_OVERDRIVE),
        "--scan-var", "dt",
        "--scan-min", str(DT_SCAN_MIN),
        "--scan-max", str(DT_SCAN_MAX),
        "--frames", str(DT_SCAN_FRAMES),
        "--frame-rate", str(DT_SCAN_FPS),
        "--out-dir", str(out_dir),
        "--wipe-dir",
        _cwd=str(REPO),
        _out=sys.stdout, _err=sys.stderr,
    )
    mp4 = out_dir / "scan.mp4"
    if not mp4.exists():
        raise SystemExit(f"dt-scan video run did not produce {mp4}")
    return mp4


def render_ei_scan(out_dir: Path) -> Path:
    """Sweep E→I coupling strength with *no* stim-window overdrive (input
    rate flat through the trial). At strength=0 the E/I populations
    decouple → async; raising the strength closes the E→I→E feedback loop
    → gamma — isolating coupling as the knob that lights up PING."""
    print(f"[ei-scan] → {out_dir.relative_to(REPO)}")
    sh.uv(
        "run", "python", str(OSCILLOSCOPE), "video",
        "--model", "ping",
        "--n-hidden", str(N_HIDDEN),
        *_DATASET_ARGS,
        "--input-rate", str(EI_SCAN_INPUT_RATE_HZ),
        "--w-in-overdrive", str(EI_SCAN_W_IN_OVERDRIVE),
        "--stim-overdrive", "1.0",
        "--scan-var", "ei_strength",
        "--scan-min", str(EI_SCAN_MIN),
        "--scan-max", str(EI_SCAN_MAX),
        "--frames", str(EI_SCAN_FRAMES),
        "--frame-rate", str(EI_SCAN_FPS),
        "--dt", str(DT_MS),
        "--out-dir", str(out_dir),
        "--wipe-dir",
        _cwd=str(REPO),
        _out=sys.stdout, _err=sys.stderr,
    )
    mp4 = out_dir / "scan.mp4"
    if not mp4.exists():
        raise SystemExit(f"ei-scan video run did not produce {mp4}")
    return mp4


def compute_summary_rates() -> dict:
    """Replay one canonical-overdrive forward pass in-process with MNIST
    d0s0 spike input, then extract pre/stim/post E/I population rates."""
    # Lazy imports so the shell-out steps above don't get polluted by the
    # module-level import of config.py (which seeds globals on import).
    import torch  # noqa: E402
    import config as C  # noqa: E402
    import models as M  # noqa: E402
    from config import make_net, patch_dt  # noqa: E402
    from oscilloscope import (
        _extract_records, _load_dataset_image, encode_image_spikes, primary_hid_key,
    )  # noqa: E402

    # Align in-process config with CLI invocation.
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
        pixel_vec, M.T_steps, DT_MS, base_rate, stim_rate,
        C.STEP_ON_MS, C.STEP_OFF_MS, C.SEED,
    ).to(C.DEVICE)

    net = make_net(C.cfg,
                   w_in=(*C.W_IN_SPIKES, "normal", C.W_IN_SPARSITY),
                   model_name="ping")
    if W_IN_OVERDRIVE != 1.0:
        with torch.no_grad():
            net.W_ff[0].mul_(W_IN_OVERDRIVE)
    net.recording = True
    with torch.no_grad():
        net.forward(input_spikes=input_spikes)
    rec = _extract_records(net)

    spk_e = rec[primary_hid_key(rec)]
    spk_i = rec["inh"]
    T_steps = spk_e.shape[0]
    t_ms = np.arange(T_steps) * DT_MS

    pre  = (t_ms >= 0)           & (t_ms < STEP_ON_MS)
    stim = (t_ms >= STEP_ON_MS)  & (t_ms < STEP_OFF_MS)
    post = (t_ms >= STEP_OFF_MS) & (t_ms <= SIM_MS)

    def mean_rate(spk, mask):
        return float(spk[mask].mean()) * 1000.0 / DT_MS

    return {
        "pre":  {"e": mean_rate(spk_e, pre),  "i": mean_rate(spk_i, pre)},
        "stim": {"e": mean_rate(spk_e, stim), "i": mean_rate(spk_i, stim)},
        "post": {"e": mean_rate(spk_e, post), "i": mean_rate(spk_i, post)},
        "base_rate_hz": float(base_rate),
        "stim_rate_hz": float(stim_rate),
    }


def _format_run_datetime(dt: datetime) -> str:
    """Long-form British-style datetime: 'Saturday, 18th April 26 at 14:30'."""
    day = dt.day
    suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return dt.strftime(f"%A, {day}{suffix} %B %y at %H:%M")


def _format_duration(seconds: float) -> str:
    """Compact duration: '42s', '12m 08s', '1h 23m'."""
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def write_numbers(rates: dict, out_path: Path, notebook_run_id: str,
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
            "input": {"mode": "dataset", "dataset": DATASET,
                      "digit": DIGIT_CLASS, "sample": SAMPLE_IDX,
                      "base_rate_hz": rates.get("base_rate_hz"),
                      "stim_rate_hz": rates.get("stim_rate_hz"),
                      "w_in_overdrive": W_IN_OVERDRIVE},
            "canonical_overdrive": CANON_OVERDRIVE,
            "scan": {"var": "stim-overdrive",
                     "min": SCAN_MIN, "max": SCAN_MAX,
                     "frames": SCAN_FRAMES, "fps": SCAN_FPS},
            "dt_scan": {"var": "dt",
                        "min": DT_SCAN_MIN, "max": DT_SCAN_MAX,
                        "frames": DT_SCAN_FRAMES, "fps": DT_SCAN_FPS,
                        "fixed_overdrive": DT_SCAN_OVERDRIVE},
            "ei_scan": {"var": "ei_strength",
                        "min": EI_SCAN_MIN, "max": EI_SCAN_MAX,
                        "frames": EI_SCAN_FRAMES, "fps": EI_SCAN_FPS,
                        "fixed_overdrive": 1.0},
            "seed": SEED,
        },
        "rates_hz": rates,
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

    # 1. Oscilloscope-rendered scan video (stim-overdrive sweep).
    scan_dir = ARTIFACTS / "scan"
    scan_src = render_scan(scan_dir)
    _overlay_stamp_video(scan_src, FIGURES / "scan_overdrive.mp4", stamp)

    # 2. Oscilloscope-rendered dt sweep at fixed overdrive=5×.
    dt_scan_dir = ARTIFACTS / "dt_scan"
    dt_scan_src = render_dt_scan(dt_scan_dir)
    _overlay_stamp_video(dt_scan_src, FIGURES / "scan_dt.mp4", stamp)

    # 3. Oscilloscope-rendered ei_strength sweep — async → PING.
    ei_scan_dir = ARTIFACTS / "ei_scan"
    ei_scan_src = render_ei_scan(ei_scan_dir)
    _overlay_stamp_video(ei_scan_src, FIGURES / "scan_ei.mp4", stamp)

    stamp.unlink(missing_ok=True)

    # 3. Numbers for the MDX.
    rates = compute_summary_rates()
    duration_s = time.monotonic() - t_start
    summary = write_numbers(rates, FIGURES / "numbers.json",
                            notebook_run_id, duration_s)
    print(f"  total duration: {summary['duration']}")

    r = summary["rates_hz"]
    print(f"  E rate  pre={r['pre']['e']:.1f} Hz  "
          f"stim={r['stim']['e']:.1f} Hz  post={r['post']['e']:.1f} Hz")
    print(f"  I rate  pre={r['pre']['i']:.1f} Hz  "
          f"stim={r['stim']['i']:.1f} Hz  post={r['post']['i']:.1f} Hz")


if __name__ == "__main__":
    main()
    sys.exit(0)
