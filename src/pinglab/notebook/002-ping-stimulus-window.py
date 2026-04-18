"""Repro for notebook entry 002 — PING stimulus window.

Invokes `oscilloscope.py video --scan-var stim-overdrive` to produce
scan.mp4, where each frame is a fresh 600 ms PING simulation at a different
stim-overdrive value. The sweep animates PING intensifying as the stim
drive grows.

Also writes numbers.json with the notebook_run_id, config snapshot, and
pre/stim/post population rates from an in-Python replay of the canonical
overdrive=5 sim (so the MDX can interpolate exact values).

Notebook entry: src/docs/src/pages/notebook/002-ping-stimulus-window.mdx
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import sh

REPO = Path(__file__).resolve().parents[3]
PINGLAB = REPO / "src" / "pinglab"
sys.path.insert(0, str(PINGLAB))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

SLUG = "002-ping-stimulus-window"
ARTIFACTS = REPO / "src" / "artifacts" / "notebook" / SLUG
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebook" / SLUG
OSCILLOSCOPE = PINGLAB / "oscilloscope.py"

# ── Simulation config ─────────────────────────────────────────────────────
# Defaults in config.py: sim_ms=600, step_on_ms=200, step_off_ms=300,
# t_e_async=0.0006. We override n_e/n_i via --n-hidden and keep everything
# else at repo defaults so the scan video matches the oscilloscope archive
# style (same renderer, same stim window).
SEED           = 42
TIER           = "tiny"  # see src/docs/src/pages/llm-context.md § 8
N_HIDDEN       = 512     # → N_E=512, N_I=128 (n_i = n_e//4)
DT_MS          = 0.1
SIM_MS         = 600.0
STEP_ON_MS     = 200.0
STEP_OFF_MS    = 300.0
T_E_ASYNC      = 0.0006  # μS — baseline (sub-threshold) drive
CANON_OVERDRIVE = 5.0    # t_e_ping = t_e_async * 5 = 0.003 μS (strong PING)

# ── Scan config ───────────────────────────────────────────────────────────
SCAN_MIN       = 1.0     # overdrive=1 → no elevation (control)
SCAN_MAX       = 10.0    # overdrive=10 → strong elevation
SCAN_FRAMES    = 180
SCAN_FPS       = 30

# dt-scan: fix overdrive at canonical 5×, sweep integration time step
DT_SCAN_MIN    = 0.05    # ms — fine temporal resolution
DT_SCAN_MAX    = 2.0     # ms — coarse (likely unstable)
DT_SCAN_FRAMES = 180
DT_SCAN_FPS    = 30


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


def render_scan(out_dir: Path) -> Path:
    """Invoke `oscilloscope.py video` sweeping stim-overdrive from 1×→10×.
    Each frame is a separate simulation; PING grows in with the sweep."""
    print(f"[scan] → {out_dir.relative_to(REPO)}")
    sh.uv(
        "run", "python", str(OSCILLOSCOPE), "video",
        "--model", "ping",
        "--n-hidden", str(N_HIDDEN),
        "--input", "synthetic-conductance",
        "--drive", str(T_E_ASYNC),
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
    """Sweep integration dt at fixed overdrive=5×. Each frame re-runs the sim
    with a different dt; fine dt is stable, coarse dt blows up."""
    print(f"[dt-scan] → {out_dir.relative_to(REPO)}")
    sh.uv(
        "run", "python", str(OSCILLOSCOPE), "video",
        "--model", "ping",
        "--n-hidden", str(N_HIDDEN),
        "--input", "synthetic-conductance",
        "--drive", str(T_E_ASYNC),
        "--stim-overdrive", str(CANON_OVERDRIVE),
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


def compute_summary_rates() -> dict:
    """Run the canonical-overdrive single sim in-process to extract
    pre/stim/post E and I population rates for numbers.json."""
    # Lazy imports so the shell-out steps above don't get polluted by the
    # module-level import of config.py (which seeds globals on import).
    import config as C  # noqa: E402
    from config import run_sim  # noqa: E402

    # Ensure the in-process sim matches the CLI invocation.
    C.cfg.n_e = N_HIDDEN
    C.cfg.n_i = N_HIDDEN // 4
    C.cfg.sim_ms = SIM_MS
    C.cfg.step_on_ms = STEP_ON_MS
    C.cfg.step_off_ms = STEP_OFF_MS
    C.cfg.t_e_async = T_E_ASYNC
    C.cfg.seed = SEED
    C._sync_globals_from_cfg(C.cfg)

    t_e_ping = T_E_ASYNC * CANON_OVERDRIVE
    rec, _, _ = run_sim(DT_MS, t_e_ping, model_name="ping")
    spk_e = rec["hid"]
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
    }


def write_numbers(rates: dict, out_path: Path, notebook_run_id: str) -> dict:
    summary = {
        "notebook_run_id": notebook_run_id,
        "config": {
            "tier": TIER,
            "model": "ping",
            "n_e": N_HIDDEN, "n_i": N_HIDDEN // 4,
            "dt_ms": DT_MS, "sim_ms": SIM_MS,
            "step_on_ms": STEP_ON_MS, "step_off_ms": STEP_OFF_MS,
            "t_e_async": T_E_ASYNC,
            "t_e_ping": T_E_ASYNC * CANON_OVERDRIVE,
            "canonical_overdrive": CANON_OVERDRIVE,
            "scan": {"var": "stim-overdrive",
                     "min": SCAN_MIN, "max": SCAN_MAX,
                     "frames": SCAN_FRAMES, "fps": SCAN_FPS},
            "dt_scan": {"var": "dt",
                        "min": DT_SCAN_MIN, "max": DT_SCAN_MAX,
                        "frames": DT_SCAN_FRAMES, "fps": DT_SCAN_FPS,
                        "fixed_overdrive": CANON_OVERDRIVE},
            "seed": SEED,
        },
        "rates_hz": rates,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out_path.relative_to(REPO)}")
    return summary


def main() -> None:
    notebook_run_id = (
        f"nb{SLUG.split('-')[0]}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )
    print(f"notebook_run_id = {notebook_run_id}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    stamp = FIGURES / "_stamp.png"
    _render_stamp_png(notebook_run_id, stamp)

    # 1. Oscilloscope-rendered scan video (stim-overdrive sweep).
    scan_dir = ARTIFACTS / "scan"
    scan_src = render_scan(scan_dir)
    _overlay_stamp_video(scan_src, FIGURES / "scan.mp4", stamp)

    # 2. Oscilloscope-rendered dt sweep at fixed overdrive=5×.
    dt_scan_dir = ARTIFACTS / "dt_scan"
    dt_scan_src = render_dt_scan(dt_scan_dir)
    _overlay_stamp_video(dt_scan_src, FIGURES / "scan_dt.mp4", stamp)

    stamp.unlink(missing_ok=True)

    # 3. Numbers for the MDX.
    rates = compute_summary_rates()
    summary = write_numbers(rates, FIGURES / "numbers.json", notebook_run_id)

    r = summary["rates_hz"]
    print(f"  E rate  pre={r['pre']['e']:.1f} Hz  "
          f"stim={r['stim']['e']:.1f} Hz  post={r['post']['e']:.1f} Hz")
    print(f"  I rate  pre={r['pre']['i']:.1f} Hz  "
          f"stim={r['stim']['i']:.1f} Hz  post={r['post']['i']:.1f} Hz")


if __name__ == "__main__":
    main()
    sys.exit(0)
