"""Notebook runner for entry 003 — PING input-rate video + constant-current control.

Sweeps the Poisson input rate (max per-pixel rate, applied uniformly
over the whole sim — no stim window, no overdrive) from 25 Hz → 1000 Hz
with everything else held fixed, and produces the *scan_input_rate.mp4*
frame-by-frame video. Companion to nb002 which sweeps the in-window
overdrive factor at fixed base rate; this entry walks the absolute drive
itself with a flat envelope. Input is MNIST digit 0 sample 0.

Also runs a control sweep with the *synthetic-conductance* input mode
(uniform constant ext_g amplitude across E, no Poisson, no image) to
isolate whether the band-jumping at low Poisson rates is encoder
stochasticity or genuine network bifurcation behavior. If the bands
are stable across the constant-current scan but jumpy across the
Poisson scan, the encoder is the cause; if they jump in both, it's
network dynamics. The control video lands as *scan_constant.mp4*.

Also writes numbers.json with mean E and I population rates from an
in-Python replay of the canonical input-rate run so the MDX can
interpolate exact values.

Notebook entry: src/docs/src/pages/notebooks/nb003.mdx
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

import sh  # noqa: E402

from _modal import append_modal_args, parse_modal_gpu  # noqa: E402
from _ping_scan import (  # noqa: E402
    DATASET, DIGIT_CLASS, DT_MS, N_HIDDEN, OSCILLOSCOPE,
    SAMPLE_IDX, SCAN_FPS, SEED, SIM_MS, STEP_OFF_MS, STEP_ON_MS,
    TIER_FRAMES, W_IN_MEAN, W_IN_STD,
    ScanSpec, _overlay_stamp_video, _render_stamp_png, run_scan,
)

SLUG = "nb003"

# Primary (Poisson) scan: sweep per-pixel input rate.
SCAN_MIN = 25.0          # 25 Hz → barely-driven control
SCAN_MAX = 1000.0        # 1 kHz → strong drive
CANON_INPUT_RATE = 250.0  # canonical input rate for the replay (suprathreshold for flat envelope)

# Companion (constant-current) scan: sweep tonic ext_g amplitude (µS).
# No image structure (uniform across all E), no Poisson stochasticity —
# isolates the encoder's contribution to the band-jumping at low rates.
# Range chosen empirically from a probe of network response:
#   0.001 µS → E=12 Hz, I=30 Hz    (sparse, async-like)
#   0.01  µS → E=38 Hz, I=96 Hz    (moderate, near PING onset)
#   0.1   µS → E=161 Hz, I=412 Hz  (saturated)
#   0.5   µS → E=334 Hz (firing-rate cap)
# Saturation kicks in by ~0.1, so picking a narrower upper end keeps
# most frames in the interesting transition zone rather than the flat
# saturated regime.
COMPANION_BIAS_MIN = 0.0
COMPANION_BIAS_MAX = 0.05


def compute_summary_rates() -> dict:
    """Replay one canonical-input-rate forward pass in-process with MNIST
    d0s0 spike input, then extract mean E/I population rates."""
    import torch  # noqa: E402
    import config as C  # noqa: E402
    import models as M  # noqa: E402
    from config import make_net, patch_dt  # noqa: E402
    from oscilloscope import (
        _extract_records, _load_dataset_image, encode_image_spikes, primary_hid_key,
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
        pixel_vec, M.T_steps, DT_MS, rate, rate,
        C.STEP_ON_MS, C.STEP_OFF_MS, C.SEED,
    ).to(C.DEVICE)

    net = make_net(C.cfg,
                   w_in=(W_IN_MEAN, W_IN_STD, "normal", C.W_IN_SPARSITY),
                   model_name="ping")
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


def _run_companion_scan(tier: str, notebook_run_id: str) -> Path | None:
    """Dispatch the constant-current control scan: same network, same dt/T,
    but --input synthetic-conductance + --scan-var bias. Output stamped to
    figures/nb003/scan_constant.mp4. Returns the published path."""
    artifacts = REPO / "src" / "artifacts" / "notebooks" / SLUG / "scan_constant"
    figures = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG
    if "--skip-training" in sys.argv:
        if not (artifacts / "scan.mp4").exists():
            print(f"  [companion] --skip-training but {artifacts}/scan.mp4 missing — skipping")
            return None
    else:
        modal_gpu = parse_modal_gpu(sys.argv)
        frames = TIER_FRAMES[tier]
        print(f"[bias] → {artifacts.relative_to(REPO)}"
              + (f"  [modal:{modal_gpu}]" if modal_gpu else ""))
        args = [
            "run", "python", str(OSCILLOSCOPE), "video",
            "--model", "ping",
            "--n-hidden", str(N_HIDDEN),
            "--t-ms", str(SIM_MS),
            "--input", "synthetic-conductance",
            "--scan-var", "bias",
            "--scan-min", str(COMPANION_BIAS_MIN),
            "--scan-max", str(COMPANION_BIAS_MAX),
            "--frames", str(frames),
            "--frame-rate", str(SCAN_FPS),
            "--out-dir", str(artifacts),
            "--wipe-dir",
            "--w-in", str(W_IN_MEAN), str(W_IN_STD),
            "--dt", str(DT_MS),
        ]
        args = append_modal_args(args, modal_gpu)
        sh.uv(*args, _cwd=str(REPO), _out=sys.stdout, _err=sys.stderr)

    src = artifacts / "scan.mp4"
    if not src.exists():
        print(f"  [companion] expected {src}, not found — skipping stamp")
        return None
    stamp = figures / "_stamp.png"
    _render_stamp_png(notebook_run_id, stamp)
    dst = figures / "scan_constant.mp4"
    _overlay_stamp_video(src, dst, stamp)
    stamp.unlink(missing_ok=True)
    return dst


def extras(tier: str, notebook_run_id: str) -> dict:
    rates = compute_summary_rates()
    print(f"  mean rate  E={rates['mean']['e']:.1f} Hz  "
          f"I={rates['mean']['i']:.1f} Hz")
    companion_video = _run_companion_scan(tier, notebook_run_id)
    return {
        "rates_hz": rates,
        "canonical_input_rate_hz": CANON_INPUT_RATE,
        "companion_scan": {
            "video": ("/" + str(companion_video.relative_to(companion_video.parents[2]))
                      if companion_video else None),
            "scan_var": "bias",
            "scan_min": COMPANION_BIAS_MIN,
            "scan_max": COMPANION_BIAS_MAX,
            "input_mode": "synthetic-conductance",
            "purpose": ("constant-current control: isolates whether band-"
                        "jumping at low Poisson rates is encoder noise or "
                        "network dynamics"),
        },
    }


def evaluate_success(figures_dir, summary):
    """Criteria: scan video rendered, and PING actually forms at the
    canonical input rate (E and I both fire over the run)."""
    primary = figures_dir / "scan_input_rate.mp4"
    primary_ok = primary.exists() and primary.stat().st_size > 0
    primary_href = ("/" + str(primary.relative_to(figures_dir.parents[2]))
                    if primary_ok else None)

    companion = figures_dir / "scan_constant.mp4"
    companion_ok = companion.exists() and companion.stat().st_size > 0
    companion_href = ("/" + str(companion.relative_to(figures_dir.parents[2]))
                      if companion_ok else None)

    rates = summary.get("rates_hz", {})
    i_mean = rates.get("mean", {}).get("i", 0.0)
    e_mean = rates.get("mean", {}).get("e", 0.0)
    ping_formed = i_mean > 1.0 and e_mean > 0.0
    return [
        {
            "label": "input-rate scan video rendered",
            "passed": bool(primary_ok),
            "detail": f"{primary.name} ({primary.stat().st_size} bytes)" if primary_ok
                      else f"missing {primary.name}",
            "detail_href": primary_href,
        },
        {
            "label": "constant-current control video rendered",
            "passed": bool(companion_ok),
            "detail": f"{companion.name} ({companion.stat().st_size} bytes)" if companion_ok
                      else f"missing {companion.name}",
            "detail_href": companion_href,
        },
        {
            "label": f"PING forms at canonical input rate ({CANON_INPUT_RATE:.0f} Hz)",
            "passed": bool(ping_formed),
            "detail": f"E={e_mean:.1f} Hz, I={i_mean:.1f} Hz",
        },
    ]


if __name__ == "__main__":
    run_scan(ScanSpec(
        slug=SLUG,
        scan_var="spike_rate",
        scan_min=SCAN_MIN,
        scan_max=SCAN_MAX,
        video_name="scan_input_rate.mp4",
        extra_osc_args=[
            "--stim-overdrive", "1.0",
            "--w-in", str(W_IN_MEAN), str(W_IN_STD),
            "--dt", str(DT_MS),
        ],
        extras_fn=extras,
        criteria_fn=evaluate_success,
    ))
    sys.exit(0)
