"""Notebook runner for entry 004 — PING E→I coupling sweep.

Sweeps the E→I coupling strength from 0 → 1 with *no* stim-window
overdrive (input rate flat through the trial), walking the network
from the async baseline (E and I effectively decoupled) through the
emergence of gamma as the E→I→E feedback loop closes. Input rate and
W_in are bumped relative to the other scans so E has enough baseline
drive to recruit I at all. Split out from the original nb002
basic-PING notebook; companion to nb002 (stim-overdrive) and nb003
(dt).

Also writes numbers.json with pre/stim/post E and I population rates
from an in-Python replay at the canonical high-ei run so the MDX can
interpolate exact values and the success-criteria check can gate on
PING actually forming once the feedback loop is closed.

Notebook entry: src/docs/src/pages/notebooks/nb004.mdx
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

import numpy as np  # noqa: E402

from _ping_scan import (  # noqa: E402
    DATASET, DIGIT_CLASS, DT_MS, INPUT_RATE_HZ, N_HIDDEN,
    SAMPLE_IDX, SEED, SIM_MS, STEP_OFF_MS, STEP_ON_MS,
    ScanSpec, run_scan,
)

SLUG = "nb004"
EI_SCAN_INPUT_RATE_HZ = 2 * INPUT_RATE_HZ
EI_SCAN_W_IN_OVERDRIVE = 3.0
EI_SCAN_MIN = 0.0
EI_SCAN_MAX = 1.0   # past 1 the E rate is already saturated-low
CANON_EI = 0.8      # well inside the PING-on regime for the replay


def compute_summary_rates() -> dict:
    """Replay one canonical-ei forward pass in-process with MNIST d0s0
    spike input (flat rate, no stim-window overdrive), then extract
    pre/stim/post E/I population rates."""
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
    s = CANON_EI
    C.cfg.w_ei = (s, s * 0.1)
    C.cfg.w_ie = (s * C.cfg.ei_ratio, s * C.cfg.ei_ratio * 0.1)
    C._sync_globals_from_cfg(C.cfg)

    pixel_vec, _ = _load_dataset_image(DATASET, DIGIT_CLASS, SAMPLE_IDX)
    M.N_IN = len(pixel_vec)
    M.N_HID = C.N_E
    M.N_INH = C.N_I
    M.max_rate_hz = EI_SCAN_INPUT_RATE_HZ
    patch_dt(DT_MS)

    base_rate = M.max_rate_hz
    input_spikes = encode_image_spikes(
        pixel_vec, M.T_steps, DT_MS, base_rate, base_rate,
        C.STEP_ON_MS, C.STEP_OFF_MS, C.SEED,
    ).to(C.DEVICE)

    net = make_net(C.cfg,
                   w_in=(*C.W_IN_SPIKES, "normal", C.W_IN_SPARSITY),
                   model_name="ping")
    if EI_SCAN_W_IN_OVERDRIVE != 1.0:
        with torch.no_grad():
            net.W_ff[0].mul_(EI_SCAN_W_IN_OVERDRIVE)
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
    }


def extras(tier: str, notebook_run_id: str) -> dict:
    rates = compute_summary_rates()
    r = rates
    print(f"  E rate  pre={r['pre']['e']:.1f} Hz  "
          f"stim={r['stim']['e']:.1f} Hz  post={r['post']['e']:.1f} Hz")
    print(f"  I rate  pre={r['pre']['i']:.1f} Hz  "
          f"stim={r['stim']['i']:.1f} Hz  post={r['post']['i']:.1f} Hz")
    return {"rates_hz": rates, "canonical_ei": CANON_EI}


def evaluate_success(figures_dir, summary):
    """Criteria: scan video rendered, and PING actually forms at the
    canonical high-ei coupling (I fires across the full trial since
    input is flat). The I-rate check mirrors nb002 / nb003 — guards
    against a regression where the network never recruits I."""
    video = figures_dir / "scan_ei.mp4"
    video_ok = video.exists() and video.stat().st_size > 0
    href = "/" + str(video.relative_to(figures_dir.parents[2])) if video_ok else None

    rates = summary.get("rates_hz", {})
    i_stim = rates.get("stim", {}).get("i", 0.0)
    e_stim = rates.get("stim", {}).get("e", 0.0)
    ping_formed = i_stim > 1.0 and e_stim > 1.0
    return [
        {
            "label": "ei-strength scan video rendered",
            "passed": bool(video_ok),
            "detail": f"{video.name} ({video.stat().st_size} bytes)" if video_ok
                      else f"missing {video.name}",
            "detail_href": href,
        },
        {
            "label": f"PING forms at canonical ei ({CANON_EI})",
            "passed": bool(ping_formed),
            "detail": f"E stim={e_stim:.1f} Hz, I stim={i_stim:.1f} Hz",
        },
    ]


if __name__ == "__main__":
    run_scan(ScanSpec(
        slug=SLUG,
        scan_var="ei_strength",
        scan_min=EI_SCAN_MIN,
        scan_max=EI_SCAN_MAX,
        video_name="scan_ei.mp4",
        extra_osc_args=[
            "--input-rate", str(EI_SCAN_INPUT_RATE_HZ),
            "--w-in-overdrive", str(EI_SCAN_W_IN_OVERDRIVE),
            "--stim-overdrive", "1.0",
            "--dt", str(DT_MS),
        ],
        config_payload={
            "fixed_overdrive": 1.0,
            "input_rate_hz": EI_SCAN_INPUT_RATE_HZ,
            "w_in_overdrive": EI_SCAN_W_IN_OVERDRIVE,
        },
        extras_fn=extras,
        criteria_fn=evaluate_success,
    ))
    sys.exit(0)
