"""Notebook runner for entry 003 — PING integration-step sweep.

Sweeps the integration timestep *dt* from 0.05 → 2 ms with the input
overdrive held well past the PING threshold (fixed 10×), isolating dt
as the only knob that can break the rhythm. Fine dt is stable; coarse
dt distorts or saturates. Split out from the original nb002 basic-PING
notebook; companion to nb002 (stim-overdrive) and nb004 (ei-strength).

Also writes numbers.json with pre/stim/post E and I population rates
from an in-Python replay at the canonical fine dt so the MDX can
interpolate exact values and the success-criteria check can gate on
PING actually forming at fine dt.

Notebook entry: src/docs/src/pages/notebooks/nb003.mdx
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

import numpy as np  # noqa: E402

from _ping_scan import (  # noqa: E402
    DATASET, DIGIT_CLASS, INPUT_RATE_HZ, N_HIDDEN,
    SAMPLE_IDX, SEED, SIM_MS, STEP_OFF_MS, STEP_ON_MS, W_IN_OVERDRIVE,
    ScanSpec, run_scan,
)

SLUG = "nb003"
DT_SCAN_OVERDRIVE = 10.0  # pinned above PING threshold so dt is the only variable
DT_SCAN_MIN = 0.05
DT_SCAN_MAX = 2.0
CANON_DT_MS = 0.1         # canonical fine dt for the replay — well inside stable band


def compute_summary_rates() -> dict:
    """Replay one canonical-dt forward pass in-process with MNIST d0s0
    spike input, then extract pre/stim/post E/I population rates."""
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
    M.max_rate_hz = INPUT_RATE_HZ
    patch_dt(CANON_DT_MS)

    base_rate = M.max_rate_hz
    stim_rate = base_rate * DT_SCAN_OVERDRIVE
    input_spikes = encode_image_spikes(
        pixel_vec, M.T_steps, CANON_DT_MS, base_rate, stim_rate,
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
    t_ms = np.arange(T_steps) * CANON_DT_MS

    pre  = (t_ms >= 0)           & (t_ms < STEP_ON_MS)
    stim = (t_ms >= STEP_ON_MS)  & (t_ms < STEP_OFF_MS)
    post = (t_ms >= STEP_OFF_MS) & (t_ms <= SIM_MS)

    def mean_rate(spk, mask):
        return float(spk[mask].mean()) * 1000.0 / CANON_DT_MS

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
    return {"rates_hz": rates, "canonical_dt_ms": CANON_DT_MS}


def evaluate_success(figures_dir, summary):
    """Criteria: scan video rendered, and PING actually forms at the
    canonical fine dt (I fires during the stim window). The I-stim check
    mirrors nb002 — guards against a regression where the stim window
    never fires and every frame lands in flat baseline."""
    video = figures_dir / "scan_dt.mp4"
    video_ok = video.exists() and video.stat().st_size > 0
    href = "/" + str(video.relative_to(figures_dir.parents[2])) if video_ok else None

    rates = summary.get("rates_hz", {})
    i_stim = rates.get("stim", {}).get("i", 0.0)
    i_pre = rates.get("pre", {}).get("i", 0.0)
    ping_formed = i_stim > 1.0 and i_stim > i_pre
    return [
        {
            "label": "dt scan video rendered",
            "passed": bool(video_ok),
            "detail": f"{video.name} ({video.stat().st_size} bytes)" if video_ok
                      else f"missing {video.name}",
            "detail_href": href,
        },
        {
            "label": f"PING forms at canonical dt ({CANON_DT_MS} ms)",
            "passed": bool(ping_formed),
            "detail": f"I pre={i_pre:.1f} Hz → stim={i_stim:.1f} Hz",
        },
    ]


if __name__ == "__main__":
    run_scan(ScanSpec(
        slug=SLUG,
        scan_var="dt",
        scan_min=DT_SCAN_MIN,
        scan_max=DT_SCAN_MAX,
        video_name="scan_dt.mp4",
        extra_osc_args=[
            "--input-rate", str(INPUT_RATE_HZ),
            "--w-in-overdrive", str(W_IN_OVERDRIVE),
            "--stim-overdrive", str(DT_SCAN_OVERDRIVE),
        ],
        config_payload={"fixed_overdrive": DT_SCAN_OVERDRIVE},
        extras_fn=extras,
        criteria_fn=evaluate_success,
    ))
    sys.exit(0)
