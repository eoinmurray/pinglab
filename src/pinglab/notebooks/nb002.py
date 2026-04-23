"""Notebook runner for entry 002 — PING stim-overdrive video.

Sweeps the Poisson-input overdrive factor (in-window rate multiplier)
from 1× → 10× while everything else is held fixed, and produces the
canonical *scan_overdrive.mp4* frame-by-frame video. Input is MNIST
digit 0 sample 0. Companion dt and ei-strength scans live in nb010
and nb011 respectively.

Also writes numbers.json with pre/stim/post E and I population rates
from an in-Python replay of the canonical overdrive=5× run so the MDX
can interpolate exact values.

Notebook entry: src/docs/src/pages/notebooks/nb002.mdx
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))

import numpy as np  # noqa: E402

from _ping_scan import (  # noqa: E402
    DATASET, DIGIT_CLASS, DT_MS, INPUT_RATE_HZ, N_HIDDEN,
    SAMPLE_IDX, SEED, SIM_MS, STEP_OFF_MS, STEP_ON_MS, W_IN_OVERDRIVE,
    ScanSpec, run_scan,
)

SLUG = "nb002"
SCAN_MIN = 1.0           # overdrive=1 → no elevation (control)
SCAN_MAX = 10.0          # overdrive=10 → strong elevation
CANON_OVERDRIVE = 5.0    # canonical in-window rate multiplier for the replay


def compute_summary_rates() -> dict:
    """Replay one canonical-overdrive forward pass in-process with MNIST
    d0s0 spike input, then extract pre/stim/post E/I population rates."""
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


def extras(tier: str, notebook_run_id: str) -> dict:
    rates = compute_summary_rates()
    r = rates
    print(f"  E rate  pre={r['pre']['e']:.1f} Hz  "
          f"stim={r['stim']['e']:.1f} Hz  post={r['post']['e']:.1f} Hz")
    print(f"  I rate  pre={r['pre']['i']:.1f} Hz  "
          f"stim={r['stim']['i']:.1f} Hz  post={r['post']['i']:.1f} Hz")
    return {"rates_hz": rates, "canonical_overdrive": CANON_OVERDRIVE}


if __name__ == "__main__":
    run_scan(ScanSpec(
        slug=SLUG,
        scan_var="stim-overdrive",
        scan_min=SCAN_MIN,
        scan_max=SCAN_MAX,
        video_name="scan_overdrive.mp4",
        extra_osc_args=[
            "--input-rate", str(INPUT_RATE_HZ),
            "--w-in-overdrive", str(W_IN_OVERDRIVE),
            "--dt", str(DT_MS),
        ],
        extras_fn=extras,
    ))
    sys.exit(0)
