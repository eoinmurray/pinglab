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

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from _ping_scan import (  # noqa: E402
    DATASET,
    DIGIT_CLASS,
    DT_MS,
    N_HIDDEN,
    SAMPLE_IDX,
    SEED,
    SIM_MS,
    STEP_OFF_MS,
    STEP_ON_MS,
    W_IN_MEAN,
    W_IN_STD,
    ScanSpec,
    run_scan,
)

SLUG = "nb004"
SCAN_MIN = 25.0  # 25 Hz → barely-driven control
SCAN_MAX = 1000.0  # 1 kHz → strong drive
CANON_INPUT_RATE = (
    250.0  # canonical input rate for the replay (suprathreshold for flat envelope)
)


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
    run_scan(
        ScanSpec(
            slug=SLUG,
            scan_var="spike_rate",
            scan_min=SCAN_MIN,
            scan_max=SCAN_MAX,
            video_name="scan_input_rate.mp4",
            extra_osc_args=[
                "--stim-overdrive",
                "1.0",
                "--w-in",
                str(W_IN_MEAN),
                str(W_IN_STD),
                "--dt",
                str(DT_MS),
            ],
            extras_fn=extras,
        )
    )
    sys.exit(0)
