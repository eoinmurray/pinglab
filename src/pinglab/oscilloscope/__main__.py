"""Oscilloscope -- PING network toolkit.

CLI entrypoint with subcommands: sim, image, video, train.

Usage:
    uv run python src/pinglab/oscilloscope.py                             # sim only (metrics)
    uv run python src/pinglab/oscilloscope.py sim                         # sim only (metrics)
    uv run python src/pinglab/oscilloscope.py image                       # snapshot
    uv run python src/pinglab/oscilloscope.py video --scan-var dt         # dt sweep video
    uv run python src/pinglab/oscilloscope.py train --epochs 10           # train on scikit digits
"""

from __future__ import annotations

import json
import math
import sys
import time as _time
from pathlib import Path

# Ensure src/pinglab/ is FIRST on sys.path so the sibling top-level modules
# (models, inputs, metrics, config, plot) are importable as bare names.
_pkg_dir = str(Path(__file__).resolve().parent.parent)  # src/pinglab/
if _pkg_dir in sys.path:
    sys.path.remove(_pkg_dir)
sys.path.insert(0, _pkg_dir)

import logging
import numpy as np
import torch
from torch import nn

log = logging.getLogger("oscilloscope")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

import models as M
from inputs import (
    make_spike_drive,
    make_step_drive,
    make_reference_noise,
    make_step_drive_from_ref,
    DT_CAL,
)
from config import (
    Config,
    cfg,
    _MODEL_CLASSES,
    N_E,
    N_I,
    FPS,
    SEED,
    SIM_MS,
    BURN_IN_MS,
    T_E_ASYNC_DEFAULT,
    SIGMA_E,
    STEP_ON_MS,
    STEP_OFF_MS,
    W_EI,
    W_IE,
    NOISE_SIGMA,
    NOISE_TAU,
    SPIKE_RATE_BASE,
    ARTIFACT_ROOT,
    W_IN_SPIKES,
    W_IN_SPARSITY,
    BIAS,
    EI_RATIO,
    DEVICE,
    patch_dt,
    run_sim,
    run_sim_batch,
    run_sim_image,
    _extract_records,
    extract_weights,
    make_net,
    make_ping_net,
    _run_sim_with_net,
    build_net,
    build_config,
    _sync_globals_from_cfg,
)
from metrics import (
    report_metrics,
    metrics_str,
    compute_metrics,
    format_metrics,
)
from plot import (
    prof,
    make_transient_fig,
    draw_transient_frame,
    reset_weight_xlims,
    LAYOUT_PRESETS,
    PANEL_CATALOG,
    ACTIVE_PANELS,
    CLR,
)


# =============================================================================
# =============================================================================
# Image → spike encoding with stimulus window
# =============================================================================


from oscilloscope.encoders import (  # noqa: E402,F401
    EVAL_SEED,
    FROZEN_MODES,
    FrozenEncoder,
    downsample_spikes_count,
    encode_batch,
    encode_image_spikes,
    encode_images_poisson,
    encode_smnist,
    transport_spikes_bin,
    upsample_spikes_zeropad,
)


# Scannable variables
# =============================================================================


def _auto_device() -> torch.device:
    """Pick the fastest available device: cuda > mps > cpu.

    Called when the user doesn't explicitly set --device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


SCAN_DEFAULTS = {
    "stim-overdrive": (1.0, "x"),
    "tau_gaba": (9.0, "ms"),
    "tau_ampa": (2.0, "ms"),
    "w_ei_mean": (W_EI[0], "\u03bcS"),
    "w_ie_mean": (W_IE[0], "\u03bcS"),
    "ei_strength": (1.0, ""),
    "spike_rate": (1.0, "Hz"),
    "bias": (1.0, "\u03bcS"),
    "dt": (1.0, "ms"),
    "digit": (0, ""),  # dataset digit class (0-9)
    "noise": (0, "Hz"),  # Poisson noise rate added to input
}


def _apply_scan_var(var_name, value):
    """Mutate model/oscilloscope state for a scan variable."""
    import config as C

    if var_name == "tau_gaba":
        M.tau_gaba = value
        M.decay_gaba = np.exp(-M.dt / value)
    elif var_name == "tau_ampa":
        M.tau_ampa = value
        M.decay_ampa = np.exp(-M.dt / value)
    elif var_name == "w_ei_mean":
        C.W_EI = (value, C.W_EI[1])
        C.cfg.w_ei = C.W_EI
    elif var_name == "w_ie_mean":
        C.W_IE = (value, C.W_IE[1])
        C.cfg.w_ie = C.W_IE
    elif var_name == "ei_strength":
        C.W_EI = (value, value * 0.1)
        C.W_IE = (value * C.EI_RATIO, value * C.EI_RATIO * 0.1)
        C.cfg.w_ei = C.W_EI
        C.cfg.w_ie = C.W_IE
    elif var_name == "bias":
        C.BIAS = value
        C.cfg.bias = value


# =============================================================================
# Scan generators
# =============================================================================

# Weight scan variables that should scale an existing matrix, not resample
_WEIGHT_SCAN_VARS = {"w_ei_mean": "W_ei", "w_ie_mean": "W_ie"}

# Human-readable x-axis labels for sweep ladder panels
_SWEEP_XLABELS = {
    "stim-overdrive": "overdrive (×)",
    "ei_strength": "E→I strength",
    "spike_rate": "input rate (Hz)",
    "bias": "bias (\u03bcS)",
    "digit": "digit class",
    "noise": "noise rate (Hz)",
    "w_ei_mean": "W_ei mean",
    "w_ie_mean": "W_ie mean",
    "dt": "dt (ms)",
}


def generate_scan(
    scan_var="stim-overdrive",
    scan_min=1.0,
    scan_max=50.0,
    n_frames=None,
    t_e_async=None,
    overdrive=12.0,
    resample_input=False,
    spike_rate=None,
    input_mode="synthetic-conductance",
    dataset="scikit",
    digit_class=0,
    sample_idx=0,
    load_weights=None,
):
    """Video scanning a variable from scan_min to scan_max."""
    import config as C

    if t_e_async is None:
        t_e_async = C.T_E_ASYNC_DEFAULT
    if spike_rate is None:
        spike_rate = C.SPIKE_RATE_BASE

    n_frames = n_frames or 10
    out_dir = Path(C.ARTIFACT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    dt = DT_CAL
    burn_steps = int(C.BURN_IN_MS / dt)

    default_val = SCAN_DEFAULTS[scan_var][0]
    unit = SCAN_DEFAULTS[scan_var][1]
    scan_values = np.linspace(
        default_val * scan_min, default_val * scan_max, n_frames
    ).tolist()

    if scan_var in ("stim-overdrive", "ei_strength", "spike_rate", "bias", "dt"):
        scan_values = np.linspace(scan_min, scan_max, n_frames).tolist()
    elif scan_var == "digit":
        # Integer digit classes 0-9 (or custom range)
        lo = int(scan_min) if scan_min >= 1 else 0
        hi = int(scan_max) if scan_max >= 1 else 9
        scan_values = list(range(lo, hi + 1))
        n_frames = len(scan_values)
    elif scan_var == "noise":
        # Linear Hz sweep
        scan_values = np.linspace(scan_min, scan_max, n_frames).tolist()

    display_values = scan_values

    prof.reset()
    log.info(
        f"scan {scan_var} {scan_values[0]:.3g}\u2192{scan_values[-1]:.3g}{unit} | {n_frames}f"
    )

    if (
        scan_var == "stim-overdrive"
        and not resample_input
        and input_mode not in ("synthetic-spikes", "dataset")
    ):
        return _scan_od_batched(
            scan_values,
            n_frames,
            dt,
            burn_steps,
            t_e_async,
            out_dir,
            display_values,
            unit,
        )

    if scan_var == "dt":
        return _scan_dt(
            scan_values,
            n_frames,
            t_e_async,
            overdrive,
            out_dir,
            display_values,
            unit,
            input_mode=input_mode,
            dataset=dataset,
            digit_class=digit_class,
            sample_idx=sample_idx,
            spike_rate=spike_rate,
        )

    _scan_streaming(
        scan_var,
        scan_values,
        n_frames,
        dt,
        burn_steps,
        t_e_async,
        overdrive,
        out_dir,
        display_values,
        unit,
        resample_input=resample_input,
        spike_rate=spike_rate,
        input_mode=input_mode,
        dataset=dataset,
        digit_class=digit_class,
        sample_idx=sample_idx,
        load_weights=load_weights,
    )


def _scan_od_batched(
    scan_values, n_frames, dt, burn_steps, t_e_async, out_dir, display_values, unit
):
    """OD scan -- batched for speed."""
    import config as C

    T_steps = int(C.SIM_MS / dt)
    t_e_ping_levels = [t_e_async * od for od in scan_values]

    ext_g_list = []
    for t_e_ping in t_e_ping_levels:
        ext_g_sim, _ = make_step_drive(
            C.N_E,
            T_steps,
            dt,
            t_e_async,
            t_e_ping,
            C.STEP_ON_MS,
            C.STEP_OFF_MS,
            C.SIGMA_E,
            C.NOISE_SIGMA,
            C.NOISE_TAU,
            C.SEED,
        )
        ext_g_list.append(ext_g_sim.numpy())

    log.info(f"  batched {n_frames} sims...")
    with prof.track_sim():
        batch_results = run_sim_batch(dt, ext_g_list)

    all_data = []
    for i, (rec, ext_g_raw) in enumerate(batch_results):
        ratio = scan_values[i]
        spk_e = rec[primary_hid_key(rec)][burn_steps:]
        spk_i = rec[primary_inh_key(rec)][burn_steps:]
        spk_o = rec.get("out")
        if spk_o is not None:
            spk_o = spk_o[burn_steps:]
        all_data.append((ratio, spk_e, spk_i, ext_g_raw[burn_steps:], spk_o))
    log.info(f"  {len(all_data)} sims done")

    _, _, sweep_weights = run_sim(dt, t_e_ping_levels[0], t_e_async=t_e_async)

    fig, axes = make_transient_fig(layout="video")
    n_total = len(all_data)
    lo, hi = display_values[0], display_values[-1]
    fname = "scan.mp4"
    writer = FFMpegWriter(fps=C.FPS, metadata=dict(title="Scan OD"))
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"  rendering {n_total}f...")
    with writer.saving(fig, str(out_dir / fname), dpi=120):
        for frame_idx in range(n_total):
            ratio, spk_e, spk_i, ext_g, spk_o = all_data[frame_idx]
            with prof.track_render():
                draw_transient_frame(
                    axes,
                    ratio,
                    spk_e,
                    spk_i,
                    ext_g,
                    dt,
                    "PING",
                    spk_o=spk_o,
                    weights=sweep_weights,
                    sweep_var="OD",
                    sweep_range=(lo, hi),
                    sweep_progress=frame_idx / max(1, n_total - 1),
                    t_e_async=t_e_async,
                    sweep_levels=display_values,
                    sweep_frame_idx=frame_idx,
                    n_e=C.N_E,
                    n_i=C.N_I,
                    step_on_ms=C.STEP_ON_MS,
                    step_off_ms=C.STEP_OFF_MS,
                    burn_in_ms=C.BURN_IN_MS,
                    w_ie=C.W_IE,
                )
            with prof.track_encode():
                fig.savefig(frames_dir / f"frame_{frame_idx + 1:04d}.png", dpi=120)
                writer.grab_frame()

    plt.close(fig)
    log.info(f"  \u2192 {out_dir / fname}")
    log.info(f"  frames \u2192 {frames_dir}/")
    prof.report(n_total)


def _scan_streaming(
    scan_var,
    scan_values,
    n_frames,
    dt,
    burn_steps,
    t_e_async,
    overdrive,
    out_dir,
    display_values,
    unit,
    resample_input=False,
    spike_rate=None,
    input_mode="synthetic-conductance",
    dataset="scikit",
    digit_class=0,
    sample_idx=0,
    load_weights=None,
):
    """Generic scan -- stream frame-by-frame to keep memory bounded."""
    import config as C

    if spike_rate is None:
        spike_rate = C.SPIKE_RATE_BASE
    T_steps = int(C.SIM_MS / dt)
    t_e_ping = t_e_async * overdrive
    is_weight_scan = scan_var in _WEIGHT_SCAN_VARS
    use_spikes = input_mode == "synthetic-spikes"
    use_dataset = input_mode == "dataset"

    input_spikes = None
    tonic_g = None
    img_tensor = None
    _digit_img = None
    _loaded_net = None

    if use_dataset:
        pixel_vec, _digit_img = _load_dataset_image(dataset, digit_class, sample_idx)
        M.N_IN = len(pixel_vec)
        patch_dt(dt)
        img_tensor = (
            torch.tensor(pixel_vec, dtype=torch.float32).unsqueeze(0).to(C.DEVICE)
        )

    # Load trained weights once if provided
    if load_weights is not None:
        M.N_HID = C.N_E
        M.N_INH = C.N_I
        patch_dt(dt)
        _loaded_net = make_net(C.cfg, w_in=(*C.W_IN_SPIKES, "normal", C.W_IN_SPARSITY))
        state = torch.load(load_weights, map_location="cpu")
        _loaded_net.load_state_dict(state, strict=False)
        _loaded_net.eval()
        _loaded_net.recording = True
        log.info(f"  loaded weights: {load_weights}")

    elif use_spikes:
        M.N_HID = C.N_E
        M.N_INH = C.N_I
        patch_dt(dt)
        stim_rate = spike_rate * overdrive
        input_spikes = make_spike_drive(
            M.N_IN,
            T_steps,
            dt,
            spike_rate,
            stim_rate,
            C.STEP_ON_MS,
            C.STEP_OFF_MS,
            C.SEED,
        ).to(C.DEVICE)
        if C.BIAS > 0:
            tonic_g = torch.full(
                (T_steps, C.N_E), C.BIAS, dtype=torch.float32, device=C.DEVICE
            )

    if is_weight_scan:
        _apply_scan_var(scan_var, scan_values[0])
        patch_dt(dt)
        net = make_ping_net(C.cfg)
        w_attr = _WEIGHT_SCAN_VARS[scan_var]
        w_base = getattr(net, w_attr).clone()
        default_val = SCAN_DEFAULTS[scan_var][0]
        _, _, sweep_weights = _run_sim_with_net(net, dt, t_e_ping, t_e_async)
    else:
        _apply_scan_var(scan_var, scan_values[0])
        if use_spikes or use_dataset:
            _net = make_ping_net(
                C.cfg, w_in=(*C.W_IN_SPIKES, "normal", C.W_IN_SPARSITY)
            )
            sweep_weights = extract_weights(_net)
        else:
            _, _, sweep_weights = run_sim(dt, t_e_ping, t_e_async=t_e_async)

    vid_layout = "sweep_video" if use_dataset else "video"
    fig, axes = make_transient_fig(layout=vid_layout)
    n_total = len(scan_values)
    sweep_label = scan_var
    lo, hi = display_values[0], display_values[-1]
    fname = "scan.mp4"

    writer = FFMpegWriter(fps=C.FPS, metadata=dict(title=f"Scan {scan_var}"))
    pred = None  # updated each frame when using dataset input
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"  sim+render {n_total}f...")
    with writer.saving(fig, str(out_dir / fname), dpi=120):
        for i, val in enumerate(scan_values):
            noise_seed = C.SEED + i if resample_input else None

            with prof.track_sim():
                if is_weight_scan:
                    scale = val / default_val
                    with torch.no_grad():
                        getattr(net, w_attr).copy_(w_base * scale)
                    rec, ext_g_np, frame_weights = _run_sim_with_net(
                        net, dt, t_e_ping, t_e_async, noise_seed=noise_seed
                    )
                elif use_dataset:
                    # digit scan: reload image for each class
                    if scan_var == "digit":
                        digit_class = int(val)
                        pixel_vec, _digit_img = _load_dataset_image(
                            dataset, digit_class, sample_idx
                        )
                    elif scan_var != "noise":
                        _apply_scan_var(scan_var, val)
                    patch_dt(dt)
                    if _loaded_net is not None:
                        net_frame = _loaded_net
                    else:
                        net_frame = make_net(
                            C.cfg, w_in=(*C.W_IN_SPIKES, "normal", C.W_IN_SPARSITY)
                        )
                    # Encode image as spikes with stimulus window
                    od_val = val if scan_var == "stim-overdrive" else overdrive
                    base_rate = val if scan_var == "spike_rate" else M.max_rate_hz
                    stim_rate = base_rate * od_val
                    frame_spikes = encode_image_spikes(
                        pixel_vec,
                        T_steps,
                        dt,
                        base_rate,
                        stim_rate,
                        C.STEP_ON_MS,
                        C.STEP_OFF_MS,
                        C.SEED,
                    ).to(C.DEVICE)
                    # noise scan: add Poisson noise spikes to input
                    if scan_var == "noise" and val > 0:
                        noise_p = val * dt / 1000.0  # per-step probability
                        noise = (torch.rand_like(frame_spikes) < noise_p).float()
                        frame_spikes = (frame_spikes + noise).clamp(max=1.0)
                    tonic_dataset = None
                    if C.BIAS > 0:
                        tonic_dataset = torch.full(
                            (T_steps, C.N_E),
                            C.BIAS,
                            dtype=torch.float32,
                            device=C.DEVICE,
                        )
                    with torch.no_grad():
                        logits = net_frame.forward(
                            input_spikes=frame_spikes, ext_g=tonic_dataset
                        )
                    pred = int(logits.argmax(dim=-1)[0].item())
                    rec = _extract_records(net_frame)
                    ext_g_np = frame_spikes.cpu().numpy()
                    frame_weights = extract_weights(net_frame)
                elif use_spikes:
                    _apply_scan_var(scan_var, val)
                    patch_dt(dt)
                    if scan_var == "stim-overdrive":
                        frame_stim_rate = spike_rate * val
                        frame_spikes = make_spike_drive(
                            M.N_IN,
                            T_steps,
                            dt,
                            spike_rate,
                            frame_stim_rate,
                            C.STEP_ON_MS,
                            C.STEP_OFF_MS,
                            C.SEED,
                        ).to(C.DEVICE)
                        frame_tonic = tonic_g
                    elif scan_var == "spike_rate":
                        frame_stim_rate = val * overdrive
                        frame_spikes = make_spike_drive(
                            M.N_IN,
                            T_steps,
                            dt,
                            val,
                            frame_stim_rate,
                            C.STEP_ON_MS,
                            C.STEP_OFF_MS,
                            C.SEED,
                        ).to(C.DEVICE)
                        frame_tonic = tonic_g
                    elif scan_var == "bias":
                        frame_spikes = input_spikes
                        frame_tonic = (
                            torch.full(
                                (T_steps, C.N_E),
                                val,
                                dtype=torch.float32,
                                device=C.DEVICE,
                            )
                            if val > 0
                            else None
                        )
                    else:
                        frame_spikes = input_spikes
                        frame_tonic = tonic_g
                    net_frame = make_ping_net(
                        C.cfg, w_in=(*C.W_IN_SPIKES, "normal", C.W_IN_SPARSITY)
                    )
                    with torch.no_grad():
                        net_frame.forward(input_spikes=frame_spikes, ext_g=frame_tonic)
                    rec = _extract_records(net_frame)
                    assert frame_spikes is not None
                    ext_g_np = frame_spikes.cpu().numpy()
                    frame_weights = extract_weights(net_frame)
                else:
                    _apply_scan_var(scan_var, val)
                    if resample_input:
                        T_steps_r = int(C.SIM_MS / dt)
                        ext_g_sim, _ = make_step_drive(
                            C.N_E,
                            T_steps_r,
                            dt,
                            t_e_async,
                            t_e_ping,
                            C.STEP_ON_MS,
                            C.STEP_OFF_MS,
                            C.SIGMA_E,
                            C.NOISE_SIGMA,
                            C.NOISE_TAU,
                            C.SEED,
                            noise_seed=noise_seed,
                        )
                        rec, ext_g_np, frame_weights = run_sim(
                            dt, t_e_ping, ext_g_override=ext_g_sim, t_e_async=t_e_async
                        )
                    else:
                        rec, ext_g_np, frame_weights = run_sim(
                            dt, t_e_ping, t_e_async=t_e_async
                        )

            spk_e = rec[primary_hid_key(rec)][burn_steps:]
            spk_i = rec[primary_inh_key(rec)][burn_steps:]
            spk_o = rec.get("out")
            if spk_o is not None:
                spk_o = spk_o[burn_steps:]

            # Build title with prediction info when using dataset input
            if use_dataset and pred is not None:
                truth = int(val) if scan_var == "digit" else digit_class
                correct = pred == truth
                mark = chr(10003) if correct else chr(10007)
                frame_title = f"d{truth} pred={pred} {mark}"
            else:
                frame_title = "PING"

            sweep_xlabel = _SWEEP_XLABELS.get(scan_var, scan_var)
            with prof.track_render():
                draw_transient_frame(
                    axes,
                    overdrive,
                    spk_e,
                    spk_i,
                    ext_g_np[burn_steps:],
                    dt,
                    frame_title,
                    spk_o=spk_o,
                    weights=frame_weights,
                    sweep_var=sweep_label,
                    sweep_range=(lo, hi),
                    sweep_progress=i / max(1, n_total - 1),
                    t_e_async=t_e_async,
                    sweep_levels=display_values,
                    sweep_frame_idx=i,
                    n_e=C.N_E,
                    n_i=C.N_I,
                    step_on_ms=C.STEP_ON_MS,
                    step_off_ms=C.STEP_OFF_MS,
                    burn_in_ms=C.BURN_IN_MS,
                    w_ie=C.W_IE,
                    digit_image=_digit_img if use_dataset else None,
                    sweep_xlabel=sweep_xlabel,
                )
            with prof.track_encode():
                fig.savefig(frames_dir / f"frame_{i + 1:04d}.png", dpi=120)
                writer.grab_frame()

            m = metrics_str(spk_e, spk_i, dt, n_e=C.N_E, n_i=C.N_I)
            pred_str = ""
            if use_dataset and pred is not None:
                truth = int(val) if scan_var == "digit" else digit_class
                mark = chr(10003) if pred == truth else chr(10007)
                pred_str = f" d{truth}\u2192pred={pred}{mark}"
            log.info(f"  {i + 1}/{n_total} {scan_var}={val:.3g}{unit} | {m}{pred_str}")

    plt.close(fig)
    log.info(f"  \u2192 {out_dir / fname}")
    log.info(f"  frames \u2192 {frames_dir}/")
    prof.report(n_total)


def _scan_dt(
    scan_values,
    n_frames,
    t_e_async,
    overdrive,
    out_dir,
    display_values,
    unit,
    input_mode="synthetic-conductance",
    dataset="scikit",
    digit_class=0,
    sample_idx=0,
    spike_rate=None,
):
    """dt sweep. Two drive modes:
    * conductance (default): reference-noise step drive held fixed across
      dt so each frame is a dt-invariant re-sampling of the same process.
    * dataset: re-encodes the same MNIST pixel vector as Poisson spikes at
      each frame's dt (expected per-pixel rate is dt-invariant); drives a
      PING net through W_in. Dataset mode is what notebook 002 uses so the
      overdrive and dt videos share an input pipeline.
    """
    import config as C

    t_e_ping = t_e_async * overdrive
    use_dataset = input_mode == "dataset"

    if use_dataset:
        pixel_vec, digit_image = _load_dataset_image(dataset, digit_class, sample_idx)
        M.N_IN = len(pixel_vec)
        M.N_HID = C.N_E
        M.N_INH = C.N_I
        patch_dt(scan_values[0])
        ref_net = make_ping_net(C.cfg, w_in=(*C.W_IN_SPIKES, "normal", C.W_IN_SPARSITY))
        dt_weights = extract_weights(ref_net)
        layout = "sweep_video"
    else:
        digit_image = None
        print("  dt scan uses conductance input (reference noise for dt-invariance)...")
        X_i, eta_ref = make_reference_noise(
            C.N_E, C.SIM_MS, C.NOISE_SIGMA, C.NOISE_TAU, C.SEED
        )
        _, _, dt_weights = run_sim(scan_values[0], t_e_ping, t_e_async=t_e_async)
        layout = "video"

    fig, axes = make_transient_fig(layout=layout)
    n_total = len(scan_values)
    lo, hi = display_values[0], display_values[-1]
    fname = "scan.mp4"

    writer = FFMpegWriter(fps=C.FPS, metadata=dict(title="dt sweep"))
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    prof.reset()
    log.info(f"  sim+render {n_total}f...")
    with writer.saving(fig, str(out_dir / fname), dpi=120):
        for i, dt_val in enumerate(scan_values):
            burn_steps = int(C.BURN_IN_MS / dt_val)
            pred = None
            with prof.track_sim():
                if use_dataset:
                    patch_dt(dt_val)
                    T_steps = M.T_steps
                    base_rate = M.max_rate_hz
                    stim_rate = base_rate * overdrive
                    frame_spikes = encode_image_spikes(
                        pixel_vec,
                        T_steps,
                        dt_val,
                        base_rate,
                        stim_rate,
                        C.STEP_ON_MS,
                        C.STEP_OFF_MS,
                        C.SEED,
                    ).to(C.DEVICE)
                    net_frame = make_ping_net(
                        C.cfg, w_in=(*C.W_IN_SPIKES, "normal", C.W_IN_SPARSITY)
                    )
                    with torch.no_grad():
                        logits = net_frame.forward(input_spikes=frame_spikes)
                    if logits is not None:
                        pred = int(logits.argmax(dim=-1)[0].item())
                    rec = _extract_records(net_frame)
                    dt_weights = extract_weights(net_frame)
                    ext_g_raw = frame_spikes.cpu().numpy()
                else:
                    ext_g_sim, ext_g_raw = make_step_drive_from_ref(
                        C.N_E,
                        dt_val,
                        t_e_async,
                        t_e_ping,
                        C.STEP_ON_MS,
                        C.STEP_OFF_MS,
                        C.SIM_MS,
                        X_i,
                        eta_ref,
                        C.SIGMA_E,
                    )
                    rec, _, _w = run_sim(
                        dt_val, t_e_ping, ext_g_override=ext_g_sim, t_e_async=t_e_async
                    )
                spk_e = rec[primary_hid_key(rec)][burn_steps:]
                spk_i = rec[primary_inh_key(rec)][burn_steps:]
                spk_o = rec["out"][burn_steps:] if rec.get("out") is not None else None
                ext_g = ext_g_raw[burn_steps:]

            with prof.track_render():
                draw_kwargs = dict(
                    spk_o=spk_o,
                    weights=dt_weights,
                    sweep_var="dt",
                    sweep_range=(lo, hi),
                    sweep_progress=i / max(1, n_total - 1),
                    t_e_async=t_e_async,
                    sweep_levels=display_values,
                    sweep_frame_idx=i,
                    n_e=C.N_E,
                    n_i=C.N_I,
                    step_on_ms=C.STEP_ON_MS,
                    step_off_ms=C.STEP_OFF_MS,
                    burn_in_ms=C.BURN_IN_MS,
                    w_ie=C.W_IE,
                )
                if use_dataset:
                    draw_kwargs["digit_image"] = digit_image
                    draw_kwargs["sweep_xlabel"] = "dt (ms)"
                    draw_kwargs["sweep_xscale"] = "log"
                draw_transient_frame(
                    axes,
                    overdrive,
                    spk_e,
                    spk_i,
                    ext_g,
                    dt_val,
                    "PING",
                    **draw_kwargs,
                )
            with prof.track_encode():
                fig.savefig(frames_dir / f"frame_{i + 1:04d}.png", dpi=120)
                writer.grab_frame()

            m = metrics_str(spk_e, spk_i, dt_val, n_e=C.N_E, n_i=C.N_I)
            log.info(f"  {i + 1}/{n_total} dt={dt_val:.3f}ms | {m}")

    plt.close(fig)
    log.info(f"  \u2192 {out_dir / fname}")
    log.info(f"  frames \u2192 {frames_dir}/")
    prof.report(n_total)


# =============================================================================
# Snapshot generators
# =============================================================================


def generate_snapshot(
    drive_mult, dt=None, fake_progress=None, model_name="ping", t_e_async=None
):
    """Generate one PNG at a given drive multiplier."""
    import config as C

    if t_e_async is None:
        t_e_async = C.T_E_ASYNC_DEFAULT
    out_dir = Path(C.ARTIFACT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dt is None:
        dt = DT_CAL
    burn_steps = int(C.BURN_IN_MS / dt)
    t_e_ping = t_e_async * drive_mult

    log.info(f"image | {model_name} OD={drive_mult:.1f}x")
    rec, ext_g_raw, weights = run_sim(
        dt, t_e_ping, model_name=model_name, t_e_async=t_e_async
    )
    spk_e = rec[primary_hid_key(rec)][burn_steps:]
    spk_i = rec[primary_inh_key(rec)][burn_steps:] if primary_inh_key(rec) else None
    spk_o = rec.get("out")
    if spk_o is not None:
        spk_o = spk_o[burn_steps:]
    ext_g_vis = ext_g_raw[burn_steps:]

    report_metrics(
        spk_e,
        spk_i,
        dt,
        model_name,
        n_e=C.N_E,
        n_i=C.N_I,
        step_on_ms=C.STEP_ON_MS,
        step_off_ms=C.STEP_OFF_MS,
        burn_in_ms=C.BURN_IN_MS,
    )

    fig, axes = make_transient_fig()
    sweep_kwargs = {}
    if fake_progress is not None:
        sweep_kwargs = dict(
            sweep_var="OD", sweep_range=(1.0, 50.0), sweep_progress=fake_progress
        )
    draw_transient_frame(
        axes,
        drive_mult,
        spk_e,
        spk_i,
        ext_g_vis,
        dt,
        model_name.upper(),
        spk_o=spk_o,
        weights=weights,
        model_name=model_name,
        t_e_async=t_e_async,
        **sweep_kwargs,
        n_e=C.N_E,
        n_i=C.N_I,
        step_on_ms=C.STEP_ON_MS,
        step_off_ms=C.STEP_OFF_MS,
        burn_in_ms=C.BURN_IN_MS,
        w_ie=C.W_IE,
    )
    fname = out_dir / "snapshot.png"
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log.info(f"  \u2192 {fname}")


def generate_spike_snapshot(
    spike_rate=None, overdrive=12.0, dt=None, model_name="ping"
):
    """Generate a snapshot with synthetic spike input."""
    import config as C

    if spike_rate is None:
        spike_rate = C.SPIKE_RATE_BASE
    out_dir = Path(C.ARTIFACT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dt is None:
        dt = DT_CAL

    M.N_HID = C.N_E
    M.N_INH = C.N_I
    patch_dt(dt)
    burn_steps = int(C.BURN_IN_MS / dt)
    T_steps = M.T_steps

    stim_rate = spike_rate * overdrive
    input_spikes = make_spike_drive(
        M.N_IN,
        T_steps,
        dt,
        spike_rate,
        stim_rate,
        C.STEP_ON_MS,
        C.STEP_OFF_MS,
        C.SEED,
    )

    log.info(f"image | {model_name} spikes {spike_rate:.0f}\u2192{stim_rate:.0f}Hz")

    net = make_ping_net(C.cfg, w_in=(*C.W_IN_SPIKES, "normal", C.W_IN_SPARSITY))

    input_spikes = input_spikes.to(C.DEVICE)
    tonic_g = None
    if C.BIAS > 0:
        tonic_g = torch.full(
            (T_steps, C.N_E), C.BIAS, dtype=torch.float32, device=C.DEVICE
        )
    with torch.no_grad():
        net.forward(input_spikes=input_spikes, ext_g=tonic_g)

    rec = _extract_records(net)
    spk_e = rec[primary_hid_key(rec)][burn_steps:]
    spk_i = rec[primary_inh_key(rec)][burn_steps:] if primary_inh_key(rec) else None
    spk_o = rec.get("out")
    if spk_o is not None:
        spk_o = spk_o[burn_steps:]

    weights = extract_weights(net)

    report_metrics(
        spk_e,
        spk_i,
        dt,
        model_name,
        n_e=C.N_E,
        n_i=C.N_I,
        step_on_ms=C.STEP_ON_MS,
        step_off_ms=C.STEP_OFF_MS,
        burn_in_ms=C.BURN_IN_MS,
    )

    fig, axes = make_transient_fig()
    draw_transient_frame(
        axes,
        overdrive,
        spk_e,
        spk_i,
        input_spikes.cpu().numpy()[burn_steps:],
        dt,
        model_name.upper() + " (spikes)",
        spk_o=spk_o,
        weights=weights,
        model_name=model_name,
        t_e_async=spike_rate,
        n_e=C.N_E,
        n_i=C.N_I,
        step_on_ms=C.STEP_ON_MS,
        step_off_ms=C.STEP_OFF_MS,
        burn_in_ms=C.BURN_IN_MS,
        w_ie=C.W_IE,
    )
    fname = out_dir / "snapshot.png"
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log.info(f"  \u2192 {fname}")


def primary_hid_key(rec):
    """Return the recording key for the deepest hidden layer.

    Single-layer models use 'hid'; multi-layer use 'hid_1', 'hid_2', etc.
    Returns the highest-numbered key, or 'hid' for single-layer.
    """
    hid_keys = sorted(k for k in rec if k.startswith("hid"))
    return hid_keys[-1] if hid_keys else "hid"


def primary_inh_key(rec):
    """Return the recording key for the deepest inhibitory layer, or None."""
    inh_keys = sorted(k for k in rec if k.startswith("inh"))
    return inh_keys[-1] if inh_keys else None


from oscilloscope.datasets import (  # noqa: E402,F401
    DATASET_N_HIDDEN_DEFAULTS,
    SHD_N_CHANNELS,
    _load_dataset_image,
    _load_shd,
    _shd_cache_dir,
    load_dataset,
)


def generate_image_snapshot(
    digit_class=0,
    sample_idx=0,
    dataset="scikit",
    dt=None,
    overdrive=1.0,
    model_name="ping",
    out_filename="snapshot.png",
    load_weights=None,
):
    """Generate a snapshot with image input."""
    import config as C

    out_dir = Path(C.ARTIFACT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dt is None:
        dt = DT_CAL

    pixel_vec, digit_image = _load_dataset_image(dataset, digit_class, sample_idx)
    n_in = len(pixel_vec)
    M.N_IN = n_in
    patch_dt(dt)

    od_str = f" OD={overdrive:.1f}x" if overdrive > 1 else ""
    log.info(f"image | dataset d{digit_class}s{sample_idx}{od_str}")

    if overdrive > 1.0:
        # Pre-encode with stimulus window
        base_rate = M.max_rate_hz
        stim_rate = base_rate * overdrive
        input_spikes = encode_image_spikes(
            pixel_vec,
            M.T_steps,
            dt,
            base_rate,
            stim_rate,
            C.STEP_ON_MS,
            C.STEP_OFF_MS,
            C.SEED,
        ).to(C.DEVICE)
        net = make_net(
            C.cfg,
            w_in=(*C.W_IN_SPIKES, "normal", C.W_IN_SPARSITY),
            model_name=model_name,
        )
        if load_weights is not None:
            state = torch.load(load_weights, map_location=C.DEVICE)
            net.load_state_dict(state, strict=False)
            net.eval()
        tonic_g = None
        if C.BIAS > 0:
            tonic_g = torch.full(
                (M.T_steps, C.N_E), C.BIAS, dtype=torch.float32, device=C.DEVICE
            )
        with torch.no_grad():
            net.forward(input_spikes=input_spikes, ext_g=tonic_g)
        rec = _extract_records(net)
        spk_in = input_spikes.cpu().numpy()
        spk_e = rec[primary_hid_key(rec)]
        spk_i = rec.get("inh")
        spk_o = rec.get("out")
        pred = None
    else:
        rec, pred, net = run_sim_image(
            dt, pixel_vec, model_name=model_name, load_weights=load_weights
        )
        spk_in = rec.get("input")
        spk_e = rec[primary_hid_key(rec)]
        spk_i = rec.get("inh")
        spk_o = rec.get("out")

    report_metrics(
        spk_e,
        spk_i,
        dt,
        model_name=model_name,
        n_e=C.N_E,
        n_i=C.N_I,
        step_on_ms=C.STEP_ON_MS,
        step_off_ms=C.STEP_OFF_MS,
        burn_in_ms=C.BURN_IN_MS,
    )

    weights = extract_weights(net)

    if spk_in is not None:
        if isinstance(spk_in, np.ndarray):
            ext_g = spk_in
        elif isinstance(spk_in, list) and len(spk_in) > 0:
            ext_g = np.stack(
                [s.numpy() if isinstance(s, torch.Tensor) else s for s in spk_in]
            )
        else:
            ext_g = np.zeros_like(spk_e)
    else:
        ext_g = np.zeros_like(spk_e)

    fig, axes = make_transient_fig(layout="dataset")
    if pred is not None:
        correct = pred == digit_class
        title = (
            f"digit={digit_class}  pred={pred}  {chr(10003) if correct else chr(10007)}"
        )
    else:
        title = f"digit={digit_class}  OD={overdrive:.1f}x"
    draw_transient_frame(
        axes,
        1.0,
        spk_e,
        spk_i,
        ext_g,
        dt,
        title,
        spk_o=spk_o,
        weights=weights,
        model_name=model_name,
        n_e=C.N_E,
        n_i=C.N_I,
        step_on_ms=C.STEP_ON_MS,
        step_off_ms=C.STEP_OFF_MS,
        burn_in_ms=C.BURN_IN_MS,
        w_ie=C.W_IE,
        digit_image=digit_image,
    )

    fname = out_dir / out_filename
    fig.savefig(fname, dpi=120)
    plt.close(fig)
    log.info(f"  \u2192 {fname}")

    npz_name = Path(out_filename).with_suffix(".npz").name
    npz_path = out_dir / npz_name
    np.savez(
        npz_path,
        spk_e=np.asarray(spk_e),
        spk_i=np.asarray(spk_i) if spk_i is not None else np.empty((0, 0)),
        dt=np.float32(dt),
        n_e=np.int32(C.N_E),
        n_i=np.int32(C.N_I),
    )
    log.info(f"  \u2192 {npz_path}")


def generate_sim_only(
    spike_rate=None,
    overdrive=12.0,
    dt=None,
    model_name="ping",
    input_mode="synthetic-conductance",
    t_e_async=None,
):
    """Run simulation and report metrics, no plot output."""
    import config as C

    if spike_rate is None:
        spike_rate = C.SPIKE_RATE_BASE
    if t_e_async is None:
        t_e_async = C.T_E_ASYNC_DEFAULT
    if dt is None:
        dt = DT_CAL

    M.N_HID = C.N_E
    M.N_INH = C.N_I
    patch_dt(dt)
    burn_steps = int(C.BURN_IN_MS / dt)
    T_steps = M.T_steps

    if input_mode == "synthetic-spikes":
        t_e_ping = t_e_async * overdrive
        stim_rate = spike_rate * overdrive
        input_spikes = make_spike_drive(
            M.N_IN,
            T_steps,
            dt,
            spike_rate,
            stim_rate,
            C.STEP_ON_MS,
            C.STEP_OFF_MS,
            C.SEED,
        )

        log.info(f"sim | {model_name} spikes {spike_rate:.0f}\u2192{stim_rate:.0f}Hz")

        net = make_net(
            C.cfg,
            w_in=(*C.W_IN_SPIKES, "normal", C.W_IN_SPARSITY),
            model_name=model_name,
        )

        input_spikes = input_spikes.to(C.DEVICE)
        tonic_g = None
        if C.BIAS > 0:
            tonic_g = torch.full(
                (T_steps, C.N_E), C.BIAS, dtype=torch.float32, device=C.DEVICE
            )
        with torch.no_grad():
            net.forward(input_spikes=input_spikes, ext_g=tonic_g)

        rec = _extract_records(net)
    else:
        t_e_ping = t_e_async * overdrive
        log.info(f"sim | {model_name} conductance OD={overdrive:.1f}x")
        rec, _, _ = run_sim(dt, t_e_ping, model_name=model_name, t_e_async=t_e_async)

    spk_e = rec[primary_hid_key(rec)][burn_steps:]
    spk_i = rec[primary_inh_key(rec)][burn_steps:] if primary_inh_key(rec) else None
    report_metrics(
        spk_e,
        spk_i,
        dt,
        model_name,
        n_e=C.N_E,
        n_i=C.N_I,
        step_on_ms=C.STEP_ON_MS,
        step_off_ms=C.STEP_OFF_MS,
        burn_in_ms=C.BURN_IN_MS,
    )


# =============================================================================
# Training (merged from train.py)
# =============================================================================

BATCH_SIZE = 64
GRAD_CLIP = 1.0
PATIENCE = 15

# DATASET_N_HIDDEN_DEFAULTS lives in oscilloscope/datasets.py — re-imported above.


def seed_everything(seed):
    """Seed Python, NumPy, and torch RNGs for reproducible runs.

    Seeds cover: Python `random`, NumPy global, torch CPU, torch CUDA
    (all devices), and torch MPS (via torch.manual_seed, which fans out
    to the active backend). Call before dataset load and model init.
    """
    if seed is None:
        return
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.info(f"  seed={seed} (Python, NumPy, torch)")


def observe_epoch(
    net,
    ref_spikes,
    epoch,
    acc,
    train_loss,
    dt,
    model_name,
    fig,
    axes,
    writer,
    burn_in_ms=20.0,
    total_epochs=100,
    grad_ratios=None,
    lr=None,
    digit_image=None,
):
    """Run reference input through network, render oscilloscope frame, grab."""
    import config as C

    C.N_E = M.N_HID
    C.N_I = M.N_INH

    net.recording = True
    with torch.no_grad():
        net(input_spikes=ref_spikes)
    net.recording = False

    rec = net.spike_record
    burn = int(burn_in_ms / dt)

    def _to_np(v):
        if isinstance(v, torch.Tensor):
            return v.cpu().numpy()
        return torch.stack(v).numpy()

    spk_e = _to_np(rec[primary_hid_key(rec)])[burn:]
    _ik = primary_inh_key(rec)
    spk_i = _to_np(rec[_ik])[burn:] if _ik else None
    spk_h1 = _to_np(rec["hid_1"])[burn:] if "hid_1" in rec else None
    spk_o = _to_np(rec["out"])[burn:] if "out" in rec else None
    ext_g = (
        _to_np(rec["input"])[burn:]
        if "input" in rec
        else np.zeros((len(spk_e), spk_e.shape[1]))
    )

    weights = extract_weights(net)

    title = f"Epoch {epoch + 1}  acc={acc:.1f}%  loss={train_loss:.3f}"
    draw_transient_frame(
        axes,
        1.0,
        spk_e,
        spk_i,
        ext_g,
        dt,
        title,
        spk_o=spk_o,
        weights=weights,
        model_name=model_name,
        sweep_frame_idx=epoch,
        n_e=M.N_HID,
        n_i=M.N_INH,
        acc=acc,
        loss=train_loss,
        grad_ratios=grad_ratios,
        lr=lr,
        total_epochs=total_epochs,
        digit_image=digit_image,
        spk_h1=spk_h1,
    )

    if writer is not None:
        writer.grab_frame()
    return compute_metrics(spk_e, spk_i, dt, model_name, n_e=M.N_HID, n_i=M.N_INH)


def train(
    model_name="ping",
    lr=0.01,
    epochs=100,
    dt=0.1,
    observe=False,
    out_dir=None,
    device_name=None,
    w_in=None,
    ei_strength=None,
    ei_ratio=2.0,
    sparsity=0.0,
    w_in_sparsity=0.0,
    dataset="scikit",
    snapshot_init=True,
    snapshot_end=True,
    t_ms=200.0,
    burn_in_ms=20.0,
    hidden_sizes=None,
    max_samples=None,
    v_grad_dampen=80.0,
    early_stopping=None,
    observe_every=1,
    adaptive_lr=False,
    kaiming_init=False,
    dales_law=True,
    w_rec=None,
    rec_layers=None,
    ei_layers=None,
    batch_size=None,
    seed=None,
    readout_w_out_scale=1.0,
    readout_mode="rate",
    fr_reg_lower_theta=0.0,
    fr_reg_lower_strength=0.0,
    fr_reg_upper_theta=0.0,
    fr_reg_upper_strength=0.0,
    skip_bad_grad_threshold=None,
    optimizer="adam",
    loss_mode="ce",
    profile_path=None,
):
    """Train on scikit digits, optionally producing oscilloscope video."""
    import time
    from torch.utils.data import DataLoader, TensorDataset

    # Seed before any RNG use (dataset split, shuffle, model init, Poisson encoding).
    seed_everything(seed)

    # Setup dt and all derived constants
    M.T_ms = t_ms
    patch_dt(dt)

    if hidden_sizes is None:
        default = DATASET_N_HIDDEN_DEFAULTS.get(dataset, 256)
        hidden_sizes = [default]
        log.info(f"  n_hidden auto → {hidden_sizes} (smart default for {dataset})")
    M.N_HID = hidden_sizes[-1]
    M.N_INH = hidden_sizes[-1] // 4
    M.HIDDEN_SIZES = list(hidden_sizes)
    M.V_GRAD_DAMPEN = v_grad_dampen
    if batch_size is not None:
        M.BATCH_SIZE = batch_size

    device = torch.device(device_name) if device_name else _auto_device()

    if out_dir is None:
        out_dir = Path(__file__).parent.parent / "artifacts" / "training" / model_name
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data — single canonical loader; smnist uses mnist data, encoded row-by-row
    use_smnist = dataset == "smnist"
    if dataset == "shd":
        X_tr, X_te, y_tr, y_te = load_dataset(
            dataset, max_samples=max_samples, split=True, dt_ms=dt, t_ms=t_ms
        )
    else:
        X_tr, X_te, y_tr, y_te = load_dataset(
            dataset, max_samples=max_samples, split=True
        )
    if dataset in ("mnist", "smnist"):
        if use_smnist:
            M.N_IN = 28
            t_ms = 28 * 10.0  # 10 ms/row × 28 rows
            M.T_ms = t_ms
            M.T_steps = int(t_ms / dt)
        else:
            M.N_IN = 784
    elif dataset == "shd":
        # X_tr shape is (N, T_steps, 700) — source n_in and T from the data.
        M.N_IN = SHD_N_CHANNELS
        M.T_steps = X_tr.shape[1]
    else:
        M.N_IN = 64
    bs = batch_size if batch_size is not None else BATCH_SIZE
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=bs,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)), batch_size=bs
    )

    # Model — symmetry-break standard-snn (dense W_in needs heterogeneous init phases).
    # Skip randomize_init when Kaiming init is used: Kaiming already gives
    # heterogeneous per-neuron tuning, so scattering mem phases is redundant.
    # Uniform randomize_init across all models — symmetry breaking matters
    # for all architectures. Only skip when kaiming (already heterogeneous).
    randomize = not kaiming_init
    net = build_net(
        model_name,
        w_in=w_in,
        w_in_sparsity=w_in_sparsity,
        ei_strength=ei_strength,
        ei_ratio=ei_ratio,
        sparsity=sparsity,
        device=device,
        randomize_init=randomize,
        kaiming_init=kaiming_init,
        dales_law=dales_law,
        w_rec=w_rec,
        hidden_sizes=hidden_sizes,
        rec_layers=rec_layers,
        ei_layers=ei_layers,
        readout_mode=readout_mode,
    )
    if randomize:
        log.info("  randomize_init=True (symmetry breaking for standard-snn)")
    if kaiming_init:
        log.info("  kaiming_init=True (signed Kaiming weights, canonical snnTorch)")
    if readout_mode != "rate":
        log.info(f"  readout_mode={readout_mode}")
    if readout_w_out_scale != 1.0:
        with torch.no_grad():
            net.W_ff[-1].mul_(readout_w_out_scale)
            if hasattr(net, "b_ff") and len(net.b_ff) > 0:
                net.b_ff[-1].mul_(readout_w_out_scale)
        log.info(
            f"  readout_w_out_scale={readout_w_out_scale:g} "
            f"(W_ff[-1] and b_ff[-1] scaled at init)"
        )
    if w_rec is not None:
        log.info("  recurrent=True (hidden→hidden connections)")
    if not dales_law:
        log.info("  dales_law=False (signed weights, no clamp)")
    n_params = sum(p.numel() for p in net.parameters())
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)

    log.info(
        f"train | {model_name} N={M.N_HID} dt={dt}ms T={M.T_ms}ms | {n_trainable:,} params"
    )
    log.info(f"  data: {len(X_tr)} train {len(X_te)} test")

    # Save config for reproducibility
    import json
    import run_log

    config = {
        "model": model_name,
        "lr": lr,
        "epochs": epochs,
        "dt": dt,
        "t_ms": M.T_ms,
        "dataset": dataset,
        "n_hidden": M.N_HID,
        "n_inh": M.N_INH,
        "n_in": M.N_IN,
        "w_in": list(w_in) if w_in else None,
        "ei_strength": ei_strength,
        "ei_ratio": ei_ratio,
        "sparsity": sparsity,
        "w_in_sparsity": w_in_sparsity,
        "input_rate": M.max_rate_hz,
        "v_grad_dampen": v_grad_dampen,
        "burn_in_ms": burn_in_ms,
        "batch_size": bs,
        "grad_clip": GRAD_CLIP,
        "early_stopping": early_stopping,
        "max_samples": max_samples,
        "n_params": n_params,
        "n_trainable": n_trainable,
        "kaiming_init": kaiming_init,
        "dales_law": dales_law,
        "readout_mode": readout_mode,
        "loss_mode": loss_mode,
        "hidden_sizes": hidden_sizes,
        "w_rec": w_rec,
        "rec_layers": list(rec_layers) if rec_layers else None,
        "ei_layers": list(ei_layers) if ei_layers else None,
        "seed": seed,
        # Provenance (git SHA, run_id, started_at, device, torch version,
        # python env hash) — keeps train-mode config.json at parity with
        # sim/image/video modes.
        **run_log.provenance(),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(out_dir / "run.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(" ".join(sys.argv) + "\n")
    log.info(f"  config \u2192 {out_dir / 'config.json'}")

    # Train uses the module-level log (already set up by save_run_artifacts)

    # Pre-generate fixed reference spike train using the canonical loader
    # (same as image mode) so init/end snapshots use the same digit-0 sample
    # regardless of train/test split shuffling.
    if dataset == "shd":
        # SHD has no canonical "digit image" — skip the image snapshot PNGs
        # (the input-raster snapshot panel is a deferred follow-up). Per-epoch
        # firing-rate metrics still want a fixed reference input, so use the
        # first train sample — already-spiked, shape (T, 700), ready for forward.
        snapshot_init = False
        snapshot_end = False
        ref_image = None
        ref_spikes = torch.from_numpy(X_tr[0]).float().to(device)
    else:
        loader_dataset = "mnist" if dataset in ("mnist", "smnist") else dataset
        ref_pixel_vec, ref_image = _load_dataset_image(
            loader_dataset, digit_class=0, sample_idx=0
        )
        ref_input = torch.from_numpy(ref_pixel_vec).float().unsqueeze(0).to(device)
        if use_smnist:
            ref_spikes = encode_smnist(ref_input, dt, M.max_rate_hz).to(device)
        else:
            torch.manual_seed(0)
            pixels = ref_input.clamp(0, 1)
            p = M.max_rate_hz * dt / 1000.0
            ref_spikes = (
                (torch.rand(M.T_steps, 1, M.N_IN, device=device) < pixels * p)
                .float()
                .squeeze(1)
            )

    if snapshot_init:
        import config as C

        C.N_E = M.N_HID
        C.N_I = M.N_INH
        net.recording = True
        with torch.no_grad():
            net(input_spikes=ref_spikes)
        net.recording = False
        rec = net.spike_record
        burn = int(burn_in_ms / dt)

        def _to_np(v):
            return (
                v.cpu().numpy()
                if isinstance(v, torch.Tensor)
                else torch.stack(v).numpy()
            )

        spk_e = _to_np(rec[primary_hid_key(rec)])[burn:]
        _ik = primary_inh_key(rec)
        spk_i = _to_np(rec[_ik])[burn:] if _ik else None
        spk_h1 = _to_np(rec["hid_1"])[burn:] if "hid_1" in rec else None
        spk_o = _to_np(rec["out"])[burn:] if "out" in rec else None
        ext_g = (
            _to_np(rec["input"])[burn:]
            if "input" in rec
            else np.zeros((len(spk_e), spk_e.shape[1]))
        )
        weights = extract_weights(net)

        snapshot_init_state = compute_metrics(
            spk_e, spk_i, dt, model_name, n_e=M.N_HID, n_i=M.N_INH
        )
        print("Init state (d0s0):")
        log.info(f"  {format_metrics(snapshot_init_state)}")

        if not observe:
            fig, axes = make_transient_fig(layout="train")
            draw_transient_frame(
                axes,
                1.0,
                spk_e,
                spk_i,
                ext_g,
                dt,
                f"Init  {model_name}",
                spk_o=spk_o,
                weights=weights,
                model_name=model_name,
                n_e=M.N_HID,
                n_i=M.N_INH,
                digit_image=ref_image,
                spk_h1=spk_h1,
            )
            fname = out_dir / "init_d0s0.png"
            fig.savefig(fname, dpi=120)
            plt.close(fig)
            log.info(f"  \u2192 {fname}")
    else:
        snapshot_init_state = None

    if epochs == 0:
        log.info(f"  epochs=0, use --epochs N to train")
        # Even in probe mode, write a minimal metrics.json so callers can
        # inspect init state without parsing logs.
        from datetime import datetime, timezone

        metrics_blob = {
            "mode": "train",
            "model": model_name,
            "run_finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": {
                "dt": dt,
                "t_ms": M.T_ms,
                "epochs": 0,
                "lr": lr,
                "input_rate": M.max_rate_hz,
                "w_in": list(w_in) if w_in else None,
                "w_in_sparsity": w_in_sparsity,
                "ei_strength": ei_strength,
                "ei_ratio": ei_ratio,
                "sparsity": sparsity,
                "n_hidden": M.N_HID,
                "n_inh": M.N_INH,
                "n_in": M.N_IN,
                "max_samples": max_samples,
                "dataset": dataset,
                "v_grad_dampen": v_grad_dampen,
                "adaptive_lr": adaptive_lr,
                "burn_in_ms": burn_in_ms,
                "n_params": n_params,
                "n_trainable": n_trainable,
            },
            "init": snapshot_init_state,
            "epochs": [],
            "end": None,
            "best_acc": 0.0,
            "best_epoch": 0,
            "total_elapsed_s": 0.0,
        }
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics_blob, f, indent=2, default=float)
        log.info(f"  done (probe only) \u2192 {out_dir}")
        return 0.0

    log.info(f"  lr={lr} epochs={epochs} batch={bs}")
    if observe:
        log.info(f"  observe → {out_dir / 'frames/'}")

    # Optimizer. Adamax uses the L∞ norm for the second moment instead of
    # the EMA of squared grads, so a single pathological batch can't poison
    # the preconditioner the way it does in Adam — the canonical SNN choice
    # (Cramer, Zenke Spytorch).
    if optimizer == "adamax":
        opt = torch.optim.Adamax(net.parameters(), lr=lr)
    elif optimizer == "adam":
        opt = torch.optim.Adam(net.parameters(), lr=lr)
    else:
        raise ValueError(f"unknown optimizer: {optimizer!r}")
    scheduler = None
    if adaptive_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=5, min_lr=1e-5
        )
    if loss_mode == "ce":
        _ce = torch.nn.CrossEntropyLoss()

        def loss_fn(logits, y):
            return _ce(logits, y)
    elif loss_mode == "mse":
        # L2 loss on logits vs one-hot targets — standard SNN-paper
        # alternative to CE (e.g. Bohte 2002, Lee 2016). No softmax;
        # the network is trained to push the correct-class logit toward
        # 1 and others toward 0. Useful when CE saturates because of
        # the readout's logit scale.
        _mse = torch.nn.MSELoss()

        def loss_fn(logits, y):
            target = torch.nn.functional.one_hot(
                y, num_classes=logits.shape[-1]
            ).float()
            return _mse(logits, target)
    else:
        raise ValueError(f"unknown loss_mode {loss_mode!r}")
    log.info(f"  loss_mode={loss_mode}")

    # Observe setup
    obs_fig = obs_axes = None
    obs_frames_dir = None
    if observe:
        reset_weight_xlims()
        obs_fig, obs_axes = make_transient_fig(layout="train")
        obs_frames_dir = out_dir / "frames"
        obs_frames_dir.mkdir(parents=True, exist_ok=True)
        assert obs_fig is not None and obs_frames_dir is not None

    # Epoch 0 frame (init state)
    init_state = None
    if observe:
        assert obs_fig is not None and obs_frames_dir is not None
        init_state = observe_epoch(
            net,
            ref_spikes,
            -1,
            0.0,
            0.0,
            dt,
            model_name,
            obs_fig,
            obs_axes,
            None,
            burn_in_ms=burn_in_ms,
            total_epochs=epochs,
            digit_image=ref_image,
        )
        if init_state:
            log.info(f"  ep 0/{epochs} init | {format_metrics(init_state)}")
        obs_fig.savefig(obs_frames_dir / "epoch_000.png", dpi=120)

    # Training loop
    import run_log

    best_acc = 0.0
    best_state = None
    no_improve = 0
    t_start = _time.perf_counter()
    prev_lr = lr
    epoch_records = []  # accumulated for metrics.json
    best_epoch = 0
    wtracker = run_log.WarningTracker()
    jsonl = run_log.MetricsJsonl(out_dir / "metrics.jsonl")
    run_log.print_progress_header(log)
    # MPS has no max_memory_allocated — sample per epoch and keep the peak.
    # CUDA tracks peak natively (queried at end of run).
    peak_mem_mps = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(epochs):
        t_epoch = _time.perf_counter()

        # Train
        net.train()
        total_loss = 0.0
        n_batches = 0
        grad_sum = 0.0
        n_grad = 0
        n_skipped_steps = 0
        layer_ratio_sum = {}
        n_samples_train = 0
        t_train_compute = _time.perf_counter()
        for batch_idx, (X_b, y_b) in enumerate(train_loader):
            # Wrap the 3rd batch of epoch 0 in torch.profiler when --profile
            # is set. Skip first two for allocator/JIT warmup. Trace overhead
            # only hits this one batch; the rest of training is unprofiled.
            # Manual __enter__/__exit__ avoids re-indenting the batch body.
            do_profile = profile_path is not None and epoch == 0 and batch_idx == 2
            _prof_handle = None
            if do_profile:
                from torch.profiler import profile as _prof, ProfilerActivity

                _activities = [ProfilerActivity.CPU]
                if device.type == "cuda":
                    _activities.append(ProfilerActivity.CUDA)
                elif device.type == "mps" and hasattr(ProfilerActivity, "MPS"):
                    _activities.append(ProfilerActivity.MPS)
                _prof_handle = _prof(
                    activities=_activities, record_shapes=True, with_stack=True
                )
                _prof_handle.__enter__()
            X_b, y_b = X_b.to(device), y_b.to(device)
            spk = encode_batch(X_b, dt, use_smnist)
            logits = net(input_spikes=spk)
            if torch.isnan(logits).any():
                opt.zero_grad()
                continue
            loss = loss_fn(logits, y_b)
            # Firing-rate regularisation (Cramer et al. 2022): penalise
            # per-neuron trial spike counts outside [θ_l, θ_u]. Off by default
            # (all strengths 0). Uses CUBANet.last_spike_counts when present.
            if (fr_reg_lower_strength > 0 or fr_reg_upper_strength > 0) and getattr(
                net, "last_spike_counts", None
            ) is not None:
                reg = 0.0
                for sc in net.last_spike_counts:
                    mean_z = sc.mean(dim=0)  # (n_hidden,)
                    if fr_reg_lower_strength > 0:
                        reg = (
                            reg
                            + fr_reg_lower_strength
                            * (torch.relu(fr_reg_lower_theta - mean_z) ** 2).sum()
                        )
                    if fr_reg_upper_strength > 0:
                        reg = (
                            reg
                            + fr_reg_upper_strength
                            * (torch.relu(mean_z - fr_reg_upper_theta) ** 2).sum()
                        )
                loss = loss + reg
            opt.zero_grad()
            loss.backward()
            # Per-layer ‖grad‖/‖W‖ ratio (weights only, skip biases) before clip.
            # CUBANet names biases b_ff.*; SNNTorchLibraryNet exposes biases
            # via nn.Linear as fc_ff.*.bias. Filter both naming conventions.
            for pname, p in net.named_parameters():
                if p.grad is None or pname.startswith("b_") or pname.endswith(".bias"):
                    continue
                wn = p.norm().item()
                if wn > 0:
                    layer_ratio_sum[pname] = (
                        layer_ratio_sum.get(pname, 0.0) + p.grad.norm().item() / wn
                    )
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            gn_f = float(gn)
            bad_grad = (not math.isfinite(gn_f)) or (
                skip_bad_grad_threshold is not None and gn_f > skip_bad_grad_threshold
            )
            if bad_grad:
                # Skip the step: a single exploded batch would otherwise
                # poison Adam's second-moment estimate and kill the run.
                opt.zero_grad(set_to_none=True)
                n_skipped_steps += 1
            else:
                opt.step()
                # Project trainable weights back onto the Dale's-law cone
                # (no-op when signed weights are allowed). Keeps the stored
                # parameter values consistent with what forward() uses.
                net.project_dales()
                total_loss += loss.item()
                n_batches += 1
                grad_sum += gn_f
                n_grad += 1
            n_samples_train += y_b.size(0)
            if _prof_handle is not None:
                _prof_handle.__exit__(None, None, None)
                assert profile_path is not None
                from pathlib import Path as _Path

                _Path(profile_path).parent.mkdir(parents=True, exist_ok=True)
                _prof_handle.export_chrome_trace(profile_path)
                log.info(f"  → profiler trace {profile_path}")
                # Top-15 ops by self CPU time (or self device time on cuda).
                _sort_key = (
                    "self_cuda_time_total"
                    if device.type == "cuda"
                    else "self_cpu_time_total"
                )
                log.info(f"  profiler top-15 ops by {_sort_key}:")
                log.info(
                    _prof_handle.key_averages().table(sort_by=_sort_key, row_limit=15)
                )
                # Per-call-site breakdown for the high-frequency ops we
                # actually want to optimise. group_by_stack_n=8 gives a
                # short-enough stack to read at a glance.
                # Dump first few stacks per hot op to a debug file so we can
                # see which Python lines are dispatching the kernel storm.
                _dbg_path = profile_path + ".stacks.txt"
                _hot_ops = (
                    "aten::_local_scalar_dense",
                    "aten::copy_",
                    "aten::where",
                    "aten::eq",
                )
                with open(_dbg_path, "w") as _dbg:
                    for _op in _hot_ops:
                        _dbg.write(f"\n=== {_op} ===\n")
                        _seen = 0
                        for _e in _prof_handle.events():
                            if _e.name != _op:
                                continue
                            if _seen >= 3:
                                break
                            _dbg.write(f"--- event {_seen} ---\n")
                            for _f in _e.stack or []:
                                _dbg.write(f"  {_f}\n")
                            if not _e.stack:
                                _dbg.write("  (no stack)\n")
                            _seen += 1
                log.info(f"  → stack debug {_dbg_path}")
                _prof_handle = None
        train_compute_s = _time.perf_counter() - t_train_compute
        if device.type == "mps":
            peak_mem_mps = max(peak_mem_mps, torch.mps.current_allocated_memory())
        avg_grad = grad_sum / max(n_grad, 1)
        grad_ratios = {n: s / max(n_grad, 1) for n, s in layer_ratio_sum.items()}

        # Eval — deterministic Poisson encoding so this matches infer()
        t_eval = _time.perf_counter()
        net.eval()
        correct = total = 0
        test_loss_sum = 0.0
        test_batches = 0
        eval_gen = torch.Generator().manual_seed(EVAL_SEED)
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                spk = encode_batch(X_b, dt, use_smnist, generator=eval_gen)
                logits_t = net(input_spikes=spk)
                test_loss_sum += loss_fn(logits_t, y_b).item()
                test_batches += 1
                correct += (logits_t.argmax(1) == y_b).sum().item()
                total += y_b.size(0)

        eval_s = _time.perf_counter() - t_eval

        acc = 100.0 * correct / total
        avg_train = total_loss / max(n_batches, 1)
        avg_test = test_loss_sum / max(test_batches, 1)

        new_best = acc > best_acc
        if new_best:
            best_acc = acc
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if scheduler is not None:
            scheduler.step(acc)
        cur_lr = opt.param_groups[0]["lr"]
        elapsed = _time.perf_counter() - t_epoch

        if cur_lr != prev_lr:
            log.info(f"  ⚡ lr → {cur_lr:.0e}")
            prev_lr = cur_lr

        # Oscilloscope frame (returns metrics dict; merged into epoch record)
        t_observe = _time.perf_counter()
        epoch_metrics = None
        if observe and (epoch + 1) % observe_every == 0:
            assert obs_fig is not None and obs_frames_dir is not None
            epoch_metrics = observe_epoch(
                net,
                ref_spikes,
                epoch,
                acc,
                avg_train,
                dt,
                model_name,
                obs_fig,
                obs_axes,
                None,
                burn_in_ms=burn_in_ms,
                total_epochs=epochs,
                grad_ratios=grad_ratios,
                lr=cur_lr,
                digit_image=ref_image,
            )
            obs_fig.savefig(obs_frames_dir / f"epoch_{epoch + 1:03d}.png", dpi=120)
        else:
            # No --observe, but still capture firing-rate metrics for the JSON.
            # Snapshot the RNG so the reference forward pass (which may use
            # randomize_init) does not perturb the training trajectory.
            rng_state = torch.get_rng_state()
            net.recording = True
            with torch.no_grad():
                net(input_spikes=ref_spikes)
            net.recording = False
            torch.set_rng_state(rng_state)
            burn = int(burn_in_ms / dt)

            def _to_np(v):
                return (
                    v.cpu().numpy()
                    if isinstance(v, torch.Tensor)
                    else torch.stack(v).numpy()
                )

            _rec = net.spike_record
            spk_e = _to_np(_rec[primary_hid_key(_rec)])[burn:]
            _ik = primary_inh_key(_rec)
            spk_i = _to_np(_rec[_ik])[burn:] if _ik else None
            epoch_metrics = compute_metrics(
                spk_e, spk_i, dt, model_name, n_e=M.N_HID, n_i=M.N_INH
            )
        observe_s = _time.perf_counter() - t_observe

        # Record this epoch into the structured metrics history
        record = {
            "ep": epoch + 1,
            "acc": acc,
            "loss": avg_train,
            "test_loss": avg_test,
            "lr": cur_lr,
            "elapsed_s": elapsed,
            "train_compute_s": train_compute_s,
            "eval_s": eval_s,
            "observe_s": observe_s,
            "samples": n_samples_train,
            "grad_norm": avg_grad,
            "grad_ratios": grad_ratios,
            "skipped_steps": n_skipped_steps,
            "new_best": new_best,
        }
        if epoch_metrics:
            record.update(epoch_metrics)
        epoch_records.append(record)

        # jsonl sidecar — one line per epoch
        jsonl.write(**{k: v for k, v in record.items() if k != "grad_ratios"})

        # Structured progress line + warning tracker
        e_rate = epoch_metrics.get("rate_e", 0.0) if epoch_metrics else 0.0
        i_rate = epoch_metrics.get("rate_i") if epoch_metrics else None
        cv = epoch_metrics.get("cv", 0.0) if epoch_metrics else 0.0
        activity = epoch_metrics.get("act", 0.0) * 100 if epoch_metrics else 0.0
        flags = wtracker.tick(epoch + 1, acc, activity, avg_train)
        eta = (epochs - epoch - 1) * elapsed
        run_log.print_epoch(
            log,
            epoch + 1,
            epochs,
            acc,
            avg_train,
            e_rate,
            i_rate,
            cv,
            activity,
            elapsed,
            eta,
            new_best=new_best,
            warnings=flags,
        )

        if early_stopping is not None and no_improve >= early_stopping:
            log.info(f"  Early stopping: no improvement for {early_stopping} epochs")
            break

    total_time = _time.perf_counter() - t_start

    # Tier 1 perf block — see nb000 for the why. Per-epoch breakdown is in
    # epoch_records (train_compute_s / eval_s / observe_s); these are the
    # whole-run aggregates the dashboards read.
    perf_block: dict = {
        "device": {"type": device.type},
        "torch_version": torch.__version__,
    }
    try:
        if device.type == "cuda":
            perf_block["device"]["name"] = torch.cuda.get_device_name(device)
            perf_block["peak_memory_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
        elif device.type == "mps" and peak_mem_mps > 0:
            # current_allocated_memory sampled per-epoch (max). Reflects active
            # tensor allocation, not the MPS driver's cached pool.
            perf_block["peak_memory_bytes"] = int(peak_mem_mps)
    except Exception:
        pass
    if epoch_records:
        perf_block["epoch1_total_s"] = float(epoch_records[0]["elapsed_s"])
        warm = epoch_records[1:] if len(epoch_records) > 1 else epoch_records
        warm_total_s = sum(r["elapsed_s"] for r in warm)
        warm_compute_s = sum(r.get("train_compute_s", 0.0) for r in warm)
        warm_samples = sum(r.get("samples", 0) for r in warm)
        perf_block["epoch_warm_mean_s"] = float(warm_total_s / max(len(warm), 1))
        if warm_compute_s > 0:
            perf_block["samples_per_sec_warm"] = float(warm_samples / warm_compute_s)

    # Snapshot end state
    end_state = None
    if snapshot_end:
        import config as C

        C.N_E = M.N_HID
        C.N_I = M.N_INH
        net.recording = True
        with torch.no_grad():
            net(input_spikes=ref_spikes)
        net.recording = False
        rec = net.spike_record
        burn = int(burn_in_ms / dt)

        def _to_np(v):
            return (
                v.cpu().numpy()
                if isinstance(v, torch.Tensor)
                else torch.stack(v).numpy()
            )

        spk_e = _to_np(rec[primary_hid_key(rec)])[burn:]
        _ik = primary_inh_key(rec)
        spk_i = _to_np(rec[_ik])[burn:] if _ik else None
        spk_h1 = _to_np(rec["hid_1"])[burn:] if "hid_1" in rec else None
        spk_o = _to_np(rec["out"])[burn:] if "out" in rec else None
        ext_g = (
            _to_np(rec["input"])[burn:]
            if "input" in rec
            else np.zeros((len(spk_e), spk_e.shape[1]))
        )
        weights = extract_weights(net)
        end_state = compute_metrics(
            spk_e, spk_i, dt, model_name, n_e=M.N_HID, n_i=M.N_INH
        )
        print("End state (d0s0):")
        log.info(f"  {format_metrics(end_state)}")
        if not observe:
            fig, axes = make_transient_fig(layout="train")
            draw_transient_frame(
                axes,
                1.0,
                spk_e,
                spk_i,
                ext_g,
                dt,
                f"End  {model_name}  acc={best_acc:.1f}%",
                spk_o=spk_o,
                weights=weights,
                model_name=model_name,
                n_e=M.N_HID,
                n_i=M.N_INH,
                digit_image=ref_image,
                spk_h1=spk_h1,
            )
            fname = out_dir / "end_d0s0.png"
            fig.savefig(fname, dpi=120)
            plt.close(fig)
            log.info(f"  \u2192 {fname}")

    # Save weights
    if best_state is not None:
        torch.save(best_state, out_dir / "weights.pth")

    # Write structured metrics for tests/analysis (parallels output.log).
    # run_finished_at is the canonical "when did this run actually produce
    # its numbers" timestamp — consumers should prefer it over file mtime,
    # which gets clobbered by git checkout, file copies, etc.
    from datetime import datetime, timezone

    metrics_path = out_dir / "metrics.json"
    metrics_blob = {
        "mode": "train",
        "model": model_name,
        "run_finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "dt": dt,
            "t_ms": M.T_ms,
            "epochs": epochs,
            "lr": lr,
            "input_rate": M.max_rate_hz,
            "w_in": list(w_in) if w_in else None,
            "w_in_sparsity": w_in_sparsity,
            "ei_strength": ei_strength,
            "ei_ratio": ei_ratio,
            "sparsity": sparsity,
            "n_hidden": M.N_HID,
            "n_inh": M.N_INH,
            "n_in": M.N_IN,
            "max_samples": max_samples,
            "dataset": dataset,
            "v_grad_dampen": v_grad_dampen,
            "adaptive_lr": adaptive_lr,
            "burn_in_ms": burn_in_ms,
            "batch_size": bs,
            "grad_clip": GRAD_CLIP,
            "early_stopping": early_stopping,
            "n_params": n_params,
            "n_trainable": n_trainable,
        },
        "init": snapshot_init_state if snapshot_init_state else init_state,
        "epochs": epoch_records,
        "end": end_state,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "total_elapsed_s": total_time,
        "perf": perf_block,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_blob, f, indent=2, default=float)
    log.info(f"  \u2192 {metrics_path}")

    # Finalize: assemble mp4 from frame PNGs
    if observe:
        log.info(f"  frames → {obs_frames_dir}/")
        if observe == "video":
            import subprocess as _sp

            assert obs_frames_dir is not None
            mp4_path = out_dir / "training.mp4"
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                "10",
                "-pattern_type",
                "glob",
                "-i",
                str(obs_frames_dir / "epoch_*.png"),
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(mp4_path),
            ]
            result = _sp.run(ffmpeg_cmd, capture_output=True)
            if result.returncode == 0:
                log.info(f"  → {mp4_path}")
            else:
                log.info(f"  ffmpeg failed (frames kept in {obs_frames_dir}/)")
        plt.close(obs_fig)

    jsonl.close()

    # test_predictions.json — per-sample inference on test set using best state
    if best_state is not None:
        net.load_state_dict(best_state, strict=False)
    net.eval()
    preds = []
    idx = 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            spk = encode_batch(X_b, dt, use_smnist)
            logits_t = net(input_spikes=spk)
            p = logits_t.argmax(1)
            for i in range(y_b.size(0)):
                preds.append(
                    {
                        "idx": idx,
                        "true": int(y_b[i].item()),
                        "pred": int(p[i].item()),
                        "correct": bool(p[i].item() == y_b[i].item()),
                        "logits": [float(x) for x in logits_t[i].tolist()],
                    }
                )
                idx += 1
    run_log.write_test_predictions(out_dir / "test_predictions.json", preds)

    # Structured summary block
    dyn = None
    if end_state:
        dyn = {
            "E rate": f"{end_state.get('rate_e', 0):.0f} Hz",
            "I rate": (
                f"{end_state.get('rate_i', 0):.0f} Hz"
                if end_state.get("rate_i") not in (None, 0.0)
                else "—"
            ),
            "CV": f"{end_state.get('cv', 0):.2f}",
            "activity": f"{end_state.get('act', 0) * 100:.0f}%",
        }
        if end_state.get("f0"):
            dyn["f0"] = f"{end_state['f0']:.0f} Hz"
    run_log.print_summary(
        log,
        best_acc=best_acc,
        final_acc=acc,
        best_epoch=best_epoch,
        runtime_s=total_time,
        dynamics=dyn,
        out_dir=out_dir,
        warnings=wtracker.summary_lines(),
    )
    return best_acc


# =============================================================================
# CLI
# =============================================================================


def _plot_dt_sweep(sweep_results, train_dt, model_name, out_dir):
    """Plot accuracy vs dt with the training dt marked."""
    dts = [r["dt"] for r in sweep_results]
    accs = [r["acc"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(dts, accs, "o-", color="#2a2a2a", linewidth=1.5, markersize=6)
    if train_dt in dts:
        ref_acc = accs[dts.index(train_dt)]
        ax.axvline(
            train_dt,
            color="#cc4444",
            linestyle="--",
            linewidth=1,
            label=f"train dt={train_dt}",
        )
        ax.plot(train_dt, ref_acc, "s", color="#cc4444", markersize=10, zorder=5)
    ax.set_xlabel("dt (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"dt inference stability — {model_name}")
    ax.legend(loc="lower left")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "dt_sweep.png", dpi=150)
    plt.close(fig)


def _render_dt_sweep_video(
    net,
    dt_values,
    sweep_results,
    train_dt,
    model_name,
    dataset,
    out_dir,
    burn_in_ms=20.0,
    frozen_inputs=False,
):
    """Render an oscilloscope video with one frame per dt value."""
    import config as C

    C.N_E = M.N_HID
    C.N_I = M.N_INH

    loader_dataset = "mnist" if dataset in ("mnist", "smnist") else dataset
    ref_pixel_vec, ref_image = _load_dataset_image(loader_dataset, 0, 0)
    ref_input = torch.from_numpy(ref_pixel_vec).unsqueeze(0)
    use_smnist = dataset == "smnist"

    frozen_ref = None
    if frozen_inputs:
        dt_ref = dt_values[0]
        g = torch.Generator()
        g.manual_seed(99)
        if use_smnist:
            frozen_ref = encode_smnist(ref_input, dt_ref, M.max_rate_hz, generator=g)
        else:
            T_ref = int(M.T_ms / dt_ref)
            frozen_ref = encode_images_poisson(
                ref_input, T_ref, dt_ref, M.max_rate_hz, generator=g
            )

    reset_weight_xlims()
    fig, axes = make_transient_fig(layout="train")
    writer = FFMpegWriter(fps=2, metadata=dict(title=f"dt sweep — {model_name}"))
    writer.setup(fig, str(out_dir / "dt_sweep.mp4"), dpi=120)

    for i, sweep_dt in enumerate(dt_values):
        patch_dt(sweep_dt)
        if frozen_ref is not None:
            spk = transport_spikes_bin(frozen_ref, dt_values[0], sweep_dt)
        else:
            spk = encode_batch(ref_input, sweep_dt, use_smnist)

        net.recording = True
        with torch.no_grad():
            net(input_spikes=spk)
        net.recording = False

        rec = net.spike_record
        burn = int(burn_in_ms / sweep_dt)

        def _to_np(v):
            if isinstance(v, torch.Tensor):
                return v.cpu().numpy()
            return torch.stack(v).numpy()

        spk_e = _to_np(rec[primary_hid_key(rec)])[burn:]
        _ik = primary_inh_key(rec)
        spk_i = _to_np(rec[_ik])[burn:] if _ik else None
        spk_h1 = _to_np(rec["hid_1"])[burn:] if "hid_1" in rec else None
        spk_o = _to_np(rec["out"])[burn:] if "out" in rec else None
        ext_g = (
            _to_np(rec["input"])[burn:]
            if "input" in rec
            else np.zeros((len(spk_e), spk_e.shape[1]))
        )

        acc = sweep_results[i]["acc"]
        marker = " ◄train" if sweep_dt == train_dt else ""
        title = f"dt={sweep_dt:.3f}ms  acc={acc:.1f}%{marker}"

        draw_transient_frame(
            axes,
            1.0,
            spk_e,
            spk_i,
            ext_g,
            sweep_dt,
            title,
            spk_o=spk_o,
            weights=extract_weights(net),
            model_name=model_name,
            sweep_frame_idx=i,
            n_e=M.N_HID,
            n_i=M.N_INH,
            acc=acc,
            digit_image=ref_image,
            total_epochs=len(dt_values),
            spk_h1=spk_h1,
        )
        writer.grab_frame()

    writer.finish()
    plt.close(fig)


def infer(
    model_name="ping",
    dt=0.25,
    load_weights=None,
    dataset="scikit",
    max_samples=None,
    t_ms=200.0,
    w_in=None,
    ei_strength=0.5,
    ei_ratio=2.0,
    sparsity=0.0,
    w_in_sparsity=0.0,
    hidden_sizes=None,
    out_dir=None,
    kaiming_init=False,
    dales_law=True,
    encode_fn=None,
    w_rec=None,
    rec_layers=None,
    ei_layers=None,
    seed=None,
):
    """Run inference with saved weights at a given dt."""
    import config as C

    # Seed before dataset load and model init (matters when load_weights is None
    # and any randomness remains in the path).
    seed_everything(seed)

    # Setup dt and all derived constants
    M.T_ms = t_ms
    patch_dt(dt)

    if hidden_sizes is None:
        default = DATASET_N_HIDDEN_DEFAULTS.get(dataset, 256)
        hidden_sizes = [default]
        log.info(f"  n_hidden auto → {hidden_sizes} (smart default for {dataset})")
    M.N_HID = hidden_sizes[-1]
    M.N_INH = hidden_sizes[-1] // 4
    M.HIDDEN_SIZES = list(hidden_sizes)

    device = _auto_device()

    # Data — same canonical loader and split as train, so the test set is
    # the same physical samples the training never saw
    _, X_te, _, y_te = load_dataset(dataset, max_samples=max_samples, split=True)
    if dataset in ("mnist", "smnist"):
        if dataset == "smnist":
            M.N_IN = 28
            M.T_ms = 28 * 10.0
            M.T_steps = int(M.T_ms / dt)
        else:
            M.N_IN = 784
    else:
        M.N_IN = 64

    from torch.utils.data import DataLoader, TensorDataset

    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)), batch_size=64
    )

    # Build model — same builder as train, same args produce same network
    # Uniform randomize_init across all models — symmetry breaking matters
    # for all architectures. Only skip when kaiming (already heterogeneous).
    randomize = not kaiming_init
    net = build_net(
        model_name,
        w_in=w_in,
        w_in_sparsity=w_in_sparsity,
        ei_strength=ei_strength,
        ei_ratio=ei_ratio,
        sparsity=sparsity,
        device=device,
        randomize_init=randomize,
        kaiming_init=kaiming_init,
        dales_law=dales_law,
        w_rec=w_rec,
        hidden_sizes=hidden_sizes,
        rec_layers=rec_layers,
        ei_layers=ei_layers,
    )

    # Load weights
    assert load_weights is not None
    state = torch.load(load_weights, map_location=device)
    net.load_state_dict(state, strict=False)
    log.info(f"  loaded {load_weights}")

    # Evaluate — pre-encode pixels as Poisson spikes (same path as train)
    use_smnist = dataset == "smnist"
    _encode = encode_fn if encode_fn is not None else encode_batch
    net.eval()
    correct = total = 0
    eval_gen = torch.Generator().manual_seed(EVAL_SEED)
    # Accumulate per-population firing rate (Hz) across batches, weighted by
    # batch size. net.rates is set by _set_meta after each forward as Hz
    # per population averaged over the batch.
    rate_sums: dict[str, float] = {}
    with torch.no_grad():
        for X_b, y_b in test_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            spk = _encode(X_b, dt, use_smnist, generator=eval_gen)
            logits = net(input_spikes=spk)
            correct += (logits.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
            batch_rates = getattr(net, "rates", None) or {}
            B = y_b.size(0)
            for k, v in batch_rates.items():
                rate_sums[k] = rate_sums.get(k, 0.0) + float(v) * B

    acc = 100.0 * correct / total
    rates_hz = {k: v / total for k, v in rate_sums.items()} if total else {}
    hid_key = max((k for k in rates_hz if k.startswith("hid")), default=None)
    hid_rate_hz = rates_hz.get(hid_key) if hid_key else None
    rate_str = f"  hid={hid_rate_hz:.1f}Hz" if hid_rate_hz is not None else ""
    log.info(f"  dt={dt:.3f}ms  acc={acc:.1f}%  ({correct}/{total}){rate_str}")

    # Structured metrics artifact for tests/analysis (parallels train's metrics.json)
    out_dir_path = Path(out_dir) if out_dir else None
    if out_dir_path and out_dir_path.exists():
        from datetime import datetime, timezone

        metrics_blob = {
            "mode": "infer",
            "model": model_name,
            "run_finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": {
                "dt": dt,
                "t_ms": M.T_ms,
                "w_in": list(w_in) if w_in else None,
                "w_in_sparsity": w_in_sparsity,
                "ei_strength": ei_strength,
                "ei_ratio": ei_ratio,
                "sparsity": sparsity,
                "n_hidden": M.N_HID,
                "n_inh": M.N_INH,
                "n_in": M.N_IN,
                "max_samples": max_samples,
                "dataset": dataset,
                "load_weights": str(load_weights),
            },
            "best_acc": acc,
            "n_correct": correct,
            "n_total": total,
        }
        metrics_blob["rates_hz"] = rates_hz
        metrics_blob["hid_rate_hz"] = hid_rate_hz
        with open(out_dir_path / "metrics.json", "w") as f:
            json.dump(metrics_blob, f, indent=2, default=float)
        log.info(f"  → {out_dir_path / 'metrics.json'}")

    return {"acc": acc, "rates_hz": rates_hz, "hid_rate_hz": hid_rate_hz}


def _apply_from_dir(args, argv):
    """Fill infer args from a training run's config.json + weights.pth.

    Only sets values the user didn't explicitly pass on the CLI.
    """
    from_dir = Path(args.from_dir)
    cfg_path = from_dir / "config.json"
    if not cfg_path.exists():
        print(f"Error: {cfg_path} not found")
        sys.exit(1)
    cfg = json.loads(cfg_path.read_text())

    # Resolve legacy model names from earlier refactors.
    from config import LEGACY_MODEL_ALIASES

    if cfg.get("model") in LEGACY_MODEL_ALIASES:
        old = cfg["model"]
        cfg["model"] = LEGACY_MODEL_ALIASES[old]
        print(f"  legacy model name {old!r} → {cfg['model']!r}")

    # Auto-detect weights
    if args.load_weights is None:
        weights_path = from_dir / "weights.pth"
        if not weights_path.exists():
            print(f"Error: {weights_path} not found (pass --load-weights explicitly)")
            sys.exit(1)
        args.load_weights = str(weights_path)

    # Auto-set out-dir to a sibling infer/ dir if not specified
    if args.out_dir is None:
        args.out_dir = str(from_dir / "infer")

    # Map config.json keys → argparse dest names.
    # Only apply if the user didn't explicitly pass the flag.
    _CONFIG_TO_ARGS = {
        "model": "model",
        "dt": "dt",
        "t_ms": "t_ms",
        "dataset": "dataset",
        "hidden_sizes": "n_hidden",
        "ei_strength": "ei_strength",
        "ei_ratio": "ei_ratio",
        "sparsity": "sparsity",
        "w_in_sparsity": "w_in_sparsity",
        "w_in": "w_in",
        "input_rate": "spike_rate",
        "max_samples": "max_samples",
        "kaiming_init": "kaiming_init",
        "dales_law": "dales_law",
        "w_rec": "w_rec",
        "rec_layers": "rec_layers",
        "ei_layers": "ei_layers",
        "seed": "seed",
    }
    # Backwards compat: old config.json has "n_hidden" as int
    if "n_hidden" in cfg and "hidden_sizes" not in cfg:
        val = cfg["n_hidden"]
        if isinstance(val, int):
            cfg["hidden_sizes"] = [val]
        elif isinstance(val, list):
            cfg["hidden_sizes"] = val

    # Build set of flags explicitly passed on CLI
    explicit = set()
    for a in argv:
        if a.startswith("--"):
            explicit.add(a.split("=")[0])

    # Reverse lookup: argparse dest → CLI flag name
    _DEST_TO_FLAG = {
        "model": "--model",
        "dt": "--dt",
        "t_ms": "--t-ms",
        "dataset": "--dataset",
        "n_hidden": "--n-hidden",
        "ei_strength": "--ei-strength",
        "ei_ratio": "--ei-ratio",
        "sparsity": "--ei-sparsity",
        "w_in_sparsity": "--w-in-sparsity",
        "w_in": "--w-in",
        "spike_rate": "--input-rate",
        "max_samples": "--max-samples",
        "kaiming_init": "--kaiming-init",
        "dales_law": "--dales-law",
        "w_rec": "--w-rec",
        "hidden_sizes": "--n-hidden",
        "rec_layers": "--rec-layers",
        "ei_layers": "--ei-layers",
        "seed": "--seed",
    }

    inherited = []
    for cfg_key, dest in _CONFIG_TO_ARGS.items():
        if cfg_key not in cfg or cfg[cfg_key] is None:
            continue
        flag = _DEST_TO_FLAG.get(dest, f"--{dest}")
        if flag in explicit or flag.replace("--", "--no-") in explicit:
            continue
        val = cfg[cfg_key]
        setattr(args, dest, val)
        inherited.append(f"{cfg_key}={val}")

    if inherited:
        print(f"  from-dir: inherited {', '.join(inherited)}")

    # Warn about critical flags missing from config.json (old training runs)
    critical = ["kaiming_init", "dales_law"]
    missing = [k for k in critical if k not in cfg]
    if missing:
        print(
            f"  WARNING: config.json missing {missing} — this training run "
            f"predates these flags. Pass them explicitly on the CLI or retrain."
        )


def parse_args():
    """Parse command-line arguments with subparsers for sim/image/video/train."""
    import argparse

    _examples = """\
Network:
  --model MODEL             ping|standard-snn|cuba (default: ping)
  --n-hidden N              N_E neurons, N_I=N_E//4 (default: 1024)
  --n-input N               N_IN input neurons (default: N_E)
  --ei-strength S           E-I coupling, W_EI=s W_IE=s*ratio (default: 0.5)
  --ei-ratio R              W_IE/W_EI ratio (default: 2.0)
  --ei-sparsity F           E-I sparsity (default: 0.2)
  --w-in-sparsity F         W_in sparsity (default: 0.95)
  --bias B                  background conductance uS (default: 0.0002)
  --dt DT                   timestep ms (default: 0.25)
  --device DEV              cpu|mps|cuda (default: cpu)

Input:
  --input MODE              synthetic-spikes|dataset (default: synthetic-spikes)
  --input-rate HZ           baseline Poisson rate (default: 25)
  --stim-overdrive X        stimulus multiplier (default: 1.0)
  --dataset NAME            scikit|mnist|smnist (default: scikit)
  --digit D                 digit class 0-9 (default: 0)

Weights (advanced):
  --w-in MEAN STD           W_in init (default: 0.3 0.06; standard-snn needs ~10 2)
  --w-ei MEAN STD           W_EI init (overrides --ei-strength)
  --w-ie MEAN STD           W_IE init (overrides --ei-strength)

Scan (video mode):
  --scan-var VAR            stim-overdrive|ei_strength|spike_rate|bias|dt|...
  --scan-min / --scan-max   sweep range
  --frames N                number of frames (default: 10)
  --frame-rate FPS          video fps (default: 10)

Train:
  --epochs N                training epochs (default: 0, probe only)
  --lr RATE                 learning rate (default: 0.01)
  --observe video|images    save oscilloscope per epoch
  --max-samples N           limit dataset size
  --v-grad-dampen S         COBA gradient dampening (default: 80)
  --early-stopping N        stop after N epochs without improvement

Output:
  --out-dir DIR             output directory

Examples:
  oscilloscope.py                                  # sim (metrics only)
  oscilloscope.py image                            # snapshot (ping default)
  oscilloscope.py video --scan-var ei_strength     # sweep E-I coupling
  oscilloscope.py video --scan-var spike_rate --scan-min 5 --scan-max 100
  oscilloscope.py train --epochs 100 --observe video
  oscilloscope.py train --epochs 0               # probe init state
  oscilloscope.py image --input dataset --dataset mnist --digit 3
  oscilloscope.py infer --load-weights weights.pth --dt 0.5
  oscilloscope.py infer --load-weights w.pth --dt 1.0 --dataset mnist

Models:
  ping (default)      COBA E-I with PING oscillations
  oscilloscope.py image --ei-strength 0.5          # PING on
  oscilloscope.py image --ei-strength 0            # PING off (E-only COBA)

  standard-snn      snnTorch-library form: mem = β·mem + I. β is a
                      dimensionless hyperparameter, no dt semantics. Not
                      dt-invariant.
  oscilloscope.py image --model standard-snn --kaiming-init

  cuba                Proper continuous-time exp-Euler CUBA: mem = β·mem + (1-β)·I,
                      β = exp(-dt/τ). Bias and weights have per-ms semantics.
                      dt-invariant by construction.
  oscilloscope.py image --model cuba --kaiming-init
"""
    parser = argparse.ArgumentParser(
        description="Oscilloscope — PING network toolkit",
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Shared parent for network/input args
    parent = argparse.ArgumentParser(add_help=False)
    net_group = parent.add_argument_group("Network")
    net_group.add_argument(
        "--model",
        type=str,
        default="ping",
        choices=list(_MODEL_CLASSES.keys()),
        help="Model to simulate (default: ping)",
    )
    net_group.add_argument(
        "--n-hidden",
        type=int,
        nargs="+",
        default=None,
        help="Hidden layer sizes. Single value = 1 layer, "
        "multiple = stacked layers (e.g. --n-hidden 128 256). "
        "(default: dataset-aware smart default; "
        "scikit=64, mnist=1024, smnist=32)",
    )
    net_group.add_argument(
        "--kaiming-init",
        action="store_true",
        help="Use plain nn.Linear Kaiming uniform init "
        "(signed weights, no fan-in normalization). "
        "Matches canonical snnTorch tutorial setup. "
        "Only applies to standard-snn / cuba; "
        "--w-in is ignored when this is set.",
    )
    net_group.add_argument(
        "--readout",
        choices=["rate", "li", "spike-count", "mem-mean"],
        default="rate",
        dest="readout_mode",
        help="Output layer: 'rate' sums last-hidden spikes "
        "and projects linearly at the final timestep "
        "(default); 'li' uses a non-spiking leaky "
        "integrator per class with max-over-time, the "
        "standard SHD-style readout.",
    )
    net_group.add_argument(
        "--dales-law",
        action="store_true",
        default=True,
        help="Enforce Dale's law: clamp weights to non-negative (default: True)",
    )
    net_group.add_argument(
        "--no-dales-law",
        dest="dales_law",
        action="store_false",
        help="Allow signed (positive + negative) weights (standard-snn / cuba only)",
    )
    net_group.add_argument(
        "--rec-layers",
        type=int,
        nargs="+",
        default=None,
        help="Which hidden layers get recurrence (1-indexed). "
        "Default: all layers when --w-rec is set.",
    )
    net_group.add_argument(
        "--ei-layers",
        type=int,
        nargs="+",
        default=None,
        help="Which hidden layers get E-I structure (1-indexed). "
        "Default: all layers (PING only).",
    )
    net_group.add_argument(
        "--n-input", type=int, default=None, help="N_IN input neurons (default: N_E)"
    )
    net_group.add_argument(
        "--ei-strength",
        type=float,
        default=0.5,
        help="E-I coupling: sets W_EI=s, W_IE=s*ratio (default: 0.5)",
    )
    net_group.add_argument(
        "--ei-ratio", type=float, default=2.0, help="W_IE/W_EI ratio (default: 2.0)"
    )
    net_group.add_argument(
        "--ei-sparsity",
        type=float,
        default=None,
        dest="sparsity",
        help="E-I connection sparsity (default: 0.2)",
    )
    net_group.add_argument(
        "--w-in-sparsity",
        type=float,
        default=None,
        help="W_in sparsity (default: 0.95)",
    )
    net_group.add_argument(
        "--bias",
        type=float,
        default=None,
        help="Background conductance to E neurons in uS",
    )
    net_group.add_argument(
        "--dt",
        type=float,
        default=0.25,
        help="Integration timestep in ms (default: 0.25)",
    )
    net_group.add_argument(
        "--t-ms",
        type=float,
        default=200.0,
        help="Total simulation duration in ms (default: 200). "
        "For synthetic-step modes, must exceed STEP_ON_MS "
        "(default 200) so the stimulus window is reached; "
        "values <= STEP_ON_MS leave the trial in flat baseline.",
    )
    net_group.add_argument(
        "--burn-in",
        type=float,
        default=None,
        help="Burn-in period in ms hidden from oscilloscope plots/videos. "
        "Default: 100 ms in shared cfg (legacy); pass 0 to make the entire "
        "trial visible.",
    )
    net_group.add_argument(
        "--tau-mem",
        type=float,
        default=None,
        help="Membrane time constant τ_mem in ms "
        "(default: 10 ms, module-level `tau_snn`). "
        "Cramer et al. SHD: 20 ms.",
    )
    net_group.add_argument(
        "--tau-syn",
        type=float,
        default=None,
        help="Synaptic time constant τ_syn in ms for the "
        "exponential synapse (default: 2 ms, "
        "module-level `tau_ampa`). Cramer et al. SHD: "
        "10 ms. Only affects models with "
        "exponential_synapse=True (e.g. cuba-exp).",
    )
    net_group.add_argument(
        "--readout-tau-out",
        type=float,
        default=None,
        help="Output-LIF time constant τ_out in ms for "
        "the spike-count readout (default: 5 ms, "
        "module-level `tau_out_ms`). Smaller values "
        "speed up output-membrane leak so it doesn't "
        "saturate under high-rate hidden drive — "
        "needed for snnTorch-family models at coarse "
        "dt. No-op for --readout rate or li.",
    )
    net_group.add_argument(
        "--readout-w-out-scale",
        type=float,
        default=1.0,
        help="Multiply the readout matrix W_ff[-1] (and "
        "bias b_ff[-1] if present) by this scalar "
        "after build_net. Use to compensate for the "
        "10× lower hidden firing rate of COBANet "
        "models vs CUBANet under mem-mean / "
        "spike-count readouts: bumping W_out by ~10 "
        "equalises the trial-level drive into the "
        "output LIF and recovers gradient signal. "
        "Train-mode only. Default 1.0.",
    )
    net_group.add_argument(
        "--exp-synapse",
        action="store_true",
        help="Promote --model standard-snn to its "
        "exponential-synapse variant (standard-snn-exp). "
        "No-op for any other model — cuba-exp / coba / "
        "ping already enable exp synapses by their "
        "registry definitions.",
    )
    net_group.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Global-norm gradient-clip threshold passed to "
        "torch.nn.utils.clip_grad_norm_. Default 1.0 "
        "(M.GRAD_CLIP). Set high (e.g. 100+) to let "
        "Adam handle wildly-scaled BPTT gradients via "
        "its second-moment preconditioner.",
    )
    net_group.add_argument(
        "--surrogate-slope",
        type=float,
        default=None,
        help="Fast-sigmoid surrogate-gradient slope β. "
        "Larger = narrower active window around "
        "threshold. pinglab default 1.0; Cramer et al. "
        "use 40 for SHD RSNNs. Applies to "
        "SurrogateSpike (cuba / standard-snn / ping) "
        "and snntorch-library's fast_sigmoid.",
    )
    net_group.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "mps", "cuda"],
        help="Compute device. If unset, auto-detects: cuda > mps > cpu.",
    )

    inp_group = parent.add_argument_group("Input")
    inp_group.add_argument(
        "--input",
        type=str,
        default="synthetic-spikes",
        choices=["synthetic-conductance", "synthetic-spikes", "dataset"],
        help="Input mode (default: synthetic-spikes)",
    )
    inp_group.add_argument(
        "--input-rate",
        type=float,
        default=25.0,
        dest="spike_rate",
        help="Baseline input rate in Hz (default: 25)",
    )
    inp_group.add_argument(
        "--stim-overdrive",
        type=float,
        default=1.0,
        dest="overdrive",
        help="Stimulus multiplier (default: 1.0)",
    )
    inp_group.add_argument(
        "--drive",
        type=float,
        default=None,
        help="Baseline tonic conductance for synthetic-conductance",
    )
    inp_group.add_argument(
        "--digit", type=int, default=0, help="Digit class for dataset input (0-9)"
    )
    inp_group.add_argument(
        "--sample", type=int, default=0, help="Sample index for dataset input"
    )
    inp_group.add_argument(
        "--dataset",
        type=str,
        default="scikit",
        choices=["scikit", "mnist", "smnist", "shd"],
        help="Dataset (default: scikit)",
    )

    wt_group = parent.add_argument_group("Weights (advanced)")
    wt_group.add_argument(
        "--w-in",
        type=float,
        nargs="+",
        default=None,
        metavar=("MEAN", "STD"),
        help="W_in init mean std (default: 0.3 0.06; standard-snn needs ~10 2 dense)",
    )
    wt_group.add_argument(
        "--w-ee",
        type=float,
        nargs=2,
        default=None,
        metavar=("MEAN", "STD"),
        help="W_EE init (mean std). E→E recurrent excitation; default (0, 0) per Börgers PING.",
    )
    wt_group.add_argument(
        "--w-ei",
        type=float,
        nargs=2,
        default=None,
        metavar=("MEAN", "STD"),
        help="W_EI init (mean std)",
    )
    wt_group.add_argument(
        "--w-ie",
        type=float,
        nargs=2,
        default=None,
        metavar=("MEAN", "STD"),
        help="W_IE init (mean std)",
    )
    wt_group.add_argument(
        "--w-rec",
        type=float,
        nargs=2,
        default=None,
        metavar=("MEAN", "STD"),
        help="W_rec recurrent init (mean std, default: 0 0.1)",
    )
    out_group = parent.add_argument_group("Output")
    out_group.add_argument("--out-dir", type=str, default=None, help="Output directory")
    out_group.add_argument(
        "--wipe-dir", action="store_true", help="Clear output directory before run"
    )
    out_group.add_argument(
        "--raster",
        type=str,
        default="scatter",
        choices=["scatter", "imshow"],
        help="Raster style (default: scatter)",
    )
    out_group.add_argument(
        "--layout",
        type=str,
        default="full",
        choices=list(LAYOUT_PRESETS.keys()),
        help="Panel layout (default: full)",
    )
    out_group.add_argument(
        "--panels", type=str, default=None, help="Comma-separated panel names"
    )

    exec_group = parent.add_argument_group("Execution")
    exec_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed. Seeds Python, NumPy, and torch "
        "(CPU + CUDA + MPS) before dataset load and "
        "model init. Persisted to config.json.",
    )
    exec_group.add_argument(
        "--modal",
        action="store_true",
        help="Run on Modal.com instead of locally. "
        "Artifacts sync back to --out-dir after completion.",
    )
    exec_group.add_argument(
        "--modal-gpu",
        type=str,
        default="T4",
        choices=["none", "T4", "L4", "A10G", "A100", "H100"],
        help="GPU type for Modal runs (default: T4). Use 'none' for CPU-only.",
    )
    exec_group.add_argument(
        "--coba-integrator",
        type=str,
        default="expeuler",
        choices=["expeuler", "fwd"],
        help="Membrane ODE integrator for COBA/PING "
        "(default: expeuler). 'fwd' falls back to "
        "forward Euler for parity comparisons.",
    )

    subparsers = parser.add_subparsers(
        dest="mode", help="Mode: sim (metrics only) | image | video | train | infer"
    )

    # -- sim subcommand --
    sim_parser = subparsers.add_parser(
        "sim",
        parents=[parent],
        help="Run simulation, report metrics, no plot",
        description="Run a single simulation and report firing-rate metrics "
        "without generating plots or video.",
        epilog="Examples:\n"
        "  oscilloscope.py sim --model ping --ei-strength 0.5\n"
        "  oscilloscope.py sim --model standard-snn --dt 0.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- image subcommand --
    image_parser = subparsers.add_parser(
        "image",
        parents=[parent],
        help="Generate an oscilloscope snapshot image",
        description="Run one forward pass and save a still-image "
        "oscilloscope figure (E/I rasters, weight histograms, PSD).",
        epilog="Examples:\n"
        "  # untrained PING on MNIST digit 3\n"
        "  oscilloscope.py image --model ping --dataset mnist --digit 3\n"
        "  # with trained weights\n"
        "  oscilloscope.py image --from-dir path/to/trained --digit 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    image_parser.add_argument(
        "--from-dir",
        type=str,
        default=None,
        help="Load trained weights + inherit config from "
        "a training run directory "
        "(e.g. src/artifacts/calibration/mnist/ping-mnist). "
        "CLI flags override inherited values.",
    )
    image_parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help="Path to a weights.pth file (alternative to --from-dir).",
    )
    image_parser.add_argument(
        "--fake-progress",
        type=float,
        default=None,
        help="Overlay a progress-bar indicator at level 0-1 "
        "(for demo / teaching; default: off).",
    )

    # -- video subcommand --
    video_parser = subparsers.add_parser(
        "video",
        parents=[parent],
        help="Sweep a parameter, save oscilloscope video",
        description="Sweep one parameter linearly between --scan-min and "
        "--scan-max over --frames, rendering one frame per value. "
        "Supports trained networks via --from-dir. "
        "Special scan vars: 'digit' iterates dataset classes, "
        "'noise' adds Poisson noise to input.",
        epilog="Examples:\n"
        "  # untrained PING ei_strength sweep (archive reproduction)\n"
        "  oscilloscope.py video --model ping --n-hidden 1024 \\\n"
        "    --scan-var ei_strength --scan-min 0 --scan-max 0.4 \\\n"
        "    --input synthetic-conductance --frames 600 --frame-rate 120\n\n"
        "  # trained network digit tour\n"
        "  oscilloscope.py video --from-dir path/to/trained \\\n"
        "    --input dataset --scan-var digit --scan-min 0 --scan-max 9",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    video_parser.add_argument(
        "--scan-var",
        type=str,
        default="stim-overdrive",
        choices=list(SCAN_DEFAULTS.keys()),
        help="Parameter to sweep. 'digit' iterates "
        "dataset classes; 'noise' adds input "
        "Poisson noise (Hz). See SCAN_DEFAULTS. "
        "(default: stim-overdrive)",
    )
    video_parser.add_argument(
        "--scan-min",
        type=float,
        default=1.0,
        help="Scan start value, in the variable's units "
        "(default: 1.0). For digit: integer class.",
    )
    video_parser.add_argument(
        "--scan-max",
        type=float,
        default=50.0,
        help="Scan end value (default: 50.0). For digit: integer class.",
    )
    video_parser.add_argument(
        "--resample-input",
        action="store_true",
        help="Use a different Poisson seed on each frame "
        "(default: same seed for all frames).",
    )
    video_parser.add_argument(
        "--frames",
        type=int,
        default=10,
        help="Number of video frames to render "
        "(default: 10). For 'digit' scan, overridden "
        "by scan range.",
    )
    video_parser.add_argument(
        "--frame-rate",
        type=int,
        default=10,
        help="Output video frame rate in fps (default: 10).",
    )
    video_parser.add_argument(
        "--from-dir",
        type=str,
        default=None,
        help="Load trained weights + inherit config from "
        "a training run directory. CLI flags override.",
    )
    video_parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help="Path to weights.pth (alternative to --from-dir).",
    )

    # -- train subcommand --
    train_parser = subparsers.add_parser(
        "train",
        parents=[parent],
        help="Train an SNN to classify digits",
        description="Train a model on MNIST / smnist / scikit-digits using "
        "surrogate-gradient BPTT. Writes weights.pth, metrics.json, "
        "metrics.jsonl, test_predictions.json plus optional video.",
        epilog="Examples:\n"
        "  # standard-snn canonical tutorial mode (full MNIST)\n"
        "  oscilloscope.py train --model standard-snn --kaiming-init \\\n"
        "    --dataset mnist --epochs 40 --lr 0.01 --adaptive-lr\n\n"
        "  # proper continuous-time CUBA LIF\n"
        "  oscilloscope.py train --model cuba --kaiming-init \\\n"
        "    --dataset mnist --epochs 40 --lr 0.01 --adaptive-lr\n\n"
        "  # PING with gamma oscillation on MNIST\n"
        "  oscilloscope.py train --model ping --dataset mnist \\\n"
        "    --ei-strength 0.5 --v-grad-dampen 1000 --lr 0.0001",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Adam learning rate. Biophysical models "
        "(coba/ping) typically need 0.0001, "
        "current-based models 0.01 (default: 0.01).",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Number of training epochs. 0 = probe only "
        "(init snapshot, no training). Default: 0.",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mini-batch size for DataLoader. Default: 64 (from models.BATCH_SIZE).",
    )
    train_parser.add_argument(
        "--observe",
        type=str,
        default=None,
        choices=["video", "images"],
        help="Save oscilloscope per epoch",
    )
    train_parser.add_argument(
        "--observe-every",
        type=int,
        default=1,
        help="Observe every Nth epoch (default: 1)",
    )
    train_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset to N samples (smoke test)",
    )
    train_parser.add_argument(
        "--v-grad-dampen",
        type=float,
        default=80.0,
        help="Gradient dampening for COBA membrane",
    )
    train_parser.add_argument(
        "--early-stopping",
        type=int,
        default=None,
        help="Stop after N epochs without improvement",
    )
    train_parser.add_argument(
        "--adaptive-lr",
        action="store_true",
        help="Enable ReduceLROnPlateau scheduler (factor=0.5, patience=5)",
    )
    train_parser.add_argument(
        "--frame-rate", type=int, default=10, help="Video fps for observe (default: 10)"
    )
    train_parser.add_argument(
        "--profile",
        type=str,
        default=None,
        metavar="PATH",
        help="Wrap the 3rd batch of epoch 0 in "
        "torch.profiler and write a Chrome-format "
        "trace JSON to PATH. Skips the first two "
        "batches to avoid allocator/JIT warmup "
        "noise. Training continues normally after.",
    )
    train_parser.add_argument(
        "--fr-reg-lower-theta",
        type=float,
        default=0.0,
        help="Firing-rate reg: lower-bound target spike "
        "count per neuron per trial (θ_l). "
        "Penalty s_l · Σ relu(θ_l − <z_i>)² is "
        "added to the loss. Default 0 = off. "
        "Cramer et al. SHD RSNN: 0.01.",
    )
    train_parser.add_argument(
        "--fr-reg-lower-strength",
        type=float,
        default=0.0,
        help="Strength s_l on the lower-bound firing-"
        "rate regulariser (default 0 = off). "
        "Cramer et al.: 1.0.",
    )
    train_parser.add_argument(
        "--fr-reg-upper-theta",
        type=float,
        default=0.0,
        help="Firing-rate reg: upper-bound target spike "
        "count per neuron per trial (θ_u). "
        "Penalty s_u · Σ relu(<z_i> − θ_u)² is "
        "added to the loss. Default 0 = off. "
        "Cramer et al. SHD RSNN: 100.",
    )
    train_parser.add_argument(
        "--fr-reg-upper-strength",
        type=float,
        default=0.0,
        help="Strength s_u on the upper-bound firing-"
        "rate regulariser (default 0 = off). "
        "Cramer et al.: 0.06.",
    )
    train_parser.add_argument(
        "--skip-bad-grad-threshold",
        type=float,
        default=None,
        help="Skip opt.step() (and zero grads) whenever "
        "the batch's clipped gradient norm is NaN, "
        "inf, or exceeds this threshold. Band-aid "
        "against single exploded batches poisoning "
        "Adam's second-moment estimate mid-run. "
        "Default off.",
    )
    train_parser.add_argument(
        "--optimizer",
        choices=["adam", "adamax"],
        default="adam",
        help="Optimizer. Adamax uses the L∞ norm for "
        "the second moment instead of an EMA of "
        "squared grads, so a single pathological "
        "batch cannot poison the preconditioner. "
        "Canonical SNN choice (Cramer, Zenke).",
    )
    train_parser.add_argument(
        "--loss",
        choices=["ce", "mse"],
        default="ce",
        dest="loss_mode",
        help="Training loss. 'ce' = cross-entropy on "
        "logits (default; expects unnormalised "
        "logits). 'mse' = L2 between logits and "
        "one-hot targets — the SNN-paper standard "
        "(Bohte 2002, Lee 2016) when the readout's "
        "logit scale makes CE softmax saturate.",
    )

    # -- infer subcommand --
    infer_parser = subparsers.add_parser(
        "infer",
        parents=[parent],
        help="Run inference with trained weights (optional dt sweep)",
        description="Evaluate a trained model on the test set. With "
        "--dt-sweep, run inference at each dt value to measure "
        "temporal-resolution stability.",
        epilog="Examples:\n"
        "  # single-dt inference\n"
        "  oscilloscope.py infer --from-dir path/to/trained --dt 0.1\n\n"
        "  # frozen-input dt-stability sweep\n"
        "  oscilloscope.py infer --from-dir path/to/trained \\\n"
        "    --dt-sweep 0.05 0.1 0.25 0.5 1.0 2.0 \\\n"
        "    --frozen-inputs-mode upsample --observe video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    infer_parser.add_argument(
        "--from-dir",
        type=str,
        default=None,
        help="Inherit params from a training run directory "
        "(reads config.json + weights.pth). "
        "CLI flags override inherited values.",
    )
    infer_parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help="Path to saved weights.pth (auto-detected when --from-dir is set)",
    )
    infer_parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit dataset to N samples"
    )
    infer_parser.add_argument(
        "--dt-sweep",
        type=float,
        nargs="+",
        default=None,
        metavar="DT",
        help="Run inference at each dt value and produce "
        "a sweep summary (e.g. --dt-sweep 0.05 0.1 0.25 0.5 1.0). "
        "Overrides --dt.",
    )
    infer_parser.add_argument(
        "--observe",
        type=str,
        default=None,
        choices=["video", "image"],
        help="Save oscilloscope visualization. "
        "With --dt-sweep: video = one frame per dt. "
        "Without: image = single snapshot.",
    )
    infer_parser.add_argument(
        "--frozen-inputs-mode",
        type=str,
        default=None,
        choices=list(FROZEN_MODES),
        help="How input spikes are transported across dt, "
        "anchored at train-dt (Parthasarathy et al. "
        "§2.1, §2.3): upsample (count-preserving "
        "zero-pad to finer eval-dt per Fig 1B, "
        "requires eval-dt <= train-dt); downsample "
        "(count-preserving sum-pool to coarser "
        "eval-dt per §2.3, requires eval-dt >= "
        "train-dt); resample (fresh Poisson at each "
        "eval-dt, works in both directions but "
        "re-introduces sampling noise).",
    )

    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        sys.exit(0)

    # Promote --model standard-snn to standard-snn-exp when --exp-synapse is
    # set. Other models already encode exponential_synapse via their registry
    # entries (cuba-exp / coba / ping), so the flag is a no-op there.
    if getattr(args, "exp_synapse", False) and args.model == "standard-snn":
        args.model = "standard-snn-exp"

    # Apply global model knobs as early as possible so every downstream
    # entrypoint (train, sim, image, video, infer) sees the right integrator.
    M.COBA_INTEGRATOR = args.coba_integrator
    if getattr(args, "surrogate_slope", None) is not None:
        M.SURROGATE_SLOPE = float(args.surrogate_slope)
    if getattr(args, "grad_clip", None) is not None:
        global GRAD_CLIP
        GRAD_CLIP = float(args.grad_clip)
        M.GRAD_CLIP = GRAD_CLIP
    # Override τ_mem / τ_syn before patch_dt() re-derives β_snn and decay_ampa.
    if getattr(args, "tau_mem", None) is not None:
        M.tau_snn = float(args.tau_mem)
    if getattr(args, "tau_syn", None) is not None:
        M.tau_ampa = float(args.tau_syn)
    if getattr(args, "readout_tau_out", None) is not None:
        M.tau_out_ms = float(args.readout_tau_out)

    # --from-dir: inherit training params from config.json, fill unset values
    if args.mode in ("infer", "video", "image") and getattr(args, "from_dir", None):
        _apply_from_dir(args, sys.argv[1:])
    if args.mode == "infer" and not getattr(args, "load_weights", None):
        print("Error: infer requires --load-weights or --from-dir")
        sys.exit(1)

    # Auto-detect: if user explicitly passed --dataset/--digit/--sample but
    # left --input at the default "synthetic-spikes", flip to "dataset". The
    # explicit dataset flags only make sense in dataset input mode, so this
    # avoids the silent footgun where "image --dataset mnist --digit 0" went
    # through the synthetic-spikes branch and ignored the digit.
    def _flag_in_argv(*names):
        for arg in sys.argv[1:]:
            for n in names:
                if arg == n or arg.startswith(n + "="):
                    return True
        return False

    args._input_auto = False
    from_dir_set_dataset = getattr(args, "from_dir", None) and getattr(
        args, "dataset", "scikit"
    ) in ("mnist", "smnist")
    if args.input == "synthetic-spikes" and (
        _flag_in_argv("--dataset", "--digit", "--sample") or from_dir_set_dataset
    ):
        args.input = "dataset"
        args._input_auto = True

    return args


def save_run_artifacts(out_dir, args, mode):
    """Save config.json (with provenance), run.sh, set up logging, print intro."""
    import json
    import logging
    import run_log

    out_dir = Path(out_dir)
    if args.wipe_dir and out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # config.json — with provenance metadata at top
    config = {"mode": mode}
    config.update(run_log.provenance())
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # run.sh
    with open(out_dir / "run.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(" ".join(sys.argv) + "\n")

    # output.log — file handler strips ANSI, stdout keeps it
    log = logging.getLogger("oscilloscope")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    class _StripAnsiFormatter(logging.Formatter):
        def format(self, record):
            msg = super().format(record)
            return run_log._strip_ansi(msg)

    fh = logging.FileHandler(out_dir / "output.log", mode="w")
    fh.setFormatter(_StripAnsiFormatter("%(message)s"))
    log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(sh)

    # Print structured intro
    _print_intro(log, config, args, mode)

    return log


def _print_intro(log, config, args, mode):
    """Group CLI args into sections and print via run_log.print_intro."""
    import run_log

    model = config.get("model", "ping")
    dataset = config.get("dataset", "scikit")

    g = lambda *keys: {k: config[k] for k in keys if k in config}

    sections = {
        "Data": g("dataset", "digit", "sample", "max_samples"),
        "Simulation": {
            "dt": f"{config.get('dt', '?')} ms",
            "t_ms": f"{config.get('t_ms', '?')}",
            "input_rate": f"{config.get('spike_rate', '?')} Hz",
            "input": config.get("input", "?"),
            "burn_in": f"{config.get('burn_in', '?')} ms",
        },
        "Network": {
            "model": model,
            "hidden_sizes": config.get("n_hidden", "?"),
            "kaiming_init": config.get("kaiming_init", False),
            "dales_law": config.get("dales_law", True),
            "ei_strength": config.get("ei_strength"),
            "ei_ratio": config.get("ei_ratio"),
        },
        "Weights": g("w_in", "w_in_sparsity", "w_ei", "w_ie", "w_rec"),
        "Training": g("epochs", "lr", "adaptive_lr", "v_grad_dampen")
        if mode == "train"
        else {},
        "Scan": g("scan_var", "scan_min", "scan_max", "frames", "frame_rate")
        if mode == "video"
        else {},
        "Output": g("out_dir", "observe", "wipe_dir"),
        "Provenance": {
            "git_sha": config.get("git_sha"),
            "device": config.get("device"),
            "run_id": config.get("run_id"),
            "started_at": config.get("started_at"),
        },
    }
    run_log.print_intro(log, mode, model, dataset, sections)


if __name__ == "__main__":
    _t0 = _time.monotonic()

    args = parse_args()
    mode = args.mode

    # --modal: re-dispatch to Modal and exit
    if getattr(args, "modal", False):
        out_dir = args.out_dir
        if out_dir is None:
            out_dir = str(Path(__file__).parent.parent / "artifacts" / "oscilloscope")
        # Rebuild CLI args without --modal / --modal-gpu
        skip = {"--modal"}
        cli_args = []
        argv = sys.argv[1:]
        i = 0
        while i < len(argv):
            if argv[i] == "--modal-gpu":
                i += 2  # skip flag and its value
            elif argv[i] in skip:
                i += 1
            else:
                cli_args.append(argv[i])
                i += 1
        from modal_app import dispatch_to_modal

        dispatch_to_modal(cli_args, out_dir, gpu=args.modal_gpu)
        sys.exit(0)

    # Build and sync config for sim/image/video modes
    if mode != "train":
        c = build_config(args)
        _sync_globals_from_cfg(c)

    import config as C

    # Determine output directory
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = str(Path(__file__).parent.parent / "artifacts" / "oscilloscope")
    out_dir = Path(out_dir)

    # Save run artifacts for all modes
    log = save_run_artifacts(out_dir, args, mode)

    # Create enriched .running marker (PID, start time, run_id, cmd).
    # Deleted on normal exit via atexit hook.
    import run_log as _rl
    import json as _json_marker

    try:
        _cfg = _json_marker.loads((out_dir / "config.json").read_text())
        _rid = _cfg.get("run_id", _rl.run_id())
    except Exception:
        _rid = _rl.run_id()
    _running_marker = _rl.write_running_marker(out_dir, _rid)

    import atexit as _atexit

    def _cleanup_running_marker():
        try:
            _running_marker.unlink(missing_ok=True)
        except Exception:
            pass

    _atexit.register(_cleanup_running_marker)

    # Single source of truth for input Poisson rate and sim duration. Every
    # code path reads M.max_rate_hz and M.T_ms, so setting them here once means
    # all dispatch branches (sim/image/video/train/infer × all input types)
    # respect --input-rate / --t-ms without per-call plumbing. Subfunctions
    # that change dt will recalc M.p_scale and M.T_steps via patch_dt as usual.
    M.max_rate_hz = args.spike_rate
    M.T_ms = args.t_ms
    if args.t_ms <= C.STEP_ON_MS:
        log.warning(
            f"  --t-ms={args.t_ms} <= STEP_ON_MS={C.STEP_ON_MS}: "
            f"the stimulus window never fires within the trial. "
            f"Consider --t-ms >= {C.STEP_OFF_MS:.0f} (STEP_OFF_MS)."
        )

    if args._input_auto:
        log.info(f"  --input auto → dataset (inferred from --dataset/--digit/--sample)")

    if mode == "sim":
        t_e_async = C.T_E_ASYNC_DEFAULT
        log.info(f"Device: {C.DEVICE}")
        generate_sim_only(
            spike_rate=args.spike_rate,
            overdrive=args.overdrive,
            dt=args.dt,
            model_name=args.model,
            input_mode=args.input,
            t_e_async=t_e_async,
        )

    elif mode == "image":
        t_e_async = C.T_E_ASYNC_DEFAULT
        log.info(f"Device: {C.DEVICE}")
        if args.input == "dataset":
            generate_image_snapshot(
                digit_class=args.digit,
                sample_idx=args.sample,
                dt=args.dt,
                dataset=args.dataset,
                overdrive=args.overdrive,
                model_name=args.model,
                load_weights=getattr(args, "load_weights", None),
            )
        elif args.input == "synthetic-spikes":
            generate_spike_snapshot(
                spike_rate=args.spike_rate,
                overdrive=args.overdrive,
                dt=args.dt,
                model_name=args.model,
            )
        else:
            generate_snapshot(
                args.overdrive,
                dt=args.dt,
                fake_progress=args.fake_progress,
                model_name=args.model,
                t_e_async=t_e_async,
            )

    elif mode == "video":
        t_e_async = C.T_E_ASYNC_DEFAULT
        log.info(f"Device: {C.DEVICE}")
        # Emit init_d0s0.png alongside the scan video when input is dataset.
        # Mirrors train mode so video runs have a comparable static reference.
        if args.input == "dataset":
            generate_image_snapshot(
                digit_class=args.digit,
                sample_idx=args.sample,
                dt=args.dt,
                dataset=args.dataset,
                overdrive=args.overdrive,
                model_name=args.model,
                out_filename="init_d0s0.png",
            )
        generate_scan(
            scan_var=args.scan_var,
            scan_min=args.scan_min,
            scan_max=args.scan_max,
            resample_input=args.resample_input,
            n_frames=args.frames,
            t_e_async=t_e_async,
            overdrive=args.overdrive,
            spike_rate=args.spike_rate,
            input_mode=args.input,
            dataset=args.dataset,
            digit_class=args.digit,
            sample_idx=args.sample,
            load_weights=getattr(args, "load_weights", None),
        )

    elif mode == "train":
        w_in = args.w_in or [0.3, 0.06]
        if len(w_in) == 1:
            w_in = [w_in[0], w_in[0] * 0.1]
        train(
            model_name=args.model,
            lr=args.lr,
            epochs=args.epochs,
            dt=args.dt or 0.1,
            observe=args.observe,
            out_dir=str(out_dir),
            device_name=args.device,
            w_in=w_in,
            ei_strength=args.ei_strength,
            ei_ratio=args.ei_ratio,
            sparsity=args.sparsity or 0.0,
            w_in_sparsity=args.w_in_sparsity or 0.0,
            dataset=args.dataset,
            snapshot_init=True,
            snapshot_end=True,
            t_ms=args.t_ms,
            burn_in_ms=args.burn_in if args.burn_in is not None else 20.0,
            hidden_sizes=args.n_hidden,
            max_samples=args.max_samples,
            v_grad_dampen=args.v_grad_dampen,
            early_stopping=args.early_stopping,
            observe_every=args.observe_every,
            adaptive_lr=args.adaptive_lr,
            kaiming_init=args.kaiming_init,
            dales_law=args.dales_law,
            w_rec=args.w_rec,
            rec_layers=args.rec_layers,
            ei_layers=args.ei_layers,
            batch_size=args.batch_size,
            seed=args.seed,
            readout_w_out_scale=args.readout_w_out_scale,
            readout_mode=args.readout_mode,
            fr_reg_lower_theta=args.fr_reg_lower_theta,
            fr_reg_lower_strength=args.fr_reg_lower_strength,
            fr_reg_upper_theta=args.fr_reg_upper_theta,
            fr_reg_upper_strength=args.fr_reg_upper_strength,
            skip_bad_grad_threshold=args.skip_bad_grad_threshold,
            optimizer=args.optimizer,
            loss_mode=args.loss_mode,
            profile_path=args.profile,
        )

    elif mode == "infer":
        w_in = args.w_in or [0.3, 0.06]
        if len(w_in) == 1:
            w_in = [w_in[0], w_in[0] * 0.1]
        infer_kwargs = dict(
            model_name=args.model,
            load_weights=args.load_weights,
            dataset=args.dataset,
            max_samples=args.max_samples,
            t_ms=args.t_ms,
            w_in=w_in,
            ei_strength=args.ei_strength,
            ei_ratio=args.ei_ratio,
            sparsity=args.sparsity or 0.0,
            w_in_sparsity=args.w_in_sparsity or 0.0,
            hidden_sizes=args.n_hidden,
            kaiming_init=args.kaiming_init,
            dales_law=args.dales_law,
            w_rec=args.w_rec,
            rec_layers=args.rec_layers,
            ei_layers=args.ei_layers,
            seed=args.seed,
        )
        if args.dt_sweep:
            dt_values = sorted(args.dt_sweep)
            log.info(f"dt sweep: {dt_values}")

            encoder = None
            frozen_mode = getattr(args, "frozen_inputs_mode", None)
            if frozen_mode is not None:
                # Anchor the reference at the training dt (paper's framing):
                # spikes are generated at dt_ref = args.dt once, then
                # transported to each sweep dt via upsample zero-pad (finer,
                # §2.1 Fig 1B) or downsample sum-pool (coarser, §2.3).
                # resample draws fresh at the target.
                dt_ref = float(args.dt)
                if frozen_mode != "resample":
                    for d in dt_values:
                        if abs(d - dt_ref) < 1e-9:
                            continue
                        if frozen_mode == "upsample" and d > dt_ref + 1e-9:
                            raise ValueError(
                                f"--frozen-inputs-mode upsample requires "
                                f"eval-dt <= train-dt; got dt={d}, "
                                f"dt_ref={dt_ref}"
                            )
                        if frozen_mode == "downsample" and d < dt_ref - 1e-9:
                            raise ValueError(
                                f"--frozen-inputs-mode downsample requires "
                                f"eval-dt >= train-dt; got dt={d}, "
                                f"dt_ref={dt_ref}"
                            )
                        ratio = max(d, dt_ref) / min(d, dt_ref)
                        if abs(ratio - round(ratio)) > 1e-6:
                            raise ValueError(
                                f"--frozen-inputs-mode {frozen_mode} requires "
                                f"integer dt ratios vs train-dt; "
                                f"dt={d}, dt_ref={dt_ref}, ratio={ratio:.4f}"
                            )
                encoder = FrozenEncoder(dt_ref, t_ms=args.t_ms, mode=frozen_mode)
                log.info(
                    f"  frozen inputs: ref dt={dt_ref} (train-dt), mode={frozen_mode}"
                )

            train_dt = float(args.dt)
            base_rate = float(args.spike_rate)

            sweep_results = []
            for sweep_dt in dt_values:
                if encoder is not None:
                    encoder.reset()
                res = infer(
                    dt=sweep_dt, out_dir=None, encode_fn=encoder, **infer_kwargs
                )
                sweep_results.append(
                    {
                        "dt": sweep_dt,
                        "acc": res["acc"],
                        "input_rate": base_rate,
                        "hid_rate_hz": res.get("hid_rate_hz"),
                        "rates_hz": res.get("rates_hz", {}),
                    }
                )
            ref = next((r for r in sweep_results if r["dt"] == train_dt), None)
            ref_acc = ref["acc"] if ref else None

            log.info(f"\n{'=' * 40}")
            log.info(f"dt sweep summary ({args.model}):")
            log.info(f"  {'dt':>8s}  {'acc':>6s}  {'Δacc':>6s}")
            for r in sweep_results:
                delta = f"{r['acc'] - ref_acc:+.1f}%" if ref_acc is not None else ""
                marker = " ←train" if r["dt"] == train_dt else ""
                log.info(f"  {r['dt']:8.4f}  {r['acc']:5.1f}%  {delta:>6s}{marker}")

            sweep_blob = {
                "model": args.model,
                "train_dt": train_dt,
                "input_rate": args.spike_rate,
                "t_ms": args.t_ms,
                "dataset": args.dataset,
                "load_weights": args.load_weights,
                "frozen_inputs_mode": frozen_mode,
                "sweep": sweep_results,
            }
            results_path = out_dir / "results.json"
            with open(results_path, "w") as f:
                json.dump(sweep_blob, f, indent=2)
            log.info(f"  → {results_path}")

            _plot_dt_sweep(sweep_results, train_dt, args.model, out_dir)
            log.info(f"  → {out_dir / 'dt_sweep.png'}")

            if args.observe == "video":
                randomize = not args.kaiming_init
                vid_net = build_net(
                    args.model,
                    w_in=w_in,
                    w_in_sparsity=args.w_in_sparsity or 0.0,
                    ei_strength=args.ei_strength,
                    ei_ratio=args.ei_ratio,
                    sparsity=args.sparsity or 0.0,
                    randomize_init=randomize,
                    kaiming_init=args.kaiming_init,
                    dales_law=args.dales_law,
                    w_rec=args.w_rec,
                    hidden_sizes=args.n_hidden,
                    rec_layers=args.rec_layers,
                    ei_layers=args.ei_layers,
                )
                vid_net.load_state_dict(
                    torch.load(args.load_weights, map_location="cpu"), strict=False
                )
                vid_net.eval()
                _render_dt_sweep_video(
                    vid_net,
                    dt_values,
                    sweep_results,
                    train_dt,
                    args.model,
                    args.dataset,
                    out_dir,
                    frozen_inputs=bool(frozen_mode),
                )
                log.info(f"  → {out_dir / 'dt_sweep.mp4'}")
        else:
            acc = infer(dt=args.dt, out_dir=str(out_dir), **infer_kwargs)["acc"]
            if args.observe:
                import config as C

                C.N_E = M.N_HID
                C.N_I = M.N_INH
                randomize = not args.kaiming_init
                vis_net = build_net(
                    args.model,
                    w_in=w_in,
                    w_in_sparsity=args.w_in_sparsity or 0.0,
                    ei_strength=args.ei_strength,
                    ei_ratio=args.ei_ratio,
                    sparsity=args.sparsity or 0.0,
                    randomize_init=randomize,
                    kaiming_init=args.kaiming_init,
                    dales_law=args.dales_law,
                    w_rec=args.w_rec,
                    hidden_sizes=args.n_hidden,
                    rec_layers=args.rec_layers,
                    ei_layers=args.ei_layers,
                )
                vis_net.load_state_dict(
                    torch.load(args.load_weights, map_location="cpu"), strict=False
                )
                vis_net.eval()
                loader_dataset = (
                    "mnist" if args.dataset in ("mnist", "smnist") else args.dataset
                )
                ref_pixel_vec, ref_image = _load_dataset_image(loader_dataset, 0, 0)
                ref_input = torch.from_numpy(ref_pixel_vec).unsqueeze(0)
                use_smnist = args.dataset == "smnist"
                ref_spikes = encode_batch(ref_input, args.dt, use_smnist)
                fig, axes = make_transient_fig(layout="train")
                observe_epoch(
                    vis_net,
                    ref_spikes,
                    0,
                    acc,
                    0.0,
                    args.dt,
                    args.model,
                    fig,
                    axes,
                    None,
                    digit_image=ref_image,
                    total_epochs=1,
                )
                fname = out_dir / "infer_d0s0.png"
                fig.savefig(fname, dpi=120)
                plt.close(fig)
                log.info(f"  → {fname}")

    _elapsed = _time.monotonic() - _t0
    _m, _s = divmod(int(_elapsed), 60)
    log.info(f"Done in {_m}m {_s}s.")
