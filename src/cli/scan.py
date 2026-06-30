"""Scan-mode generators for the CLI.

Sweeps a variable from scan_min to scan_max, rendering one video frame per
value. Three internal paths:
    _scan_od_batched     — fast batched path for stim-overdrive scans
    _scan_streaming      — generic per-frame path (all other vars)
    _scan_dt             — dt sweep with dt-invariant reference noise
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from figkit import plt, FFMpegWriter

import models as M
from config import (
    _extract_records,
    _run_sim_with_net,
    extract_weights,
    make_net,
    make_ping_net,
    run_sim,
    run_sim_batch,
    W_EI,
    W_IE,
)
from inputs import (
    DT_CAL,
    make_reference_noise,
    make_spike_drive,
    make_step_drive,
    make_step_drive_from_ref,
)
from metrics import metrics_str
from plot import draw_transient_frame, make_transient_fig
from profiling import prof

from datasets import _load_dataset_image
from encoders import encode_image_spikes

log = logging.getLogger("cli")


# ── Shared utilities (imported by snapshot/train/infer/__main__) ──────────
def _auto_device() -> torch.device:
    """Pick the fastest available device: cuda > mps > cpu.

    Called when the user doesn't explicitly set --device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Recording-key resolvers (shared with cli.py and future modules) ──
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


# Scannable variables
# =============================================================================

SCAN_DEFAULTS = {
    "stim-overdrive": (1.0, "x"),
    "tau_gaba": (9.0, "ms"),
    "tau_ampa": (2.0, "ms"),
    "w_ei_mean": (W_EI[0], "μS"),
    "w_ie_mean": (W_IE[0], "μS"),
    "ei_strength": (1.0, ""),
    "spike_rate": (1.0, "Hz"),
    "bias": (1.0, "μS"),
    "dt": (1.0, "ms"),
    "digit": (0, ""),  # dataset digit class (0-9)
    "noise": (0, "Hz"),  # Poisson noise rate added to input
}


def _apply_scan_var(var_name, value):
    """Apply a scan variable that mutates global state (M globals or cfg).

    Handles tau_* overrides for the M module. Other scan variables like
    stim-overdrive, spike_rate, noise are handled directly in scan loops.
    """
    import config as C

    if var_name == "tau_gaba":
        M.tau_gaba = value
        M.decay_gaba = np.exp(-M.dt / value)
    elif var_name == "tau_ampa":
        M.tau_ampa = value
        M.decay_ampa = np.exp(-M.dt / value)
    elif var_name in ("w_ei_mean", "w_ie_mean", "ei_strength", "bias"):
        # Config param mutations
        C.cfg.apply_frame_param(var_name, value)
    # Other scan vars (stim-overdrive, spike_rate, noise, digit) don't mutate globals


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
    "bias": "bias (μS)",
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
        f"scan {scan_var} {scan_values[0]:.3g}→{scan_values[-1]:.3g}{unit} | {n_frames}f"
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
    log.info(f"  → {out_dir / fname}")
    log.info(f"  frames → {frames_dir}/")
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
    _digit_img = None
    _loaded_net = None

    # Ensure M.T_steps matches dt (needed for forward loop and encoding)
    M.T_steps = int(C.SIM_MS / dt)

    if use_dataset:
        pixel_vec, _digit_img = _load_dataset_image(dataset, digit_class, sample_idx)
        M.N_IN = len(pixel_vec)

    # Load trained weights once if provided
    if load_weights is not None:
        M.N_HID = C.N_E
        M.N_INH = C.N_I
        _loaded_net = make_net(C.cfg, w_in=(*C.W_IN_SPIKES, "normal", C.W_IN_SPARSITY))
        state = torch.load(load_weights, map_location="cpu")
        _loaded_net.load_state_dict(state, strict=False)
        _loaded_net.eval()
        _loaded_net.recording = True
        log.info(f"  loaded weights: {load_weights}")

    elif use_spikes:
        M.N_HID = C.N_E
        M.N_INH = C.N_I
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
                pred_str = f" d{truth}→pred={pred}{mark}"
            log.info(f"  {i + 1}/{n_total} {scan_var}={val:.3g}{unit} | {m}{pred_str}")

    plt.close(fig)
    log.info(f"  → {out_dir / fname}")
    log.info(f"  frames → {frames_dir}/")
    prof.report(n_total)
