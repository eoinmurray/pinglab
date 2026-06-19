"""Snapshot-mode generators for the CLI.

One-shot renderers that produce a single PNG (or a sim-only metrics dump)
for a given drive level / model / input mode. Four variants:
    generate_snapshot          — synthetic conductance drive (default)
    generate_spike_snapshot    — synthetic Poisson spike drive
    generate_image_snapshot    — dataset image as input
    generate_sim_only          — no plot, metrics only
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from figkit import plt

import models as M
from config import (
    _extract_records,
    extract_weights,
    make_net,
    make_ping_net,
    patch_dt,
    run_sim,
    run_sim_image,
)
from inputs import DT_CAL, make_spike_drive
from metrics import report_metrics
from plot import draw_transient_frame, make_transient_fig

from datasets import _load_dataset_image
from encoders import encode_image_spikes
from scan import primary_hid_key, primary_inh_key

log = logging.getLogger("cli")


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
    log.info(f"  → {fname}")


def generate_spike_snapshot(
    spike_rate=None, overdrive=12.0, dt=None, model_name="ping",
    independent_drive=None, independent_drive_i=None,
    quenched_drive=None, quenched_drive_i=None,
    lyapunov_eps=0.0,
):
    """Generate a snapshot with synthetic spike input.

    ``independent_drive``: optional (rate_hz, g_per_spike) tuple. When set,
    generates per-E-cell independent Poisson streams (instead of routing
    everything through W_in) and feeds them as ext_g. Used for V&S/Brunel
    balanced-network experiments where input correlations should be zero.

    ``independent_drive_i``: same as above but targeting the I population's
    excitatory conductance. Required for the full V&S-AI state where
    I cells need uncorrelated noise distinct from the E-mediated W^EI input.

    ``quenched_drive`` / ``quenched_drive_i``: optional (mean, std) tuple.
    Per-cell *constant-in-time* excitatory conductance, drawn once from
    N(mean, std) (clamped ≥ 0) and held frozen for the whole trial — V&S's
    quenched random input. Unlike the Poisson drive it has no per-timestep
    fluctuation, so it cannot pin spike times (Mainen-Sejnowski reliability)
    and the Lyapunov probe sees the network's *autonomous* chaos rather than
    input entrainment.

    ``lyapunov_eps``: if > 0, run a second forward pass on the *same* input
    with all membrane voltages perturbed by an ε-mV random offset at t=0,
    then save the per-timestep *spike-train* divergence D(t) = number of E
    cells whose spike differs between the clean and perturbed copy (keys
    lyap_t_ms / lyap_dist / lyap_eps). Spike-train divergence (not ‖ΔV‖) is
    the right measure for spiking nets: the reset contracts voltage between
    spikes, so chaos lives in spike-flip events. D(t) grows for the chaotic
    V&S balanced state and stays bounded / re-locks for cycle-locked PING.
    """
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

    log.info(f"image | {model_name} spikes {spike_rate:.0f}→{stim_rate:.0f}Hz")

    net = make_ping_net(C.cfg, w_in=(*C.W_IN_SPIKES, "normal", C.W_IN_SPARSITY))

    input_spikes = input_spikes.to(C.DEVICE)
    tonic_g = None
    if C.BIAS > 0:
        tonic_g = torch.full(
            (T_steps, C.N_E), C.BIAS, dtype=torch.float32, device=C.DEVICE
        )
    if independent_drive is not None:
        ind_rate, ind_g = float(independent_drive[0]), float(independent_drive[1])
        gen = torch.Generator(device="cpu").manual_seed(C.SEED + 1)
        p_per_step = ind_rate * dt / 1000.0
        # (T, N_E) independent Bernoulli spikes — one Poisson stream per E cell.
        ind_spikes = (
            torch.rand(T_steps, C.N_E, generator=gen) < p_per_step
        ).to(torch.float32).to(C.DEVICE)
        log.info(
            f"  + independent per-E-cell drive: {ind_rate:.0f} Hz × {ind_g:.4f} μS"
        )
        ind_drive_g = ind_spikes * ind_g
        tonic_g = (tonic_g + ind_drive_g) if tonic_g is not None else ind_drive_g
    tonic_g_i = None
    if independent_drive_i is not None:
        ind_rate_i, ind_g_i = (
            float(independent_drive_i[0]), float(independent_drive_i[1])
        )
        gen_i = torch.Generator(device="cpu").manual_seed(C.SEED + 2)
        p_per_step_i = ind_rate_i * dt / 1000.0
        ind_spikes_i = (
            torch.rand(T_steps, C.N_I, generator=gen_i) < p_per_step_i
        ).to(torch.float32).to(C.DEVICE)
        log.info(
            f"  + independent per-I-cell drive: {ind_rate_i:.0f} Hz × {ind_g_i:.4f} μS"
        )
        tonic_g_i = ind_spikes_i * ind_g_i
    if quenched_drive is not None:
        q_mean, q_std = float(quenched_drive[0]), float(quenched_drive[1])
        gen_q = torch.Generator(device="cpu").manual_seed(C.SEED + 3)
        # one frozen value per E cell, broadcast across all timesteps
        q_cell = torch.clamp(
            torch.normal(q_mean, q_std, (C.N_E,), generator=gen_q), min=0.0
        ).to(C.DEVICE)
        q_drive_g = q_cell.unsqueeze(0).expand(T_steps, -1)
        log.info(
            f"  + quenched per-E-cell DC drive: N({q_mean:.4f}, {q_std:.4f}) μS"
        )
        tonic_g = (tonic_g + q_drive_g) if tonic_g is not None else q_drive_g
    if quenched_drive_i is not None:
        q_mean_i, q_std_i = float(quenched_drive_i[0]), float(quenched_drive_i[1])
        gen_qi = torch.Generator(device="cpu").manual_seed(C.SEED + 4)
        q_cell_i = torch.clamp(
            torch.normal(q_mean_i, q_std_i, (C.N_I,), generator=gen_qi), min=0.0
        ).to(C.DEVICE)
        q_drive_gi = q_cell_i.unsqueeze(0).expand(T_steps, -1)
        log.info(
            f"  + quenched per-I-cell DC drive: N({q_mean_i:.4f}, {q_std_i:.4f}) μS"
        )
        tonic_g_i = (tonic_g_i + q_drive_gi) if tonic_g_i is not None else q_drive_gi
    with torch.no_grad():
        net.forward(input_spikes=input_spikes, ext_g=tonic_g, ext_g_i=tonic_g_i)

    rec = _extract_records(net)

    # Lyapunov check: rerun on identical input with an ε-perturbed initial
    # membrane state and measure how fast the two *spike trains* diverge.
    # Done before downstream record mutation so `rec` still holds the clean
    # run's E spikes.
    lyap_t_ms = lyap_dist = None
    if lyapunov_eps > 0:
        hid_k = primary_hid_key(rec)
        s_clean = np.asarray(rec[hid_k])  # (T, B, N_E) or (T, N_E)
        with torch.no_grad():
            net.forward(
                input_spikes=input_spikes, ext_g=tonic_g, ext_g_i=tonic_g_i,
                v_perturb_eps=float(lyapunov_eps), v_perturb_seed=C.SEED + 7,
            )
        rec_p = _extract_records(net)
        s_pert = np.asarray(rec_p[hid_k])
        n_t = min(s_clean.shape[0], s_pert.shape[0])
        # D(t) = number of E cells whose spike state differs at step t.
        diff = (s_clean[:n_t] != s_pert[:n_t]).reshape(n_t, -1)
        lyap_dist = diff.sum(axis=1).astype(np.float64)
        lyap_t_ms = np.arange(n_t) * dt
        log.info(
            f"  + lyapunov ε={lyapunov_eps:g} mV: "
            f"spike-diff D {lyap_dist[0]:.0f} → {lyap_dist[-1]:.0f} cells "
            f"(max {lyap_dist.max():.0f})"
        )

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
    log.info(f"  → {fname}")

    npz_path = out_dir / "snapshot.npz"
    extra = {}
    for key in ("v_e_1", "ge_e_1", "gi_e_1", "v_i_1", "ge_i_1", "gi_i_1"):
        if key in rec:
            extra[key] = np.asarray(rec[key])
    if lyap_dist is not None:
        extra["lyap_t_ms"] = np.asarray(lyap_t_ms)
        extra["lyap_dist"] = np.asarray(lyap_dist)
        extra["lyap_eps"] = np.float32(lyapunov_eps)
    np.savez(
        npz_path,
        spk_e=np.asarray(spk_e),
        spk_i=np.asarray(spk_i) if spk_i is not None else np.empty((0, 0)),
        dt=np.float32(dt),
        n_e=np.int32(C.N_E),
        n_i=np.int32(C.N_I),
        **extra,
    )
    log.info(f"  → {npz_path}")


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
    log.info(f"  → {fname}")

    npz_name = Path(out_filename).with_suffix(".npz").name
    npz_path = out_dir / npz_name
    extra = {}
    for key in ("v_e_1", "ge_e_1", "gi_e_1", "v_i_1", "ge_i_1", "gi_i_1"):
        if key in rec:
            extra[key] = np.asarray(rec[key])
    np.savez(
        npz_path,
        spk_e=np.asarray(spk_e),
        spk_i=np.asarray(spk_i) if spk_i is not None else np.empty((0, 0)),
        dt=np.float32(dt),
        n_e=np.int32(C.N_E),
        n_i=np.int32(C.N_I),
        **extra,
    )
    log.info(f"  → {npz_path}")


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

        log.info(f"sim | {model_name} spikes {spike_rate:.0f}→{stim_rate:.0f}Hz")

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
