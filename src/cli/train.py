"""Training driver for the oscilloscope CLI.

Holds seed_everything, observe_epoch (per-epoch oscilloscope frame), and
the main train() loop that supports CUBA / COBA / PING / snnTorch-library
models across MNIST / FashionMNIST / Yin-Yang / sMNIST / SHD with a
configurable readout, loss, dataset encoder, gradient stabilizer, and
optimizer.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time as _time
from pathlib import Path

_pkg_dir = str(Path(__file__).resolve().parent.parent)
if _pkg_dir in sys.path:
    sys.path.remove(_pkg_dir)
sys.path.insert(0, _pkg_dir)
_cli_dir = str(Path(__file__).resolve().parent)
if _cli_dir in sys.path:
    sys.path.remove(_cli_dir)
sys.path.insert(0, _cli_dir)

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import models as M
from config import (
    _extract_records,
    build_net,
    extract_weights,
    make_net,
    patch_dt,
)
from inputs import DT_CAL, make_spike_drive
from metrics import compute_metrics, format_metrics
from plot import draw_transient_frame, make_transient_fig, reset_weight_xlims
from runlog import (
    MetricsJsonl,
    WarningTracker,
    print_epoch,
    print_progress_header,
    print_summary,
    provenance,
    run_id,
    write_test_predictions,
)

from cli.datasets import (
    DATASET_N_HIDDEN_DEFAULTS,
    SHD_N_CHANNELS,
    _load_dataset_image,
    load_dataset,
)
from cli.encoders import EVAL_SEED, encode_batch, encode_smnist
from cli.scan import _auto_device, primary_hid_key, primary_inh_key

log = logging.getLogger("oscilloscope")


BATCH_SIZE = 64
GRAD_CLIP = 1.0

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
    w_ei=None,
    w_ie=None,
    ei_strength=None,
    ei_ratio=2.0,
    w_in_sparsity=0.0,
    dataset="scikit",
    snapshot_init=True,
    snapshot_end=True,
    t_ms=200.0,
    hidden_sizes=None,
    max_samples=None,
    v_grad_dampen=80.0,
    dales_law=True,
    ei_layers=None,
    batch_size=None,
    seed=None,
    readout_w_out_scale=1.0,
    readout_mode="rate",
    tau_gaba=None,
    fr_reg_upper_theta=0.0,
    fr_reg_upper_strength=0.0,
    fr_reg_lower_theta=0.0,
    fr_reg_lower_strength=0.0,
    fr_reg_mode="per-neuron",
    trainable_w_ee=False,
    trainable_w_ei=False,
    trainable_w_ie=False,
    tbptt_window=None,
):
    """Train on scikit digits, optionally producing oscilloscope video."""
    import time
    from torch.utils.data import DataLoader, TensorDataset

    burn_in_ms = 20.0

    seed_everything(seed)

    # Setup dt and all derived constants
    M.T_ms = t_ms
    patch_dt(dt)
    # τ_GABA override: must apply AFTER patch_dt (which recomputes
    # decay_gaba from the current M.tau_gaba). nb041 sweeps this to
    # vary the gamma frequency f_γ across retrained networks.
    if tau_gaba is not None:
        import numpy as _np
        M.tau_gaba = float(tau_gaba)
        M.decay_gaba = float(_np.exp(-M.dt / float(tau_gaba)))

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
    net = build_net(
        model_name,
        w_in=w_in,
        w_in_sparsity=w_in_sparsity,
        w_ei=w_ei,
        w_ie=w_ie,
        ei_strength=ei_strength,
        ei_ratio=ei_ratio,
        device=device,
        randomize_init=True,
        dales_law=dales_law,
        hidden_sizes=hidden_sizes,
        ei_layers=ei_layers,
        readout_mode=readout_mode,
        trainable_w_ee=trainable_w_ee,
        trainable_w_ei=trainable_w_ei,
        trainable_w_ie=trainable_w_ie,
        tbptt_window=tbptt_window,
    )
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
    import runlog

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
        "w_in_sparsity": w_in_sparsity,
        "input_rate": M.max_rate_hz,
        "v_grad_dampen": v_grad_dampen,
        "burn_in_ms": burn_in_ms,
        "batch_size": bs,
        "grad_clip": GRAD_CLIP,
        "max_samples": max_samples,
        "n_params": n_params,
        "n_trainable": n_trainable,
        "dales_law": dales_law,
        "readout_mode": readout_mode,
        "hidden_sizes": hidden_sizes,
        "ei_layers": list(ei_layers) if ei_layers else None,
        "trainable_w_ee": trainable_w_ee,
        "trainable_w_ei": trainable_w_ei,
        "trainable_w_ie": trainable_w_ie,
        "tbptt_window": getattr(net, "tbptt_window", None),
        "seed": seed,
        "tau_gaba_ms": float(M.tau_gaba),
        # Provenance (git SHA, run_id, started_at, device, torch version,
        # python env hash) — keeps train-mode config.json at parity with
        # sim/image/video modes.
        **runlog.provenance(),
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
                "n_hidden": M.N_HID,
                "n_inh": M.N_INH,
                "n_in": M.N_IN,
                "max_samples": max_samples,
                "dataset": dataset,
                "v_grad_dampen": v_grad_dampen,
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

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    _ce = torch.nn.CrossEntropyLoss()

    def loss_fn(logits, y):
        return _ce(logits, y)

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
    import runlog

    best_acc = 0.0
    best_state = None
    no_improve = 0
    t_start = _time.perf_counter()
    prev_lr = lr
    epoch_records = []  # accumulated for metrics.json
    best_epoch = 0
    wtracker = runlog.WarningTracker()
    jsonl = runlog.MetricsJsonl(out_dir / "metrics.jsonl")
    runlog.print_progress_header(log)
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
            X_b, y_b = X_b.to(device), y_b.to(device)
            spk = encode_batch(X_b, dt, use_smnist)
            logits = net(input_spikes=spk)
            if torch.isnan(logits).any():
                opt.zero_grad()
                continue
            loss = loss_fn(logits, y_b)
            if fr_reg_upper_strength > 0 and getattr(
                net, "last_spike_counts", None
            ) is not None:
                reg = 0.0
                for sc in net.last_spike_counts:
                    if fr_reg_mode == "population":
                        # Single scalar penalty on the grand mean across batch
                        # and neurons; scale by n_neurons so s_u retains its
                        # per-neuron-recipe magnitude when all cells overshoot.
                        pop_mean = sc.mean()
                        n_neurons = sc.shape[-1]
                        reg = (
                            reg
                            + fr_reg_upper_strength
                            * n_neurons
                            * torch.relu(pop_mean - fr_reg_upper_theta) ** 2
                        )
                    else:
                        mean_z = sc.mean(dim=0)
                        reg = (
                            reg
                            + fr_reg_upper_strength
                            * (torch.relu(mean_z - fr_reg_upper_theta) ** 2).sum()
                        )
                loss = loss + reg
            if fr_reg_lower_strength > 0 and getattr(
                net, "last_spike_counts", None
            ) is not None:
                # Symmetric lower-bound: penalise being BELOW θ_l, i.e.
                # reward firing rate at or above θ_l. Same per-neuron /
                # population pooling rules as the upper-bound block.
                reg_l = 0.0
                for sc in net.last_spike_counts:
                    if fr_reg_mode == "population":
                        pop_mean = sc.mean()
                        n_neurons = sc.shape[-1]
                        reg_l = (
                            reg_l
                            + fr_reg_lower_strength
                            * n_neurons
                            * torch.relu(fr_reg_lower_theta - pop_mean) ** 2
                        )
                    else:
                        mean_z = sc.mean(dim=0)
                        reg_l = (
                            reg_l
                            + fr_reg_lower_strength
                            * (torch.relu(fr_reg_lower_theta - mean_z) ** 2).sum()
                        )
                loss = loss + reg_l
            opt.zero_grad()
            loss.backward()
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
            if not math.isfinite(gn_f):
                opt.zero_grad(set_to_none=True)
                n_skipped_steps += 1
            else:
                opt.step()
                net.project_dales()
                total_loss += loss.item()
                n_batches += 1
                grad_sum += gn_f
                n_grad += 1
            n_samples_train += y_b.size(0)
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
        # Accumulate per-cell rate (Hz) over the full test set so the
        # per-epoch metrics record reflects test-set means rather than
        # only the single-trial observation rate set by observe_epoch.
        test_rate_e_sum = 0.0
        test_rate_i_sum = 0.0
        eval_gen = torch.Generator().manual_seed(EVAL_SEED)
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                spk = encode_batch(X_b, dt, use_smnist, generator=eval_gen)
                logits_t = net(input_spikes=spk)
                test_loss_sum += loss_fn(logits_t, y_b).item()
                test_batches += 1
                correct += (logits_t.argmax(1) == y_b).sum().item()
                B = y_b.size(0)
                total += B
                # net.rates is set by _set_meta after every forward pass;
                # values are already per-cell Hz averaged over the batch.
                batch_rates = getattr(net, "rates", None) or {}
                for k, v in batch_rates.items():
                    if k.startswith("hid"):
                        test_rate_e_sum += float(v) * B
                    elif k.startswith("inh"):
                        test_rate_i_sum += float(v) * B

        eval_s = _time.perf_counter() - t_eval

        acc = 100.0 * correct / total
        avg_train = total_loss / max(n_batches, 1)
        avg_test = test_loss_sum / max(test_batches, 1)
        test_rate_e = test_rate_e_sum / total if total else 0.0
        test_rate_i = test_rate_i_sum / total if total else 0.0

        new_best = acc > best_acc
        if new_best:
            best_acc = acc
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        cur_lr = opt.param_groups[0]["lr"]
        elapsed = _time.perf_counter() - t_epoch

        if cur_lr != prev_lr:
            log.info(f"  ⚡ lr → {cur_lr:.0e}")
            prev_lr = cur_lr

        # Oscilloscope frame (returns metrics dict; merged into epoch record)
        t_observe = _time.perf_counter()
        epoch_metrics = None
        if observe:
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

        # Trainable-parameter Frobenius norms — surfaced per epoch so
        # convergence audits (nb024) can tell whether the optimiser is
        # still actively moving each named parameter. Mirrors the
        # structure of grad_ratios (one key per named parameter, scalar
        # value). All grads-requiring parameters included.
        weight_norms = {
            name: float(p.detach().norm().item())
            for name, p in net.named_parameters()
            if p.requires_grad
        }

        # Record this epoch into the structured metrics history
        record = {
            "ep": epoch + 1,
            "acc": acc,
            "loss": avg_train,
            "test_loss": avg_test,
            "test_rate_e": test_rate_e,
            "test_rate_i": test_rate_i,
            "lr": cur_lr,
            "elapsed_s": elapsed,
            "train_compute_s": train_compute_s,
            "eval_s": eval_s,
            "observe_s": observe_s,
            "samples": n_samples_train,
            "grad_norm": avg_grad,
            "grad_ratios": grad_ratios,
            "weight_norms": weight_norms,
            "skipped_steps": n_skipped_steps,
            "new_best": new_best,
        }
        if epoch_metrics:
            record.update(epoch_metrics)
        epoch_records.append(record)

        # jsonl sidecar — one line per epoch (skip dict fields, they
        # round-trip via metrics.json).
        jsonl.write(**{
            k: v for k, v in record.items()
            if k not in ("grad_ratios", "weight_norms")
        })

        # Structured progress line + warning tracker
        e_rate = epoch_metrics.get("rate_e", 0.0) if epoch_metrics else 0.0
        i_rate = epoch_metrics.get("rate_i") if epoch_metrics else None
        cv = epoch_metrics.get("cv", 0.0) if epoch_metrics else 0.0
        activity = epoch_metrics.get("act", 0.0) * 100 if epoch_metrics else 0.0
        flags = wtracker.tick(epoch + 1, acc, activity, avg_train)
        eta = (epochs - epoch - 1) * elapsed
        runlog.print_epoch(
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

    total_time = _time.perf_counter() - t_start

    # Tier 1 perf block. Per-epoch breakdown is in epoch_records
    # (train_compute_s / eval_s / observe_s); these are the whole-run
    # aggregates the dashboards read.
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
            "n_hidden": M.N_HID,
            "n_inh": M.N_INH,
            "n_in": M.N_IN,
            "max_samples": max_samples,
            "dataset": dataset,
            "v_grad_dampen": v_grad_dampen,
            "burn_in_ms": burn_in_ms,
            "batch_size": bs,
            "grad_clip": GRAD_CLIP,
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
    runlog.write_test_predictions(out_dir / "test_predictions.json", preds)

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
    runlog.print_summary(
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

