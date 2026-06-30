"""Inference driver for the CLI.

Replays a trained network over a held-out dataset split, recomputes test
accuracy and rates, and optionally writes per-sample predictions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

import models as M
from config import build_net, setup_model_globals
from datasets import DATASET_N_HIDDEN_DEFAULTS, load_dataset
from encoders import EVAL_SEED, encode_batch
from scan import _auto_device
from train import seed_everything

log = logging.getLogger("cli")


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
    w_in_sparsity=0.0,
    hidden_sizes=None,
    out_dir=None,
    dales_law=True,
    encode_fn=None,
    ei_layers=None,
    seed=None,
):
    """Run inference with saved weights at a given dt."""

    # Seed before dataset load and model init (matters when load_weights is None
    # and any randomness remains in the path).
    seed_everything(seed)

    # Setup dt (constants are computed locally in model.forward)
    M.T_ms = t_ms

    if hidden_sizes is None:
        default = DATASET_N_HIDDEN_DEFAULTS.get(dataset, 256)
        hidden_sizes = [default]
        log.info(f"  n_hidden auto → {hidden_sizes} (smart default for {dataset})")
    setup_model_globals(hidden_sizes)

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
    net = build_net(
        model_name,
        w_in=w_in,
        w_in_sparsity=w_in_sparsity,
        ei_strength=ei_strength,
        ei_ratio=ei_ratio,
        device=device,
        randomize_init=True,
        dales_law=dales_law,
        hidden_sizes=hidden_sizes,
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


def infer_and_snapshot(
    model_name="ping",
    dt=0.25,
    load_weights=None,
    dataset="scikit",
    t_ms=200.0,
    w_in=None,
    ei_strength=0.5,
    ei_ratio=2.0,
    w_in_sparsity=0.0,
    hidden_sizes=None,
    out_dir=None,
    dales_law=True,
    ei_layers=None,
    seed=None,
    digit=0,
    sample=0,
):
    """Run inference on a single sample and save full spike trajectory to snapshot.npz."""
    import numpy as np
    from scan import primary_hid_key, primary_inh_key

    seed_everything(seed)

    M.T_ms = t_ms
    if hidden_sizes is None:
        default = DATASET_N_HIDDEN_DEFAULTS.get(dataset, 256)
        hidden_sizes = [default]
    setup_model_globals(hidden_sizes)

    device = _auto_device()

    # Load dataset
    _, X_te, _, y_te = load_dataset(dataset, max_samples=None, split=True)
    if dataset in ("mnist", "smnist"):
        if dataset == "smnist":
            M.N_IN = 28
            M.T_ms = 28 * 10.0
            M.T_steps = int(M.T_ms / dt)
        else:
            M.N_IN = 784
    else:
        M.N_IN = 64

    # Select single sample (by digit class and index within that class)
    if dataset == "mnist":
        digit_mask = (y_te == digit)
        digit_indices = np.where(digit_mask)[0]
        if sample >= len(digit_indices):
            log.warning(
                f"Sample {sample} out of range for digit {digit} "
                f"({len(digit_indices)} available); using sample 0"
            )
            sample = 0
        sample_idx = digit_indices[sample]
    else:
        sample_idx = sample if sample < len(X_te) else 0

    X_single = torch.from_numpy(X_te[sample_idx : sample_idx + 1]).to(device)
    y_single = torch.from_numpy(y_te[sample_idx : sample_idx + 1]).to(device)

    # Build model
    net = build_net(
        model_name,
        w_in=w_in,
        w_in_sparsity=w_in_sparsity,
        ei_strength=ei_strength,
        ei_ratio=ei_ratio,
        device=device,
        randomize_init=True,
        dales_law=dales_law,
        hidden_sizes=hidden_sizes,
        ei_layers=ei_layers,
    )

    # Load weights
    assert load_weights is not None
    state = torch.load(load_weights, map_location=device)
    net.load_state_dict(state, strict=False)

    # Run forward pass with recording
    net.recording = True
    net.eval()
    with torch.no_grad():
        spk = encode_batch(X_single, dt, dataset == "smnist")
        logits = net(input_spikes=spk)

    # Extract spike data
    rec = net.spike_record
    spk_e = rec[primary_hid_key(rec)].cpu().numpy()
    spk_i_key = primary_inh_key(rec)
    spk_i = (
        rec[spk_i_key].cpu().numpy()
        if spk_i_key
        else np.zeros((spk_e.shape[0], 0), dtype=np.float32)
    )

    # Save snapshot
    out_path = Path(out_dir) / "snapshot.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    npz_data = {
        "spk_e": spk_e,
        "spk_i": spk_i,
        "dt": np.float32(dt),
        "n_e": np.int32(M.N_HID),
        "n_i": np.int32(M.N_INH),
    }

    # Include optional data like Lyapunov divergence if present
    for key in ["lyap_dist", "lyap_t_ms"]:
        if key in rec:
            v = rec[key]
            npz_data[key] = v.cpu().numpy() if hasattr(v, "cpu") else v

    np.savez(out_path, **npz_data)
    log.info(f"Saved snapshot to {out_path}")

    return {"acc": None}

