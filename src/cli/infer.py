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
from config import build_net, setup_model_globals, save_snapshot_npz
from datasets import DATASET_N_HIDDEN_DEFAULTS, load_dataset
from encoders import EVAL_SEED, encode_batch
from scan import _auto_device, primary_hid_key, primary_inh_key
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
    emit_per_cell_rates=False,
):
    """Run inference with saved weights at a given dt."""

    # Seed before dataset load and model init (matters when load_weights is None
    # and any randomness remains in the path).
    seed_everything(seed)

    # ─────────────────────────────────────────────────────────────────────────
    # M MODULE GLOBALS INITIALIZATION (same as in train.py)
    # ─────────────────────────────────────────────────────────────────────────
    # Must initialize M globals before building/loading the network.
    # ─────────────────────────────────────────────────────────────────────────

    M.T_ms = t_ms  # Total simulation time (ms)

    if hidden_sizes is None:
        default = DATASET_N_HIDDEN_DEFAULTS.get(dataset, 256)
        hidden_sizes = [default]
        log.info(f"  n_hidden auto → {hidden_sizes} (smart default for {dataset})")

    # Initialize M module globals before building/loading network
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

    # E1: per-cell rate emission. When requested, turn on spike recording and
    # accumulate per-cell spike counts (summed over time and batch) for the
    # primary E and I layers. Notebooks that need the per-cell rate distribution
    # (rather than the population mean) consume per_cell_rates.npz instead of
    # rebuilding the net in-process. Recording is off by default so the common
    # accuracy-only path stays cheap.
    if emit_per_cell_rates:
        net.recording = True
    per_cell_e = None  # running (N_E,) spike-count sum across the test set
    per_cell_i = None  # running (N_I,) spike-count sum across the test set

    def _accum_per_cell(rec, key, acc):
        """Sum a recorded raster over time (and batch) to per-cell counts."""
        r = rec.get(key) if key else None
        if r is None:
            return acc
        # (T, B, N) → sum over T and B; (T, N) at B=1 → sum over T.
        cnt = r.sum(dim=(0, 1)) if r.ndim == 3 else r.sum(dim=0)
        cnt = cnt.detach().cpu().numpy()
        return cnt if acc is None else acc + cnt

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
            if emit_per_cell_rates:
                rec = net.spike_record
                per_cell_e = _accum_per_cell(rec, primary_hid_key(rec), per_cell_e)
                per_cell_i = _accum_per_cell(rec, primary_inh_key(rec), per_cell_i)

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

    # E1: write per-cell rate arrays if requested. rate = total spikes per cell
    # over the test set / (n_trials × physical trial time in seconds) → Hz.
    if emit_per_cell_rates and out_dir_path and out_dir_path.exists():
        import numpy as np

        t_sec = float(M.T_ms) / 1000.0
        denom = total * t_sec if total else 1.0
        dump = {}
        if per_cell_e is not None:
            dump["rate_e_per_cell"] = (per_cell_e / denom).astype(np.float32)
        if per_cell_i is not None:
            dump["rate_i_per_cell"] = (per_cell_i / denom).astype(np.float32)
        out_npz = out_dir_path / "per_cell_rates.npz"
        np.savez(out_npz, **dump)
        log.info(f"  → {out_npz}  ({len(dump)} arrays)")

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
    sample_index=None,
):
    """Run inference on a single sample and save full spike trajectory to snapshot.npz.

    Sample selection: by default the trial is chosen by (digit class, index within
    class) for MNIST, or raw index for other datasets. Pass sample_index to force a
    raw test-set index regardless of dataset — notebooks that grab "test trial N"
    (rather than "digit D instance S") use this to reproduce a specific raster.
    """
    import numpy as np

    # Seed RNG for reproducibility of inference results
    seed_everything(seed)

    # Initialize M module globals (same setup as infer() above)
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

    # Select single sample. A raw sample_index overrides digit-class selection for
    # any dataset; otherwise MNIST selects by (digit class, index within class) and
    # other datasets use the raw index.
    if sample_index is not None:
        sample_idx = sample_index if 0 <= sample_index < len(X_te) else 0
    elif dataset == "mnist":
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

    # Run forward pass with spike recording enabled
    # Recording captures all spike trains and intermediate state (voltages, conductances)
    net.recording = True
    net.eval()  # Disable dropout, batch norm, etc.
    with torch.no_grad():
        # Encode pixel data as Poisson spike train using EVAL_SEED for determinism
        spk = encode_batch(X_single, dt, dataset == "smnist")
        logits = net(input_spikes=spk)

    # Extract and save spike recording to NPZ file for notebook analysis
    # The recording dict contains spike tensors keyed by layer ('hid', 'inh', etc.),
    # plus optional traces (voltages, conductances, etc.) if the network recorded them.
    rec = net.spike_record
    out_path = Path(out_dir) / "snapshot.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save snapshot with canonical field names and metadata
    # (Uses shared utility to ensure all snapshot fields are consistent across code paths)
    save_snapshot_npz(out_path, rec, dt, M.N_HID, M.N_INH)

    return {"acc": None}


def dump_weights(
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
    readout_mode="rate",
    trainable_w_ei=False,
    trainable_w_ie=False,
    kaiming_init=False,
):
    """Emit initialisation and trained weight matrices to weights_dump.npz.

    Rebuilds the network under its training seed to recover the deterministic
    INIT weights (build_net with randomize_init is seed-reproducible), then reads
    the trained values straight out of the saved state_dict. Lets notebooks
    compare init-vs-trained anatomical weights without importing build_net.

    For every fixed E-I weight matrix (W_ei, W_ie, W_ee, W_ii) and layer key k,
    emits two arrays: <name>_<k>_init and <name>_<k>_trained. E.g. a single
    E-I-layer PING gives W_ei_1_init, W_ei_1_trained, W_ie_1_init, W_ie_1_trained.

    Args mirror infer() so the CLI maps --load-config → this the same way.
    """
    import numpy as np

    # Seed BEFORE build_net so the randomized init is byte-identical to training's
    # init (this is the whole point — reproduce the pre-training weights).
    seed_everything(seed)

    M.T_ms = t_ms
    M.dt = dt
    M.T_steps = int(M.T_ms / M.dt)
    if hidden_sizes is None:
        default = DATASET_N_HIDDEN_DEFAULTS.get(dataset, 256)
        hidden_sizes = [default]
    setup_model_globals(hidden_sizes)
    if dataset in ("mnist", "smnist"):
        M.N_IN = 28 if dataset == "smnist" else 784
    else:
        M.N_IN = 64

    device = _auto_device()
    net = build_net(
        model_name,
        w_in=w_in,
        w_in_sparsity=w_in_sparsity,
        ei_strength=ei_strength,
        ei_ratio=ei_ratio,
        device=device,
        # kaiming cells are already heterogeneous; everything else needs the
        # seeded symmetry-break, exactly as train/infer do.
        randomize_init=not kaiming_init,
        dales_law=dales_law,
        hidden_sizes=hidden_sizes,
        ei_layers=ei_layers,
        readout_mode=readout_mode,
        trainable_w_ei=trainable_w_ei,
        trainable_w_ie=trainable_w_ie,
    )

    _WEIGHT_DICTS = ("W_ei", "W_ie", "W_ee", "W_ii")

    # INIT weights: read straight off the freshly built (untrained) net.
    dump: dict[str, "np.ndarray"] = {}
    for name in _WEIGHT_DICTS:
        pdict = getattr(net, name, None)
        if pdict is None:
            continue
        for k, w in pdict.items():
            dump[f"{name}_{k}_init"] = w.detach().cpu().numpy()

    # TRAINED weights: pulled from the saved state_dict (keys look like "W_ei.1").
    assert load_weights is not None, "dump_weights requires --load-weights"
    state = torch.load(load_weights, map_location="cpu")
    for sk, sv in state.items():
        name, _, key = sk.partition(".")
        if name in _WEIGHT_DICTS and key:
            arr = sv.detach().cpu().numpy() if hasattr(sv, "detach") else np.asarray(sv)
            dump[f"{name}_{key}_trained"] = arr

    out_path = Path(out_dir) / "weights_dump.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **dump)
    log.info(f"  → {out_path}  ({len(dump)} arrays)")

    return {"n_arrays": len(dump), "path": str(out_path)}

