"""Study 15 — Modal GPU test — Poisson PING-SNN on MNIST.

Fork of study.14 with parameterized paths so it can run both locally
and on Modal serverless GPU.  Tiny subset by default for fast round-trip.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT

from pinglab.io.compiler import compile_graph, compile_graph_to_runtime
from pinglab.backends.pytorch import (
    prepare_runtime_tensors, run_batch, get_device, simulate_network, surrogate_lif_step,
)
from pinglab.io.graph_renderer import save_graph_diagram
from pinglab.io.training import encode_poisson, train_epoch, eval_epoch

from plots import (
    save_line, save_stacked_lines, save_raster_grid, save_raster_layers,
    save_confusion_matrix, save_input_raster, save_voltage_traces,
)


def main(
    artifacts_dir: Path | str | None = None,
    data_dir: Path | str | None = None,
    checkpoint_dir: Path | str | None = None,
    on_epoch_end: callable | None = None,
) -> dict:
    """Run training.  Returns the results dict.

    Parameters
    ----------
    artifacts_dir : path where plots / results.json / config.json are written.
        Default: ``ARTIFACTS_ROOT / <experiment folder name>``
    data_dir : path used for MNIST download cache.
        Default: ``<experiment_dir> / data``
    checkpoint_dir : path for saving/loading ``checkpoint.pt``.
        When set, training resumes from the last checkpoint if one exists,
        and a new checkpoint is written after every epoch.
    on_epoch_end : callable invoked after each epoch checkpoint save.
        Modal passes ``volume.commit`` here to flush the Volume.
    """
    device = get_device()
    print(f"Using {device} device")

    experiment_dir = Path(__file__).parent.resolve()

    if artifacts_dir is None:
        data_path = ARTIFACTS_ROOT / Path(__file__).parent.name
    else:
        data_path = Path(artifacts_dir)

    # Resolve checkpoint path and check for a resumable checkpoint.
    ckpt_path = Path(checkpoint_dir) / "checkpoint.pt" if checkpoint_dir else None
    resuming = ckpt_path is not None and ckpt_path.exists()

    # Only wipe artifacts on a fresh run — a checkpoint means prior epochs
    # already wrote partial artifacts we want to keep.
    if not resuming:
        if data_path.exists() and not data_path.is_symlink():
            shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    # ── tee stdout to a log file ──────────────────────────────────────────
    import io

    class Tee(io.TextIOBase):
        """Write to both a file and the original stream."""
        def __init__(self, stream, path):
            self._stream = stream
            self._file = open(path, "a")
        def write(self, data):
            self._stream.write(data)
            self._file.write(data)
            self._file.flush()
            return len(data)
        def flush(self):
            self._stream.flush()
            self._file.flush()
        def close(self):
            self._file.close()

    log_path = data_path / "train.log"
    if not resuming:
        log_path.unlink(missing_ok=True)
    sys.stdout = Tee(sys.__stdout__, log_path)
    sys.stderr = Tee(sys.__stderr__, log_path)

    if data_dir is None:
        mnist_root = experiment_dir / "data"
    else:
        mnist_root = Path(data_dir)

    config_path = experiment_dir / "config.json"
    with config_path.open(encoding="utf-8") as f:
        spec = json.load(f)
    shutil.copy2(config_path, data_path / "config.json")

    save_graph_diagram(spec, data_path / "graph")

    # ── network config ──────────────────────────────────────────────────────
    meta = spec.get("meta", {})
    batch_size        = int(meta.get("batch_size", 16))
    epochs            = int(meta.get("epochs", 5))
    lr                = float(meta.get("lr", 1e-3))
    input_scale       = float(meta.get("input_scale", 1.5))
    readout_mode      = str(meta.get("readout", "hybrid"))
    readout_alpha     = float(meta.get("readout_alpha", 0.01))
    subset_size       = meta.get("subset_size")       # None = full MNIST (60K)
    test_subset_size  = meta.get("test_subset_size")  # None = full test set (10K)
    if subset_size is not None:
        subset_size = int(subset_size)
    if test_subset_size is not None:
        test_subset_size = int(test_subset_size)

    sim_cfg = spec.get("sim", {})
    dt      = float(sim_cfg.get("dt_ms", 1.0))
    T_ms    = float(sim_cfg.get("T_ms", 100.0))
    T_steps = int(T_ms / dt)
    burn_in_ms    = float(sim_cfg.get("burn_in_ms", 0.0))
    burn_in_steps = int(burn_in_ms / dt)
    cm_backward_scale = float(meta.get("cm_backward_scale", 1.0))

    # ── compile graph ───────────────────────────────────────────────────────
    plan    = compile_graph(spec)
    pop_idx = plan["population_index"]
    out_start = int(pop_idx["E_out"]["start"])
    out_stop  = int(pop_idx["E_out"]["stop"])
    N_E       = int(plan["totals"]["N_E"])
    N_I       = int(plan["totals"]["N_I"])
    n_total   = N_E + N_I
    n_input   = int(pop_idx["E_in"]["stop"] - pop_idx["E_in"]["start"])

    print(f"Network: N_E={N_E}  N_I={N_I}  n_total={n_total}  E_in=[0,{n_input})  E_out=[{out_start},{out_stop})")
    print(f"Sim: T={T_ms}ms  dt={dt}ms  steps={T_steps}  burn_in={burn_in_ms}ms ({burn_in_steps} steps)")

    runtime = compile_graph_to_runtime(spec, backend="pytorch", trainable=True, device=device)

    # With cm_backward_scale > 1, the surrogate Jacobian tames the explosive
    # driving-force gradient, allowing BPTT through the spike buffer.  All
    # weight matrices can then receive gradient.
    use_surrogate_cm = cm_backward_scale > 1.0
    trainable_params = [runtime.weights.W_ee]
    param_counts = {"W_ee": runtime.weights.W_ee.numel()}
    if runtime.weights.W_ei is not None:
        if use_surrogate_cm:
            trainable_params.append(runtime.weights.W_ei)
            param_counts["W_ei"] = runtime.weights.W_ei.numel()
        else:
            param_counts["W_ei (frozen)"] = runtime.weights.W_ei.numel()
    if runtime.weights.W_ie is not None:
        if use_surrogate_cm:
            trainable_params.append(runtime.weights.W_ie)
            param_counts["W_ie"] = runtime.weights.W_ie.numel()
        else:
            param_counts["W_ie (frozen)"] = runtime.weights.W_ie.numel()
    total_params = sum(param_counts.values())
    n_trainable = sum(p.numel() for p in trainable_params)
    detach_spikes = not use_surrogate_cm
    print(f"Trainable params: {n_trainable:,}  total: {total_params:,}"
          f"  cm_backward_scale={cm_backward_scale}  detach_spikes={detach_spikes}")

    # Pre-build simulation state once — reset cheaply between batches.
    sim_state = prepare_runtime_tensors(runtime, training_mode=True, batch_size=batch_size)
    print(f"State pre-built: N={sim_state.N}  B={sim_state.batch_size}  buf_len={sim_state.buf_len}  device={device}")

    # ── data ────────────────────────────────────────────────────────────────
    train_data = MNIST(root=mnist_root, train=True,  download=True, transform=ToTensor())
    test_data  = MNIST(root=mnist_root, train=False, download=True, transform=ToTensor())

    if subset_size is not None:
        train_data = Subset(train_data, list(range(min(subset_size, len(train_data)))))
    if test_subset_size is not None:
        test_data = Subset(test_data, list(range(min(test_subset_size, len(test_data)))))

    print(f"Training on {len(train_data)} samples, evaluating on {len(test_data)} samples")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

    # ── optimizer ────────────────────────────────────────────────────────
    optimizer = Adam(trainable_params, lr=lr)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=1)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - 1)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[1])

    # ── gradient norm tracking via hooks ─────────────────────────────────
    _grad_norms: dict[str, list[float]] = {"W_ee": []}

    def _make_hook(name):
        def _hook(grad):
            _grad_norms[name].append(grad.norm().item())
        return _hook
    runtime.weights.W_ee.register_hook(_make_hook("W_ee"))
    if use_surrogate_cm:
        if runtime.weights.W_ei is not None:
            _grad_norms["W_ei"] = []
            runtime.weights.W_ei.register_hook(_make_hook("W_ei"))
        if runtime.weights.W_ie is not None:
            _grad_norms["W_ie"] = []
            runtime.weights.W_ie.register_hook(_make_hook("W_ie"))

    # ── forward closure (Poisson encoding + hybrid readout) ─────────────
    _batch_kwargs = dict(
        T_steps=T_steps, n_total=n_total, n_input=n_input,
        out_start=out_start, out_stop=out_stop,
        input_scale=input_scale, sim_state=sim_state,
        burn_in_steps=burn_in_steps, readout=readout_mode,
        readout_alpha=readout_alpha, encoding="poisson",
        cm_backward_scale=cm_backward_scale,
        detach_spikes=detach_spikes,
    )

    def forward(X: torch.Tensor) -> torch.Tensor:
        return run_batch(runtime, X, **_batch_kwargs)

    # Per-iteration firing rate tracking.  forward_with_rates() is used
    # during training; it stashes per-layer Hz after each batch.
    _iter_rates: dict[str, list[float]] = {
        "E_in": [], "E_hid": [], "E_out": [],
    }
    # Layer boundaries in the spike tensor [B, T, N_E] (E neurons only).
    _layer_slices = {
        "E_in":  (int(pop_idx["E_in"]["start"]),  int(pop_idx["E_in"]["stop"])),
        "E_hid": (int(pop_idx["E_hid"]["start"]), int(pop_idx["E_hid"]["stop"])),
        "E_out": (int(pop_idx["E_out"]["start"]), int(pop_idx["E_out"]["stop"])),
    }
    T_sec = T_steps * dt / 1000.0  # simulation window in seconds

    def forward_with_rates(X: torch.Tensor) -> torch.Tensor:
        logits, spikes = run_batch(runtime, X, **_batch_kwargs, return_spikes=True)
        # spikes: [B, T, N_E] — compute mean firing rate per layer
        with torch.no_grad():
            for name, (s, e) in _layer_slices.items():
                # mean spikes per neuron per sample, converted to Hz
                rate = spikes[:, burn_in_steps:, s:e].sum(dim=1).mean().item() / T_sec
                _iter_rates[name].append(rate)
        return logits

    # ── training loop ───────────────────────────────────────────────────────
    import time
    test_losses, test_accuracies = [], []
    all_iter_losses: list[float] = []
    all_iter_accs: list[float] = []
    start_epoch = 0
    prior_elapsed = 0.0

    # ── resume from checkpoint if available ──────────────────────────────
    if resuming:
        print(f"\nResuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Restore weights into the live runtime
        runtime.weights.W_ee.data.copy_(ckpt["W_ee"])
        if "W_ei" in ckpt and runtime.weights.W_ei is not None:
            runtime.weights.W_ei.data.copy_(ckpt["W_ei"])
        if "W_ie" in ckpt and runtime.weights.W_ie is not None:
            runtime.weights.W_ie.data.copy_(ckpt["W_ie"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        test_losses = ckpt.get("test_losses", [])
        test_accuracies = ckpt.get("test_accuracies", [])
        all_iter_losses = ckpt.get("all_iter_losses", [])
        all_iter_accs = ckpt.get("all_iter_accs", [])
        for name in _iter_rates:
            _iter_rates[name] = ckpt.get(f"rates_{name}", [])
        for name in _grad_norms:
            _grad_norms[name] = ckpt.get(f"grad_norm_{name}", [])
        prior_elapsed = ckpt.get("elapsed_seconds", 0.0)
        print(f"  Restored epoch {start_epoch}/{epochs}, "
              f"{len(all_iter_losses)} iter metrics, "
              f"{len(test_losses)} epoch metrics, "
              f"{prior_elapsed:.0f}s prior elapsed")

    run_start = time.perf_counter()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}", flush=True)
        train_loss, iter_losses, iter_accs = train_epoch(
            train_loader, optimizer, forward_with_rates,
            n_total_samples=len(train_data),
            device=device,
            max_grad_norm=float(meta.get("max_grad_norm", 1.0)),
            per_param_clip=use_surrogate_cm,
        )
        all_iter_losses.extend(iter_losses)
        all_iter_accs.extend(iter_accs)

        train_elapsed = time.perf_counter() - epoch_start
        print(f"  train done in {train_elapsed:.1f}s  avg loss: {train_loss:.4f}", flush=True)

        test_loss, acc = eval_epoch(
            test_loader, forward,
            device=device,
        )
        test_losses.append(test_loss)
        test_accuracies.append(acc)
        scheduler.step()
        epoch_elapsed = time.perf_counter() - epoch_start
        total_elapsed = time.perf_counter() - run_start
        epochs_left = epochs - (epoch + 1)
        eta = epochs_left * epoch_elapsed
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  EPOCH {epoch+1}/{epochs}  train_loss={train_loss:.4f}"
            f"  test_loss={test_loss:.4f}  acc={100*acc:.1f}%"
            f"  lr={current_lr:.2e}"
            f"  epoch_time={epoch_elapsed:.0f}s  ETA={eta:.0f}s  total={total_elapsed:.0f}s",
            flush=True,
        )

        # ── per-epoch checkpoint ────────────────────────────────────────────
        if ckpt_path is not None:
            ckpt_data = {
                "epoch": epoch + 1,
                "W_ee": runtime.weights.W_ee.detach().cpu(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "test_losses": test_losses,
                "test_accuracies": test_accuracies,
                "all_iter_losses": all_iter_losses,
                "all_iter_accs": all_iter_accs,
                **{f"rates_{k}": v for k, v in _iter_rates.items()},
                **{f"grad_norm_{k}": v for k, v in _grad_norms.items()},
                "elapsed_seconds": prior_elapsed + (time.perf_counter() - run_start),
            }
            if runtime.weights.W_ei is not None:
                ckpt_data["W_ei"] = runtime.weights.W_ei.detach().cpu()
            if runtime.weights.W_ie is not None:
                ckpt_data["W_ie"] = runtime.weights.W_ie.detach().cpu()
            tmp_ckpt = ckpt_path.with_suffix(".pt.tmp")
            torch.save(ckpt_data, tmp_ckpt)
            tmp_ckpt.rename(ckpt_path)  # atomic on POSIX
            print(f"  Checkpoint saved → {ckpt_path}", flush=True)
            if on_epoch_end is not None:
                on_epoch_end()

    # ── clean up checkpoint (training completed successfully) ───────────────
    if ckpt_path is not None and ckpt_path.exists():
        ckpt_path.unlink()
        print("Checkpoint removed (training complete).")

    # ── save weights ────────────────────────────────────────────────────────
    mnist_root.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "W_ee": runtime.weights.W_ee.detach(),
    }
    if runtime.weights.W_ei is not None:
        checkpoint["W_ei"] = runtime.weights.W_ei.detach()
    if runtime.weights.W_ie is not None:
        checkpoint["W_ie"] = runtime.weights.W_ie.detach()
    torch.save(checkpoint, mnist_root / "weights.pth")

    # ── per-class accuracy (on final weights) ────────────────────────────────
    class_correct = [0] * 10
    class_total = [0] * 10
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            pred = forward(X)
            predicted = pred.argmax(1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            for p, t in zip(predicted.cpu(), y.cpu()):
                class_total[t.item()] += 1
                if p.item() == t.item():
                    class_correct[t.item()] += 1

    class_accs = [100 * c / max(t, 1) for c, t in zip(class_correct, class_total)]
    print("\nPer-class accuracy:")
    for d in range(10):
        print(f"  digit {d}: {class_accs[d]:5.1f}%  ({class_correct[d]}/{class_total[d]})")

    # ── raster plots ─────────────────────────────────────────────────────────
    # Gather one canonical example per digit class from the test set.
    print("\nCollecting canonical samples for raster plots...", flush=True)
    canonical: dict[int, torch.Tensor] = {}
    for X_batch, y_batch in test_loader:
        for img, lbl in zip(X_batch, y_batch):
            d = lbl.item()
            if d not in canonical:
                canonical[d] = img
        if len(canonical) == 10:
            break

    # Run each canonical sample through the trained network (no grad).
    # Use unbatched simulate_network (no spike_fn) to capture I neuron spikes
    # and voltage traces.
    out_spike_tensors = []
    full_spike_arrays = {}
    voltage_traces = {}
    input_arrays = {}

    def _spikes_to_array(result, T_steps: int, n_total: int, dt: float) -> np.ndarray:
        """Build [T, N_total] binary spike array from SimulationResult.spikes."""
        arr = np.zeros((T_steps, n_total), dtype=np.float32)
        times = result.spikes.times
        ids = result.spikes.ids
        for t_ms, nid in zip(times, ids):
            step = int(t_ms / dt)
            if 0 <= step < T_steps and 0 <= nid < n_total:
                arr[step, nid] = 1.0
        return arr

    available_digits = sorted(canonical.keys())
    if len(available_digits) < 10:
        print(f"  Warning: only found canonical samples for digits {available_digits}")

    print(f"\nInput scale: {input_scale:.1f} (fixed)")

    with torch.no_grad():
        for d in available_digits:
            img = canonical[d]
            ext = encode_poisson(
                img, T_steps=T_steps, n_total=n_total, n_input=n_input, scale=input_scale,
            )  # [T, N] — unbatched, stochastic
            input_arrays[d] = ext.numpy()
            result = simulate_network(
                runtime,
                external_input=ext,
            )
            full_arr = _spikes_to_array(result, T_steps, n_total, dt)
            out_spike_tensors.append(full_arr[:, out_start:out_stop])
            full_spike_arrays[d] = full_arr
            voltage_traces[d] = result.voltage_trace

            # Diagnostic: count spikes per layer
            n_out_spikes = int(full_arr[:, out_start:out_stop].sum())
            n_hid_spikes = int(full_arr[:, pop_idx["E_hid"]["start"]:pop_idx["E_hid"]["stop"]].sum())
            n_in_spikes = int(input_arrays[d][:, :n_input].sum())
            print(f"  digit {d}: input_spikes={n_in_spikes}  E_hid_spikes={n_hid_spikes}  E_out_spikes={n_out_spikes}")

    # 1. 2×5 output-layer raster grid — one panel per available digit
    save_raster_grid(
        data_path / "raster_output_all_all",
        out_spike_tensors,
        dt=dt,
        digit_labels=available_digits,
        suptitle="Output layer spikes per digit class (trained PING)",
    )

    # 2. Layer raster for each digit (E_hid + I_global + E_out)
    for d in available_digits:
        save_raster_layers(
            data_path / f"raster_layers_digit_{d:02d}",
            full_spike_arrays[d],
            dt=dt,
            pop_idx=pop_idx,
            input_ext=input_arrays[d],
            n_input=n_input,
            title=f"All layers — digit {d}",
        )

    # 3. Input rasters — Poisson spike trains for each digit
    for d in available_digits:
        save_input_raster(
            data_path / f"raster_input_digit_{d:02d}",
            input_arrays[d],
            dt=dt,
            n_input=n_input,
            title=f"Poisson input — digit {d}",
        )

    # 4. Output neuron voltage traces for each digit
    v_th = float(spec.get("biophysics", {}).get("V_th", -50.0))
    for d in available_digits:
        save_voltage_traces(
            data_path / f"voltage_output_digit_{d:02d}",
            voltage_traces[d],
            dt=dt,
            out_start=out_start,
            out_stop=out_stop,
            title=f"Output neuron voltages — digit {d}",
            v_th=v_th,
        )

    # ── plots ────────────────────────────────────────────────────────────────
    iters_x = list(range(1, len(all_iter_losses) + 1))
    epochs_x = list(range(1, epochs + 1))
    save_line(data_path / "loss_train", x=iters_x, y=all_iter_losses,
              title="Train Loss", xlabel="Iteration", ylabel="Cross-entropy loss")
    save_line(data_path / "loss_test",  x=epochs_x, y=test_losses,
              title="Test Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    save_line(data_path / "accuracy",   x=iters_x, y=[a * 100 for a in all_iter_accs],
              title="Train Accuracy", xlabel="Iteration", ylabel="Accuracy (%)")

    # Firing rates per layer over iterations
    if _iter_rates["E_in"]:
        save_stacked_lines(
            data_path / "firing_rates",
            x=iters_x,
            series={k: v for k, v in _iter_rates.items() if v},
            suptitle="Mean Firing Rate per Layer",
            xlabel="Iteration",
            ylabel="Hz",
        )

    # Gradient norms per weight matrix over iterations
    if _grad_norms.get("W_ee"):
        save_stacked_lines(
            data_path / "grad_norms",
            x=iters_x,
            series={k: v for k, v in _grad_norms.items() if v},
            suptitle="Gradient Norm per Weight Matrix",
            xlabel="Iteration",
            ylabel="||grad||",
        )

    save_line(
        data_path / "accuracy_per_class",
        x=list(range(10)),
        y=class_accs,
        title="Per-class Test Accuracy",
        xlabel="Digit",
        ylabel="Accuracy (%)",
    )

    save_confusion_matrix(
        data_path / "confusion",
        all_labels,
        all_preds,
        title="Confusion Matrix",
    )

    # ── save results ──────────────────────────────────────────────────────
    import uuid
    total_elapsed = prior_elapsed + (time.perf_counter() - run_start)
    results = {
        "run_id": uuid.uuid4().hex[:8],
        "epochs": epochs,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "final_train_loss": round(all_iter_losses[-1], 4),
        "final_test_loss": round(test_losses[-1], 4),
        "best_test_loss": round(min(test_losses), 4),
        "best_test_loss_epoch": int(test_losses.index(min(test_losses)) + 1),
        "final_test_accuracy": round(100 * test_accuracies[-1], 1),
        "best_test_accuracy": round(100 * max(test_accuracies), 1),
        "best_test_accuracy_epoch": int(test_accuracies.index(max(test_accuracies)) + 1),
        "test_losses_per_epoch": [round(x, 4) for x in test_losses],
        "test_accuracies_per_epoch": [round(100 * x, 1) for x in test_accuracies],
        "encoding": "poisson",
        "readout": readout_mode,
        "readout_alpha": readout_alpha,
        "input_scale": input_scale,
        "trainable_params": n_trainable,
        "trainable_params_breakdown": param_counts,
        "cm_backward_scale": cm_backward_scale,
        "detach_spikes": detach_spikes,
        "elapsed_seconds": round(total_elapsed, 1),
        "device": str(device),
        "runtime": "modal" if os.environ.get("MODAL_IS_REMOTE") else "local",
    }
    with open(data_path / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Done!")
    return results


if __name__ == "__main__":
    main()
