"""Study 9 — Trainable SNN on MNIST via surrogate gradients.

Three feedforward E populations (784 → 256 → 10), no I neurons.
Trained with BPTT through surrogate spike function using pinglab's
simulate_network with spike_fn=surrogate_lif_step.
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from torch.optim import Adam
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
from pinglab.io.training import encode_rate, train_epoch, eval_epoch

from plots import save_line, save_raster_grid, save_raster_layers


def main() -> None:
    device = get_device()
    print(f"Using {device} device")

    experiment_dir = Path(__file__).parent.resolve()
    data_path = ARTIFACTS_ROOT / Path(__file__).parent.name
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    config_path = experiment_dir / "config.json"
    with config_path.open(encoding="utf-8") as f:
        spec = json.load(f)
    shutil.copy2(config_path, data_path / "config.json")

    # ── network config ──────────────────────────────────────────────────────
    meta = spec.get("meta", {})
    batch_size        = int(meta.get("batch_size", 16))
    epochs            = int(meta.get("epochs", 5))
    lr                = float(meta.get("lr", 1e-3))
    input_scale       = float(meta.get("input_scale", 1.5))
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

    # ── compile graph ───────────────────────────────────────────────────────
    plan    = compile_graph(spec)
    pop_idx = plan["population_index"]
    out_start = int(pop_idx["E_out"]["start"])
    out_stop  = int(pop_idx["E_out"]["stop"])
    n_total   = int(plan["totals"]["N_E"])  # N_I = 0
    n_input   = int(pop_idx["E_in"]["stop"] - pop_idx["E_in"]["start"])

    print(f"Network: N_E={n_total}  E_in=[0,{n_input})  E_out=[{out_start},{out_stop})")
    print(f"Sim: T={T_ms}ms  dt={dt}ms  steps={T_steps}")

    runtime = compile_graph_to_runtime(spec, backend="pytorch", trainable=True, device=device)
    print(f"W_ee trainable params: {runtime.weights.W_ee.numel():,}")

    # Pre-build simulation state once — reset cheaply between batches.
    sim_state = prepare_runtime_tensors(runtime, training_mode=True, batch_size=batch_size)
    print(f"State pre-built: N={sim_state.N}  B={sim_state.batch_size}  buf_len={sim_state.buf_len}  device={device}")

    # ── data ────────────────────────────────────────────────────────────────
    train_data = MNIST(root=experiment_dir / "data", train=True,  download=True, transform=ToTensor())
    test_data  = MNIST(root=experiment_dir / "data", train=False, download=True, transform=ToTensor())

    if subset_size is not None:
        train_data = Subset(train_data, list(range(min(subset_size, len(train_data)))))
    if test_subset_size is not None:
        test_data = Subset(test_data, list(range(min(test_subset_size, len(test_data)))))

    print(f"Training on {len(train_data)} samples, evaluating on {len(test_data)} samples")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False)

    # ── optimizer ───────────────────────────────────────────────────────────
    optimizer = Adam([runtime.weights.W_ee], lr=lr)

    # ── forward closure ─────────────────────────────────────────────────────
    # Captures all SNN details so train_epoch / eval_epoch stay generic.
    def forward(X: torch.Tensor) -> torch.Tensor:
        return run_batch(
            runtime, X,
            T_steps=T_steps, n_total=n_total, n_input=n_input,
            out_start=out_start, out_stop=out_stop,
            input_scale=input_scale,
            sim_state=sim_state,
        )

    # ── training loop ───────────────────────────────────────────────────────
    import time
    test_losses, test_accuracies = [], []
    all_iter_losses: list[float] = []
    all_iter_accs: list[float] = []
    run_start = time.perf_counter()

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}", flush=True)
        train_loss, iter_losses, iter_accs = train_epoch(
            train_loader, optimizer, forward,
            n_total_samples=len(train_data),
            device=device,
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
        epoch_elapsed = time.perf_counter() - epoch_start
        total_elapsed = time.perf_counter() - run_start
        epochs_left = epochs - (epoch + 1)
        eta = epochs_left * epoch_elapsed
        print(
            f"  EPOCH {epoch+1}/{epochs}  train_loss={train_loss:.4f}"
            f"  test_loss={test_loss:.4f}  acc={100*acc:.1f}%"
            f"  epoch_time={epoch_elapsed:.0f}s  ETA={eta:.0f}s  total={total_elapsed:.0f}s",
            flush=True,
        )

    # ── save weights ────────────────────────────────────────────────────────
    (experiment_dir / "data").mkdir(exist_ok=True)
    torch.save({"W_ee": runtime.weights.W_ee.detach()}, experiment_dir / "data" / "weights.pth")

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
    out_spike_tensors = []   # [T, n_out] per digit for the output-grid plot
    full_spike_tensors = {}  # digit -> [T, N_E] for the layers plot

    with torch.no_grad():
        for d in range(10):
            img = canonical[d]
            ext = encode_rate(
                img, T_steps=T_steps, n_total=n_total, n_input=n_input, scale=input_scale,
            ).unsqueeze(0)  # [1, T, N]
            _, spikes = simulate_network(
                runtime,
                external_input=ext,
                spike_fn=surrogate_lif_step,
                return_spike_tensor=True,
            )
            s = spikes[0].cpu()  # [T, N_E]
            out_spike_tensors.append(s[:, out_start:out_stop])  # [T, 10]
            full_spike_tensors[d] = s

    # 1. 2×5 output-layer raster grid — one panel per digit
    save_raster_grid(
        data_path / "raster_output_all_all",
        out_spike_tensors,
        dt=dt,
        suptitle="Output layer spikes per digit class (trained)",
    )

    # 2. Layer raster for each digit (hidden + output)
    pop_idx = plan["population_index"]
    for d in range(10):
        save_raster_layers(
            data_path / f"raster_layers_digit_{d:02d}",
            full_spike_tensors[d],
            dt=dt,
            pop_idx=pop_idx,
            title=f"Hidden + output layer — digit {d}",
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

    print("Done!")


if __name__ == "__main__":
    main()
