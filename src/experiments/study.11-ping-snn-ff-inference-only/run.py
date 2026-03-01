"""Study 11 — PING rhythms in an inference-only SNN classifier.

Extends study.10's trained E-only feedforward network (784→64→10) with a
global inhibitory population (I_global, 50 neurons) forming a PING loop
with E_hid. Loads W_ee from study.9 and evaluates classification on MNIST
while observing E↔I rhythmic dynamics.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT

from pinglab.io.compiler import compile_graph, compile_graph_to_runtime
from pinglab.backends.pytorch import (
    get_device, prepare_runtime_tensors, run_batch, simulate_network, surrogate_lif_step,
)
from pinglab.io.training import encode_rate, eval_epoch

from plots import save_line, save_raster_grid, save_raster_layers, save_confusion_matrix


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
    batch_size       = int(meta.get("batch_size", 16))
    input_scale      = float(meta.get("input_scale", 3.0))
    test_subset_size = meta.get("test_subset_size")
    weights_source   = str(meta.get("weights_source", "study.9-ai-snn-ff"))
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
    N_E       = int(plan["totals"]["N_E"])
    N_I       = int(plan["totals"]["N_I"])
    n_total   = N_E + N_I
    n_input   = int(pop_idx["E_in"]["stop"] - pop_idx["E_in"]["start"])

    print(f"Network: N_E={N_E}  N_I={N_I}  n_total={n_total}  E_in=[0,{n_input})  E_out=[{out_start},{out_stop})")
    print(f"Sim: T={T_ms}ms  dt={dt}ms  steps={T_steps}")

    runtime = compile_graph_to_runtime(spec, backend="pytorch", device=device)

    # ── load trained weights ────────────────────────────────────────────────
    weights_dir = experiment_dir.parent / weights_source / "data"
    weights_path = weights_dir / "weights.pth"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Trained weights not found at {weights_path}. Run study.9 first."
        )
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    runtime.weights.W_ee.copy_(checkpoint["W_ee"])
    print(f"Loaded trained W_ee from {weights_path}  ({runtime.weights.W_ee.numel():,} params)")

    # Pre-build simulation state for batched inference.
    # training_mode=True forces the simple-buffer path which supports batched input;
    # the delay-override path (training_mode=False) uses unbatched buffers.
    sim_state = prepare_runtime_tensors(runtime, training_mode=True, batch_size=batch_size)

    # ── data ────────────────────────────────────────────────────────────────
    # Use study.9's cached MNIST data if available, otherwise download fresh.
    mnist_dir = experiment_dir.parent / weights_source / "data"
    if not (mnist_dir / "MNIST").exists():
        mnist_dir = experiment_dir / "data"
    test_data = MNIST(root=mnist_dir, train=False, download=True, transform=ToTensor())
    if test_subset_size is not None:
        test_data = Subset(test_data, list(range(min(test_subset_size, len(test_data)))))
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print(f"Evaluating on {len(test_data)} test samples")

    # ── forward closure ─────────────────────────────────────────────────────
    def forward(X: torch.Tensor) -> torch.Tensor:
        return run_batch(
            runtime, X,
            T_steps=T_steps, n_total=n_total, n_input=n_input,
            out_start=out_start, out_stop=out_stop,
            input_scale=input_scale,
            sim_state=sim_state,
        )

    # ── evaluation ──────────────────────────────────────────────────────────
    import time
    run_start = time.perf_counter()

    with torch.no_grad():
        test_loss, test_acc = eval_epoch(test_loader, forward, device=device)

    elapsed = time.perf_counter() - run_start
    print(f"\nTest loss: {test_loss:.4f}  accuracy: {100*test_acc:.1f}%  ({elapsed:.1f}s)")

    # ── per-class accuracy ──────────────────────────────────────────────────
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
    print("\nCollecting canonical samples for raster plots...", flush=True)
    canonical: dict[int, torch.Tensor] = {}
    for X_batch, y_batch in test_loader:
        for img, lbl in zip(X_batch, y_batch):
            d = lbl.item()
            if d not in canonical:
                canonical[d] = img
        if len(canonical) == 10:
            break

    out_spike_tensors = []
    full_spike_arrays = {}

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

    with torch.no_grad():
        for d in range(10):
            img = canonical[d]
            ext = encode_rate(
                img, T_steps=T_steps, n_total=n_total, n_input=n_input, scale=input_scale,
            )  # [T, N] — unbatched; no spike_fn so spikes are recorded
            result = simulate_network(
                runtime,
                external_input=ext,
            )
            # Build full array including I neurons from the spikes object
            full_arr = _spikes_to_array(result, T_steps, n_total, dt)
            out_spike_tensors.append(full_arr[:, out_start:out_stop])
            full_spike_arrays[d] = full_arr

    # Output-layer raster grid
    save_raster_grid(
        data_path / "raster_output_all_all",
        out_spike_tensors,
        dt=dt,
        suptitle="Output layer spikes per digit class (inference)",
    )

    # Layer rasters per digit (now including I_global)
    for d in range(10):
        save_raster_layers(
            data_path / f"raster_layers_digit_{d:02d}",
            full_spike_arrays[d],
            dt=dt,
            pop_idx=pop_idx,
            title=f"E_hid + I_global + E_out — digit {d}",
        )



    # ── plots ────────────────────────────────────────────────────────────────
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

    print(f"\nSaved artifacts to {data_path}")
    print("Done!")


if __name__ == "__main__":
    main()
