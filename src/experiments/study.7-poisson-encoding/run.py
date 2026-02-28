import shutil
import sys
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT

from pinglab.backends.types import Spikes
from pinglab.io import RuntimeBundle, compile_graph_to_runtime, layer_bounds_from_spec
from pinglab.backends.pytorch import simulate_network
from pinglab.plots.raster import save_raster
from plots import (
    save_total_spikes_vs_neuron_id,
)


def generate_poisson_input(
    *,
    rates_hz: np.ndarray,
    t_ms: float,
    dt_ms: float,
    seed: int,
) -> np.ndarray:
    num_neurons = int(rates_hz.size)
    num_steps = int(np.ceil(t_ms / dt_ms))
    rng = np.random.default_rng(seed)

    p_spike = np.clip(rates_hz * (dt_ms / 1000.0), 0.0, 1.0)
    spike_mask = rng.random((num_steps, num_neurons)) < p_spike[None, :]
    return spike_mask.astype(np.float32)


def poisson_mask_to_spikes(poisson_mask: np.ndarray, dt_ms: float) -> Spikes:
    t_idx, neuron_ids = np.nonzero(poisson_mask > 0.0)
    times_ms = t_idx.astype(np.float64) * float(dt_ms)
    ids = neuron_ids.astype(np.int64)
    types = np.zeros_like(ids, dtype=np.int64)
    return Spikes(times=times_ms, ids=ids, types=types)


def main() -> None:
    # Boilerplate

    # Resolve artifacts path
    data_path = ARTIFACTS_ROOT / Path(__file__).parent.name
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    # Read and copy config into the artifacts directory
    config_path = Path(__file__).parent / "config.json"
    with config_path.open(encoding="utf-8") as f:
        spec = json.load(f)
    shutil.copy2(config_path, data_path / "config.json")

    # Experiment: inject linearly spaced Poisson trains into E neurons
    scan_id = str(spec.get("meta", {}).get("scan_id", "poisson"))
    runtime = compile_graph_to_runtime(spec, backend="pytorch")

    t_ms = float(runtime.config.T)
    dt_ms = float(runtime.config.dt)
    seed = int(runtime.config.seed or 0)
    n_e = int(runtime.config.N_E)
    n_i = int(runtime.config.N_I)
    meta = spec.get("meta", {})
    min_rate_hz = float(meta.get("min_rate_hz", 5.0))
    max_rate_hz = float(meta.get("max_rate_hz", 50.0))
    input_amp = float(meta.get("input_amp", 3.0))
    rates_hz = np.linspace(min_rate_hz, max_rate_hz, n_e, dtype=np.float64)

    poisson_mask = generate_poisson_input(
        rates_hz=rates_hz,
        t_ms=t_ms,
        dt_ms=dt_ms,
        seed=seed,
    )
    input_spikes = poisson_mask_to_spikes(poisson_mask, dt_ms)
    external_input = runtime.external_input.detach().cpu().numpy().astype(np.float32, copy=True)
    external_input[:, :n_e] += poisson_mask * input_amp

    runtime_override = RuntimeBundle(
        config=runtime.config,
        external_input=torch.as_tensor(external_input, dtype=torch.float32),
        weights=runtime.weights,
        model=runtime.model,
        backend=runtime.backend,
        device="cpu",
    )
    result = simulate_network(runtime_override)
    layer_bounds = layer_bounds_from_spec(spec)

    # Diagnostics
    duration_s = t_ms / 1000.0
    input_spikes_per_neuron = poisson_mask.sum(axis=0).astype(np.int64)
    input_empirical_hz = input_spikes_per_neuron.astype(np.float64) / max(duration_s, 1e-9)
    total_input_current_per_neuron = external_input[:, :n_e].sum(axis=0).astype(np.float64)

    output_spikes_per_neuron = np.bincount(
        result.spikes.ids[result.spikes.ids < n_e],
        minlength=n_e,
    ).astype(np.int64)
    output_empirical_hz = output_spikes_per_neuron.astype(np.float64) / max(duration_s, 1e-9)

    corr_input_target_vs_output = float(
        np.corrcoef(rates_hz, output_empirical_hz)[0, 1]
    ) if n_e > 1 else 0.0
    corr_input_empirical_vs_output = float(
        np.corrcoef(input_empirical_hz, output_empirical_hz)[0, 1]
    ) if n_e > 1 else 0.0

    print("[study.7] --- diagnostics ---")
    print(f"[study.7] duration_ms={t_ms:.1f} dt_ms={dt_ms:.3f} seed={seed}")
    print(
        f"[study.7] tonic_baseline_mean={float(runtime.external_input.detach().cpu().numpy()[:, :n_e].mean()):.4f} "
        f"input_amp={input_amp:.4f}"
    )
    print(
        "[study.7] corr(target_rate_hz, output_rate_hz)="
        f"{corr_input_target_vs_output:.4f}"
    )
    print(
        "[study.7] corr(empirical_input_rate_hz, output_rate_hz)="
        f"{corr_input_empirical_vs_output:.4f}"
    )
    print(
        "[study.7] totals: "
        f"input_spikes={int(input_spikes_per_neuron.sum())} "
        f"output_spikes={int(output_spikes_per_neuron.sum())}"
    )
    print(
        "[study.7] per-neuron: "
        "id target_hz empirical_in_hz in_spikes total_input_current out_spikes out_hz"
    )
    for i in range(n_e):
        print(
            "[study.7] "
            f"{i:02d} "
            f"{rates_hz[i]:8.3f} "
            f"{input_empirical_hz[i]:14.3f} "
            f"{int(input_spikes_per_neuron[i]):9d} "
            f"{total_input_current_per_neuron[i]:19.3f} "
            f"{int(output_spikes_per_neuron[i]):10d} "
            f"{output_empirical_hz[i]:7.3f}"
        )

    info_lines = [
        f"rates_hz: {min_rate_hz:.1f} -> {max_rate_hz:.1f}",
        f"input_amp: {input_amp:.2f}",
    ]

    save_raster(
        input_spikes,
        data_path / f"raster_row-1_{scan_id}_input-poisson",
        label="Input raster | poisson encoding",
        info_lines=[f"rates_hz: {min_rate_hz:.1f} -> {max_rate_hz:.1f}"],
        xlim=(0.0, t_ms),
        layer_bounds=[(0, n_e, "E")],
        layer_order=["E"],
    )

    save_raster(
        result.spikes,
        data_path / f"raster_row-2_{scan_id}_network-poisson",
        label="Network raster | poisson encoding",
        info_lines=info_lines,
        dt=dt_ms,
        xlim=(0.0, t_ms),
        layer_bounds=layer_bounds,
        layer_order=["E"],
    )

    e_ids = np.arange(n_e, dtype=np.int64)
    e_spike_counts = output_spikes_per_neuron
    save_total_spikes_vs_neuron_id(
        data_path / f"spikes_row-3_{scan_id}_total-spikes-vs-neuron-id",
        neuron_ids=e_ids,
        spike_counts=e_spike_counts,
    )


if __name__ == "__main__":
    main()
