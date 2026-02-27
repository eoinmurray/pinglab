import shutil
import sys
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT

from pinglab.io import compile_graph_to_runtime, layer_bounds_from_spec
from pinglab.backends.pytorch import simulate_network
from pinglab.plots.raster import save_raster
from plots import save_mean_input_plot


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

    # Experiment
    scan_id = str(spec.get("meta", {}).get("scan_id", "main"))
    runtime = compile_graph_to_runtime(spec, backend="pytorch")
    temporal = dict(spec.get("meta", {}).get("temporal_signal", {}))
    dt_ms = float(runtime.config.dt)
    T_ms = float(runtime.config.T)
    t_ms = np.arange(int(np.ceil(T_ms / dt_ms)), dtype=float) * dt_ms
    t_s = t_ms / 1000.0

    base = float(temporal.get("base", 1.2))
    amp = float(temporal.get("amp", 0.4))
    freq_hz = float(temporal.get("freq_hz", 3.0))
    phase = float(temporal.get("phase_rad", 0.0))
    signal = base + amp * np.sin(2.0 * np.pi * freq_hz * t_s + phase)
    signal = np.clip(signal, 0.0, None)

    external_input = runtime.external_input.detach().clone()
    n_e = int(runtime.config.N_E)
    external_input[:, :n_e] = external_input.new_tensor(signal)[:, None]
    save_mean_input_plot(
        data_path / f"input_row-1_{scan_id}_mean-e-input.png",
        t_ms=t_ms,
        mean_input=external_input[:, :n_e].mean(dim=1).detach().cpu().numpy(),
    )

    runtime_i = replace(runtime, external_input=external_input)
    result = simulate_network(runtime_i)
    layer_bounds = layer_bounds_from_spec(spec)

    save_raster(
        result.spikes,
        data_path / f"raster_row-2_{scan_id}_main",
        external_input=external_input.detach().cpu().numpy(),
        dt=float(runtime.config.dt),
        label="Raster | temporal encoding",
        xlim=(0.0, float(runtime.config.T)),
        layer_bounds=layer_bounds,
        layer_order=["I", "E"],
    )


if __name__ == "__main__":
    main()
