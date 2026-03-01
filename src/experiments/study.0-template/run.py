import shutil
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT

from pinglab.io import compile_graph_to_runtime, layer_bounds_from_spec, save_graph_diagram
from pinglab.backends.pytorch import simulate_network
from pinglab.plots.raster import save_raster


def main() -> None:
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

    # Graph diagram
    save_graph_diagram(spec, data_path / "graph_main_main_00")

    # Run
    scan_id = str(spec.get("meta", {}).get("scan_id", "main"))
    runtime = compile_graph_to_runtime(spec, backend="pytorch")
    result = simulate_network(runtime)
    layer_bounds = layer_bounds_from_spec(spec)

    save_raster(
        result.spikes,
        data_path / f"raster_main_{scan_id}_00",
        dt=float(runtime.config.dt),
        label="Raster",
        xlim=(0.0, float(runtime.config.T)),
        layer_bounds=layer_bounds,
        layer_order=["I", "E"],
    )

    print(f"Saved artifacts to {data_path}")


if __name__ == "__main__":
    main()
