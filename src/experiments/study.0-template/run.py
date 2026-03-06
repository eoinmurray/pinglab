import shutil
import sys
import json
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT

from pinglab.io import compile_graph_to_runtime, layer_bounds_from_spec, save_graph_diagram
from pinglab.backends.pytorch import simulate_network
from pinglab.plots.raster import save_raster


def main(
    artifacts_dir: Path | str | None = None,
    raw_data_dir: Path | str | None = None,
) -> None:
    experiment_dir = Path(__file__).parent.resolve()

    if artifacts_dir is None:
        data_path = ARTIFACTS_ROOT / Path(__file__).parent.name
    else:
        data_path = Path(artifacts_dir)

    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    config_path = experiment_dir / "config.json"
    with config_path.open(encoding="utf-8") as f:
        spec = json.load(f)

    # Run
    scan_id = str(spec.get("meta", {}).get("scan_id", "main"))
    runtime = compile_graph_to_runtime(spec, backend="pytorch")
    result = simulate_network(runtime)
    layer_bounds = layer_bounds_from_spec(spec)

    # Save raw data
    run_id = uuid.uuid4().hex[:8]
    if raw_data_dir is not None:
        run_data_dir = Path(raw_data_dir)
    else:
        run_data_dir = experiment_dir / "data" / run_id
    run_data_dir.mkdir(parents=True, exist_ok=True)

    # np.savez(run_data_dir / "training_metrics.npz", ...)
    # np.savez(run_data_dir / "inference_data.npz", ...)
    shutil.copy2(config_path, run_data_dir / "config.json")

    # Generate plots
    from plots import main as plots_main
    plots_main(data_dir=run_data_dir, artifacts_dir=data_path)

    # Also save raster directly (simple study example)
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
