import json
import shutil
import sys
from pathlib import Path

import numpy as np

from pinglab.analysis import spike_count_for_range
from pinglab.io import compile_graph_to_runtime, layer_bounds_from_spec, set_edges_enabled
from pinglab.backends.pytorch import simulate_network
from pinglab.plots.raster import save_raster
from pinglab.io import linspace_from_scan, overwrite_spec_value

from plots import save_e2_spikes_vs_delay

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT


def _count_for_label(spike_ids: np.ndarray, layer_bounds: list[tuple[int, int, str]], label: str) -> int:
    for start, stop, layer_label in layer_bounds:
        if layer_label == label:
            return spike_count_for_range(spike_ids, int(start), int(stop))
    return 0


def main() -> None:
    data_path = ARTIFACTS_ROOT / Path(__file__).parent.name
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    config_path = Path(__file__).parent / "config.json"
    with config_path.open(encoding="utf-8") as f:
        spec = json.load(f)
    shutil.copy2(config_path, data_path / "config.json")

    scan_cfg = dict(spec.get("meta", {}).get("scan", {}))
    scan_id = str(scan_cfg.get("scan_id", "delay_scan"))
    values = linspace_from_scan(scan_cfg)

    baseline_spec = set_edges_enabled(spec, {"e_src_to_e_a", "e_src_to_e_b"})
    baseline_runtime = compile_graph_to_runtime(baseline_spec, backend="pytorch")
    baseline_result = simulate_network(baseline_runtime)
    baseline_bounds = layer_bounds_from_spec(baseline_spec)
    save_raster(
        baseline_result.spikes,
        data_path / "raster_row-0_baseline_no_ff",
        external_input=baseline_runtime.external_input.detach().cpu().numpy(),
        dt=baseline_runtime.config.dt,
        label="baseline no E_src->targets",
        xlim=(0.0, float(baseline_runtime.config.T)),
        layer_bounds=baseline_bounds,
        layer_order=["I", "E_src", "E_A", "E_B"],
    )
    print("[baseline] source feedforward disabled -> raster saved")

    delays: list[float] = []
    spikes_a: list[int] = []
    spikes_b: list[int] = []

    for idx, delay_ms in enumerate(values):
        variant_spec = overwrite_spec_value(spec, "edges[e_src_to_e_b].delay_ms", float(delay_ms))

        runtime = compile_graph_to_runtime(variant_spec, backend="pytorch")
        config = runtime.config
        result = simulate_network(runtime)

        layer_bounds = layer_bounds_from_spec(variant_spec)
        a_count = _count_for_label(result.spikes.ids, layer_bounds, "E_A")
        b_count = _count_for_label(result.spikes.ids, layer_bounds, "E_B")

        delays.append(float(delay_ms))
        spikes_a.append(int(a_count))
        spikes_b.append(int(b_count))

        save_raster(
            result.spikes,
            data_path / f"raster_row-1_{scan_id}_{idx:02d}_delay-{delay_ms:.3f}",
            external_input=runtime.external_input.detach().cpu().numpy(),
            dt=config.dt,
            label=f"E_src->E_B delay_ms {delay_ms:.3f}",
            xlim=(0.0, float(config.T)),
            layer_bounds=layer_bounds,
            layer_order=["I", "E_src", "E_A", "E_B"],
        )

        print(
            f"[{idx + 1}/{len(values)}] e_src->e_b.delay_ms={float(delay_ms):.3f} "
            f"-> raster saved | E_A={a_count} E_B={b_count}"
        )

    save_e2_spikes_vs_delay(
        data_path / f"spikes_row-2_{scan_id}_targets-vs-delay.png",
        delay_ms=delays,
        e1_spikes=spikes_a,
        e2_spikes=spikes_b,
    )


if __name__ == "__main__":
    main()
