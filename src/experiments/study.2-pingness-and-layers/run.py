import shutil
import sys
import json
import copy
from pathlib import Path
from typing import Any

from pinglab.analysis import spike_count_for_range
from pinglab.io import compile_graph_to_runtime, layer_bounds_from_spec, set_edges_enabled
from pinglab.backends.pytorch import simulate_network
from pinglab.plots.raster import save_raster

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import ARTIFACTS_ROOT

def _build_independent_i_spec(spec: dict[str, Any]) -> dict[str, Any]:
    updated: dict[str, Any] = copy.deepcopy(spec)

    original_i_node = next(
        (
            node
            for node in updated.get("nodes", [])
            if node.get("kind") == "population" and str(node.get("id")) == "I"
        ),
        None,
    )
    if original_i_node is None:
        raise ValueError("Expected an inhibitory population node with id 'I'")

    original_i_size = int(original_i_node.get("size", 0))
    if original_i_size <= 1:
        raise ValueError("Expected inhibitory population size > 1 for I1/I2 split")

    i1_size = original_i_size // 2
    i2_size = original_i_size - i1_size

    nodes: list[dict[str, Any]] = []
    for node in updated.get("nodes", []):
        if str(node.get("id")) == "I":
            node_i1 = copy.deepcopy(node)
            node_i1["id"] = "I1"
            node_i1["size"] = i1_size
            node_i2 = copy.deepcopy(node)
            node_i2["id"] = "I2"
            node_i2["size"] = i2_size
            nodes.extend([node_i1, node_i2])
        else:
            nodes.append(node)
    updated["nodes"] = nodes

    edges = updated.get("edges", [])
    e1_to_i_template = next(
        (edge for edge in edges if str(edge.get("id")) == "e1_to_i"), None
    )
    e2_to_i_template = next(
        (edge for edge in edges if str(edge.get("id")) == "e2_to_i"), None
    )
    i_to_e1_template = next(
        (edge for edge in edges if str(edge.get("id")) == "i_to_e1"), None
    )
    i_to_e2_template = next(
        (edge for edge in edges if str(edge.get("id")) == "i_to_e2"), None
    )
    if (
        e1_to_i_template is None
        or e2_to_i_template is None
        or i_to_e1_template is None
        or i_to_e2_template is None
    ):
        raise ValueError("Missing I-coupling template edges for independent variant")

    preserved_edges = [
        edge
        for edge in edges
        if str(edge.get("id"))
        not in {"e1_to_i", "e2_to_i", "i_to_e1", "i_to_e2"}
    ]

    e1_to_i1 = copy.deepcopy(e1_to_i_template)
    e1_to_i1["id"] = "e1_to_i1"
    e1_to_i1["to"] = "I1"

    i1_to_e1 = copy.deepcopy(i_to_e1_template)
    i1_to_e1["id"] = "i1_to_e1"
    i1_to_e1["from"] = "I1"

    e2_to_i2 = copy.deepcopy(e2_to_i_template)
    e2_to_i2["id"] = "e2_to_i2"
    e2_to_i2["to"] = "I2"

    i2_to_e2 = copy.deepcopy(i_to_e2_template)
    i2_to_e2["id"] = "i2_to_e2"
    i2_to_e2["from"] = "I2"

    updated["edges"] = preserved_edges + [e1_to_i1, i1_to_e1, e2_to_i2, i2_to_e2]
    return updated

def main() -> None:
    data_path = ARTIFACTS_ROOT / Path(__file__).parent.name
    if data_path.exists():
        shutil.rmtree(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    config_path = Path(__file__).parent / "config.json"
    with config_path.open(encoding="utf-8") as f:
        spec = json.load(f)
    shutil.copy2(config_path, data_path / "config.json")

    variants = [
        {
            "scan_id": "fully_connected",
            "label": "Fully connected",
            "disabled_edges": set(),
            "independent_i": False,
        },
        {
            "scan_id": "partially_connected",
            "label": "No E2<->I",
            "disabled_edges": {"e2_to_i", "i_to_e2"},
            "independent_i": False,
        },
        {
            "scan_id": "independently_connected",
            "label": "Split I pools (E1-I1, E2-I2)",
            "disabled_edges": set(),
            "independent_i": True,
        },
    ]

    for variant in variants:
        if bool(variant.get("independent_i", False)):
            variant_spec = _build_independent_i_spec(spec)
        else:
            variant_spec = set_edges_enabled(spec, variant["disabled_edges"])
        runtime = compile_graph_to_runtime(variant_spec, backend="pytorch")
        config = runtime.config
        result = simulate_network(runtime)

        layer_bounds = layer_bounds_from_spec(variant_spec)
        scan_id = str(variant["scan_id"])
        label = str(variant["label"])
        layer_order = ["I", "E1", "E2"]
        if bool(variant.get("independent_i", False)):
            layer_order = ["I1", "I2", "E1", "E2"]
        save_raster(
            result.spikes,
            data_path / f"raster_row-1_raster_{scan_id}.png",
            external_input=runtime.external_input.detach().cpu().numpy(),
            dt=config.dt,
            label=label,
            xlim=(0.0, float(config.T)),
            layer_bounds=layer_bounds,
            layer_order=layer_order,
        )

        spike_ids = result.spikes.ids
        i_total = 0
        for start, stop, layer_label in layer_bounds:
            count = spike_count_for_range(spike_ids, start, stop)
            print(f"[{scan_id}] [spikes] {layer_label}: {count}")
            if layer_label.upper().startswith("I"):
                i_total += count
        print(f"[{scan_id}] [spikes] I_total: {i_total}")


if __name__ == "__main__":
    main()
