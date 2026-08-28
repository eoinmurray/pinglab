from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools"), str(REPO / "tools/snnsim")]

import numpy as np
import snnlang as snn
from experiments.exp077 import recipe
from experiments.helpers import snnlang_stages as stages
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def phase_diagnostics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    trace_a = a.astype(np.float32).mean(axis=(1, 2))
    trace_b = b.astype(np.float32).mean(axis=(1, 2))
    if trace_a.std() == 0 or trace_b.std() == 0:
        return {"zero_lag_correlation": 0.0, "peak_lag_steps": 0.0}
    corr = np.correlate(trace_a - trace_a.mean(), trace_b - trace_b.mean(), mode="full")
    lags = np.arange(-len(trace_a) + 1, len(trace_a))
    return {
        "zero_lag_correlation": float(np.corrcoef(trace_a, trace_b)[0, 1]),
        "peak_lag_steps": float(lags[int(np.argmax(corr))]),
    }


def parity_measurements(directory):
    acquisition = load_json(directory / "acquisition.json")
    with np.load(directory / "parity.npz") as arrays:

        def discrepancy(a, b):
            return float(np.max(np.abs(arrays[a] - arrays[b])))

        parameters = [
            key for key in arrays.files if key.startswith("legacy_parameter__")
        ]
        parameter_max = max(
            discrepancy(key, key.replace("legacy_parameter__", "native_parameter__"))
            for key in parameters
        )
        output_max = discrepancy("legacy_output", "native_output")
        replay_max = discrepancy("replay_output", "native_output")
        compiled_max = discrepancy("compiled_first", "compiled_warm")
        mismatches = sum(
            int(
                np.count_nonzero(
                    arrays[f"legacy_{population}"] != arrays[f"native_{population}"]
                )
            )
            for population in ("e", "i")
        )
    legacy_s = float(np.median(acquisition["timings"]["legacy"]))
    graph_s = float(np.median(acquisition["timings"]["graph"]))
    return {
        "parameter_max_abs": parameter_max,
        "output_max_abs": output_max,
        "spike_mismatch_count": mismatches,
        "checkpoint_replay_max_abs": replay_max,
        "legacy_median_s": legacy_s,
        "graph_median_s": graph_s,
        "graph_overhead_percent": 100 * (graph_s / legacy_s - 1),
        "legacy_peak_python_bytes": acquisition["peaks"]["legacy"],
        "graph_peak_python_bytes": acquisition["peaks"]["graph"],
        **{
            key: value
            for key, value in acquisition.items()
            if key.startswith("compile_")
        },
        "compiled_warm_median_s": float(np.median(acquisition["compiled_times"])),
        "compiled_replay_max_abs": compiled_max,
        "performance_gate_percent": recipe.PARITY["gate_percent"],
        "performance_gate_pass": graph_s
        <= (1 + recipe.PARITY["gate_percent"] / 100) * legacy_s,
        "parity_pass": parameter_max
        == output_max
        == replay_max
        == compiled_max
        == mismatches
        == 0,
    }


def analyse(identity, *, run_id=None):
    source = stages.source(REPO, recipe, identity, "compute")
    with stages.execution(
        REPO, recipe, "analyse", sources={"compute": source}, run_id=run_id
    ) as run:
        variants = {}
        for name, _ in recipe.VARIANTS:
            bundle = snn.load_bundle(source.export / "variants" / f"{name}.bundle")
            # Topology inspection performs no simulation or graph planning.
            coupling = [
                p
                for p in bundle.graph["projections"]
                if p["id"].startswith(("a_I_to_b", "b_I_to_a"))
            ]
            with np.load(
                source.export / "variants" / f"{name}-recordings.npz"
            ) as arrays:
                variants[name] = {
                    "graph_digest": bundle.manifest["graph_digest"],
                    "coupling_projection_count": len(coupling),
                    "coupling_delay_steps": sorted(
                        {round(p["delay"]["value"] / recipe.DT_MS) for p in coupling}
                    ),
                    "spikes": {
                        key: int(arrays[key].sum())
                        for key in (
                            "population_0",
                            "population_1",
                            "population_2",
                            "population_3",
                        )
                    },
                    "diagnostics": phase_diagnostics(
                        arrays["population_0"], arrays["population_2"]
                    ),
                }
        delay = load_json(source.export / "delay-gates.json")
        if delay.get("passed") is not True or any(
            delay.get(key) != 0 for key in ("failures", "errors", "skipped")
        ):
            raise PingstoreError("missing successful delay evidence")
        write_json_atomic(
            run.export / "results.json",
            {
                "schema": "exp077.analysis/v1",
                "purpose": "graph-native architecture and causality validation",
                "config": recipe.SCALE,
                "variants": variants,
                "parity_performance": parity_measurements(source.export / "parity"),
                "delay_timing": {
                    "gate": delay,
                    "zero_additional_delay_steps": 1,
                    "explicit_delay_steps": 5,
                },
                "exit": {
                    "variants_graph_only": True,
                    "milestone_4_sweep_performed": False,
                    "paid_compute_usd": 0.0,
                },
            },
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id", help="unused source-neutral reservation")
    args = parser.parse_args()
    try:
        analyse(args.source, run_id=args.run_id)
    except (PingstoreError, OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
