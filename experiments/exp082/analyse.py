"""Measure saved exp082 counts and streams; never run inference or plotting."""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from experiments.exp082 import evidence, inputs, measurements, recipe
from pingstore.contracts import PingstoreError, load_json, write_json_atomic


def analyse(identity, showcase_identity, *, run_id=None):
    source = inputs.source(REPO, identity, "compute")
    cfg, bank, contract = inputs.compute_evidence(REPO, source)
    showcase = inputs.source(REPO, showcase_identity, "compute")
    showcase_bank, showcase_record = evidence.showcase_evidence(REPO, showcase)
    if showcase_bank.reference != bank.reference:
        raise PingstoreError("showcase and evaluation use different training banks")
    rows = [evidence.condition(source, j, cfg) for j in recipe.jobs(cfg)]
    preflight = measurements.grid_output_preflight(rows)
    streams = {
        name: measurements.stream_result(*evidence.stream(source, name))
        for name in ("matched", "variable")
    }
    streams.update(
        {
            name: measurements.stream_result(
                *evidence.stream(
                    showcase, name, conditions=recipe.SHOWCASE_CONDITIONS
                )
            )
            for name in recipe.SHOWCASE_TARGETS
        }
    )
    selected = list(streams["matched"]["correct"]).index(1)
    streams["single_trial"] = measurements.first_correct_trial_from_stream(
        streams["matched"]
    )
    metadata, derived = {}, {}
    for name, stream in streams.items():
        stream.update(measurements.display_values(stream))
        metadata[name] = {
            k: v for k, v in stream.items() if not isinstance(v, np.ndarray)
        }
        for key in ("probabilities", "counts", "final_counts"):
            derived[f"{name}_{key}"] = stream[key]
    result = {
        "schema": "exp082.analysis/v2",
        "status": "complete",
        "profile": cfg["profile"],
        "condition_evidence": load_json(source.export / "evidence.json").get(
            "condition_evidence", "per-presentation-counts/v1"
        ),
        "config": {
            k: v
            for k, v in cfg.items()
            if k not in ("schema", "profile", "checkpoint_policy")
        },
        "checkpoint_policy": recipe.CHECKPOINT_POLICY,
        "checkpoint_provenance": contract["checkpoints"],
        "training_cells": [recipe.training_cell_name(s) for s in recipe.SEEDS],
        "training_source": "exp022 variable-rate streaming training",
        "readout": {
            "mode": "spike-count",
            "definition": "total output-LIF spikes over the matched presentation window",
            "reported_activity": "output spike rate in Hz may be derived as count divided by window duration in seconds",
        },
        **{
            name + ("_stream" if name != "single_trial" else ""): meta
            for name, meta in metadata.items()
        },
        "single_trial_segment_index": selected,
        "showcase_selection": {
            key: showcase_record[key]
            for key in ("configuration", "candidates", "selected")
        },
        "grid_per_seed": rows,
        "duration_200ms_psychometric": [r for r in rows if r["duration_ms"] == 200.0],
        "scientific_preflight": {
            "evaluation_grid": preflight,
            **{
                name + "_stream": streams[name]["output_activity"]
                for name in ("matched", "variable")
            },
        },
        "plot_data": measurements.plot_data(rows, cfg),
    }
    with inputs.execution(
        REPO,
        "analyse",
        sources={"compute": source, "showcase": showcase, "bank": bank},
        run_id=run_id,
        configuration={
            "schema": "exp082.analysis/v2",
            "uncertainty": "sample SD / sqrt(3)",
        },
    ) as run:
        write_json_atomic(run.export / "numbers.json", result)
        np.savez_compressed(run.export / "display.npz", **derived)
    return run.run_id


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--showcase-source", required=True)
    p.add_argument("--run-id")
    a = p.parse_args()
    try:
        print(analyse(a.source, a.showcase_source, run_id=a.run_id))
    except (PingstoreError, OSError, KeyError, ValueError, RuntimeError) as exc:
        p.exit(1, f"exp082 analyse: {exc}\n")


if __name__ == "__main__":
    main()
