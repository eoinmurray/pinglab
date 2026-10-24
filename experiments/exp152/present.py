"""Export the callback replay and native controller circuit.

Consumes an explicit analysis run without executing the model. Full traces remain
in its ancestry; replay states appear only at recorded callback completion times.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run

NODES = ("flexor", "extensor", "plant")


def recording(values, summary):
    """Keep every timing sample and expose controller states when callbacks finish."""
    nodes = {}
    for name in NODES:
        prefix = name + "_"
        times = np.asarray(values[prefix + "time_s"]) * 1000
        callback = np.asarray(values[prefix + "callback_ms"])
        row = dict(
            time_ms=times.tolist(),
            callback_ms=callback.tolist(),
            lag_ms=np.asarray(values[prefix + "lag_ms"]).tolist(),
            lateness_ms=np.asarray(values[prefix + "lateness_ms"]).tolist(),
            model_s=np.asarray(values[prefix + "model_time_s"]).tolist(),
            stop_reason=summary["stop_reasons"][name],
            statistics=summary["full_callback_ms"][name],
        )
        if len(times) != len(callback) or np.any(np.diff(times) < 0):
            raise ValueError(f"Invalid callback timeline for {name}")
        if name != "plant":
            row.update(
                voltage=np.asarray(values[prefix + "voltage"]).tolist(),
                spikes=np.asarray(values[prefix + "spikes"]).tolist(),
            )
        else:
            row.update(
                angle=np.asarray(values["angle"]).tolist(),
                velocity=np.asarray(values["velocity"]).tolist(),
                activation=np.asarray(values["activation"]).tolist(),
            )
        for field, samples in row.items():
            if not isinstance(samples, list):
                continue
            array = np.asarray(samples)
            if len(array) != len(times) or not np.isfinite(array).all():
                raise ValueError(f"Invalid callback field for {name}: {field}")
        if name != "plant" and (np.asarray(row["spikes"]) < 0).any():
            raise ValueError(f"Negative spike count for {name}")
        nodes[name] = row
    return dict(
        schema="pinglab.exp152.replay/v1",
        duration_ms=summary["duration_seconds"] * 1000,
        period_ms=summary["timer_ms"],
        neuron_model=summary.get("configuration", {}).get("neuron_model", "coba_lif"),
        voltage_range_mv=[-80, 40]
        if summary.get("configuration", {}).get("neuron_model") == "hodgkin_huxley"
        else [-75, -45],
        neurons_per_population=summary["neurons_per_population"],
        snnlab_ref=summary["snnlab_ref"],
        behaviour_passed=summary["passed"],
        timing_passed=summary["timing_passed"],
        clock="Monotonic callback completion time relative to each node's mapped shared start",
        nodes=nodes,
    )


def present(source, *, run_id=None):
    import shutil

    upstream = source_run(
        REPO / ".pingstore", source, stage="analyse", experiment="exp152"
    )
    numbers = load_json(upstream.file("numbers.json"))
    cfg = numbers["configuration"]
    with np.load(upstream.file("traces.npz"), allow_pickle=False) as data:
        values = dict(data)
    payload = recording(values, numbers)
    reference = upstream.record["inputs"]["compute"]
    compute = source_run(
        REPO / ".pingstore",
        reference["run_id"],
        stage="compute",
        experiment="exp152",
        reference=reference,
    )
    with stage_run(
        REPO,
        "exp152",
        "present",
        inputs={"analyse": upstream},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        write_json_atomic(run.export / "circuit.json", payload)
        circuit = compute.file("flexor", "circuit.svg")
        if circuit.is_file():
            shutil.copyfile(circuit, run.export / "circuit.svg")
        write_json_atomic(run.export / "numbers.json", numbers)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    present(**vars(parser.parse_args()))
