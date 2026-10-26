"""Prepare a recorded classifier replay from explicitly selected SNNLab evidence.

The CLI requires explicit completed analysis evidence. Figures and browser
replays are written to a new presentation run; upstream execution is never
launched automatically.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp127.recipe import SLUG
from pingstore.contracts import load_json, write_json_atomic
from pingstore.stages import source_run, stage_run
from snnlab import lang


def events(array, dt):
    steps, channels = np.nonzero(array)
    return [
        [float(step * dt), int(channel)]
        for step, channel in zip(steps, channels, strict=True)
    ]


def present(identity):
    source = source_run(REPO / ".pingstore", identity, stage="analyse", experiment=SLUG)
    compute = source_run(
        REPO / ".pingstore",
        source.record["inputs"]["compute"]["run_id"],
        stage="compute",
        experiment=SLUG,
    )
    if compute.reference != source.record["inputs"]["compute"]:
        raise ValueError("analysis input digest does not match compute evidence")
    numbers = load_json(source.file("numbers.json"))
    cfg = numbers["configuration"]
    replay = {
        "schema": "exp127.classifier/v2",
        "dt_ms": cfg["dt_ms"],
        "duration_ms": cfg["duration_ms"],
        "delay_ms": cfg["delay_ms"],
        "selected_epoch": numbers["selected_epoch"],
        "test_samples": cfg["test_samples"],
        "class_names": ["A", "B"],
        "weight_unit": "nA",
        "decision_min_spikes": cfg["decision_min_spikes"],
        "decision_min_lead": cfg["decision_min_lead"],
        "modes": {},
        "examples": [],
    }
    graph = load_json(compute.file("classifier.bundle/graph.json"))
    neuron = graph["populations"][0]["neuron"]
    replay["rest_mv"] = neuron["resting_mv"]
    replay["reset_mv"] = neuron["reset_mv"]
    replay["threshold_mv"] = neuron["threshold_mv"]
    with (
        np.load(compute.file("dataset.npz")) as dataset,
        np.load(compute.file("recording.npz")) as recording,
    ):
        for mode in ("before", "after"):
            replay["modes"][mode] = {
                "accuracy": numbers["test"][mode]["accuracy"],
                "weights": recording[f"{mode}_weights"].tolist(),
            }
        for index in numbers["example_indices"]:
            example = {
                "index": index,
                "label": int(dataset["test_labels"][index]),
                "rates_hz": dataset["test_rates"][index].tolist(),
                "inputs": events(dataset["test_spikes"][:, index], cfg["dt_ms"]),
                "modes": {},
            }
            for mode in ("before", "after"):
                metrics = numbers["test"][mode]
                example["modes"][mode] = {
                    "outputs": events(
                        recording[f"{mode}_output_spikes"][:, index], cfg["dt_ms"]
                    ),
                    "voltages_mv": recording[f"{mode}_output_voltage"][
                        :, index
                    ].tolist(),
                    "counts": metrics["counts"][index],
                    "prediction": metrics["predictions"][index],
                    "output_spikes": metrics["output_spikes"][index],
                    "latency_ms": metrics["latency_ms"][index],
                    "decision_class": metrics["decision_class"][index],
                }
            replay["examples"].append(example)
    teaching = {
        "schema": "exp127.training/v1",
        "dt_ms": cfg["dt_ms"],
        "duration_ms": cfg["duration_ms"],
        "rest_mv": replay["rest_mv"],
        "threshold_mv": replay["threshold_mv"],
        **numbers["training_trace"],
        "examples": [],
        "steps": [],
    }
    with (
        np.load(compute.file("training_trace.npz")) as trace,
        np.load(compute.file("dataset.npz")) as dataset,
    ):
        for index in trace["indices"]:
            teaching["examples"].append(
                {
                    "label": int(dataset["train_labels"][index]),
                    "rates_hz": dataset["train_rates"][index].tolist(),
                    "inputs": events(dataset["train_spikes"][:, index], cfg["dt_ms"]),
                }
            )
        start = teaching["window_start"]
        for step in range(start, start + teaching["window_length"]):
            teaching["steps"].append(
                {
                    "iteration": step + 1,
                    "loss": float(trace["losses"][step]),
                    "weights_before": trace["weights"][step].tolist(),
                    "weights_after": trace["weights"][step + 1].tolist(),
                    "gradients": trace["gradients"][step].tolist(),
                    "examples": [
                        {
                            "outputs": events(
                                trace["outputs"][step, :, example], cfg["dt_ms"]
                            ),
                            "voltages_mv": trace["voltages"][step, :, example].tolist(),
                        }
                        for example in range(2)
                    ],
                }
            )
    with stage_run(
        REPO,
        SLUG,
        "present",
        inputs={"analyse": source, "compute": compute},
        configuration=cfg,
    ) as run:
        if compute.record.get("scientific_artifacts", {}).get("native_paths"):
            bundle_path = compute.restore_bundle(
                "classifier.bundle", run.scratch / "classifier.bundle"
            )
        else:
            bundle_path = compute.unit("classifier.bundle")
        lang.load_bundle(bundle_path).visualise(
            run.export / "circuit.svg", view="circuit"
        )
        write_json_atomic(run.export / "numbers.json", numbers)
        write_json_atomic(run.export / "classifier.json", replay)
        write_json_atomic(run.export / "training.json", teaching)
    print(run.run_id)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    present(parser.parse_args().source)
