"""Simulate the fixed MNIST image with SNNLab's native execution API.

Invoke compute.py with optional --protocol comparison, screen or long. Each
seed supplies the same pixel spikes to every condition. The authored bundle
contains all weights, including the one-to-one inhibitory background; no
runtime parameter editing or training is needed.

One simulate() call executes each case from rest. SNNLab streams coordinate-
aware recording blocks and saves its native simulation-start snapshot. Pingstore
retains these artifacts and native metrics in a new immutable compute run.
Imports neither load the dataset nor execute a simulation.
"""

import argparse
import sys
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
import torch
from experiments.exp144 import dataset, recipe
from pingstore.stages import stage_run
from snnlab.sim import (
    DenseArrayBinding,
    NPZRecordingSink,
    Performance,
    RecordingSpec,
    SignalRecording,
    SnapshotSpec,
    simulate,
)


def compute(args):
    cfg = recipe.configuration(args.protocol)
    pixels, provenance = dataset.load_sample(REPO, cfg)
    torch.set_num_threads(cfg["torch_threads"])
    with stage_run(
        REPO, "exp144", "compute", configuration=cfg, run_id=args.run_id
    ) as run:
        run.record["execution"].update(
            snnlab_version=version("snnlab"), dataset=provenance
        )
        run.record["scientific_artifacts"] = dict(snapshot_stages={})
        np.savez_compressed(
            run.export / "sample.npz",
            pixels=pixels.numpy().reshape(28, 28),
            label=np.int64(cfg["sample_label"]),
        )
        for seed in cfg["seeds"]:
            image, backgrounds = dataset.encode_drive(pixels, cfg, seed)
            # Retain the exact inputs, independently of each condition's recording.
            np.savez_compressed(
                run.export / f"input-seed-{seed}.npz",
                image=image.numpy().astype(bool),
                **{
                    f"background_i_{rate:g}hz": values.numpy().astype(bool)
                    for rate, values in backgrounds.items()
                },
            )
            for index, case in enumerate(cfg["cases"]):
                bundle = recipe.author_network(cfg, case)
                directory = run.export / case["id"] / f"seed-{seed}"
                directory.mkdir(parents=True)
                simulation = directory / "simulation"
                bundle.write(simulation / "network.bundle", visualise=False)
                if index == 0 and seed == cfg["seeds"][0]:
                    bundle.visualise(run.export / "circuit.svg", view="circuit")
                prefix = f"{case['id']}/seed-{seed}/simulation"
                run.record["scientific_artifacts"]["snapshot_stages"][prefix] = [
                    "start"
                ]
                sink = NPZRecordingSink(directory / "recording")
                result = simulate(
                    bundle=bundle,
                    input_bindings=(
                        DenseArrayBinding("image", image),
                        DenseArrayBinding(
                            "background_i", backgrounds[case["i_rate_hz"]]
                        ),
                    ),
                    protocol=provenance,
                    seed=seed,
                    device=cfg["device"],
                    performance=Performance(**cfg["performance"]),
                    diagnostics=False,
                    recording=RecordingSpec(
                        signals=tuple(
                            SignalRecording(signal)
                            for signal in cfg["recording_signals"]
                        ),
                        sink=sink,
                        sink_id=sink.identity,
                        block_steps=cfg["recording_block_steps"],
                    ),
                    snapshot=SnapshotSpec(
                        path=simulation / "start", stage="start", parameters=True
                    ),
                    retain_outputs=False,
                    retain_parameters=False,
                    retain_final_state=False,
                    retain_runtime_state=False,
                )
                # Check actual native parameters rather than trusting a recorded knob.
                weights = result.snapshots["start"].parameters
                if tuple(weights["input_to_E.weight"].shape) != (
                    cfg["n_input"],
                    cfg["n_e"],
                ):
                    raise ValueError("unexpected input matrix shape")
                for name in ("E_to_I.weight", "I_to_E.weight"):
                    actual = float((weights[name] > 0).float().mean())
                    if abs(actual - case["recurrence_density"]) > 0.02:
                        raise ValueError(
                            "native recurrence density differs from recipe"
                        )
                torch.testing.assert_close(
                    weights["background_to_I.weight"],
                    torch.eye(cfg["n_i"]) * cfg["i_background_jump_us"],
                    rtol=0,
                    atol=0,
                )
                result.write_metrics(directory / "metrics.json")
                print(f"Completed {case['id']} seed {seed}", flush=True)
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", choices=("comparison", "screen", "long"), default="comparison"
    )
    parser.add_argument("--run-id")
    print(compute(parser.parse_args()))


if __name__ == "__main__":
    main()
