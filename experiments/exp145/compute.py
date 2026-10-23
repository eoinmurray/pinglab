"""Execute a shared-input COBA probe with 50 ms burn-in in a new Pingstore run.

Run compute.py with optional --seed, --device, --measurement-ms and --run-id. Native recording
retains all E/I spikes plus sampled voltages and projection conductances.
The bundle, start snapshot, common external spikes and execution metrics are saved;
no training, analysis or presentation is launched automatically.
"""

import argparse
import sys
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
import torch
from experiments.exp145 import dataset, recipe
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


def recording_signals(cfg):
    signals = [SignalRecording(f"{p}.spikes") for p in ("E", "I")]
    for p in ("E", "I"):
        cells = tuple(
            np.linspace(
                0,
                cfg[f"n_{p.lower()}"] - 1,
                min(cfg["recording_cells"], cfg[f"n_{p.lower()}"]),
                dtype=int,
            ).tolist()
        )
        signals.append(SignalRecording(f"{p}.voltage", cells=cells))
        for source in ("E", "I", "X"):
            for quantity in ("conductance",):
                signals.append(
                    SignalRecording(f"{source}_to_{p}.{quantity}", cells=cells)
                )
    return tuple(signals)


def compute(args):
    cfg = recipe.configuration(
        seed=args.seed, device=args.device, measurement_ms=args.measurement_ms
    )
    torch.set_num_threads(cfg["torch_threads"])
    bundle = recipe.author_network(cfg)
    external = dataset.encode_drive(cfg)
    with stage_run(
        REPO, "exp145", "compute", configuration=cfg, run_id=args.run_id
    ) as run:
        run.record["execution"]["snnlab_version"] = version("snnlab")
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={"simulation": ["start"]}
        )
        np.savez_compressed(
            run.export / "external.npz", spikes=external.numpy().astype(bool)
        )
        bundle.write(run.export / "simulation" / "network.bundle", visualise=False)
        bundle.visualise(run.export / "circuit.svg", view="circuit")
        sink = NPZRecordingSink(run.export / "recording")
        result = simulate(
            bundle=bundle,
            input_bindings=(DenseArrayBinding("X", external),),
            seed=cfg["seed"],
            device=cfg["device"],
            performance=Performance(**cfg["performance"]),
            diagnostics=False,
            recording=RecordingSpec(
                signals=recording_signals(cfg),
                sink=sink,
                sink_id=sink.identity,
                block_steps=cfg["recording_block_steps"],
            ),
            snapshot=SnapshotSpec(
                path=run.export / "simulation" / "start", stage="start", parameters=True
            ),
            retain_outputs=False,
            retain_parameters=False,
            retain_final_state=False,
            retain_runtime_state=False,
        )
        result.write_metrics(run.export / "metrics.json")
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--measurement-ms", type=float, default=1000.0)
    parser.add_argument("--run-id")
    print(compute(parser.parse_args()))


if __name__ == "__main__":
    main()
