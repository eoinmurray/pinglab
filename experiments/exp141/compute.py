"""Execute COBA and PING training and save the evidence for each mode.

Run `uv run python experiments/exp141/compute.py` from the repository root.
The recipe supplies the scientific settings; SNNLAB_DEVICE can override native
device selection (CUDA when available, otherwise CPU). One compute run retains one
authored bundle and circuit diagram per model, native parameter snapshots for the
final and best-validation weights, consolidated results.json and recording.npz.
The dataset and resumable checkpoint used for replay are temporary scratch files.

SNNLab owns spike encoding, shuffled minibatches, surrogate-gradient training
and epoch-level validation. The lowest validation loss selects a completed epoch;
SNNLab tests it once on the disjoint held-out split after training. Online
training metrics come from minibatches. This file connects those native APIs to the
repository's run storage and native snapshots. It does not launch analysis
or plotting: those stages consume explicitly named completed runs through
their own commands. Importing this file does not start an experiment."""

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path

# Support direct script execution while resolving repository-local helpers.
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

import numpy as np
from experiments.exp141 import dataset, recipe
from pingstore.library import stage_run, write_json_atomic
from snnlab.sim import (
    CheckpointSelection,
    DenseArrayBinding,
    EpochLogger,
    Performance,
    RecordingSpec,
    SignalRecording,
    SnapshotSpec,
    TestSpec,
    TrainingMetrics,
    ValidationSpec,
    resolve_device,
    simulate,
    train,
)


def record_sample(mode, bundle, drive, checkpoint, directory, device, cfg):
    """Replay the saved image from fresh state using the selected checkpoint."""
    populations = ["E", "readout"] + (["I"] if mode == "ping" else [])
    # Restore learned weights, not the membrane state at the end of training.
    return simulate(
        bundle=bundle,
        checkpoint=checkpoint,
        input_bindings=(DenseArrayBinding("drive", drive),),
        seed=cfg["seed"],
        device=device,
        performance=Performance(**cfg["performance"]),
        diagnostics=False,
        recording=RecordingSpec(
            signals=(
                SignalRecording("drive.value"),
                *(SignalRecording(f"{name}.spikes") for name in populations),
            ),
        ),
        snapshot=SnapshotSpec(
            path=directory / "best-validation" / "start",
            stage="start",
            parameters=True,
        ),
    )


def compute(*, run_id=None):
    cfg = recipe.configuration()
    device = str(resolve_device())
    # Pingstore records the configuration and owns the compute run lifecycle.
    with stage_run(
        REPO,
        cfg["experiment"],
        "compute",
        run_id=run_id,
        configuration=cfg,
    ) as run:
        # Both models use the same split; downstream needs only the diagnostic image.
        digits = dataset.prepare(run.scratch / "dataset.npz", cfg)
        run.record["execution"].update(
            dataset={
                **digits.provenance,
                **{f"{split}_samples": count for split, count in digits.counts.items()},
            },
            device=device,
            performance=cfg["performance"],
        )
        partition = digits.partitions[digits.provenance["sample_partition"]]
        sample_row = partition.sample_ids.tolist().index(cfg["sample_index"])
        recordings = dict(pixels=partition.features[sample_row].reshape(8, 8).copy())
        results = dict(
            schema="exp141.results/v2",
            sample=dict(
                index=cfg["sample_index"],
                label=int(partition.labels[sample_row]),
                partition=run.record["execution"]["dataset"]["sample_partition"],
                checkpoint="validation_selected",
            ),
            models={},
            recording_info={},
        )
        # Explicit retention permits single end snapshots without weakening the
        # default start/end-pair check for other scientific executions.
        run.record["scientific_artifacts"] = dict(
            snapshot_stages={
                f"{mode}/{weights}": [stage]
                for mode in cfg["models"]
                for weights, stage in (("final", "end"), ("best-validation", "start"))
            }
        )
        drive = None
        # Each mode has a different topology, so each gets its own circuit diagram.
        for mode in cfg["models"]:
            bundle = recipe.author_network(mode, cfg)
            if drive is None:
                # One encoder draw guarantees an identical input for the two replays.
                drive = dataset.sample_drive(digits, bundle, cfg)
            mode_directory = run.export / mode
            mode_directory.mkdir()
            bundle.visualise(mode_directory / "circuit.svg", view="circuit")
            bundle.write(run.export / f"{mode}.bundle", visualise=False)
            selected_checkpoint = run.scratch / mode / "selected-checkpoint"
            # Retain final weights natively, and use a temporary selected checkpoint
            # to capture the best-validation weights during the diagnostic replay.
            result = train(
                bundle=bundle,
                input_bindings=(dataset.binding(digits, "train", cfg),),
                validation=ValidationSpec(
                    input_bindings=(dataset.binding(digits, "validation", cfg),),
                    encoding_seeds=[cfg["seed"]],
                ),
                test=TestSpec(
                    input_bindings=(dataset.binding(digits, "test", cfg),),
                    encoding_seeds=[cfg["seed"]],
                ),
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                seed=cfg["seed"],
                device=device,
                performance=Performance(**cfg["performance"]),
                # Shuffle training only; keep incomplete batches in every partition.
                shuffle=True,
                metrics=TrainingMetrics(**cfg["training_metrics"]),
                checkpoint_selection=CheckpointSelection.from_dict(
                    cfg["checkpoint_selection"]
                ),
                save_selected_checkpoint=selected_checkpoint,
                diagnostics=False,
                on_epoch_end=EpochLogger(metrics=("accuracy",), precision=3),
                snapshot=SnapshotSpec(
                    path=mode_directory / "final" / "end",
                    stage="end",
                    parameters=True,
                ),
            )
            test_metrics = result.metrics["test"]
            print(
                f"Selected epoch {test_metrics['epoch']}: "
                f"test={test_metrics['accuracy']:.3f}",
                flush=True,
            )
            replay = record_sample(
                mode, bundle, drive, selected_checkpoint, mode_directory, device, cfg
            )
            results["models"][mode] = dict(
                training=result.metrics, sample=replay.metrics
            )
            sample = replay.numpy(batch=0)
            recordings[f"{mode}--scores"] = sample.outputs["class_scores"]
            for signal, spikes in sample.recorded_signals.items():
                name = (
                    "drive"
                    if signal == "drive.value"
                    else f"{mode}--{signal.removesuffix('.spikes')}"
                )
                if name in recordings:
                    np.testing.assert_array_equal(recordings[name], spikes)
                    continue
                recordings[name] = spikes.astype(np.uint8)
                # Keep native axes and coordinates with the lossless binary export.
                results["recording_info"][name] = asdict(
                    replace(
                        sample.recording.info[signal],
                        array_dtype="uint8",
                        conversion="binary_spikes_to_uint8",
                    )
                )
            print(f"Completed {mode}", flush=True)
        write_json_atomic(run.export / "results.json", results)
        np.savez_compressed(run.export / "recording.npz", **recordings)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    compute(**vars(parser.parse_args()))
