"""Validate saved training histories and extract the learning-curve data.

Run `uv run python experiments/exp141/analyse.py --source <compute-run-id>`
after compute has completed. The explicit input identifies one Pingstore
compute run. Its saved configuration must match the current recipe before
its metrics are interpreted. For each mode, the history must include every
epoch, starting with the untrained validation baseline. Training curves use
online minibatch metrics from completed epochs; validation uses fixed weights.
The native held-out test identifies the validation-selected epoch and its score.

The source provides consolidated results.json and recording.npz; parameter
tables come from its final-training snapshots, which retain trained weights.
The output is a separate analysis run containing numbers.json: the recipe
and simple per-epoch loss/accuracy records for training and validation. It also
validates post-training sample recordings and saves them in rasters.npz.
This stage reads saved measurements only; it neither constructs networks nor
acquires a dataset, performs inference or resumes training. Keeping extraction
separate lets presentation reuse the validated numbers without execution."""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
from experiments.exp141.recipe import configuration
from pingstore.library import load_json, source_run, stage_run, write_json_atomic
from snnlab.sim import (
    CheckpointSelection,
    NetworkSnapshot,
    RecordingData,
    SignalInfo,
    TrainingHistory,
)


def sample_recording(saved, mode, cfg):
    """Validate dense spike arrays and the spike-count classification contract."""
    steps = round(cfg["duration_ms"] / cfg["dt_ms"])
    sizes = {"E": cfg["n_hidden"], "readout": cfg["n_classes"]}
    if mode == "ping":
        sizes["I"] = cfg["n_inhibitory"]
    arrays = {name: saved[f"{mode}--{name}"].copy() for name in sizes}
    scores = saved[f"{mode}--scores"].copy()
    for name, spikes in arrays.items():
        if spikes.shape != (steps, sizes[name]) or not np.isin(spikes, (0, 1)).all():
            raise ValueError(f"invalid sample spikes: {mode}/{name}")
    if scores.shape != (cfg["n_classes"],):
        raise ValueError("invalid sample class scores")
    np.testing.assert_array_equal(scores, arrays["readout"].sum(axis=0))
    # Store compact, validated binary recordings; presentation only reads these.
    arrays = {name: values.astype(np.uint8) for name, values in arrays.items()}
    return arrays, dict(scores=scores.tolist(), prediction=int(scores.argmax()))


def parameter_rows(upstream, mode):
    """Build the report table from the native final-training snapshot.

    Initializer statistics still describe initialization; the accompanying
    parameters.npz holds the actual final weights rather than their initial draw.
    """
    snapshot = NetworkSnapshot.read(
        upstream.file(mode, "final", "end", "snapshot.json")
    )
    if (
        snapshot.stage != "end"
        or snapshot.execution.kind != "train"
        or not all(parameter.stored for parameter in snapshot.parameter_info.values())
    ):
        raise ValueError("expected a native final-training parameter snapshot")
    execution, training = snapshot.execution, snapshot.training
    rows = [
        ["Seed", execution.seed],
        ["Timestep (ms)", execution.timestep_ms],
        ["Presentation (ms)", execution.duration_ms],
        ["Epochs", training.epochs],
        ["Batch size", execution.batch_size],
        ["Optimizer", training.optimizer.kind],
        ["Learning rate", training.optimizer.config.lr],
        ["Weight decay", training.optimizer.config.weight_decay],
        ["Gradient clipping", training.gradient_clip],
        ["Surrogate", training.surrogate.kind],
        [
            "Regularizers",
            "None" if not training.regularizers else str(training.regularizers),
        ],
    ]
    rows.append(
        [
            "Maximum pixel rate (Hz)",
            execution.protocol.dataset_binding.encoder.max_rate_hz,
        ]
    )
    for name, population in snapshot.populations.items():
        neuron = population.parameters
        rows.extend(
            [
                [f"{name}: neurons", population.size],
                [f"{name}: membrane model", population.neuron.kind],
                [f"{name}: capacitance (nF)", neuron["capacitance_nf"].value],
                [f"{name}: leak (µS)", neuron["leak_us"].value],
                [f"{name}: membrane time constant (ms)", neuron["tau_mem"].value],
                [
                    f"{name}: rest / reset / threshold (mV)",
                    f"{neuron['resting_mv'].value} / {neuron['reset_mv'].value} / {neuron['threshold_mv'].value}",
                ],
                [f"{name}: refractory period (ms)", neuron["refractory"].value],
                [
                    f"{name}: voltage gradient dampening",
                    neuron["voltage_grad_dampen"].value,
                ],
            ]
        )
    for name, projection in snapshot.projections.items():
        initializer = projection.initializer
        rows.extend(
            [
                [
                    f"{name}: synapse / decay (ms)",
                    f"{projection.synapse.kind} / {projection.synapse_parameters['tau'].value}",
                ],
                [
                    f"{name}: delay (ms)",
                    projection.delay.value if projection.get("delay") else 0,
                ],
                [
                    f"{name}: matrix shape",
                    " × ".join(map(str, projection.runtime_matrix_shape)),
                ],
                [
                    f"{name}: normal mean / SD (µS)",
                    f"{initializer['mean']:.6g} / {initializer['std']:.6g}; clamped nonnegative",
                ],
                [
                    f"{name}: weights",
                    "learned" if projection.trainable_parameters else "fixed",
                ],
            ]
        )
    return rows


def training_history(metrics, cfg, dataset_info):
    """Validate native evidence against this experiment's measurement protocol."""
    history = TrainingHistory.from_metrics(metrics)
    history.validate(
        expected_epochs=cfg["epochs"],
        require_complete=True,
        selection=CheckpointSelection.from_dict(cfg["checkpoint_selection"]),
        expected_examples={"test": dataset_info["test_samples"]},
        expected_encoding_draws={"test": (cfg["seed"],)},
    )
    if history.test.checkpoint != "selected":
        raise ValueError("expected a held-out test of the selected checkpoint")
    # SNNLab validates the format; exp141 requires these particular measurements.
    for epoch in history.epochs:
        for split, semantics in (
            ("train", "online_pre_update"),
            ("validation", "fixed_weight_evaluation"),
        ):
            measured = epoch.splits.get(split)
            if split == "train" and epoch.epoch == 0:
                if measured is not None:
                    raise ValueError("online training has no epoch-zero baseline")
                continue
            if measured is None or measured.measurement != semantics:
                raise ValueError(f"missing or unexpected {split} measurement")
            # With one cross-entropy objective and no penalties, total loss is CE.
            if measured.loss < 0 or measured.get("accuracy") is None:
                raise ValueError("invalid classification metrics")
    return history


def learning_curves(history):
    """Keep the report's compact curve format while using native split records."""
    rows = []
    for epoch in history.epochs:
        row = dict(epoch=epoch.epoch)
        for split in ("train", "validation"):
            measured = epoch.splits.get(split)
            row[split] = (
                {key: measured[key] for key in ("loss", "accuracy", "measurement")}
                if measured is not None
                else None
            )
        rows.append(row)
    return rows


def analyse(source, *, run_id=None):
    cfg = configuration()
    # source_run checks the explicit source run and its Pingstore provenance.
    upstream = source_run(
        REPO / ".pingstore",
        source,
        stage="compute",
        experiment=cfg["experiment"],
    )
    # Reject histories from another recipe instead of silently mixing settings.
    if upstream.record["execution"]["configuration"] != cfg:
        raise ValueError("compute configuration differs from this recipe")
    results = load_json(upstream.file("results.json"))
    if results["schema"] != "exp141.results/v2":
        raise ValueError("expected consolidated exp141 compute results")
    # The diagnostic is original dataset sample zero, not the first shuffled row.
    with np.load(upstream.file("recording.npz"), allow_pickle=False) as saved:
        pixels = saved["pixels"].copy()
        drive = saved["drive"].copy()
        recordings = {
            mode: sample_recording(saved, mode, cfg) for mode in cfg["models"]
        }
    if (
        pixels.shape != (8, 8)
        or not np.isfinite(pixels).all()
        or not ((pixels >= 0) & (pixels <= 1)).all()
    ):
        raise ValueError("invalid diagnostic image")
    steps = round(cfg["duration_ms"] / cfg["dt_ms"])
    if drive.shape != (steps, cfg["n_input"]) or not np.isin(drive, (0, 1)).all():
        raise ValueError("invalid diagnostic input spikes")
    rasters = dict(pixels=pixels, drive=drive.astype(np.uint8))
    sample = {**results["sample"], "modes": {}}
    if (
        sample["index"] != cfg["sample_index"]
        or not 0 <= sample["label"] < cfg["n_classes"]
        or sample["partition"]
        != upstream.record["execution"]["dataset"]["sample_partition"]
        or sample["checkpoint"] != "validation_selected"
    ):
        raise ValueError("invalid diagnostic sample metadata")
    curves = {}
    training = {}
    test_metrics = {}
    for mode in cfg["models"]:
        spikes, sample["modes"][mode] = recordings[mode]
        rasters.update({f"{mode}--{name}": values for name, values in spikes.items()})
        training[mode] = results["models"][mode]["training"]
        history = training_history(
            training[mode], cfg, upstream.record["execution"]["dataset"]
        )
        test_metrics[mode] = training[mode]["test"]
        sample["modes"][mode]["epoch"] = history.test.epoch
        curves[mode] = learning_curves(history)
    recording = RecordingData(
        signals={name: values for name, values in rasters.items() if name != "pixels"},
        info={
            name: SignalInfo(**values)
            for name, values in results["recording_info"].items()
        },
    )
    for name, info in recording.info.items():
        if (
            info.unit != "spike"
            or info.dt_ms != cfg["dt_ms"]
            or info.step_indices != tuple(range(steps))
            or info.cell_ids != tuple(range(recording.signals[name].shape[1]))
        ):
            raise ValueError(f"unexpected diagnostic recording coordinates: {name}")
    # Save derived numbers in a new run linked to the immutable compute source.
    with stage_run(
        REPO,
        cfg["experiment"],
        "analyse",
        inputs={"compute": upstream},
        run_id=run_id,
        configuration=cfg,
    ) as run:
        np.savez_compressed(run.export / "rasters.npz", **rasters)
        write_json_atomic(
            run.export / "numbers.json",
            dict(
                configuration=cfg,
                curves=curves,
                training=training,
                recording_info=results["recording_info"],
                test=test_metrics,
                sample=sample,
                dataset=upstream.record["execution"]["dataset"],
                parameters={
                    mode: parameter_rows(upstream, mode) for mode in cfg["models"]
                },
            ),
        )
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    analyse(**vars(parser.parse_args()))
