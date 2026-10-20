"""Prepare scikit-learn digits and describe how SNNLab should encode them.

Each bundled example is an 8-by-8 grayscale image, flattened to 64 channels,
with an integer label from zero to nine. Dividing the original pixel values
by 16 produces float32 intensities in [0, 1]. Seeded, stratified splits keep
approximately the same digit proportions in training, validation and test. The
temporary compressed dataset file retains all partitions and their original
indices, so both network modes consume exactly the same examples. Compute keeps
the diagnostic image and dataset provenance in its completed run.

The saved features are intensities, not precomputed spikes. binding() tells
SNNLab which arrays to read and how to turn intensities into Poisson spike
trains during execution: brighter pixels have higher expected firing rates.
Dataset preparation happens only when prepare() is called; importing this
module does not load or save the digits dataset."""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from snnlab.sim import DatasetEncoder, DatasetPartition, DatasetSnapshot


def prepare(path, cfg):
    pixels, labels = load_digits(return_X_y=True)
    # Scaling is fixed by the dataset range; no fitted preprocessing is needed.
    features = pixels.astype(np.float32) / 16.0
    # Split indices so saved rows can always be traced to the original dataset.
    development_indices, test_indices = train_test_split(
        np.arange(len(labels)),
        test_size=cfg["test_fraction"],
        random_state=cfg["seed"],
        stratify=labels,
    )
    train_indices, validation_indices = train_test_split(
        development_indices,
        # Both fractions describe the full dataset, not the remaining subset.
        test_size=cfg["validation_fraction"] / (1 - cfg["test_fraction"]),
        random_state=cfg["seed"],
        stratify=labels[development_indices],
    )
    indices = dict(
        train=train_indices, validation=validation_indices, test=test_indices
    )
    sample_partition = next(
        split for split, rows in indices.items() if cfg["sample_index"] in rows
    )
    # Keep sklearn's exact split and row order; SNNLab owns archive validation.
    snapshot = DatasetSnapshot.from_partitions(
        dataset_id="sklearn_digits",
        partitions={
            split: DatasetPartition(features[rows], labels[rows], rows)
            for split, rows in indices.items()
        },
        disjoint=("train", "validation", "test"),
        preprocessing=dict(operation="divide", divisor=16.0, dtype="float32"),
        provenance=dict(
            source="sklearn.datasets.load_digits",
            pixel_scale=16.0,
            split_policy="sklearn.train_test_split/two-stage",
            split_seed=cfg["seed"],
            validation_fraction=cfg["validation_fraction"],
            test_fraction=cfg["test_fraction"],
            sample_index=cfg["sample_index"],
            sample_label=int(labels[cfg["sample_index"]]),
            sample_partition=sample_partition,
        ),
    )
    snapshot.write(path)
    return DatasetSnapshot.read(path)


def binding(snapshot, split, cfg):
    # The input and target identifiers match the graph and cross-entropy objective.
    diagnostic = split == "sample"
    return snapshot.binding(
        snapshot.provenance["sample_partition"] if diagnostic else split,
        input_id="drive",
        target_id=None if diagnostic else "digit",
        sample_ids=(cfg["sample_index"],) if diagnostic else None,
        # At intensity 1 a pixel has input_rate_hz as its expected Poisson rate;
        # lower intensities scale that rate over the same presentation duration.
        encoder=DatasetEncoder(
            kind="rate_poisson",
            duration_ms=cfg["duration_ms"],
            max_rate_hz=cfg["input_rate_hz"],
            seed=cfg["sample_encoder_seed"] if diagnostic else cfg["seed"],
        ),
    )


def sample_drive(snapshot, bundle, cfg):
    """Encode the retained image once on CPU for exact replay in both modes."""
    from snnlab.sim import resolve_dataset_snapshot_binding

    resolved, _ = resolve_dataset_snapshot_binding(
        bundle.graph,
        binding(snapshot, "sample", cfg),
        device="cpu",
        execution_seed=cfg["sample_encoder_seed"],
    )
    return resolved.tensors["drive"]
