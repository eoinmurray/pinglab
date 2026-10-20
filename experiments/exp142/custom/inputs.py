"""Encode one diagnostic image and authenticate its physical spike coordinates.

Use SNNLab's native rate encoder once on the fixed input grid. Coordinate hashes
allow every solver timestep to verify exact replay without re-encoding events."""

from hashlib import sha256

import numpy as np


def event_digest(events):
    coordinates = np.column_stack(
        [events[key] for key in ("time_ns", "batches", "channels")]
    )
    if coordinates.dtype.kind not in "iu":
        raise ValueError("physical input coordinates must be integers")
    return sha256(coordinates.astype("<i8").tobytes()).hexdigest()


def encode_sample(graph, dataset_path, cfg, *, max_rate_hz):
    """Use SNNLab once on the fixed input grid, never once per solver timestep."""
    from snnlab.sim import (
        DatasetEncoder,
        DatasetSnapshotBinding,
        resolve_dataset_snapshot_binding,
    )

    if graph["timebase"]["dt"]["value"] * 1e6 != cfg["input_grid_ns"]:
        raise ValueError("encoder graph must use the fixed physical input grid")
    binding = DatasetSnapshotBinding(
        path=dataset_path,
        input_id="drive",
        dataset_id="sklearn_digits",
        split="sample",
        feature_key="sample_features",
        label_key="sample_labels",
        target_id=None,
        encoder=DatasetEncoder(
            kind="rate_poisson",
            duration_ms=cfg["duration_ns"] / 1e6,
            max_rate_hz=max_rate_hz,
            seed=cfg["input_seed"],
        ),
        shuffle=False,
    )
    resolved, _ = resolve_dataset_snapshot_binding(
        graph,
        binding,
        device="cpu",
        execution_seed=cfg["input_seed"],
    )
    indices = np.column_stack(
        np.nonzero(resolved.tensors["drive"].cpu().numpy())
    ).astype(np.int64)
    events = dict(
        time_ns=indices[:, 0] * cfg["input_grid_ns"],
        batches=indices[:, 1],
        channels=indices[:, 2],
    )
    return events, resolved.protocol
