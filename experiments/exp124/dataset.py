"""Acquire SHD and encode its selected event streams for native training.

The official training archive is checksum-verified before reading. A seeded
per-class shuffle separates training and validation utterances; local checks
retain one of each per class. The official test archive is never accessed.
Acquisition records counts separately from the resolved scientific recipe.

The custom DatasetEncoder combines event binning with the calibrated independent
insertion/deletion noise, which SNNLab's event_bin encoder does not supply.
Bindings authenticate the temporary ragged-event NPZ; selection and source
checksums remain in the completed run. Importing this module does no I/O.
"""

import gzip
import hashlib
import shutil
import urllib.request

import h5py
import numpy as np
import torch
from snnlab import extensions

BASE = "https://zenkelab.org/datasets/"


def digest(path, algorithm="sha256"):
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def select_indices(all_labels, cfg):
    """Stratified held-out split; full mode retains every official train item."""
    rng = np.random.default_rng(cfg["seed"])
    train, validation = [], []
    for label in range(cfg["n_classes"]):
        available = np.flatnonzero(all_labels == label)
        selected = rng.permutation(available)
        held_out = max(1, round(len(available) * cfg["validation_fraction"]))
        n_train = len(available) - held_out
        if n_train < 1:
            raise ValueError("each class needs training and validation examples")
        if not cfg["full_dataset"]:
            selected = np.concatenate((selected[:1], selected[n_train : n_train + 1]))
            n_train = 1
        train.extend(selected[:n_train])
        validation.extend(selected[n_train:])
    return train, validation


def load_inputs(cfg, cache):
    cache.mkdir(parents=True, exist_ok=True)
    filename = "shd_train.h5.gz"
    expected = "d47c9825dee33347913e8ce0f2be08b0"  # Official archive, pinned in the retained evidence.
    archive = cache / filename
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        urllib.request.urlretrieve(BASE + filename, temporary)
        if digest(temporary, "md5") != expected:
            raise ValueError("SHD download checksum mismatch")
        temporary.rename(archive)
    if digest(archive, "md5") != expected:
        raise ValueError("cached SHD checksum mismatch")
    path = archive.with_suffix("")
    temporary = path.with_suffix(".unpack")
    with gzip.open(archive, "rb") as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target)
    temporary.replace(path)
    with h5py.File(path, "r") as handle:
        all_labels = np.asarray(handle["labels"], dtype=np.int64)
        train, validation = select_indices(all_labels, cfg)
        indices = np.asarray(train + validation, dtype=np.int64)
        offsets, channels, times = [0], [], []
        for index in indices:
            t = np.asarray(handle["spikes/times"][index], dtype=np.float64) * 1000
            ch = np.asarray(handle["spikes/units"][index], dtype=np.int64)
            if (
                not np.isfinite(t).all()
                or (t < 0).any()
                or (ch < 0).any()
                or (ch >= cfg["n_input"]).any()
            ):
                raise ValueError("invalid SHD events")
            keep = t < cfg["duration_ms"]
            offsets.append(offsets[-1] + int(keep.sum()))
            channels.append(ch[keep])
            times.append(t[keep])
        labels = all_labels[indices]
        speakers = np.asarray(handle["extra/speaker"])[indices]
    counts = dict(
        train_class_counts=np.bincount(
            labels[: len(train)], minlength=cfg["n_classes"]
        ).tolist(),
        validation_class_counts=np.bincount(
            labels[len(train) :], minlength=cfg["n_classes"]
        ).tolist(),
    )
    if cfg.get("full_dataset", False) and len(indices) != len(all_labels):
        raise ValueError(
            "full-data selection did not retain every official training sample"
        )
    arrays = dict(
        event_offsets=np.asarray(offsets),
        event_time_ms=np.concatenate(times),
        event_channel=np.concatenate(channels),
        labels=labels,
        official_train_indices=indices,
        speakers=speakers,
        split=np.asarray([0] * len(train) + [1] * len(validation)),
    )
    metadata = dict(
        url=BASE + filename,
        archive_md5=expected,
        archive_sha256=digest(archive),
        hdf5_sha256=digest(path),
        selection_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        train_count=len(train),
        validation_count=len(validation),
        official_train_count=len(all_labels),
        noise_seeds={
            "training": cfg["seed"] + 2,
            "validation_evaluation": cfg["seed"] + 4,
        },
        evaluation_noise="same corruption each epoch; matched across models",
        retained_events=int(sum(len(t) for t in times)),
        official_test_accessed=False,
    )
    return labels, arrays, metadata, counts


def encode_shd(arrays, selected, *, dt_ms, channels, config, seed):
    """Bin ragged SHD events and apply the experiment's insertion/deletion noise."""
    spikes = np.zeros(
        (round(config["duration_ms"] / dt_ms), len(selected), channels),
        dtype=np.float32,
    )
    for column, local_index in enumerate(selected):
        index = int(local_index) + config["sample_offset"]
        start, end = arrays["event_offsets"][index : index + 2]
        steps = np.floor(arrays["event_time_ms"][start:end] / dt_ms).astype(np.int64)
        keep = steps < len(spikes)
        spikes[steps[keep], column, arrays["event_channel"][start:end][keep]] = 1
    noise = np.random.default_rng(seed).random(spikes.shape, dtype=np.float32)
    occupied = spikes != 0
    spikes[occupied & (noise < config["spike_deletion_probability"])] = 0
    spikes[~occupied & (noise < config["input_noise_hz"] * dt_ms / 1000)] = 1
    return torch.from_numpy(spikes)


extensions.register_encoder("exp124.shd_events/v1", encode_shd)
