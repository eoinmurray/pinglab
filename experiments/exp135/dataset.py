"""Verified public training archive, subject split and binary event snapshots."""

from __future__ import annotations

import hashlib
import io
import math
import re
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

import numpy as np
from experiments.exp135 import recipe


def checksum(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_archive(path):
    """Accept only complete gzip responses, then authenticate the pinned bytes."""
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(
                recipe.TRAIN_ARCHIVE_URL,
                headers={"User-Agent": "pinglab-dataset/1.0"},
            )
            print(
                f"Downloading DVS Gesture training archive (attempt {attempt}/3)",
                flush=True,
            )
            with urllib.request.urlopen(request, timeout=120) as source:
                if source.status != 200 or source.headers.get("Content-Range"):
                    raise ValueError(
                        f"expected a full HTTP 200 response, received {source.status}"
                    )
                content_type = source.headers.get("Content-Type", "").split(";", 1)[0]
                if content_type.startswith("text/"):
                    raise ValueError(f"expected gzip archive, received {content_type}")
                length = source.headers.get("Content-Length")
                expected = int(length) if length is not None else None
                digest, received, next_progress = hashlib.md5(), 0, 128 * 1024**2
                with partial.open("wb") as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        if received == 0 and not block.startswith(b"\x1f\x8b"):
                            raise ValueError("download response is not a gzip archive")
                        target.write(block)
                        digest.update(block)
                        received += len(block)
                        if received >= next_progress:
                            print(
                                f"DVS Gesture download: {received:,} / {expected or 'unknown'} bytes",
                                flush=True,
                            )
                            next_progress += 128 * 1024**2
                if received == 0 or (expected is not None and received != expected):
                    raise ValueError(
                        f"incomplete archive: received {received} of {expected} bytes"
                    )
            actual = digest.hexdigest()
            if actual != recipe.TRAIN_ARCHIVE_MD5:
                raise ValueError(
                    f"archive checksum mismatch: expected {recipe.TRAIN_ARCHIVE_MD5}, received {actual} ({received} bytes)"
                )
            partial.replace(path)
            print(
                f"Verified DVS Gesture archive: {received:,} bytes; MD5 {actual}",
                flush=True,
            )
            return path
        except (OSError, ValueError) as exc:
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(
                    "DVS Gesture archive download failed after three attempts"
                ) from exc
            print(f"Archive attempt {attempt} rejected: {exc}; retrying", flush=True)
            time.sleep(attempt)
    raise AssertionError("unreachable download state")


def verified_archive(path, *, download=False):
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"dataset archive missing: {path}; use --download")
        path.parent.mkdir(parents=True, exist_ok=True)
        return download_archive(path)
    if checksum(path, "md5") != recipe.TRAIN_ARCHIVE_MD5:
        raise ValueError("training archive checksum mismatch")
    print(
        f"Verified cached DVS Gesture archive: {path.stat().st_size:,} bytes",
        flush=True,
    )
    return path


def inventory(path):
    rows = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.suffix != ".npy":
                continue
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("unsafe dataset member path")
            subject = re.fullmatch(r"user(\d+)_(.+)", name.parent.name)
            if "ibmGestureTrain" not in name.parts or subject is None:
                raise ValueError(f"unexpected training sample: {name}")
            label = int(name.stem)
            if not 0 <= label < recipe.N_CLASSES:
                raise ValueError("invalid gesture label")
            rows.append(
                dict(sample_id=member.name, subject=int(subject[1]), label=label)
            )
    if len(rows) != recipe.EXPECTED_TRAIN_SAMPLES:
        raise ValueError(
            f"expected {recipe.EXPECTED_TRAIN_SAMPLES} samples, got {len(rows)}"
        )
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate archive sample identities")
    return sorted(rows, key=lambda r: r["sample_id"])


def balanced_selection(rows, count, *, seed):
    """Select within classes deterministically, distributing residual slots fairly."""
    rng = np.random.default_rng(seed)
    classes = rng.permutation(recipe.N_CLASSES)
    queues = {}
    for label in classes:
        candidates = sorted(
            (r for r in rows if r["label"] == label), key=lambda r: r["sample_id"]
        )
        queues[int(label)] = [candidates[i] for i in rng.permutation(len(candidates))]
    selected = []
    while len(selected) < count:
        for label in classes:
            if not queues[int(label)]:
                raise ValueError("insufficient examples for a balanced subset")
            selected.append(queues[int(label)].pop())
            if len(selected) == count:
                break
    return selected


def select(rows):
    subjects = sorted({r["subject"] for r in rows})
    if len(subjects) < 3:
        raise ValueError("at least three training subjects are required")
    held_out = subjects[-2:]
    training = [r for r in rows if r["subject"] not in held_out]
    validation = [r for r in rows if r["subject"] in held_out]
    count = max(recipe.N_CLASSES, math.floor(len(rows) * recipe.TRAIN_FRACTION + 0.5))
    return dict(
        train=balanced_selection(training, count, seed=recipe.SELECTION_SEED),
        validation=balanced_selection(
            validation,
            recipe.N_CLASSES * recipe.VALIDATION_PER_CLASS,
            seed=recipe.SELECTION_SEED + 1,
        ),
    ), held_out


def encode(events):
    """Tonic archive columns are x,y,polarity,time_ms; retain full event extent."""
    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] != 4 or len(events) == 0:
        raise ValueError("expected a nonempty [events,4] array")
    if not np.isfinite(events).all():
        raise ValueError("event coordinates/timestamps must be finite")
    if np.any(events[:, :3] != np.floor(events[:, :3])):
        raise ValueError("event coordinates/polarities must be integers")
    x, y, polarity = events[:, :3].astype(np.int64).T
    timestamp = events[:, 3].astype(np.float64)
    if (
        np.any((x < 0) | (x >= 128) | (y < 0) | (y >= 128))
        or np.any((polarity != 0) & (polarity != 1))
        or np.any(timestamp < 0)
    ):
        raise ValueError("event coordinate/polarity/timestamp out of bounds")
    origin = float(timestamp.min())
    steps = np.floor((timestamp - origin) / recipe.DT_MS).astype(np.int64)
    channels = polarity * 1024 + (y // 4) * 32 + x // 4
    occupied = np.unique(steps * 2048 + channels)
    return dict(
        event_time_ms=(occupied // 2048).astype(np.float64) * recipe.DT_MS,
        event_channel=(occupied % 2048).astype(np.int64),
        duration_ms=float((steps.max() + 1) * recipe.DT_MS),
        timestamp_origin_ms=origin,
        original_events=len(events),
        binary_events=len(occupied),
        collisions=len(events) - len(occupied),
    )


def prepare(path, destination, *, local_check=None):
    """Keep selected binary event data and per-example identity as scientific output."""
    rows = inventory(path)
    selected, held_out = select(rows)
    if local_check:
        selected = {
            split: chosen[: local_check[f"{split}_examples"]]
            for split, chosen in selected.items()
        }
    wanted = {r["sample_id"] for split in selected.values() for r in split}
    encoded = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            if member.name in wanted:
                with archive.extractfile(member) as handle:
                    events = np.load(io.BytesIO(handle.read()), allow_pickle=False)
                    source_events = len(events)
                    source_extent = float(events[:, 3].max() - events[:, 3].min())
                    if local_check and local_check["presentation_ms"] is not None:
                        elapsed = events[:, 3] - events[:, 3].min()
                        events = events[elapsed < local_check["presentation_ms"]]
                    encoded[member.name] = encode(events)
                    if local_check:
                        encoded[member.name].update(
                            source_events=source_events,
                            source_event_extent_ms=source_extent,
                            prefix_limit_ms=local_check["presentation_ms"],
                        )
    if set(encoded) != wanted:
        raise ValueError("archive is missing selected examples")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        source_samples=len(rows),
        archive_sha256=checksum(path),
        validation_subjects=held_out,
        splits={},
        presentation_ms=(
            local_check["presentation_ms"]
            if local_check and local_check["presentation_ms"] is not None
            else max(e["duration_ms"] for e in encoded.values())
        ),
    )
    for split, chosen in selected.items():
        samples, times, channels, descriptions = [], [], [], []
        for index, row in enumerate(chosen):
            values = encoded[row["sample_id"]]
            times.append(values["event_time_ms"])
            channels.append(values["event_channel"])
            samples.append(np.full(values["binary_events"], index, dtype=np.int64))
            descriptions.append(
                {
                    **row,
                    **{k: v for k, v in values.items() if not k.startswith("event_")},
                }
            )
        np.savez_compressed(
            destination / f"{split}-events.npz",
            event_sample=np.concatenate(samples),
            event_time_ms=np.concatenate(times),
            event_channel=np.concatenate(channels),
            labels=np.asarray([r["label"] for r in chosen], dtype=np.int64),
            durations_ms=np.asarray(
                [r["duration_ms"] for r in descriptions], dtype=np.int64
            ),
            event_offsets=np.concatenate(([0], np.cumsum([len(v) for v in times]))),
        )
        metadata["splits"][split] = descriptions
    return metadata


def crop_start(duration_ms, window_ms, *, seed, sample_index, random=True):
    """Choose an integer-ms crop; shorter examples are right-zero-padded."""
    available = max(0, int(duration_ms) - int(window_ms))
    if not random:
        return available // 2
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(sample_index)]))
    return int(rng.integers(available + 1))


def encode_crops(arrays, selected, *, dt_ms, channels, config, seed):
    """Vectorised binary crops through SNNLab's batch-local encoder interface."""
    import torch

    window = float(config["duration_ms"])
    if dt_ms != 1.0 or window <= 0 or window != int(window):
        raise ValueError("gesture crops require integer duration and 1 ms resolution")
    times = arrays["event_time_ms"]
    offsets = arrays["event_offsets"]
    durations = arrays["durations_ms"]
    spikes = np.zeros((int(window), len(selected), channels), dtype=np.float32)
    for batch, sample in enumerate(selected):
        left, right = int(offsets[sample]), int(offsets[sample + 1])
        sample_times = times[left:right]
        start = crop_start(
            durations[sample],
            window,
            seed=seed,
            sample_index=sample,
            random=config["random"],
        )
        lo, hi = np.searchsorted(sample_times, [start, start + window])
        steps = (sample_times[lo:hi] - start).astype(np.int64)
        indices = arrays["event_channel"][left + lo : left + hi]
        spikes[steps, batch, indices] = 1.0
    return torch.from_numpy(spikes)


from snnlab import extensions

extensions.register_encoder("exp135.random_crop/v1", encode_crops)
