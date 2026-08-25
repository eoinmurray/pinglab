"""Notebook-side dataset access (data, not CLI internals).

Mirrors the MNIST source partitions used by tools/snnsim: the official training
partition remains separate from the untouched official test partition.

Notebooks that only *run* the network never need this (the CLI loads data itself);
it exists for the few notebooks that must build custom stimuli from raw pixels
(e.g. nb048's sequential digit streams).
"""

from __future__ import annotations

import gzip
import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np

# Comparable downstream analyses of reduced-pool exp022 cells use 10% of the
# untouched 10,000-image official MNIST test partition.
MNIST_REDUCED_EVAL_SAMPLES = 1_000

# ── SHD (Spiking Heidelberg Digits) ───────────────────────────────────────
# Class id → spoken word. SHD is digits 0-9 in German (labels 0-9) then English
# (labels 10-19), 20 classes total. Kept here so figures can title a raster with
# the word rather than a bare integer.
SHD_LABELS = [
    "null", "eins", "zwei", "drei", "vier",
    "fünf", "sechs", "sieben", "acht", "neun",
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
]
_SHD_DIR = "/tmp/shd"
_SHD_URLS = {
    "train": "https://zenkelab.org/datasets/shd_train.h5.gz",
    "test": "https://zenkelab.org/datasets/shd_test.h5.gz",
}


def _valid_h5(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            return "spikes" in handle and "labels" in handle
    except Exception:  # noqa: BLE001 — any unreadable HDF5 is a bad cache entry
        return False


def _shd_h5(split: str) -> str:
    """Fetch + gunzip one SHD split to _SHD_DIR, returning the local .h5 path.

    Mirrors the tool-side download (tools/snnsim/datasets.py) rather than importing
    it — same tool↔experiment boundary the mnist path respects. Reuses the cache
    the CLI may already have populated at /tmp/shd.  A stale/truncated HDF5 is
    deleted and rebuilt; this protects cloud runs from treating a partial
    download as a valid cache hit.
    """
    os.makedirs(_SHD_DIR, exist_ok=True)
    h5_path = os.path.join(_SHD_DIR, f"shd_{split}.h5")
    if _valid_h5(h5_path):
        return h5_path
    if os.path.exists(h5_path):
        os.unlink(h5_path)
    gz_path = h5_path + ".gz"
    for attempt in range(3):
        if not os.path.exists(gz_path):
            urllib.request.urlretrieve(_SHD_URLS[split], gz_path)
        tmp = h5_path + ".tmp"
        try:
            with gzip.open(gz_path, "rb") as f_in, open(tmp, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            Path(tmp).replace(h5_path)
            if _valid_h5(h5_path):
                return h5_path
        except Exception:  # noqa: BLE001 — retry after clearing partial cache
            pass
        for candidate in (tmp, h5_path, gz_path):
            if os.path.exists(candidate):
                os.unlink(candidate)
        if attempt == 2:
            break
    raise RuntimeError(f"failed to download a valid SHD {split!r} split after 3 attempts")
    return h5_path


def load_shd_events(split: str = "train", max_samples: int | None = None):
    """Return (events, labels) for one SHD split — raw spike events, not binned.

    events: object ndarray, events[i] = (units_int16, times_float32) — the spike
            unit index in [0, 700) and its time in seconds for utterance i.
    labels: int64 ndarray of class ids in [0, 20).

    Raw events are exactly what a raster needs; binning to a spike tensor is the
    trainer's job (see ShdBinnedDataset in tools/snnsim/train.py). Deterministic
    subset at seed 42 when max_samples is set.
    """
    import h5py

    with h5py.File(_shd_h5(split), "r") as f:
        times = f["spikes"]["times"]
        units = f["spikes"]["units"]
        labels = np.asarray(f["labels"], dtype=np.int64)
        events = np.empty(len(labels), dtype=object)
        for i in range(len(labels)):
            events[i] = (
                np.asarray(units[i], dtype=np.int16),
                np.asarray(times[i], dtype=np.float32),
            )
    if max_samples is not None and max_samples < len(labels):
        idx = np.random.RandomState(42).choice(len(labels), max_samples, replace=False)
        events, labels = events[idx], labels[idx]
    return events, labels


def load_mnist_split(max_samples: int | None = None):
    """Return official MNIST train/test partitions, optionally capping train.

    The cap never applies to the official 10,000-image test partition.
    """
    from torchvision import datasets, transforms

    tr = datasets.MNIST(root="/tmp/mnist", train=True, download=True,
                        transform=transforms.ToTensor())
    te = datasets.MNIST(root="/tmp/mnist", train=False, download=True,
                        transform=transforms.ToTensor())
    X = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y = tr.targets.numpy().astype(np.int64)
    X_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_te = te.targets.numpy().astype(np.int64)
    if max_samples is not None and max_samples < len(X):
        idx = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]
    return X, X_te, y, y_te
