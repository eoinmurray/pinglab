"""Notebook-side dataset access (data, not CLI internals).

Mirrors the mnist path of cli.datasets.load_dataset EXACTLY (same torchvision
source, same concat order, same stratified 80/20 split at seed 42) so a notebook's
test split is the identical physical samples the trained network never saw. Kept in
sync with cli.datasets by hand — if the CLI's split recipe changes, update here too.

Notebooks that only *run* the network never need this (the CLI loads data itself);
it exists for the few notebooks that must build custom stimuli from raw pixels
(e.g. nb048's sequential digit streams).
"""

from __future__ import annotations

import gzip
import os
import shutil
import urllib.request

import numpy as np

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


def _shd_h5(split: str) -> str:
    """Fetch + gunzip one SHD split to _SHD_DIR, returning the local .h5 path.

    Mirrors the tool-side download (tools/snn/datasets.py) rather than importing
    it — same tool↔experiment boundary the mnist path respects. Reuses the cache
    the CLI may already have populated at /tmp/shd.
    """
    os.makedirs(_SHD_DIR, exist_ok=True)
    h5_path = os.path.join(_SHD_DIR, f"shd_{split}.h5")
    if os.path.exists(h5_path):
        return h5_path
    gz_path = h5_path + ".gz"
    if not os.path.exists(gz_path):
        urllib.request.urlretrieve(_SHD_URLS[split], gz_path)
    with gzip.open(gz_path, "rb") as f_in, open(h5_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return h5_path


def load_shd_events(split: str = "train", max_samples: int | None = None):
    """Return (events, labels) for one SHD split — raw spike events, not binned.

    events: object ndarray, events[i] = (units_int16, times_float32) — the spike
            unit index in [0, 700) and its time in seconds for utterance i.
    labels: int64 ndarray of class ids in [0, 20).

    Raw events are exactly what a raster needs; binning to a spike tensor is the
    trainer's job (see ShdBinnedDataset in tools/snn/train.py). Deterministic
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
    """Return (X_tr, X_te, y_tr, y_te) for MNIST, matching cli.datasets.load_dataset.

    X in [0, 1] float32 (N, 784); y int64. Stratified 80/20, random_state=42.
    """
    from sklearn.model_selection import train_test_split
    from torchvision import datasets, transforms

    tr = datasets.MNIST(root="/tmp/mnist", train=True, download=True,
                        transform=transforms.ToTensor())
    te = datasets.MNIST(root="/tmp/mnist", train=False, download=True,
                        transform=transforms.ToTensor())
    X = np.concatenate([
        tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0,
        te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0,
    ])
    y = np.concatenate([tr.targets.numpy(), te.targets.numpy()]).astype(np.int64)
    if max_samples is not None and max_samples < len(X):
        idx = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
