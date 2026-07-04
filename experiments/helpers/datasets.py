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

import numpy as np


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
