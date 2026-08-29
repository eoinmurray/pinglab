"""Notebook-side dataset access (data, not CLI internals).

Mirrors the MNIST source partitions used by tools/snnsim: the official training
partition remains separate from the untouched official test partition.

Notebooks that only *run* the network never need this (the CLI loads data itself);
it exists for the few notebooks that must build custom stimuli from raw pixels
(e.g. nb048's sequential digit streams).
"""

from __future__ import annotations

import numpy as np

# Comparable downstream analyses of reduced-pool exp022 cells use 10% of the
# untouched 10,000-image official MNIST test partition.
MNIST_REDUCED_EVAL_SAMPLES = 1_000

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
