"""Dataset loaders — MNIST.

Pulled out of cli.py. The single
canonical entry point is `load_dataset(name, ...)`. Image-mode notebooks
(snapshot rendering) use `_load_dataset_image()` for a single sample.
"""

from __future__ import annotations

import numpy as np

# Smart per-dataset hidden-size defaults (used by build_net auto-config).
DATASET_N_HIDDEN_DEFAULTS = {
    "mnist": 1024,  # n_in = 784, next pow2
}


def load_dataset(name, max_samples=None, split=False, dt_ms=None, t_ms=None):
    """Load full dataset as (X, y) numpy arrays in [0, 1] / int64.

    Args:
        name: "mnist"
        max_samples: optional cap; deterministic random subset (seed 42)
        split: if True, return (X_tr, X_te, y_tr, y_te) via stratified 80/20
               train_test_split(seed 42); otherwise return (X, y)
        dt_ms, t_ms: currently unused; kept for call-site compatibility

    Single canonical loader used by train, infer, and image paths so
    "first digit-0 sample" means the same physical sample everywhere.
    """
    if name == "mnist":
        from torchvision import datasets, transforms

        mnist_train = datasets.MNIST(
            root="/tmp/mnist",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        mnist_test = datasets.MNIST(
            root="/tmp/mnist",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        X = np.concatenate(
            [
                mnist_train.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0,
                mnist_test.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0,
            ]
        )
        y = np.concatenate(
            [
                mnist_train.targets.numpy(),
                mnist_test.targets.numpy(),
            ]
        ).astype(np.int64)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if max_samples is not None and max_samples < len(X):
        idx = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]

    if split:
        from sklearn.model_selection import train_test_split

        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X, y


def _load_dataset_image(dataset="mnist", digit_class=0, sample_idx=0):
    """Load a single image from a dataset. Returns (pixel_vec, digit_image)."""
    if dataset == "mnist":
        from torchvision import datasets, transforms

        mnist = datasets.MNIST(
            root="/tmp/mnist",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        data = mnist.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        targets = mnist.targets.numpy()
        images = mnist.data.numpy()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    idx = np.where(targets == digit_class)[0][sample_idx]
    return data[idx], images[idx]
