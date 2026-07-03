"""Dataset loaders — MNIST/sMNIST, SHD.

Pulled out of cli.py. The single
canonical entry point is `load_dataset(name, ...)`. Image-mode notebooks
(snapshot rendering) use `_load_dataset_image()` for a single sample.
"""

from __future__ import annotations

import numpy as np

SHD_N_CHANNELS = 700

# Smart per-dataset hidden-size defaults (used by build_net auto-config).
DATASET_N_HIDDEN_DEFAULTS = {
    "mnist": 1024,  # n_in = 784, next pow2
    "smnist": 32,  # n_in = 28, next pow2
    "shd": 256,  # n_in = 700; 256 keeps the ladder tractable locally
}


def _shd_cache_dir():
    """Root dir for SHD HDF5 files. Override with $PINGLAB_SHD_DIR."""
    import os

    return os.environ.get("PINGLAB_SHD_DIR", "/tmp/shd/SHD")


def _load_shd(dt_ms, t_ms, max_samples=None):
    """Load SHD as a dense (N, T_steps, 700) float32 spike tensor + int64 labels.

    Reads the official train+test H5 files, concatenates them, then bins
    each sample's (time, unit) events into the grid at dt. `max_samples`
    is a per-file cap applied before binning — keeps the smoke-test cost
    bounded without reading the full 10k-sample set.
    """
    from pathlib import Path

    import h5py

    root = Path(_shd_cache_dir())
    train_path = root / "shd_train.h5"
    test_path = root / "shd_test.h5"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"SHD HDF5 files not found at {root}. Download shd_train.h5 + "
            f"shd_test.h5 from https://zenkelab.org/datasets/ or set "
            f"$PINGLAB_SHD_DIR to an existing cache."
        )
    T_steps = int(round(t_ms / dt_ms))
    dt_s = dt_ms / 1000.0
    xs, ys = [], []
    per_file_cap = max_samples if max_samples is not None else None
    for path in (train_path, test_path):
        with h5py.File(path, "r") as f:
            labels = f["labels"][:].astype(np.int64)
            n = len(labels) if per_file_cap is None else min(per_file_cap, len(labels))
            times_ds = f["spikes/times"]
            units_ds = f["spikes/units"]
            X = np.zeros((n, T_steps, SHD_N_CHANNELS), dtype=np.float32)
            for i in range(n):
                t_idx = np.clip((times_ds[i] / dt_s).astype(np.int64), 0, T_steps - 1)
                u_idx = np.clip(units_ds[i].astype(np.int64), 0, SHD_N_CHANNELS - 1)
                # multiple events per cell → clip to 1 spike (dense binary raster)
                X[i, t_idx, u_idx] = 1.0
            xs.append(X)
            ys.append(labels[:n])
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def load_dataset(name, max_samples=None, split=False, dt_ms=None, t_ms=None):
    """Load full dataset as (X, y) numpy arrays in [0, 1] / int64.

    Args:
        name: "mnist" | "smnist" | "shd"
              - smnist is mnist data, encoded row-by-row at run time
              - shd is natively event-based; X shape is (N, T_steps, 700)
        max_samples: optional cap; deterministic random subset (seed 42)
        split: if True, return (X_tr, X_te, y_tr, y_te) via stratified 80/20
               train_test_split(seed 42); otherwise return (X, y)
        dt_ms, t_ms: required for "shd" (event binning grid); ignored otherwise

    Single canonical loader used by train, infer, and image/video paths so
    "first digit-0 sample" means the same physical sample everywhere.
    """
    if name in ("mnist", "smnist"):
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
    elif name == "shd":
        if dt_ms is None or t_ms is None:
            raise ValueError("load_dataset('shd', ...) requires dt_ms and t_ms")
        X, y = _load_shd(dt_ms=dt_ms, t_ms=t_ms, max_samples=max_samples)
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
    if dataset in ("mnist", "smnist"):
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
