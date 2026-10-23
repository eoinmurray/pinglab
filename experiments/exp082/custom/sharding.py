"""Authenticate exp082 shard reuse against its inputs and local implementation.

The compute stage uses this identity for worker resume and collection. Dataset,
runtime and source hashes reject reuse after scientific or execution changes.
"""

from pathlib import Path

import numpy as np
import torch
from experiments.exp082.custom import inference
from pingstore.contracts import file_sha256
from snnlab.sim import load_dataset


def shard_identity(cfg, bank, recipe_module, *, dataset=None):
    """Freeze expected scientific/execution content before shard reuse.

    ``auto`` remains an explicit device-selection request; actual devices are
    recorded separately in worker provenance and each inference request.
    This identity does not rely on an artifact validator's inferred coverage.
    """
    import hashlib
    import os
    from importlib.metadata import version

    import snnlab

    if dataset is None:
        _, pixels, _, labels = load_dataset(
            "mnist", split=True, evaluation_only=True, evaluation_split="test"
        )
    else:
        pixels, labels = dataset
    source = Path(snnlab.__file__).parent
    return {
        "recipe": cfg,
        "bank": bank.reference,
        "dataset": {
            "pixels": {
                "dtype": str(pixels.dtype),
                "shape": list(pixels.shape),
                "sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
            },
            "labels": {
                "dtype": str(labels.dtype),
                "shape": list(labels.shape),
                "sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            },
        },
        "runtime": {
            "snnlab": version("snnlab"),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "requested_device": os.environ.get("PINGLAB_DEVICE", "auto"),
            "cuda": torch.version.cuda,
            "default_dtype": str(torch.get_default_dtype()),
            "threads": torch.get_num_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        },
        "snnlab_source": {
            str(p.relative_to(source)): file_sha256(p)
            for p in sorted(source.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".json")
        },
        "experiment_source": {
            str(p): file_sha256(p)
            for p in sorted(Path(recipe_module.__file__).parent.rglob("*.py"))
        },
        "inference_source": file_sha256(Path(inference.__file__)),
    }
