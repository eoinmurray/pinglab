"""Declare the one-to-one inhibitory background through a SNNLab initializer.

SNNLab's built-in constant initializer fills every matrix entry. This extension
instead gives each private background stream exactly one I target, with the
configured conductance jump in µS. It consumes no random numbers.

The versioned initializer is referenced by the compiled bundle and its native
snapshots. Registration happens during authoring; no model is built or executed
when this module is imported.
"""

from functools import lru_cache

import torch
from snnlab import extensions


def diagonal_background(shape, config, *, device, dtype):
    """Initialize a square source-by-target matrix without altering RNG state."""
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("private background requires equal source and target counts")
    return torch.eye(shape[0], device=device, dtype=dtype) * config["jump_us"]


@lru_cache(maxsize=1)
def register():
    extensions.register_initializer(
        "exp144.diagonal_background/v1", diagonal_background
    )
