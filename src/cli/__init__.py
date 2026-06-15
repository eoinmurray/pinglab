"""CLI package — entry-point at cli.py, runnable as `python -m cli`.

The submodules import their siblings by bare name (`import models as M`,
`from config import ...`). That resolves automatically when a module is run
as a script (its own directory is sys.path[0]) and under pytest (configured
via `pythonpath` in pyproject.toml). When the package is imported instead
(e.g. `from cli import theme` in a notebook runner), this __init__ runs first
and puts the package directory on sys.path so those bare imports still resolve.

This module also re-exports the symbols imported by tests and notebooks.
"""

import sys as _sys
from pathlib import Path as _Path

_pkg_dir = str(_Path(__file__).resolve().parent)
if _pkg_dir in _sys.path:
    _sys.path.remove(_pkg_dir)
_sys.path.insert(0, _pkg_dir)

from .cli import M  # noqa: E402,F401  # re-exported for tests/_apply_scan_var
from .encoders import FROZEN_MODES, FrozenEncoder  # noqa: E402,F401
from .cli import (  # noqa: E402,F401
    EVAL_SEED,
    _apply_scan_var,
    _auto_device,
    _extract_records,
    _load_dataset_image,
    _load_shd,
    _shd_cache_dir,
    downsample_spikes_count,
    encode_batch,
    encode_image_spikes,
    encode_images_poisson,
    encode_smnist,
    infer,
    load_dataset,
    parse_args,
    primary_hid_key,
    primary_inh_key,
    seed_everything,
    train,
    transport_spikes_bin,
    upsample_spikes_zeropad,
)
