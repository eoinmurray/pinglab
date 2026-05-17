"""Oscilloscope package — CLI entry-point at __main__.py.

Re-exports the symbols imported by tests and notebooks. The CLI is invoked
as `python src/pinglab/cli/__main__.py …` (or via the runners,
which already point at __main__.py).
"""

from .__main__ import M  # noqa: F401  # re-exported for tests/_apply_scan_var
from .__main__ import (  # noqa: F401
    EVAL_SEED,
    FROZEN_MODES,
    FrozenEncoder,
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
