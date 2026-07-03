"""Coverage for datasets.py error branches.

SHD loader tests were removed — SHD is not a supported path yet. What remains
covers the unknown-dataset guards in load_dataset / _load_dataset_image.
"""

from __future__ import annotations

import pytest
from datasets import _load_dataset_image, load_dataset


def test_load_dataset_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("not_a_dataset")


def test_load_dataset_image_unknown_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        _load_dataset_image(dataset="not_a_dataset")
