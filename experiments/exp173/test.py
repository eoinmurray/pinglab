"""Non-simulation checks for intervention and rate measurement semantics."""

import numpy as np
import pytest
import torch
from experiments.exp173 import recipe
from experiments.exp173.analyse import summarize


def source_weights():
    return {
        "W_ff.0": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "W_ff.1": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "W_ei.1": torch.arange(3 * 256, dtype=torch.float32).reshape(3, 256),
        "W_ie.1": torch.arange(256 * 3, dtype=torch.float32).reshape(256, 3),
    }


def test_control_is_exact_and_subsets_share_the_same_neurons():
    source = source_weights()
    control, indices = recipe.resize_weights(source, 256)
    np.testing.assert_array_equal(indices, np.arange(256))
    for name, key in recipe.MAPPING.items():
        assert torch.equal(control[name], source[key])
    previous = set()
    for size in (32, 64, 128):
        reduced, ids = recipe.resize_weights(source, size)
        assert previous.issubset(set(ids))
        previous = set(ids)
        assert torch.equal(reduced["input_to_E.weight"], source["W_ff.0"])
        assert torch.equal(reduced["readout_projection.weight"], source["W_ff.1"])
        assert torch.equal(reduced["E_to_I.weight"], source["W_ei.1"][:, ids])
        assert torch.equal(
            reduced["I_to_E.weight"], source["W_ie.1"][ids] * (256 / size)
        )


def test_rate_normalization_is_independent_of_pool_size():
    counts = np.array([[2, 4], [4, 6]])
    reference = summarize(counts, 10, 200)
    larger = summarize(counts * 2, 20, 200)
    assert reference["mean_hz"] == larger["mean_hz"] == 2
    assert reference["total_spikes"] == 16
    assert reference["images"] == 2 and reference["draws"] == 2
    with pytest.raises(ValueError):
        summarize(np.array([[-1]]), 10, 200)


def test_accuracy_subset_matches_manuscript_order_and_partition_bounds():
    selected = recipe.test_sample_indices()
    assert len(selected) == len(np.unique(selected)) == 1000
    np.testing.assert_array_equal(selected[:5], [6252, 4684, 1731, 4742, 4521])
    assert selected.min() >= 0 and selected.max() < 10000
    np.testing.assert_array_equal(selected, recipe.test_sample_indices())
    with pytest.raises(ValueError):
        recipe.test_sample_indices(samples=10001)
