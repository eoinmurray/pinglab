"""Non-simulation checks for retrospective threshold semantics and native recording reduction."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from experiments.exp174.analyse import summarize, threshold_index
from experiments.exp174.compute import PrefixCollector
from experiments.exp174.recipe import configuration


def test_threshold_uses_final_unbroken_suffix():
    assert threshold_index([0.7, 0.8, 0.79, 0.81, 0.8], 0.8) == 3
    assert threshold_index([0.8, 0.9, 0.79], 0.8) is None
    assert threshold_index([0.8, 0.9], 0.8) == 0


def test_budget_uses_all_images_and_current_step():
    cfg = dict(steps=3, samples=2, threshold_accuracy=0.8, dt_ms=0.1)
    predictions = np.array([[0, 0], [0, 1], [0, 1]])
    counts = np.array([[2, 4], [4, 8], [8, 16]])
    row, _ = summarize(
        predictions,
        np.array([0, 1]),
        {"e_spikes": counts, "i_spikes": counts * 0, "artificial_events": counts * 0},
        cfg,
    )
    assert row["latency_ms"] == 0.2
    assert row["e_spikes_at_threshold"] == 9


def test_recording_reduction_respects_blocks_and_batch_offsets():
    collector = PrefixCollector(4, 3, 2)
    collector(SimpleNamespace(kind="output"))
    for offset, width in ((0, 2), (2, 1)):
        for start in (0, 2):
            v = np.broadcast_to([1.0, 2.0], (2, width, 2)).copy()
            collector(
                SimpleNamespace(
                    signal="readout.pre_reset_voltage",
                    kind="dense",
                    values=torch.from_numpy(v),
                    start_step=start,
                    end_step=start + 2,
                    batch_offset=offset,
                )
            )
            for p in ("E", "I"):
                collector(
                    SimpleNamespace(
                        signal=p + ".spikes",
                        kind="dense",
                        values=torch.ones(2, width, 4),
                        start_step=start,
                        end_step=start + 2,
                        batch_offset=offset,
                    )
                )
    assert all(v.all() for v in collector.seen.values())
    np.testing.assert_array_equal(collector.predictions, np.ones((4, 3)))
    np.testing.assert_array_equal(collector.counts["E"], np.full((4, 3), 4))
    np.testing.assert_array_equal(collector.score_sum, [[4, 8]] * 3)


def test_configuration_is_fresh_and_prototype_is_bounded():
    a, b = configuration(), configuration()
    a["samples"] = 10
    assert b["samples"] == 1000
    with pytest.raises(ValueError):
        configuration(samples=10000)
