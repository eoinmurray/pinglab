import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from tools.snnviz import (  # noqa: E402
    FrameTimeline,
    Recording,
    RecordingError,
    exponential_trace,
    grid_layout,
    representative_frame,
)
from tools.snnviz.loaders import load_snnsim_recording


def test_recording_validates_shared_timeline():
    recording = Recording(
        0.25, {"spk_e": np.zeros((8, 3)), "v_e": np.zeros((8, 3))}
    )
    assert recording.steps == 8
    assert recording.duration_ms == 2.0
    with pytest.raises(RecordingError):
        Recording(
            0.25, {"a": np.zeros((8, 1)), "b": np.zeros((7, 1))}
        )


def test_grid_and_trace_are_backend_independent():
    assert grid_layout(5, columns=3).shape == (5, 2)
    events = np.zeros((4, 2))
    events[0, 0] = 1
    trace = exponential_trace(events, dt_ms=1, tau_ms=2)
    assert trace[1, 0] == pytest.approx(1)
    assert trace[2, 0] == pytest.approx(np.exp(-0.5))


def test_timeline_and_representative_frame():
    timeline = FrameTimeline.sample(100, frames=10, dt_ms=0.25)
    assert timeline.steps[[0, -1]].tolist() == [0, 99]
    activity = np.asarray([[0, 0], [1, 0], [1, 2]])
    assert representative_frame(activity) == 2


def test_timeline_composes_slow_motion_repeats_and_holds():
    timeline = FrameTimeline.compose(
        [(0, 9, 3), (5, 5, 2), (9, 0, 3)], dt_ms=0.25
    )
    assert timeline.steps.tolist() == [0, 4, 9, 5, 5, 9, 4, 0]


def test_snnsim_loader_separates_retained_static_arrays(tmp_path):
    np.savez(
        tmp_path / "snapshot.npz",
        dt=np.asarray(0.25),
        spk_e=np.zeros((8, 3)),
        input_excitatory_e_executed=np.zeros((8, 3)),
        input_excitatory_e_rate_scale=np.ones(3),
    )

    recording = load_snnsim_recording(tmp_path)

    assert recording.steps == 8
    assert "input_excitatory_e_executed" in recording.signals
    assert "input_excitatory_e_rate_scale" not in recording.signals
    np.testing.assert_array_equal(
        recording.metadata["retained_static"]["input_excitatory_e_rate_scale"],
        np.ones(3),
    )
