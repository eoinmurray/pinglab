"""Physical weights, independent counts and delayed chunked execution."""

import numpy as np
import torch
from experiments.exp099 import recipe
from experiments.exp099.compute import build_model


def small_cfg():
    return {**recipe.configuration(), "n_e": 40, "n_i": 10, "t_ms": 1000.0}


def test_physical_weights_and_refractory_durations():
    cfg = small_cfg()
    _, model = build_model(cfg)
    for p in model.plan.projections:
        assert p.delay_steps == 15
        w = model.parameter_map()[p.parameter].detach().numpy()
        if p.id.startswith("private"):
            np.testing.assert_allclose(w, np.eye(len(w)) * 0.004)
        else:
            np.testing.assert_allclose(
                w[w > 0], 0.00125 if p.id.startswith("E") else 0.00334
            )
    for row, duration in zip(model.plan.populations, (3.0, 1.5)):
        assert row["neuron"]["refractory_steps"] * cfg["dt_ms"] == duration


def test_count_superposition_keeps_multiplets_and_independence():
    cfg = small_cfg()
    counts = recipe.afferent_counts(cfg)
    for array in counts.values():
        assert array.max() >= 2
        assert abs(array.mean() - 0.024) < 0.002
        assert abs(np.corrcoef(array[:, 0], array[:, 1])[0, 1]) < 0.04
    assert not np.array_equal(counts["private_e"][:, :10], counts["private_i"])


def test_diagonal_count_pulse_and_delay_survive_chunk_boundary():
    torch.set_num_threads(1)
    _, model = build_model(small_cfg())
    drive = {"private_e": torch.zeros(25, 1, 40), "private_i": torch.zeros(25, 1, 10)}
    drive["private_e"][3, 0, 2] = 2
    fields = ["private_e_to_E.conductance", "E.voltage", "E.spikes"]
    with torch.inference_mode():
        full = model(drive, recording_fields=fields)
        first = model({k: v[:10] for k, v in drive.items()}, recording_fields=fields)
        second = model(
            {k: v[10:] for k, v in drive.items()},
            recording_fields=fields,
            runtime_state=first.runtime_state,
        )
    g = full.recordings[fields[0]][:, 0].numpy()
    assert not g[:18].any()
    np.testing.assert_allclose(g[18, 2], 0.008)
    assert np.count_nonzero(g[18]) == 1
    for key in fields:
        torch.testing.assert_close(
            full.recordings[key],
            torch.cat([first.recordings[key], second.recordings[key]]),
            rtol=0,
            atol=0,
        )


def test_paper_passive_scale_is_authored_in_graph():
    _, model = build_model(small_cfg())
    for pop in model.plan.populations:
        neuron = pop["neuron"]
        assert neuron["capacitance_nf"] == 0.15
        assert neuron["leak_us"] == 0.01
        assert neuron["tau_mem"]["value"] == 15.0


def test_visual_grid_and_recording_contract():
    import pytest
    from experiments.exp099.render import frame_grid
    from tools.snnviz import Recording, RecordingError  # noqa: TID251

    grid, response = frame_grid()
    assert grid.rect("A").height > grid.rect("B").height
    assert response.rect("D").y > response.rect("E").y
    with pytest.raises(RecordingError):
        Recording(0.1, {"e": np.zeros((10, 2)), "i": np.zeros((9, 1))})
