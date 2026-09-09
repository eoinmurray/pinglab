"""Physical weights, independent counts and delayed chunked execution."""

import numpy as np
import torch
from experiments.exp099 import recipe
from experiments.exp099.compute import build_model


def small_cfg():
    return {**recipe.configuration(), "n_e": 40, "n_i": 10, "t_ms": 150.0}


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
                w[w > 0], 0.001 if p.id.startswith("E") else 0.00334
            )
    for row, duration in zip(model.plan.populations, (3.0, 1.5)):
        assert row["neuron"]["refractory_steps"] * cfg["dt_ms"] == duration


def test_count_superposition_keeps_multiplets_and_independence():
    cfg = {**small_cfg(), "t_ms": 1000.0, "stimulus_e_hz": 0.8, "stimulus_i_hz": 0.8}
    counts = recipe.afferent_counts(cfg)
    for array in counts.values():
        assert array.max() >= 2
        assert abs(array.mean() - 0.032) < 0.002
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


def test_pulse_schedule_and_epochs():
    from experiments.exp099.analyse import measure

    cfg = {**small_cfg(), "t_ms": recipe.DURATION_MS}
    e, i = recipe.source_rates(np.array([0, 700, 725, 750, 850, 875, 900, 1099.9]), cfg)
    np.testing.assert_allclose(e, [0.8, 0.8, 1.0, 1.2, 1.2, 1.0, 0.8, 0.8])
    np.testing.assert_allclose(i, e)
    old = {k: v for k, v in cfg.items() if k != "stimulus_i_hz"}
    _, old_i = recipe.source_rates(np.array([0, 500, 1000, 1499.9]), old)
    np.testing.assert_allclose(old_i, 0.8)
    steps = round(cfg["t_ms"] / cfg["dt_ms"])
    data = {f"spk_{p}": np.zeros((steps, cfg[f"n_{p}"]), bool) for p in ("e", "i")}
    for name in (
        "mean_E_to_E",
        "mean_private_e_to_E",
        "mean_I_to_E",
        "mean_v_e",
        "mean_v_i",
    ):
        data[name] = np.zeros(steps)
    # Burn-in spikes must not enter any visible epoch.
    for p in ("e", "i"):
        data[f"spk_{p}"][: round(cfg["burn_in_ms"] / cfg["dt_ms"])] = True
    _, result = measure(data, cfg)
    assert cfg["view_end_ms"] - cfg["view_start_ms"] == 600
    assert all(
        epoch["e_hz"] == epoch["i_hz"] == 0 for epoch in result["epochs"].values()
    )
    assert set(result["epochs"]) == {"visible_baseline", "plateau", "recovery"}
