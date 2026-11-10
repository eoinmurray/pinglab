"""Focused checks of spatial mapping and the declared selection estimator."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from experiments.exp172.analyse import copy_fidelity, fit_profile, spatial_timecourse
from experiments.exp172.recipe import configuration, gaussian, topographic


def test_relative_phase_samples_c_cycles_and_preserves_accumulated_drift():
    from experiments.exp172.present import relative_phases

    spikes = {}
    for population, period in zip("ABC", [20, 24, 25], strict=True):
        values = np.zeros((800, 1))
        values[period:790:period] = 1
        spikes[population + "_E"] = values
    times, relative = relative_phases(spikes, dict(dt_ms=1))
    np.testing.assert_array_equal(times, np.arange(25, 790, 25))
    for population, period in [("A", 20), ("B", 24)]:
        valid = times <= np.flatnonzero(spikes[population + "_E"][:, 0])[-1]
        np.testing.assert_allclose(
            relative[population][valid] / (2 * np.pi),
            times[valid] * (1 / period - 1 / 25),
            atol=1e-12,
        )
        assert np.isnan(relative[population][~valid]).all()
    assert relative["A"][-1] / (2 * np.pi) > 1
    spikes["C_E"][:] = 0
    times, relative = relative_phases(spikes, dict(dt_ms=1))
    assert times.size == 0
    assert all(values.size == 0 for values in relative.values())


def test_inhibition_phase_reference_has_correct_sign_and_native_delay_shift():
    from experiments.exp172.analyse import measured_inhibition_profile

    time = np.arange(800, dtype=float)
    inhibition = 1 - np.cos(2 * np.pi * (time - 18) / 20)
    cycles = [
        dict(peak_ms=t, start_ms=t - 10, end_ms=t + 10) for t in np.arange(20, 780, 20)
    ]
    phase = np.linspace(-0.5, 0.5, 201)
    for delay, expected in [(0, 0.1), (2, 0.2)]:
        profile, kept = measured_inhibition_profile(
            time,
            inhibition,
            cycles,
            [[100, 700]],
            phase,
            delay,
        )
        assert len(kept) > 0
        assert phase[np.argmin(profile)] == pytest.approx(expected)


def test_probe_response_window_uses_positive_half_peak_and_wraps():
    from experiments.exp172.analyse import probe_response_window

    phase = np.linspace(-0.5, 0.5, 5)
    measured = probe_response_window(phase, [-4, 0, 4, 0, -4], 0.5)
    assert measured["threshold_extra_spikes"] == 2
    np.testing.assert_allclose(measured["bounds_cycles"], [[-0.125, 0.125]])
    measured = probe_response_window(phase, [4, 0, -1, 0, 4], 0.5)
    np.testing.assert_allclose(
        measured["bounds_cycles"], [[-0.5, -0.375], [0.375, 0.5]]
    )
    assert probe_response_window(phase, [-1, 0, -1, 0, -1], 0.5)["bounds_cycles"] == []


def test_probe_adds_only_one_uniform_excitatory_projection():
    from experiments.exp172.recipe import author_network, probe_configuration

    cfg = configuration()
    base, _ = author_network(cfg)
    settings = probe_configuration()
    probed, _ = author_network(cfg, probe=settings)
    for kind in ["populations", "outputs"]:
        assert probed.graph[kind] == base.graph[kind]
    added = [p for p in probed.graph["projections"] if p["id"] == "probe_to_C_E"]
    assert len(added) == 1
    assert [
        p for p in probed.graph["projections"] if p["id"] != "probe_to_C_E"
    ] == base.graph["projections"]
    assert settings["conductance_us"] == 0.05
    assert settings["response_ms"] == 5


def test_long_representative_protocol_keeps_reference_and_circuit_settings():
    from experiments.exp172.recipe import representative_configuration

    reference = configuration()
    cfg = representative_configuration(reference)
    assert (cfg["duration_ms"], cfg["switch_ms"]) == (5000, 2500)
    assert cfg["stability_search"]["windows_ms"] == [[250, 2400], [2750, 4900]]
    assert cfg["windows_ms"] == [[1950, 2400], [4450, 4900]]
    assert cfg["network"] == reference["network"]
    assert cfg["background_hz"] == reference["background_hz"]
    assert reference["duration_ms"] == 2400 and reference["switch_ms"] == 1200
    assert reference["stability_search"]["windows_ms"] == [[250, 1100], [1450, 2300]]


def test_natural_volley_plan_uses_observed_packets_and_the_plotted_phase_reference():
    from experiments.exp172.analyse import natural_volley_plan

    cfg = configuration(dt_ms=1, duration_ms=800, switch_ms=400)
    cfg["stability_search"]["windows_ms"] = [[100, 350], [450, 750]]
    spikes = {}
    for sender, first in [("A", 20), ("B", 24), ("C", 22)]:
        values = np.zeros((800, 1), dtype=np.uint8)
        values[first:780:20] = 1
        spikes[sender + "_E"] = values
    plan = natural_volley_plan(spikes, cfg)
    assert plan and len({row["id"] for row in plan}) == len(plan)
    for row in plan:
        assert row["packet_spikes"] == 1
        assert row["phase_cycles"] == pytest.approx(
            0.1 if row["sender"] == "A" else -0.1
        )
        assert (
            row["start_step"]
            < row["packet_start"]
            < row["packet_end"]
            <= row["end_step"]
        )
        assert row["response_start"] < row["response_end"] < row["response_end_two"]


def test_used_window_excludes_sparse_and_nonpersistent_effects():
    from experiments.exp172.analyse import used_phase_summary
    from experiments.exp172.recipe import used_window_configuration

    events = [
        dict(phase_cycles=0.1, persistent_effect=value, sender="A", attended="A")
        for value in [2, 3, 4]
    ]
    # A large response at a single rare phase cannot establish an actually used band.
    events.append(
        dict(phase_cycles=0.31, persistent_effect=100, sender="B", attended="B")
    )
    events.extend(
        dict(phase_cycles=-0.2, persistent_effect=-1, sender="B", attended="B")
        for _ in range(4)
    )
    result = used_phase_summary(events, used_window_configuration())
    np.testing.assert_allclose(result["bounds_cycles"], [[0.0625, 0.125]])
    assert result["peak_effect"] == 3
    assert result["threshold_effect"] == 1.5


def test_spatial_count_response_is_invariant_to_within_cycle_spike_timing():
    from experiments.exp172.analyse import spatial_count_amplitudes

    cfg = configuration()
    values = np.zeros((40, 400))
    values[5, 70:130] = 1
    moved = np.roll(values, 15, axis=0)
    np.testing.assert_allclose(
        spatial_count_amplitudes(values, cfg), spatial_count_amplitudes(moved, cfg)
    )
    assert (
        spatial_count_amplitudes(values, cfg)[0]
        > spatial_count_amplitudes(values, cfg)[1]
    )


def test_low_inhibition_window_interpolates_crossings_without_using_dominance():
    from experiments.exp172.analyse import low_inhibition_intervals

    phase = np.linspace(-0.5, 0.5, 5)
    measured = low_inhibition_intervals(phase, [4, 2, 0, 2, 4], 0.5)
    np.testing.assert_allclose(measured["bounds_cycles"], [[-0.25, 0.25]])
    assert measured["threshold_us"] == 2
    measured = low_inhibition_intervals(phase, [0, 2, 4, 2, 0], 0.5)
    np.testing.assert_allclose(measured["bounds_cycles"], [[-0.5, -0.25], [0.25, 0.5]])
    assert low_inhibition_intervals(phase, [1] * 5, 0.5)["bounds_cycles"] == []
    with pytest.raises(ValueError):
        low_inhibition_intervals(phase, [0, 1, np.nan, 1, 0], 0.5)


def test_cycle_selection_success_and_flicker_are_independent():
    from experiments.exp172.analyse import cycle_selection_scores

    def scores(a, b):
        return cycle_selection_scores(
            [
                dict(attended="A", cycles=[dict(dominant=s) for s in a]),
                dict(attended="B", cycles=[dict(dominant=s) for s in b]),
            ]
        )

    perfect = scores(["A"] * 3, ["B"] * 7)
    assert perfect["switching_success"] == 1
    assert perfect["flicker_fraction"] == 0
    stuck = scores(["A"] * 3, ["A"] * 7)
    assert stuck["switching_success"] == 0.5
    assert stuck["flicker_fraction"] == 0
    wrong = scores(["B"] * 3, ["A"] * 7)
    assert wrong["switching_success"] == 0
    assert wrong["flicker_fraction"] == 0
    alternating = scores(["A", "B", "A", "B"], ["B", "A", "B", "A"])
    assert alternating["switching_success"] == 0.5
    assert alternating["flicker_fraction"] == 1
    interrupted = scores(["A", None, "B"], ["B", "B"])
    assert interrupted["eligible_pairs"] == 1
    assert interrupted["flicker_fraction"] == 0
    assert interrupted["switching_success"] == pytest.approx(2 / 3)
    assert scores([None], [None])["switching_success"] == 0
    assert scores([None], [None])["flicker_fraction"] is None
    assert scores([], ["B"])["switching_success"] is None


def test_cycle_latency_reports_run_onset_and_resets_after_unresolved_cycles():
    from experiments.exp172.analyse import cycle_switch_latency

    def latency(states):
        cycles = [
            dict(start_ms=i * 20, end_ms=(i + 1) * 20, dominant=s)
            for i, s in enumerate(states)
        ]
        return cycle_switch_latency(cycles, 10, 3)

    immediate = latency(["A", "B", "B", "B"])
    assert immediate["cycles"] == 1
    assert immediate["onset_ms"] == 10
    assert immediate["confirmation_ms"] == 70
    assert latency(["B", "B", "B"])["cycles"] is None
    assert latency(["A", "B", None, "B", "B", "B"])["cycles"] == 3
    assert latency(["A", "B", "A", "B", "B"])["cycles"] is None
    cycles = [
        dict(start_ms=i * 20, end_ms=(i + 1) * 20, dominant="A") for i in range(4)
    ]
    assert cycle_switch_latency(cycles, 0, 3, attended="A", end_ms=60)["cycles"] == 1
    assert cycle_switch_latency(cycles, 0, 3, attended="A", end_ms=50)["cycles"] is None


def test_known_mixture_and_silent_response():
    x = np.arange(400) / 400
    centres = [0.25, 0.75]
    y = 7 + 30 * gaussian(x, 0.25, 0.09) + 10 * gaussian(x, 0.75, 0.09)
    fit, _, _ = fit_profile(y, centres, 0.09, 1.0)
    np.testing.assert_allclose(fit["amplitudes_hz"], [30, 10], atol=1e-8)
    assert abs(fit["fraction_A"] - 0.75) < 1e-10
    swapped, _, _ = fit_profile(y, centres[::-1], 0.09, 1.0)
    assert abs(swapped["fraction_A"] - 0.25) < 1e-10
    silent, _, _ = fit_profile(np.zeros(400), centres, 0.09, 1.0)
    assert silent["fraction_A"] is None


def test_topography_preserves_incoming_totals_and_sender_symmetry():
    original = {
        "sensory_A.weight": torch.arange(1, 25, dtype=torch.float32).reshape(6, 4),
        "sensory_B.weight": torch.ones(6, 4),
        "A_E_to_C_E.weight": torch.ones(4, 4) * 2,
        "B_E_to_C_E.weight": torch.ones(4, 4) * 2,
        "C_I_to_E.weight": torch.ones(2, 4),
    }
    result = topographic(
        original, dict(sensory_kernel_sigma=0.08, cross_kernel_sigma=0.08)
    )
    for key in original:
        torch.testing.assert_close(result[key].sum(0), original[key].sum(0))
    torch.testing.assert_close(result["A_E_to_C_E.weight"], result["B_E_to_C_E.weight"])
    torch.testing.assert_close(result["C_I_to_E.weight"], original["C_I_to_E.weight"])
    # A spatial source at index 1 preferentially reaches target index 1.
    assert result["A_E_to_C_E.weight"][1].argmax().item() == 1


def test_disconnected_receiver_gain_compression_stays_finite():
    original = {
        "sensory_A.weight": torch.ones(6, 4),
        "sensory_B.weight": torch.ones(6, 4),
        "sensory_C.weight": torch.arange(1, 25, dtype=torch.float32).reshape(6, 4),
        "A_E_to_C_E.weight": torch.ones(4, 4),
        "B_E_to_C_E.weight": torch.ones(4, 4),
        "C_E_to_E.weight": torch.ones(4, 4),
        "C_I_to_E.weight": torch.ones(2, 4),
    }
    original["A_E_to_C_E.weight"].zero_()
    original["B_E_to_C_E.weight"].zero_()
    result = topographic(
        original,
        dict(
            sensory_kernel_sigma=0.08,
            cross_kernel_sigma=0.08,
            c_gain_scale=0.25,
        ),
    )
    assert all(torch.isfinite(v).all() for v in result.values())
    assert result["A_E_to_C_E.weight"].count_nonzero() == 0
    assert result["B_E_to_C_E.weight"].count_nonzero() == 0


def test_copy_score_penalises_background_and_gain_errors():
    profile = np.repeat(np.array([10.0, 20.0, 30.0, 20.0]), 8)
    assert copy_fidelity(profile, profile)["copy_r_squared"] == 1.0
    assert copy_fidelity(profile, np.full_like(profile, 20.0))["copy_r_squared"] == 0.0
    scaled = copy_fidelity(profile, 2 * profile)
    assert abs(scaled["correlation"] - 1) < 1e-12
    assert scaled["copy_r_squared"] < 0


def test_configuration_is_fresh_and_checks_overrides():
    a = configuration()
    a["network"]["populations"]["E"]["size"] = 3
    assert configuration()["network"]["populations"]["E"]["size"] == 400
    assert configuration(dt_ms=0.2)["network"]["dt_ms"] == 0.2
    for overrides in [
        dict(unknown=1),
        dict(cue_strengths_hz=[3]),
        dict(spatial_mean_hz=-1),
    ]:
        with pytest.raises(ValueError):
            configuration(**overrides)


def test_imports_do_not_execute_or_touch_scratch():
    root = Path(__file__).resolve().parents[2]
    code = """
import pathlib
from unittest.mock import patch
with patch('snnlab.sim.simulate',side_effect=AssertionError('simulation on import')):
    import experiments.exp172.recipe
    import experiments.exp172.compute
    import experiments.exp172.analyse
    import experiments.exp172.present
"""
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)
    for name in ["recipe.py", "compute.py", "analyse.py", "present.py"]:
        text = (root / "experiments/exp172" / name).read_text()
        assert "from run import" not in text
        assert ".scratch/" not in text
        assert '".scratch"' not in text
        assert "'.scratch'" not in text


def test_spatial_timecourse_recovers_known_components():
    cfg = configuration(dt_ms=1.0, duration_ms=200.0, switch_ms=100.0)
    cfg["network"]["populations"]["E"]["size"] = 400
    x = np.arange(400) / 400
    sigma = np.sqrt(
        sum(
            cfg[k] ** 2
            for k in ["spatial_sigma", "sensory_kernel_sigma", "cross_kernel_sigma"]
        )
    )
    # Analytic rate-valued input checks the estimator, without neural simulation.
    rate = 7 + 30 * gaussian(x, 0.25, sigma) + 10 * gaussian(x, 0.75, sigma)
    spikes = {"C_E": np.tile(rate / 1000, (200, 1))}
    result = spatial_timecourse(spikes, cfg, False)
    assert result["window_ms"] == 100
    for row in result["points"]:
        np.testing.assert_allclose(
            [row["background_hz"], row["A_amplitude_hz"], row["B_amplitude_hz"]],
            [7, 30, 10],
            atol=1e-10,
        )


def test_stability_distinguishes_correct_switch_from_wrong_spatial_assignment():
    from experiments.exp172.analyse import stability_metrics

    cfg = configuration(dt_ms=1.0)
    x = np.arange(400) / 400
    sigma = np.sqrt(
        sum(
            cfg[k] ** 2
            for k in ["spatial_sigma", "sensory_kernel_sigma", "cross_kernel_sigma"]
        )
    )
    patterns = [gaussian(x, c, sigma) for c in cfg["centres"]]
    rates = np.vstack(
        [
            np.tile(7 + 30 * patterns[0], (1200, 1)),
            np.tile(7 + 30 * patterns[1], (1200, 1)),
        ]
    )
    correct = stability_metrics({"C_E": rates / 1000}, cfg)
    assert correct["worst_window_wrong_fraction"] == 0
    assert correct["switch_latency_ms"] == 40
    wrong = stability_metrics({"C_E": rates / 1000}, cfg, swap=True)
    assert wrong["worst_window_wrong_fraction"] == 1
    assert wrong["switch_latency_ms"] is None


def test_whole_volley_decoder_rejects_split_half_artifact_and_keeps_real_excursion():
    from experiments.exp172.analyse import cycle_selection_metrics
    from experiments.exp172.present import phase_evidence
    from scipy.optimize import nnls

    cfg = configuration(dt_ms=0.1, duration_ms=800.0, switch_ms=400.0)
    cfg["stability_search"]["windows_ms"] = [[100.0, 350.0], [450.0, 750.0]]
    t = np.arange(8000) * cfg["dt_ms"]
    x = np.arange(400) / 400
    sigma = np.sqrt(
        sum(
            cfg[k] ** 2
            for k in ["spatial_sigma", "sensory_kernel_sigma", "cross_kernel_sigma"]
        )
    )
    patterns = [gaussian(x, c, sigma) for c in cfg["centres"]]
    design = np.column_stack([np.ones(400), *patterns])
    for real_excursion in [False, True]:
        rates = np.zeros((len(t), 400))
        for i, centre in enumerate(np.arange(50.0, 780.0, 22.0)):
            attended = 0 if centre < cfg["switch_ms"] else 1
            other = 1 - attended
            lead = -1.0 if i % 2 == 0 else 1.0
            correct, competitor = (
                (25.0, 30.0) if real_excursion and centre == 248.0 else (30.0, 25.0)
            )
            rates += (
                correct
                * np.exp(-0.5 * ((t - centre - lead) / 0.45) ** 2)[:, None]
                * patterns[attended]
            )
            rates += (
                competitor
                * np.exp(-0.5 * ((t - centre + lead) / 0.45) ** 2)[:, None]
                * patterns[other]
            )
        # Analytic rate profiles test only decoding; no neural simulation.
        values = rates * cfg["dt_ms"] / 1000
        measured = cycle_selection_metrics({"C_E": values}, cfg, sustained_cycles=3)
        a, b = measured["epochs"]
        if real_excursion:
            assert a["switching_success"] < 1
            assert a["flicker_fraction"] > 0
        else:
            assert a["switching_success"] == b["switching_success"] == 1
            assert a["flicker_fraction"] == b["flicker_fraction"] == 0
        for epoch in measured["epochs"]:
            for cycle in epoch["cycles"]:
                assert cycle["start_ms"] < cycle["peak_ms"] < cycle["end_ms"]
        # Demonstrate that peak-to-peak bins mislabel these same complete,
        # attended-dominant volleys when their spatial halves differ in timing.
        _, _, _, detected = phase_evidence(
            {"A_E": values, "B_E": values, "C_E": values}, cfg
        )
        wrong = 0
        for lo, hi in zip(detected["C"], detected["C"][1:], strict=False):
            if lo * cfg["dt_ms"] >= 100 and hi * cfg["dt_ms"] <= 350:
                _, aa, bb = nnls(design, values[lo:hi].mean(0) * 1000 / cfg["dt_ms"])[0]
                wrong += bb > aa
        assert wrong > 0


def test_whole_volley_decoder_handles_no_detected_cycles():
    from experiments.exp172.analyse import cycle_selection_metrics

    cfg = configuration(dt_ms=1.0, duration_ms=800.0, switch_ms=400.0)
    cfg["stability_search"]["windows_ms"] = [[100, 350], [450, 750]]
    measured = cycle_selection_metrics(
        {"C_E": np.zeros((800, 400))}, cfg, sustained_cycles=3
    )
    assert measured["cycle_counts"] == [0, 0]
    assert measured["switching_success"] is None
    assert measured["flicker_fraction"] is None
    assert measured["switching_latency"]["cycles"] is None


def test_report_scope_requires_every_planned_trial_and_records_historical_extras():
    from experiments.exp172.analyse import report_trial_scope
    from experiments.exp172.recipe import trial_plan

    cfg = configuration()
    trials = [dict(id=item["id"]) for item in trial_plan(cfg)]
    selected, excluded = report_trial_scope([*trials, dict(id="historical-extra")], cfg)
    assert selected == trials
    assert excluded == ["historical-extra"]
    with pytest.raises(ValueError, match="Incomplete trial grid"):
        report_trial_scope(trials[1:], cfg)
    with pytest.raises(ValueError, match="Duplicate trial"):
        report_trial_scope([*trials, trials[0]], cfg)


def test_packet_delay_conserves_each_channel_and_rejects_wrapped_spikes():
    from experiments.exp172.recipe import shifted_packet

    source = np.zeros((30, 2), dtype=np.uint8)
    source[6, 0] = 1
    source[9, 1] = 1
    settings = dict(packet_window_ms=[4, 20], delay_ms=4)
    delayed = shifted_packet(source, 1, settings)
    assert np.flatnonzero(delayed[:, 0]).tolist() == [10]
    assert np.flatnonzero(delayed[:, 1]).tolist() == [13]
    np.testing.assert_array_equal(delayed.sum(0), source.sum(0))
    np.testing.assert_array_equal(delayed[:4], source[:4])
    np.testing.assert_array_equal(delayed[20:], source[20:])
    source[18, 0] = 1
    with pytest.raises(ValueError, match="wrap"):
        shifted_packet(source, 1, settings)


def test_replay_architecture_changes_only_one_projection_source():
    from experiments.exp172.recipe import author_network

    cfg = configuration()
    ordinary, _ = author_network(cfg)
    replayed, _ = author_network(cfg, replay_sender="B")
    graph = replayed.graph
    port = next(i for i in graph["inputs"] if i["id"] == "replay_B")
    assert port["shape"][-1] == cfg["network"]["populations"]["E"]["size"]
    graph["inputs"].remove(port)
    pathway = next(p for p in graph["projections"] if p["id"] == "B_E_to_C_E")
    assert pathway["source"] == "replay_B.value"
    pathway["source"] = "B_E.spikes"
    assert graph == ordinary.graph


@pytest.mark.parametrize("control_key", ["background_hz", "cue_hz"])
def test_background_aggregation_equal_weights_seeds_and_preserves_missing_latency(
    control_key,
):
    from experiments.exp172.analyse import aggregate_background

    settings = dict(seeds=[1, 2, 3, 4, 5], rates_hz=[28.0])
    rows = []
    for i, seed in enumerate(settings["seeds"]):
        for att in "AB":
            rows.append(
                dict(
                    seed=seed,
                    **{control_key: 28.0},
                    attended=att,
                    S=[0.0, 1.0, 1.0, 1.0, 1.0][i],
                    F=i / 4,
                    latency_cycles=[None, 1, 3, 5, None][i],
                    C_E_hz=10 + i,
                    C_I_hz=20 + i,
                    cycles=100 if i == 0 else 1,
                )
            )
    aggregated = aggregate_background(rows, settings, control_key=control_key)
    for result in aggregated:
        assert result["S"]["mean"] == pytest.approx(0.8)
        assert result["F"]["mean"] == pytest.approx(0.5)
        assert result["F"]["sd"] == pytest.approx(
            np.std([0, 0.25, 0.5, 0.75, 1], ddof=1)
        )
        assert result["latency_cycles"] == dict(n=3, total=5, mean=3.0, sd=2.0)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        aggregate_background(rows[:-1], settings, control_key=control_key)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        aggregate_background([*rows, rows[0]], settings, control_key=control_key)


def test_seed_summary_does_not_turn_unobserved_results_into_zeroes():
    from experiments.exp172.analyse import seed_summary

    assert seed_summary([None] * 5) == dict(n=0, total=5, mean=None, sd=None)
    assert seed_summary([None, 2]) == dict(n=1, total=2, mean=2.0, sd=None)
    with pytest.raises(ValueError, match="Non-finite"):
        seed_summary([np.nan])


def test_five_seed_background_plan_is_complete_and_paired():
    from experiments.exp172.recipe import (
        background_sweep_configuration,
        background_sweep_plan,
    )

    settings = background_sweep_configuration()
    plan = background_sweep_plan(settings)
    assert len(plan) == len({item["id"] for item in plan}) == 45
    assert {item["seed"] for item in plan} == {185, 186, 187, 188, 189}
    for seed in settings["seeds"]:
        assert (
            sorted(i["background_hz"] for i in plan if i["seed"] == seed)
            == settings["rates_hz"]
        )
    assert all(i["cue_hz"] == 3 and i["active"] == "AB" and not i["swap"] for i in plan)


def test_circular_arrival_density_normalises_and_wraps_phase_boundaries():
    from experiments.exp172.analyse import circular_arrival_density

    phases = [-0.49, 0.49, -0.02, 0.01, 0.03]
    first = circular_arrival_density(phases, 24, 1.0)
    shifted = circular_arrival_density(np.array(phases) + 1, 24, 1.0)
    np.testing.assert_allclose(first["density"], shifted["density"])
    assert sum(first["density"]) / 24 == pytest.approx(1.0)
    assert first["n"] == 5
    assert 0 <= first["concentration"] <= 1
    assert first["density"][0] > 0 and first["density"][-1] > 0
    assert circular_arrival_density([], 24, 1.0)["density"] is None
    with pytest.raises(ValueError, match="Non-finite"):
        circular_arrival_density([np.nan], 24, 1.0)


def test_cue_sweep_selects_joint_background_and_preserves_control_scope():
    from experiments.exp172.recipe import (
        best_background,
        cue_sweep_configuration,
        cue_sweep_plan,
    )

    def summary(rate, a, b, flicker):
        return [
            dict(
                background_hz=rate,
                attended=att,
                S=dict(mean=value),
                F=dict(mean=flicker),
                latency_cycles=dict(mean=2.0),
            )
            for att, value in zip("AB", (a, b), strict=True)
        ]

    # A perfect response in just one attention period must not win the joint screen.
    rows = summary(28, 1.0, 0.2, 0.1) + summary(32, 0.7, 0.7, 0.2)
    assert best_background(rows) == 32
    settings = dict(cue_sweep_configuration(), background_hz=best_background(rows))
    plan = cue_sweep_plan(settings)
    assert len(plan) == len({r["id"] for r in plan}) == 35
    assert all(r["background_hz"] == 32 for r in plan)
    for seed in settings["seeds"]:
        assert (
            sorted(r["cue_hz"] for r in plan if r["seed"] == seed)
            == settings["rates_hz"]
        )


def test_contrast_cue_switches_excess_without_changing_background_or_uniform_cue():
    from experiments.exp172.recipe import contrast_rate_profiles, inputs

    cfg = configuration(dt_ms=1.0, duration_ms=40.0, switch_ms=20.0)
    cfg["network"]["input_channels"] = 64
    original, rates = inputs(cfg, 185, 0, False)
    equal, equal_rates = inputs(dict(cfg, contrast_ratio=1.0), 185, 0, False)
    for key in original:
        np.testing.assert_array_equal(original[key], equal[key])
    for key in ("X_A", "X_B"):
        np.testing.assert_array_equal(
            equal_rates[key], np.stack([rates[key], rates[key]])
        )
    stronger, changed = inputs(dict(cfg, contrast_ratio=3.0), 185, 0, False)
    for sender, attended_epoch in (("A", 0), ("B", 1)):
        key = "X_" + sender
        np.testing.assert_allclose(
            changed[key][attended_epoch] - cfg["spatial_baseline_hz"],
            3 * (rates[key] - cfg["spatial_baseline_hz"]),
        )
        np.testing.assert_array_equal(changed[key][1 - attended_epoch], rates[key])
        window = slice(0, 20) if attended_epoch == 0 else slice(20, 40)
        assert np.all(stronger[key][window] >= original[key][window])
    np.testing.assert_array_equal(stronger["X_C"], original["X_C"])
    assert not stronger["U"].any()
    with pytest.raises(ValueError):
        contrast_rate_profiles(cfg, 0.5)
