"""Focused checks of spatial mapping and the declared selection estimator."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from experiments.exp172.analyse import copy_fidelity, fit_profile
from experiments.exp172.present import spatial_timecourse
from experiments.exp172.recipe import configuration, gaussian, topographic


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
