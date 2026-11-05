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
        assert ".scratch/gaussian-selection" not in text


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
