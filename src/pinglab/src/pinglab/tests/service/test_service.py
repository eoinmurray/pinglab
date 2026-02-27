import numpy as np
import pytest

from pinglab.service.service import (
    _resolve_targeted_pulse_ids,
    run_simulation,
)


def test_resolve_targeted_pulse_ids_disabled_returns_full_populations() -> None:
    e_ids = np.arange(6, dtype=int)
    i_ids = np.arange(6, 10, dtype=int)

    selected_e, selected_i = _resolve_targeted_pulse_ids(
        e_ids=e_ids,
        i_ids=i_ids,
        targeted_subset_enabled=False,
        target_population="all",
        target_strategy="random",
        target_fraction=0.25,
        target_seed=7,
    )

    assert np.array_equal(selected_e, e_ids)
    assert np.array_equal(selected_i, i_ids)


def test_resolve_targeted_pulse_ids_random_subset_is_seeded() -> None:
    e_ids = np.arange(10, dtype=int)
    i_ids = np.arange(10, 20, dtype=int)

    selected_e_1, selected_i_1 = _resolve_targeted_pulse_ids(
        e_ids=e_ids,
        i_ids=i_ids,
        targeted_subset_enabled=True,
        target_population="all",
        target_strategy="random",
        target_fraction=0.3,
        target_seed=9,
    )
    selected_e_2, selected_i_2 = _resolve_targeted_pulse_ids(
        e_ids=e_ids,
        i_ids=i_ids,
        targeted_subset_enabled=True,
        target_population="all",
        target_strategy="random",
        target_fraction=0.3,
        target_seed=9,
    )

    assert np.array_equal(selected_e_1, selected_e_2)
    assert np.array_equal(selected_i_1, selected_i_2)
    assert selected_e_1.size == 3
    assert selected_i_1.size == 3


def test_resolve_targeted_pulse_ids_first_strategy_and_population_filter() -> None:
    e_ids = np.arange(5, dtype=int)
    i_ids = np.arange(5, 10, dtype=int)

    selected_e, selected_i = _resolve_targeted_pulse_ids(
        e_ids=e_ids,
        i_ids=i_ids,
        targeted_subset_enabled=True,
        target_population="e",
        target_strategy="first",
        target_fraction=0.4,
        target_seed=0,
    )

    assert np.array_equal(selected_e, np.array([0, 1]))
    assert selected_i.size == 0


def test_run_simulation_accepts_sine_input_type() -> None:
    result = run_simulation(
        config_overrides={"T": 80.0, "dt": 1.0, "N_E": 20, "N_I": 10},
        inputs_overrides={
            "input_type": "sine",
            "I_E_base": 0.2,
            "I_I_base": 0.1,
            "noise_std_E": 0.0,
            "noise_std_I": 0.0,
            "seed": 3,
            "sine_freq_hz": 20.0,
            "sine_amp": 0.4,
            "sine_y_offset": 1.0,
            "sine_phase": 0.0,
            "sine_phase_offset_i": 1.57,
        },
        performance_mode=False,
        max_spikes=2000,
    )

    assert len(result["input_t_ms"]) > 0
    assert len(result["input_t_ms"]) == len(result["input_mean_E"])
    assert len(result["input_t_ms"]) == len(result["input_mean_I"])
    assert np.std(np.array(result["input_mean_E"])) > 0.0
    assert np.mean(np.array(result["input_mean_E"])) > 1.0


def test_run_simulation_accepts_external_spike_train_input_type() -> None:
    result = run_simulation(
        config_overrides={"T": 300.0, "dt": 1.0, "N_E": 32, "N_I": 8},
        inputs_overrides={
            "input_type": "external_spike_train",
            "input_population": "e",
            "I_E_base": 0.0,
            "I_I_base": 0.0,
            "noise_std_E": 0.0,
            "noise_std_I": 0.0,
            "seed": 7,
            "lambda0_hz": 25.0,
            "mod_depth": 0.6,
            "envelope_freq_hz": 5.0,
            "phase_rad": 0.2,
            "w_in": 0.3,
            "tau_in_ms": 3.0,
        },
        performance_mode=False,
        max_spikes=4000,
    )

    assert len(result["input_t_ms"]) > 0
    assert len(result["input_spike_fraction_E"]) == len(result["input_t_ms"])
    assert len(result["input_spike_fraction_layers"]) >= 1
    assert len(result["input_raw_raster_times_ms_layers"]) == len(result["layer_labels"])
    assert len(result["input_raw_raster_ids_layers"]) == len(result["layer_labels"])
    assert np.mean(np.array(result["input_spike_fraction_E"], dtype=float)) > 0.0
    assert len(result["decode_lowpass_hz_E"]) == len(result["population_rate_t_ms"])
    assert len(result["decode_lowpass_hz_layers"]) == len(result["population_rate_hz_layers"])
    assert len(result["input_envelope_hz"]) == len(result["input_t_ms"])
    assert len(result["input_envelope_hz_layers"]) == len(result["layer_labels"])
    assert len(result["decode_envelope_hz"]) == len(result["population_rate_t_ms"])
    assert len(result["decode_envelope_hz_layers"]) == len(result["layer_labels"])
    assert len(result["decode_corr_layers"]) == len(result["layer_labels"])
    assert len(result["decode_rmse_layers"]) == len(result["layer_labels"])
    assert -1.0 <= float(result["decode_corr"]) <= 1.0
    assert float(result["decode_rmse"]) >= 0.0


def test_run_simulation_defaults_to_e_only_input_population() -> None:
    base_config = {"T": 30.0, "dt": 1.0, "N_E": 10, "N_I": 8}
    common_inputs = {
        "input_type": "ramp",
        "I_E_start": 1.0,
        "I_E_end": 1.0,
        "I_I_start": 1.5,
        "I_I_end": 1.5,
        "noise_std_E": 0.0,
        "noise_std_I": 0.0,
        "seed": 1,
    }

    e_only = run_simulation(
        config_overrides=base_config,
        inputs_overrides=common_inputs,
        performance_mode=False,
        max_spikes=2000,
    )
    all_pops = run_simulation(
        config_overrides=base_config,
        inputs_overrides={**common_inputs, "input_population": "all"},
        performance_mode=False,
        max_spikes=2000,
    )

    assert np.max(np.abs(np.array(e_only["input_mean_I"]))) == 0.0
    assert np.mean(np.array(all_pops["input_mean_I"])) > 1.0


def test_run_simulation_clips_negative_input_to_zero() -> None:
    result = run_simulation(
        config_overrides={"T": 60.0, "dt": 1.0, "N_E": 12, "N_I": 6},
        inputs_overrides={
            "input_type": "sine",
            "input_population": "all",
            "I_E_base": 0.0,
            "I_I_base": 0.0,
            "noise_std_E": 0.0,
            "noise_std_I": 0.0,
            "seed": 5,
            "sine_freq_hz": 10.0,
            "sine_amp": 1.5,
            "sine_y_offset": -1.0,
            "sine_phase": 0.0,
            "sine_phase_offset_i": 0.0,
        },
        performance_mode=False,
        max_spikes=2000,
    )

    input_e = np.array(result["input_mean_E"], dtype=float)
    input_i = np.array(result["input_mean_I"], dtype=float)
    assert np.min(input_e) >= 0.0
    assert np.min(input_i) >= 0.0

def test_run_simulation_exposes_single_e_layer_label_without_templates() -> None:
    result = run_simulation(
        config_overrides={"T": 120.0, "dt": 1.0, "N_E": 30, "N_I": 10},
        inputs_overrides={
            "input_type": "sine",
            "input_population": "e",
            "I_E_base": 1.2,
            "I_I_base": 0.0,
            "noise_std_E": 0.0,
            "noise_std_I": 0.0,
            "seed": 2,
            "sine_freq_hz": 20.0,
            "sine_amp": 0.6,
            "sine_y_offset": 0.4,
        },
        performance_mode=False,
        max_spikes=5000,
        burn_in_ms=20.0,
    )

    assert result["layer_labels"] == ["L1", "I"]


def test_e_rate_invariant_when_only_ei_std_changes_and_ie_is_zero() -> None:
    config = {"T": 1000.0, "dt": 0.1, "N_E": 400, "N_I": 200, "seed": 0}
    inputs = {
        "input_type": "ramp",
        "input_population": "e",
        "I_E_start": 0.0,
        "I_E_end": 0.0,
        "I_I_start": 0.0,
        "I_I_end": 0.0,
        "noise_std_E": 2.0,
        "noise_std_I": 0.0,
        "seed": 0,
    }

    def _run(ei_std: float) -> float:
        out = run_simulation(
            config_overrides=config,
            inputs_overrides=inputs,
            weights_overrides={
                "ee": {"mean": 0.0, "std": 0.0},
                "ei": {"mean": 0.0, "std": ei_std},
                "ie": {"mean": 0.0, "std": 0.0},
                "ii": {"mean": 0.0, "std": 0.0},
                "clamp_min": 0.0,
                "seed": 0,
            },
            performance_mode=False,
            max_spikes=30000,
            burn_in_ms=200.0,
        )
        return float(out["mean_rate_E"])

    base = _run(0.0)
    for std in (0.05, 0.1, 0.2, 0.4):
        assert _run(std) == pytest.approx(base, abs=0.2)
