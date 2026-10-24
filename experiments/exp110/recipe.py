"""Committed manuscript-figure definitions; no execution on import.

configuration() resolves fresh scientific settings before execution. Network
authoring and analysis definitions do not load data or launch a stage on import.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

TAU_GABA_REFERENCE_MS = 6.0


CELLS = [
    {
        "name": "coba__off__seed42",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "off",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": None,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__off__seed43",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "off",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": None,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__off__seed44",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "off",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": None,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt25hz__seed42",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "25 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 25.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt25hz__seed43",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "25 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 25.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt25hz__seed44",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "25 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 25.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt10hz__seed42",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "10 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 10.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt10hz__seed43",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "10 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 10.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt10hz__seed44",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "10 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 10.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt5hz__seed42",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "5 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 5.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt5hz__seed43",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "5 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 5.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt5hz__seed44",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "5 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 5.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt2p5hz__seed42",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "2.5 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 2.5,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt2p5hz__seed43",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "2.5 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 2.5,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt2p5hz__seed44",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "2.5 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 2.5,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt1hz__seed42",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "1 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 1.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt1hz__seed43",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "1 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 1.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt1hz__seed44",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "1 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 1.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__off__seed42",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "off",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": None,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__off__seed43",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "off",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": None,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__off__seed44",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "off",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": None,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt25hz__seed42",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "25 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 25.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt25hz__seed43",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "25 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 25.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt25hz__seed44",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "25 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 25.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt10hz__seed42",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "10 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 10.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt10hz__seed43",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "10 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 10.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt10hz__seed44",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "10 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 10.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt5hz__seed42",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "5 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 5.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt5hz__seed43",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "5 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 5.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt5hz__seed44",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "5 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 5.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt2p5hz__seed42",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "2.5 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 2.5,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt2p5hz__seed43",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "2.5 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 2.5,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt2p5hz__seed44",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "2.5 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 2.5,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt1hz__seed42",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "1 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 1.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt1hz__seed43",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "1 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 1.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt1hz__seed44",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "1 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 1.0,
        "training_run_id": "TR-02",
    },
]


def training_run_cells(training_run_id):
    cells = [
        deepcopy(cell) for cell in CELLS if cell["training_run_id"] == training_run_id
    ]
    if not cells:
        raise ValueError(f"unsupported training family: {training_run_id}")
    return tuple(cells)


def training_run_values(training_run_id, field):
    values = []
    for cell in training_run_cells(training_run_id):
        value = cell[field]
        if value not in values:
            values.append(value)
    return tuple(values)


REFRACTORY_E_MS = 1.2

REFRACTORY_I_MS = 0.6

REFRACTORY_POLICY = "exact"

source_exp054_recipe_FIGURES = (
    "turnon_maps_compound.svg",
    "turnon_compound.svg",
    "grid_maps.svg",
    "grid_rasters.svg",
    "grid_autocorr.svg",
    "rate_invariance.svg",
    "null_autocorr.svg",
)


def refractory_configuration() -> dict:
    return {
        "refractory_e_ms": REFRACTORY_E_MS,
        "refractory_i_ms": REFRACTORY_I_MS,
        "refractory_policy": REFRACTORY_POLICY,
    }


def source_exp054_recipe_configuration(*, smoke=False, version=8):
    if version not in (7, 8):
        raise ValueError("unsupported exp054 recipe version")
    cfg = {
        "schema": f"exp054.recipe/v{version}",
        **refractory_configuration(),
        "profile": "smoke" if smoke else "production",
        "dt_ms": 0.1,
        "tau_gaba_ms": 6.0,
        "sim_ms": 400.0 if smoke else 1000.0,
        "burn_ms": 100.0,
        "n_e": 1024,
        "n_i": 256,
        "seed": 42,
        "input_rate_hz": 100.0,
        "private_w_in": 0.5,
        "shared_n_in": 200,
        "shared_w_in": 0.2,
        "shared_zero_fraction": 0.95,
        "max_lag_ms": 100.0,
        "bin_ms": 1.0,
        "wei_mean": [round(v, 3) for v in np.linspace(0, 3, 6 if smoke else 11)],
        "wie_mean": [round(v, 3) for v in np.linspace(0, 6, 6 if smoke else 11)],
        "private_null_hz": [1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 70.0, 100.0],
        "shared_null_hz": [8.0, 12.0, 16.0, 20.0, 28.0, 40.0, 60.0, 100.0],
        "display_window_ms": 200.0,
        "display_e": 160,
        "display_i": 48,
        "display_stride": 1 if smoke else 2,
    }
    if version == 8:
        cfg.update(
            executor="snnlab.sim.GraphExecutor",
            biophysics=dict(BIOPHYSICS),
            initialization_stream="exp054 ordered CPU draws/v1",
            encoder_seed=43,
            weight_initialization={
                "relative_sd": 0.1,
                "discarded_readout": {"width": 10, "mean": 5.1, "sd": 3.8},
                "discarded_ee": {"mean": 0.0, "sd": 0.0},
            },
            reset="fresh state per probe; resting voltage; zero conductances and refractory",
        )
    return cfg


def validate(cfg):
    if cfg not in tuple(
        source_exp054_recipe_configuration(smoke=smoke, version=version)
        for smoke in (False, True)
        for version in (7, 8)
    ):
        from pingstore.contracts import PingstoreError

        raise PingstoreError("inconsistent exp054 recipe")
    return cfg


def turnon_points(cfg):
    weak = 1 if cfg["profile"] == "smoke" else 2
    return [
        ("A", 0, 0),
        ("B", weak, weak),
        ("C", len(cfg["wei_mean"]) - 1, len(cfg["wie_mean"]) - 1),
    ]


BIOPHYSICS = {
    "capacitance_e_nf": 1.0,
    "capacitance_i_nf": 0.5,
    "leak_e_us": 0.05,
    "leak_i_us": 0.10,
    "resting_mv": -65.0,
    "threshold_mv": -50.0,
    "reset_mv": -65.0,
    "tau_ampa_ms": 2.0,
    "reversal_ampa_mv": 0.0,
    "reversal_gaba_mv": -80.0,
}


SCHEMA = "exp117.recipe/v3"

IMPORTED_PARAMETERS = {
    "population_sizes": {"N_E": 1024, "N_I": 256},
    "capacitance_nF": {"E": 1.0, "I": 0.5},
    "leak_conductance_uS": {"E": 0.05, "I": 0.10},
    "rest_mV": -65.0,
    "reset_mV": -65.0,
    "threshold_mV": -50.0,
    "reversal_mV": {"excitatory": 0.0, "inhibitory": -80.0},
    "refractory_ms": {"E": 1.2, "I": 0.6},
    "tau_AMPA_ms": 2.0,
    "tau_GABA_reference_ms": 6.0,
    "summed_coupling_uS": {"E_to_I": 1.0, "I_to_E": 2.0},
}

MEAN_FIELD_CHOICES = {
    "rate_relaxation_ms": {"E": 20.0, "I": 5.0},
    "effective_voltage_noise_mV": 4.0,
    "driving_force_reference": "resting_voltage",
    "external_drive_nA": {"minimum": 0.0, "maximum": 4.0},
    "tau_GABA_sweep_ms": (4.5, 6.0, 9.0, 12.0, 18.0, 27.0),
}

NUMERICAL_PROTOCOL = {
    "drive_grid_nA": (0.0, 4.0, 401),
    "equilibrium_root": {
        "method": "brentq",
        "xtol": 1e-13,
        "rtol": 1e-12,
        "residual_tolerance_per_ms": 1e-11,
    },
    "complex_eigenvalue_threshold_per_ms": 1e-8,
    "hopf_refinement": {
        "method": "brentq",
        "xtol_nA": 1e-12,
        "rtol": 1e-12,
        "transversality_step_nA": 1e-5,
    },
    "criticality_ramp": {
        "span_relative_to_hopf_nA": (-0.1, 0.55),
        "points": 25,
        "duration_ms": 2000.0,
        "measurement_start_ms": 1500.0,
        "measurement_samples": 1001,
        "initial_rate_perturbation_per_ms": 1e-3,
        "amplitude_threshold_per_ms": 1e-4,
        "maximum_branch_gap_per_ms": 1e-4,
        "minimum_amplitude_squared_r2": 0.9,
        "solver": "LSODA",
        "rtol": 1e-7,
        "atol": 1e-10,
        "max_step_ms": 1.0,
    },
}


def source_exp117_recipe_configuration() -> dict:
    """Return the complete independent model definition."""
    return {
        "schema": SCHEMA,
        "imported_parameters": {
            key: dict(value) if isinstance(value, dict) else value
            for key, value in IMPORTED_PARAMETERS.items()
        },
        "mean_field_choices": {
            key: dict(value)
            if isinstance(value, dict)
            else list(value)
            if isinstance(value, tuple)
            else value
            for key, value in MEAN_FIELD_CHOICES.items()
        },
        "numerical_protocol": {
            "drive_grid_nA": list(NUMERICAL_PROTOCOL["drive_grid_nA"]),
            "equilibrium_root": dict(NUMERICAL_PROTOCOL["equilibrium_root"]),
            "complex_eigenvalue_threshold_per_ms": NUMERICAL_PROTOCOL[
                "complex_eigenvalue_threshold_per_ms"
            ],
            "hopf_refinement": dict(NUMERICAL_PROTOCOL["hopf_refinement"]),
            "criticality_ramp": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in NUMERICAL_PROTOCOL["criticality_ramp"].items()
            },
        },
        "runtime_dependencies": [],
        "planned_outputs": {
            "hopf_location": True,
            "hopf_frequency_Hz": True,
            "criticality": ["supercritical", "subcritical", "unresolved"],
            "frequency_vs_tau_GABA": True,
        },
    }


def source_exp117_recipe_validate(candidate: dict) -> dict:
    """Require the exact committed scientific configuration."""
    if candidate != source_exp117_recipe_configuration():
        raise ValueError("inconsistent exp117 recipe")
    return candidate


SLUG = "exp110"
RATE_FREQUENCY_SOURCE = "rate_vs_fgamma.svg"
CYCLE_COUNT_SOURCE = "spikes_per_cycle_distribution_equal_network.svg"
PERTURBATION_SOURCE = "numbers.json"
TIMESTEP_SOURCE = "numbers.json"
FIGURES = (
    "onset_super_compound.svg",
    "onset_super_compound.pdf",
    "cycle_participation_compound.svg",
    "cycle_participation_compound.pdf",
    "robustness_compound.svg",
    "robustness_compound.pdf",
)


def configuration(exp054_recipe: dict, exp117_recipe: dict) -> dict:
    return deepcopy(
        dict(
            schema="exp110.presentation/v15",
            cycle_count_weighting="equal_network",
            figures=(
                "gamma-onset",
                "rate-frequency-and-cycle-participation",
                "spike-perturbation-and-timestep-robustness",
            ),
            source_stage="analyse",
            source_recipes=dict(
                exp054=exp054_recipe,
                exp117=exp117_recipe,
            ),
        )
    )


MODELS = list(training_run_values("TR-02", "model"))
