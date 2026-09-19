"""Self-contained scientific definition for exp117; no execution on import."""

from __future__ import annotations

SLUG = "exp117"
SCHEMA = "exp117.recipe/v3"

# Literal copies of established conductance-based system parameters.  Exp117
# deliberately does not import another experiment's recipe at runtime.
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

# Choices that define the mean-field closure rather than the underlying
# conductance-based neuron system.
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


def configuration() -> dict:
    """Return the complete independent model definition."""
    return {
        "schema": SCHEMA,
        "imported_parameters": {
            key: dict(value) if isinstance(value, dict) else value
            for key, value in IMPORTED_PARAMETERS.items()
        },
        "mean_field_choices": {
            key: dict(value) if isinstance(value, dict) else list(value)
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


def validate(candidate: dict) -> dict:
    """Require the exact committed scientific configuration."""
    if candidate != configuration():
        raise ValueError("inconsistent exp117 recipe")
    return candidate
