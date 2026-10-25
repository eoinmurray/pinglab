"""Self-contained scientific definition for exp117; no execution on import.

configuration() resolves fresh scientific settings before execution. Network
authoring and analysis definitions do not load data or launch a stage on import.
"""

from __future__ import annotations

from copy import deepcopy

from snnlab import lang
from snnlab import meanfield as mf
from snnlab.meanfield import closures

SLUG = "exp117"
SCHEMA = "exp117.recipe/v4"

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
    "authoring_dt_ms": 0.1,
    "gain_quadrature_nodes": 48,
    "equilibrium_guess": [2.0, 6.0, 0.01, 0.02],
    "equilibrium_root": {
        "method": "snnlab.meanfield.find_equilibrium",
        "scaled_derivative_tolerance": 1e-9,
    },
    "complex_eigenvalue_threshold_per_ms": 1e-8,
    "hopf_refinement": {
        "method": "snnlab.meanfield.find_stability_crossings",
        "xtol_nA": 1e-12,
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
    """Return the complete scientific and numerical definition."""
    return deepcopy(
        dict(
            schema=SCHEMA,
            imported_parameters={
                key: dict(value) if isinstance(value, dict) else value
                for key, value in IMPORTED_PARAMETERS.items()
            },
            mean_field_choices={
                key: dict(value)
                if isinstance(value, dict)
                else list(value)
                if isinstance(value, tuple)
                else value
                for key, value in MEAN_FIELD_CHOICES.items()
            },
            numerical_protocol=dict(
                authoring_dt_ms=NUMERICAL_PROTOCOL["authoring_dt_ms"],
                gain_quadrature_nodes=NUMERICAL_PROTOCOL["gain_quadrature_nodes"],
                equilibrium_guess=list(NUMERICAL_PROTOCOL["equilibrium_guess"]),
                drive_grid_nA=list(NUMERICAL_PROTOCOL["drive_grid_nA"]),
                equilibrium_root=dict(NUMERICAL_PROTOCOL["equilibrium_root"]),
                complex_eigenvalue_threshold_per_ms=NUMERICAL_PROTOCOL[
                    "complex_eigenvalue_threshold_per_ms"
                ],
                hopf_refinement=dict(NUMERICAL_PROTOCOL["hopf_refinement"]),
                criticality_ramp={
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in NUMERICAL_PROTOCOL["criticality_ramp"].items()
                },
            ),
            runtime_dependencies=["snnlab>=0.27.0"],
            planned_outputs=dict(
                hopf_location=True,
                hopf_frequency_Hz=True,
                criticality=["supercritical", "subcritical", "unresolved"],
                frequency_vs_tau_GABA=True,
            ),
        )
    )


def validate(candidate: dict, *, historical=False) -> dict:
    """Require the current compute recipe; retain recorded recipes for readers.

    Authenticated v3 evidence remains analysable without replacing its recorded
    solver settings with the current implementation's defaults.
    """
    if historical and candidate.get("schema") in {"exp117.recipe/v3", SCHEMA}:
        if not all(
            isinstance(candidate.get(key), dict)
            for key in (
                "imported_parameters",
                "mean_field_choices",
                "numerical_protocol",
            )
        ):
            raise ValueError("incomplete recorded exp117 recipe")
        return candidate
    if candidate != configuration():
        raise ValueError("inconsistent exp117 recipe")
    return candidate


def author_network(cfg):
    """Compile the reference cross-coupled architecture without simulating it."""
    p = cfg["imported_parameters"]
    net = lang.Network(
        "exp117", dt=cfg["numerical_protocol"]["authoring_dt_ms"] * lang.ms
    )
    populations = {}
    for name in ("E", "I"):
        populations[name] = net.population(
            name,
            size=p["population_sizes"]["N_" + name],
            neuron=lang.COBA_LIF(
                tau_mem=p["capacitance_nF"][name]
                / p["leak_conductance_uS"][name]
                * lang.ms,
                capacitance_nf=p["capacitance_nF"][name],
                leak_us=p["leak_conductance_uS"][name],
                resting_mv=p["rest_mV"],
                reset_mv=p["reset_mV"],
                threshold_mv=p["threshold_mV"],
                excitatory_reversal_mv=p["reversal_mV"]["excitatory"],
                inhibitory_reversal_mv=p["reversal_mV"]["inhibitory"],
                refractory=p["refractory_ms"][name] * lang.ms,
            ),
        )
    for path, coupling in p["summed_coupling_uS"].items():
        source, target = path.split("_to_")
        inhibitory = source == "I"
        net.connect(
            populations[source].spikes,
            populations[target].inhibitory
            if inhibitory
            else populations[target].excitatory,
            name=path,
            synapse=lang.GABA(tau=p["tau_GABA_reference_ms"] * lang.ms)
            if inhibitory
            else lang.AMPA(tau=p["tau_AMPA_ms"] * lang.ms),
            weight=lang.Constant(coupling / p["population_sizes"]["N_" + source]),
            initialization_scaling="direct",
            connection="recurrent",
        )
    return lang.compile(net)


def mean_field_model(cfg, *, bundle=None):
    """Reduce the authored circuit to native Hz/uS rate–conductance equations."""
    choices = cfg["mean_field_choices"]
    return mf.reduce(
        author_network(cfg) if bundle is None else bundle,
        closure=closures.RateConductance(
            gain=closures.NoisyLIF(
                sigma=choices["effective_voltage_noise_mV"],
                quadrature_nodes=cfg["numerical_protocol"]["gain_quadrature_nodes"],
            ),
            tau_rate=choices["rate_relaxation_ms"],
        ),
        weights=mf.MeanIncomingWeights(
            cfg["imported_parameters"]["summed_coupling_uS"],
            provenance="exp117 imported reference cross couplings; no matrix sampling",
        ),
        applied_current={"E": 0 * lang.nA},
    )
