"""The retained 18-network cycle-counting recipe; no storage side effects.

configuration() resolves fresh scientific settings before execution. Network
authoring and analysis definitions do not load data or launch a stage on import.
"""

from copy import deepcopy

from experiments.exp046.custom.source_exp041_recipe import ANALYSIS_PURPOSE
from experiments.exp046.custom.source_exp041_recipe import CHECKPOINT_POLICY
from experiments.exp046.custom.source_exp041_recipe import CHECKPOINT_ROLE
from experiments.exp046.custom.source_exp041_recipe import EVAL_MAX_SAMPLES
from experiments.exp046.custom.source_exp041_recipe import SEEDS
from experiments.exp046.custom.source_exp041_recipe import SMOKE_MAX_SAMPLES
from experiments.exp046.custom.source_exp041_recipe import TAU_GABA_SWEEP
from experiments.exp046.custom.source_exp041_recipe import cell_name

SLUG = "exp046"
REFRACTORY_E_MS = 1.2
REFRACTORY_I_MS = 0.6
REFRACTORY_POLICY = "exact"
__all__ = ["ANALYSIS_PURPOSE", "CHECKPOINT_ROLE", "cell_name"]
TAU_GABA_SWEEP_MS = TAU_GABA_SWEEP
FIGURES = tuple(
    name + "." + ext
    for name in ("spikes_per_cycle_distribution", "ceiling_vs_fgamma")
    for ext in ("svg", "pdf")
)
FIGURES += tuple(
    "spikes_per_cycle_distribution_equal_network." + ext for ext in ("svg", "pdf")
)


def refractory_configuration() -> dict:
    return {
        "refractory_e_ms": REFRACTORY_E_MS,
        "refractory_i_ms": REFRACTORY_I_MS,
        "refractory_policy": REFRACTORY_POLICY,
    }


def configuration(*, smoke=False, version=2):
    if type(smoke) is not bool:
        raise ValueError("smoke must be boolean")
    if version not in (1, 2):
        raise ValueError("unsupported exp046 recipe version")
    return deepcopy(
        dict(
            schema=f"exp046.recipe/v{version}",
            **(refractory_configuration() if version >= 2 else dict()),
            profile="smoke" if smoke else "production",
            tau_gaba_sweep_ms=list(TAU_GABA_SWEEP_MS),
            seeds=list(SEEDS),
            evaluation_samples=SMOKE_MAX_SAMPLES if smoke else EVAL_MAX_SAMPLES,
            checkpoint_policy=CHECKPOINT_POLICY,
        )
    )


BIOPHYSICS = {
    "capacitance_e_nf": 1.0,
    "capacitance_i_nf": 0.5,
    "leak_e_us": 0.05,
    "leak_i_us": 0.10,
    "resting_mv": -65.0,
    "threshold_mv": -50.0,
    "reset_mv": -65.0,
    "readout_tau_ms": 2.0,
    "readout_threshold": 1.0,
}


def author_network(training, *, observables=()):
    """Compile this experiment's E/I classifier and requested observables."""
    from types import SimpleNamespace

    from snnlab import lang

    biophysics = BIOPHYSICS
    refractory = refractory_configuration()
    dt = training["dt"]
    net = lang.Network(SLUG, dt=dt * lang.ms)
    drive = net.input(
        "drive",
        shape=("time", "batch", training["n_in"]),
        signal_type="spikes",
        unit="spike",
    )
    neurons = {}
    for label in ("e", "i"):
        capacitance, leak = (
            biophysics[f"capacitance_{label}_nf"],
            biophysics[f"leak_{label}_us"],
        )
        neurons[label] = lang.COBA_LIF(
            tau_mem=capacitance / leak * lang.ms,
            capacitance_nf=capacitance,
            leak_us=leak,
            resting_mv=biophysics["resting_mv"],
            threshold_mv=biophysics["threshold_mv"],
            reset_mv=biophysics["reset_mv"],
            initial_voltage_mv=biophysics["resting_mv"],
            refractory=refractory[f"refractory_{label}_ms"] * lang.ms,
            voltage_grad_dampen=training["v_grad_dampen"],
        )
    # Reciprocal E/I loop with excitatory drive to E only.
    ping_ampa = lang.AMPA(tau=training["tau_ampa_ms"] * lang.ms)
    ping_gaba = lang.GABA(tau=training["tau_gaba_ms"] * lang.ms)
    ping_recurrent_delay = dt * lang.ms
    ping_e = net.population("E", size=training["n_hidden"], neuron=neurons["e"])
    ping_i = net.population("I", size=training["n_inh"], neuron=neurons["i"])
    ping_input_to_e = net.connect(
        drive,
        ping_e.excitatory,
        name="input_to_E",
        synapse=ping_ampa,
        weight=lang.Constant(0),
        constraint=lang.NonNegative(),
        initialization_scaling="direct",
    )
    ping_e_to_i = net.connect(
        ping_e.spikes,
        ping_i.excitatory,
        name="E_to_I",
        synapse=ping_ampa,
        weight=lang.Constant(0),
        constraint=lang.NonNegative(),
        connection="recurrent",
        delay=ping_recurrent_delay,
        initialization_scaling="direct",
    )
    ping_i_to_e = net.connect(
        ping_i.spikes,
        ping_e.inhibitory,
        name="I_to_E",
        synapse=ping_gaba,
        weight=lang.Constant(0),
        constraint=lang.NonNegative(),
        connection="recurrent",
        delay=ping_recurrent_delay,
        initialization_scaling="direct",
    )
    circuit = SimpleNamespace(
        e=ping_e,
        i=ping_i,
        input_to_e=ping_input_to_e,
        e_to_i=ping_e_to_i,
        i_to_e=ping_i_to_e,
    )
    output = net.population(
        "readout",
        size=training["n_out"],
        spiking=True,
        neuron=lang.LeakyIntegrator(
            tau=biophysics["readout_tau_ms"] * lang.ms,
            soft_reset_threshold=biophysics["readout_threshold"],
            initial_voltage=0,
            surrogate_slope=training["surrogate_slope"],
        ),
    )
    net.connect(
        circuit.e.spikes,
        output.excitatory,
        name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=biophysics["readout_tau_ms"] * lang.ms),
        weight=lang.Constant(0),
        constraint=lang.NonNegative(),
        initialization_scaling="direct",
    )
    net.output(
        "class_scores",
        lang.reduce(
            output.pre_reset_voltage
            if training.get("readout_mode", "mem-mean") == "mem-mean"
            else output.spikes,
            operation="mean"
            if training.get("readout_mode", "mem-mean") == "mem-mean"
            else "sum",
            over="time",
            name="readout_reduction",
        ),
    )
    for label, pop in (("e", circuit.e), ("i", circuit.i)):
        net.output(
            f"spk_{label}_count",
            lang.reduce(
                pop.spikes, operation="sum", over="time", name=f"{label}_counts"
            ),
        )
        if "spikes" in observables:
            net.expose(pop.spikes, name=f"spk_{label}")
    if "traces" in observables:
        for key, value in {
            "v_e": circuit.e.voltage,
            "v_i": circuit.i.voltage,
            "ge_e": circuit.input_to_e.conductance,
            "ge_i": circuit.e_to_i.conductance,
            "gi_e": circuit.i_to_e.conductance,
        }.items():
            net.expose(value, name=key)
    return lang.compile(
        net,
        target="tools/snnsim",
        refractory_timing="legacy_coba",
    )


def inference_request(training, cfg):
    return {
        "checkpoint_role": CHECKPOINT_ROLE,
        "input": "dataset",
        "t_ms": training["t_ms"],
        "input_rate_hz": training["input_rate"],
        "samples": cfg["evaluation_samples"],
        "batch_size": 64,
        "subset_seed": 42,
        "encoder_seed": 20260415,
        "observables": [],
        "products": ["rasters", "rates"],
    }
