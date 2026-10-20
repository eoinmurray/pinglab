"""Preserved paired pool-size controls, with no execution on import."""

from snnlab.lang import duration_metadata, refractory_metadata

SLUG = "exp047"
REFRACTORY_E_MS = 1.2
REFRACTORY_I_MS = 0.6
REFRACTORY_POLICY = "exact"
FIGURES = ("pool_size_controls.svg", "pool_size_controls.pdf")
DEFINITION = "j_ie_synapse = g_ie_total / n_i"
MEASUREMENT = {
    "schema": "exp047.measurement/v1",
    "aggregation": "seed mean and sample SD",
    "sd_ddof": 1,
}


def refractory_configuration() -> dict:
    return {
        "refractory_e_ms": REFRACTORY_E_MS,
        "refractory_i_ms": REFRACTORY_I_MS,
        "refractory_policy": REFRACTORY_POLICY,
    }




def duration_configuration(duration_ms: float, dt_ms: float) -> dict:
    return duration_metadata(duration_ms, dt_ms)


def refractory_execution_configuration(dt_ms: float) -> dict:
    return refractory_metadata(
        REFRACTORY_E_MS,
        REFRACTORY_I_MS,
        dt_ms,
    )


def configuration(*, smoke=False, version=2):
    if version not in (1, 2):
        raise ValueError("unsupported exp047 recipe version")
    return {
        "schema": f"exp047.recipe/v{version}",
        **(refractory_configuration() if version >= 2 else {}),
        "profile": "smoke" if smoke else "production",
        "n_e": 1024,
        "n_in": 784,
        "n_i_sweep": [16, 64, 256],
        "n_i_reference": 256,
        "reference_g_ie": [1.0, 2.0, 4.0],
        "reference_j_ie": [1.0 / 256, 2.0 / 256, 4.0 / 256],
        "g_ei_total": 1.0,
        "input_rate_hz": 25.0,
        "t_ms": 200.0 if smoke else 500.0,
        "dt_ms": 0.1,
        "n_batch": 2 if smoke else 8,
        "seeds": [40, 41] if smoke else [40, 41, 42],
        "w_in_mean": 1.2,
        "w_in_initial_zero_fraction": 0.95,
    }


def report_config(cfg):
    """Keep the historical article's configuration interface unchanged."""
    return {
        k: v
        for k, v in cfg.items()
        if k not in {"schema", "profile", "w_in_mean", "w_in_initial_zero_fraction"}
    }


def conditions(cfg):
    for control, key in (
        ("fixed_total", "reference_g_ie"),
        ("fixed_synapse", "reference_j_ie"),
    ):
        for level in cfg[key]:
            for n_i in cfg["n_i_sweep"]:
                g_ie = level if control == "fixed_total" else n_i * level
                yield control, f"{level:.12g}", n_i, g_ie


def job(n_i, g_ie, seed):
    return {
        "id": f"nI{n_i}_g{g_ie:.8g}_s{seed}",
        "n_i": n_i,
        "g_ie_total": g_ie,
        "seed": seed,
    }


def jobs(cfg):
    seen, result = set(), []
    for _, _, n_i, g_ie in conditions(cfg):
        for seed in cfg["seeds"]:
            key = (n_i, round(g_ie, 12), seed)
            if key not in seen:
                seen.add(key)
                result.append(job(n_i, g_ie, seed))
    return result




BIOPHYSICS = {"capacitance_e_nf":1.0,"capacitance_i_nf":0.5,
    "leak_e_us":0.05,"leak_i_us":0.10,"resting_mv":-65.0,"threshold_mv":-50.0,
    "reset_mv":-65.0,"readout_tau_ms":2.0,"readout_threshold":1.0}


def network_settings(cfg, item):
    return {
        "n_in": cfg["n_in"],
        "n_hidden": cfg["n_e"],
        "n_inh": item["n_i"],
        "n_out": 10,
        "dt": cfg["dt_ms"],
        "tau_ampa_ms": 2.0,
        "tau_gaba_ms": 9.0,
        "seed": item["seed"],
        "v_grad_dampen": 80.0,
        "surrogate_slope": 1.0,
        "ei_strength": cfg["g_ei_total"],
        "ei_ratio": item["g_ie_total"] / cfg["g_ei_total"],
        "w_in": [cfg["w_in_mean"], cfg["w_in_mean"] * 0.1],
        "w_in_initial_zero_fraction": cfg["w_in_initial_zero_fraction"],
        "readout_w_out_scale": 1.0,
    }


def author_network(cfg, item):
    """Compile the pool-size control with this condition's network settings."""
    from types import SimpleNamespace

    from snnlab import lang

    training = network_settings(cfg, item)
    observables = ()
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
