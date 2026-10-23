"""PING fundamentals: one network setup across raster, spectrum and drive response.

configuration() resolves fresh scientific settings before execution. Network
authoring and analysis definitions do not load data or launch a stage on import.
"""

from copy import deepcopy

SLUG = "exp023"
DT_MS = 0.1
REFRACTORY_E_MS = 1.2
REFRACTORY_I_MS = 0.6
REFRACTORY_POLICY = "exact"
N_E, N_I, N_IN = 1024, 256, 1024
FI_N_IN = N_IN
SEED = 42
COBA_INPUT_RATE_HZ, PING_INPUT_RATE_HZ = 5, 45
CELLS = ("coba", "ping")
FI_EI = {"coba": 0.0, "ping": 1.5}
FI_RATES_HZ = [2, 5, 10, 20, 40, 70, 100]
F_GAMMA_BAND_HZ = (5.0, 150.0)
BIOPHYSICS = {
    "E_L_mV": -65.0,
    "E_E_mV": 0.0,
    "E_I_mV": -80.0,
    "g_L_E_uS": 0.05,
    "g_L_I_uS": 0.10,
    "threshold_mV": -50.0,
    "reset_mV": -65.0,
    "tau_ampa_ms": 2.0,
    "tau_gaba_ms": 6.0,
    "C_m_E_nF": 1.0,
    "C_m_I_nF": 0.5,
    "refractory_E_ms": REFRACTORY_E_MS,
    "refractory_I_ms": REFRACTORY_I_MS,
}


def refractory_configuration() -> dict:
    return {
        "refractory_e_ms": REFRACTORY_E_MS,
        "refractory_i_ms": REFRACTORY_I_MS,
        "refractory_policy": REFRACTORY_POLICY,
    }


def configuration(*, smoke: bool = False, version: int = 5) -> dict:
    if type(version) is not int or version not in (4, 5):
        raise ValueError("unsupported exp023 recipe version")
    if type(smoke) is not bool:
        raise ValueError("smoke must be boolean")
    cfg = dict(
        schema=f"exp023.recipe/v{version}",
        **refractory_configuration(),
        profile="smoke" if smoke else "production",
        model="ping",
        cells=list(CELLS),
        n_e=N_E,
        n_i=N_I,
        seed=SEED,
        trials_per_condition=1,
        initial_voltage_mV=-65.0,
        initial_conductance_uS=0.0,
        input_weight_parent_mean=1.5,
        input_weight_parent_sd=0.3,
        input_initial_zero_fraction=0.95,
        recurrent_initial_zero_fraction=0.0,
        ei_ratio=2.0,
        integration="exponential_euler",
        drive=graph_drive(smoke=smoke),
        executor="snnlab.sim.GraphExecutor",
        initialization_stream="native graph parameter order; distinct from the historical CLI stream",
        biophysics=dict(BIOPHYSICS),
    )

    if version >= 5:
        from dataclasses import asdict
        from snnlab.sim import Performance

        cfg["performance"] = asdict(Performance())
    return deepcopy(cfg)


def operating_point(cell: str, rate_hz: int, n_in: int, *, smoke=False) -> dict:
    if cell not in CELLS:
        raise ValueError(f"unknown loop condition: {cell}")
    return {
        "input": "synthetic-spikes",
        "input_rate_hz": float(rate_hz),
        "ei_strength": float(FI_EI[cell]),
        "t_ms": 200 if smoke else 400,
        "dt_ms": DT_MS,
        "n_in": n_in,
        "seed": SEED,
    }


def graph_drive(*, smoke=False) -> dict:
    return {
        "raster_operating_points": {
            cell: operating_point(
                cell,
                COBA_INPUT_RATE_HZ if cell == "coba" else PING_INPUT_RATE_HZ,
                N_IN,
                smoke=smoke,
            )
            for cell in CELLS
        },
        "fi_sweep": {
            "input": "synthetic-spikes",
            "input_rates_hz": list(FI_RATES_HZ),
            "ei_strength_by_cell": {cell: float(FI_EI[cell]) for cell in CELLS},
            "t_ms": 200 if smoke else 400,
            "dt_ms": DT_MS,
            "n_in": FI_N_IN,
            "seed": SEED,
        },
    }


def trials(*, smoke=False):
    for cell, point in graph_drive(smoke=smoke)["raster_operating_points"].items():
        yield f"scope/{cell}", cell, point, True
    for cell in CELLS:
        for rate in FI_RATES_HZ:
            yield (
                f"fi/{cell}__r{rate}",
                cell,
                operating_point(cell, rate, FI_N_IN, smoke=smoke),
                False,
            )


def author_network(cfg: dict, point: dict, *, traces: bool):
    """Declare the untrained circuit, with no stimulus or simulator globals."""
    from snnlab import lang

    b = cfg["biophysics"]
    net = lang.Network("exp023_ping_fundamentals", dt=point["dt_ms"] * lang.ms)
    drive = net.input(
        "drive",
        shape=("time", "batch", point["n_in"]),
        signal_type="spikes",
        unit="spike",
    )
    neurons = {}
    for label in ("E", "I"):
        neurons[label] = lang.COBA_LIF(
            tau_mem=(b[f"C_m_{label}_nF"] / b[f"g_L_{label}_uS"]) * lang.ms,
            capacitance_nf=b[f"C_m_{label}_nF"],
            leak_us=b[f"g_L_{label}_uS"],
            resting_mv=b["E_L_mV"],
            threshold_mv=b["threshold_mV"],
            reset_mv=b["reset_mV"],
            initial_voltage_mv=cfg["initial_voltage_mV"],
            refractory=b[f"refractory_{label}_ms"] * lang.ms,
            voltage_grad_dampen=80.0,
        )
    strength = point["ei_strength"]
    ie_strength = strength * cfg["ei_ratio"]
    # Reciprocal E/I loop with excitatory drive to E only.
    ping_ampa = lang.AMPA(tau=b["tau_ampa_ms"] * lang.ms)
    ping_gaba = lang.GABA(tau=b["tau_gaba_ms"] * lang.ms)
    ping_recurrent_delay = point["dt_ms"] * lang.ms
    ping_e = net.population("E", size=cfg["n_e"], neuron=neurons["E"])
    ping_i = net.population("I", size=cfg["n_i"], neuron=neurons["I"])
    ping_input_to_e = net.connect(
        drive,
        ping_e.excitatory,
        name="input_to_E",
        synapse=ping_ampa,
        weight=lang.LowerClampedNormal(
            cfg["input_weight_parent_mean"],
            cfg["input_weight_parent_sd"],
            initial_zero_fraction=cfg["input_initial_zero_fraction"],
            zeroing="bernoulli",
        ),
        constraint=lang.NonNegative(),
        initialization_scaling="fan_in_normalized",
    )
    ping_e_to_i = net.connect(
        ping_e.spikes,
        ping_i.excitatory,
        name="E_to_I",
        synapse=ping_ampa,
        weight=lang.LowerClampedNormal(strength, strength * 0.1),
        constraint=lang.NonNegative(),
        connection="recurrent",
        delay=ping_recurrent_delay,
        initialization_scaling="fan_in_normalized",
    )
    ping_i_to_e = net.connect(
        ping_i.spikes,
        ping_e.inhibitory,
        name="I_to_E",
        synapse=ping_gaba,
        weight=lang.LowerClampedNormal(ie_strength, ie_strength * 0.1),
        constraint=lang.NonNegative(),
        connection="recurrent",
        delay=ping_recurrent_delay,
        initialization_scaling="fan_in_normalized",
    )
    e, i = ping_e, ping_i
    if traces:
        for name, signal in {
            "spk_e": e.spikes,
            "spk_i": i.spikes,
            "v_e": e.voltage,
            "v_i": i.voltage,
            "ge_e": ping_input_to_e.conductance,
            "ge_i": ping_e_to_i.conductance,
            "gi_e": ping_i_to_e.conductance,
        }.items():
            net.expose(signal, name=name)
    for label, population in (("E", e), ("I", i)):
        count = lang.reduce(
            population.spikes, operation="sum", over="time", name=f"{label}_counts"
        )
        net.output(f"spk_{label.lower()}_count", count)
    return lang.compile(net, target="tools/snnsim", refractory_timing="legacy_coba")
