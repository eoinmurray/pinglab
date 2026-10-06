"""PING fundamentals: preserved scientific settings, without execution on import."""

SLUG = "exp023"
DT_MS = 0.1
REFRACTORY_E_MS = 1.2
REFRACTORY_I_MS = 0.6
REFRACTORY_POLICY = "exact"
N_E, N_I, N_IN = 1024, 256, 1024
# The old f–I command omitted --n-in and used the simulator's 784-channel default.
# Preserve this distinction; unifying the protocols is a separate scientific change.
FI_N_IN = 784
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


def configuration(*, smoke: bool = False) -> dict:
    return {
        "schema": "exp023.recipe/v3",
        **refractory_configuration(),
        "profile": "smoke" if smoke else "production",
        "model": "ping",
        "cells": list(CELLS),
        "n_e": N_E,
        "n_i": N_I,
        "seed": SEED,
        "trials_per_condition": 1,
        "initial_voltage_mV": -65.0,
        "initial_conductance_uS": 0.0,
        "input_weight_parent_mean": 1.5,
        "input_weight_parent_sd": 0.3,
        "input_initial_zero_fraction": 0.95,
        "recurrent_initial_zero_fraction": 0.0,
        "ei_ratio": 2.0,
        "integration": "exponential_euler",
        "drive": graph_drive(smoke=smoke),
        "executor": "snnlab.sim.GraphExecutor",
        "initialization_stream": "native graph parameter order; distinct from the historical CLI stream",
        "biophysics": dict(BIOPHYSICS),
    }


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
    from experiments.helpers.ping import build_ping
    from snnlab import lang
    from snnlab.sim.timing import refractory_steps

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
            refractory_steps=refractory_steps(
                b[f"refractory_{label}_ms"],
                point["dt_ms"],
                policy=cfg["refractory_policy"],
            ),
            voltage_grad_dampen=80.0,
        )
    strength = point["ei_strength"]
    ie_strength = strength * cfg["ei_ratio"]
    circuit = build_ping(
        net,
        source=drive,
        n_e=cfg["n_e"],
        n_i=cfg["n_i"],
        neuron_e=neurons["E"],
        neuron_i=neurons["I"],
        ampa=lang.AMPA(tau=b["tau_ampa_ms"] * lang.ms),
        gaba=lang.GABA(tau=b["tau_gaba_ms"] * lang.ms),
        input_weight=lang.LowerClampedNormal(
            cfg["input_weight_parent_mean"],
            cfg["input_weight_parent_sd"],
            initial_zero_fraction=cfg["input_initial_zero_fraction"],
            zeroing="bernoulli",
        ),
        ei_weight=lang.LowerClampedNormal(strength, strength * 0.1),
        ie_weight=lang.LowerClampedNormal(ie_strength, ie_strength * 0.1),
        recurrent_delay=point["dt_ms"] * lang.ms,
        initialization_scaling="fan_in_normalized",
    )
    e, i = circuit.e, circuit.i
    if traces:
        for name, signal in {
            "spk_e": e.spikes,
            "spk_i": i.spikes,
            "v_e": e.voltage,
            "v_i": i.voltage,
            "ge_e": circuit.input_to_e.conductance,
            "ge_i": circuit.e_to_i.conductance,
            "gi_e": circuit.i_to_e.conductance,
        }.items():
            net.expose(signal, name=name)
    else:
        for label, population in (("E", e), ("I", i)):
            count = lang.ops.reduce(
                population.spikes, operation="sum", over="time", name=f"{label}_counts"
            )
            net.output(f"spk_{label.lower()}_count", count)
    return lang.compile(net, target="tools/snnsim")


def execution_request(bundle, point: dict, *, traces: bool):
    from snnlab.sim.execution import ExecutionSpec, PoissonInputBinding
    from snnlab.sim.timing import duration_steps

    return ExecutionSpec(
        kind="simulate",
        executor="graph",
        graph=bundle.graph,
        seed=point["seed"],
        device="cpu",
        diagnostics=traces,
        input_bindings=(
            PoissonInputBinding(
                input_id="drive",
                steps_count=duration_steps(point["t_ms"], point["dt_ms"]),
                batch_size=1,
                rates_hz=(point["input_rate_hz"],),
                seed=point["seed"],
            ),
        ),
    )
