"""Author one Toolbox half-centre oscillator with explicit graded graph edges.

Two identical ionic neurons receive the proposal's asymmetric current onset.
Each voltage drives a separate sigmoid-gated inhibitory synapse. Configuration
and bundle authoring are side-effect free apart from extension registration.
"""

from dataclasses import asdict

import numpy as np
import torch
from experiments.exp147.custom import dynamics
from snnlab import extensions, lang
from snnlab.sim import Performance


def configuration(*, duration_ms=200.0):
    if not np.isfinite(duration_ms) or not 0 < duration_ms <= 200:
        raise ValueError("duration_ms must be positive and at most 200 ms")
    cfg = dict(
        experiment="exp147",
        schema="exp147.recipe/v1",
        seed=0,
        device="cpu",
        dtype="float64",
        torch_threads=1,
        dt_ms=0.001,
        duration_ms=float(duration_ms),
        integration="gate-first forward Euler; beginning-of-step voltage sources",
        neuron=dict(
            initial_voltage_mv=-70.0,
            capacitance_nf=0.01,
            leak_us=0.01,
            leak_reversal_mv=-70.0,
            g_na_us=1.0,
            g_k_us=10.0,
            e_na_mv=50.0,
            e_k_mv=-85.0,
            tau_na_ms=1.0,
            tau_k_ms=10.0,
            gamma_na_per_mv=1.0,
            gamma_k_per_mv=1.0,
            bias_na_mv=-60.0,
            bias_k_mv=-60.0,
            spike_threshold_mv=-20.0,
            synapse_reversal_mv=-85.0,
        ),
        synapse=dict(
            initial_gate_mv=-70.0,
            g_max_us=1.0,
            tau_ms=10.0,
            gamma_per_mv=1.0,
            bias_mv=-55.0,
        ),
        currents=dict(
            n1=dict(
                kick_start_ms=5.0,
                kick_stop_ms=10.0,
                kick_na=1.0,
                drive_start_ms=10.0,
                drive_na=0.8,
            ),
            n2=dict(
                kick_start_ms=0.0,
                kick_stop_ms=0.0,
                kick_na=0.0,
                drive_start_ms=10.0,
                drive_na=0.8,
            ),
        ),
        projection_weight=1.0,
        delay_ms=0.0,
        batch_size=1,
        performance=asdict(Performance(compile_mode=None, cuda_graphs="off")),
        reference_status="Toolbox reference traces not supplied",
        unit_convention="mV, ms, nF, uS, nA; C/g/I units assumed",
        validation=dict(voltage_atol_mv=0.1, spike_atol_ms=0.05),
    )
    steps = cfg["duration_ms"] / cfg["dt_ms"]
    if not np.isclose(steps, round(steps), rtol=0, atol=1e-8):
        raise ValueError("duration must contain an integer number of timesteps")
    return cfg


def validate_configuration(cfg):
    if cfg != configuration(duration_ms=cfg["duration_ms"]):
        raise ValueError("unsupported exp147 configuration")


def author_network(cfg):
    validate_configuration(cfg)
    definitions = {definition.name for definition in extensions.definitions()}
    if "pinglab.exp147.toolbox_ei/v1" not in definitions:
        extensions.register_neuron(
            "pinglab.exp147.toolbox_ei/v1",
            dynamics.neuron_step,
            initialize=dynamics.neuron_initial_state,
            input_unit="uS",
            state_units={"z_na": "mV", "z_k": "mV", "current": "nA"},
            capture_safe=True,
        )
    if "pinglab.exp147.graded/v1" not in definitions:
        extensions.register_synapse(
            "pinglab.exp147.graded/v1",
            dynamics.synapse_step,
            initialize=dynamics.synapse_initial_state,
            input_unit="mV",
            output_unit="uS",
            state_units={"gate": "mV"},
            capture_safe=True,
        )
    net = lang.Network("exp147", dt=cfg["dt_ms"] * lang.ms)
    # An unconnected clock supplies only time/batch axes. The fixed current
    # schedule lives in each neuron because custom inputs share one unit.
    net.input("clock", shape=("time", "batch", 1), signal_type="continuous", unit="1")
    cells = {}
    for name, schedule in cfg["currents"].items():
        schedule_config = dict(
            kick_start_step=round(schedule["kick_start_ms"] / cfg["dt_ms"]),
            kick_stop_step=round(schedule["kick_stop_ms"] / cfg["dt_ms"]),
            drive_start_step=round(schedule["drive_start_ms"] / cfg["dt_ms"]),
            kick_na=schedule["kick_na"],
            drive_na=schedule["drive_na"],
        )
        cells[name] = net.population(
            name,
            size=1,
            neuron=lang.CustomNeuron(
                "pinglab.exp147.toolbox_ei/v1", **cfg["neuron"], **schedule_config
            ),
        )
        net.output(f"{name}_voltage_mv", cells[name].voltage)
        net.output(f"{name}_spikes", cells[name].spikes)
        for state, unit in (("z_na", "mv"), ("z_k", "mv"), ("current", "na")):
            net.output(f"{name}_{state}_{unit}", cells[name].state(state))
    for pre, post in (("n1", "n2"), ("n2", "n1")):
        name = f"{pre}_to_{post}"
        edge = net.connect(
            cells[pre].voltage,
            cells[post].inhibitory,
            name=name,
            synapse=lang.CustomSynapse("pinglab.exp147.graded/v1", **cfg["synapse"]),
            # Explicit mV drives use dimensionless weights before gating.
            # Keep these at one and set physical g_max inside the synapse.
            weight=lang.Constant(cfg["projection_weight"]),
            initialization_scaling="direct",
            connection="recurrent",
            delay=cfg["delay_ms"] * lang.ms,
        )
        net.output(f"{name}_gate_mv", edge.state("gate"))
        net.output(f"{name}_conductance_us", edge.conductance)
    return lang.compile(net, target="tools/snnsim")


def clock(cfg):
    return torch.zeros(
        (round(cfg["duration_ms"] / cfg["dt_ms"]), 1, 1), dtype=torch.float64
    )
