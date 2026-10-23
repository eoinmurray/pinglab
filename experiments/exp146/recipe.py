"""Author the Neuromorphic Toolbox Task 1 neuron and current-pulse protocol.

The proposal's sigmoid-gated sodium/potassium model uses gate-first forward
Euler, not classical Hodgkin–Huxley kinetics. This module resolves scientific
settings and compiles a native SNNLab bundle; it never executes a simulation.
"""

from dataclasses import asdict

import numpy as np
import torch
from experiments.exp146.custom import dynamics
from snnlab import extensions, lang
from snnlab.sim import Performance


def configuration(*, duration_ms=50.0):
    """Resolve the proposal protocol; shorter durations support focused tests."""
    if not np.isfinite(duration_ms) or not 0 < duration_ms <= 50:
        raise ValueError("duration_ms must be positive and at most 50 ms")
    cfg = dict(
        experiment="exp146",
        schema="exp146.recipe/v1",
        seed=0,
        device="cpu",
        dtype="float64",
        torch_threads=1,
        dt_ms=0.001,
        duration_ms=float(duration_ms),
        integration="gate-first forward Euler",
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
        pulse_start_ms=10.0,
        pulse_current_na=0.8,
        spike_threshold_mv=-20.0,
        performance=asdict(Performance(compile_mode=None, cuda_graphs="off")),
        reference_status="Toolbox reference traces not supplied",
        # The proposal labels voltage/time but not C/g/I. nF/uS/nA are a
        # dimensionally consistent embedding with unchanged numerical values.
        unit_convention="mV, ms, nF, uS, nA; C/g/I units assumed",
        validation=dict(voltage_atol_mv=0.1, spike_atol_ms=0.05),
    )
    steps = cfg["duration_ms"] / cfg["dt_ms"]
    if not np.isclose(steps, round(steps), rtol=0, atol=1e-8):
        raise ValueError("duration must contain an integer number of timesteps")
    return cfg


def validate_configuration(cfg):
    if cfg != configuration(duration_ms=cfg["duration_ms"]):
        raise ValueError("unsupported exp146 configuration")


def author_network(cfg):
    """Register importable callbacks and compile current → one custom neuron."""
    validate_configuration(cfg)
    try:
        extensions.get("neuron", "pinglab.exp146.toolbox_ei/v1")
    except ValueError:
        extensions.register_neuron(
            "pinglab.exp146.toolbox_ei/v1",
            dynamics.step,
            initialize=dynamics.initial_state,
            input_unit="nA",
            state_units={"z_na": "mV", "z_k": "mV", "g_na": "uS", "g_k": "uS"},
            capture_safe=True,
        )
    try:
        extensions.get("synapse", "pinglab.exp146.injected_current/v1")
    except ValueError:
        extensions.register_synapse(
            "pinglab.exp146.injected_current/v1",
            dynamics.inject_current,
            output_unit="nA",
            capture_safe=True,
        )
    net = lang.Network("exp146", dt=cfg["dt_ms"] * lang.ms)
    drive = net.input(
        "drive",
        shape=("time", "batch", 1),
        signal_type="continuous",
        unit="1",
    )
    keys = (
        "initial_voltage_mv",
        "capacitance_nf",
        "leak_us",
        "leak_reversal_mv",
        "g_na_us",
        "g_k_us",
        "e_na_mv",
        "e_k_mv",
        "tau_na_ms",
        "tau_k_ms",
        "gamma_na_per_mv",
        "gamma_k_per_mv",
        "bias_na_mv",
        "bias_k_mv",
        "spike_threshold_mv",
    )
    neuron = net.population(
        "cell",
        size=1,
        neuron=lang.CustomNeuron(
            "pinglab.exp146.toolbox_ei/v1",
            **{key: cfg[key] for key in keys},
        ),
    )
    injection = net.connect(
        drive,
        neuron.excitatory,
        name="injection",
        synapse=lang.CustomSynapse("pinglab.exp146.injected_current/v1"),
        weight=lang.Constant(1.0),
        initialization_scaling="direct",
    )
    net.output("voltage_mv", neuron.voltage)
    net.output("spikes", neuron.spikes)
    net.output("current_na", injection.current)
    for name in ("z_na", "z_k", "g_na", "g_k"):
        suffix = "mv" if name.startswith("z_") else "us"
        net.output(f"{name}_{suffix}", neuron.state(name))
    return lang.compile(net, target="tools/snnsim")


def current_waveform(cfg):
    """Inputs apply over intervals [k*dt, (k+1)*dt), in nA at unit weight."""
    steps = round(cfg["duration_ms"] / cfg["dt_ms"])
    current = torch.zeros((steps, 1, 1), dtype=torch.float64)
    current[round(cfg["pulse_start_ms"] / cfg["dt_ms"]) :] = cfg["pulse_current_na"]
    return current
