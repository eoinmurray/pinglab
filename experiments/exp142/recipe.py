"""Define the frozen digit networks and their physical-time comparison.

configuration() returns independent scientific and execution settings, including
an explicit source pin. author_network() compiles each discrete clock variant;
it neither reads source artifacts nor launches simulation or training."""

from copy import deepcopy

from dataclasses import asdict
from math import isfinite

from snnlab import lang
from snnlab.sim import Performance


def configuration(*, input_rate_hz=None):
    if input_rate_hz is None:
        input_rate_hz = 100.0
    if (
        isinstance(input_rate_hz, bool)
        or not isfinite(input_rate_hz)
        or not 0 < input_rate_hz <= 5000
    ):
        raise ValueError(
            "input_rate_hz must be positive, finite and supported by the 0.2 ms grid"
        )
    return deepcopy(
        dict(
            schema="exp142.recipe/v5",
            experiment="exp142",
            source=dict(
                run_id="exp141-r012-compute",
                payload_digest="sha256:45238b946b7b75cd34cdc05c944f2ee3860d46631f07835147e294f49c195bcd",
            ),
            seed=42,
            cpu_threads=1,
            performance=asdict(Performance()),
            voltage_grad_dampen=1000.0,
            models=["coba", "ping"],
            n_input=64,
            n_e=128,
            n_i=32,
            n_output=10,
            duration_ns=200_000_000,
            sample_dt_ns=100_000,
            input_grid_ns=200_000,
            native_sample_dt_ns=200_000,
            input_rate_hz=float(input_rate_hz),
            source_input_rate_hz=25.0,
            input_seed=0,
            sample_index=0,
            physics=dict(
                capacitance_e_nf=1.0,
                capacitance_i_nf=0.5,
                leak_e_us=0.05,
                leak_i_us=0.1,
                resting_mv=-65.0,
                reset_mv=-65.0,
                threshold_mv=-50.0,
                excitatory_reversal_mv=0.0,
                inhibitory_reversal_mv=-80.0,
                tau_ampa_ms=2.0,
                tau_gaba_ms=6.0,
                refractory_e_ms=1.2,
                refractory_i_ms=0.6,
                recurrent_delay_ms=0.2,
            ),
            cases=[
                dict(id="dt0p1", dt_ns=100_000),
                dict(id="dt0p05", dt_ns=50_000),
                dict(id="dt0p025", dt_ns=25_000),
                dict(id="dt0p0125", dt_ns=12_500),
            ],
            solver=dict(rtol=1e-8, atol=1e-9, max_step_ms=0.05),
            refinement=dict(rtol=1e-10, atol=1e-11, max_step_ms=0.025),
            refinement_acceptance=dict(
                max_spike_shift_ms=1e-3, max_guarded_voltage_mae_mv=1e-3
            ),
            guard_ms=2.0,
            target_e=0,
            target_output=0,
            training=False,
            source_checkpoint="validation_selected",
        )
    )


def author_network(mode, cfg, dt_ns, *, matched):
    if mode not in cfg["models"]:
        raise ValueError("unknown mode")
    p = cfg["physics"]
    dt = dt_ns / 1e6
    net = lang.Network("exp142_frozen_digits", dt=dt * lang.ms)
    drive = net.input(
        "drive",
        shape=("time", "batch", cfg["n_input"]),
        signal_type="spikes",
        unit="spike",
    )
    populations = {}
    for name, size, kind in [
        ("E", cfg["n_e"], "e"),
        ("readout", cfg["n_output"], "e"),
        *([("I", cfg["n_i"], "i")] if mode == "ping" else []),
    ]:
        cap, leak = p[f"capacitance_{kind}_nf"], p[f"leak_{kind}_us"]
        pop = net.population(
            name,
            size=size,
            neuron=lang.COBA_LIF(
                tau_mem=cap / leak * lang.ms,
                capacitance_nf=cap,
                leak_us=leak,
                resting_mv=p["resting_mv"],
                reset_mv=p["reset_mv"],
                threshold_mv=p["threshold_mv"],
                initial_voltage_mv=p["resting_mv"],
                refractory=p[f"refractory_{kind}_ms"] * lang.ms,
                voltage_grad_dampen=cfg["voltage_grad_dampen"],
            ),
        )
        populations[name] = pop
        net.expose(pop.voltage, name=f"{name}_voltage")
        net.expose(pop.spikes, name=f"{name}_spikes")

    def connect(source, target, name, *, inhibitory=False, delay=None):
        projection = net.connect(
            source,
            target,
            name=name,
            synapse=(
                lang.GABA(tau=p["tau_gaba_ms"] * lang.ms)
                if inhibitory
                else lang.AMPA(tau=p["tau_ampa_ms"] * lang.ms)
            ),
            weight=lang.Constant(0),
            constraint=lang.NonNegative(),
            initialization_scaling="direct",
            connection="recurrent" if delay else "feedforward",
            delay=delay * lang.ms if delay else None,
        )
        net.expose(projection.conductance, name=f"{name}_conductance")

    connect(drive, populations["E"].excitatory, "input_to_E")
    # End-stamped hidden spikes arrive at the next slot start with no physical lag.
    connect(
        populations["E"].spikes,
        populations["readout"].excitatory,
        "readout_projection",
        delay=dt if matched else None,
    )
    if mode == "ping":
        # Start-stamped receptor arrivals need one extra slot beyond physical delay.
        delay = p["recurrent_delay_ms"] + (dt if matched else 0)
        connect(
            populations["E"].spikes, populations["I"].excitatory, "E_to_I", delay=delay
        )
        connect(
            populations["I"].spikes,
            populations["E"].inhibitory,
            "I_to_E",
            inhibitory=True,
            delay=delay,
        )
    net.output(
        "class_scores",
        lang.reduce(
            populations["readout"].spikes,
            operation="sum",
            over="time",
            name="spike_counts",
        ),
    )
    return lang.compile(net, target="tools/snnsim")
