"""Define the paired spiking controllers and the rotary-joint protocol.

The report protocol uses single-cell Elevate controllers and 1 ms logical
coupling. This module owns settings and native network authoring; execution
and ROS imports live in the compute stage and custom adapters.
"""

import hashlib
import math
from pathlib import Path

from snnlab import lang

SIDES = ("flexor", "extensor")
SNNLAB_REF = "3551f5a7a1a628808ef67de55253cb30bba2b752"


def window_steps(dt_ms, timer_ms):
    """Retain the report's single 1 ms model step per callback window."""
    if dt_ms != 1.0 or timer_ms != 1.0:
        raise ValueError("The report protocol requires 1 ms model and control steps")
    return 1


def configuration(
    *,
    duration=30.0,
    dt_ms=1.0,
    timer_ms=1.0,
    neurons=1,
    max_lag_ms=20.0,
    max_spike_age_ms=5.0,
    controller_warmup_ms=1000.0,
    flexor_bias_ms=20.0,
    flexor_initial_us=0.5,
    extensor_initial_us=0.0,
    compile_mode=None,
    initial_angle_deg=90.0,
    tip_mass_kg=0.002,
    arm_length_m=0.2,
    gravity_m_s2=9.81,
    tonic_us=0.04,
    sensory_drive_us=0.035,
    peer_inhibition_us=0.2,
    sensory_tau_ms=10.0,
    peer_tau_ms=50.0,
    activation_increment=0.003,
):
    """Resolve numerical, scheduling, mechanical and recording settings."""
    if not math.isfinite(duration) or duration < 2:
        raise ValueError("Use duration >= 2 seconds")
    if type(neurons) is not int or neurons != 1:
        raise ValueError("The report protocol requires one neuron per controller")
    for value in (max_lag_ms, max_spike_age_ms):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Timing budgets must be positive and finite")
    if compile_mode not in (None, "default"):
        raise ValueError("Choose default compilation or eager execution")
    if not math.isfinite(controller_warmup_ms) or controller_warmup_ms <= 0:
        raise ValueError("Controller warmup must be positive and finite")
    if not math.isfinite(flexor_bias_ms) or flexor_bias_ms < 0:
        raise ValueError("Startup duration must be finite and nonnegative")
    if any(
        not math.isfinite(value) or value < 0
        for value in (flexor_initial_us, extensor_initial_us)
    ):
        raise ValueError("Startup conductance must be finite and nonnegative")
    if not math.isfinite(initial_angle_deg):
        raise ValueError("Initial angle must be finite")
    if any(not math.isfinite(v) or v < 0 for v in (tip_mass_kg, gravity_m_s2)):
        raise ValueError("Tip mass and gravity must be finite and nonnegative")
    if not math.isfinite(arm_length_m) or arm_length_m <= 0:
        raise ValueError("Arm length must be finite and positive")
    if any(
        not math.isfinite(v) or v < 0
        for v in (tonic_us, sensory_drive_us, peer_inhibition_us)
    ):
        raise ValueError("Neural conductances must be finite and nonnegative")
    if any(
        not math.isfinite(v) or v <= 0
        for v in (sensory_tau_ms, peer_tau_ms, activation_increment)
    ):
        raise ValueError(
            "Synaptic timescales and muscle increment must be finite and positive"
        )
    reference = math.radians(initial_angle_deg)
    return dict(
        experiment="exp152",
        duration=duration,
        dt_ms=dt_ms,
        timer_ms=timer_ms,
        neurons=neurons,
        max_lag_ms=max_lag_ms,
        max_spike_age_ms=max_spike_age_ms,
        controller_warmup_ms=controller_warmup_ms,
        substeps=window_steps(dt_ms, timer_ms),
        seed=0,
        device="cpu",
        dtype="float64",
        compile_mode=compile_mode,
        spike_transport="logical",
        snnlab_ref=SNNLAB_REF,
        adaptation="snnlab-ros2 three-process spike-only joint controller",
        joint=dict(
            inertia=0.005,
            tip_mass_kg=tip_mass_kg,
            arm_length_m=arm_length_m,
            gravity_m_s2=gravity_m_s2,
            total_inertia=0.005 + tip_mass_kg * arm_length_m**2,
            initial_angle=reference,
            reference_angle=reference,
            spring_rest_angle=reference,
            angle_convention="zero_downward_positive_right",
            damping=0.02,
            stiffness=0.02,
            max_torque=0.8,
            activation_tau=0.08,
            sensory_limit=0.25,
            hard_limit=1.0,
            sensory_interval=0.01,
            activation_increment=activation_increment,
        ),
        neuron_model="elevate",
        controller_populations=["motor"],
        neuron=dict(
            capacitance_nf=1.0,
            g_na_us=100.0,
            g_k_us=65.0,
            g_ca_us=2.0,
            g_kca_us=10.0,
            g_leak_us=2.0,
            e_na_mv=40.0,
            e_k_mv=-90.0,
            e_ca_mv=120.0,
            e_leak_mv=-50.0,
            excitatory_reversal_mv=0.0,
            inhibitory_reversal_mv=-80.0,
            tau_na_activation_ms=0.1,
            tau_na_inactivation_ms=1.0,
            tau_k_ms=1.0,
            tau_ca_ms=1.0,
            tau_kca_ms=75.0,
            initial_voltage_mv=-60.0,
            current_na=-5.0,
            max_step_ms=0.025,
        ),
        neuron_implementation="snnlab_custom_neuron_torch_exponential_split",
        neuron_source_sha256=hashlib.sha256(
            (Path(__file__).parent / "custom/elevate.py").read_bytes()
        ).hexdigest(),
        weights=dict(
            sensory_drive=sensory_drive_us, peer_inhibition=peer_inhibition_us
        ),
        synapse_tau_ms=dict(sensory_drive=sensory_tau_ms, peer_inhibition=peer_tau_ms),
        tonic_us=tonic_us,
        flexor_initial_us=flexor_initial_us,
        extensor_initial_us=extensor_initial_us,
        flexor_bias_ms=flexor_bias_ms,
        analysis=dict(
            movement_velocity_threshold_rad_s=0.05,
            minimum_reversals=4,
            cycle_count=10,
            cycle_tolerance=0.01,
            cycle_burn_in_s=5.0,
            cycle_phase_samples=128,
        ),
        recording="All callback diagnostics, substep population means/counts, topic events and node logs",
        interpretation="Simulated loaded joint; no hard-real-time guarantee",
    )


def author_network(cfg):
    """Compile one motor neuron population, shared by both controller sides."""
    from experiments.exp152.custom.elevate import DEFINITION, register

    register()
    dt_ms, neurons = cfg["dt_ms"], cfg["neurons"]
    net = lang.Network("joint_controller", dt=dt_ms * lang.ms)

    def connect(source, target, name, weight, tau, *, inhibitory=False):
        return net.connect(
            source,
            target,
            name=name,
            synapse=(lang.GABA if inhibitory else lang.AMPA)(tau=tau * lang.ms),
            weight=lang.Constant(weight),
            initialization_scaling="direct",
            constraint=lang.NonNegative(),
        )

    motor = net.population(
        "motor",
        size=neurons,
        neuron=lang.CustomNeuron(DEFINITION, **cfg["neuron"]),
    )
    tonic = net.input(
        "tonic", shape=("time", "batch", 1), signal_type="continuous", unit="uS"
    )
    sensory = net.input(
        "sensory", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
    peer = net.input(
        "peer", shape=("time", "batch", 1), signal_type="spikes", unit="spike"
    )
    # The validated startup conductance is direct, with no additional filter.
    net.connect(
        tonic,
        motor.excitatory,
        name="local_bias",
        synapse=lang.CustomSynapse("pinglab.exp152.direct_conductance/v1"),
        weight=lang.Constant(1.0),
        initialization_scaling="direct",
    )
    connect(
        sensory,
        motor.excitatory,
        "sensory_drive",
        cfg["weights"]["sensory_drive"],
        cfg["synapse_tau_ms"]["sensory_drive"],
    )
    connect(
        peer,
        motor.inhibitory,
        "peer_inhibition",
        cfg["weights"]["peer_inhibition"] / neurons,
        cfg["synapse_tau_ms"]["peer_inhibition"],
        inhibitory=True,
    )
    net.output("motor_voltage", motor.voltage)
    net.output("motor_spikes", motor.spikes)
    return lang.compile(net)


def controller_settings(cfg):
    """Flatten recorded settings for the controller adapter without live defaults."""
    return {
        **{
            key: cfg[key]
            for key in (
                "duration",
                "flexor_bias_ms",
                "flexor_initial_us",
                "extensor_initial_us",
                "controller_warmup_ms",
                "tonic_us",
            )
        },
        "sensory_drive_us": cfg["weights"]["sensory_drive"],
        "peer_inhibition_us": cfg["weights"]["peer_inhibition"],
        "sensory_tau_ms": cfg["synapse_tau_ms"]["sensory_drive"],
        "peer_tau_ms": cfg["synapse_tau_ms"]["peer_inhibition"],
        "initial_angle_deg": math.degrees(cfg["joint"]["initial_angle"]),
        **{
            key: cfg["joint"][key]
            for key in (
                "tip_mass_kg",
                "arm_length_m",
                "gravity_m_s2",
                "activation_increment",
            )
        },
    }
