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
    max_lag_ms=8000.0,
    compile_mode="default",
    controller_warmup_ms=1000.0,
):
    """Resolve the fixed report model and its acquisition settings.

    The extended lag allowance reproduces the displayed diagnostic recording;
    it does not relax the 1 ms deadline or the 5 ms spike-delivery verdict.
    """
    if not math.isfinite(duration) or duration < 2:
        raise ValueError("Use duration >= 2 seconds")
    if not math.isfinite(max_lag_ms) or max_lag_ms <= 0:
        raise ValueError("Maximum lag must be positive and finite")
    if compile_mode not in (None, "default"):
        raise ValueError("Choose default compilation or eager validation")
    if not math.isfinite(controller_warmup_ms) or controller_warmup_ms <= 0:
        raise ValueError("Controller warmup must be positive and finite")
    reference = math.pi / 2
    return dict(
        experiment="exp152",
        duration=duration,
        dt_ms=1.0,
        timer_ms=1.0,
        neurons=1,
        max_lag_ms=max_lag_ms,
        max_spike_age_ms=5.0,
        controller_warmup_ms=controller_warmup_ms,
        substeps=1,
        seed=0,
        device="cpu",
        dtype="float64",
        compile_mode=compile_mode,
        spike_transport="logical",
        snnlab_ref=SNNLAB_REF,
        joint=dict(
            inertia=0.005,
            tip_mass_kg=0.002,
            arm_length_m=0.2,
            gravity_m_s2=9.81,
            total_inertia=0.005 + 0.002 * 0.2**2,
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
            activation_increment=0.003,
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
        weights=dict(sensory_drive=0.035, peer_inhibition=0.2),
        synapse_tau_ms=dict(sensory_drive=10.0, peer_inhibition=50.0),
        tonic_us=0.04,
        flexor_initial_us=0.5,
        extensor_initial_us=0.0,
        flexor_bias_ms=20.0,
        analysis=dict(
            movement_velocity_threshold_rad_s=0.05,
            minimum_reversals=4,
            cycle_count=10,
            cycle_tolerance=0.01,
            cycle_burn_in_s=5.0,
            cycle_phase_samples=128,
        ),
        recording="Per-node callback diagnostics, single-cell voltage/spikes, joint state, topic events and node logs",
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
