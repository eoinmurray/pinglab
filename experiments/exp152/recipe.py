"""Define the paired spiking controllers and the rotary-joint protocol.

Adapted from snnlab-ros2 at the source revision recorded in configuration.
This module authors the native bundle; execution and ROS imports live elsewhere.
"""

import hashlib
import math
from pathlib import Path

from snnlab import lang

SIDES = ("flexor", "extensor")
DT_CHOICES = (0.1, 0.25, 0.5, 1.0, 2.0)
SNNLAB_REF = "3551f5a7a1a628808ef67de55253cb30bba2b752"


def window_steps(dt_ms, timer_ms):
    """Require a whole number of numerical steps in each callback window."""
    if dt_ms not in DT_CHOICES or not math.isfinite(timer_ms) or timer_ms <= 0:
        raise ValueError("Choose a supported dt and a positive, finite timer period")
    ratio = timer_ms / dt_ms
    if ratio < 1 or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-9):
        raise ValueError(
            "Timer period must be an integer multiple of the simulation step"
        )
    return round(ratio)


def configuration(
    *,
    duration=30.0,
    dt_ms=1.0,
    timer_ms=1.0,
    neurons=1,
    max_lag_ms=20.0,
    max_spike_age_ms=5.0,
    profile_timing=False,
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
    spike_transport="logical",
):
    """Resolve numerical, scheduling, mechanical and recording settings."""
    if not math.isfinite(duration) or duration < 2:
        raise ValueError("Use duration >= 2 seconds")
    if type(neurons) is not int or neurons < 1:
        raise ValueError("Use an integer neurons >= 1")
    for value in (max_lag_ms, max_spike_age_ms):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Timing budgets must be positive and finite")
    if spike_transport not in ("arrival", "logical") or compile_mode not in (
        None,
        "default",
    ):
        raise ValueError(
            "Choose arrival/logical transport and default compilation/eager execution"
        )
    if type(profile_timing) is not bool:
        raise ValueError("profile_timing must be boolean")
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
        profile_timing=profile_timing,
        controller_warmup_ms=controller_warmup_ms,
        substeps=window_steps(dt_ms, timer_ms),
        seed=0,
        device="cpu",
        dtype="float64",
        compile_mode=compile_mode,
        spike_transport=spike_transport,
        snnlab_ref=SNNLAB_REF,
        adaptation="snnlab-ros2 three-process spike-only joint controller",
        source_revision="901d5b624c85aa4858aa0a0007e479d64e60e0ea",
        source_worktree_dirty=True,
        source_file_sha256={
            "recipe.py": "8ca23269bd2c9fa44d074aad6d9ddfa092f38c43b5640f41a542cd0c115519c9",
            "node_flexor.py": "86a327a44e1f96bb1286d795d77f6b6ddd5a0ba08115ed98f028c5c2459090c7",
            "node_extensor.py": "e6394e0a441197a70bb4a081f3a85e41b341fafb787b7c796b3ec41884d0545f",
            "node_plant.py": "ed986bb51b633a6b63416b1efbabc7fd8afb6ce76f0c64546c91904ee969881a",
            "test.py": "662f4d6eb88e4afc8bd1867ac3c365c0c3456b207d6a2b30e6b4da48e260ac5c",
            "replay.py": "c99210c0159b4995ec4411538b62c4572995f857d92b9e0e3f8223dd2176b541",
        },
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
        cycle_selection_source=dict(
            run_id="exp152-r104-analyse",
            role="native-weight cycle validation of default settings",
        ),
        load_selection_source=dict(
            run_id="exp152-r091-analyse",
            role="tip-mass operating point selection; overrides are exploratory",
        ),
        feasibility_source=dict(
            run_id="exp152-r072-analyse",
            role="parameter selection only; new executions use experiment-local code",
        ),
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
        interpretation="Synchronized cell replicas increase load; simulated plant, no hard real-time guarantee",
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


def startup_conditions():
    """Compare bounded startup pulses for the single-neuron architecture."""
    return [
        dict(flexor_bias_ms=duration, flexor_initial_us=drive, extensor_initial_us=0.0)
        for duration, drive in ((10, 0.5), (20, 0.5), (50, 0.5), (20, 0.25))
    ]


def cycle_conditions(grid="synaptic"):
    """Bounded search; each row is an independent paired-controller condition."""
    if grid == "actuation":
        return [
            dict(
                tonic_us=tonic,
                peer_inhibition_us=0.2,
                sensory_drive_us=sensory,
                peer_tau_ms=50.0,
                activation_increment=gain,
            )
            for tonic in (0.04, 0.05, 0.06)
            for sensory in (0.01, 0.02, 0.025, 0.035, 0.05, 0.075, 0.1)
            for gain in (0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009)
        ]
    return [
        dict(
            tonic_us=tonic,
            peer_inhibition_us=inhibition,
            sensory_drive_us=sensory,
            peer_tau_ms=tau,
        )
        for tau in (50.0, 150.0)
        for tonic in (0.0, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2)
        for inhibition in (0.2, 0.5, 1.0)
        for sensory in (0.025, 0.005)
    ]


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
