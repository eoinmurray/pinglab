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
    duration=8.0,
    dt_ms=1.0,
    timer_ms=1.0,
    neurons=1,
    max_lag_ms=20.0,
    max_spike_age_ms=5.0,
    profile_timing=False,
    flexor_bias_ms=200.0,
    flexor_initial_us=0.02,
    extensor_initial_us=0.005,
    compile_mode=None,
):
    """Resolve numerical, scheduling, mechanical and recording settings."""
    if not math.isfinite(duration) or duration < 2:
        raise ValueError("Use duration >= 2 seconds")
    if type(neurons) is not int or neurons < 1:
        raise ValueError("Use an integer neurons >= 1")
    for value in (max_lag_ms, max_spike_age_ms):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Timing budgets must be positive and finite")
    if compile_mode not in (None, "default"):
        raise ValueError("Choose default compilation or eager execution")
    if type(profile_timing) is not bool:
        raise ValueError("profile_timing must be boolean")
    if not math.isfinite(flexor_bias_ms) or flexor_bias_ms < 0:
        raise ValueError("Startup duration must be finite and nonnegative")
    if any(
        not math.isfinite(value) or value < 0
        for value in (flexor_initial_us, extensor_initial_us)
    ):
        raise ValueError("Startup conductance must be finite and nonnegative")
    return dict(
        experiment="exp152",
        duration=duration,
        dt_ms=dt_ms,
        timer_ms=timer_ms,
        neurons=neurons,
        max_lag_ms=max_lag_ms,
        max_spike_age_ms=max_spike_age_ms,
        profile_timing=profile_timing,
        substeps=window_steps(dt_ms, timer_ms),
        seed=0,
        device="cpu",
        dtype="float64",
        compile_mode=compile_mode,
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
            damping=0.02,
            stiffness=0.02,
            max_torque=0.8,
            activation_tau=0.08,
            sensory_limit=0.25,
            hard_limit=1.0,
            sensory_interval=0.01,
            activation_increment=0.08,
        ),
        neuron_model="hodgkin_huxley",
        hh=dict(
            rk4_max_step_ms=0.025,
            motor=dict(
                capacitance_nf=0.01,
                g_na_us=1.2,
                g_k_us=0.36,
                g_leak_us=0.003,
                e_na_mv=50.0,
                e_k_mv=-77.0,
                e_leak_mv=-54.387,
                excitatory_reversal_mv=0.0,
                inhibitory_reversal_mv=-80.0,
                initial_voltage_mv=-65.0,
            ),
            slow=dict(
                capacitance_nf=0.02,
                g_na_us=1.2,
                g_k_us=0.36,
                g_leak_us=0.003,
                e_na_mv=50.0,
                e_k_mv=-77.0,
                e_leak_mv=-54.387,
                excitatory_reversal_mv=0.0,
                inhibitory_reversal_mv=-80.0,
                initial_voltage_mv=-65.0,
            ),
        ),
        hh_native_source_sha256=hashlib.sha256(
            (Path(__file__).parent / "custom/hh_kernel.c").read_bytes()
            + (Path(__file__).parent / "custom/hh_operator.cpp").read_bytes()
        ).hexdigest(),
        hh_reference_example_sha256="aa9d12ee5a620892a3a6653b55c45fe45b07ee275074d01e3ea7ec09443aeb90",
        hh_source_sha256=hashlib.sha256(
            (Path(__file__).parent / "custom/hodgkin_huxley.py").read_bytes()
        ).hexdigest(),
        weights=dict(
            sensory_drive=0.003,
            peer_inhibition=0.01,
            recruit_slow=0.002,
            adaptation=0.003,
        ),
        synapse_tau_ms=dict(
            local_bias=0.001,
            sensory_drive=10,
            peer_inhibition=50,
            recruit_slow=160,
            adaptation=250,
        ),
        tonic_us=0.005,
        flexor_initial_us=flexor_initial_us,
        extensor_initial_us=extensor_initial_us,
        flexor_bias_ms=flexor_bias_ms,
        analysis=dict(movement_velocity_threshold_rad_s=0.05, minimum_reversals=4),
        recording="All callback diagnostics, substep population means/counts, topic events and node logs",
        interpretation="Synchronized cell replicas increase load; simulated plant, no hard real-time guarantee",
    )


def author_network(cfg):
    """Compile one shared motor/slow architecture for both controller sides."""
    from experiments.exp152.custom.hodgkin_huxley import DEFINITION, register

    register()
    dt_ms, neurons = cfg["dt_ms"], cfg["neurons"]
    net = lang.Network("joint_controller", dt=dt_ms * lang.ms)

    def connect(
        source, target, name, weight, tau, *, inhibitory=False, recurrent=False
    ):
        return net.connect(
            source,
            target,
            name=name,
            synapse=(lang.GABA if inhibitory else lang.AMPA)(tau=tau * lang.ms),
            weight=lang.Constant(weight),
            initialization_scaling="direct",
            constraint=lang.NonNegative(),
            **(
                {"connection": "recurrent", "delay": dt_ms * lang.ms}
                if recurrent
                else {}
            ),
        )

    motor = net.population(
        "motor",
        size=neurons,
        neuron=lang.CustomNeuron(DEFINITION, **cfg["hh"]["motor"]),
    )
    slow = net.population(
        "slow",
        size=neurons,
        neuron=lang.CustomNeuron(DEFINITION, **cfg["hh"]["slow"]),
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
    connect(
        tonic, motor.excitatory, "local_bias", 1, cfg["synapse_tau_ms"]["local_bias"]
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
    connect(
        motor.spikes,
        slow.excitatory,
        "recruit_slow",
        cfg["weights"]["recruit_slow"] / neurons,
        cfg["synapse_tau_ms"]["recruit_slow"],
        recurrent=True,
    )
    connect(
        slow.spikes,
        motor.inhibitory,
        "adaptation",
        cfg["weights"]["adaptation"] / neurons,
        cfg["synapse_tau_ms"]["adaptation"],
        inhibitory=True,
        recurrent=True,
    )
    for name, cell in [("motor", motor), ("slow", slow)]:
        net.output(name + "_voltage", cell.voltage)
        net.output(name + "_spikes", cell.spikes)
    return lang.compile(net)


def startup_conditions():
    """Compare selected excitatory pulse with brief opposing-drive suppression."""
    return [
        dict(
            flexor_bias_ms=duration,
            flexor_initial_us=flexor,
            extensor_initial_us=extensor,
        )
        for duration, flexor, extensor in (
            (20, 0.01, 0.005),
            (200, 0.02, 0.005),
            (20, 0.01, 0),
            (50, 0.01, 0),
            (100, 0.01, 0),
            (200, 0.01, 0),
            (100, 0.005, 0),
            (100, 0.01, 0.0025),
            (200, 0.01, 0.0025),
        )
    ]
