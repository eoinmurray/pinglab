"""Define an MNIST-driven excitatory/inhibitory activity probe.

One fixed training image drives a full pixel-to-E projection and a ten-cell
readout. E-to-I-to-E feedback and private I background excitation are compared
with all weights fixed.

configuration() resolves three committed protocols. author_network() compiles
the shared architecture with legacy COBA refractory scheduling; neither function
loads MNIST or executes a simulation. No training specification is authored.
"""

from dataclasses import asdict

from experiments.exp144.custom.initializers import register
from snnlab import lang
from snnlab.sim import Performance


def configuration(protocol="comparison"):
    if protocol not in ("comparison", "screen", "long"):
        raise ValueError(f"unknown protocol: {protocol}")
    screen = protocol == "screen"
    cases = [
        dict(
            id=f"win{weight:g}-i{rate:g}",
            input_parent_us=weight,
            i_rate_hz=rate,
            recurrence_density=1.0 if screen or rate == 0 else 0.1,
        )
        for weight in ([0.9, 1.5, 3.0, 6.0] if screen else [3.0])
        for rate in ([0.0, 2.0, 5.0, 10.0, 15.0] if screen else [0.0, 5.0, 10.0])
    ]
    return dict(
        experiment="exp144",
        schema="exp144.recipe/v3",
        protocol=protocol,
        dataset="official MNIST training",
        dataset_directory="data",
        sample_index=0,
        sample_label=5,
        n_input=784,
        n_e=1024,
        n_i=256,
        n_output=10,
        dt_ms=0.1,
        duration_ms={"comparison": 500.0, "screen": 2000.0, "long": 11000.0}[protocol],
        burn_in_ms=0.0 if protocol == "comparison" else 1000.0,
        seeds=[42, 43, 44] if protocol == "long" else [42],
        max_pixel_rate_hz=25.0,
        input_initial_zero_fraction=0.95,
        input_relative_sd=0.1,
        reciprocal_parent_us=[1.5, 3.0],
        reciprocal_relative_sd=0.1,
        i_background_jump_us=0.15,
        resting_mv=-65.0,
        reset_mv=-65.0,
        threshold_mv=-50.0,
        capacitance_e_nf=1.0,
        capacitance_i_nf=0.5,
        leak_e_us=0.05,
        leak_i_us=0.1,
        refractory_e_ms=1.2,
        refractory_i_ms=0.6,
        tau_ampa_ms=2.0,
        tau_gaba_ms=6.0,
        recurrent_delay_ms=0.1,
        readout_parent_us=1.0,
        readout_relative_sd=0.1,
        encoding="native SNNLab pixel-rate Bernoulli Poisson discretization",
        image_seed_offset=10000,
        background_seed_offset=20000,
        recording_block_steps=1000,
        device="cpu",
        torch_threads=1,
        # Preserve float32 eager execution for comparison with the scratch probe.
        performance=asdict(
            Performance(compile_mode=None, precision="highest", cuda_graphs="off")
        ),
        recording_signals=["image.value", "E.spikes", "I.spikes"],
        analysis=dict(
            min_cv_spikes=20,
            correlation_cells=200,
            correlation_bins_ms=[5.0, 10.0, 50.0],
            rate_bin_ms=1.0,
            psd_max_segment_bins=1000,
            onset_ms=200.0,
        ),
        presentation=dict(
            seed=42,
            cases=["win3-i0", "win3-i5", "win3-i10"],
            input_window_ms=200.0,
            rate_window_ms=400.0,
        ),
        cases=cases,
        training=False,
        selection="exploratory screening; fixed image and full simulation raster",
    )


def validate_configuration(cfg):
    """Require a recorded, supported recipe before interpreting saved arrays."""
    if cfg != configuration(cfg.get("protocol")):
        raise ValueError(
            "source configuration differs from the supported exp144 recipe"
        )


def author_network(cfg, case):
    register()
    net = lang.Network("exp144", dt=cfg["dt_ms"] * lang.ms)
    image = net.input(
        "image",
        shape=("time", "batch", cfg["n_input"]),
        signal_type="spikes",
        unit="spike",
    )
    background = net.input(
        "background_i",
        shape=("time", "batch", cfg["n_i"]),
        signal_type="spikes",
        unit="spike",
    )
    pops = {}
    for name, size, kind in [
        ("E", cfg["n_e"], "e"),
        ("I", cfg["n_i"], "i"),
        ("readout", cfg["n_output"], "e"),
    ]:
        cap, leak = cfg[f"capacitance_{kind}_nf"], cfg[f"leak_{kind}_us"]
        pops[name] = net.population(
            name,
            size=size,
            neuron=lang.COBA_LIF(
                tau_mem=cap / leak * lang.ms,
                capacitance_nf=cap,
                leak_us=leak,
                resting_mv=cfg["resting_mv"],
                threshold_mv=cfg["threshold_mv"],
                reset_mv=cfg["reset_mv"],
                initial_voltage_mv=cfg["resting_mv"],
                refractory=cfg[f"refractory_{kind}_ms"] * lang.ms,
            ),
        )
    ampa = lang.AMPA(tau=cfg["tau_ampa_ms"] * lang.ms)
    gaba = lang.GABA(tau=cfg["tau_gaba_ms"] * lang.ms)
    net.connect(
        image,
        pops["E"].excitatory,
        name="input_to_E",
        synapse=ampa,
        weight=lang.LowerClampedNormal(
            case["input_parent_us"],
            case["input_parent_us"] * cfg["input_relative_sd"],
            initial_zero_fraction=cfg["input_initial_zero_fraction"],
        ),
        constraint=lang.NonNegative(),
        initialization_scaling="fan_in_normalized",
    )
    for source, target, mean, synapse in [
        ("E", "I", cfg["reciprocal_parent_us"][0], ampa),
        ("I", "E", cfg["reciprocal_parent_us"][1], gaba),
    ]:
        net.connect(
            pops[source].spikes,
            pops[target].excitatory if source == "E" else pops[target].inhibitory,
            name=f"{source}_to_{target}",
            synapse=synapse,
            weight=lang.LowerClampedNormal(
                mean,
                mean * cfg["reciprocal_relative_sd"],
                initial_zero_fraction=1.0 - case["recurrence_density"],
            ),
            constraint=lang.NonNegative(),
            initialization_scaling="fan_in_normalized",
            connection="recurrent",
            delay=cfg["recurrent_delay_ms"] * lang.ms,
        )
    net.connect(
        background,
        pops["I"].excitatory,
        name="background_to_I",
        synapse=ampa,
        weight=lang.CustomInitializer(
            "exp144.diagonal_background/v1",
            jump_us=cfg["i_background_jump_us"],
        ),
        constraint=lang.NonNegative(),
        initialization_scaling="direct",
    )
    net.connect(
        pops["E"].spikes,
        pops["readout"].excitatory,
        name="readout_projection",
        synapse=ampa,
        weight=lang.LowerClampedNormal(
            cfg["readout_parent_us"],
            cfg["readout_parent_us"] * cfg["readout_relative_sd"],
        ),
        constraint=lang.NonNegative(),
        initialization_scaling="fan_in_normalized",
    )
    scores = lang.reduce(
        pops["readout"].spikes, operation="sum", over="time", name="class_counts"
    )
    net.output("class_scores", scores)
    return lang.compile(net, refractory_timing="legacy_coba")
