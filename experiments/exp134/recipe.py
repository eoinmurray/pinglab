"""Define paired membrane interventions in a frozen trained PING circuit.

configuration() resolves the sampling, numerical, network and recording settings.
Each call returns fresh values; image pixels are loaded separately by compute.

author_network() compiles the reciprocal E/I circuit for one intervention.
The trained checkpoint supplies all weights; this module neither loads artifacts
nor runs simulations. The two sweeps share one baseline and identical inputs.
"""

from dataclasses import asdict

import numpy as np
from snnlab import lang
from snnlab.sim import Performance

SLUG = "exp134"
UNIT = "ping__tg6__seed42"
CHECKPOINT_ROLE = "final_epoch"
CHECKPOINT_FILE = "weights_final.pth"
CHECKPOINT_SHA256 = "34a94d90e652059ef754a214670d2fe642af504a6fcc01380d33bd22854278c6"
PARAMETERS = {
    "input_to_E.weight": "W_ff.0",
    "E_to_I.weight": "W_ei.1",
    "I_to_E.weight": "W_ie.1",
    "readout_projection.weight": "W_ff.1",
}


def conditions(cfg=None):
    cfg = configuration() if cfg is None else cfg
    rows = [
        dict(
            condition_id="baseline",
            sweep="baseline",
            multiplier=1.0,
            capacitance_nf=list(cfg["capacitance_nf"]),
            leak_us=list(cfg["leak_us"]),
        )
    ]
    for sweep in ("leak", "capacitance"):
        for index, factor in enumerate(cfg["multipliers"]):
            if factor == 1.0:
                continue
            rows.append(
                dict(
                    condition_id=f"{sweep}-{index:02d}",
                    sweep=sweep,
                    multiplier=factor,
                    capacitance_nf=[
                        c * (factor if sweep == "capacitance" else 1)
                        for c in cfg["capacitance_nf"]
                    ],
                    leak_us=[
                        g * (factor if sweep == "leak" else 1) for g in cfg["leak_us"]
                    ],
                )
            )
    return rows


def select_images(images, cfg):
    rng = np.random.default_rng(cfg["image_selection_seed"])
    return sorted(
        int(i)
        for digit in range(10)
        for i in rng.choice(
            sorted(k for k, (_, label) in images.items() if label == digit),
            cfg["images_per_digit"],
            replace=False,
        )
    )


def configuration(images=None, *, images_per_digit=None, batch_images=None):
    images_per_digit = 50 if images_per_digit is None else images_per_digit
    batch_images = 5 if batch_images is None else batch_images
    if not isinstance(images_per_digit, int) or not 1 <= images_per_digit <= 50:
        raise ValueError("images_per_digit must be an integer from 1 to 50")
    if not isinstance(batch_images, int) or batch_images < 1:
        raise ValueError("batch_images must be a positive integer")
    cfg = dict(
        schema="exp134.compute/v1",
        seeds=list(range(42, 52)),
        partition="official_mnist_test",
        image_selection="balanced sample without replacement from the validated 500-image bank",
        image_selection_seed=134,
        images_per_digit=images_per_digit,
        dt_ms=0.1,
        duration_ms=200.0,
        input_rate_hz=25.0,
        n_input=784,
        n_excitatory=1024,
        n_inhibitory=256,
        n_output=10,
        capacitance_nf=[1.0, 0.5],
        leak_us=[0.05, 0.1],
        refractory_ms=[1.2, 0.6],
        resting_mv=-65.0,
        reset_mv=-65.0,
        threshold_mv=-50.0,
        voltage_grad_dampen=1000.0,
        tau_ampa_ms=2.0,
        tau_gaba_ms=6.0,
        readout_tau_ms=2.0,
        readout_threshold=1.0,
        readout_initial_voltage=0.0,
        readout_surrogate_slope=1.0,
        burn_in_ms=0.0,
        batch_images=batch_images,
        torch_threads=1,
        model_seed=0,
        performance=asdict(Performance(compile_mode=None)),
        recording_signals=["spk_e", "spk_i"],
        multipliers=[float(x) for x in np.geomspace(0.5, 2.0, 19)],
        checkpoint_unit=UNIT,
        checkpoint_role=CHECKPOINT_ROLE,
        checkpoint_file=CHECKPOINT_FILE,
        checkpoint_sha256=CHECKPOINT_SHA256,
        parameter_mapping=dict(PARAMETERS),
        device="cpu",
        checkpoint_contract=dict(
            dt=0.1,
            input_rate=25.0,
            t_ms=200.0,
            tau_gaba_ms=6.0,
            tau_ampa_ms=2.0,
            n_in=784,
            n_hidden=1024,
            n_inh=256,
            n_out=10,
            train_leak=False,
            adaptive_threshold=False,
            input_rate_sampling="fixed",
            v_grad_dampen=1000.0,
            surrogate_slope=1.0,
            state_clamp=False,
            readout_mode="mem-mean",
            signed_readout=False,
            readout_bias=False,
        ),
        generator_seed_rule="1000 * image_index + draw identifier",
        baseline_shared_between_sweeps=True,
    )
    cfg["conditions"] = conditions(cfg)
    if images is not None:
        cfg["image_indices"] = select_images(images, cfg)
    cfg["presentations"] = (
        10 * images_per_digit * len(cfg["seeds"]) * len(cfg["conditions"])
    )
    return cfg


def measurement():
    return dict(
        schema="exp134.analysis/v1",
        population_bin_ms=1.0,
        rhythmicity_max_lag_ms=100.0,
        frequency_search_hz=[5.0, 150.0],
        minimum_cycles=1,
        volley_smoothing_ms=1.0,
        volley_minimum_spacing_ms=2.0,
        volley_relative_prominence=0.2,
        volley_minimum_prominence_spikes=1.0,
        minimum_mean_i_participation_pct=50.0,
        aggregation="peak of mean full-trial E PSD; mean E spikes/s and finite rhythm contrast across draws/images; participation averages accepted draws within image then eligible images",
        show_uncertainty_estimates=False,
    )


def author_network(condition, cfg=None):
    cfg = configuration() if cfg is None else cfg
    net = lang.Network("exp134_membrane_modulation", dt=cfg["dt_ms"] * lang.ms)
    drive = net.input(
        "image",
        shape=("time", "batch", cfg["n_input"]),
        signal_type="spikes",
        unit="spike",
    )
    neurons = [
        lang.COBA_LIF(
            tau_mem=(c / g) * lang.ms,
            capacitance_nf=c,
            leak_us=g,
            resting_mv=cfg["resting_mv"],
            reset_mv=cfg["reset_mv"],
            threshold_mv=cfg["threshold_mv"],
            initial_voltage_mv=cfg["resting_mv"],
            voltage_grad_dampen=cfg["voltage_grad_dampen"],
            refractory=r * lang.ms,
        )
        for c, g, r in zip(
            condition["capacitance_nf"],
            condition["leak_us"],
            cfg["refractory_ms"],
            strict=True,
        )
    ]
    # Reciprocal E/I loop with excitatory drive to E only.
    ping_ampa = lang.AMPA(tau=cfg["tau_ampa_ms"] * lang.ms)
    ping_gaba = lang.GABA(tau=cfg["tau_gaba_ms"] * lang.ms)
    ping_recurrent_delay = cfg["dt_ms"] * lang.ms
    ping_e = net.population("E", size=cfg["n_excitatory"], neuron=neurons[0])
    ping_i = net.population("I", size=cfg["n_inhibitory"], neuron=neurons[1])
    net.connect(
        drive,
        ping_e.excitatory,
        name="input_to_E",
        synapse=ping_ampa,
        weight=lang.Constant(0.0),
        constraint=lang.NonNegative(),
        initialization_scaling="direct",
    )
    net.connect(
        ping_e.spikes,
        ping_i.excitatory,
        name="E_to_I",
        synapse=ping_ampa,
        weight=lang.Constant(0.0),
        constraint=lang.NonNegative(),
        connection="recurrent",
        delay=ping_recurrent_delay,
        initialization_scaling="direct",
    )
    net.connect(
        ping_i.spikes,
        ping_e.inhibitory,
        name="I_to_E",
        synapse=ping_gaba,
        weight=lang.Constant(0.0),
        constraint=lang.NonNegative(),
        connection="recurrent",
        delay=ping_recurrent_delay,
        initialization_scaling="direct",
    )
    output = net.population(
        "readout",
        size=cfg["n_output"],
        spiking=True,
        neuron=lang.LeakyIntegrator(
            tau=cfg["readout_tau_ms"] * lang.ms,
            soft_reset_threshold=cfg["readout_threshold"],
            surrogate_slope=cfg["readout_surrogate_slope"],
            initial_voltage=cfg["readout_initial_voltage"],
        ),
    )
    net.connect(
        ping_e.spikes,
        output.excitatory,
        name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=cfg["readout_tau_ms"] * lang.ms),
        weight=lang.Constant(0.0),
        constraint=lang.NonNegative(),
        initialization_scaling="direct",
    )
    for signal, name in (
        (ping_e.spikes, "spk_e"),
        (ping_i.spikes, "spk_i"),
    ):
        net.expose(signal, name=name)
    return lang.compile(net, target="tools/snnsim")
