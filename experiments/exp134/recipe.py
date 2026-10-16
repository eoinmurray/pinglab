"""Paired leak and capacitance interventions in a frozen trained PING circuit."""

import hashlib
from pathlib import Path

import numpy as np
from experiments.helpers.ping import build_ping
from snnlab import lang

REPO = Path(__file__).resolve().parents[2]
SLUG = "exp134"
DT_MS = 0.1
DURATION_MS = 200.0
STEPS = round(DURATION_MS / DT_MS)
RATE_HZ = 25.0
SEEDS = tuple(range(42, 52))
SELECTION_SEED = 134
IMAGES_PER_DIGIT = 50
BATCH_IMAGES = 5
MULTIPLIERS = tuple(float(x) for x in np.geomspace(0.5, 2.0, 19))
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


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditions():
    rows = [
        dict(
            condition_id="baseline",
            sweep="baseline",
            multiplier=1.0,
            capacitance_nf=[1.0, 0.5],
            leak_us=[0.05, 0.1],
        )
    ]
    for sweep in ("leak", "capacitance"):
        for index, factor in enumerate(MULTIPLIERS):
            if factor == 1.0:
                continue
            rows.append(
                dict(
                    condition_id=f"{sweep}-{index:02d}",
                    sweep=sweep,
                    multiplier=factor,
                    capacitance_nf=[
                        c * (factor if sweep == "capacitance" else 1)
                        for c in (1.0, 0.5)
                    ],
                    leak_us=[
                        g * (factor if sweep == "leak" else 1) for g in (0.05, 0.1)
                    ],
                )
            )
    return rows


def sweep_conditions(sweep):
    if sweep not in ("leak", "capacitance"):
        raise ValueError("unknown modulation sweep")
    return sorted(
        (c for c in conditions() if c["sweep"] in (sweep, "baseline")),
        key=lambda c: c["multiplier"],
    )


def select_images(images):
    rng = np.random.default_rng(SELECTION_SEED)
    return sorted(
        int(i)
        for digit in range(10)
        for i in rng.choice(
            sorted(k for k, (_, label) in images.items() if label == digit),
            IMAGES_PER_DIGIT,
            replace=False,
        )
    )


def configuration(images):
    indices = select_images(images)
    return dict(
        schema="exp134.compute/v1",
        conditions=conditions(),
        image_indices=indices,
        seeds=SEEDS,
        partition="official_mnist_test",
        image_selection="all 50 per digit from exp123's balanced 500-image bank",
        image_selection_seed=SELECTION_SEED,
        dt_ms=DT_MS,
        duration_ms=DURATION_MS,
        input_rate_hz=RATE_HZ,
        tau_gaba_ms=6.0,
        burn_in_ms=0.0,
        batch_images=BATCH_IMAGES,
        checkpoint_sha256=CHECKPOINT_SHA256,
        generator_seed_rule="1000 * image_index + draw identifier",
        baseline_shared_between_sweeps=True,
        presentations=len(indices) * len(SEEDS) * len(conditions()),
        script_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        helper_sha256=sha256(REPO / "experiments/helpers/ping.py"),
    )


def measurement():
    return dict(
        schema="exp134.analysis/v1",
        population_bin_ms=1.0,
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


def network(condition):
    net = lang.Network("exp134_membrane_modulation", dt=DT_MS * lang.ms)
    drive = net.input(
        "image", shape=("time", "batch", 784), signal_type="spikes", unit="spike"
    )
    neurons = [
        lang.COBA_LIF(
            tau_mem=(c / g) * lang.ms,
            capacitance_nf=c,
            leak_us=g,
            resting_mv=-65.0,
            reset_mv=-65.0,
            threshold_mv=-50.0,
            initial_voltage_mv=-65.0,
            voltage_grad_dampen=1000.0,
            refractory=r * lang.ms,
        )
        for c, g, r in zip(
            condition["capacitance_nf"], condition["leak_us"], (1.2, 0.6), strict=True
        )
    ]
    circuit = build_ping(
        net,
        source=drive,
        n_e=1024,
        n_i=256,
        neuron_e=neurons[0],
        neuron_i=neurons[1],
        ampa=lang.AMPA(tau=2 * lang.ms),
        gaba=lang.GABA(tau=6.0 * lang.ms),
        input_weight=lang.Constant(0.0),
        ei_weight=lang.Constant(0.0),
        ie_weight=lang.Constant(0.0),
        recurrent_delay=DT_MS * lang.ms,
        initialization_scaling="direct",
    )
    output = net.population(
        "readout",
        size=10,
        spiking=True,
        neuron=lang.LeakyIntegrator(
            tau=2 * lang.ms,
            soft_reset_threshold=1.0,
            surrogate_slope=1.0,
            initial_voltage=0.0,
        ),
    )
    net.connect(
        circuit.e.spikes,
        output.excitatory,
        name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=2 * lang.ms),
        weight=lang.Constant(0.0),
        constraint=lang.NonNegative(),
        initialization_scaling="direct",
    )
    for signal, name in (
        (circuit.e.spikes, "spk_e"),
        (circuit.i.spikes, "spk_i"),
        (output.spikes, "spk_out"),
    ):
        net.expose(signal, name=name)
    return lang.compile(net, target="tools/snnsim")
