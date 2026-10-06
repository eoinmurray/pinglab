"""Frozen-network confidence trajectories in physical time and volley position."""

from pathlib import Path
import hashlib
import numpy as np
from snnlab import lang
from snnlab.sim.timing import refractory_steps
from experiments.helpers.ping import build_ping

REPO = Path(__file__).resolve().parents[2]
SLUG = "exp123"
IMAGE_INDICES = tuple(range(10))
SEEDS = tuple(range(42, 142))
ACCURACY_SEEDS = (42, 43, 44)
ACCURACY_SELECTION_SEED = 123
ACCURACY_IMAGES_PER_DIGIT = 50
ACCURACY_BATCH_IMAGES = 32
TAUS_MS = (3.0, 6.0, 12.0)
DT_MS = 0.1
DURATION_MS = 200.0
STEPS = round(DURATION_MS / DT_MS)
RATE_HZ = 25.0
PARETO_NEW_RATES_HZ = (12.5, 50.0)
PARETO_RATES_HZ = (12.5, 25.0, 50.0)
PARETO_DEADLINES_MS = (20.0, 40.0, 80.0, 120.0, 200.0)
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


def accuracy_images(labels):
    rng = np.random.default_rng(ACCURACY_SELECTION_SEED)
    return tuple(
        sorted(
            int(i)
            for digit in range(10)
            for i in rng.choice(
                np.flatnonzero(np.asarray(labels) == digit),
                ACCURACY_IMAGES_PER_DIGIT,
                replace=False,
            )
        )
    )


def recording_unit(image_index, rate, protocol):
    suffix = f"-rate{rate:g}".replace(".", "p") if protocol == "pareto" else ""
    return f"image{image_index:05d}{suffix}"


def configuration(protocol="trajectories", image_indices=None):
    return dict(
        schema="exp123.compute/v1",
        protocol=protocol,
        image_indices=IMAGE_INDICES if image_indices is None else image_indices,
        image_selection="first ten"
        if protocol == "trajectories"
        else "50 randomly sampled per digit without replacement",
        image_selection_seed=None
        if protocol == "trajectories"
        else ACCURACY_SELECTION_SEED,
        partition="official_mnist_test",
        seeds=SEEDS if protocol == "trajectories" else ACCURACY_SEEDS,
        execution_batch_images=1
        if protocol == "trajectories"
        else ACCURACY_BATCH_IMAGES,
        generator_seed_rule="1000 * image_index + encoding_seed",
        tau_gaba_ms=TAUS_MS,
        dt_ms=DT_MS,
        duration_ms=DURATION_MS,
        burn_in_ms=0.0,
        input_rate_hz=None if protocol == "pareto" else RATE_HZ,
        input_rates_hz=PARETO_NEW_RATES_HZ if protocol == "pareto" else (RATE_HZ,),
        cross_rate_pairing="same per-image/draw random uniforms thresholded at each rate",
        checkpoint_unit=UNIT,
        checkpoint_role=CHECKPOINT_ROLE,
        checkpoint_sha256=CHECKPOINT_SHA256,
        checkpoint_epoch=50,
        biological_defaults=dict(
            capacitance_nf=[1.0, 0.5],
            leak_us=[0.05, 0.1],
            resting_mv=-65.0,
            reset_mv=-65.0,
            threshold_mv=-50.0,
            refractory_ms=[1.2, 0.6],
            tau_ampa_ms=2.0,
            recurrent_delay_ms=DT_MS,
            voltage_grad_dampen=1000.0,
            output_tau_ms=2.0,
            output_threshold=1.0,
        ),
        script_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        ping_helper_sha256=sha256(REPO / "experiments/helpers/ping.py"),
    )


def measurement():
    return dict(
        schema="exp123.measurement/v1",
        score="softmax of cumulative output spike counts",
        softmax_temperature=1.0,
        time_sampling="zero state plus each completed integration step",
        primary_score="true class",
        volley_smoothing_ms=1.0,
        volley_spacing_ms=2.0,
        volley_relative_prominence=0.2,
        volley_minimum_prominence=1.0,
        minimum_i_participation_pct=50.0,
        phase_origin="first detected I volley = zero",
        phase_interpolation="linear between successive volley times; no extrapolation",
        phase_grid_step=0.05,
        phase_coverage="common complete volley intervals across every setting and draw within each image",
        saturation_threshold=0.99,
        decision_rule="earliest unique true-class cumulative-count winner sustained through the final update; ties are failures",
        decision_cycle_mapping="interpolate within each accepted I-volley sequence only; no extrapolation",
        decision_summary="median and quartiles among successful trials; cycle summaries require an in-range accepted cycle coordinate",
        script_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
    )


def network(tau_ms):
    net = lang.Network("exp123_evidence", dt=DT_MS * lang.ms)
    drive = net.input(
        "image", shape=("time", "batch", 784), signal_type="spikes", unit="spike"
    )
    neurons = [
        lang.COBA_LIF(
            tau_mem=t * lang.ms,
            capacitance_nf=c,
            leak_us=g,
            resting_mv=-65.0,
            reset_mv=-65.0,
            threshold_mv=-50.0,
            initial_voltage_mv=-65.0,
            voltage_grad_dampen=1000.0,
            refractory_steps=refractory_steps(r, DT_MS, policy="exact"),
        )
        for t, c, g, r in ((20, 1.0, 0.05, 1.2), (5, 0.5, 0.1, 0.6))
    ]
    circuit = build_ping(
        net,
        source=drive,
        n_e=1024,
        n_i=256,
        neuron_e=neurons[0],
        neuron_i=neurons[1],
        ampa=lang.AMPA(tau=2 * lang.ms),
        gaba=lang.GABA(tau=tau_ms * lang.ms),
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
