"""Frozen-network confidence trajectories in physical time and volley position.

The configuration resolves scientific settings and the authoring functions compile
the network. Data loading, simulation and run storage belong to the execution stages.
"""

import hashlib
from pathlib import Path

import numpy as np
from snnlab import lang

REPO = Path(__file__).resolve().parents[2]
SLUG = "exp123"
FIGURE_SIZE_MM = (180, 101.25)
IMAGE_INDICES = tuple(range(9))
SEEDS = tuple(range(42, 142))
ACCURACY_SEEDS = (42, 43, 44)
ACCURACY_SELECTION_SEED = 123
ACCURACY_IMAGES_PER_DIGIT = 50
ACCURACY_BATCH_IMAGES = 32
TAUS_MS = (3.0, 6.0, 12.0)
SWEEP_TAUS_MS = tuple(
    float(t)
    for t in np.concatenate(
        (
            np.geomspace(3, 6, 7),
            np.geomspace(6, 12, 7)[1:],
            np.geomspace(12, 30, 7)[1:],
        )
    )
)
SWEEP_SEEDS = tuple(range(42, 52))
DT_MS = 0.1
DURATION_MS = 200.0
STEPS = round(DURATION_MS / DT_MS)
RATE_HZ = 25.0
TARGET_ACCURACY_PCT = 80.0
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
    if protocol == "trajectories":
        return f"trajectory-image{image_index:05d}"
    suffix = f"-rate{rate:g}".replace(".", "p")
    return f"image{image_index:05d}{suffix}"


def shared_configuration(labels):
    """One balanced sample and paired decay sweep, with no extra simulations."""
    return shared_configuration_for_images(accuracy_images(labels))


def shared_configuration_for_images(image_indices):
    if len(image_indices) != 500 or len(set(image_indices)) != 500:
        raise ValueError("shared sweep requires 500 distinct test images")
    cfg = configuration("benchmark", tuple(image_indices))
    cfg.update(
        seeds=SWEEP_SEEDS,
        tau_gaba_ms=SWEEP_TAUS_MS,
        input_rates_hz=(RATE_HZ,),
        execution_batch_images=1,
    )
    return dict(
        schema="exp123.compute/v3",
        sweep=cfg,
        illustrative_image_indices=list(cfg["image_indices"][:9]),
        presentations=len(cfg["image_indices"]) * len(SWEEP_SEEDS) * len(SWEEP_TAUS_MS),
        sampling="all 500 balanced images have the same ten encodings at every decay; illustrative images are a view of this sample",
    )


def configuration(protocol="benchmark", image_indices=None):
    if protocol not in ("benchmark", "trajectories"):
        raise ValueError("unsupported sampling protocol")
    return dict(
        schema="exp123.compute/v1",
        protocol=protocol,
        image_indices=IMAGE_INDICES if image_indices is None else image_indices,
        image_selection="first nine"
        if protocol == "trajectories"
        else "50 randomly sampled per digit without replacement",
        image_selection_seed=None
        if protocol == "trajectories"
        else ACCURACY_SELECTION_SEED,
        partition="official_mnist_test",
        n_input=784,
        n_e=1024,
        n_i=256,
        n_output=10,
        model_seed=0,
        device="cpu",
        torch_threads=1,
        performance=dict(compile_mode=None),
        seeds=SEEDS if protocol == "trajectories" else ACCURACY_SEEDS,
        execution_batch_images=1
        if protocol == "trajectories"
        else ACCURACY_BATCH_IMAGES,
        generator_seed_rule="1000 * image_index + encoding_seed",
        tau_gaba_ms=TAUS_MS,
        dt_ms=DT_MS,
        duration_ms=DURATION_MS,
        burn_in_ms=0.0,
        input_rate_hz=RATE_HZ,
        input_rates_hz=(RATE_HZ,),
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
        script_sha256={
            str(p.relative_to(Path(__file__).parent)): sha256(p)
            for p in Path(__file__).parent.rglob("*.py")
        },
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
        volley_count_origin="first accepted I volley = one; each subsequent accepted volley increments count",
        volley_scores="sample cumulative scores at the first minimum of smoothed I activity strictly between each accepted peak and the next; last completed update; final peak has no completion",
        completed_volley_count_origin="one at the trough following the first accepted I peak; zero is illustrative time-zero state",
        decision_volley_count="number of accepted I volleys at or before decision; accepted coverage only",
        decision_completed_volley_count="first inhibitory trough completion at or after stable decision; one-based; accepted sequences and shared Figure 2 count range only; decisions after last bounded completion excluded",
        phase_coverage="common complete volley intervals across every setting and draw within each image",
        saturation_threshold=0.99,
        decision_rule="earliest unique true-class cumulative-count winner sustained through the final update; ties are failures",
        decision_cycle_mapping="interpolate within each accepted I-volley sequence only; no extrapolation",
        decision_summary="median and quartiles among successful trials; cycle summaries require an in-range accepted cycle coordinate",
        script_sha256={
            str(p.relative_to(Path(__file__).parent)): sha256(p)
            for p in Path(__file__).parent.rglob("*.py")
        },
    )


def author_network(tau_ms, cfg=None):
    cfg = configuration() if cfg is None else cfg
    if not np.isfinite(tau_ms) or tau_ms <= 0:
        raise ValueError("inhibitory decay must be positive and finite")
    net = lang.Network("exp123_evidence", dt=cfg["dt_ms"] * lang.ms)
    drive = net.input(
        "image",
        shape=("time", "batch", cfg["n_input"]),
        signal_type="spikes",
        unit="spike",
    )
    biology = cfg["biological_defaults"]
    neurons = [
        lang.COBA_LIF(
            tau_mem=(c / g) * lang.ms,
            capacitance_nf=c,
            leak_us=g,
            resting_mv=biology["resting_mv"],
            reset_mv=biology["reset_mv"],
            threshold_mv=biology["threshold_mv"],
            initial_voltage_mv=biology["resting_mv"],
            voltage_grad_dampen=biology["voltage_grad_dampen"],
            refractory=r * lang.ms,
        )
        for c, g, r in zip(
            biology["capacitance_nf"],
            biology["leak_us"],
            biology["refractory_ms"],
            strict=True,
        )
    ]
    # Reciprocal E/I loop with excitatory drive to E only.
    ping_ampa = lang.AMPA(tau=biology["tau_ampa_ms"] * lang.ms)
    ping_gaba = lang.GABA(tau=tau_ms * lang.ms)
    ping_recurrent_delay = cfg["dt_ms"] * lang.ms
    ping_e = net.population("E", size=cfg["n_e"], neuron=neurons[0])
    ping_i = net.population("I", size=cfg["n_i"], neuron=neurons[1])
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
            tau=biology["output_tau_ms"] * lang.ms,
            soft_reset_threshold=biology["output_threshold"],
            surrogate_slope=1.0,
            initial_voltage=0.0,
        ),
    )
    net.connect(
        ping_e.spikes,
        output.excitatory,
        name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=biology["output_tau_ms"] * lang.ms),
        weight=lang.Constant(0.0),
        constraint=lang.NonNegative(),
        initialization_scaling="direct",
    )
    for signal, name in (
        (ping_e.spikes, "spk_e"),
        (ping_i.spikes, "spk_i"),
        (output.spikes, "spk_out"),
    ):
        net.expose(signal, name=name)
    return lang.compile(net, target=None)


def dynamics_configuration(source_configuration=None):
    """Resolve the native spectrum and cycle measurements for the shared sweep."""
    cfg = dict(
        configuration() if source_configuration is None else source_configuration
    )
    cfg.update(population_bin_ms=1.0, transient_ms=0.0)
    cfg.update(
        frequency_search_hz=[5.0, 150.0],
        estimator_version="snnlab-analysis-v1",
        frequency_estimator="E population full-trial Welch PSD peak in 5–150 Hz, parabolic interpolation; condition summary is peak of mean PSD",
        volley_frequency_estimator="1000 / mean interval between detected inhibitory volleys in ms",
        volley_smoothing_ms=1.0,
        volley_minimum_spacing_ms=2.0,
        volley_relative_prominence=0.2,
        volley_minimum_prominence_spikes=1.0,
        minimum_mean_i_participation_pct=50.0,
        minimum_cycles=1,
        primary_rhythmicity="snnlab.analysis.rhythmicity_scalars contrast; E raster; 1 ms bins; 100 ms maximum lag",
        rhythmicity_estimator="shared rate-normalized population autocorrelogram and pooled-event IEI histogram; helper contrast=(lobe-trough)/(lobe+trough)",
        analysis_library="snnlab.analysis",
        spectrum_settings=dict(
            method="welch",
            center=True,
            detrend=False,
            window="hann",
            nperseg="full_trial",
            scaling="density",
        ),
        spectral_peak_settings=dict(
            interpolation="parabolic",
            interpolation_boundary="spectrum",
            clamp_band=False,
        ),
        analysis_script_sha256={
            str(p.relative_to(Path(__file__).parent)): sha256(p)
            for p in Path(__file__).parent.rglob("*.py")
        },
    )
    return cfg
