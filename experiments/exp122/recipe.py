"""One-image paired-encoding inhibitory-decay calibration."""

from pathlib import Path
import hashlib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SLUG = "exp122"
SOURCE = "exp022-r001-compute"
UNIT = "ping__tg6__seed42"
CHECKPOINT_ROLE = "final_epoch"
CHECKPOINT_FILE = "weights_final.pth"
CHECKPOINT_SHA256 = "34a94d90e652059ef754a214670d2fe642af504a6fcc01380d33bd22854278c6"
SEEDS = tuple(range(42, 52))
CAPACITANCE_SCALES = tuple(float(x) for x in np.geomspace(0.5, 2.0, 10))
LEAK_SCALES = tuple(float(x) for x in np.geomspace(0.5, 2.0, 10))
TAUS_MS = tuple(float(x) for x in np.geomspace(3, 30, 10))
DT_MS = 0.1
DURATION_MS = 200.0
TRANSIENT_MS = 0.0
BIN_MS = 1.0
SEARCH_HZ = (5.0, 150.0)


def analysis_configuration(source_configuration):
    cfg = dict(source_configuration)
    for key in (
        "minimum_ac_peak",
        "minimum_ac_prominence",
        "maximum_cycle_interval_cv",
        "cycle_interval_period_ratio_bounds",
    ):
        cfg.pop(key, None)
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
            p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")
        },
    )
    return cfg


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditions():
    return [
        dict(
            condition_id=f"{sweep}{j:02d}",
            sweep=sweep,
            condition_index=j,
            axis_value=value,
            tau_gaba_ms=value if sweep == "tau" else 6.0,
            leak_scale=value if sweep == "leak" else 1.0,
            capacitance_scale=value if sweep == "cap" else 1.0,
            capacitance_e_nf=1.0 * (value if sweep == "cap" else 1.0),
            capacitance_i_nf=0.5 * (value if sweep == "cap" else 1.0),
            leak_e_us=0.05 * (value if sweep == "leak" else 1.0),
            leak_i_us=0.1 * (value if sweep == "leak" else 1.0),
        )
        for sweep, values in (
            ("tau", TAUS_MS),
            ("leak", LEAK_SCALES),
            ("cap", CAPACITANCE_SCALES),
        )
        for j, value in enumerate(values)
    ]


def configuration():
    return dict(
        image_index=0,
        partition="official_mnist_test",
        seeds=SEEDS,
        tau_gaba_ms=TAUS_MS,
        leak_scales=LEAK_SCALES,
        capacitance_scales=CAPACITANCE_SCALES,
        conditions=conditions(),
        dt_ms=DT_MS,
        duration_ms=DURATION_MS,
        transient_ms=TRANSIENT_MS,
        population_bin_ms=BIN_MS,
        frequency_search_hz=SEARCH_HZ,
        training_source=SOURCE,
        training_unit=UNIT,
        checkpoint_role=CHECKPOINT_ROLE,
        checkpoint_sha256=CHECKPOINT_SHA256,
        biological_defaults=dict(
            capacitance_nf=[1.0, 0.5],
            leak_us=[0.05, 0.1],
            resting_mv=-65.0,
            threshold_mv=-50.0,
            reset_mv=-65.0,
            refractory_ms=[1.2, 0.6],
            voltage_grad_dampen=1000.0,
            synapse_tau_ampa_ms=2.0,
            readout_tau_ms=2.0,
            readout_threshold=1.0,
            initial_voltage_mv=-65.0,
            recurrent_delay_ms=DT_MS,
        ),
        biological_default_basis="literal working protocol from exp022/exp041/exp044; paired with exp041's trained 6 ms cell",
        script_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        helper_sha256=sha256(REPO / "experiments/helpers/ping.py"),
    )


def network(tau_ms, leak_scale=1.0, capacitance_scale=1.0):
    from snnlab import lang
    from experiments.helpers.ping import build_ping
    from snnlab.sim.timing import refractory_steps

    net = lang.Network("exp122_gamma_calibration", dt=DT_MS * lang.ms)
    drive = net.input(
        "image", shape=("time", "batch", 784), signal_type="spikes", unit="spike"
    )
    neurons = [
        lang.COBA_LIF(
            tau_mem=t * capacitance_scale / leak_scale * lang.ms,
            capacitance_nf=c * capacitance_scale,
            leak_us=g * leak_scale,
            resting_mv=-65.0,
            threshold_mv=-50.0,
            reset_mv=-65.0,
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
    net.expose(circuit.e.spikes, name="spk_e")
    net.expose(circuit.i.spikes, name="spk_i")
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
    net.expose(output.spikes, name="spk_out")
    return lang.compile(net, target="tools/snnsim")
