"""Self-contained delayed rate-conductance circuits with reciprocal coupling.

Own the neuronal reference graph, seeded weight-initialization settings, unfitted
closure and matched grid. Loading, initialization and integration belong to the
execution stage; no scientific inputs come from another experiment.
"""

import copy
import math
from itertools import product

from snnlab import lang
from snnlab import meanfield as mf
from snnlab.meanfield import closures

SLUG = "exp153"
SCHEMA = "exp153.recipe/v7"
SNAPSHOT = "network-initialization/start"


def configuration(
    *,
    difference_hz=-4.0,
    coupling_us=0.09,
    incoming_weights_us=None,
    onset_only=False,
    tau_rate_e_ms=5.0,
    tau_rate_i_ms=5.0,
    sensory_gain=9.2,
    recurrent_ee_us=0.4,
    recurrent_ii_us=0.8,
    local_feedback_scale=1.5,
    cross_delay_ms=2.0,
    cross_tau_ms=0.5,
):
    """Resolve locally authored physics, numerical settings and initialized weights."""
    differences = [float(value) for value in range(-10, 11, 2)]
    couplings = [value / 100 for value in range(13)]
    if not math.isfinite(difference_hz) or difference_hz not in differences:
        raise ValueError("onset diagnostic must select an existing input difference")
    if (
        not math.isfinite(coupling_us)
        or coupling_us not in couplings
        or coupling_us <= 0
    ):
        raise ValueError("onset diagnostic must select a positive grid coupling")
    weights = dict(incoming_weights_us or {})
    if any(
        not math.isfinite(v) or v <= 0
        for v in (
            tau_rate_e_ms,
            tau_rate_i_ms,
            sensory_gain,
            recurrent_ee_us,
            recurrent_ii_us,
            local_feedback_scale,
            cross_delay_ms,
            cross_tau_ms,
        )
    ):
        raise ValueError(
            "rate relaxation times, delays, gains and recurrent weights must be finite and positive"
        )
    if any(not math.isfinite(value) or value <= 0 for value in weights.values()):
        raise ValueError("mean incoming effective weights must be finite and positive")
    network = dict(
        dt_ms=0.1,
        network_seed=85,
        input_channels=1024,
        mean_input_hz=25.0,
        input_rates_hz={
            "drive_A": [25.0 + value / 2 for value in differences],
            "drive_B": [25.0 - value / 2 for value in differences],
        },
        base_coupling_us=0.12,
        onset_ms=500.0,
        settling_ms=300.0,
        resting_mv=-65.0,
        threshold_mv=-50.0,
        reset_mv=-65.0,
        populations=dict(
            E=dict(size=80, capacitance_nf=1.0, leak_us=0.05, refractory_ms=3.0),
            I=dict(size=20, capacitance_nf=0.5, leak_us=0.1, refractory_ms=1.5),
        ),
        sensory=dict(mean_us=0.2, std_us=0.03, gain=float(sensory_gain), tau_ms=2.0),
        local=dict(
            E_to_I=dict(
                mean_us=0.5 * local_feedback_scale,
                std_us=0.05 * local_feedback_scale,
                tau_ms=1.0,
            ),
            I_to_E=dict(
                mean_us=1.0 * local_feedback_scale,
                std_us=0.1 * local_feedback_scale,
                tau_ms=9.0,
            ),
            E_to_E=dict(total_us=float(recurrent_ee_us), tau_ms=2.0),
            I_to_I=dict(total_us=float(recurrent_ii_us), tau_ms=6.0),
        ),
        cross=dict(
            tau_ms=float(cross_tau_ms),
            delay_ms=float(cross_delay_ms),
            zero_fraction=0.9,
        ),
        input_gain_application="applied once to initialized sensory sums during mean-field reduction",
    )
    return dict(
        schema=SCHEMA,
        protocol="onset-only" if onset_only else "matched-grid",
        network=network,
        duration_ms=1500.0 if onset_only else 5000.0,
        sweep=dict(differences_hz=differences, couplings_us=couplings),
        example=dict(
            difference_hz=float(difference_hz),
            coupling_us=float(coupling_us),
            display_window_ms=[0.0, 1500.0],
            phase_window_ms=[800.0, 1500.0],
            frequency_windows_ms=dict(first=[0.0, 500.0], last=[1000.0, 1500.0]),
            rationale="Selected condition from this experiment's own mean-field grid",
        ),
        incoming_weights_us=weights,
        reduction_projection_scales={
            f"PING_{circuit}_input_to_E": network["sensory"]["gain"]
            for circuit in ("A", "B")
        },
        native_snapshot=SNAPSHOT + "/snapshot.json",
        weight_interpretation="sum(native initialized W)/N_target, including zero edges and excluded autapses; sensory execution gain applied once during reduction",
        closure=dict(
            name="snnlab.meanfield.closures.RateConductance",
            sigma_mv=4.0,
            tau_rate_ms={
                f"PING_{circuit}_{population}": (
                    tau_rate_e_ms if population == "E" else tau_rate_i_ms
                )
                for circuit in ("A", "B")
                for population in ("E", "I")
            },
            quadrature_nodes=48,
            provenance="Unfitted rate-conductance closure; equal E/I relaxation times across circuits, independently specified for this closure",
            external_filters="dynamic",
            delays="explicit",
            modulation="translate",
        ),
        omitted_effects=[
            "input covariance and finite-population fluctuations",
            "within-population weight heterogeneity",
            "voltage-dependent driving force and shunting",
        ],
        initial=dict(rate_hz=1.0, conductance_us=0.0, history="constant"),
        solver=dict(method="DOP853", rtol=1e-8, atol=1e-10, max_step_ms=0.1),
        sample_interval_ms=0.5,
        gate=dict(
            windows_ms=[[800.0, 2900.0], [2900.0, 5000.0]],
            min_amplitude_hz=0.5,
            min_relative_amplitude=0.01,
            min_late_to_early_amplitude=0.5,
            min_peaks=20,
            max_interval_cv=0.2,
            frequency_band_hz=None,
            interpretation="Sustained rhythms at every grid input; no imposed spiking gamma band",
        ),
        phase=dict(
            start_ms=800.0,
            min_peak_distance_ms=15.0,
            prominence_fraction=0.1,
            min_amplitude_hz=0.5,
            min_peaks=20,
            max_interval_cv=0.2,
            frequency_band_hz=None,
            max_abs_drift_hz=0.25,
            min_concentration=0.95,
            max_phase_span_cycles=1.0,
            smoothing="none; deterministic population rates, not binned spikes",
        ),
    )


def analysis_configuration(compute_configuration):
    """Resolve onset analysis choices alongside the immutable compute settings."""
    cfg = copy.deepcopy(compute_configuration)
    start, end = cfg["example"]["phase_window_ms"]
    cfg["onset_analysis"] = dict(
        min_peaks=5,
        gate_windows_ms=[[start, (start + end) / 2], [(start + end) / 2, end]],
    )
    return cfg


def susceptibility_configuration(compute_configuration):
    cfg = copy.deepcopy(compute_configuration)
    cfg["susceptibility"] = dict(
        sender="A",
        receiver="B",
        window_ms=[1200.0, 1400.0],
        definition="analytic dPhi_B_E/dI evaluated at saved effective current; Hz/nA",
        arrivals="sender E-rate peak plus saved cross-pathway transmission delay",
        interpretation="instantaneous local susceptibility, not a finite-window response",
    )
    return cfg


def conditions(cfg):
    """Stable drive-major grid order, including one zero-coupling calibration per drive."""
    if cfg["protocol"] == "onset-only":
        return [
            dict(
                label="onset",
                difference_hz=cfg["example"]["difference_hz"],
                coupling_us=cfg["example"]["coupling_us"],
            )
        ]
    return [
        dict(
            label=f"condition-{index:03d}",
            difference_hz=difference,
            coupling_us=coupling,
        )
        for index, (difference, coupling) in enumerate(
            product(cfg["sweep"]["differences_hz"], cfg["sweep"]["couplings_us"])
        )
    ]


def add_circuit(net, circuit, drive, settings):
    """Author the complete local E/I reference and its private sensory projection."""
    cells = {}
    with net.group(f"PING_{circuit}"):
        for population, physical in settings["populations"].items():
            cells[population] = net.population(
                f"PING_{circuit}_{population}",
                size=physical["size"],
                neuron=lang.COBA_LIF(
                    tau_mem=physical["capacitance_nf"] / physical["leak_us"] * lang.ms,
                    capacitance_nf=physical["capacitance_nf"],
                    leak_us=physical["leak_us"],
                    resting_mv=settings["resting_mv"],
                    threshold_mv=settings["threshold_mv"],
                    reset_mv=settings["reset_mv"],
                    initial_voltage_mv=settings["resting_mv"],
                    refractory=physical["refractory_ms"] * lang.ms,
                    voltage_grad_dampen=80.0,
                ),
            )
        sensory = settings["sensory"]
        net.connect(
            drive,
            cells["E"].excitatory,
            name=f"PING_{circuit}_input_to_E",
            synapse=lang.AMPA(tau=sensory["tau_ms"] * lang.ms),
            weight=lang.LowerClampedNormal(
                sensory["mean_us"],
                sensory["std_us"],
            ),
            constraint=lang.NonNegative(),
        )
        for pathway, physical in settings["local"].items():
            source, target = pathway.split("_to_")
            inhibitory = source == "I"
            fixed = "total_us" in physical
            weight = (
                lang.Constant(
                    physical["total_us"] / settings["populations"][source]["size"]
                )
                if fixed
                else lang.LowerClampedNormal(physical["mean_us"], physical["std_us"])
            )
            net.connect(
                cells[source].spikes,
                getattr(cells[target], "inhibitory" if inhibitory else "excitatory"),
                name=f"PING_{circuit}_{pathway}",
                synapse=(lang.GABA if inhibitory else lang.AMPA)(
                    tau=physical["tau_ms"] * lang.ms
                ),
                weight=weight,
                initialization_scaling="direct" if fixed else "fan_in_normalized",
                constraint=lang.NonNegative(),
                connection="recurrent",
                delay=settings["dt_ms"] * lang.ms,
            )
    return cells


def author_network(cfg=None):
    """Compile this experiment's own fourteen-pathway reference circuit."""
    settings = (configuration() if cfg is None else cfg)["network"]
    net = lang.Network("exp153_reciprocal_ping", dt=settings["dt_ms"] * lang.ms)
    drives = {
        c: net.input(
            f"drive_{c}",
            shape=("time", "batch", settings["input_channels"]),
            signal_type="spikes",
            unit="spike",
        )
        for c in ("A", "B")
    }
    circuits = {c: add_circuit(net, c, drives[c], settings) for c in ("A", "B")}
    cross = settings["cross"]
    for source, target in (("A", "B"), ("B", "A")):
        for population in ("E", "I"):
            net.connect(
                circuits[source]["E"].spikes,
                circuits[target][population].excitatory,
                name=f"PING_{source}_E_to_PING_{target}_{population}_K_E{population}",
                synapse=lang.AMPA(tau=cross["tau_ms"] * lang.ms),
                weight=lang.LowerClampedNormal(
                    settings["base_coupling_us"],
                    0.0,
                    initial_zero_fraction=cross["zero_fraction"],
                    zeroing="exact_k",
                ),
                constraint=lang.NonNegative(),
                connection="feedback",
                delay=cross["delay_ms"] * lang.ms,
                modulation=lang.StepGain(
                    at=settings["onset_ms"] * lang.ms, before=0.0, after=1.0
                ),
            )
    for circuit in ("A", "B"):
        for population in ("E", "I"):
            net.expose(
                circuits[circuit][population].spikes, name=f"{circuit}_{population}"
            )
    return lang.compile(net, target=None)


def input_rates(source, difference_hz):
    """Resolve this experiment's two private stationary sensory rates."""
    return {
        "drive_A": source["mean_input_hz"] + difference_hz / 2,
        "drive_B": source["mean_input_hz"] - difference_hz / 2,
    }


def build_model(bundle, cfg, *, difference_hz=0.0, coupling_us=0.0):
    """Scale cross-circuit sums from the locally initialized reference exactly once."""
    source = cfg["network"]
    if (
        not math.isfinite(difference_hz)
        or abs(difference_hz) >= 2 * source["mean_input_hz"]
    ):
        raise ValueError("input difference must leave both drives positive")
    if not 0 <= coupling_us <= source["base_coupling_us"]:
        raise ValueError("coupling outside the recorded grid range")
    weights = {
        name: value * coupling_us / source["base_coupling_us"]
        if "_K_E" in name
        else value * cfg.get("reduction_projection_scales", {}).get(name, 1.0)
        for name, value in cfg["incoming_weights_us"].items()
    }
    closure = cfg["closure"]
    return mf.reduce(
        bundle,
        closure=closures.RateConductance(
            gain=closures.NoisyLIF(
                sigma=closure["sigma_mv"],
                quadrature_nodes=closure["quadrature_nodes"],
            ),
            tau_rate=closure["tau_rate_ms"],
        ),
        weights=mf.MeanIncomingWeights(
            weights, provenance=cfg["weight_interpretation"]
        ),
        inputs={
            name: mf.StationaryRate(rate)
            for name, rate in input_rates(source, difference_hz).items()
        },
        external_filters=closure["external_filters"],
        delays=closure["delays"],
        modulation=closure["modulation"],
    )


def initial_state(model, cfg):
    return {
        name: cfg["initial"]["rate_hz" if name.endswith(".rate") else "conductance_us"]
        for name in model.states
    }
