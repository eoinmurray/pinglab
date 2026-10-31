"""One matched mean-field pair: one-way excitation, no feedback or training."""

import math

from snnlab import lang
from snnlab import meanfield as mf
from snnlab.meanfield import closures

SLUG = "exp159"
SCHEMA = "exp159.recipe/v1"
SNAPSHOT = "network-initialization/start"


def configuration(*, difference_hz=2.0, coupling_us=0.06, incoming_weights_us=None):
    if not math.isfinite(difference_hz) or abs(difference_hz) >= 50:
        raise ValueError("input difference must leave both drives positive")
    if not math.isfinite(coupling_us) or not 0 < coupling_us <= 0.12:
        raise ValueError("coupling must be positive and at most 0.12 uS")
    weights = dict(incoming_weights_us or {})
    if any(not math.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError("incoming weights must be positive and finite")
    return dict(
        schema=SCHEMA,
        protocol="oneway-pair",
        duration_ms=1500.0,
        difference_hz=float(difference_hz),
        coupling_us=float(coupling_us),
        network=dict(
            dt_ms=0.1,
            network_seed=85,
            input_channels=1024,
            mean_input_hz=25.0,
            input_rates_hz=dict(
                drive_A=25 + difference_hz / 2, drive_B=25 - difference_hz / 2
            ),
            base_coupling_us=0.12,
            onset_ms=500.0,
            resting_mv=-65.0,
            threshold_mv=-50.0,
            reset_mv=-65.0,
            populations=dict(
                E=dict(size=80, capacitance_nf=1.0, leak_us=0.05, refractory_ms=3.0),
                I=dict(size=20, capacitance_nf=0.5, leak_us=0.1, refractory_ms=1.5),
            ),
            sensory=dict(mean_us=0.2, std_us=0.03, gain=9.2, tau_ms=2.0),
            local=dict(
                E_to_I=dict(mean_us=0.75, std_us=0.07500000000000001, tau_ms=1.0),
                I_to_E=dict(mean_us=1.5, std_us=0.15000000000000002, tau_ms=9.0),
                E_to_E=dict(total_us=0.4, tau_ms=2.0),
                I_to_I=dict(total_us=0.8, tau_ms=6.0),
            ),
            cross=dict(tau_ms=0.5, delay_ms=2.0, zero_fraction=0.9),
        ),
        incoming_weights_us=weights,
        reduction_projection_scales={f"PING_{c}_input_to_E": 9.2 for c in ("A", "B")},
        weight_interpretation="sum(native initialized W)/N_target; sensory gain applied once during reduction",
        closure=dict(
            name="snnlab.meanfield.closures.RateConductance",
            sigma_mv=4.0,
            tau_rate_ms={f"PING_{c}_{p}": 5.0 for c in ("A", "B") for p in ("E", "I")},
            quadrature_nodes=48,
            external_filters="dynamic",
            delays="explicit",
            modulation="translate",
        ),
        initial=dict(rate_hz=1.0, conductance_us=0.0, history="constant"),
        solver=dict(method="DOP853", rtol=1e-8, atol=1e-10, max_step_ms=0.1),
        sample_interval_ms=0.1,
        phase=dict(
            window_ms=[800.0, 1500.0],
            min_peak_distance_ms=15.0,
            prominence_fraction=0.1,
            min_amplitude_hz=0.5,
            min_peaks=5,
            frequency_band_hz=[30.0, 80.0],
            max_interval_cv=0.2,
            max_abs_drift_hz=0.25,
            min_concentration=0.95,
            max_phase_span_cycles=1.0,
        ),
        timing=dict(
            window_ms=[1000.0, 1450.0],
            low_inhibition_fraction=0.25,
            input_rise_fraction=0.2,
            preceding_ms=20.0,
            output_search_ms=20.0,
            inhibition_search_ms=15.0,
        ),
        closeup_cycles=4,
        omitted_effects=[
            "input covariance and finite-population fluctuations",
            "within-population weight heterogeneity",
            "voltage-dependent driving force and shunting",
        ],
    )


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


def author_network(cfg):
    n = cfg["network"]
    net = lang.Network("exp159_oneway_ctc", dt=n["dt_ms"] * lang.ms)
    drives = {
        c: net.input(
            f"drive_{c}",
            shape=("time", "batch", n["input_channels"]),
            signal_type="spikes",
            unit="spike",
        )
        for c in ("A", "B")
    }
    cells = {c: add_circuit(net, c, drives[c], n) for c in ("A", "B")}
    for p in ("E", "I"):
        net.connect(
            cells["A"]["E"].spikes,
            cells["B"][p].excitatory,
            name=f"PING_A_E_to_PING_B_{p}_K_E{p}",
            synapse=lang.AMPA(tau=n["cross"]["tau_ms"] * lang.ms),
            weight=lang.LowerClampedNormal(
                n["base_coupling_us"],
                0,
                initial_zero_fraction=n["cross"]["zero_fraction"],
                zeroing="exact_k",
            ),
            constraint=lang.NonNegative(),
            connection="feedback",
            delay=n["cross"]["delay_ms"] * lang.ms,
            modulation=lang.StepGain(at=n["onset_ms"] * lang.ms, before=0, after=1),
        )
    for c in ("A", "B"):
        for p in ("E", "I"):
            net.expose(cells[c][p].spikes, name=f"{c}_{p}")
    return lang.compile(net, target=None)


def input_rates(source, difference_hz):
    return {
        "drive_A": source["mean_input_hz"] + difference_hz / 2,
        "drive_B": source["mean_input_hz"] - difference_hz / 2,
    }


def build_model(bundle, cfg, *, difference_hz=2.0, coupling_us=0.0):
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
