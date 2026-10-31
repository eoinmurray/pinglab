"""Symmetric mean-field competition with a redirected constant-rate cue."""

import copy
import math

from snnlab import lang
from snnlab import meanfield as mf
from snnlab.meanfield import closures

SLUG = "exp161"
SCHEMA = "exp161.recipe/v1"
SNAPSHOT = "network-initialization/start"


def configuration(
    *,
    reference_network,
    incoming_weights_us,
    coupling_us=0.04,
    cue_hz=6.0,
    background_hz=28.0,
):
    if not all(
        math.isfinite(v) and v > 0 for v in (coupling_us, cue_hz, background_hz)
    ):
        raise ValueError("coupling, cue and background must be finite and positive")
    n = reference_network
    return dict(
        schema=SCHEMA,
        network=dict(
            populations=copy.deepcopy(n["populations"]),
            local={k: dict(tau_ms=v["tau_ms"]) for k, v in n["local"].items()},
            dt_ms=0.1,
            input_channels=1024,
            resting_mv=-65.0,
            threshold_mv=-50.0,
            reset_mv=-65.0,
            sensory=dict(gain=9.2, tau_ms=2.0, delay_ms=0.1),
            cross=dict(tau_ms=0.5, delay_ms=2.0),
        ),
        incoming_weights_us=copy.deepcopy(incoming_weights_us),
        input_rates_hz=dict(
            X_A=24.0,
            X_B=24.0,
            X_C=float(background_hz),
            U_A=float(cue_hz),
            U_B=float(cue_hz),
        ),
        coupling_us=float(coupling_us),
        cue_hz=float(cue_hz),
        sensory_hz=24.0,
        background_hz=float(background_hz),
        duration_ms=2400.0,
        switch_ms=1200.0,
        dt_ms=0.1,
        windows_ms=[[650.0, 1100.0], [1850.0, 2300.0]],
        closure=dict(sigma_mv=4.0, tau_rate_ms=5.0, quadrature_nodes=48),
        seed=85,
        low_inhibition_fraction=0.25,
        phase=dict(
            min_peak_distance_ms=15.0,
            min_prominence_hz=0.05,
            prominence_fraction=0.1,
            min_plv=0.95,
            max_abs_drift_cycles_s=0.25,
            max_cued_phase_span_cycles=0.3,
            min_uncued_phase_span_cycles=0.5,
            min_receiver_amplitude_hz=0.5,
        ),
        timing=dict(
            reference_start_ms=500.0, closeup_before_ms=100.0, closeup_after_ms=250.0
        ),
        initial=dict(rate_hz=1.0, conductance_us=0.0, history="constant"),
        solver=dict(method="DOP853", rtol=1e-8, atol=1e-10, max_step=0.1),
    )


def author_network(cfg):
    n = cfg["network"]
    net = lang.Network("exp161_competitive_entrainment", dt=n["dt_ms"] * lang.ms)
    source_weights = cfg["incoming_weights_us"]
    weights = {}
    cells = {}
    inputs = {}

    def connect(src, target, name, total, tau, delay=0.1, modulation=None):
        # Native direct matrices have the requested incoming sum at every cell.
        source_size = (
            n["input_channels"]
            if name.startswith(("sensory_", "cue_"))
            else n["populations"]["E" if "_E_to_" in name else "I"]["size"]
        )
        if name.endswith(("_E_to_E", "_I_to_I")):
            source_size -= 1
        net.connect(
            src,
            target,
            name=name,
            synapse=(lang.GABA if "_I_to_" in name else lang.AMPA)(tau=tau * lang.ms),
            weight=lang.Constant(total / source_size),
            initialization_scaling="direct",
            constraint=lang.NonNegative(),
            delay=delay * lang.ms,
            **(dict(modulation=modulation) if modulation is not None else {}),
        )
        weights[name] = total

    for c in "ABC":
        inputs[c] = net.input(
            "X_" + c,
            shape=("time", "batch", n["input_channels"]),
            signal_type="spikes",
            unit="spike",
        )
        cells[c] = {}
        for p, phys in n["populations"].items():
            cells[c][p] = net.population(
                c + "_" + p,
                size=phys["size"],
                neuron=lang.COBA_LIF(
                    tau_mem=phys["capacitance_nf"] / phys["leak_us"] * lang.ms,
                    capacitance_nf=phys["capacitance_nf"],
                    leak_us=phys["leak_us"],
                    resting_mv=n["resting_mv"],
                    threshold_mv=n["threshold_mv"],
                    reset_mv=n["reset_mv"],
                    initial_voltage_mv=n["resting_mv"],
                    refractory=phys["refractory_ms"] * lang.ms,
                    voltage_grad_dampen=80.0,
                ),
            )
        connect(
            inputs[c],
            cells[c]["E"].excitatory,
            "sensory_" + c,
            source_weights["PING_B_input_to_E"] * n["sensory"]["gain"],
            n["sensory"]["tau_ms"],
        )
        for path, phys in n["local"].items():
            s, t = path.split("_to_")
            connect(
                cells[c][s].spikes,
                getattr(cells[c][t], "inhibitory" if s == "I" else "excitatory"),
                c + "_" + path,
                source_weights["PING_B_" + path],
                phys["tau_ms"],
            )
        for p in "EI":
            net.expose(cells[c][p].spikes, name=c + "_" + p + "_spikes")
    for c in "AB":
        cue = net.input(
            "U_" + c,
            shape=("time", "batch", n["input_channels"]),
            signal_type="spikes",
            unit="spike",
        )
        connect(
            cue,
            cells[c]["E"].excitatory,
            "cue_" + c,
            source_weights["PING_B_input_to_E"] * n["sensory"]["gain"],
            n["sensory"]["tau_ms"],
            modulation=lang.StepGain(
                at=cfg["switch_ms"] * lang.ms,
                before=1 if c == "A" else 0,
                after=0 if c == "A" else 1,
            ),
        )
        for p in "EI":
            connect(
                cells[c]["E"].spikes,
                cells["C"][p].excitatory,
                c + "_E_to_C_" + p,
                cfg["coupling_us"],
                n["cross"]["tau_ms"],
                delay=n["cross"]["delay_ms"],
            )
    return lang.compile(net, target=None), weights


def build_model(bundle, cfg, weights):
    return mf.reduce(
        bundle,
        closure=closures.RateConductance(
            gain=closures.NoisyLIF(
                sigma=cfg["closure"]["sigma_mv"],
                quadrature_nodes=cfg["closure"]["quadrature_nodes"],
            ),
            tau_rate=cfg["closure"]["tau_rate_ms"],
        ),
        weights=mf.MeanIncomingWeights(
            weights,
            provenance="native symmetric constant matrices; local sums from pinned exp159 B; sensory gain applied once",
        ),
        inputs={
            name: mf.StationaryRate(rate)
            for name, rate in cfg["input_rates_hz"].items()
        },
        external_filters="dynamic",
        delays="explicit",
        modulation="translate",
    )
