"""Author three unidirectionally connected E/I rate-conductance circuits.

A compiled COBA graph defines the native mean-field reduction; no spiking
simulation is claimed. Settings are synthetic operating-point choices, rather
than fitted estimates. This module has no execution or storage side effects.
"""

import math

import numpy as np
from snnlab import lang
from snnlab import meanfield as mf
from snnlab.meanfield import closures

SLUG = "exp157"
CIRCUITS = ("A", "B", "C")


def configuration(
    *,
    b_hold_drive_hz=4.45,
    b_final_drive_hz=5.0,
    tau_rate_ms=None,
    self_recurrence_scale=0.5,
):
    if any(
        not math.isfinite(value) or value <= 2.5
        for value in (b_hold_drive_hz, b_final_drive_hz)
    ):
        raise ValueError("B drive levels must exceed A's fixed 2.5 Hz/source")
    if b_final_drive_hz < b_hold_drive_hz:
        raise ValueError("B's final ramp must be ascending")
    if not math.isfinite(self_recurrence_scale) or not 0 < self_recurrence_scale <= 1:
        raise ValueError("self recurrence scale must be finite and in (0, 1]")
    times = (
        dict(A=dict(E=24.0, I=6.0), B=dict(E=16.0, I=4.0), C=dict(E=20.0, I=5.0))
        if tau_rate_ms is None
        else tau_rate_ms
    )
    if set(times) != set(CIRCUITS) or any(
        set(times[c]) != {"E", "I"} for c in CIRCUITS
    ):
        raise ValueError("rate relaxation must specify E and I for A, B and C")
    if any(
        not math.isfinite(value) or value <= 0
        for pair in times.values()
        for value in pair.values()
    ):
        raise ValueError("rate relaxation times must be finite and positive")
    return dict(
        schema="exp157.recipe/v6",
        protocol="nine-500ms-windows-e-only",
        circuits=list(CIRCUITS),
        sizes=dict(E=80, I=20, X=80),
        neuron=dict(
            capacitance_nf=0.25,
            leak_us=0.0167,
            resting_mv=-70.0,
            reset_mv=-60.0,
            threshold_mv=-50.0,
            excitatory_reversal_mv=0.0,
            inhibitory_reversal_mv=-80.0,
        ),
        refractory_ms=dict(E=2.0, I=1.0),
        local_incoming_us=dict(
            E_E=0.3857 * self_recurrence_scale,
            E_I=0.7709,
            I_E=1.6070,
            I_I=1.6073 * self_recurrence_scale,
        ),
        self_recurrence_scale=float(self_recurrence_scale),
        calibration_policy="I relaxation calibrated to closer autonomous frequencies; drives, weights and E relaxation inherited without firing-rate recalibration",
        input_incoming_us=0.8677,
        a_drive_hz=2.5,
        c_drive_hz=2.3,
        b_hold_drive_hz=float(b_hold_drive_hz),
        b_final_drive_hz=float(b_final_drive_hz),
        cross_incoming_us=dict(E=0.1, I=0.0),
        synapse_ms=dict(E=5.0, I=5.0, X=5.0, cross=5.0),
        tau_rate_ms={c: dict(pair) for c, pair in times.items()},
        frequency_calibration=dict(
            targets_hz=dict(A=34.0, B=38.0, C=36.0),
            i_tau_brackets_ms=dict(A=[5.0, 8.0], B=[3.0, 4.0], C=[4.0, 6.0]),
            duration_ms=2500.0,
            measure_start_ms=1500.0,
            tolerance_hz=0.25,
            tau_tolerance_ms=0.00001,
            interpretation="Uncoupled constant inputs at each circuit's inherited hold drive. Keep E relaxation, synapses, weights and static gain fixed; tune only I relaxation to 34/36/38 Hz for A/C/B. Finite-window calibration, not a guarantee across drive levels.",
        ),
        voltage_noise_mv=4.0,
        quadrature_nodes=32,
        window_ms=500.0,
        duration_ms=4500.0,
        sample_interval_ms=0.25,
        initial_rate_hz=0.0,
        target=dict(
            b_to_a_e_rate=2.0,
            ramp_end_ms=4000.0,
            terminal_average_ms=100.0,
            relative_tolerance=0.005,
            definition="Mean excitatory population firing rate in 3900–4000 ms, at the end of the second B ramp; window 9 holds these inputs",
        ),
        solver=dict(method="LSODA", rtol=1e-11, atol=1e-13, max_step_ms=0.125),
        analysis=dict(
            gamma_band_hz=[25.0, 100.0],
            min_amplitude_hz=0.5,
            min_relative_amplitude=0.01,
            min_late_early_ratio=0.7,
            min_peaks=8,
            max_interval_cv=0.15,
            hold_discard_ms=100.0,
            frequency_window_ms=200.0,
            frequency_step_ms=25.0,
        ),
        interpretation="Synthetic deterministic mean-field protocol; one continuous initial-value trajectory with nine 500 ms windows. No pre-run warm-up or resets. A/B feed forward to C only. Ramp windows are nonstationary; locking is tested in holds.",
    )


def windows(cfg):
    labels = (
        "All inputs off",
        "C drive ramps",
        "C drive holds",
        "A drive ramps",
        "A drive holds",
        "B drive ramps",
        "B drive holds",
        "B input ramps again",
        "All inputs hold",
    )
    return [
        dict(
            number=i + 1,
            label=label,
            start_ms=i * cfg["window_ms"],
            stop_ms=(i + 1) * cfg["window_ms"],
            ramp=i in (1, 3, 5, 7),
        )
        for i, label in enumerate(labels)
    ]


def drive_controls(cfg):
    times = np.arange(10) * cfg["window_ms"]
    a, b, c, end = (
        cfg["a_drive_hz"],
        cfg["b_hold_drive_hz"],
        cfg["c_drive_hz"],
        cfg["b_final_drive_hz"],
    )
    return {
        f"X_{circuit}.rate": mf.PiecewiseLinear(times=times, values=values)
        for circuit, values in {
            "A": [0, 0, 0, 0, a, a, a, a, a, a],
            "B": [0, 0, 0, 0, 0, 0, b, b, end, end],
            "C": [0, 0, c, c, c, c, c, c, c, c],
        }.items()
    }


def incoming_weights(cfg):
    weights = {}
    for circuit in CIRCUITS:
        for pathway, value in cfg["local_incoming_us"].items():
            source, target = pathway.split("_")
            weights[f"{circuit}_{source}_to_{circuit}_{target}"] = value
        weights[f"X_{circuit}_to_{circuit}_E"] = cfg["input_incoming_us"]
    for sender in ("A", "B"):
        for target, value in cfg["cross_incoming_us"].items():
            if value:
                weights[f"{sender}_E_to_C_{target}"] = value
    return weights


def author_network(cfg):
    net = lang.Network(SLUG, dt=0.05 * lang.ms)
    drives = {
        c: net.input(
            f"X_{c}",
            shape=("time", "batch", cfg["sizes"]["X"]),
            signal_type="spikes",
            unit="spike",
        )
        for c in CIRCUITS
    }
    cells = {}
    for circuit in CIRCUITS:
        for kind in ("E", "I"):
            name = f"{circuit}_{kind}"
            cells[name] = net.population(
                name,
                size=cfg["sizes"][kind],
                neuron=lang.COBA_LIF(
                    **cfg["neuron"],
                    tau_mem=cfg["neuron"]["capacitance_nf"]
                    / cfg["neuron"]["leak_us"]
                    * lang.ms,
                    refractory=cfg["refractory_ms"][kind] * lang.ms,
                ),
            )
    for name, incoming in incoming_weights(cfg).items():
        source, target = name.split("_to_")
        external = source.startswith("X_")
        inhibitory = not external and source.endswith("_I")
        source_kind = "X" if external else source[-1]
        cross = not external and source.split("_")[0] != target.split("_")[0]
        source_size = cfg["sizes"][source_kind]
        net.connect(
            drives[source[2:]] if external else cells[source].spikes,
            cells[target].inhibitory if inhibitory else cells[target].excitatory,
            name=name,
            synapse=(lang.GABA if inhibitory else lang.AMPA)(
                tau=cfg["synapse_ms"]["cross" if cross else source_kind] * lang.ms
            ),
            weight=lang.Constant(incoming / (source_size - int(source == target))),
            initialization_scaling="direct",
            allow_autapses=False,
            connection="feedforward" if external or cross else "recurrent",
        )
    for name, cell in cells.items():
        net.output(name + "_spikes", cell.spikes)
    return lang.compile(net)


def build_model(cfg, bundle):
    return mf.reduce(
        bundle,
        closure=closures.RateConductance(
            gain=closures.NoisyLIF(
                sigma=cfg["voltage_noise_mv"], quadrature_nodes=cfg["quadrature_nodes"]
            ),
            tau_rate={
                f"{c}_{p}": cfg["tau_rate_ms"][c][p]
                for c in CIRCUITS
                for p in ("E", "I")
            },
        ),
        weights=mf.MeanIncomingWeights(
            incoming_weights(cfg),
            provenance="Explicit synthetic effective sum(W)/N_target in µS, including excluded autapses; no sampling",
        ),
        inputs={f"X_{c}": mf.StationaryRate(0.0) for c in CIRCUITS},
        external_filters="dynamic",
        delays="omit",
    )


def initial_state(model, cfg):
    return {
        name: cfg["initial_rate_hz"] if name.endswith(".rate") else 0.0
        for name in model.states
    }
