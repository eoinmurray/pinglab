"""Fixed S1/S2-to-R PING circuit with sustained sender-specific neuronal cues."""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

from snnlab import lang as snn

SLUG = "exp136"
DT_MS = 0.1
N_INPUT, N_E, N_I, N_CUE = 1024, 80, 20, 32
N_CUE_INPUT = 128
BACKGROUND_HZ = {"S1": 33.0, "S2": 33.0, "R": 35.0}
NETWORK_SEED = 85
INPUT_SEEDS = {"S1": 8501, "S2": 8503, "R": 8502, "cue": 8504}
CUE_ACTIVE_INPUT_HZ = 25.0
INPUT_GAIN = 9.2
FORWARD_US = 0.045
CUE_TO_E_US = 0.06
PARAMETER_SOURCE = "exp136-r059-compute"
SELECTION_EVIDENCE = "exp136-r064-present"
PARAMETER_UNIT = "k0450-q0600-noise0-execution"


def configuration():
    """Exactly two continuous trials at the selected fixed parameters."""
    return dict(
        schema="exp136.fixed/v1",
        dt_ms=DT_MS,
        duration_ms=4000.0,
        segment_ms=1000.0,
        frequency_window_ms=1000,
        conditions=["coupled", "zero"],
        background_hz=BACKGROUND_HZ.copy(),
        network_seed=NETWORK_SEED,
        input_seeds=INPUT_SEEDS.copy(),
        input_channels=N_INPUT,
        cue_input_channels=N_CUE_INPUT,
        cue_drive_hz=CUE_ACTIVE_INPUT_HZ,
        cue_segments=[["S1", 1000.0, 2000.0], ["S2", 2000.0, 3000.0]],
        forward_strength_us=FORWARD_US,
        cue_to_e_strength_us=CUE_TO_E_US,
        parameter_unit=PARAMETER_UNIT,
        analysis_windows=[
            ["baseline", 100, 1000],
            ["cue_S1", 1100, 2000],
            ["cue_S2", 2100, 3000],
            ["recovery", 3100, 4000],
        ],
        analysis=dict(
            smoothing_ms=1.0,
            peak_distance_ms=8.0,
            prominence_fraction=0.1,
            max_phase_gap_ms=50.0,
            min_volleys=10,
            gamma_hz=[30.0, 80.0],
            max_iei_cv=0.2,
            plv_window_ms=200.0,
            plv_step_ms=10.0,
            plv_min_coverage=0.9,
        ),
        limitation="one selected network/input realisation; no stimulus decoding",
    )


def attention_configuration():
    """Fixed signal-routing assay; no parameter selection on the test trials."""
    cfg = configuration()
    trials = [
        dict(
            split=split,
            trial=i,
            noise_offset=offset + 1000 * i,
            signal_seed=136000 + offset + 10 * i,
        )
        for split, offset in (("train", 100000), ("test", 200000))
        for i in range(3)
    ]
    cfg.update(
        schema="exp136.attention/v1",
        conditions=["coupled", "zero", "jittered"],
        signal=dict(
            bin_ms=10.0,
            smoothing_ms=50.0,
            amplitude_hz=1.5,
            construction="independent smoothed Gaussian sequences, tanh bounded, centred within each trial",
        ),
        decoder=dict(
            bin_ms=20.0,
            sample_ms=10.0,
            lags_ms=[0, 10, 20, 30, 40],
            target_delay_ms=20.0,
            ridge_alpha=10.0,
            fitting="one pooled cue-independent decoder per condition and sender; training-only scaling",
            features="population E/I rates in trailing windows",
        ),
        jitter=dict(block_ms=50.0, seed=136500),
        cue_orders=[["S1", "S2"], ["S2", "S1"]],
        units=[
            dict(unit=f"signals-{i // 2}", trials=trials[i : i + 2])
            for i in range(0, len(trials), 2)
        ],
        baseline_presentation="exp136-r067-present",
        limitation="three held-out stimulus/noise pairs; one fixed network; no behavioural readout",
    )
    return cfg


SEARCH_AXES = {
    "amplitude_hz": (1.5, 15.0),
    "smoothing_ms": (25.0, 250.0),
    "forward_strength_us": (0.015, 0.135),
    "cue_to_e_strength_us": (0.015, 0.18),
}


def routing_candidates(level, centre=None):
    """Fixed, space-filling coarse sample and a narrower local fine sample."""
    from scipy.stats import qmc

    baseline = dict(
        amplitude_hz=1.5,
        smoothing_ms=50.0,
        forward_strength_us=FORWARD_US,
        cue_to_e_strength_us=CUE_TO_E_US,
    )
    if level not in ("coarse", "fine") or (level == "fine" and centre is None):
        raise ValueError("fine search requires an explicit coarse-selected centre")
    bounds = dict(SEARCH_AXES)
    if level == "fine":
        bounds = {
            k: (max(lo, centre[k] * 0.65), min(hi, centre[k] * 1.35))
            for k, (lo, hi) in bounds.items()
        }
    sample = qmc.LatinHypercube(d=4, seed=136701 if level == "coarse" else 136702)
    points = qmc.scale(
        sample.random(16),
        [v[0] for v in bounds.values()],
        [v[1] for v in bounds.values()],
    )
    definitions = [baseline if level == "coarse" else dict(centre)]
    definitions += [dict(zip(bounds, map(float, row))) for row in points]
    return [dict(v, candidate=f"{level}-{i:02d}") for i, v in enumerate(definitions)]


def routing_configuration(candidate, *, confirmation=False):
    """Keep Fig. 1–2 wiring and mean drives; fit readouts, never neural weights."""
    cfg = attention_configuration()
    cfg["signal"].update({k: candidate[k] for k in ("amplitude_hz", "smoothing_ms")})
    cfg.update(
        {k: candidate[k] for k in ("forward_strength_us", "cue_to_e_strength_us")}
    )
    cfg["decoder"].update(bin_ms=50.0, lags_ms=[0, 25, 50], target_delay_ms=50.0)
    offsets = (
        (("train", 400000), ("test", 500000))
        if confirmation
        else (("train", 300000), ("test", 310000))
    )
    trials = [
        dict(
            split=split,
            trial=i,
            noise_offset=offset + i * 1000,
            signal_seed=136000 + offset + 10 * i,
        )
        for split, offset in offsets
        for i in range(3 if confirmation else 1)
    ]
    cfg["units"] = [
        dict(
            unit=f"{candidate['candidate']}-signals-{i // 2}", trials=trials[i : i + 2]
        )
        for i in range(0, len(trials), 2)
    ]
    cfg["conditions"] = ["coupled", "zero", "jittered"] if confirmation else ["coupled"]
    cfg["selection"] = dict(
        local_min_r2=0.1,
        split="fresh test" if confirmation else "validation",
        rule="sender readability first; then positive cued R scores and minimum same-signal cue gain",
    )
    cfg["limitation"] = (
        "one wiring realisation; validation-selected parameters; descriptive fresh-trial confirmation"
    )
    return cfg


@dataclass(frozen=True)
class Circuit:
    E: snn.Population
    I: snn.Population


def neuron(*, inhibitory=False):
    """Use the local conductance-LIF cell settings of the reference PING pair."""
    cap, leak, refractory = (0.5, 0.1, 1.5) if inhibitory else (1.0, 0.05, 3.0)
    return snn.COBA_LIF(
        tau_mem=(cap / leak) * snn.ms,
        capacitance_nf=cap,
        leak_us=leak,
        resting_mv=-65.0,
        threshold_mv=-50.0,
        reset_mv=-65.0,
        refractory=refractory * snn.ms,
        initial_voltage_mv=-65.0,
        voltage_grad_dampen=80.0,
    )


def spike_input(net, name, size):
    return net.input(
        name, shape=("time", "batch", size), signal_type="spikes", unit="spike"
    )


def add_ping(net, name):
    """Define each PING circuit locally, without cross-experiment imports."""
    with net.group(name):
        drive = spike_input(net, f"{name}_sensory_drive", N_INPUT)
        e = net.population(f"{name}_E", size=N_E, neuron=neuron())
        i = net.population(f"{name}_I", size=N_I, neuron=neuron(inhibitory=True))
        net.connect(
            drive,
            e.excitatory,
            name=f"{name}_input_to_E",
            synapse=snn.AMPA(tau=2 * snn.ms),
            weight=snn.LowerClampedNormal(0.2 * INPUT_GAIN, 0.03 * INPUT_GAIN),
            constraint=snn.NonNegative(),
        )
        net.connect(
            e.spikes,
            i.excitatory,
            name=f"{name}_E_to_I",
            synapse=snn.AMPA(tau=1 * snn.ms),
            weight=snn.LowerClampedNormal(0.5, 0.05),
            constraint=snn.NonNegative(),
            connection="recurrent",
            delay=DT_MS * snn.ms,
        )
        net.connect(
            i.spikes,
            e.inhibitory,
            name=f"{name}_I_to_E",
            synapse=snn.GABA(tau=9 * snn.ms),
            weight=snn.LowerClampedNormal(1.0, 0.1),
            constraint=snn.NonNegative(),
            connection="recurrent",
            delay=DT_MS * snn.ms,
        )
    return Circuit(e, i)


def author_network():
    """Fixed one-way circuit; compute restores the selected native matrices."""
    net = snn.Network("exp136_sustained_cue_ping", dt=DT_MS * snn.ms)
    circuits = {name: add_ping(net, name) for name in ("S1", "S2", "R")}
    for name in ("S1", "S2"):
        sender = circuits[name]
        for pop in ("E", "I"):
            net.connect(
                sender.E.spikes,
                getattr(circuits["R"], pop).excitatory,
                name=f"{name}_E_to_R_{pop}",
                synapse=snn.AMPA(tau=2 * snn.ms),
                weight=snn.LowerClampedNormal(
                    FORWARD_US, 0.0, initial_zero_fraction=0.9, zeroing="exact_k"
                ),
                constraint=snn.NonNegative(),
                delay=2 * snn.ms,
            )
        with net.group(f"cue_{name}"):
            cue_drive = spike_input(net, f"cue_{name}_drive", N_CUE_INPUT)
            cue = net.population(f"cue_{name}_E", size=N_CUE, neuron=neuron())
            net.connect(
                cue_drive,
                cue.excitatory,
                name=f"cue_{name}_input_to_E",
                synapse=snn.AMPA(tau=2 * snn.ms),
                weight=snn.LowerClampedNormal(0.2 * INPUT_GAIN, 0.03 * INPUT_GAIN),
                constraint=snn.NonNegative(),
            )
        net.connect(
            cue.spikes,
            sender.E.excitatory,
            name=f"cue_{name}_E_to_{name}_E",
            synapse=snn.AMPA(tau=2 * snn.ms),
            weight=snn.LowerClampedNormal(
                CUE_TO_E_US, 0.0, initial_zero_fraction=0.75, zeroing="exact_k"
            ),
            constraint=snn.NonNegative(),
            delay=2 * snn.ms,
        )
        net.output(f"cue_{name}_spikes", cue.spikes)
    for name, circuit in circuits.items():
        for pop in ("E", "I"):
            net.output(f"{name}_{pop}_spikes", getattr(circuit, pop).spikes)
    return snn.compile(net, target=None)


def write_network_bundle(bundle, output):
    """Retain the complete bundle; presentation renders one circuit view."""
    output = Path(output)
    bundle.write(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="New/empty bundle directory; default: fresh temp dir.",
    )
    args = parser.parse_args(argv)
    output = args.output or Path(tempfile.mkdtemp(prefix=f"{SLUG}-network-"))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output must name a new or empty directory")
    write_network_bundle(author_network(), output)
    print(f"Fixed network bundle: {output.resolve()}")
    print("Authoring preview only; no simulation or scientific run performed.")
    return 0




def readout_candidates():
    """Common target times/samples prevent window-dependent evaluation changes."""
    return [
        dict(
            readout=f"{mode}-{width:03d}-a{alpha:04d}",
            feature_mode=mode,
            bin_ms=float(width),
            ridge_alpha=float(alpha),
            lags_ms=[0, 25, 50],
            sample_ms=10.0,
            target_delay_ms=50.0,
            observation_start_ms=250.0,
            cue_exclusion_ms=350.0,
        )
        for mode in ("population", "individual")
        for width in (25, 50, 100, 200)
        for alpha in (1, 10, 100, 1000)
    ]


def receiver_candidates(network):
    """Target the receiver while holding the successful sensory encoding fixed."""
    return [
        dict(
            network,
            candidate=f"receiver-r{drive:02d}-k{index}",
            receiver_background_hz=float(drive),
            forward_strength_us=float(coupling),
        )
        for drive in (29, 32, 35)
        for index, coupling in enumerate((0.04, network["forward_strength_us"], 0.12))
    ]


def receiver_configuration(network, readout, *, confirmation=False):
    cfg = routing_configuration(network, confirmation=True)
    cfg["background_hz"]["R"] = network.get("receiver_background_hz", 35.0)
    cfg["decoder"].update(readout)
    # Earlier exposed tests are development validation; new offsets stay unused until confirmation.
    offsets = (
        (("train", 600000), ("test", 700000))
        if confirmation
        else (("train", 400000), ("validation", 500000))
    )
    trials = [
        dict(
            split=split,
            trial=i,
            noise_offset=offset + 1000 * i,
            signal_seed=136000 + offset + 10 * i,
        )
        for split, offset in offsets
        for i in range(3)
    ]
    cfg["units"] = [
        dict(unit=f"{network['candidate']}-signals-{i // 2}", trials=trials[i : i + 2])
        for i in range(0, len(trials), 2)
    ]
    cfg["conditions"] = ["coupled", "zero", "jittered"] if confirmation else ["coupled"]
    cfg["selection"] = dict(
        split="fresh test" if confirmation else "development validation",
        local_min_r2=0.1,
        rule="readable senders and positive cue benefit, then minimum cued receiver R2",
    )
    return cfg


if __name__ == "__main__":
    raise SystemExit(main())
