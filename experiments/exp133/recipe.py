"""Committed Arnold-tongue grid and native SNNLab authoring; no execution.

The configuration resolves scientific settings and the authoring functions compile
the network. Data loading, simulation and run storage belong to the execution stages.
"""

from dataclasses import dataclass

import numpy as np
from snnlab import lang as snn

SLUG = "exp133"
DT_MS = 0.1
T_MS = 5000.0
COUPLING_ONSET_MS = 500.0
SETTLING_MS = 300.0
NETWORK_SEED = 85
INPUT_SEEDS = (8501, 8502)
N_INPUT, N_E, N_I = 1024, 80, 20
MEAN_INPUT_HZ = 25.0
INPUT_DIFFERENCES_HZ = np.arange(-10.0, 10.1, 2.0)
INPUT_GAIN = 9.2
K_VALUES_US = np.arange(13, dtype=float) / 100
BASE_K_US = 0.12
MAX_DRIFT_HZ = 0.25
MIN_CONCENTRATION = 0.95
MAX_PHASE_SPAN_CYCLES = 1.0


@dataclass(frozen=True)
class Circuit:
    E: snn.Population
    I: snn.Population


def add_ping(net, name, drive):
    """Retain the working 80-E/20-I conductance-LIF operating point."""
    with net.group(name):
        cells = {}
        for pop, count, cap, leak, refractory in (
            ("E", N_E, 1.0, 0.05, 3.0),
            ("I", N_I, 0.5, 0.1, 1.5),
        ):
            cells[pop] = net.population(
                f"{name}_{pop}",
                size=count,
                neuron=snn.COBA_LIF(
                    tau_mem=(cap / leak) * snn.ms,
                    capacitance_nf=cap,
                    leak_us=leak,
                    resting_mv=-65.0,
                    threshold_mv=-50.0,
                    reset_mv=-65.0,
                    refractory=refractory * snn.ms,
                    initial_voltage_mv=-65.0,
                    voltage_grad_dampen=80.0,
                ),
            )
        net.connect(
            drive,
            cells["E"].excitatory,
            name=f"{name}_input_to_E",
            synapse=snn.AMPA(tau=2 * snn.ms),
            weight=snn.LowerClampedNormal(0.2, 0.03),
            constraint=snn.NonNegative(),
        )
        net.connect(
            cells["E"].spikes,
            cells["I"].excitatory,
            name=f"{name}_E_to_I",
            synapse=snn.AMPA(tau=1 * snn.ms),
            weight=snn.LowerClampedNormal(0.5, 0.05),
            constraint=snn.NonNegative(),
            connection="recurrent",
            delay=DT_MS * snn.ms,
        )
        net.connect(
            cells["I"].spikes,
            cells["E"].inhibitory,
            name=f"{name}_I_to_E",
            synapse=snn.GABA(tau=9 * snn.ms),
            weight=snn.LowerClampedNormal(1.0, 0.1),
            constraint=snn.NonNegative(),
            connection="recurrent",
            delay=DT_MS * snn.ms,
        )
    return Circuit(**cells)


def cross_ids():
    return tuple(
        f"PING_{source}_E_to_PING_{target}_{pop}_K_E{pop}"
        for source, target in (("A", "B"), ("B", "A"))
        for pop in ("E", "I")
    )


def author_network():
    """One portable bundle; native request scales set the coupling strengths."""
    net = snn.Network("exp133_arnold_tongue", dt=DT_MS * snn.ms)
    drives = {
        name: net.input(
            f"drive_{name}",
            shape=("time", "batch", N_INPUT),
            signal_type="spikes",
            unit="spike",
        )
        for name in ("A", "B")
    }
    circuits = {
        name: add_ping(net, f"PING_{name}", drives[name]) for name in ("A", "B")
    }
    for source, target in (("A", "B"), ("B", "A")):
        for pop in ("E", "I"):
            net.connect(
                circuits[source].E.spikes,
                getattr(circuits[target], pop).excitatory,
                name=f"PING_{source}_E_to_PING_{target}_{pop}_K_E{pop}",
                synapse=snn.AMPA(tau=2 * snn.ms),
                weight=snn.LowerClampedNormal(
                    BASE_K_US, 0.0, initial_zero_fraction=0.9, zeroing="exact_k"
                ),
                constraint=snn.NonNegative(),
                connection="feedback",
                delay=2 * snn.ms,
                modulation=snn.StepGain(
                    at=COUPLING_ONSET_MS * snn.ms, before=0.0, after=1.0
                ),
            )
    for name in ("A", "B"):
        for pop in ("E", "I"):
            net.expose(getattr(circuits[name], pop).spikes, name=f"{name}_{pop}")
    return snn.compile(net, target=None)


def coupling_scales(k):
    if not np.isfinite(k) or not 0 <= k <= BASE_K_US:
        raise ValueError("coupling must be within the committed nominal range")
    return {
        **{name: float(k / BASE_K_US) for name in cross_ids()},
        **{f"PING_{name}_input_to_E": INPUT_GAIN for name in ("A", "B")},
    }


def label(k):
    return f"coupling-{round(k * 1000):03d}"


def rates():
    return {
        "drive_A": MEAN_INPUT_HZ + INPUT_DIFFERENCES_HZ / 2,
        "drive_B": MEAN_INPUT_HZ - INPUT_DIFFERENCES_HZ / 2,
    }


def make_inputs(steps=None):
    """Common random numbers across detuning; exactly shared across coupling."""
    steps = round(T_MS / DT_MS) if steps is None else steps
    result = {}
    for (name, rate), seed in zip(rates().items(), INPUT_SEEDS, strict=True):
        uniform = np.random.default_rng(seed).random(
            (steps, 1, N_INPUT), dtype=np.float32
        )
        result[name] = (uniform < rate[None, :, None] * DT_MS / 1000).astype(np.uint8)
    return result


def configuration():
    bundle = author_network()
    return dict(
        schema="exp133.recipe/v1",
        dt_ms=DT_MS,
        duration_ms=T_MS,
        onset_ms=COUPLING_ONSET_MS,
        settling_ms=SETTLING_MS,
        network_seed=NETWORK_SEED,
        input_seeds=list(INPUT_SEEDS),
        input_differences_hz=INPUT_DIFFERENCES_HZ.tolist(),
        input_rates_hz={key: value.tolist() for key, value in rates().items()},
        mean_input_hz=MEAN_INPUT_HZ,
        input_channels=N_INPUT,
        input_projection_gain=INPUT_GAIN,
        coupling_us=K_VALUES_US.tolist(),
        base_coupling_us=BASE_K_US,
        bundle_digest=bundle.manifest["graph_digest"],
        batch_order="ascending signed input difference; each batch is a condition, not a replicate",
        input_pairing="common underlying uniforms across detuning; exact reuse across coupling",
        scale_protocol="native projection_scales: equal cross-network coupling and equal fixed input gain",
        classification=dict(
            max_abs_drift_hz=MAX_DRIFT_HZ,
            min_concentration=MIN_CONCENTRATION,
            max_phase_span_cycles=MAX_PHASE_SPAN_CYCLES,
        ),
        rate_smoothing_ms=1.0,
        volley_distance_ms=15.0,
        volley_prominence_fraction=0.1,
        min_volleys=20,
        max_iei_cv=0.2,
        gamma_band_hz=[30.0, 80.0],
        analysis="entire suffix after settling; common phase-valid window per pair",
    )
