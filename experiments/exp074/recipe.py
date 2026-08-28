"""Fixed scientific definitions; importing this module performs no execution."""

import snnlang as snn
from experiments.helpers.snnlang_stages import configuration as canonical_configuration

SLUG = "exp074"

DT_MS = 0.1

T_MS = 200.0

N_BATCH = 4

N_INPUT = 784

N_E = 256

N_I = 64

N_CLASSES = 10

INPUT_RATE_HZ = 100.0

SEED = 74

DISPLAY_TRIAL = 0

SCALE = {
    "dt_ms": DT_MS,
    "t_ms": T_MS,
    "n_batch": N_BATCH,
    "n_input": N_INPUT,
    "n_e": N_E,
    "n_i": N_I,
    "input_rate_hz": INPUT_RATE_HZ,
    "seed": SEED,
}


def author_network() -> snn.Bundle:
    """Define the graph in Python; no simulator implementation leaks in here."""
    net = snn.Network("snnlang_ping_demo", dt=DT_MS * snn.ms)
    spikes = net.input(
        "spike_input",
        shape=("time", "batch", N_INPUT),
        signal_type="spikes",
        unit="spike",
    )
    cell = snn.components.ping(
        net,
        name="sensory_ping",
        n_e=N_E,
        n_i=N_I,
        source=spikes,
    )
    logits = snn.readouts.MeanVoltage(
        source=cell.E.spikes,
        classes=N_CLASSES,
        name="classifier",
        tau=2 * snn.ms,
        weight=snn.Normal(5.1, 3.8),
    )
    net.output("class_logits", logits)
    net.expose(cell.E.spikes, cell.I.spikes, name="cell")
    return snn.compile(net, target="tools/snnsim")


def configuration():
    return canonical_configuration(
        {
            "DT_MS": DT_MS,
            "T_MS": T_MS,
            "N_BATCH": N_BATCH,
            "N_INPUT": N_INPUT,
            "N_E": N_E,
            "N_I": N_I,
            "N_CLASSES": N_CLASSES,
            "INPUT_RATE_HZ": INPUT_RATE_HZ,
            "SEED": SEED,
            "DISPLAY_TRIAL": DISPLAY_TRIAL,
            "SCALE": SCALE,
        },
        {"network": author_network().manifest["graph_digest"]},
    )
