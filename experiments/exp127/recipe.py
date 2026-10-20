"""A six-weight SNNLab classifier for the introductory learning replay.

The configuration resolves scientific settings and the authoring functions compile
the network. Data loading, simulation and run storage belong to the execution stages.
"""

from copy import deepcopy

from snnlab import lang

SLUG = "exp127"
CONFIG = {
    "schema": "exp127.learning/v1",
    "seed": 127,
    "device": "cpu",
    "torch_threads": 1,
    "performance": {"compile_mode": None},
    "dt_ms": 1.0,
    "duration_ms": 200,
    "inputs": 3,
    "outputs": 2,
    "train_samples": 128,
    "validation_samples": 64,
    "test_samples": 128,
    "rate_patterns_hz": [[125, 12.5, 31.25], [12.5, 125, 31.25]],
    "rate_jitter": [0.85, 1.15],
    "epochs": 24,
    "trace_updates": 48,
    "trace_window": 8,
    "batch_size": 64,
    "input_encoding": "independent Poisson counts in 1 ms bins, clipped to binary occupancy for SNNLab spike inputs",
    "learning_rate": 0.02,
    "surrogate_slope": 1.0,
    "gradient_clip": 5.0,
    "tau_mem_ms": 20.0,
    "tau_syn_ms": 2.0,
    "capacitance_nf": 0.2,
    "initial_weight_na": 1.05,
    "delay_ms": 1.0,
    "decision_min_spikes": 5,
    "decision_min_lead": 3,
    "selection": "minimum validation cross-entropy, earliest epoch on ties, including initial",
    "readout": "total output spike counts over 200 ms; ties choose class A",
    "replay_selection": "first held-out example of each class, independent of model responses",
}


def configuration():
    """Return independent resolved scientific settings for the learning protocol."""
    return deepcopy(CONFIG)


def author_network(cfg=None):
    cfg = configuration() if cfg is None else cfg
    net = lang.Network("two_class_intro", dt=cfg["dt_ms"] * lang.ms)
    inputs = net.input(
        "inputs",
        shape=("time", "batch", cfg["inputs"]),
        signal_type="spikes",
        unit="spike",
    )
    outputs = net.population(
        "outputs",
        size=cfg["outputs"],
        neuron=lang.CUBA_LIF(
            tau_mem=cfg["tau_mem_ms"] * lang.ms,
            capacitance_nf=cfg["capacitance_nf"],
        ),
    )
    projection = net.connect(
        inputs,
        outputs.excitatory,
        name="input_to_output",
        synapse=lang.ExponentialCurrent(tau=cfg["tau_syn_ms"] * lang.ms),
        weight=lang.Constant(cfg["initial_weight_na"]),
        constraint=lang.NonNegative(),
        delay=cfg["delay_ms"] * lang.ms,
    )
    counts = lang.reduce(
        outputs.spikes, operation="sum", over="time", name="spike_counts"
    )
    net.output("scores", counts)
    net.expose(outputs.spikes, name="output_spikes")
    net.expose(outputs.voltage, name="output_voltage")
    net.expose(projection.current, name="output_current")
    spec = lang.TrainSpec(
        objectives=(lang.CrossEntropy(prediction=counts, target="class"),),
        parameter_groups=(
            lang.ParameterGroup(
                (projection.weight,), name="synapses", lr=cfg["learning_rate"]
            ),
        ),
        optimizer=lang.AdamW(weight_decay=0.0),
        surrogate=lang.FastSigmoid(slope=cfg["surrogate_slope"]),
        gradient_clip=cfg["gradient_clip"],
        presentation_duration=cfg["duration_ms"] * lang.ms,
    )
    return lang.compile(net, training=spec, target=None)
