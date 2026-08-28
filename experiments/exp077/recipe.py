"""Fixed scientific definitions; importing this module performs no execution."""

import snnlang as snn
import torch
from experiments.helpers.snnlang_stages import configuration as canonical_configuration
from snnlang.examples.build_examples import ping_classifier

SLUG = "exp077"

DT_MS = 0.1

STEPS = 300

BATCH = 2

SEED = 77

VARIANTS = (
    ("uncoupled", None),
    ("unidirectional", 1),
    ("reciprocal_zero_additional", 1),
    ("reciprocal_delayed", 5),
)

SCALE = {
    "dt_ms": DT_MS,
    "steps": STEPS,
    "batch": BATCH,
    "seed": SEED,
    "variants": len(VARIANTS),
}


def author_variant(name: str, delay_steps: int | None) -> snn.Bundle:
    net = snn.Network(f"two_ping_{name}", dt=DT_MS * snn.ms)
    drive_a = net.input(
        "drive_a", shape=("time", "batch", 8), signal_type="spikes", unit="spike"
    )
    drive_b = net.input(
        "drive_b", shape=("time", "batch", 6), signal_type="spikes", unit="spike"
    )
    a = snn.components.ping(
        net, name="a", n_e=16, n_i=4, source=drive_a, include_silent_recurrence=True
    )
    b = snn.components.ping(
        net, name="b", n_e=12, n_i=3, source=drive_b, include_silent_recurrence=True
    )
    delay = (delay_steps or 1) * DT_MS * snn.ms
    if name != "uncoupled":
        net.connect(
            a.I.spikes,
            b.E.inhibitory,
            name="a_I_to_b_E",
            synapse=snn.GABA(tau=9 * snn.ms),
            weight=snn.Constant(3.0),
            constraint=snn.NonNegative(),
            connection="feedback",
            delay=delay,
        )
    if name.startswith("reciprocal"):
        net.connect(
            b.I.spikes,
            a.E.inhibitory,
            name="b_I_to_a_E",
            synapse=snn.GABA(tau=9 * snn.ms),
            weight=snn.Constant(3.0),
            constraint=snn.NonNegative(),
            connection="feedback",
            delay=delay,
        )
    net.expose(a.E.spikes, a.I.spikes, b.E.spikes, b.I.spikes, name="population")
    return snn.compile(net, target="tools/snnsim")


def independent_inputs() -> dict[str, torch.Tensor]:
    a = torch.zeros(STEPS, BATCH, 8)
    b = torch.zeros(STEPS, BATCH, 6)
    a[0::10, :, :] = 1.0
    b[5::13, :, :] = 1.0
    a[:, 1] = torch.roll(a[:, 1], 3, dims=0)
    b[:, 1] = torch.roll(b[:, 1], 7, dims=0)
    return {"drive_a": a, "drive_b": b}


DELAY_TESTS = [
    "tools/snnsim/tests/test_execution.py::test_input_delay_pulse_arrives_on_exact_timestep_and_handles_boundary",
    "tools/snnsim/tests/test_execution.py::test_delay_lowering_is_exact_in_steps_and_feedback_is_causal",
]


def configuration():
    return canonical_configuration(
        {
            "DT_MS": DT_MS,
            "STEPS": STEPS,
            "BATCH": BATCH,
            "SEED": SEED,
            "VARIANTS": VARIANTS,
            "SCALE": SCALE,
            "PARITY": PARITY,
            "PARITY_MODEL": PARITY_MODEL,
            "DELAY_TESTS": DELAY_TESTS,
        },
        {
            **{
                name: author_variant(name, delay).manifest["graph_digest"]
                for name, delay in VARIANTS
            },
            "parity": ping_classifier().manifest["graph_digest"],
        },
    )


PARITY = {
    "dt_ms": 0.1,
    "t_ms": 10.0,
    "steps": 100,
    "batch": 8,
    "seed": 17,
    "replay_seed": 999,
    "input_probability": 0.02,
    "warmups": 2,
    "repetitions": 5,
    "compiled_repetitions": 3,
    "compile_steps": 20,
    "compile_batch": 2,
    "gate_percent": 10.0,
}

PARITY_MODEL = {
    "n_input": 784,
    "n_classes": 10,
    "hidden_sizes": [256],
    "w_in": [0.2, 0.03],
    "w_in_initial_zero_fraction": 0.0,
    "w_ei": [0.5, 0.05],
    "w_ie": [1.0, 0.1],
    "readout_mode": "mem-mean",
}
