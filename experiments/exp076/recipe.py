"""Fixed scientific definitions; importing this module performs no execution."""

import snnlang as snn
from experiments.helpers.snnlang_stages import configuration as canonical_configuration
from snnlang import training

SLUG = "exp076"

DT_MS = 0.5

T_MS = 40.0

N_E = 64

N_I = 16

N_INPUT = 784

N_CLASSES = 10

MAX_SAMPLES = 160

BATCH_SIZE = 32

EPOCHS = 2

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

SEED = 76

W_IN = (0.2, 0.03)

W_EI = (0.5, 0.05)

W_IE = (1.0, 0.1)

TAU_GABA_MS = 9.0

READOUT_INIT = (5.1, 3.8)

READOUT_TAU_MS = 2.0

SCALE = {
    "dt_ms": DT_MS,
    "t_ms": T_MS,
    "n_e": N_E,
    "n_i": N_I,
    "max_samples": MAX_SAMPLES,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "seed": SEED,
}


def author_bundle() -> snn.Bundle:
    net = snn.Network("mnist_ping_replay_gate", dt=DT_MS * snn.ms)
    image_spikes = net.input(
        "image_spikes",
        shape=("time", "batch", N_INPUT),
        signal_type="spikes",
        unit="spike",
    )
    cell = snn.components.ping(
        net,
        name="sensory_ping",
        n_e=N_E,
        n_i=N_I,
        source=image_spikes,
        tau_gaba=TAU_GABA_MS * snn.ms,
    )
    logits = snn.readouts.MeanVoltage(
        source=cell.E.spikes,
        classes=N_CLASSES,
        name="classifier",
        tau=READOUT_TAU_MS * snn.ms,
        weight=snn.Normal(*READOUT_INIT),
    )
    net.output("class_logits", logits)
    net.expose(cell.E.spikes, cell.I.spikes, name="cell")

    recurrent_ids = {
        "sensory_ping_E_to_I.weight",
        "sensory_ping_I_to_E.weight",
    }
    trainable_ids = [
        row["id"] for row in net.parameters if row["id"] not in recurrent_ids
    ]
    recipe = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=logits, target="digit")],
        parameter_groups=[
            training.ParameterGroup(
                trainable_ids,
                name="input_and_readout_trainable",
                lr=LEARNING_RATE,
            ),
            training.ParameterGroup(
                sorted(recurrent_ids),
                name="recurrent_ei_frozen",
                lr=0.0,
                frozen=True,
            ),
        ],
        optimizer=training.AdamW(weight_decay=WEIGHT_DECAY),
        epochs=EPOCHS,
        gradient_clip=1.0,
    )
    return snn.compile(net, training=recipe, target="tools/snnsim")


PARITY_TEST = "tools/snnsim/tests/test_bundle.py::test_bundle_and_legacy_one_step_training_are_exactly_equivalent"


def configuration():
    return canonical_configuration(
        {
            "DT_MS": DT_MS,
            "T_MS": T_MS,
            "N_E": N_E,
            "N_I": N_I,
            "N_INPUT": N_INPUT,
            "N_CLASSES": N_CLASSES,
            "MAX_SAMPLES": MAX_SAMPLES,
            "BATCH_SIZE": BATCH_SIZE,
            "EPOCHS": EPOCHS,
            "LEARNING_RATE": LEARNING_RATE,
            "WEIGHT_DECAY": WEIGHT_DECAY,
            "SEED": SEED,
            "W_IN": W_IN,
            "W_EI": W_EI,
            "W_IE": W_IE,
            "TAU_GABA_MS": TAU_GABA_MS,
            "READOUT_INIT": READOUT_INIT,
            "READOUT_TAU_MS": READOUT_TAU_MS,
            "SCALE": SCALE,
        },
        {"network": author_bundle().manifest["graph_digest"]},
    )
