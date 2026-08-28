"""Fixed scientific definitions; importing this module performs no execution."""

import snnlang as snn
from experiments.helpers.snnlang_stages import configuration as canonical_configuration
from snnlang import training

SLUG = "exp075"

DT_MS = 0.5

T_MS = 100.0

N_E = 128

N_I = 32

N_INPUT = 784

N_CLASSES = 10

MAX_SAMPLES = 1_000

BATCH_SIZE = 64

EPOCHS = 4

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

SEED = 75

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
    net = snn.Network("mnist_ping_training_demo", dt=DT_MS * snn.ms)
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

    recurrent_ids = {
        "sensory_ping_E_to_I.weight",
        "sensory_ping_I_to_E.weight",
    }
    feedforward_ids = [
        row["id"] for row in net.parameters if row["id"] not in recurrent_ids
    ]
    recipe = snn.TrainSpec(
        objectives=[training.CrossEntropy(prediction=logits, target="digit")],
        parameter_groups=[
            training.ParameterGroup(
                feedforward_ids,
                name="feedforward_trainable",
                lr=LEARNING_RATE,
            ),
            training.ParameterGroup(
                sorted(recurrent_ids),
                name="recurrent_frozen",
                lr=0.0,
                frozen=True,
            ),
        ],
        optimizer=training.AdamW(weight_decay=WEIGHT_DECAY),
        epochs=EPOCHS,
        gradient_clip=1.0,
    )
    return snn.compile(net, training=recipe, target="tools/snnsim")


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
            "SCALE": SCALE,
        },
        {"network": author_bundle().manifest["graph_digest"]},
    )
