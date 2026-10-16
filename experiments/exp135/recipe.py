"""Paired convolutional COBA and E/I networks for DVS Gesture; authoring only.

Running this module compiles a portable bundle and its native diagrams. It does
not acquire data, simulate activity, train weights or create a Pingstore run.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from snnlab import extensions, lang

SLUG = "exp135"
MODELS = ("coba", "ping")
SHARDS = 2
COMMON_WEIGHTS = (
    "block1_input_to_E.weight",
    "block2_input_to_E.weight",
    "readout_projection.weight",
)
POPULATIONS = {
    "coba": ("block1_E", "block2_E"),
    "ping": ("block1_E", "block1_I", "block2_E", "block2_I"),
}
RASTER_E_CELLS = 256
RASTER_I_CELLS = 64
ACF_MAX_LAG_MS = 200.0
DT_MS = 1.0
CROP_MS = 500.0
PRESENTATION_MS = CROP_MS
COMPILE_MODE = "default"
INPUT_SHAPE = (2, 32, 32)
N_CLASSES = 11
LEARNING_RATE = 1e-3
TRAIN_FRACTION = 0.10
SELECTION_SEED = 135
TRAIN_SEED = 42
EPOCHS = 20
BATCH_SIZE = 54
VALIDATION_PER_CLASS = 1
TRAIN_ARCHIVE_URL = "https://ndownloader.figshare.com/files/38022171"
TRAIN_ARCHIVE_MD5 = "3a8f0d4120a166bac7591f77409cb105"
EXPECTED_TRAIN_SAMPLES = 1077
# Proposed operating point, pending activity and timestep calibration.
BLOCKS = ((16, 4, 32), (32, 8, 16))
FEEDFORWARD_MEAN_US = 0.02
E_TO_I_MEAN_US = 0.05
I_TO_E_MEAN_US = 0.10


def neuron(*, inhibitory=False):
    """Conductance LIF cells with faster inhibitory membrane dynamics."""
    capacitance, leak = (0.5, 0.1) if inhibitory else (1.0, 0.05)
    return lang.COBA_LIF(
        tau_mem=(capacitance / leak) * lang.ms,
        capacitance_nf=capacitance,
        leak_us=leak,
        resting_mv=-65.0,
        reset_mv=-65.0,
        threshold_mv=-50.0,
        initial_voltage_mv=-65.0,
        refractory=(1.0 if inhibitory else 2.0) * lang.ms,
        voltage_grad_dampen=80.0,
    )


def paired_normal(shape, config, *, device, dtype):
    """Parameter-local draws keep common weights identical across architectures."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    value = torch.randn(shape, generator=generator) * config["std"] + config["mean"]
    if config["nonnegative"]:
        value.clamp_(min=0)
    return value.to(device=device, dtype=dtype)


if __name__ != "__main__":
    extensions.register_initializer("exp135.paired_normal/v1", paired_normal)


def initializer(mean, std, seed, *, nonnegative=True):
    return lang.CustomInitializer(
        "exp135.paired_normal/v1",
        mean=mean,
        std=std,
        seed=seed,
        nonnegative=nonnegative,
    )


def work_items():
    return [dict(id=model, gpu_index=index) for index, model in enumerate(MODELS)]


def author_network(model="ping", *, presentation_ms=PRESENTATION_MS, seed=TRAIN_SEED):
    """Compile the uncalibrated proposal, including its native training recipe.

    Both cases have 3x3 feedforward kernels; PING adds local 1x1 reciprocal
    E/I kernels. All conductance weights stay nonnegative. Average pooling carries
    fractional spike activity; it does not create new binary spikes.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model: {model}")
    net = lang.Network(f"exp135_dvs_gesture_{model}", dt=DT_MS * lang.ms)
    source = net.input(
        "events",
        shape=("time", "batch", 2048),
        spatial_shape=INPUT_SHAPE,
        signal_type="spikes",
        unit="spike",
    )
    weights = []
    for index, (n_e, n_i, width) in enumerate(BLOCKS, start=1):
        name = f"block{index}"
        e = net.population(
            f"{name}_E", spatial_shape=(n_e, width, width), neuron=neuron()
        )
        feedforward = net.connect(
            source,
            e.excitatory,
            name=f"{name}_input_to_E",
            connectivity=lang.Conv2d(kernel_size=3, padding=1),
            synapse=lang.AMPA(tau=2 * lang.ms),
            weight=initializer(
                FEEDFORWARD_MEAN_US, FEEDFORWARD_MEAN_US / 5, seed + index
            ),
            constraint=lang.NonNegative(),
            initialization_scaling="direct",
        )
        weights.append(feedforward.weight)
        if model == "ping":
            i = net.population(
                f"{name}_I",
                spatial_shape=(n_i, width, width),
                neuron=neuron(inhibitory=True),
            )
            ei = net.connect(
                e.spikes,
                i.excitatory,
                name=f"{name}_E_to_I",
                connectivity=lang.Conv2d(kernel_size=1),
                synapse=lang.AMPA(tau=1 * lang.ms),
                weight=initializer(
                    E_TO_I_MEAN_US, E_TO_I_MEAN_US / 5, seed + 10 + index
                ),
                constraint=lang.NonNegative(),
                initialization_scaling="direct",
                connection="recurrent",
                delay=DT_MS * lang.ms,
            )
            ie = net.connect(
                i.spikes,
                e.inhibitory,
                name=f"{name}_I_to_E",
                connectivity=lang.Conv2d(kernel_size=1),
                synapse=lang.GABA(tau=6 * lang.ms),
                weight=initializer(
                    I_TO_E_MEAN_US, I_TO_E_MEAN_US / 5, seed + 20 + index
                ),
                constraint=lang.NonNegative(),
                initialization_scaling="direct",
                connection="feedback",
                delay=DT_MS * lang.ms,
            )
            weights.extend((ei.weight, ie.weight))
        source = lang.pool2d(
            e.spikes, kernel_size=2, stride=2, mode="avg", name=f"{name}_avg_pool"
        )

    readout = net.population(
        "readout",
        size=N_CLASSES,
        spiking=False,
        neuron=lang.LeakyIntegrator(tau=20 * lang.ms),
    )
    projection = net.connect(
        source,
        readout.excitatory,
        name="readout_projection",
        synapse=lang.LeakyIntegrator(tau=20 * lang.ms),
        weight=initializer(0.0, 0.1, seed + 30, nonnegative=False),
        initialization_scaling="direct",
    )
    weights.append(projection.weight)
    scores = lang.reduce(
        readout.voltage, operation="mean", over="time", name="mean_readout_voltage"
    )
    net.output("class_scores", scores)
    training = lang.TrainSpec(
        objectives=(lang.CrossEntropy(prediction=scores, target="gesture"),),
        parameter_groups=(
            lang.ParameterGroup(weights, name="all_weights", lr=LEARNING_RATE),
        ),
        optimizer=lang.AdamW(weight_decay=0.0),
        gradient_clip=1.0,
        surrogate=lang.FastSigmoid(slope=1.0),
        presentation_duration=(
            presentation_ms * lang.ms if presentation_ms is not None else None
        ),
    )
    return lang.compile(net, training=training, target="tools/snnsim")


def configuration(*, epochs=EPOCHS, batch_size=BATCH_SIZE, seed=TRAIN_SEED):
    if min(epochs, batch_size) < 1 or seed < 0:
        raise ValueError("epochs/batch size must be positive and seed nonnegative")
    return dict(
        schema="exp135.training/v3",
        models=list(MODELS),
        work_items=work_items(),
        gpu_count=SHARDS,
        paired_initialization="common kernels/readout use identical parameter-local CPU RNG draws",
        sample_selection="lexicographically first selected validation sample; final weights; fixed centred validation crop",
        raster_cells=dict(excitatory=RASTER_E_CELLS, inhibitory=RASTER_I_CELLS),
        train_fraction=TRAIN_FRACTION,
        selection_seed=SELECTION_SEED,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        dt_ms=DT_MS,
        input_shape=list(INPUT_SHAPE),
        n_classes=N_CLASSES,
        learning_rate=LEARNING_RATE,
        source=dict(url=TRAIN_ARCHIVE_URL, md5=TRAIN_ARCHIVE_MD5),
        source_format="Tonic segmented training archive; x,y,p,t with millisecond timestamps",
        expected_source_samples=EXPECTED_TRAIN_SAMPLES,
        validation_per_class=VALIDATION_PER_CLASS,
        split="highest two training-subject IDs reserved for validation; official test archive unused",
        selection="seeded class-balanced round-robin, 10% of source training count",
        crop_ms=CROP_MS,
        compile_mode=COMPILE_MODE,
        crop_seed=TRAIN_SEED,
        validation_crop="fixed centred 500 ms per example",
        presentation="fresh seeded random 500 ms training crop per epoch; fixed centred validation crop",
        readout="mean voltage over 500 ms crop; zero-pad only examples shorter than crop",
        temporal_encoding="binary occupancy; spatial downsampling by integer division by four",
        training_metrics="online pre-update; validation fixed-weight at epoch boundaries",
        checkpoint_selection="minimum validation loss, earliest epoch on ties",
    )


def local_check_configuration(*, full_gestures=False):
    """A bounded real-data CPU diagnostic, separate from the full-gesture trial."""
    cfg = configuration(epochs=1)
    cfg["presentation"] = (
        "complete selected gesture extents; CPU diagnostic"
        if full_gestures
        else "first 250 ms from each first event; bounded CPU diagnostic"
    )
    cfg.pop("crop_ms", None)
    cfg["compile_mode"] = None
    cfg["local_check"] = {
        "train_examples": 2,
        "validation_examples": 2,
        "presentation_ms": None if full_gestures else 250.0,
        "purpose": "one-epoch real-data pipeline check; no performance claim",
    }
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Empty/new bundle directory; defaults to a new temporary directory.",
    )
    args = parser.parse_args(argv)
    output = args.output or Path(tempfile.mkdtemp(prefix=f"{SLUG}-network-"))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--output must name a new or empty directory")
    for model in MODELS:
        bundle = author_network(model)
        bundle.write(output / model, visualise=True)
        print(f"{model.upper()} bundle and diagrams: {(output / model).resolve()}")
    print("Authoring preview only; no data loaded or training performed.")
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.exp135.recipe import main as cli_main

    raise SystemExit(cli_main())


def benchmark_configuration():
    """Compiled, random-crop throughput only; no accuracy claim."""
    return dict(
        schema="exp135.gpu-benchmark/v2",
        models=list(MODELS),
        batches=[1, 2, 4, 8, 16, 32, 54],
        compile_modes=[COMPILE_MODE],
        duration_ms=CROP_MS,
        epochs=4,
        warmup_epochs=1,
        seed=TRAIN_SEED,
        case_timeout_s=300,
        purpose="compiled random-crop throughput and memory",
    )
