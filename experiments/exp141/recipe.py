"""Define the two digit classifiers and their shared scientific settings.

The COBA mode is a feed-forward conductance-based leaky integrate-and-fire
(COBA-LIF) network: pixel spikes drive an excitatory hidden population, which
then drives ten spiking readout neurons. PING adds an inhibitory population
that receives excitation from the hidden layer and sends inhibition back.
Both modes learn the input and readout weights; PING's feedback weights stay
fixed. Readout spike counts act as classification logits, with one neuron per
digit. Cross-entropy is the only objective; there are no spike penalties.

This module authors a SNNLab graph and training specification; it does not
load data or execute training. configuration() returns a fresh dictionary
so compute, analysis and presentation can record and compare the same
resolved settings. author_network() compiles a bundle for the chosen mode;
compute.py supplies the dataset bindings, execution device and random seed."""

from dataclasses import asdict

from snnlab import lang
from snnlab.sim import (
    CheckpointSelection,
    Performance,
    SelectionMetric,
    TrainingMetrics,
)


def configuration():
    # Keep all scientific defaults here; every call creates fresh mutable values.
    return dict(
        experiment="exp141",
        schema="exp141.recipe/v9",
        models=["coba", "ping"],
        seed=42,
        # Optimization and presentation length are shared by both modes.
        epochs=30,
        batch_size=32,
        # Native device-dependent optimizations, including CUDA mixed precision.
        # Record every requested policy; SNNLab metrics retain the resolved paths.
        performance=asdict(Performance()),
        # Retain the baseline and test the validation-selected weights once.
        training_metrics=TrainingMetrics().to_dict(),
        checkpoint_selection=CheckpointSelection(
            metric=SelectionMetric("loss")
        ).to_dict(),
        lr=0.0004,
        gradient_clip=1.0,
        dt_ms=0.2,
        duration_ms=200.0,
        # Maximum rate for a fully bright pixel; darker pixels scale this rate.
        input_rate_hz=25.0,
        # One input per pixel and one readout neuron per digit.
        n_input=64,
        n_hidden=128,
        n_inhibitory=32,
        n_classes=10,
        validation_fraction=0.1,
        test_fraction=0.1,
        # Replay original dataset index zero with identical input spikes in both modes.
        sample_index=0,
        sample_encoder_seed=0,
        # Membrane voltages are in mV; capacitance/leak determine C/g_L in ms.
        resting_mv=-65.0,
        threshold_mv=-50.0,
        reset_mv=-65.0,
        capacitance_e_nf=1.0,
        capacitance_i_nf=0.5,
        leak_e_us=0.05,
        leak_i_us=0.10,
        refractory_e_ms=1.2,
        refractory_i_ms=0.6,
        tau_ampa_ms=2.0,
        tau_gaba_ms=6.0,
        # These weight statistics are divided by fan-in when connections are built.
        input_weight_mean=0.9,
        input_weight_std=0.09,
        readout_weight_mean=1.0,
        readout_weight_std=0.1,
        # Copied from exp022's canonical fixed-feedback settings:
        # E→I: max(0, Normal(1, 0.1)) / N_E;
        # I→E: max(0, Normal(2, 0.2)) / N_I.
        # Native SNNLab draws matrices at this experiment's population sizes.
        feedback_weight_mean=1.0,
        feedback_weight_std=0.1,
        inhibitory_strength=2.0,
        voltage_grad_dampen=1000.0,
    )


def author_network(mode, cfg):
    if mode not in cfg["models"]:
        raise ValueError(f"unknown mode: {mode}")
    # Shared feed-forward classifier.
    net = lang.Network("exp141", dt=cfg["dt_ms"] * lang.ms)
    drive = net.input(
        "drive",
        shape=("time", "batch", cfg["n_input"]),
        signal_type="spikes",
        unit="spike",
    )

    # Hidden E and readout use the same excitatory-cell membrane settings.
    # The optional I population selects its own capacitance, leak and refractory time.
    def population(name, size, *, kind="e"):
        capacitance = cfg[f"capacitance_{kind}_nf"]
        leak = cfg[f"leak_{kind}_us"]
        return net.population(
            name,
            size=size,
            neuron=lang.COBA_LIF(
                # nF / microSiemens gives milliseconds for the passive time constant.
                tau_mem=capacitance / leak * lang.ms,
                capacitance_nf=capacitance,
                leak_us=leak,
                resting_mv=cfg["resting_mv"],
                threshold_mv=cfg["threshold_mv"],
                reset_mv=cfg["reset_mv"],
                initial_voltage_mv=cfg["resting_mv"],
                refractory=cfg[f"refractory_{kind}_ms"] * lang.ms,
                voltage_grad_dampen=cfg["voltage_grad_dampen"],
            ),
        )

    excitatory = population("E", cfg["n_hidden"])
    readout = population("readout", cfg["n_classes"])
    ampa = lang.AMPA(tau=cfg["tau_ampa_ms"] * lang.ms)
    gaba = lang.GABA(tau=cfg["tau_gaba_ms"] * lang.ms)

    def connect(
        source, target, *, name, mean, std, fan_in, synapse=ampa, recurrent=False
    ):
        """Initialize nonnegative conductances, normalized by source population size."""
        net.connect(
            source,
            target,
            name=name,
            synapse=synapse,
            # Normalize once here; direct scaling prevents another fan-in rescaling.
            weight=lang.Normal(mean / fan_in, std / fan_in),
            constraint=lang.NonNegative(),
            initialization_scaling="direct",
            # Feedback projections explicitly carry a one-timestep recurrent delay.
            connection="recurrent" if recurrent else "feedforward",
            delay=cfg["dt_ms"] * lang.ms if recurrent else None,
        )

    connect(
        drive,
        excitatory.excitatory,
        name="input_to_E",
        mean=cfg["input_weight_mean"],
        std=cfg["input_weight_std"],
        fan_in=cfg["n_input"],
    )
    connect(
        excitatory.spikes,
        readout.excitatory,
        name="readout_projection",
        mean=cfg["readout_weight_mean"],
        std=cfg["readout_weight_std"],
        fan_in=cfg["n_hidden"],
    )
    # Only the two feed-forward matrices are optimized in either mode.
    groups = [
        lang.ParameterGroup(
            ["input_to_E.weight", "readout_projection.weight"],
            name="learned",
            lr=cfg["lr"],
        )
    ]

    # PING adds a fixed excitatory–inhibitory feedback loop.
    if mode == "ping":
        inhibitory = population("I", cfg["n_inhibitory"], kind="i")
        connect(
            excitatory.spikes,
            inhibitory.excitatory,
            name="E_to_I",
            mean=cfg["feedback_weight_mean"],
            std=cfg["feedback_weight_std"],
            fan_in=cfg["n_hidden"],
            recurrent=True,
        )
        connect(
            inhibitory.spikes,
            excitatory.inhibitory,
            name="I_to_E",
            mean=cfg["feedback_weight_mean"] * cfg["inhibitory_strength"],
            std=cfg["feedback_weight_std"] * cfg["inhibitory_strength"],
            fan_in=cfg["n_inhibitory"],
            synapse=gaba,
            recurrent=True,
        )
        # The loop affects activity but its synaptic weights are held constant.
        groups.append(
            lang.ParameterGroup(
                ["E_to_I.weight", "I_to_E.weight"],
                name="feedback",
                lr=0.0,
                frozen=True,
            )
        )

    # Each digit's logit is its readout neuron's total spike count.
    counts = lang.reduce(
        readout.spikes, operation="sum", over="time", name="spike_counts"
    )
    net.output("class_scores", counts)
    # Cross-entropy consumes raw counts as logits; FastSigmoid supplies the
    # backward-pass approximation for the nondifferentiable spike threshold.
    training = lang.TrainSpec(
        objectives=(lang.CrossEntropy(prediction=counts, target="digit"),),
        parameter_groups=groups,
        optimizer=lang.AdamW(lr=cfg["lr"], weight_decay=0.0),
        surrogate=lang.FastSigmoid(slope=1.0),
        gradient_clip=cfg["gradient_clip"],
        epochs=cfg["epochs"],
        presentation_duration=cfg["duration_ms"] * lang.ms,
    )
    return lang.compile(net, training=training, target="tools/snnsim")
