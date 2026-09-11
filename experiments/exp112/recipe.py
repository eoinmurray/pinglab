"""Self-contained scientific recipe for the exp112 four-condition comparison."""

from __future__ import annotations

from pathlib import Path

SLUG = "exp112"
SCHEMA = "exp112.recipe/v1"

# The four conditions differ only in recurrent E/I-loop engagement and the
# dimensionless voltage-gradient damping divisor.  The remaining values are
# literal copies of the exp022 unpenalised activity-frontier recipe as inspected
# at repository commit be48653b; exp112 never imports or reads exp022.
CASES = (
    {"id": "coba-d1", "architecture": "coba", "ei_strength": 0.0, "v_grad_dampen": 1.0},
    {
        "id": "coba-d1000",
        "architecture": "coba",
        "ei_strength": 0.0,
        "v_grad_dampen": 1000.0,
    },
    {"id": "ping-d1", "architecture": "ping", "ei_strength": 1.0, "v_grad_dampen": 1.0},
    {
        "id": "ping-d1000",
        "architecture": "ping",
        "ei_strength": 1.0,
        "v_grad_dampen": 1000.0,
    },
)

SEED = 42
MNIST_TRAIN_SAMPLES = 60_000
TRAINING_POOL_SAMPLES = 1_200
OPTIMIZER_TRAIN_SAMPLES = 1_080
VALIDATION_SAMPLES = 120
MNIST_TEST_SAMPLES = 10_000
MNIST_SUBSET_SEED = 42
MNIST_SPLIT_SEED = 42
RASTER_DIGIT = 0
RASTER_SAMPLE_WITHIN_CLASS = 0

EPOCHS = 50
BATCH_SIZE = 256
N_INPUT = 784
N_EXCITATORY = 1024
N_INHIBITORY = 256
N_OUTPUT = 10
DT_MS = 0.1
PRESENTATION_MS = 200.0
REFRACTORY_E_MS = 1.2
REFRACTORY_I_MS = 0.6
REFRACTORY_POLICY = "exact"
TAU_AMPA_MS = 2.0
TAU_GABA_MS = 6.0
INPUT_RATE_HZ = 25.0
EI_RATIO = 2.0
W_IN_SUMMED_PARENT_MEAN = 0.9
W_IN_INITIAL_ZERO_FRACTION = 0.95
RECURRENT_INITIAL_ZERO_FRACTION = 0.0
READOUT = "mem-mean"
READOUT_W_INIT_MEAN = 1.12060546875
READOUT_W_INIT_STD = 0.8349609375
SURROGATE_SLOPE = 1.0
LEARNING_RATE = 0.0004
WEIGHT_DECAY = 0.0
GRADIENT_CLIP_NORM = 1.0
DALES_LAW = True
RATE_PENALTY_TARGET_HZ = 0.0
RATE_PENALTY_STRENGTH = 0.0


def case_for_shard(index: int) -> dict:
    if not 0 <= index < len(CASES):
        raise ValueError(f"shard index must be in [0, {len(CASES)})")
    return dict(CASES[index])


def configuration(case: dict) -> dict:
    if case not in CASES:
        raise ValueError(f"unknown exp112 condition: {case!r}")
    return {
        "schema": SCHEMA,
        "condition": dict(case),
        "design": {
            "factors": {
                "architecture": ["coba", "ping"],
                "voltage_gradient_damping_divisor": [1.0, 1000.0],
            },
            "pairing": "one common seed, MNIST subset, validation split, data order and encoding streams",
        },
        "dataset": {
            "name": "mnist",
            "source_train_partition": "official_mnist_train",
            "source_test_partition": "official_mnist_test",
            "training_pool_samples": TRAINING_POOL_SAMPLES,
            "training_fraction": TRAINING_POOL_SAMPLES / MNIST_TRAIN_SAMPLES,
            "subset_selection": "without replacement using numpy RandomState",
            "subset_seed": MNIST_SUBSET_SEED,
            "split": "stratified_90_10_within_selected_training_pool",
            "split_seed": MNIST_SPLIT_SEED,
            "optimizer_train_samples": OPTIMIZER_TRAIN_SAMPLES,
            "validation_samples": VALIDATION_SAMPLES,
            "checkpoint_selection_partition": "validation",
            "official_test_checkpoint_role": "final_epoch",
            "official_test_samples": MNIST_TEST_SAMPLES,
            "official_test_used_during_training": False,
            "illustrative_raster": {
                "partition": "official_mnist_test",
                "digit": RASTER_DIGIT,
                "sample_within_class": RASTER_SAMPLE_WITHIN_CLASS,
                "checkpoint_role": "final_epoch",
            },
        },
        "input": {
            "encoding": "Bernoulli-discretized Poisson from normalized pixels",
            "channels": N_INPUT,
            "maximum_pixel_rate_hz": INPUT_RATE_HZ,
        },
        "topology": {
            "simulator_model": "ping",
            "excitatory_neurons": N_EXCITATORY,
            "inhibitory_neurons": N_INHIBITORY,
            "output_neurons": N_OUTPUT,
            "ei_strength": case["ei_strength"],
            "ei_ratio": EI_RATIO,
            "ei_loop_enabled": case["architecture"] == "ping",
            "recurrent_weights_trainable": False,
        },
        "dynamics": {
            "dt_ms": DT_MS,
            "presentation_duration_ms": PRESENTATION_MS,
            "presentation_steps": round(PRESENTATION_MS / DT_MS),
            "refractory_e_ms": REFRACTORY_E_MS,
            "refractory_i_ms": REFRACTORY_I_MS,
            "refractory_policy": REFRACTORY_POLICY,
            "tau_ampa_ms": TAU_AMPA_MS,
            "tau_gaba_ms": TAU_GABA_MS,
        },
        "initialization": {
            "seed": SEED,
            "w_in_summed_parent_mean": W_IN_SUMMED_PARENT_MEAN,
            "w_in_parent_std": W_IN_SUMMED_PARENT_MEAN * 0.1,
            "w_in_initial_zero_fraction": W_IN_INITIAL_ZERO_FRACTION,
            "recurrent_initial_zero_fraction": RECURRENT_INITIAL_ZERO_FRACTION,
            "readout_distribution": "lower_clamped_normal",
            "readout_w_mean": READOUT_W_INIT_MEAN,
            "readout_w_std": READOUT_W_INIT_STD,
        },
        "training": {
            "optimizer": "adamw",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "voltage_gradient_damping_divisor": case["v_grad_dampen"],
            "surrogate": "fast_sigmoid",
            "surrogate_slope": SURROGATE_SLOPE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "dales_law": DALES_LAW,
            "rate_penalty_target_hz": RATE_PENALTY_TARGET_HZ,
            "rate_penalty_strength": RATE_PENALTY_STRENGTH,
        },
        "readout": {
            "kind": "output LIF with mean pre-reset voltage",
            "mode": READOUT,
            "bias": False,
            "signed_weights": False,
        },
        "parameter_origin": {
            "description": "values transcribed into exp112 from the exp022 unpenalised activity-frontier recipe",
            "repository_commit_inspected": "be48653b",
            "runtime_dependency": False,
        },
    }


def training_args(case: dict, output: Path) -> list[str]:
    """Return one fixed legacy-executor CLI command for a committed condition."""
    configuration(case)
    return [
        "train",
        "--model",
        "ping",
        "--dataset",
        "mnist",
        "--n-hidden",
        str(N_EXCITATORY),
        "--input-rate",
        str(INPUT_RATE_HZ),
        "--max-samples",
        str(TRAINING_POOL_SAMPLES),
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        str(BATCH_SIZE),
        "--t-ms",
        str(PRESENTATION_MS),
        "--dt",
        str(DT_MS),
        "--refractory-e-ms",
        str(REFRACTORY_E_MS),
        "--refractory-i-ms",
        str(REFRACTORY_I_MS),
        "--refractory-policy",
        REFRACTORY_POLICY,
        "--tau-gaba",
        str(TAU_GABA_MS),
        "--seed",
        str(SEED),
        "--ei-strength",
        str(case["ei_strength"]),
        "--ei-ratio",
        str(EI_RATIO),
        "--recurrent-initial-zero-fraction",
        str(RECURRENT_INITIAL_ZERO_FRACTION),
        "--v-grad-dampen",
        str(case["v_grad_dampen"]),
        "--w-in",
        str(W_IN_SUMMED_PARENT_MEAN),
        "--w-in-initial-zero-fraction",
        str(W_IN_INITIAL_ZERO_FRACTION),
        "--readout",
        READOUT,
        "--readout-w-init-mean",
        str(READOUT_W_INIT_MEAN),
        "--readout-w-init-std",
        str(READOUT_W_INIT_STD),
        "--surrogate-slope",
        str(SURROGATE_SLOPE),
        "--lr",
        str(LEARNING_RATE),
        "--weight-decay",
        str(WEIGHT_DECAY),
        "--fr-reg-upper-target-hz",
        str(RATE_PENALTY_TARGET_HZ),
        "--fr-reg-upper-strength",
        str(RATE_PENALTY_STRENGTH),
        "--dales-law",
        "--no-signed-readout",
        "--no-readout-bias",
        "--no-train-leak",
        "--no-adaptive-threshold",
        "--out-dir",
        str(output),
        "--wipe-dir",
    ]


def test_args(training: Path, output: Path) -> list[str]:
    """Evaluate the epoch-50 checkpoint on all official test images."""
    return [
        "sim",
        "--infer",
        "--load-config",
        str(training / "config.json"),
        "--load-weights",
        str(training / "weights_final.pth"),
        "--max-samples",
        str(MNIST_TEST_SAMPLES),
        "--out-dir",
        str(output),
        "--wipe-dir",
    ]


def raster_args(training: Path, output: Path) -> list[str]:
    """Record one shared digit-0 trial from the epoch-50 checkpoint."""
    return [
        "sim",
        "--infer",
        "--load-config",
        str(training / "config.json"),
        "--load-weights",
        str(training / "weights_final.pth"),
        "--input",
        "dataset",
        "--dataset",
        "mnist",
        "--digit",
        str(RASTER_DIGIT),
        "--sample",
        str(RASTER_SAMPLE_WITHIN_CLASS),
        "--recording-mode",
        "spikes",
        "--output-fields",
        "spk_e",
        "spk_i",
        "--out-dir",
        str(output),
        "--wipe-dir",
    ]
