"""Scientific definitions; importing this module does not execute training."""

SLUG = "exp126"
PROFILE_CONFIGURATION = {
    "schema": "exp126.profile/v1",
    "batch_size": 128,
    "steps": 50,
    "warmups": 3,
    "repeats": 15,
    "source_seed": 42,
    "events_seed": 642,
    "variants": [
        "snntorch",
        "snnlab",
        "no_finite",
        "no_finite_binary",
        "instantaneous",
        "cached_surrogate",
    ],
    "timing_scope": "preencoded forward or forward/loss/backward; no optimizer, encoding, construction or profilers",
    "diagnostic_only": True,
    "order_seed": 20261014,
}
CONFIG = {
    "schema": "exp126.recipe/v1",
    "seed": 42,
    "inputs": 784,
    "hidden": 256,
    "outputs": 10,
    "train_per_class": 600,
    "validation_per_class": 100,
    "epochs": 30,
    "batch_size": 128,
    "evaluation_batch_size": 256,
    "steps": 50,
    "dt_ms": 1.0,
    "peak_input_rate_hz": 250.0,
    "alpha": 0.8,
    "beta": 0.9,
    "threshold": 1.0,
    "surrogate_slope": 25.0,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "gradient_clip": 1.0,
    "inference_repeats": 5,
    "warmup_batches": 3,
    "selection": "minimum validation cross-entropy across epochs 1–30; earliest tie",
}


def comparison_configuration(seed=42):
    return {
        **CONFIG,
        "schema": "exp126.matched/v3",
        "seed": seed,
        "timing_repeats": 5,
        "probe_examples": 8,
        "state_atol": 1e-5,
        "state_rtol": 1e-5,
        "gradient_atol": 1e-5,
        "gradient_rtol": 1e-4,
        "affine": "source-target matrix multiply plus separate trainable bias",
        "encoding": "CPU uniform stream copied to CUDA for Bernoulli thresholding; dense time-major events; encoding and transfer timed separately from optimization",
        "snnlab": "GraphExecutor with matched custom neuron and surrogate; structural extension validation",
        "snnlab_extension_validation": "structural",
        "snnlab_policy_selection": "library default, verified against the declared structural policy",
        "strict_validation_microbenchmark": True,
    }


# Each phase is one complete trained implementation replicate, including its
# initialization probe, benchmarks, learning history and selected-model test pass.
COMPARISON_PHASES = ("snntorch", "snnlab")
PHASE_EXPORT_FILES = (
    "dataset.npz",
    "initial_weights.npz",
    "agreement.npz",
    "gradients.npz",
    "selected_weights.npz",
    "final_weights.npz",
    "learning.json",
    "test_predictions.npz",
)


def condition_role(phase, seed):
    if phase not in COMPARISON_PHASES or type(seed) is not int or seed < 0:
        raise ValueError("invalid comparison phase or seed")
    return f"{phase}-seed-{seed}"


def assembled_configuration(seeds):
    seeds = sorted(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("assembly needs unique paired seeds")
    for seed in seeds:
        condition_role("snntorch", seed)
    return {
        "schema": "exp126.assembled/v1",
        "seeds": seeds,
        "phases": list(COMPARISON_PHASES),
        "protocol": {
            key: value
            for key, value in comparison_configuration().items()
            if key != "seed"
        },
    }


def implementation_source_digest(repo):
    import hashlib

    digest = hashlib.sha256()
    for name in (
        "recipe.py",
        "compute.py",
        "compute_parts/matched_models.py",
        "compute_parts/comparison_compute.py",
    ):
        digest.update(name.encode() + b"\0")
        digest.update((repo / "experiments/exp126" / name).read_bytes())
    return digest.hexdigest()


CURRENT_PROFILE_CONFIGURATION = {
    "schema": "exp126.current-profile/v1",
    "warmups": 3,
    "repeats": 11,
    "batch_size": 128,
    "states": ["initial", "trained"],
    "variants": ["snntorch", "snnlab", "zero_decay", "no_snapshots", "combined"],
}
