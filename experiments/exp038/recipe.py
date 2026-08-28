"""The retained exp038 inference recipe; training is owned by exp022."""

import numpy as np
from experiments.exp022 import FR_STRENGTH_UPPER as FR_STRENGTH_UPPER
from experiments.exp022 import training_run_cell, training_run_values
from experiments.helpers.checkpoints import checkpoint_policy
from experiments.helpers.datasets import MNIST_REDUCED_EVAL_SAMPLES

SLUG = "exp038"
ANALYSIS_PURPOSE = "deployment_performance"
CHECKPOINT_POLICY = checkpoint_policy(ANALYSIS_PURPOSE)
CHECKPOINT_ROLE = CHECKPOINT_POLICY["role"]
MODELS = list(training_run_values("TR-02", "model"))
SEEDS_BASELINE = list(training_run_values("TR-02", "seed"))
RATE_TARGET_GRID_HZ = list(training_run_values("TR-02", "rate_target_hz"))
EVAL_MAX_SAMPLES = MNIST_REDUCED_EVAL_SAMPLES
EI_RASTER_N_E_PLOT, EI_RASTER_N_I_PLOT = 200, 64
EI_SWEEP = [round(0.1 * i, 1) for i in range(11)]
EI_RASTER = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
FI_UNIFORM_RATES_HZ = [
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    12.0,
    14.0,
    16.0,
    18.0,
    20.0,
    25.0,
    30.0,
    35.0,
    40.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    100.0,
]
FIGURES = tuple(
    name + "." + ext
    for name, extensions in (
        ("rate_rasters__ping", ("png", "pdf")),
        ("fi_curve__ping", ("svg", "pdf")),
        ("fi_curve_uniform", ("svg", "pdf")),
        ("ei_rasters", ("png", "pdf")),
        ("loop_transfer_compound", ("png", "pdf")),
    )
    for ext in extensions
)


def cell_name(model: str, rate_target_hz: float | None, seed: int) -> str:
    return training_run_cell(
        "TR-02", model=model, rate_target_hz=rate_target_hz, seed=seed
    )["name"]


def rate_target_display(rate_target_hz: float | None) -> str:
    """Human label for plots / numbers.json."""
    if rate_target_hz is None:
        return "off"
    return f"{rate_target_hz:g}"


def seeds_for(rate_target_hz: float | None) -> list[int]:
    """Return the independent seeds used at every frontier point."""
    return list(SEEDS_BASELINE)


def bank_cells():
    return [
        {
            "cell_name": cell_name(m, t, s),
            "model": m,
            "rate_target_hz": t,
            "seed": s,
            "w_in": 0.9,
        }
        for m in MODELS
        for t in RATE_TARGET_GRID_HZ
        for s in SEEDS_BASELINE
    ]


def configuration(*, smoke=False):
    return {
        "schema": "exp038.recipe/v1",
        "profile": "smoke" if smoke else "production",
        "checkpoint_policy": CHECKPOINT_POLICY,
        "evaluation_samples": 100 if smoke else EVAL_MAX_SAMPLES,
        "seeds": SEEDS_BASELINE,
        "illustrative_seed": 42,
        "sample_index": 0,
        "ei_strengths": [0.0, 0.5, 1.0] if smoke else EI_SWEEP,
        "ei_rasters": [0.0, 1.0] if smoke else EI_RASTER,
        "rate_rasters": [0.0, 10.0, 100.0]
        if smoke
        else np.linspace(0.0, 100.0, 40)[:10].tolist(),
        "uniform_rates": [0.0, 10.0, 100.0] if smoke else FI_UNIFORM_RATES_HZ,
        "uniform_trials": 2 if smoke else 32,
    }


def jobs(cfg):
    result = []

    def add(kind, model, seed, value, **extra):
        result.append(
            {
                "kind": kind,
                "model": model,
                "seed": seed,
                "cell_name": cell_name(model, None, seed),
                "path": f"{kind}/{model}__seed{seed}__{value:g}",
                **extra,
            }
        )

    for rate in cfg["rate_rasters"]:
        add("rate_raster", "ping", 42, rate, input_rate=rate, sample_index=0)
    for model in MODELS:
        for rate in cfg["uniform_rates"]:
            add(
                "fi_uniform",
                model,
                42,
                rate,
                input_rate=rate,
                trials=cfg["uniform_trials"],
            )
    for seed in cfg["seeds"]:
        for strength in cfg["ei_strengths"]:
            add(
                "ei_sweep",
                "coba",
                seed,
                strength,
                ei_strength=strength,
                samples=cfg["evaluation_samples"],
            )
    for strength in cfg["ei_rasters"]:
        add(
            "ei_raster",
            "coba",
            42,
            strength,
            ei_strength=strength,
            sample_index=0,
            samples=cfg["evaluation_samples"],
        )
    return result


def inference_args(train, checkpoint, output, job):
    args = [
        "sim",
        "--load-config",
        str(train / "config.json"),
        "--load-weights",
        str(checkpoint),
        "--device",
        "auto",
    ]
    if job["kind"] == "fi_uniform":
        args += [
            "--input",
            "synthetic-spikes",
            "--n-in",
            "784",
            "--input-rate",
            str(job["input_rate"]),
            "--n-batch",
            str(job["trials"]),
        ]
    else:
        args += ["--infer"]
        if "ei_strength" in job:
            args += [
                "--ei-strength",
                str(job["ei_strength"]),
                "--skip-load",
                "W_ei.",
                "W_ie.",
            ]
        if "input_rate" in job:
            args += ["--input-rate", str(job["input_rate"])]
        if "samples" in job:
            args += ["--max-samples", str(job["samples"])]
        if "sample_index" in job:
            args += ["--sample-index", str(job["sample_index"])]
    return args + ["--out-dir", str(output)]
