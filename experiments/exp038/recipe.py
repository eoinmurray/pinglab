"""The retained exp038 inference recipe; training is owned by exp022."""

import numpy as np
from experiments.exp022 import FR_STRENGTH_UPPER as FR_STRENGTH_UPPER
from experiments.exp022 import training_run_cell, training_run_values
from experiments.exp022.checkpoints import checkpoint_policy
from experiments.helpers.datasets import MNIST_REDUCED_EVAL_SAMPLES

SLUG = "exp038"
REFRACTORY_E_MS = 1.2
REFRACTORY_I_MS = 0.6
REFRACTORY_POLICY = "exact"
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


def refractory_configuration() -> dict:
    return {
        "refractory_e_ms": REFRACTORY_E_MS,
        "refractory_i_ms": REFRACTORY_I_MS,
        "refractory_policy": REFRACTORY_POLICY,
    }




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


def configuration(*, smoke=False, version=2):
    if version not in (1, 2):
        raise ValueError("unsupported exp038 recipe version")
    return {
        "schema": f"exp038.recipe/v{version}",
        **(refractory_configuration() if version >= 2 else {}),
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




BIOPHYSICS = {
    "capacitance_e_nf": 1.0, "capacitance_i_nf": 0.5,
    "leak_e_us": 0.05, "leak_i_us": 0.10,
    "resting_mv": -65.0, "threshold_mv": -50.0, "reset_mv": -65.0,
    "readout_tau_ms": 2.0, "readout_threshold": 1.0,
}


def author_network(training, *, observables=()):
    from experiments.helpers.checkpoint_graph import author_network as author

    return author(
        SLUG, training, BIOPHYSICS, refractory_configuration(), observables=observables
    )


def inference_request(training, job):
    snapshot = "sample_index" in job
    request = {
        "checkpoint_role": CHECKPOINT_ROLE,
        "input": "synthetic"
        if job["kind"] == "fi_uniform"
        else "snapshot"
        if snapshot
        else "dataset",
        "t_ms": training["t_ms"],
        "input_rate_hz": job.get("input_rate", training["input_rate"]),
        "samples": job.get("samples"),
        "trials": job.get("trials"),
        "batch_size": job["trials"]
        if job["kind"] == "fi_uniform"
        else 1
        if snapshot
        else 64,
        "subset_seed": 42,
        "encoder_seed": 20260415,
        "sample_index": job.get("sample_index"),
        "observables": ["spikes"] if snapshot else [],
        "products": [],
    }
    if "ei_strength" in job:
        request["ei_strength"] = job["ei_strength"]
    return request
