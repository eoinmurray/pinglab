"""Preserved exp037 perturbation recipe; no execution or storage on import."""

from experiments.exp022 import FR_STRENGTH_UPPER as FR_STRENGTH_UPPER
from experiments.exp022 import training_run_cell, training_run_values
from experiments.helpers.checkpoints import checkpoint_policy
from experiments.helpers.datasets import MNIST_REDUCED_EVAL_SAMPLES

SLUG = "exp037"
SHARDS = 6
ANALYSIS_PURPOSE = "deployment_performance"
CHECKPOINT_POLICY = checkpoint_policy(ANALYSIS_PURPOSE)
CHECKPOINT_ROLE = CHECKPOINT_POLICY["role"]
MODELS = list(training_run_values("TR-02", "model"))
SEEDS_BASELINE = list(training_run_values("TR-02", "seed"))
RATE_TARGET_GRID_HZ = list(training_run_values("TR-02", "rate_target_hz"))
EVAL_MAX_SAMPLES = MNIST_REDUCED_EVAL_SAMPLES
EI_RASTER_N_E_PLOT, EI_RASTER_N_I_PLOT = 200, 64
PERTURB_DROP_LEVELS = [round(0.1 * i, 1) for i in range(11)]
PERTURB_ADD_LEVELS = [float(2 * i) for i in range(21)]
PERTURB_RASTER_DROP_LEVELS = [0.0, 0.5, 1.0]
PERTURB_RASTER_ADD_LEVELS = [0.0, 20.0, 40.0]
SNAPSHOT_ARRAYS = ("dt", "n_e", "n_i", "label", "spk_e", "spk_i")
FIGURES = (
    "perturbation_curves.svg",
    "perturbation_curves.pdf",
    *(
        f"perturb_rasters__{mode}__{model}.{ext}"
        for model in MODELS
        for mode in ("drop", "add")
        for ext in ("png", "pdf")
    ),
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


def _level_tag(level) -> str:
    if isinstance(level, (list, tuple)):
        return "_".join(_level_tag(x) for x in level)
    return f"{float(level):g}".replace(".", "p")


def _parse_job(job_id: str) -> tuple[str, str, int, str, str]:
    """Return (kind, model, seed, mode, level_tag)."""
    parts = job_id.split("__")
    if len(parts) != 5 or not parts[2].startswith("seed"):
        raise ValueError(f"bad job id {job_id!r}")
    return parts[0], parts[1], int(parts[2].removeprefix("seed")), parts[3], parts[4]


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
        "schema": "exp037.recipe/v1",
        "profile": "smoke" if smoke else "production",
        "checkpoint_policy": CHECKPOINT_POLICY,
        "evaluation_samples": 100 if smoke else EVAL_MAX_SAMPLES,
        "seeds": SEEDS_BASELINE,
        "illustrative_seed": SEEDS_BASELINE[0],
        "sample_index": 0,
        "drop_levels": [0.0, 0.5, 0.8, 1.0] if smoke else PERTURB_DROP_LEVELS,
        "add_levels": [0.0, 20.0, 40.0] if smoke else PERTURB_ADD_LEVELS,
        "raster_drop_levels": PERTURB_RASTER_DROP_LEVELS,
        "raster_add_levels": PERTURB_RASTER_ADD_LEVELS,
    }


def jobs(cfg):
    result = []
    for model in MODELS:
        for kind, seeds in (
            ("sweep", cfg["seeds"]),
            ("raster", [cfg["illustrative_seed"]]),
        ):
            for seed in seeds:
                for mode in ("drop", "add"):
                    levels = cfg[
                        ("raster_" if kind == "raster" else "") + mode + "_levels"
                    ]
                    for level in levels:
                        identity = (
                            f"{kind}__{model}__seed{seed}__{mode}__{_level_tag(level)}"
                        )
                        result.append(
                            {
                                "id": identity,
                                "path": f"jobs/{identity}",
                                "kind": kind,
                                "model": model,
                                "seed": seed,
                                "mode": mode,
                                "level": level,
                                "cell_name": cell_name(model, None, seed),
                                "samples": cfg["evaluation_samples"],
                                **(
                                    {"sample_index": cfg["sample_index"]}
                                    if kind == "raster"
                                    else {}
                                ),
                            }
                        )
    return result


def infer_jobs():
    import os

    return [
        j["id"]
        for j in jobs(configuration(smoke=os.environ.get("PINGLAB_SMOKE") == "1"))
    ]


def inference_args(train, checkpoint, output, job):
    args = [
        "sim",
        "--infer",
        "--load-config",
        str(train / "config.json"),
        "--load-weights",
        str(checkpoint),
        "--perturb-mode",
        job["mode"],
        "--perturb-level",
        str(job["level"]),
        "--max-samples",
        str(job["samples"]),
    ]
    if job["kind"] == "raster":
        args += ["--sample-index", str(job["sample_index"])] + [
            "--recording-mode",
            "spikes",
            "--output-fields",
            "spk_e",
            "spk_i",
        ]
    return args + ["--out-dir", str(output)]
