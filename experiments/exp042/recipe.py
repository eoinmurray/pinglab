"""The reduced timing intervention recipe; no execution or data selection on import."""

import re

from experiments.exp022.recipe import training_run_cell, training_run_values
from experiments.helpers.checkpoints import checkpoint_policy
from experiments.helpers.datasets import MNIST_REDUCED_EVAL_SAMPLES
from experiments.helpers.operating_point import F_GAMMA_HZ

SLUG = "exp042"
TRAINING_RUN = "TR-02"
ANALYSIS_PURPOSE = "endpoint_dynamics"
CHECKPOINT_POLICY = checkpoint_policy(ANALYSIS_PURPOSE)
CHECKPOINT_ROLE = CHECKPOINT_POLICY["role"]
EVAL_SEED = 20260415
SEEDS = training_run_values(TRAINING_RUN, "seed")
CONDITIONS = ("baseline", "phase_shuffled_i", "poisson_matched_i")
JITTER_SIGMAS_MS = (0.0, 1.0, 3.0, 7.0, 14.0, 21.0, 28.0, 42.0, 60.0, 100.0)
CELL_JITTER_SIGMAS_MS = (0.0, 0.5, 1.0, 2.0, 5.0, 9.0, 14.0, 21.0, 50.0)
F_GAMMA_REFERENCE_HZ = F_GAMMA_HZ
EVAL_MAX_SAMPLES = MNIST_REDUCED_EVAL_SAMPLES
SMOKE_MAX_SAMPLES = 100
RASTER_SAMPLE_IDX = 0
RASTER_N_E_PLOT = 200
RASTER_N_I_PLOT = 64
COMPOUND_SIGMA_MS = 14.0
SHARDS = 8
FIGURES = (
    "rhythm_compound.png",
    "rhythm_compound.pdf",
    "cell_jitter_sweep.svg",
    "cell_jitter_sweep.pdf",
    "jitter_sweep.svg",
    "jitter_sweep.pdf",
)


def cell_name(seed):
    return training_run_cell(
        TRAINING_RUN, model="ping", rate_target_hz=None, seed=seed
    )["name"]


def configuration(*, smoke=False):
    return {
        "schema": "exp042.recipe/v1",
        "profile": "smoke" if smoke else "production",
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "jitter_sigmas_ms": list((0.0, 14.0, 100.0) if smoke else JITTER_SIGMAS_MS),
        "cell_jitter_sigmas_ms": list(
            (0.0, 0.5, 1.0, 2.0, 5.0, 9.0, 14.0) if smoke else CELL_JITTER_SIGMAS_MS
        ),
        "evaluation_samples": SMOKE_MAX_SAMPLES if smoke else EVAL_MAX_SAMPLES,
        "evaluation_partition": "official_mnist_test",
        "evaluation_seed": EVAL_SEED,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "f_gamma_reference_hz": F_GAMMA_REFERENCE_HZ,
        "raster": {
            "seed": SEEDS[0],
            "sample_index": RASTER_SAMPLE_IDX,
            "sigma_ms": COMPOUND_SIGMA_MS,
            "selection_seed": 0,
            "n_e_plot": RASTER_N_E_PLOT,
            "n_i_plot": RASTER_N_I_PLOT,
        },
    }


def jobs(cfg):
    result = []
    for seed in cfg["seeds"]:
        groups = [("results", c, seed, None) for c in cfg["conditions"]]
        groups += [
            ("jitter_sweep", f"jitter_sigma_{s:g}", seed + int(s), s)
            for s in cfg["jitter_sigmas_ms"]
        ]
        groups += [
            ("cell_jitter_sweep", f"cell_jitter_sigma_{s:g}", seed + int(s * 13), s)
            for s in cfg["cell_jitter_sigmas_ms"]
        ]
        for group, condition, offset, sigma in groups:
            identity = re.sub(
                r"(\d)\.(\d)", r"\1p\2", f"eval__{cell_name(seed)}__{condition}"
            )
            result.append(
                {
                    "id": identity,
                    "seed": seed,
                    "cell": cell_name(seed),
                    "group": group,
                    "condition": condition,
                    "seed_offset": offset,
                    "sigma_ms": sigma,
                }
            )
    return result
