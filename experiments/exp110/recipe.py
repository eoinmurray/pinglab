"""Committed manuscript-figure definitions; no execution on import."""

SLUG = "exp110"
SOURCE_EXPERIMENT = "exp054"
SOURCE_STAGE = "analyse"
PERFORMANCE_SOURCE = "results_compound.png"
TRANSFER_SOURCE = "loop_transfer_compound.png"
RATE_FREQUENCY_SOURCE = "rate_vs_fgamma.svg"
CYCLE_COUNT_SOURCE = "spikes_per_cycle_distribution.svg"
PERTURBATION_SOURCE = "numbers.json"
TIMESTEP_SOURCE = "numbers.json"
FIGURES = (
    "onset_super_compound.png",
    "onset_super_compound.pdf",
    "performance_transfer_compound.png",
    "performance_transfer_compound.pdf",
    "cycle_participation_compound.png",
    "cycle_participation_compound.pdf",
    "robustness_compound.png",
    "robustness_compound.pdf",
)


def configuration(source_recipe: dict) -> dict:
    return {
        "schema": "exp110.presentation/v5",
        "figures": (
            "gamma-onset",
            "performance-and-loop-transfer",
            "rate-frequency-and-cycle-participation",
            "spike-perturbation-and-timestep-robustness",
        ),
        "source_experiment": SOURCE_EXPERIMENT,
        "source_stage": SOURCE_STAGE,
        "source_recipe": source_recipe,
    }
