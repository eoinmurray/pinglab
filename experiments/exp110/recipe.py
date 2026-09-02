"""Committed manuscript-figure definitions; no execution on import."""

SLUG = "exp110"
SOURCE_EXPERIMENT = "exp054"
SOURCE_STAGE = "analyse"
RATE_FREQUENCY_SOURCE = "rate_vs_fgamma.svg"
CYCLE_COUNT_SOURCE = "spikes_per_cycle_distribution.svg"
PERTURBATION_SOURCE = "numbers.json"
TIMESTEP_SOURCE = "numbers.json"
FIGURES = (
    "onset_super_compound.png",
    "onset_super_compound.pdf",
    "cycle_participation_compound.png",
    "cycle_participation_compound.pdf",
    "robustness_compound.png",
    "robustness_compound.pdf",
)


def configuration(source_recipe: dict) -> dict:
    return {
        "schema": "exp110.presentation/v6",
        "figures": (
            "gamma-onset",
            "rate-frequency-and-cycle-participation",
            "spike-perturbation-and-timestep-robustness",
        ),
        "source_experiment": SOURCE_EXPERIMENT,
        "source_stage": SOURCE_STAGE,
        "source_recipe": source_recipe,
    }
