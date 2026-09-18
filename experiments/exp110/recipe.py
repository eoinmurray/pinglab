"""Committed manuscript-figure definitions; no execution on import."""

SLUG = "exp110"
RATE_FREQUENCY_SOURCE = "rate_vs_fgamma.svg"
CYCLE_COUNT_SOURCE = "spikes_per_cycle_distribution_equal_network.svg"
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


def configuration(exp054_recipe: dict, exp115_recipe: dict) -> dict:
    return {
        "schema": "exp110.presentation/v14",
        "cycle_count_weighting": "equal_network",
        "figures": (
            "gamma-onset",
            "rate-frequency-and-cycle-participation",
            "spike-perturbation-and-timestep-robustness",
        ),
        "source_stage": "analyse",
        "source_recipes": {
            "exp054": exp054_recipe,
            "exp115": exp115_recipe,
        },
    }
