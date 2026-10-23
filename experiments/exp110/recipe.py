"""Committed manuscript-figure definitions; no execution on import.

configuration() resolves fresh scientific settings before execution. Network
authoring and analysis definitions do not load data or launch a stage on import.
"""

from copy import deepcopy

SLUG = "exp110"
RATE_FREQUENCY_SOURCE = "rate_vs_fgamma.svg"
CYCLE_COUNT_SOURCE = "spikes_per_cycle_distribution_equal_network.svg"
PERTURBATION_SOURCE = "numbers.json"
TIMESTEP_SOURCE = "numbers.json"
FIGURES = (
    "onset_super_compound.svg",
    "onset_super_compound.pdf",
    "cycle_participation_compound.svg",
    "cycle_participation_compound.pdf",
    "robustness_compound.svg",
    "robustness_compound.pdf",
)


def configuration(exp054_recipe: dict, exp117_recipe: dict) -> dict:
    return deepcopy(
        dict(
            schema="exp110.presentation/v15",
            cycle_count_weighting="equal_network",
            figures=(
                "gamma-onset",
                "rate-frequency-and-cycle-participation",
                "spike-perturbation-and-timestep-robustness",
            ),
            source_stage="analyse",
            source_recipes=dict(
                exp054=exp054_recipe,
                exp117=exp117_recipe,
            ),
        )
    )
