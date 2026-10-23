"""Convergence audit definitions; training identities remain owned by exp022.

configuration() resolves fresh scientific settings before execution. Network
authoring and analysis definitions do not load data or launch a stage on import.
"""

from copy import deepcopy

from experiments.exp024.custom.bank_contract import (
    training_run_cell,
    training_run_values,
)

SLUG = "exp024"
TRAINING_RUN = "TR-02"
MODELS = training_run_values(TRAINING_RUN, "model")
SEEDS = training_run_values(TRAINING_RUN, "seed")
WINDOW = 10
ACCURACY_THRESHOLD = 0.1  # percentage points per epoch
RATE_THRESHOLD = 0.05  # Hz per epoch
ACCURACY_FRACTION = 0.99
FIELDS = ("acc", "loss", "test_loss", "test_rate_e", "test_rate_i")
PARAMETERS = ("W_ff.0", "W_ff.1")
FIGURES = ("coba_curves.svg", "ping_curves.svg", "confidence_inflation.svg")


def cell_name(model: str, seed: int) -> str:
    return training_run_cell(TRAINING_RUN, model=model, rate_target_hz=None, seed=seed)[
        "name"
    ]


def slope_last_n(values: list[float], n: int = WINDOW) -> float:
    """Historical endpoint secant, not a least-squares regression slope."""
    if len(values) < 2 or n < 2:
        raise ValueError("a slope requires at least two epochs")
    tail = values[-n:]
    return (tail[-1] - tail[0]) / (len(tail) - 1)


def accuracy_marker(values: list[float]) -> int | None:
    """First epoch reaching 99% of final accuracy; not sustained convergence."""
    if values[-1] <= 0:
        return None
    return next(
        i + 1
        for i, value in enumerate(values)
        if value >= ACCURACY_FRACTION * values[-1]
    )


def configuration():
    """Resolve the history audit without importing the bank producer."""
    return deepcopy(
        dict(
            training_run=TRAINING_RUN,
            window_epochs=WINDOW,
            accuracy_threshold_pp_per_epoch=ACCURACY_THRESHOLD,
            rate_threshold_hz_per_epoch=RATE_THRESHOLD,
            accuracy_marker_fraction=ACCURACY_FRACTION,
            slope_estimator="endpoint_secant",
            threshold_rule="absolute_slope_strictly_below",
        )
    )
