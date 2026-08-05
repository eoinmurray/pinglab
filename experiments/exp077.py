"""Experiment 077: filter-matched calibration for variable-rate PING training.

The complete, staged scientific contract lives in ``writings/exp077.typ``.
Implement stages in dependency order and register each completed function in
``IMPLEMENTED_STEPS``.  ``EXP077_THROUGH_STEP=N`` is a meta-control: it changes
only how far the registered protocol runs, never the protocol itself.

Large regenerable intermediates belong under ``temp/experiments/exp077``.
Plot-ready arrays, summaries, figures, provenance, and the reproducer belong
under ``artifacts/data/exp077``.  Plot code must read only from the latter.
"""

from __future__ import annotations

import os
from collections.abc import Callable

SLUG = "exp077"
N_STEPS = 7

STAGE_NAMES: dict[int, str] = {
    1: "generate and validate filter-matched pixel features",
    2: "generate the empirical pixel-response library",
    3: "calculate and test the dependent linear-filter prediction",
    4: "construct and validate complete sampled feature images",
    5: "train the mixed-rate nonlinear and linear decoders",
    6: "evaluate held-out psychometric curves and select thresholds",
    7: "write the variable-rate training-range decision",
}


def _not_implemented(step: int) -> None:
    raise NotImplementedError(
        f"exp077 Step {step} is specified but not implemented: "
        f"{STAGE_NAMES[step]}. Follow writings/exp077.typ and register the "
        "completed stage in IMPLEMENTED_STEPS."
    )


def step_1() -> None:
    _not_implemented(1)


def step_2() -> None:
    _not_implemented(2)


def step_3() -> None:
    _not_implemented(3)


def step_4() -> None:
    _not_implemented(4)


def step_5() -> None:
    _not_implemented(5)


def step_6() -> None:
    _not_implemented(6)


def step_7() -> None:
    _not_implemented(7)


STAGE_FUNCTIONS: dict[int, Callable[[], None]] = {
    1: step_1,
    2: step_2,
    3: step_3,
    4: step_4,
    5: step_5,
    6: step_6,
    7: step_7,
}

# Add a step number only after its method, focused validation, committed
# plot-ready record, and Results figure are all implemented.
IMPLEMENTED_STEPS: frozenset[int] = frozenset()


def requested_through_step() -> int:
    raw = os.environ.get("EXP077_THROUGH_STEP", str(N_STEPS))
    try:
        step = int(raw)
    except ValueError as exc:
        raise SystemExit("EXP077_THROUGH_STEP must be an integer from 1 through 7") from exc
    if step not in STAGE_NAMES:
        raise SystemExit("EXP077_THROUGH_STEP must be an integer from 1 through 7")
    return step


def main() -> None:
    through_step = requested_through_step()
    for step in range(1, through_step + 1):
        if step not in IMPLEMENTED_STEPS:
            _not_implemented(step)
        STAGE_FUNCTIONS[step]()


if __name__ == "__main__":
    main()
