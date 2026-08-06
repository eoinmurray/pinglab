"""Experiment 078: predicted acquisition of synchrony between PING circuits.

The staged scientific contract lives in ``writings/exp078.typ``. Implement the
stages in order and add a stage number to ``IMPLEMENTED_STEPS`` only after its
focused validation and publication artifacts exist.
"""

from __future__ import annotations

import os
from collections.abc import Callable

SLUG = "exp078"
N_STEPS = 5

STAGE_NAMES: dict[int, str] = {
    1: "calibrate two independent mature PING oscillators",
    2: "measure the macroscopic E-to-I phase-response curve",
    3: "predict stable phase relationships from the measured response",
    4: "test the predicted locking and non-locking conditions",
    5: "replicate the prediction test across held-out states and inputs",
}


def _not_implemented(step: int) -> None:
    raise NotImplementedError(
        f"exp078 Step {step} is specified but not implemented: "
        f"{STAGE_NAMES[step]}. Follow writings/exp078.typ and register the "
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


STAGE_FUNCTIONS: dict[int, Callable[[], None]] = {
    1: step_1,
    2: step_2,
    3: step_3,
    4: step_4,
    5: step_5,
}

IMPLEMENTED_STEPS: frozenset[int] = frozenset()


def requested_through_step() -> int:
    raw = os.environ.get("EXP078_THROUGH_STEP", str(N_STEPS))
    try:
        step = int(raw)
    except ValueError as exc:
        raise SystemExit("EXP078_THROUGH_STEP must be an integer from 1 through 5") from exc
    if step not in STAGE_NAMES:
        raise SystemExit("EXP078_THROUGH_STEP must be an integer from 1 through 5")
    return step


def main() -> None:
    through_step = requested_through_step()
    for step in range(1, through_step + 1):
        if step not in IMPLEMENTED_STEPS:
            _not_implemented(step)
        STAGE_FUNCTIONS[step]()


if __name__ == "__main__":
    main()
