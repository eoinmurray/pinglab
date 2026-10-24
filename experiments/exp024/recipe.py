"""Convergence audit definitions; training identities remain owned by exp022.

configuration() resolves fresh scientific settings before execution. Network
authoring and analysis definitions do not load data or launch a stage on import.
"""

from copy import deepcopy

FR_STRENGTH_UPPER = 0.041
CELLS = [
    {
        "name": "coba__off__seed42",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "off",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": None,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__off__seed43",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "off",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": None,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__off__seed44",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "off",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": None,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt25hz__seed42",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "25 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 25.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt25hz__seed43",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "25 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 25.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt25hz__seed44",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "25 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 25.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt10hz__seed42",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "10 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 10.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt10hz__seed43",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "10 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 10.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt10hz__seed44",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "10 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 10.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt5hz__seed42",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "5 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 5.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt5hz__seed43",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "5 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 5.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt5hz__seed44",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "5 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 5.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt2p5hz__seed42",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "2.5 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 2.5,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt2p5hz__seed43",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "2.5 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 2.5,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt2p5hz__seed44",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "2.5 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 2.5,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt1hz__seed42",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "1 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 1.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt1hz__seed43",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "1 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 1.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "coba__rt1hz__seed44",
        "model": "coba",
        "family": "activity_frontier",
        "tag": "1 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 1.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__off__seed42",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "off",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": None,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__off__seed43",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "off",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": None,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__off__seed44",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "off",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": None,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt25hz__seed42",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "25 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 25.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt25hz__seed43",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "25 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 25.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt25hz__seed44",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "25 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 25.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt10hz__seed42",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "10 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 10.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt10hz__seed43",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "10 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 10.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt10hz__seed44",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "10 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 10.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt5hz__seed42",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "5 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 5.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt5hz__seed43",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "5 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 5.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt5hz__seed44",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "5 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 5.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt2p5hz__seed42",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "2.5 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 2.5,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt2p5hz__seed43",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "2.5 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 2.5,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt2p5hz__seed44",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "2.5 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 2.5,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt1hz__seed42",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "1 Hz",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 1.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt1hz__seed43",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "1 Hz",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 1.0,
        "training_run_id": "TR-02",
    },
    {
        "name": "ping__rt1hz__seed44",
        "model": "ping",
        "family": "activity_frontier",
        "tag": "1 Hz",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "rate_target_hz": 1.0,
        "training_run_id": "TR-02",
    },
]


def training_run_cells(training_run_id):
    cells = [
        deepcopy(cell) for cell in CELLS if cell["training_run_id"] == training_run_id
    ]
    if not cells:
        raise ValueError(f"unsupported training family: {training_run_id}")
    return tuple(cells)


def training_run_values(training_run_id, field):
    values = []
    for cell in training_run_cells(training_run_id):
        value = cell[field]
        if value not in values:
            values.append(value)
    return tuple(values)


def training_run_cell(training_run_id, **identity):
    cells = [
        cell
        for cell in training_run_cells(training_run_id)
        if all(cell.get(key) == value for key, value in identity.items())
    ]
    if len(cells) != 1:
        raise ValueError(f"expected one {training_run_id} cell matching {identity}")
    return cells[0]


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
