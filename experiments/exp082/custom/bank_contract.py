"""Local expectations for validated upstream training-bank identities.

These literal identities describe this experiment's supported scientific inputs.
The input validators compare recorded configurations and checkpoint provenance;
this module never imports the producer or selects a run.
"""

from copy import deepcopy

FR_STRENGTH_UPPER = 0.041
CELLS = [
    {
        "name": "ping__variable_rate__seed42",
        "model": "ping",
        "family": "variable_rate",
        "tag": "categorical 0.5–25 Hz · output spike count",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "max_samples": 7000,
        "readout": "spike-count",
        "input_rates_hz": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 25.0],
        "rate_sampling": "uniform categorical per presentation",
        "consumer": "exp082",
        "readout_w_init_mean": 0.05,
        "readout_w_init_std": 0.04,
        "status": "ready_to_train",
        "training_run_id": "TR-06",
    },
    {
        "name": "ping__variable_rate__seed43",
        "model": "ping",
        "family": "variable_rate",
        "tag": "categorical 0.5–25 Hz · output spike count",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "max_samples": 7000,
        "readout": "spike-count",
        "input_rates_hz": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 25.0],
        "rate_sampling": "uniform categorical per presentation",
        "consumer": "exp082",
        "readout_w_init_mean": 0.05,
        "readout_w_init_std": 0.04,
        "status": "ready_to_train",
        "training_run_id": "TR-06",
    },
    {
        "name": "ping__variable_rate__seed44",
        "model": "ping",
        "family": "variable_rate",
        "tag": "categorical 0.5–25 Hz · output spike count",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "max_samples": 7000,
        "readout": "spike-count",
        "input_rates_hz": [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 25.0],
        "rate_sampling": "uniform categorical per presentation",
        "consumer": "exp082",
        "readout_w_init_mean": 0.05,
        "readout_w_init_std": 0.04,
        "status": "ready_to_train",
        "training_run_id": "TR-06",
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
