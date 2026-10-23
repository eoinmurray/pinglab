"""Local expectations for validated upstream training-bank identities.

These literal identities describe this experiment's supported scientific inputs.
The input validators compare recorded configurations and checkpoint provenance;
this module never imports the producer or selects a run.
"""

from copy import deepcopy

FR_STRENGTH_UPPER = 0.041
CELLS = [
    {
        "name": "ping__tg4p5__seed42",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=4.5",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 4.5,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg4p5__seed43",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=4.5",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 4.5,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg4p5__seed44",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=4.5",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 4.5,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg6__seed42",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=6",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg6__seed43",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=6",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg6__seed44",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=6",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg9__seed42",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=9",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 9.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg9__seed43",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=9",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 9.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg9__seed44",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=9",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 9.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg12__seed42",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=12",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 12.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg12__seed43",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=12",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 12.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg12__seed44",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=12",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 12.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg18__seed42",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=18",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 18.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg18__seed43",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=18",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 18.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg18__seed44",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=18",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 18.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg27__seed42",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=27",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 27.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg27__seed43",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=27",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 27.0,
        "training_run_id": "TR-03",
    },
    {
        "name": "ping__tg27__seed44",
        "model": "ping",
        "family": "tau_gaba",
        "tag": "τ=27",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 27.0,
        "training_run_id": "TR-03",
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
