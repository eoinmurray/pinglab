"""Local expectations for validated upstream training-bank identities.

These literal identities describe this experiment's supported scientific inputs.
The input validators compare recorded configurations and checkpoint provenance;
this module never imports the producer or selects a run.
"""

from copy import deepcopy

FR_STRENGTH_UPPER = 0.041
CELLS = [
    {
        "name": "frozen_ping__seed42",
        "model": "ping_init",
        "family": "init",
        "tag": "frozen_ping",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "ei_strength": 1.0,
        "trainable_w_ei": False,
        "trainable_w_ie": False,
        "training_run_id": "TR-05",
    },
    {
        "name": "frozen_ping__seed43",
        "model": "ping_init",
        "family": "init",
        "tag": "frozen_ping",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "ei_strength": 1.0,
        "trainable_w_ei": False,
        "trainable_w_ie": False,
        "training_run_id": "TR-05",
    },
    {
        "name": "frozen_ping__seed44",
        "model": "ping_init",
        "family": "init",
        "tag": "frozen_ping",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "ei_strength": 1.0,
        "trainable_w_ei": False,
        "trainable_w_ie": False,
        "training_run_id": "TR-05",
    },
    {
        "name": "trainable_ping_init__seed42",
        "model": "ping_init",
        "family": "init",
        "tag": "trainable_ping_init",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "ei_strength": 1.0,
        "trainable_w_ei": True,
        "trainable_w_ie": True,
        "training_run_id": "TR-05",
    },
    {
        "name": "trainable_ping_init__seed43",
        "model": "ping_init",
        "family": "init",
        "tag": "trainable_ping_init",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "ei_strength": 1.0,
        "trainable_w_ei": True,
        "trainable_w_ie": True,
        "training_run_id": "TR-05",
    },
    {
        "name": "trainable_ping_init__seed44",
        "model": "ping_init",
        "family": "init",
        "tag": "trainable_ping_init",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "ei_strength": 1.0,
        "trainable_w_ei": True,
        "trainable_w_ie": True,
        "training_run_id": "TR-05",
    },
    {
        "name": "trainable_zero_init__seed42",
        "model": "ping_init",
        "family": "init",
        "tag": "trainable_zero_init",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "ei_strength": 0.0,
        "trainable_w_ei": True,
        "trainable_w_ie": True,
        "training_run_id": "TR-05",
    },
    {
        "name": "trainable_zero_init__seed43",
        "model": "ping_init",
        "family": "init",
        "tag": "trainable_zero_init",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "ei_strength": 0.0,
        "trainable_w_ei": True,
        "trainable_w_ie": True,
        "training_run_id": "TR-05",
    },
    {
        "name": "trainable_zero_init__seed44",
        "model": "ping_init",
        "family": "init",
        "tag": "trainable_zero_init",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "ei_strength": 0.0,
        "trainable_w_ei": True,
        "trainable_w_ie": True,
        "training_run_id": "TR-05",
    },
    {
        "name": "trainable_small_init__seed42",
        "model": "ping_init",
        "family": "init",
        "tag": "trainable_small_init",
        "seed": 42,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "ei_strength": 0.1,
        "trainable_w_ei": True,
        "trainable_w_ie": True,
        "training_run_id": "TR-05",
    },
    {
        "name": "trainable_small_init__seed43",
        "model": "ping_init",
        "family": "init",
        "tag": "trainable_small_init",
        "seed": 43,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "ei_strength": 0.1,
        "trainable_w_ei": True,
        "trainable_w_ie": True,
        "training_run_id": "TR-05",
    },
    {
        "name": "trainable_small_init__seed44",
        "model": "ping_init",
        "family": "init",
        "tag": "trainable_small_init",
        "seed": 44,
        "dt_ms": 0.1,
        "tau_gaba": 6.0,
        "ei_strength": 0.1,
        "trainable_w_ei": True,
        "trainable_w_ie": True,
        "training_run_id": "TR-05",
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
