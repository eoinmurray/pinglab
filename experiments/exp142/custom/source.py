"""Read the pinned exp141 diagnostic and its validation-selected weights.

Validate the recorded recipe, sample and matrix inventory before replaying any
network. The source is a validated Pingstore run; no exp141 code is imported.
"""

import json

import numpy as np
from experiments.exp142.analyse import arrays
from experiments.exp142.custom.network_artifacts import snapshot_parameters


def source_data(source, cfg):
    saved_cfg = source.record["execution"]["configuration"]
    expected = dict(
        schema="exp141.recipe/v9",
        n_input=cfg["n_input"],
        n_hidden=cfg["n_e"],
        n_inhibitory=cfg["n_i"],
        n_classes=cfg["n_output"],
        dt_ms=0.2,
        duration_ms=200.0,
        sample_index=cfg["sample_index"],
        input_rate_hz=cfg["source_input_rate_hz"],
    )
    expected.update({k: v for k, v in cfg["physics"].items() if k in saved_cfg})
    for key, value in expected.items():
        if saved_cfg.get(key) != value:
            raise ValueError(f"unsupported source setting: {key}")
    recording = arrays(source.file("recording.npz"))
    drive = recording["drive"][:, None, :]
    if drive.shape != (1000, 1, cfg["n_input"]) or not np.isin(drive, (0, 1)).all():
        raise ValueError("invalid saved source input")
    indices = np.column_stack(np.nonzero(drive)).astype(np.int64)
    events = dict(
        time_ns=indices[:, 0] * cfg["input_grid_ns"],
        batches=indices[:, 1],
        channels=indices[:, 2],
    )
    result = json.loads(source.file("results.json").read_text())
    data = dict(
        sample_features=recording["pixels"].reshape(1, -1),
        sample_labels=np.array([result["sample"]["label"]]),
    )
    if data["sample_features"].shape != (1, cfg["n_input"]):
        raise ValueError("invalid original sample")
    partition = result["sample"]["partition"]
    if (
        result["sample"]["checkpoint"] != "validation_selected"
        or result["sample"]["index"] != cfg["sample_index"]
    ):
        raise ValueError("unexpected source diagnostic checkpoint or sample")
    models = {}
    for mode in cfg["models"]:
        w = snapshot_parameters(source, mode, "best-validation", "start")
        shapes = {
            "input_to_E.weight": (cfg["n_input"], cfg["n_e"]),
            "readout_projection.weight": (cfg["n_e"], cfg["n_output"]),
        }
        if mode == "ping":
            shapes.update(
                {
                    "E_to_I.weight": (cfg["n_e"], cfg["n_i"]),
                    "I_to_E.weight": (cfg["n_i"], cfg["n_e"]),
                }
            )
        if set(w) != set(shapes):
            raise ValueError("unexpected source projections")
        for name, shape in shapes.items():
            value = w[name]
            if (
                value.shape != shape
                or value.dtype != np.float32
                or not np.isfinite(value).all()
                or np.any(value < 0)
            ):
                raise ValueError(f"invalid source matrix {name}")
        models[mode] = w
    return (
        events,
        models,
        data["sample_features"][0],
        int(data["sample_labels"][0]),
        partition,
    )


def source_recording(source, mode):
    """Select original native diagnostic spikes and scores from the pinned run."""
    saved = arrays(source.file("recording.npz"))
    return {
        **{
            name: saved[f"{mode}--{name}"]
            for name in ("E", "readout", *(["I"] if mode == "ping" else []))
        },
        "scores": saved[f"{mode}--scores"],
    }
