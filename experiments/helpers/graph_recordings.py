"""Convert named graph spike observables to the standard sparse event arrays."""

from pathlib import Path

import numpy as np


def write_spike_events(source: Path, destination: Path, config: dict) -> None:
    """Keep integer step coordinates and the paired single-trial recording layout."""
    steps = round(config["duration_ms"] / config["dt_ms"])
    arrays = {
        "dt": np.float32(config["dt_ms"]),
        "n_trials": np.int32(1),
        "T": np.int32(steps),
        "n_e": np.int32(config["n_e"]),
        "n_i": np.int32(config["n_i"]),
    }
    with np.load(source, allow_pickle=False) as recording:
        for population, size in (
            ("e", config["n_e"]),
            ("i", config["n_i"]),
            ("out", 10),
        ):
            spikes = recording[f"{population}_spikes"]
            if spikes.shape != (steps, 1, size):
                raise ValueError(f"Unexpected {population} spike shape: {spikes.shape}")
            if not np.all((spikes == 0) | (spikes == 1)):
                raise ValueError(f"Non-binary {population} spike recording")
            time, trial, cell = np.nonzero(spikes)
            arrays.update(
                {
                    f"{population}_t": time.astype(np.int32),
                    f"{population}_trial": trial.astype(np.int32),
                    f"{population}_cell": cell.astype(np.int32),
                }
            )
    np.savez_compressed(destination, **arrays)
