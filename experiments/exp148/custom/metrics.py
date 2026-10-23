"""Validate sparse events and measure firing and population-cycle intervals.

Window rates include silent cells. Oscillation estimates use population-count
peaks in the hold window; they do not classify AI or identify a PING mechanism.
"""

import numpy as np
from experiments.exp148.recipe import windows
from scipy.signal import find_peaks


def validate_events(events, cfg):
    for population in ("E", "I"):
        times = np.asarray(events[population + "_time_ms"])
        cells = np.asarray(events[population + "_cell"])
        if (
            times.ndim != 1
            or cells.shape != times.shape
            or cells.dtype.kind not in "iu"
        ):
            raise ValueError("events require matching time and integer-cell vectors")
        if (
            not np.isfinite(times).all()
            or np.any(times < 0)
            or np.any(times >= cfg["duration_ms"])
        ):
            raise ValueError("spike time outside the simulation")
        if np.any(np.diff(times) < 0) or not np.allclose(
            times / cfg["dt_ms"], np.rint(times / cfg["dt_ms"]), rtol=0, atol=1e-8
        ):
            raise ValueError(
                "spike times must be ordered and lie on the simulation grid"
            )
        if np.any(cells < 0) or np.any(cells >= cfg["n_" + population.lower()]):
            raise ValueError("spike cell outside its population")
        if len(times) > 1:
            same_time = times[1:] == times[:-1]
            if np.any(same_time & (cells[1:] <= cells[:-1])):
                raise ValueError(
                    "simultaneous events must contain distinct sorted cells"
                )


def measure(events, cfg):
    validate_events(events, cfg)
    rows = []
    for label, start, end in windows(cfg):
        row = dict(window=label, start_ms=start, end_ms=end)
        for population in ("E", "I"):
            times = events[population + "_time_ms"]
            count = int(((times >= start) & (times < end)).sum())
            row[population + "_spikes"] = count
            row[population + "_rate_hz"] = (
                count / cfg["n_" + population.lower()] / ((end - start) / 1000)
            )
        rows.append(row)
    start, end = windows(cfg)[-1][1:]
    analysis = cfg["analysis"]
    width = analysis["population_bin_ms"]
    edges = np.arange(start, end + width / 2, width)
    cycles = {}
    for population in ("E", "I"):
        counts, _ = np.histogram(events[population + "_time_ms"], bins=edges)
        peaks, _ = find_peaks(
            counts,
            distance=max(1, round(analysis["peak_separation_ms"] / width)),
            prominence=counts.max() * analysis["peak_prominence_fraction"],
        )
        times = edges[peaks] + width / 2
        period = float(np.median(np.diff(times))) if len(times) > 1 else None
        cycles[population] = dict(
            peak_times_ms=times.tolist(),
            peak_count=len(times),
            median_cycle_ms=period,
            frequency_hz=1000 / period if period is not None else None,
        )
    return dict(
        windows=rows,
        hold_cycles=cycles,
        interpretation="Population oscillations measured; AI classification and PING mechanism untested",
    )
