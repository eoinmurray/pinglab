"""Spike data slicing utilities."""

from pinglab.backends.types import Spikes


def slice_spikes(spikes: Spikes, start_time: float, stop_time: float) -> Spikes:
    """
    Slice spikes to only include those within the specified time window.

    Parameters:
        spikes: Spikes object with times, ids, and optional types arrays
        start_time: Start time of the slice in ms (inclusive)
        stop_time: Stop time of the slice in ms (exclusive)

    Returns:
        New Spikes object containing only spikes within [start_time, stop_time)

    Raises:
        ValueError: If start_time >= stop_time
    """
    if start_time >= stop_time:
        raise ValueError(f"start_time ({start_time}) must be < stop_time ({stop_time})")

    mask = (spikes.times >= start_time) & (spikes.times < stop_time)
    return Spikes(
        times=spikes.times[mask],
        ids=spikes.ids[mask],
        types=spikes.types[mask] if spikes.types is not None else None,
    )
