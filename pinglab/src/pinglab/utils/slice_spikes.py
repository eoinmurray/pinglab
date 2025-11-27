
from pinglab.types import Spikes


def slice_spikes(spikes: Spikes, start_time, stop_time):
    """
    Slice spikes to only include those within the specified time window.

    Args:
        spikes: An object with 'times', 'ids', and 'types' attributes (lists or arrays).
        start_time: Start time of the slice (inclusive).
        stop_time: Stop time of the slice (exclusive).

    Returns:
        A new spikes object containing only the spikes within the time window.
    """
    mask = (spikes.times >= start_time) & (spikes.times < stop_time)
    sliced_spikes = type(spikes)(
        times=spikes.times[mask], ids=spikes.ids[mask], types=spikes.types[mask] if spikes.types is not None else None
    )
    return sliced_spikes
