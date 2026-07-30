"""Small serialisable operation vocabulary."""

from __future__ import annotations

from .core import Constant, Signal


def linear(source: Signal, *, size: int, name: str, trainable: bool = True) -> Signal:
    net = source.network
    weight = net.parameter(
        f"{name}.weight",
        shape=(size, source.shape[-1]),
        initializer=Constant(0.0),
        constraint=None,
    )
    return net.operation(
        "linear",
        source,
        name=name,
        shape=(*source.shape[:-1], size),
        unit=source.unit,
        signal_type="continuous",
        parameters=(weight,),
        size=size,
        trainable=trainable,
    )


def reduce(
    source: Signal,
    *,
    operation: str,
    over: str,
    name: str,
    window: str = "full",
    mask: Signal | None = None,
) -> Signal:
    if over != "time" or len(source.shape) < 2:
        raise ValueError("v1 reductions support only an explicit time axis")
    sources = [source] + ([mask] if mask else [])
    return source.network.operation(
        f"reduce_{operation}",
        sources,
        name=name,
        shape=(source.shape[0], *source.shape[2:]),
        unit=source.unit,
        window=window,
        mask=mask.id if mask else None,
    )


def divide(source: Signal, denominator: Signal, *, name: str, unit: str) -> Signal:
    return source.network.operation(
        "divide",
        [source, denominator],
        name=name,
        shape=source.shape,
        unit=unit,
    )
