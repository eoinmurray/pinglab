"""Concise readouts expanded into ordinary graph operations."""

from __future__ import annotations

from dataclasses import dataclass

from . import ops
from .core import LeakyIntegrator, Signal, ms


@dataclass(frozen=True)
class Readout:
    signal: Signal
    parameters: tuple = ()

    def __getattr__(self, item):
        return getattr(self.signal, item)

    @property
    def id(self) -> str:
        return self.signal.id


def MeanVoltage(*, source: Signal, classes: int, name: str, tau=20 * ms) -> Readout:
    net = source.network
    with net.group(name):
        projected = ops.linear(source, size=classes, name=f"{name}_projection")
        layer = net.population(
            f"{name}_integrator",
            size=classes,
            neuron=LeakyIntegrator(tau=tau),
            spiking=False,
        )
        drive = net.connect(
            projected,
            layer.excitatory,
            name=f"{name}_drive",
            synapse=LeakyIntegrator(tau=tau),
            connection="feedforward",
        )
        mean = ops.reduce(
            layer.voltage, operation="mean", over="time", name=f"{name}_mean"
        )
    params = tuple(
        dict.fromkeys(
            [p["id"] for p in net.parameters if p["id"].startswith(name)]
            + list(drive.parameter_ids)
        )
    )
    return Readout(mean, params)


def FinalVoltage(*, source: Signal, classes: int, name: str) -> Readout:
    with source.network.group(name):
        projected = ops.linear(source, size=classes, name=f"{name}_projection")
        signal = source.network.operation(
            "select_final",
            projected,
            name=f"{name}_final",
            shape=(projected.shape[0], projected.shape[-1]),
            unit=projected.unit,
        )
    return Readout(signal)


def SpikeCount(*, source: Signal, classes: int, name: str) -> Readout:
    with source.network.group(name):
        projected = ops.linear(source, size=classes, name=f"{name}_projection")
        result = ops.reduce(
            projected, operation="sum", over="time", name=f"{name}_count"
        )
    return Readout(result)


def SpikeRate(
    *,
    source: Signal,
    classes: int,
    name: str,
    duration: float | None = None,
    mask: Signal | None = None,
    window: str = "full",
) -> Readout:
    if duration is None and mask is None:
        raise ValueError("SpikeRate requires a duration or valid-time mask")
    with source.network.group(name):
        projected = ops.linear(source, size=classes, name=f"{name}_projection")
        count = ops.reduce(
            projected,
            operation="sum",
            over="time",
            name=f"{name}_count",
            window=window,
            mask=mask,
        )
        signal = source.network.operation(
            "duration_normalise",
            [count] + ([mask] if mask else []),
            name=f"{name}_rate",
            shape=count.shape,
            unit="Hz",
            duration=duration,
            mask=mask.id if mask else None,
            window=window,
        )
    return Readout(signal)


def CumulativePotential(*, source: Signal, classes: int, name: str) -> Readout:
    with source.network.group(name):
        projected = ops.linear(source, size=classes, name=f"{name}_projection")
        signal = source.network.operation(
            "cumulative_sum",
            projected,
            name=f"{name}_cumulative",
            shape=projected.shape,
            unit=projected.unit,
        )
    return Readout(signal)
