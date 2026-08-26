"""Data-only declarations for reproducible simulation inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .core import Population, Signal, Spec


def _id(value: Population | Signal | str) -> str:
    return value if isinstance(value, str) else value.id


@dataclass(frozen=True)
class ShotNoise:
    rate_hz: float
    amplitude: float
    tau_ms: float

    def json(self) -> dict[str, Any]:
        return {
            "kind": "shot_noise",
            "rate_hz": float(self.rate_hz),
            "amplitude": float(self.amplitude),
            "tau_ms": float(self.tau_ms),
        }


@dataclass(frozen=True)
class GlobalShotNoise(ShotNoise):
    def json(self) -> dict[str, Any]:
        return {**super().json(), "kind": "global_shot_noise"}


@dataclass(frozen=True)
class CellDistribution:
    """Reproducible multiplicative cell heterogeneity laws."""

    rate: Spec = field(default_factory=lambda: Spec("constant", {"value": 1.0}))
    amplitude: Spec = field(default_factory=lambda: Spec("constant", {"value": 1.0}))

    def json(self) -> dict[str, Any]:
        return {"rate": self.rate.json(), "amplitude": self.amplitude.json()}


@dataclass(frozen=True)
class BackgroundChannel:
    private: ShotNoise
    shared: GlobalShotNoise
    heterogeneity: CellDistribution = field(default_factory=CellDistribution)

    def json(self) -> dict[str, Any]:
        return {
            "private": self.private.json(),
            "shared": self.shared.json(),
            "heterogeneity": self.heterogeneity.json(),
        }


@dataclass(frozen=True)
class ConductanceBackground:
    target: Population | str
    excitatory: BackgroundChannel
    inhibitory: BackgroundChannel

    def json(self) -> dict[str, Any]:
        return {
            "target": _id(self.target),
            "excitatory": self.excitatory.json(),
            "inhibitory": self.inhibitory.json(),
        }


@dataclass(frozen=True)
class StructuredPoisson:
    input: Signal | str
    rate_hz: float

    def json(self) -> dict[str, Any]:
        return {
            "kind": "structured_poisson",
            "input": _id(self.input),
            "rate_hz": float(self.rate_hz),
        }


@dataclass(frozen=True)
class ConductanceSchedule:
    targets: Sequence[Population | str]
    start_ms: float
    end_ms: float
    start_scale: float = 1.0
    end_scale: float = 1.0
    shape: str = "smoothstep"

    def json(self) -> dict[str, Any]:
        return {
            "kind": "conductance_schedule",
            "targets": [_id(target) for target in self.targets],
            "shape": self.shape,
            "start_ms": float(self.start_ms),
            "end_ms": float(self.end_ms),
            "start_scale": float(self.start_scale),
            "end_scale": float(self.end_scale),
        }


@dataclass(frozen=True)
class SimulationSpec:
    spike_sources: Sequence[StructuredPoisson] = ()
    backgrounds: Sequence[ConductanceBackground] = ()
    modulation: Sequence[ConductanceSchedule] = ()


def simulation_dict(spec: SimulationSpec, graph_digest: str) -> dict[str, Any]:
    return {
        "schema": "snnlang.simulation/v1",
        "graph_digest": graph_digest,
        "seed_derivation": "request_seed+stable_channel_index",
        "spike_sources": [source.json() for source in spec.spike_sources],
        "backgrounds": [background.json() for background in spec.backgrounds],
        "modulation": [schedule.json() for schedule in spec.modulation],
    }


def validate_simulation(graph: dict[str, Any], recipe: dict[str, Any]) -> None:
    if recipe.get("schema") != "snnlang.simulation/v1":
        raise ValueError("unsupported simulation schema")
    input_ids = {row["id"] + ".value" for row in graph.get("inputs", [])}
    population_ids = {
        row["id"] for row in graph.get("populations", []) if row.get("spiking")
    }
    for source in recipe.get("spike_sources", []):
        if source.get("input") not in input_ids or float(source.get("rate_hz", 0)) < 0:
            raise ValueError(
                "simulation spike source must reference an input and use rate_hz >= 0"
            )
    targets = []
    for background in recipe.get("backgrounds", []):
        target = background.get("target")
        targets.append(target)
        if target not in population_ids:
            raise ValueError("conductance background must target a spiking population")
        for polarity in ("excitatory", "inhibitory"):
            channel = background.get(polarity, {})
            for ownership, expected in (
                ("private", "shot_noise"),
                ("shared", "global_shot_noise"),
            ):
                noise = channel.get(ownership, {})
                if noise.get("kind") != expected:
                    raise ValueError(
                        f"{polarity} background requires one {ownership} stream"
                    )
                if (
                    any(
                        float(noise.get(key, 0)) < 0 for key in ("rate_hz", "amplitude")
                    )
                    or float(noise.get("tau_ms", 0)) <= 0
                ):
                    raise ValueError(
                        "background rates/amplitudes must be non-negative and tau_ms positive"
                    )
    if len(targets) != len(set(targets)):
        raise ValueError("simulation may declare only one background per population")
    for schedule in recipe.get("modulation", []):
        if (
            schedule.get("shape") != "smoothstep"
            or float(schedule.get("start_ms", -1)) < 0
            or float(schedule.get("end_ms", 0)) <= float(schedule.get("start_ms", -1))
        ):
            raise ValueError(
                "conductance schedule requires smoothstep and 0 <= start_ms < end_ms"
            )
        if not set(schedule.get("targets", ())) <= population_ids:
            raise ValueError("conductance schedule references an unknown population")
