"""Deterministically realize authenticated SNNLang simulation recipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class RealizedInputs:
    input_spikes: torch.Tensor | None
    excitatory_e: torch.Tensor | None
    inhibitory_e: torch.Tensor | None
    excitatory_i: torch.Tensor | None
    inhibitory_i: torch.Tensor | None
    retained: dict[str, np.ndarray]


def _distribution(
    spec: dict[str, Any], size: int, generator: torch.Generator
) -> torch.Tensor:
    kind = spec.get("kind")
    if kind == "constant":
        values = torch.full((size,), float(spec.get("value", 1.0)))
    elif kind in {"normal", "lower_clamped_normal"}:
        values = torch.normal(
            float(spec["mean"]), float(spec["std"]), (size,), generator=generator
        )
        if kind == "lower_clamped_normal":
            values.clamp_(min=0)
    elif kind == "uniform":
        values = torch.empty(size).uniform_(
            float(spec["low"]), float(spec["high"]), generator=generator
        )
    else:
        raise ValueError(f"unsupported cell distribution: {kind!r}")
    return values.clamp(min=0)


def _smoothstep(
    recipe: dict[str, Any], target: str, t_steps: int, dt: float
) -> torch.Tensor:
    scale = torch.ones(t_steps)
    for schedule in recipe.get("modulation", []):
        if target not in schedule.get("targets", []):
            continue
        time = torch.arange(t_steps, dtype=torch.float32) * float(dt)
        u = (
            (time - float(schedule["start_ms"]))
            / (float(schedule["end_ms"]) - float(schedule["start_ms"]))
        ).clamp(0, 1)
        alpha = u * u * (3 - 2 * u)
        scale *= float(schedule.get("start_scale", 1.0)) + alpha * (
            float(schedule.get("end_scale", 1.0))
            - float(schedule.get("start_scale", 1.0))
        )
    return scale


def realize_simulation_inputs(
    recipe: dict[str, Any],
    *,
    seed: int,
    dt: float,
    t_steps: int,
    e_id: str,
    i_id: str,
    n_e: int,
    n_i: int,
    input_size: int | None = None,
) -> RealizedInputs:
    """Generate mutually independent private/shared streams with stable seeds."""
    retained: dict[str, np.ndarray] = {}
    spike_sources = recipe.get("spike_sources", [])
    if len(spike_sources) > 1:
        raise ValueError("legacy execution supports one structured spike source")
    input_spikes = None
    if spike_sources:
        source = spike_sources[0]
        generator = torch.Generator().manual_seed(int(seed))
        p = float(source["rate_hz"]) * float(dt) / 1000.0
        source_size = n_e if input_size is None else int(input_size)
        input_spikes = (
            torch.rand(t_steps, source_size, generator=generator) < p
        ).float()
        retained["input_structured_spikes"] = input_spikes.numpy()

    resolved: dict[tuple[str, str], torch.Tensor] = {}
    channel_index = 1
    sizes = {e_id: n_e, i_id: n_i}
    names = {e_id: "e", i_id: "i"}
    for background in recipe.get("backgrounds", []):
        target = background["target"]
        if target not in sizes:
            raise ValueError(
                f"simulation background targets unsupported population {target!r}"
            )
        size = sizes[target]
        modulation = _smoothstep(recipe, target, t_steps, dt)
        retained[f"input_modulation_{names[target]}"] = modulation.numpy()
        for polarity in ("excitatory", "inhibitory"):
            channel = background[polarity]
            heterogeneity = channel["heterogeneity"]
            private_gen = torch.Generator().manual_seed(int(seed) + channel_index)
            channel_index += 1
            shared_gen = torch.Generator().manual_seed(int(seed) + channel_index)
            channel_index += 1
            rate_scale = _distribution(heterogeneity["rate"], size, private_gen)
            amplitude_scale = _distribution(
                heterogeneity["amplitude"], size, private_gen
            )
            private = channel["private"]
            private_p = (
                float(private["rate_hz"]) * float(dt) / 1000.0 * rate_scale
            ).clamp(max=1)
            private_events = (
                torch.rand(t_steps, size, generator=private_gen)
                < private_p.unsqueeze(0)
            ).float() * (float(private["amplitude"]) * amplitude_scale).unsqueeze(0)
            shared = channel["shared"]
            shared_p = min(1.0, float(shared["rate_hz"]) * float(dt) / 1000.0)
            shared_events = (
                torch.rand(t_steps, 1, generator=shared_gen) < shared_p
            ).float() * float(shared["amplitude"])
            shared_cells = shared_events.expand(-1, size)
            total = private_events + shared_cells
            executed = total * modulation.unsqueeze(1)
            stem = f"input_{polarity}_{names[target]}"
            retained[f"{stem}_private"] = private_events.numpy()
            retained[f"{stem}_shared"] = shared_cells.numpy()
            retained[f"{stem}_total"] = total.numpy()
            retained[f"{stem}_executed"] = executed.numpy()
            retained[f"{stem}_rate_scale"] = rate_scale.numpy()
            retained[f"{stem}_amplitude_scale"] = amplitude_scale.numpy()
            resolved[(target, polarity)] = executed

    return RealizedInputs(
        input_spikes=input_spikes,
        excitatory_e=resolved.get((e_id, "excitatory")),
        inhibitory_e=resolved.get((e_id, "inhibitory")),
        excitatory_i=resolved.get((i_id, "excitatory")),
        inhibitory_i=resolved.get((i_id, "inhibitory")),
        retained=retained,
    )
