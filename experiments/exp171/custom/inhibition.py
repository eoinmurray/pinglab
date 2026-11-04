"""Initialize native batched trajectories with private inhibitory history only."""

import numpy as np
import torch
from scipy.signal import lfilter
from snnlab.sim import (
    GraphRuntimeState,
    runtime_state_compatibility,
    runtime_state_signature,
)


def history_state(model, history):
    """Amplitude-scaled history through the frozen identity GABA projection.

    This filters external events, without evolving neurons or readout. The final
    event remains queued for delivery at the first scored step. All other state
    is freshly initialized, including time and reductions.
    """
    if history.ndim != 3 or not len(history):
        raise ValueError("Expected nonempty time × batch × private-channel history")
    plan = model.plan
    private = next(p for p in plan.projections if p.id == "artificial_to_E")
    if private.delay_steps != 1:
        raise ValueError("Private inhibitory initialization requires one-step delay")
    rows = {p["id"]: p for p in plan.populations}
    if history.shape[2] != rows["E"]["size"]:
        raise ValueError("Private history channel count differs from E population")
    parameter = next(iter(model.parameter_map().values()))
    batch = history.shape[1]

    def zeros(shape):
        return torch.zeros(shape, device=parameter.device, dtype=parameter.dtype)

    def tensor(array):
        return torch.as_tensor(array, device=parameter.device, dtype=parameter.dtype)

    tau = next(
        p["synapse"]["tau"]["value"]
        for p in plan.graph["projections"]
        if p["id"] == private.id
    )
    delayed = np.concatenate([np.zeros_like(history[:1]), history[:-1]], axis=0)
    conductance = lfilter([1.0], [1.0, -np.exp(-plan.dt_ms / tau)], delayed, axis=0)[-1]
    voltages = {
        name: zeros((batch, p["size"]))
        + (
            0.0
            if p["neuron"]["kind"] == "leaky_integrator"
            else p["neuron"].get("initial_voltage_mv", -65.0)
        )
        for name, p in rows.items()
    }
    conductances = {
        p.id: zeros((batch, rows[p.target.partition(".")[0]]["size"]))
        for p in plan.projections
    }
    conductances[private.id] = tensor(conductance)
    return GraphRuntimeState(
        signature=runtime_state_signature(plan, projection_mode=model.projection_mode),
        compatibility=runtime_state_compatibility(
            plan, projection_mode=model.projection_mode
        ),
        completed_steps=0,
        voltages=voltages,
        refractory={
            name: torch.zeros_like(v, dtype=torch.long) for name, v in voltages.items()
        },
        conductances=conductances,
        population_histories={
            name: zeros(
                (
                    max(
                        (
                            p.delay_steps
                            for p in plan.projections
                            if p.root_source.startswith(name + ".")
                            and p.delay_steps > 0
                        ),
                        default=1,
                    ),
                    batch,
                    row["size"],
                )
            )
            for name, row in rows.items()
        },
        input_histories={"artificial": tensor(history[-1:])},
    )
