"""Native runtime initialization carrying only private inhibitory prehistory."""

import numpy as np
import torch
from scipy.signal import lfilter
from snnlab.sim import (
    GraphRuntimeState,
    runtime_state_compatibility,
    runtime_state_signature,
)


def filtered_events(events, cfg):
    delayed = np.vstack([np.zeros((1, events.shape[1])), events[:-1]])
    return lfilter(
        [1.0], [1.0, -np.exp(-cfg["dt_ms"] / cfg["tau_gaba_ms"])], delayed, axis=0
    )


def initial_state(model, prehistory, amplitudes, cfg):
    """Fresh cells/readout and empty reductions; only private GABA has a past.

    The last history event remains pending in the native one-step input queue.
    No sensory input or neuronal evolution occurs before the scored trial.
    """
    plan = model.plan
    history_steps = round(cfg["prehistory_ms"] / cfg["dt_ms"])
    if prehistory.shape != (history_steps, cfg["n_e"]) or not history_steps:
        raise ValueError("Expected nonempty configured inhibitory prehistory")
    pops = {p["id"]: p for p in plan.populations}
    voltages = {
        name: torch.full(
            (1, p["size"]), 0.0 if name == "readout" else cfg["initial_voltage_mv"]
        )
        for name, p in pops.items()
    }
    conductances = {
        p.id: torch.zeros((1, pops[p.target.partition(".")[0]]["size"]))
        for p in plan.projections
    }
    conductances["artificial_to_E"] = torch.from_numpy(
        (filtered_events(prehistory, cfg)[-1] * amplitudes).astype(np.float32)[None, :]
    )
    return GraphRuntimeState(
        signature=runtime_state_signature(plan),
        compatibility=runtime_state_compatibility(plan),
        completed_steps=0,
        voltages=voltages,
        refractory={
            name: torch.zeros_like(v, dtype=torch.long) for name, v in voltages.items()
        },
        conductances=conductances,
        population_histories={
            name: torch.zeros(
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
                    1,
                    row["size"],
                )
            )
            for name, row in pops.items()
        },
        input_histories={
            "artificial": torch.from_numpy(prehistory[-1:, None, :]).float()
        },
    )
