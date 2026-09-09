"""Physical parameters and retained timing conventions for gamma-gated sparsity.

The adopted hidden-neuron model uses 1.2/0.6-ms E/I refractory holds and normally
6-ms GABA decay. Timestep sweeps must represent both holds exactly; generic
snnsim defaults are independent of this collection boundary.

F_GAMMA_HZ is a retained 43.95-Hz reference from an earlier spiking analysis.
It fixes the approximately 22.8-ms clock windows used by existing inhibitory
replay experiments. It is not the current exp041 spectral estimate, the exp033
Hopf frequency, or a detected-cycle boundary. Preserve this constant when
reusing those experiments; changing it would change the intervention and
require a separately versioned protocol and new evidence.
"""

from __future__ import annotations

# Canonical inhibitory (GABA) synaptic decay — the collection's operating point.
# Passed to the CLI as --tau-gaba when training every non-sweep exp022 cell.
TAU_GABA_GAMMA_MS: float = 6.0

# Retained clock convention for existing replay protocols; not a live measurement.
F_GAMMA_HZ: float = 43.95


# The collection's spiking model. Generic snnsim defaults remain independent.
REFRACTORY_E_MS: float = 1.2
REFRACTORY_I_MS: float = 0.6
REFRACTORY_POLICY = "exact"


def refractory_configuration() -> dict:
    """Explicit parameters for new collection execution, including old checkpoints."""
    return {
        "refractory_e_ms": REFRACTORY_E_MS,
        "refractory_i_ms": REFRACTORY_I_MS,
        "refractory_policy": REFRACTORY_POLICY,
    }


def refractory_args() -> list[str]:
    """Override absent retained config fields at the collection boundary."""
    return [
        "--refractory-e-ms",
        str(REFRACTORY_E_MS),
        "--refractory-i-ms",
        str(REFRACTORY_I_MS),
        "--refractory-policy",
        REFRACTORY_POLICY,
    ]


def duration_steps(duration_ms: float, dt_ms: float) -> int:
    """Whole trial steps with tolerance for binary representations of integers.

    This collection contract matches snnsim's public execution convention without
    importing simulator implementation across the experiment/tool boundary.
    """
    import math

    if not math.isfinite(duration_ms) or duration_ms < 0:
        raise ValueError("duration_ms must be finite and nonnegative")
    if not math.isfinite(dt_ms) or dt_ms <= 0:
        raise ValueError("dt_ms must be finite and positive")
    ratio = duration_ms / dt_ms
    nearest = round(ratio)
    return (
        nearest
        if math.isclose(ratio, nearest, rel_tol=1e-12, abs_tol=1e-12)
        else math.floor(ratio)
    )


def duration_configuration(duration_ms: float, dt_ms: float) -> dict:
    steps = duration_steps(duration_ms, dt_ms)
    return {
        "nominal_duration_ms": float(duration_ms),
        "duration_steps": steps,
        "realized_duration_ms": steps * float(dt_ms),
    }


def refractory_execution_configuration(dt_ms: float) -> dict:
    """Expected metadata of an execution of the exact collection model."""
    import math

    result = refractory_configuration()
    for population, duration in (("e", REFRACTORY_E_MS), ("i", REFRACTORY_I_MS)):
        steps = round(duration / dt_ms)
        if steps < 1 or not math.isclose(
            steps * dt_ms, duration, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(
                "collection refractory duration is not exactly representable"
            )
        result[f"refractory_{population}_steps"] = steps
        result[f"realized_refractory_{population}_ms"] = steps * dt_ms
    return result
