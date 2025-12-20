from __future__ import annotations

from .hotloop import hotloop_burnin


def tune_I_E_for_target_rate(
    *,
    config,
    target_rate_E: float,
    added_current_E: dict[int, float],
    seed: int,
    burnin_ms: float = 300.0,
    max_iters: int = 10,
    eta: float = 0.05,
    tol: float = 1.0,
) -> tuple[float, list[dict]]:
    """
    Iteratively adjust I_E until mean_rate_E matches target_rate_E.

    Uses a simple proportional controller: I_E <- I_E + eta * (target - measured).
    Runs short burn-in simulations to measure rate, avoiding full simulation cost.

    Args:
        config: Experiment config (LocalConfig). Will be copied internally.
        target_rate_E: Desired mean E firing rate (Hz).
        added_current_E: Image current dict {neuron_id: amplitude}.
        seed: Random seed for simulations.
        burnin_ms: Duration of each burn-in simulation (ms).
        max_iters: Maximum tuning iterations.
        eta: Learning rate for I_E updates (nA / Hz).
        tol: Convergence tolerance (Hz). Stop if |error| < tol.

    Returns:
        I_E_tuned: The tuned I_E value that yields approximately target_rate_E.
        history: List of dicts with {I_E, rate_E, error} for each iteration.
    """
    I_E = config.default_inputs.I_E
    history = []

    for i in range(max_iters):
        rate_E = hotloop_burnin(
            config=config,
            burnin_ms=burnin_ms,
            added_current_E=added_current_E,
            I_E_override=I_E,
            seed=seed + i,  # vary seed each iteration for robustness
        )

        error = target_rate_E - rate_E
        history.append({"iter": i, "I_E": I_E, "rate_E": rate_E, "error": error})

        if abs(error) < tol:
            break

        I_E = I_E + eta * error

    return I_E, history
