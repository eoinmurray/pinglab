"""The retained 18-network cycle-counting recipe; no storage side effects."""

from experiments.exp041.recipe import (
    ANALYSIS_PURPOSE,
    CHECKPOINT_POLICY,
    CHECKPOINT_ROLE,
    EVAL_MAX_SAMPLES,
    SEEDS,
    SMOKE_MAX_SAMPLES,
    TAU_GABA_SWEEP,
    cell_name,
)

SLUG = "exp046"
REFRACTORY_E_MS = 1.2
REFRACTORY_I_MS = 0.6
REFRACTORY_POLICY = "exact"
__all__ = ["ANALYSIS_PURPOSE", "CHECKPOINT_ROLE", "cell_name"]
TAU_GABA_SWEEP_MS = TAU_GABA_SWEEP
FIGURES = tuple(
    name + "." + ext
    for name in ("spikes_per_cycle_distribution", "ceiling_vs_fgamma")
    for ext in ("svg", "pdf")
)
FIGURES += tuple(
    "spikes_per_cycle_distribution_equal_network." + ext
    for ext in ("svg", "pdf")
)


def refractory_configuration() -> dict:
    return {
        "refractory_e_ms": REFRACTORY_E_MS,
        "refractory_i_ms": REFRACTORY_I_MS,
        "refractory_policy": REFRACTORY_POLICY,
    }




def configuration(*, smoke=False, version=2):
    if version not in (1, 2):
        raise ValueError("unsupported exp046 recipe version")
    return {
        "schema": f"exp046.recipe/v{version}",
        **(refractory_configuration() if version >= 2 else {}),
        "profile": "smoke" if smoke else "production",
        "tau_gaba_sweep_ms": list(TAU_GABA_SWEEP_MS),
        "seeds": list(SEEDS),
        "evaluation_samples": SMOKE_MAX_SAMPLES if smoke else EVAL_MAX_SAMPLES,
        "checkpoint_policy": CHECKPOINT_POLICY,
    }




BIOPHYSICS = {
    "capacitance_e_nf": 1.0, "capacitance_i_nf": 0.5,
    "leak_e_us": 0.05, "leak_i_us": 0.10,
    "resting_mv": -65.0, "threshold_mv": -50.0, "reset_mv": -65.0,
    "readout_tau_ms": 2.0, "readout_threshold": 1.0,
}


def author_network(training, *, observables=()):
    from experiments.helpers.checkpoint_graph import author_network as author

    return author(
        SLUG, training, BIOPHYSICS, refractory_configuration(), observables=observables
    )


def inference_request(training, cfg):
    return {
        "checkpoint_role": CHECKPOINT_ROLE,
        "input": "dataset",
        "t_ms": training["t_ms"],
        "input_rate_hz": training["input_rate"],
        "samples": cfg["evaluation_samples"],
        "batch_size": 64,
        "subset_seed": 42,
        "encoder_seed": 20260415,
        "observables": [],
        "products": ["rasters", "rates"],
    }
