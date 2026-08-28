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
__all__ = ["ANALYSIS_PURPOSE", "CHECKPOINT_ROLE", "cell_name"]
TAU_GABA_SWEEP_MS = TAU_GABA_SWEEP
FIGURES = tuple(
    name + "." + ext
    for name in ("spikes_per_cycle_distribution", "ceiling_vs_fgamma")
    for ext in ("svg", "pdf")
)


def configuration(*, smoke=False):
    return {
        "schema": "exp046.recipe/v1",
        "profile": "smoke" if smoke else "production",
        "tau_gaba_sweep_ms": list(TAU_GABA_SWEEP_MS),
        "seeds": list(SEEDS),
        "evaluation_samples": SMOKE_MAX_SAMPLES if smoke else EVAL_MAX_SAMPLES,
        "checkpoint_policy": CHECKPOINT_POLICY,
    }


def inference_args(train, checkpoint, output, *, samples, tau_gaba_ms):
    return [
        "sim",
        "--infer",
        "--device",
        "auto",
        "--load-config",
        str(train / "config.json"),
        "--load-weights",
        str(checkpoint),
        "--tau-gaba",
        str(tau_gaba_ms),
        "--max-samples",
        str(samples),
        "--out-dir",
        str(output),
        "--outputs",
        "rasters",
        "per_cell_rates",
        "--recording-mode",
        "spikes",
        "--output-fields",
        "rate_e_per_cell",
        "e_trial",
        "e_t",
        "e_cell",
        "i_trial",
        "i_t",
        "i_cell",
    ]
