"""Frozen design for twenty fast snnsim--Brian2 comparisons."""

from __future__ import annotations

SLUG = "exp111"
COLLECTION = "gamma-gated-sparsity"
SOURCE_EXPERIMENT = "exp022"
MASTER_SEED = 111_2026

DT_MS = 0.1
REDUCED_DURATION_MS = 500.0
PRODUCTION_DURATION_MS = 200.0
N_E_REDUCED = 20
N_I_REDUCED = 5

TESTS = (
    ("lif-passive", "Passive conductance LIF", "core"),
    ("lif-spiking", "Threshold, reset and refractory", "core"),
    ("synapse-impulses", "AMPA and GABA impulse responses", "core"),
    ("event-causality", "Event scheduling through the E-I loop", "core"),
    ("projection-scaling", "Projection and weight scaling", "core"),
    ("matched-loop", "Matched-drive loop-off and loop-on activity", "exp023"),
    ("input-response", "Input-response curves", "exp023"),
    ("coupling-onset", "Coupling-plane oscillation onset", "exp033/exp054"),
    ("uncoupled-nulls", "Private and shared uncoupled nulls", "exp054"),
    ("gaba-frequency", "GABA timescale and gamma frequency", "exp041"),
    ("cycle-participation", "Excitatory spikes per inhibitory cycle", "exp046"),
    ("checkpoint-replay", "Selected and final checkpoint replay", "exp024"),
    ("coba-ping-endpoints", "COBA and PING endpoint activity", "exp025"),
    ("loop-transfer", "Post-training loop-strength transfer", "exp038"),
    ("spike-perturbations", "Shared hidden-spike perturbation inputs", "exp037"),
    ("inhibitory-jitter", "Shared inhibitory-jitter inputs", "exp042"),
    ("timestep-robustness", "Integration-timestep robustness", "exp044"),
    ("recurrent-training", "Frozen and trained recurrent weights", "exp049"),
    ("stream-resets", "Continuous hidden state and readout resets", "exp082"),
    ("duration-rate", "Continuous-stream duration and input rate", "exp082"),
)

FIGURES = tuple(f"{identifier}.svg" for identifier, _, _ in TESTS)


def configuration() -> dict:
    return {
        "schema": "exp111.recipe/v2",
        "master_seed": MASTER_SEED,
        "dt_ms": DT_MS,
        "reduced_duration_ms": REDUCED_DURATION_MS,
        "production_duration_ms": PRODUCTION_DURATION_MS,
        "reduced_population": {"n_e": N_E_REDUCED, "n_i": N_I_REDUCED},
        "tests": [
            {"id": identifier, "title": title, "source_experiment": source}
            for identifier, title, source in TESTS
        ],
    }
