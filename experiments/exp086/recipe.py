"""Fixed exp086 recipe, reusing exp085 definitions without executing exp085."""

import numpy as np
import torch
from experiments.exp085 import (
    COUPLING_DELAY_MS,
    DT_MS,
    INPUT_RATE_A_HZ,
    INPUT_RATE_B_HZ,
    INPUT_SEEDS,
    N_E,
    N_I,
    N_INPUT,
    NETWORK_SEED,
    author_network,
    poisson_input,
)
from experiments.exp085 import PING_GROUPS as PING_GROUPS

SLUG = "exp086"
STATUS = "draft"

T_MS = 5_000.0
COUPLING_ONSET_MS = 500.0
ANALYSIS_START_MS = 300.0
DISPLAY_WINDOW_MS = 1_500.0
RATE_SMOOTH_MS = 1.0
VELOCITY_SMOOTH_MS = 8.0
PHASE_BINS = 24
K_VALUES = np.asarray([0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0])

SCALE = {
    "status": STATUS,
    "completed_methods": [1, 2, 3],
    "dt_ms": DT_MS,
    "t_ms": T_MS,
    "coupling_onset_ms": COUPLING_ONSET_MS,
    "analysis_start_ms_after_coupling": ANALYSIS_START_MS,
    "input_rate_a_hz": INPUT_RATE_A_HZ,
    "input_rate_b_hz": INPUT_RATE_B_HZ,
    "coupling_delay_ms": COUPLING_DELAY_MS,
    "k_values_us": K_VALUES.tolist(),
    "same_k_for_e_to_e_and_e_to_i": True,
    "trajectories_per_k": 1,
}


def make_inputs() -> dict[str, torch.Tensor]:
    """Generate the one fixed input realization shared by every branch."""
    steps = round(T_MS / DT_MS)
    return {
        f"drive_A_{INPUT_RATE_A_HZ:g}_Hz": poisson_input(
            rate_hz=INPUT_RATE_A_HZ,
            seed=INPUT_SEEDS[0],
            steps=steps,
        ),
        f"drive_B_{INPUT_RATE_B_HZ:g}_Hz": poisson_input(
            rate_hz=INPUT_RATE_B_HZ,
            seed=INPUT_SEEDS[1],
            steps=steps,
        ),
    }


MEASUREMENT = {
    "schema": "exp086.measurement/v1",
    "analysis_start_ms_after_coupling": ANALYSIS_START_MS,
    "rate_smooth_ms": RATE_SMOOTH_MS,
    "velocity_smooth_ms": VELOCITY_SMOOTH_MS,
    "volley_distance_ms": 15.0,
    "volley_prominence_fraction": 0.1,
    "phase_bins": PHASE_BINS,
    "slips": "floor(abs(net_phase_change_cycles) + 1e-9)",
    "frequency_summary": "all detected suffix volleys, before phase-valid mask",
    "selection": "concentration * peak_to_mean * max(slowing, 0) * exp(-alignment)",
    "selection_candidates": "0 < K < max(K), at least two net slips; first maximum",
}
FIGURES = (
    "network.svg",
    "uncoupled.png",
    "coupling_regimes_measured.png",
    "intermittent_attraction_measured.png",
)
ARRAY_KEYS = (
    "time_ms",
    "rate_e_a",
    "rate_i_a",
    "rate_e_b",
    "rate_i_b",
    "peaks_a",
    "peaks_b",
    "wrapped_phase",
    "unwrapped_phase",
    "relative_velocity_rad_s",
    "relative_velocity_smoothed_rad_s",
    "phase_bin_centres",
    "phase_density",
    "mean_velocity_by_phase",
)


def label(k):
    return f"k_{k:.3f}".replace(".", "p")


def branches():
    return [{"k": float(k), "label": label(k)} for k in K_VALUES]


def configuration():
    return {
        "schema": "exp086.recipe/v1",
        **SCALE,
        "input_seeds": list(INPUT_SEEDS),
        "network_seed": NETWORK_SEED,
        "n_input": N_INPUT,
        "n_e": N_E,
        "n_i": N_I,
        "graphs": {
            label(k): author_network(k_ee=float(k), k_ei=float(k)).manifest[
                "graph_digest"
            ]
            for k in K_VALUES
        },
        "shared_state": "uncoupled prefix; detached copy per branch",
    }
