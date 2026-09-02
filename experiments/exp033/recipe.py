"""Committed mean-field recipe; no storage or execution on import."""

from experiments.helpers.operating_point import TAU_GABA_GAMMA_MS

SLUG = "exp033"
# ── Timescales (ms) ───────────────────────────────────────────────────
TAU_E_MS = 20.0  # E membrane (= CELL_E tau_m)
TAU_I_MS = 5.0  # I membrane (= CELL_I tau_m)
TAU_AMPA_MS = 2.0
# Canonical operating point (was 9.0); single source of truth in helpers so the
# mean-field analysis tracks the spiking collection's τ_GABA.
TAU_GABA_MS = TAU_GABA_GAMMA_MS

# ── COBANet-grounded gain with a free effective-noise scale ───────────
# Couplings are the ei-strength values, fan-in normalised so the lumped
# W̃ = w·N = s (E→I) and r·s (I→E); σ_V is not derived from COBANet.
E_L_MV, V_TH_MV, V_RESET_MV = -65.0, -50.0, -65.0
CELL_E = {"tau_m": TAU_E_MS, "g_L": 0.05, "tau_ref": 3.0}
CELL_I = {"tau_m": TAU_I_MS, "g_L": 0.10, "tau_ref": 1.5}
DV_INH_MV, DV_EXC_MV = 15.0, 65.0  # |V_rest − E_rev| driving forces
WT_EI, WT_IE = 1.0, 2.0  # lumped couplings (µS): s and r·s
SIGMA_V_MV = 4.0  # membrane-noise std
SIGMA_V_GRID_MV = (3.0, 4.0, 5.0, 6.0)
HYSTERESIS_SPAN_NA = (-0.1, 0.55)
HYSTERESIS_POINTS = 25
LIMIT_CYCLE_OFFSET_NA = 0.4


FIGURES = (
    "bifurcation_compound.svg",
    "sigma_sensitivity.svg",
    "limit_cycle.svg",
    "timeseries.svg",
    "phase_planes.svg",
    "reduction_ladder.svg",
)
TAU_GRID_MS = (4.5, 6.0, 9.0, 12.0, 18.0, 27.0)


def configuration():
    return {
        "schema": "exp033.recipe/v1",
        "profile": "production",
        "tau_E_ms": TAU_E_MS,
        "tau_I_ms": TAU_I_MS,
        "tau_AMPA_ms": TAU_AMPA_MS,
        "tau_GABA_ms": TAU_GABA_MS,
        "W_tilde_EI": WT_EI,
        "W_tilde_IE": WT_IE,
        "dV_inh_mV": DV_INH_MV,
        "dV_exc_mV": DV_EXC_MV,
        "sigma_V_mV": SIGMA_V_MV,
        "cell_E": dict(CELL_E),
        "cell_I": dict(CELL_I),
        "rest_mV": E_L_MV,
        "threshold_mV": V_TH_MV,
        "reset_mV": V_RESET_MV,
        "drive_grid": [0.0, 4.0, 401],
        "sigma_grid_mV": list(SIGMA_V_GRID_MV),
        "sensitivity_grid": [0.0, 1.2, 121],
        "convergence_grid": [0.0, 1.2, 241],
        "tau_grid_ms": list(TAU_GRID_MS),
        "jacobian_eps": 1e-6,
        "hopf_refinement": {"method": "brentq", "xtol": 1e-10, "rtol": 1e-12},
        "hysteresis": {
            "span_nA": list(HYSTERESIS_SPAN_NA),
            "points": HYSTERESIS_POINTS,
            "t_max_ms": 2000.0,
            "settle_start_ms": 1500.0,
            "threshold": 1e-4,
            "rtol": 1e-7,
            "atol": 1e-10,
            "max_step": 1.0,
        },
        "cycle": {
            "offset_nA": LIMIT_CYCLE_OFFSET_NA,
            "t_max_ms": 700.0,
            "rtol": 1e-9,
            "atol": 1e-12,
            "max_step": 0.25,
            "waveform_periods": 3,
            "waveform_points": 1500,
            "phase_periods": 4,
            "phase_points": 2000,
        },
        "comparison": {"offset_nA": 1.0, "t_max_ms": 300.0, "measure_after_ms": 150.0},
        "ladder": {"drive_nA": 1.0, "t_max_ms": 400.0},
        "comparison_and_ladder_solver": {"rtol": 1e-8, "atol": 1e-11, "max_step": 0.5},
        "solver": "LSODA",
        "upstream_aggregation": "median_across_three_seeds",
    }
