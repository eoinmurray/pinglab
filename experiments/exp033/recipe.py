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
CELL_E = {"tau_m": TAU_E_MS, "g_L": 0.05, "tau_ref": 1.2}
CELL_I = {"tau_m": TAU_I_MS, "g_L": 0.10, "tau_ref": 0.6}
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


def configuration(*, version=2):
    """Versioned gain parameters; historical definitions never use live defaults."""
    if version not in (1, 2):
        raise ValueError("unsupported exp033 recipe version")
    return {
        "schema": f"exp033.recipe/v{version}",
        **({"gain_integral": "erfcx_for_negative_arguments"} if version == 2 else {}),
        "profile": "production",
        "tau_E_ms": 20.0,
        "tau_I_ms": 5.0,
        "tau_AMPA_ms": 2.0,
        "tau_GABA_ms": 6.0,
        "W_tilde_EI": 1.0,
        "W_tilde_IE": 2.0,
        "dV_inh_mV": 15.0,
        "dV_exc_mV": 65.0,
        "sigma_V_mV": 4.0,
        "cell_E": {"tau_m": 20.0, "g_L": 0.05, "tau_ref": 3.0 if version == 1 else 1.2},
        "cell_I": {"tau_m": 5.0, "g_L": 0.1, "tau_ref": 1.5 if version == 1 else 0.6},
        "rest_mV": -65.0,
        "threshold_mV": -50.0,
        "reset_mV": -65.0,
        "drive_grid": [0.0, 4.0, 401],
        "sigma_grid_mV": [3.0, 4.0, 5.0, 6.0],
        "sensitivity_grid": [0.0, 1.2, 121],
        "convergence_grid": [0.0, 1.2, 241],
        "tau_grid_ms": [4.5, 6.0, 9.0, 12.0, 18.0, 27.0],
        "jacobian_eps": 1e-06,
        "hopf_refinement": {"method": "brentq", "xtol": 1e-10, "rtol": 1e-12},
        "hysteresis": {
            "span_nA": [-0.1, 0.55],
            "points": 25,
            "t_max_ms": 2000.0,
            "settle_start_ms": 1500.0,
            "threshold": 0.0001,
            "rtol": 1e-07,
            "atol": 1e-10,
            "max_step": 1.0,
        },
        "cycle": {
            "offset_nA": 0.4,
            "t_max_ms": 700.0,
            "rtol": 1e-09,
            "atol": 1e-12,
            "max_step": 0.25,
            "waveform_periods": 3,
            "waveform_points": 1500,
            "phase_periods": 4,
            "phase_points": 2000,
        },
        "comparison": {"offset_nA": 1.0, "t_max_ms": 300.0, "measure_after_ms": 150.0},
        "ladder": {"drive_nA": 1.0, "t_max_ms": 400.0},
        "comparison_and_ladder_solver": {"rtol": 1e-08, "atol": 1e-11, "max_step": 0.5},
        "solver": "LSODA",
        "upstream_aggregation": "median_across_three_seeds",
    }


def validate(cfg):
    from pingstore.contracts import PingstoreError

    if cfg not in (configuration(version=1), configuration(version=2)):
        raise PingstoreError("inconsistent exp033 recipe")
    return cfg
