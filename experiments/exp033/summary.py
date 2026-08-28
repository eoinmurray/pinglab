"""Historical scientific summary definitions; no execution."""

from .recipe import (
    CELL_E,
    CELL_I,
    DV_EXC_MV,
    DV_INH_MV,
    SIGMA_V_MV,
    SLUG,
    TAU_AMPA_MS,
    TAU_E_MS,
    TAU_GABA_MS,
    TAU_I_MS,
    WT_EI,
    WT_IE,
)


def summary(
    hopf, criticality, twod, limitcyc, mf_freq, meas_fgamma, hopf3, two_d, sensitivity
):
    summary = {
        "slug": SLUG,
        "config": {
            "tau_E_ms": TAU_E_MS,
            "tau_I_ms": TAU_I_MS,
            "tau_AMPA_ms": TAU_AMPA_MS,
            "tau_GABA_ms": TAU_GABA_MS,
            "W_tilde_EI": WT_EI,
            "W_tilde_IE": WT_IE,
            "dV_inh_mV": DV_INH_MV,
            "dV_exc_mV": DV_EXC_MV,
            "sigma_V_mV": SIGMA_V_MV,
            "cell_E": CELL_E,
            "cell_I": CELL_I,
        },
        "results": {
            "hopf": hopf,
            "criticality": criticality,
            "two_d_vs_four_d": twod,
            "limit_cycle": limitcyc,
            "frequency_vs_tau_gaba": {
                "mean_field": mf_freq,
                "spiking_exp041": meas_fgamma,
            },
            "reductions": {
                "three_d_qss": hopf3,
                "two_d_all_pairs": two_d,
            },
            "sigma_sensitivity": sensitivity,
        },
        "success_criteria": [
            {
                "label": "Reference 4D Hopf in the gamma band",
                "passed": bool(hopf and 20.0 <= hopf["freq_star_Hz"] <= 80.0),
                "detail": (
                    f"I_ext* = {hopf['I_ext_star']:.3f} nA, "
                    f"f* = {hopf['freq_star_Hz']:.2f} Hz"
                    if hopf
                    else "no Hopf found"
                ),
            },
            {
                "label": "Hopf is supercritical (reversible onset, no hysteresis)",
                "passed": bool(
                    criticality and criticality["verdict"] == "supercritical"
                ),
                "detail": (
                    f"up/down sweeps coincide (gap {criticality['hyst_gap']:.2e}, "
                    f"width {criticality['hyst_width_nA']} nA); "
                    f"A² ∝ (I−I*) slope {criticality['A2_slope']:.3e}, "
                    f"R² = {criticality['A2_r2']:.3f}"
                    if criticality
                    else "not evaluated"
                ),
            },
            {
                "label": "2D Wilson-Cowan reduction cannot sustain (rings down)",
                "passed": bool(twod and twod["pp_2d"] < 1e-4 <= twod["pp_4d"]),
                "detail": (
                    f"at I*+{twod['I_ext'] - hopf['I_ext_star']:.2g} nA: "
                    f"4D peak-to-peak {twod['pp_4d']:.3e}, 2D {twod['pp_2d']:.3e}"
                    if twod
                    else "not evaluated"
                ),
            },
            {
                "label": "Minimal dimension is 3: 3D-by-QSS keeps the Hopf, all six 2D pairs lose it",
                "passed": bool(
                    hopf3 is not None and all(v is None for v in two_d.values())
                ),
                "detail": (
                    f"3D (AMPA slaved): Hopf at I*={hopf3['I_ext_star']:.3f} nA, "
                    f"f*={hopf3['freq_star_Hz']:.2f} Hz; "
                    f"2D pairs with a Hopf: "
                    f"{[k for k, v in two_d.items() if v] or 'none (all six ring down)'}"
                    if hopf3
                    else "3D Hopf not found"
                ),
            },
        ],
    }
    return summary
