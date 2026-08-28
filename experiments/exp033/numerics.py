"""Pure numerical model helpers, retained for exp054's existing scientific API.

Stage execution lives in compute/analyse/present. These helpers never resolve runs.
"""

from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq, fsolve
from scipy.special import erf

REPO = Path(__file__).resolve().parents[2]
from .recipe import (
    CELL_E,
    CELL_I,
    DV_EXC_MV,
    DV_INH_MV,
    E_L_MV,
    HYSTERESIS_POINTS,
    HYSTERESIS_SPAN_NA,
    LIMIT_CYCLE_OFFSET_NA,
    SIGMA_V_MV,
    TAU_AMPA_MS,
    TAU_E_MS,
    TAU_GABA_MS,
    TAU_I_MS,
    V_RESET_MV,
    V_TH_MV,
    WT_EI,
    WT_IE,
)


def lif_fi(mu_I, cell, sigma=SIGMA_V_MV):
    """Ricciardi/Siegert LIF f-I rate (1/ms) for mean input current mu_I (nA)."""
    muV = E_L_MV + mu_I / cell["g_L"]
    y_th = (V_TH_MV - muV) / sigma
    y_r = (V_RESET_MV - muV) / sigma
    val, _ = quad(
        lambda u: np.exp(min(u * u, 700.0)) * (1.0 + erf(u)), y_r, y_th, limit=200
    )
    return 1.0 / (cell["tau_ref"] + cell["tau_m"] * np.sqrt(np.pi) * val)


def gE(mu, sigma=SIGMA_V_MV):
    return lif_fi(mu, CELL_E, sigma)


def gI(mu, sigma=SIGMA_V_MV):
    return lif_fi(mu, CELL_I, sigma)


def rhs_4d(t, y, I_ext, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV):
    E, I, g_eI, g_iE = y
    return [
        (-E + gE(I_ext - g_iE * DV_INH_MV, sigma)) / TAU_E_MS,
        (-I + gI(g_eI * DV_EXC_MV, sigma)) / TAU_I_MS,
        -g_eI / TAU_AMPA_MS + WT_EI * E,
        -g_iE / tau_gaba + WT_IE * I,
    ]


def fixed_point(I_ext, tau_gaba=TAU_GABA_MS, x0=(0.005, 0.002), sigma=SIGMA_V_MV):
    """Silent fixed point; returns the 4D state or None."""

    def residual(x):
        E, I = x
        g_iE = tau_gaba * WT_IE * max(I, 0.0)
        g_eI = TAU_AMPA_MS * WT_EI * max(E, 0.0)
        return [
            E - gE(I_ext - g_iE * DV_INH_MV, sigma),
            I - gI(g_eI * DV_EXC_MV, sigma),
        ]

    sol, _, ier, _ = fsolve(residual, x0, full_output=True)
    if ier != 1:
        return None
    E, I = sol
    return np.array([E, I, TAU_AMPA_MS * WT_EI * E, tau_gaba * WT_IE * I])


def jacobian(fp, I_ext, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV, eps=1e-6):
    """Numerical 4D Jacobian at a fixed point fp = (E, I, g_e^I, g_i^E)."""

    def f(y):
        return np.array(rhs_4d(0.0, y, I_ext, tau_gaba, sigma))

    J = np.zeros((4, 4))
    y0 = np.asarray(fp, dtype=float)
    for k in range(4):
        yp = y0.copy()
        yp[k] += eps
        ym = y0.copy()
        ym[k] -= eps
        J[:, k] = (f(yp) - f(ym)) / (2 * eps)
    return J


def sweep(I_ext_grid, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV):
    results = []
    x = None
    for I_ext in I_ext_grid:
        fp = fixed_point(
            I_ext,
            tau_gaba,
            x0=(x[0], x[1]) if x is not None else (0.005, 0.002),
            sigma=sigma,
        )
        if fp is None:
            continue
        x = fp
        eigs = linalg.eigvals(jacobian(fp, I_ext, tau_gaba, sigma))
        results.append(
            {
                "I_ext": float(I_ext),
                "fp": fp.tolist(),
                "eigs": [(float(e.real), float(e.imag)) for e in eigs],
            }
        )
    return results


def _leading_complex_eigenvalue(fp, I_ext, tau_gaba, sigma):
    eigs = linalg.eigvals(jacobian(fp, I_ext, tau_gaba, sigma))
    complex_eigs = [e for e in eigs if abs(e.imag) > 1e-6]
    return max(complex_eigs, key=lambda e: e.real) if complex_eigs else None


def find_hopf(results, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV, refine=True):
    """Locate and refine the first complex-pair stability crossing.

    ``results`` supplies a coarse continuation grid.  Once a bracket is found,
    Brent's method refines the drive at which the leading complex eigenvalue's
    real part is zero; the fixed point and Jacobian are recomputed there.
    """
    previous = None
    for r in results:
        complex_eigs = [complex(e[0], e[1]) for e in r["eigs"] if abs(e[1]) > 1e-6]
        leading = max(complex_eigs, key=lambda e: e.real) if complex_eigs else None
        if (
            previous is not None
            and previous[1] is not None
            and leading is not None
            and previous[1].real < 0 <= leading.real
        ):
            lo, hi = previous[0]["I_ext"], r["I_ext"]
            if not refine:
                return {
                    "I_ext_star": float(r["I_ext"]),
                    "omega_star": float(abs(leading.imag)),
                    "freq_star_Hz": float(1000.0 * abs(leading.imag) / (2 * np.pi)),
                    "fp_at_star": r["fp"],
                    "leading_eigenvalue": [float(leading.real), float(leading.imag)],
                    "coarse_bracket_nA": [float(lo), float(hi)],
                }
            x0 = tuple(previous[0]["fp"][:2])

            def real_part(drive):
                fp = fixed_point(drive, tau_gaba, x0=x0, sigma=sigma)
                if fp is None:
                    raise RuntimeError(f"fixed-point solve failed at I_ext={drive}")
                eig = _leading_complex_eigenvalue(fp, drive, tau_gaba, sigma)
                if eig is None:
                    raise RuntimeError(f"no complex eigenvalue at I_ext={drive}")
                return float(eig.real)

            drive_star = float(brentq(real_part, lo, hi, xtol=1e-10, rtol=1e-12))
            fp_star = fixed_point(drive_star, tau_gaba, x0=x0, sigma=sigma)
            eig_star = _leading_complex_eigenvalue(fp_star, drive_star, tau_gaba, sigma)
            return {
                "I_ext_star": drive_star,
                "omega_star": float(abs(eig_star.imag)),
                "freq_star_Hz": float(1000.0 * abs(eig_star.imag) / (2 * np.pi)),
                "fp_at_star": fp_star.tolist(),
                "leading_eigenvalue": [float(eig_star.real), float(eig_star.imag)],
                "coarse_bracket_nA": [float(lo), float(hi)],
            }
        previous = (r, leading)
    return None


def settle(
    I_ext, y0, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV, t_max=2000.0, t_settle=1500.0
):
    """Integrate from y0 to steady state; return (peak-to-peak E amplitude,
    final state). The final state is carried into the next sweep step so a
    coexisting cycle, if any, is followed (quasi-static continuation)."""
    sol = solve_ivp(
        rhs_4d,
        (0, t_max),
        y0,
        args=(I_ext, tau_gaba, sigma),
        method="LSODA",
        rtol=1e-7,
        atol=1e-10,
        max_step=1.0,
    )
    y_end = sol.y[:, -1]
    E = sol.y[0][sol.t >= t_settle]
    amp = float(E.max() - E.min()) if E.size >= 10 else 0.0
    return amp, y_end


def hysteresis_sweep(
    i_star, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV, span=(-0.1, 0.55), n=25
):
    """Quasi-static up/down ramp of I_ext across I*. Supercritical onset is
    reversible (branches coincide); subcritical leaves a hysteresis loop."""
    grid = np.linspace(i_star + span[0], i_star + span[1], n)
    thr = 1e-4
    # rising branch: start from the silent fixed point with a small kick
    y = fixed_point(grid[0], tau_gaba, sigma=sigma).copy()
    y[0] += 1e-3
    up = []
    for I in grid:
        amp, y = settle(I, y, tau_gaba, sigma)
        up.append({"I_ext": float(I), "amp": amp})
    # falling branch: continue from the high-drive end state
    down = []
    for I in grid[::-1]:
        amp, y = settle(I, y, tau_gaba, sigma)
        down.append({"I_ext": float(I), "amp": amp})
    down.reverse()
    # max amplitude gap between branches at equal drive = hysteresis size
    hyst_gap = float(max(abs(d["amp"] - u["amp"]) for u, d in zip(up, down)))
    on = next((u["I_ext"] for u in up if u["amp"] > thr), None)
    off = next((d["I_ext"] for d in down if d["amp"] > thr), None)
    hyst_width = float(on - off) if (on is not None and off is not None) else None
    # A^2 vs (I - I*) on the rising branch above threshold
    above = [(u["I_ext"] - i_star, u["amp"]) for u in up if u["I_ext"] > i_star + 1e-9]
    slope, r2 = 0.0, 0.0
    if len(above) >= 2:
        x = np.array([a[0] for a in above])
        ysq = np.array([a[1] ** 2 for a in above])
        m, c = np.polyfit(x, ysq, 1)
        ss_res = float(np.sum((ysq - (m * x + c)) ** 2))
        ss_tot = float(np.sum((ysq - ysq.mean()) ** 2))
        slope = float(m)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    supercritical = hyst_gap < thr and slope > 0 and r2 > 0.9
    return {
        "verdict": "supercritical" if supercritical else "subcritical/inconclusive",
        "hyst_gap": hyst_gap,
        "hyst_width_nA": hyst_width,
        "A2_slope": slope,
        "A2_r2": r2,
        "up": up,
        "down": down,
    }


def rhs_2d(t, y, I_ext, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV):
    """Same DC coupling as 4D, synapses slaved instantaneously to the rates."""
    E, I = y
    g_eI = TAU_AMPA_MS * WT_EI * E
    g_iE = tau_gaba * WT_IE * I
    return [
        (-E + gE(I_ext - g_iE * DV_INH_MV, sigma)) / TAU_E_MS,
        (-I + gI(g_eI * DV_EXC_MV, sigma)) / TAU_I_MS,
    ]


def compute_2d_vs_4d(hopf, offset=1.0, sigma=SIGMA_V_MV):
    """Same drive above the 4D Hopf: 2D rings down, 4D sustains. Numeric
    check for the analytic Bendixson-Dulac rejection of the 2D field."""
    I_ext = hopf["I_ext_star"] + offset
    fp4 = fixed_point(I_ext, sigma=sigma)
    sol4 = solve_ivp(
        rhs_4d,
        (0, 300),
        fp4 + np.array([2e-3, 0, 0, 0]),
        args=(I_ext, TAU_GABA_MS, sigma),
        method="LSODA",
        rtol=1e-8,
        atol=1e-11,
        max_step=0.5,
    )
    sol2 = solve_ivp(
        rhs_2d,
        (0, 300),
        [fp4[0] + 2e-3, fp4[1]],
        args=(I_ext, TAU_GABA_MS, sigma),
        method="LSODA",
        rtol=1e-8,
        atol=1e-11,
        max_step=0.5,
    )
    d4, d2 = sol4.y[0] - fp4[0], sol2.y[0] - fp4[0]
    pp4 = float(d4[sol4.t > 150].max() - d4[sol4.t > 150].min())
    pp2 = float(d2[sol2.t > 150].max() - d2[sol2.t > 150].min())
    print(f"  2D-vs-4D at I={I_ext:.2f} nA: 4D peak-to-peak={pp4:.4e}, 2D={pp2:.4e}")
    return {"I_ext": float(I_ext), "pp_4d": pp4, "pp_2d": pp2}


def limit_cycle_metrics(hopf, offset=0.4, sigma=SIGMA_V_MV):
    """Return reference limit-cycle metrics at a drive relative to onset."""
    I_ext = hopf["I_ext_star"] + offset
    fp = fixed_point(I_ext, sigma=sigma)
    sol = solve_ivp(
        rhs_4d,
        (0, 700),
        fp + np.array([1e-3, 0, 0, 0]),
        args=(I_ext, TAU_GABA_MS, sigma),
        method="LSODA",
        rtol=1e-9,
        atol=1e-12,
        max_step=0.25,
        dense_output=True,
    )
    period = 1000.0 / hopf["freq_star_Hz"]
    tt = np.linspace(700 - 3 * period, 700, 1500)
    Y = sol.sol(tt)
    E, I = Y[0], Y[1]
    Ez, Iz = E - E.mean(), I - I.mean()
    lags = (np.arange(len(tt)) - len(tt) // 2) * (tt[1] - tt[0])
    xc = np.correlate(Iz, Ez, mode="same")
    return {
        "I_ext": float(I_ext),
        "e_leads_i_ms": float(abs(lags[np.argmax(xc)])),
        "e_peak_to_peak": float(np.ptp(E)),
        "t_ms": tt,
        "E": E,
        "I": I,
    }


def frequency_vs_tau_gaba(tau_list, I_grid, sigma=SIGMA_V_MV):
    out = []
    for tg in tau_list:
        h = find_hopf(sweep(I_grid, tau_gaba=tg, sigma=sigma), tg, sigma)
        out.append(
            {
                "tau_gaba_ms": tg,
                "f_star_Hz": h["freq_star_Hz"] if h else None,
                "I_ext_star": h["I_ext_star"] if h else None,
            }
        )
    return out


def rhs_2d_qss(t, y, I_ext, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV):
    """Rates slaved to their f-I steady state: 2D in (g_e^I, g_i^E)."""
    g_eI, g_iE = y
    E = gE(I_ext - g_iE * DV_INH_MV, sigma)
    Inh = gI(g_eI * DV_EXC_MV, sigma)
    return [-g_eI / TAU_AMPA_MS + WT_EI * E, -g_iE / tau_gaba + WT_IE * Inh]


def fixed_point_2d_qss(I_ext, tau_gaba=TAU_GABA_MS, x0=None, sigma=SIGMA_V_MV):
    if x0 is None:
        x0 = (0.01, 0.02)
    sol, _, ier, _ = fsolve(
        lambda y: rhs_2d_qss(0.0, y, I_ext, tau_gaba, sigma), x0, full_output=True
    )
    return np.asarray(sol) if ier == 1 else None


def rhs_3d_qss(t, y, I_ext, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV):
    """Fast AMPA conductance slaved (g_e^I = tau_AMPA W^EI E): 3D in (E, I, g_i^E)."""
    E, I, g_iE = y
    g_eI = TAU_AMPA_MS * WT_EI * E
    return [
        (-E + gE(I_ext - g_iE * DV_INH_MV, sigma)) / TAU_E_MS,
        (-I + gI(g_eI * DV_EXC_MV, sigma)) / TAU_I_MS,
        -g_iE / tau_gaba + WT_IE * I,
    ]


def fixed_point_3d_qss(I_ext, tau_gaba=TAU_GABA_MS, x0=None, sigma=SIGMA_V_MV):
    if x0 is None:
        x0 = (0.005, 0.002, 0.02)
    sol, _, ier, _ = fsolve(
        lambda y: rhs_3d_qss(0.0, y, I_ext, tau_gaba, sigma), x0, full_output=True
    )
    return np.asarray(sol) if ier == 1 else None


def rhs_2d_fastslow(t, y, I_ext, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV):
    """Fast/slow lump (route 3): slave the fast pair
    {g_e^I (tau=2), I (tau=5)} to quasi-steady state and keep the slow
    {E (tau=20), g_i^E (tau=9)} -> 2D in (E, g_i^E)."""
    E, g_iE = y
    g_eI = TAU_AMPA_MS * WT_EI * E
    I = gI(g_eI * DV_EXC_MV, sigma)
    return [
        (-E + gE(I_ext - g_iE * DV_INH_MV, sigma)) / TAU_E_MS,
        -g_iE / tau_gaba + WT_IE * I,
    ]


def fixed_point_2d_fastslow(I_ext, tau_gaba=TAU_GABA_MS, x0=None, sigma=SIGMA_V_MV):
    if x0 is None:
        x0 = (0.005, 0.02)
    sol, _, ier, _ = fsolve(
        lambda y: rhs_2d_fastslow(0.0, y, I_ext, tau_gaba, sigma), x0, full_output=True
    )
    return np.asarray(sol) if ier == 1 else None


def fixed_point_2d_wc(I_ext, tau_gaba=TAU_GABA_MS, x0=None, sigma=SIGMA_V_MV):
    """Fixed point of the Wilson-Cowan field rhs_2d (keep E, I)."""
    if x0 is None:
        x0 = (0.005, 0.002)
    sol, _, ier, _ = fsolve(
        lambda y: rhs_2d(0.0, y, I_ext, tau_gaba, sigma), x0, full_output=True
    )
    return np.asarray(sol) if ier == 1 else None


def rhs_2d_E_ge(t, y, I_ext, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV):
    """Keep (E, g_e^I); slave I = Phi_I(g_e^I) and g_i^E = tau_GABA W^IE I."""
    E, g_eI = y
    I = gI(g_eI * DV_EXC_MV, sigma)
    g_iE = tau_gaba * WT_IE * I
    return [
        (-E + gE(I_ext - g_iE * DV_INH_MV, sigma)) / TAU_E_MS,
        -g_eI / TAU_AMPA_MS + WT_EI * E,
    ]


def fixed_point_2d_E_ge(I_ext, tau_gaba=TAU_GABA_MS, x0=None, sigma=SIGMA_V_MV):
    if x0 is None:
        x0 = (0.005, 0.01)
    sol, _, ier, _ = fsolve(
        lambda y: rhs_2d_E_ge(0.0, y, I_ext, tau_gaba, sigma), x0, full_output=True
    )
    return np.asarray(sol) if ier == 1 else None


def rhs_2d_I_gi(t, y, I_ext, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV):
    """Keep (I, g_i^E); slave E = Phi_E(I_ext - g_i^E) and g_e^I = tau_AMPA W^EI E."""
    I, g_iE = y
    E = gE(I_ext - g_iE * DV_INH_MV, sigma)
    g_eI = TAU_AMPA_MS * WT_EI * E
    return [(-I + gI(g_eI * DV_EXC_MV, sigma)) / TAU_I_MS, -g_iE / tau_gaba + WT_IE * I]


def fixed_point_2d_I_gi(I_ext, tau_gaba=TAU_GABA_MS, x0=None, sigma=SIGMA_V_MV):
    if x0 is None:
        x0 = (0.002, 0.02)
    sol, _, ier, _ = fsolve(
        lambda y: rhs_2d_I_gi(0.0, y, I_ext, tau_gaba, sigma), x0, full_output=True
    )
    return np.asarray(sol) if ier == 1 else None


def rhs_2d_I_ge(t, y, I_ext, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV):
    """Keep (I, g_e^I); slave g_i^E = tau_GABA W^IE I and E = Phi_E(I_ext - g_i^E)."""
    I, g_eI = y
    g_iE = tau_gaba * WT_IE * I
    E = gE(I_ext - g_iE * DV_INH_MV, sigma)
    return [
        (-I + gI(g_eI * DV_EXC_MV, sigma)) / TAU_I_MS,
        -g_eI / TAU_AMPA_MS + WT_EI * E,
    ]


def fixed_point_2d_I_ge(I_ext, tau_gaba=TAU_GABA_MS, x0=None, sigma=SIGMA_V_MV):
    if x0 is None:
        x0 = (0.002, 0.01)
    sol, _, ier, _ = fsolve(
        lambda y: rhs_2d_I_ge(0.0, y, I_ext, tau_gaba, sigma), x0, full_output=True
    )
    return np.asarray(sol) if ier == 1 else None


def reduction_sweep(rhs, fp_fn, I_grid, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV):
    """Fixed point + numeric-Jacobian eigenvalues across an I_ext sweep,
    for a reduced model (so find_hopf can run on it)."""
    results = []
    x = None
    for I_ext in I_grid:
        fp = fp_fn(I_ext, tau_gaba, tuple(x) if x is not None else None, sigma)
        if fp is None:
            continue
        x = fp
        y0 = np.asarray(fp, dtype=float)

        def f(y):
            return np.asarray(rhs(0.0, y, I_ext, tau_gaba, sigma))

        n = y0.size
        J = np.zeros((n, n))
        for k in range(n):
            yp = y0.copy()
            yp[k] += 1e-6
            ym = y0.copy()
            ym[k] -= 1e-6
            J[:, k] = (f(yp) - f(ym)) / 2e-6
        eigs = linalg.eigvals(J)
        results.append(
            {
                "I_ext": float(I_ext),
                "fp": [float(v) for v in y0],
                "eigs": [(float(e.real), float(e.imag)) for e in eigs],
            }
        )
    return results


def sigma_sensitivity(sigma_grid, coarse_grid, convergence_grid):
    """Evaluate every headline bifurcation quantity across effective noise."""
    rows: list[dict] = []
    for sigma in sigma_grid:
        coarse = sweep(coarse_grid, sigma=sigma)
        hopf = find_hopf(coarse, sigma=sigma)
        fine_hopf = find_hopf(sweep(convergence_grid, sigma=sigma), sigma=sigma)
        if hopf is None:
            rows.append({"sigma_V_mV": float(sigma), "hopf_exists": False})
            continue
        criticality = hysteresis_sweep(
            hopf["I_ext_star"],
            sigma=sigma,
            span=HYSTERESIS_SPAN_NA,
            n=HYSTERESIS_POINTS,
        )
        cycle = limit_cycle_metrics(hopf, offset=LIMIT_CYCLE_OFFSET_NA, sigma=sigma)
        rows.append(
            {
                "sigma_V_mV": float(sigma),
                "hopf_exists": True,
                "hopf": hopf,
                "fixed_point_at_hopf": {
                    "E_per_ms": hopf["fp_at_star"][0],
                    "I_per_ms": hopf["fp_at_star"][1],
                    "g_eI": hopf["fp_at_star"][2],
                    "g_iE": hopf["fp_at_star"][3],
                },
                "criticality": {
                    k: v for k, v in criticality.items() if k not in {"up", "down"}
                },
                "limit_cycle": {
                    "relative_drive_nA": LIMIT_CYCLE_OFFSET_NA,
                    "e_peak_to_peak": cycle["e_peak_to_peak"],
                    "e_leads_i_ms": cycle["e_leads_i_ms"],
                },
                "convergence_check": {
                    "comparison_grid_step_nA": float(np.diff(convergence_grid[:2])[0]),
                    "I_ext_star": fine_hopf["I_ext_star"] if fine_hopf else None,
                    "absolute_difference_nA": (
                        abs(hopf["I_ext_star"] - fine_hopf["I_ext_star"])
                        if fine_hopf
                        else None
                    ),
                },
            }
        )
    return {
        "sigma_grid_mV": [float(s) for s in sigma_grid],
        "reference_sigma_mV": SIGMA_V_MV,
        "settings": {
            "drive_interval_nA": [float(coarse_grid[0]), float(coarse_grid[-1])],
            "coarse_grid_step_nA": float(np.diff(coarse_grid[:2])[0]),
            "hopf_refinement": "Brent root of leading complex eigenvalue real part",
            "hysteresis_span_relative_nA": list(HYSTERESIS_SPAN_NA),
            "hysteresis_points": HYSTERESIS_POINTS,
            "amplitude_threshold": 1e-4,
            "integration_t_max_ms": 2000.0,
            "integration_settle_start_ms": 1500.0,
            "limit_cycle_relative_drive_nA": LIMIT_CYCLE_OFFSET_NA,
        },
        "rows": rows,
        "topology_retained": all(row.get("hopf_exists") for row in rows),
        "supercritical_retained": all(
            row.get("criticality", {}).get("verdict") == "supercritical" for row in rows
        ),
    }
