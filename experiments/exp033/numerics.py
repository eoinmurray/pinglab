"""Pure numerical model helpers, retained for exp054's existing scientific API.

Stage execution lives in compute/analyse/present. These helpers never resolve runs.
"""

from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
from scipy import linalg
from scipy.integrate import quad
from scipy.optimize import brentq, fsolve
from scipy.special import erf, erfcx

from .recipe import (
    CELL_E,
    CELL_I,
    DV_EXC_MV,
    DV_INH_MV,
    E_L_MV,
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

_gain_cells = ContextVar("exp033_gain_cells", default=None)


@contextmanager
def gain_parameters(cfg):
    """Bind gains to an explicit recipe for all reductions and nested solvers."""
    token = _gain_cells.set(
        (
            dict(cfg["cell_E"]),
            dict(cfg["cell_I"]),
            cfg.get("gain_integral") == "erfcx_for_negative_arguments",
        )
    )
    try:
        yield
    finally:
        _gain_cells.reset(token)


def lif_fi(mu_I, cell, sigma=SIGMA_V_MV):
    """Ricciardi/Siegert LIF f-I rate (1/ms) for mean input current mu_I (nA)."""
    muV = E_L_MV + mu_I / cell["g_L"]
    y_th = (V_TH_MV - muV) / sigma
    y_r = (V_RESET_MV - muV) / sigma
    cells = _gain_cells.get()
    stable = cells is None or cells[2]

    def integrand(u):
        # erfcx(-u) avoids cancellation in 1 + erf(u) for strong inputs.
        # Keep the positive-tail overflow guard and frozen v1 arithmetic.
        if stable and u < 0:
            return erfcx(-u)
        return np.exp(min(u * u, 700.0)) * (1.0 + erf(u))

    val, _ = quad(integrand, y_r, y_th, limit=200)
    return 1.0 / (cell["tau_ref"] + cell["tau_m"] * np.sqrt(np.pi) * val)


def gE(mu, sigma=SIGMA_V_MV):
    cells = _gain_cells.get()
    return lif_fi(mu, cells[0] if cells is not None else CELL_E, sigma)


def gI(mu, sigma=SIGMA_V_MV):
    cells = _gain_cells.get()
    return lif_fi(mu, cells[1] if cells is not None else CELL_I, sigma)


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


def sweep(I_ext_grid, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV, eps=1e-6):
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
        eigs = linalg.eigvals(jacobian(fp, I_ext, tau_gaba, sigma, eps))
        results.append(
            {
                "I_ext": float(I_ext),
                "fp": fp.tolist(),
                "eigs": [(float(e.real), float(e.imag)) for e in eigs],
            }
        )
    return results


def _leading_complex_eigenvalue(fp, I_ext, tau_gaba, sigma, eps=1e-6):
    eigs = linalg.eigvals(jacobian(fp, I_ext, tau_gaba, sigma, eps))
    complex_eigs = [e for e in eigs if abs(e.imag) > 1e-6]
    return max(complex_eigs, key=lambda e: e.real) if complex_eigs else None


def find_hopf(
    results,
    tau_gaba=TAU_GABA_MS,
    sigma=SIGMA_V_MV,
    refine=True,
    *,
    eps=1e-6,
    xtol=1e-10,
    rtol=1e-12,
):
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
                eig = _leading_complex_eigenvalue(fp, drive, tau_gaba, sigma, eps)
                if eig is None:
                    raise RuntimeError(f"no complex eigenvalue at I_ext={drive}")
                return float(eig.real)

            drive_star = float(brentq(real_part, lo, hi, xtol=xtol, rtol=rtol))
            fp_star = fixed_point(drive_star, tau_gaba, x0=x0, sigma=sigma)
            eig_star = _leading_complex_eigenvalue(
                fp_star, drive_star, tau_gaba, sigma, eps
            )
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


def rhs_2d(t, y, I_ext, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV):
    """Same DC coupling as 4D, synapses slaved instantaneously to the rates."""
    E, I = y
    g_eI = TAU_AMPA_MS * WT_EI * E
    g_iE = tau_gaba * WT_IE * I
    return [
        (-E + gE(I_ext - g_iE * DV_INH_MV, sigma)) / TAU_E_MS,
        (-I + gI(g_eI * DV_EXC_MV, sigma)) / TAU_I_MS,
    ]


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


def reduction_sweep(
    rhs, fp_fn, I_grid, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV, eps=1e-6
):
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
            yp[k] += eps
            ym = y0.copy()
            ym[k] -= eps
            J[:, k] = (f(yp) - f(ym)) / (2 * eps)
        eigs = linalg.eigvals(J)
        results.append(
            {
                "I_ext": float(I_ext),
                "fp": [float(v) for v in y0],
                "eigs": [(float(e.real), float(e.imag)) for e in eigs],
            }
        )
    return results
