r"""033 — Mean-field PING: the 4D conductance Hopf with a LIF population gain.

Numerics for the theory in src/docs/src/pages/notebooks/exp033.mdx.

One analysis on one model. The 4D reduction in state $(E, I, g_e^I, g_i^E)$
uses COBANet's own leaky-integrate-and-fire f-I curve (Ricciardi/Siegert)
as the population gain, with the recurrent couplings read off the
biophysics (the ei-strength scalar times the fan-in times the synaptic
driving force). The effective membrane-noise std σ_V is free and sensitivity-tested.

The runner sweeps the external drive $I_\text{ext}$ (nA), tracks the
silent fixed point, diagonalises the Jacobian, and locates the Hopf
(smallest $I_\text{ext}^\star$ where max Re(λ) crosses 0). It then:

  - reads off the gamma frequency $f^\star = \omega^\star/2\pi$ at the crossing;
  - re-finds the Hopf across the exp041 $\tau_\text{GABA}$ sweep and compares
    $f^\star$ to the spiking $f_\gamma$;
  - tests super- vs subcritical onset by direct ODE simulation of the
    amplitude across the bifurcation;
  - contrasts the 4D field against its 2D Wilson-Cowan reduction (synapses
    adiabatically eliminated), which rings down at the same drive.

Writing: writings/exp033.typ · figures + numbers.json: artifacts/data/exp033/
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq, fsolve
from scipy.special import erf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import theme  # noqa: E402
from helpers.operating_point import TAU_GABA_GAMMA_MS  # noqa: E402
from helpers.paths import (  # noqa: E402
    artifacts_and_figures,
    log_runner_event,
    runner_paths,
)
from helpers.stamp import stamp_figure  # noqa: E402

SLUG = "exp033"
RUN_PATHS = runner_paths(SLUG)
_, FIGURES = artifacts_and_figures(SLUG)

# ── Timescales (ms) ───────────────────────────────────────────────────
TAU_E_MS = 20.0    # E membrane (= CELL_E tau_m)
TAU_I_MS = 5.0     # I membrane (= CELL_I tau_m)
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
DV_INH_MV, DV_EXC_MV = 15.0, 65.0   # |V_rest − E_rev| driving forces
WT_EI, WT_IE = 1.0, 2.0             # lumped couplings (µS): s and r·s
SIGMA_V_MV = 4.0                    # membrane-noise std
SIGMA_V_GRID_MV = (3.0, 4.0, 5.0, 6.0)
HYSTERESIS_SPAN_NA = (-0.1, 0.55)
HYSTERESIS_POINTS = 25
LIMIT_CYCLE_OFFSET_NA = 0.4


def lif_fi(mu_I, cell, sigma=SIGMA_V_MV):
    """Ricciardi/Siegert LIF f-I rate (1/ms) for mean input current mu_I (nA)."""
    muV = E_L_MV + mu_I / cell["g_L"]
    y_th = (V_TH_MV - muV) / sigma
    y_r = (V_RESET_MV - muV) / sigma
    val, _ = quad(lambda u: np.exp(min(u * u, 700.0)) * (1.0 + erf(u)),
                  y_r, y_th, limit=200)
    return 1.0 / (cell["tau_ref"] + cell["tau_m"] * np.sqrt(np.pi) * val)


def gE(mu, sigma=SIGMA_V_MV):
    return lif_fi(mu, CELL_E, sigma)


def gI(mu, sigma=SIGMA_V_MV):
    return lif_fi(mu, CELL_I, sigma)


# ── 4D mean-field, state (E, I, g_e^I, g_i^E) ─────────────────────────


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
        return [E - gE(I_ext - g_iE * DV_INH_MV, sigma),
                I - gI(g_eI * DV_EXC_MV, sigma)]

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
        fp = fixed_point(I_ext, tau_gaba,
                         x0=(x[0], x[1]) if x is not None else (0.005, 0.002),
                         sigma=sigma)
        if fp is None:
            continue
        x = fp
        eigs = linalg.eigvals(jacobian(fp, I_ext, tau_gaba, sigma))
        results.append({
            "I_ext": float(I_ext),
            "fp": fp.tolist(),
            "eigs": [(float(e.real), float(e.imag)) for e in eigs],
        })
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
        complex_eigs = [complex(e[0], e[1]) for e in r["eigs"]
                        if abs(e[1]) > 1e-6]
        leading = max(complex_eigs, key=lambda e: e.real) if complex_eigs else None
        if (previous is not None and previous[1] is not None and leading is not None
                and previous[1].real < 0 <= leading.real):
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
            eig_star = _leading_complex_eigenvalue(
                fp_star, drive_star, tau_gaba, sigma
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


# ── Super- vs subcritical by direct simulation ────────────────────────


def settle(I_ext, y0, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV,
           t_max=2000.0, t_settle=1500.0):
    """Integrate from y0 to steady state; return (peak-to-peak E amplitude,
    final state). The final state is carried into the next sweep step so a
    coexisting cycle, if any, is followed (quasi-static continuation)."""
    sol = solve_ivp(rhs_4d, (0, t_max), y0, args=(I_ext, tau_gaba, sigma),
                    method="LSODA", rtol=1e-7, atol=1e-10, max_step=1.0)
    y_end = sol.y[:, -1]
    E = sol.y[0][sol.t >= t_settle]
    amp = float(E.max() - E.min()) if E.size >= 10 else 0.0
    return amp, y_end


def hysteresis_sweep(i_star, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV,
                     span=(-0.1, 0.55), n=25):
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
    above = [(u["I_ext"] - i_star, u["amp"]) for u in up
             if u["I_ext"] > i_star + 1e-9]
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


def plot_hysteresis(sweep, hopf, out_path, run_id):
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    i_star = hopf["I_ext_star"]
    xu = [d["I_ext"] for d in sweep["up"]]
    yu = [d["amp"] for d in sweep["up"]]
    xd = [d["I_ext"] for d in sweep["down"]]
    yd = [d["amp"] for d in sweep["down"]]
    ax.plot(xu, yu, "o-", color=theme.INK_BLACK, lw=1.2, ms=5,
            label="drive increasing")
    ax.plot(xd, yd, "s--", color=theme.DEEP_RED, lw=1.0, ms=5,
            markerfacecolor="none", label="drive decreasing")
    ax.axvline(i_star, color=theme.AMBER, lw=0.6, ls=":")
    ax.annotate("no hysteresis:\nbranches coincide",
                xy=(i_star, 0.0),
                xytext=(i_star - 0.085, max(yu) * 0.55),
                fontsize=theme.SIZE_ANNOTATION, color=theme.GREY_DARK,
                ha="left", va="center")
    ax.set_xlabel("$I_\\text{ext}$ (nA)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("E amplitude (peak-to-peak)", fontsize=theme.SIZE_LABEL)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── 2D Wilson-Cowan field (synapses adiabatically eliminated) ──────────


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
    sol4 = solve_ivp(rhs_4d, (0, 300), fp4 + np.array([2e-3, 0, 0, 0]),
                     args=(I_ext, TAU_GABA_MS, sigma), method="LSODA",
                     rtol=1e-8, atol=1e-11, max_step=0.5)
    sol2 = solve_ivp(rhs_2d, (0, 300), [fp4[0] + 2e-3, fp4[1]],
                     args=(I_ext, TAU_GABA_MS, sigma), method="LSODA",
                     rtol=1e-8, atol=1e-11, max_step=0.5)
    d4, d2 = sol4.y[0] - fp4[0], sol2.y[0] - fp4[0]
    pp4 = float(d4[sol4.t > 150].max() - d4[sol4.t > 150].min())
    pp2 = float(d2[sol2.t > 150].max() - d2[sol2.t > 150].min())
    print(f"  2D-vs-4D at I={I_ext:.2f} nA: 4D peak-to-peak={pp4:.4e}, 2D={pp2:.4e}")
    return {"I_ext": float(I_ext), "pp_4d": pp4, "pp_2d": pp2}


def plot_eigenvalues_complex(results, hopf, out_path, run_id):
    """The four 4D eigenvalues in the complex plane, coloured by drive."""
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    xs = np.array([r["I_ext"] for r in results])
    eig_re = np.array([[e[0] for e in r["eigs"]] for r in results])
    eig_im = np.array([[e[1] for e in r["eigs"]] for r in results])
    sc = None
    for k in range(eig_re.shape[1]):
        sc = ax.scatter(eig_re[:, k], eig_im[:, k], c=xs, cmap="magma",
                        s=5, linewidths=0)
    assert sc is not None  # eig_re has 4 columns, so the loop always assigns sc
    ax.axvline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    if hopf:
        w = hopf["omega_star"]
        ax.scatter([0, 0], [w, -w], facecolors="none",
                   edgecolors=theme.ELECTRIC_CYAN, s=70, lw=1.4, zorder=5)
        ax.annotate(f"crossing at $\\pm i\\omega^\\star$\n"
                    f"$f^\\star = {hopf['freq_star_Hz']:.1f}$ Hz",
                    xy=(0, w), xytext=(0.10 * eig_re.max(), w + 0.12 * w),
                    fontsize=theme.SIZE_ANNOTATION, color=theme.GREY_DARK,
                    ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=theme.ELECTRIC_CYAN,
                                    lw=0.8))
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("$I_\\text{ext}$ (nA)", fontsize=theme.SIZE_LABEL)
    ax.set_xlabel("Re$(\\lambda)$", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Im$(\\lambda)$", fontsize=theme.SIZE_LABEL)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def limit_cycle_metrics(hopf, offset=0.4, sigma=SIGMA_V_MV):
    """Return reference limit-cycle metrics at a drive relative to onset."""
    I_ext = hopf["I_ext_star"] + offset
    fp = fixed_point(I_ext, sigma=sigma)
    sol = solve_ivp(rhs_4d, (0, 700), fp + np.array([1e-3, 0, 0, 0]),
                    args=(I_ext, TAU_GABA_MS, sigma), method="LSODA",
                    rtol=1e-9, atol=1e-12, max_step=0.25, dense_output=True)
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


def plot_limit_cycle(hopf, out_path, run_id, offset=0.4, sigma=SIGMA_V_MV):
    """4D limit cycle just above onset: E and I waveforms and the E→I lag."""
    theme.apply()
    metrics = limit_cycle_metrics(hopf, offset=offset, sigma=sigma)
    I_ext, tt, E, I = (metrics[k] for k in ("I_ext", "t_ms", "E", "I"))
    print(f"  limit cycle at I={I_ext:.2f} nA: "
          f"I lags E by {metrics['e_leads_i_ms']:.2f} ms")
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax.plot(tt - tt[0], E, color=theme.INK_BLACK, lw=1.3, label="$E$")
    ax.set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("$E$ rate", fontsize=theme.SIZE_LABEL, color=theme.INK_BLACK)
    ax2 = ax.twinx()
    ax2.plot(tt - tt[0], I, color=theme.DEEP_RED, lw=1.3, label="$I$")
    ax2.set_ylabel("$I$ rate", fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {k: metrics[k] for k in ("I_ext", "e_leads_i_ms", "e_peak_to_peak")}


# ── Frequency vs inhibitory decay, against the exp041 spiking sweep ─────


def load_exp041_fgamma():
    collection_root = os.environ.get("PINGLAB_COLLECTION_DERIVED_ROOT")
    root = Path(collection_root) if collection_root else FIGURES.parent
    p = root / "exp041" / "numbers.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    meas = {}
    for r in d.get("results", []):
        meas.setdefault(r["tau_gaba_ms"], []).append(r["f_gamma_hz"])
    return {k: float(np.median(v)) for k, v in meas.items()}


def frequency_vs_tau_gaba(tau_list, I_grid, sigma=SIGMA_V_MV):
    out = []
    for tg in tau_list:
        h = find_hopf(sweep(I_grid, tau_gaba=tg, sigma=sigma), tg, sigma)
        out.append({"tau_gaba_ms": tg,
                    "f_star_Hz": h["freq_star_Hz"] if h else None,
                    "I_ext_star": h["I_ext_star"] if h else None})
    return out


def plot_frequency_vs_tau_gaba(mf, meas, out_path, run_id):
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    tg = [d["tau_gaba_ms"] for d in mf if d["f_star_Hz"] is not None]
    fs = [d["f_star_Hz"] for d in mf if d["f_star_Hz"] is not None]
    ax.plot(tg, fs, "o-", color=theme.INK_BLACK, lw=1.4,
            label="reference mean-field $f^\\star$")
    if meas:
        mt = sorted(meas)
        ax.plot(mt, [meas[t] for t in mt], "s--", color=theme.DEEP_RED, lw=1.3,
                label="spiking $f_\\gamma$ (exp041)")
    ax.set_xlabel("$\\tau_\\text{GABA}$ (ms)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("gamma frequency (Hz)", fontsize=theme.SIZE_LABEL)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Dimensional reductions (answering "is it really 4D?", issue #38) ────
# The Hopf needs a multi-lag negative-feedback ring. Eliminating variables
# by quasi-steady state tests how low the dimension can go:
#   - slave the rates -> 2D in (g_e^I, g_i^E): gains sit off-diagonal, so
#     the divergence is the constant -1/tau_AMPA - 1/tau_GABA < 0 and
#     Bendixson-Dulac forbids a cycle (the dual of the Wilson-Cowan
#     rejection, which slaves the conductances instead);
#   - slave only the fast AMPA conductance -> 3D in (E, I, g_i^E): a
#     three-lag ring, which retains the Hopf.


def rhs_2d_qss(t, y, I_ext, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV):
    """Rates slaved to their f-I steady state: 2D in (g_e^I, g_i^E)."""
    g_eI, g_iE = y
    E = gE(I_ext - g_iE * DV_INH_MV, sigma)
    Inh = gI(g_eI * DV_EXC_MV, sigma)
    return [-g_eI / TAU_AMPA_MS + WT_EI * E,
            -g_iE / tau_gaba + WT_IE * Inh]


def fixed_point_2d_qss(I_ext, tau_gaba=TAU_GABA_MS, x0=None, sigma=SIGMA_V_MV):
    if x0 is None:
        x0 = (0.01, 0.02)
    sol, _, ier, _ = fsolve(
        lambda y: rhs_2d_qss(0.0, y, I_ext, tau_gaba, sigma), x0,
        full_output=True)
    return np.asarray(sol) if ier == 1 else None


def rhs_3d_qss(t, y, I_ext, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV):
    """Fast AMPA conductance slaved (g_e^I = tau_AMPA W^EI E): 3D in (E, I, g_i^E)."""
    E, I, g_iE = y
    g_eI = TAU_AMPA_MS * WT_EI * E
    return [(-E + gE(I_ext - g_iE * DV_INH_MV, sigma)) / TAU_E_MS,
            (-I + gI(g_eI * DV_EXC_MV, sigma)) / TAU_I_MS,
            -g_iE / tau_gaba + WT_IE * I]


def fixed_point_3d_qss(I_ext, tau_gaba=TAU_GABA_MS, x0=None, sigma=SIGMA_V_MV):
    if x0 is None:
        x0 = (0.005, 0.002, 0.02)
    sol, _, ier, _ = fsolve(
        lambda y: rhs_3d_qss(0.0, y, I_ext, tau_gaba, sigma), x0,
        full_output=True)
    return np.asarray(sol) if ier == 1 else None


def rhs_2d_fastslow(t, y, I_ext, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV):
    """Fast/slow lump (issue #38, route 3): slave the fast pair
    {g_e^I (tau=2), I (tau=5)} to quasi-steady state and keep the slow
    {E (tau=20), g_i^E (tau=9)} -> 2D in (E, g_i^E)."""
    E, g_iE = y
    g_eI = TAU_AMPA_MS * WT_EI * E
    I = gI(g_eI * DV_EXC_MV, sigma)
    return [(-E + gE(I_ext - g_iE * DV_INH_MV, sigma)) / TAU_E_MS,
            -g_iE / tau_gaba + WT_IE * I]


def fixed_point_2d_fastslow(I_ext, tau_gaba=TAU_GABA_MS, x0=None, sigma=SIGMA_V_MV):
    if x0 is None:
        x0 = (0.005, 0.02)
    sol, _, ier, _ = fsolve(
        lambda y: rhs_2d_fastslow(0.0, y, I_ext, tau_gaba, sigma), x0,
        full_output=True)
    return np.asarray(sol) if ier == 1 else None


def fixed_point_2d_wc(I_ext, tau_gaba=TAU_GABA_MS, x0=None, sigma=SIGMA_V_MV):
    """Fixed point of the Wilson-Cowan field rhs_2d (keep E, I)."""
    if x0 is None:
        x0 = (0.005, 0.002)
    sol, _, ier, _ = fsolve(
        lambda y: rhs_2d(0.0, y, I_ext, tau_gaba, sigma), x0, full_output=True)
    return np.asarray(sol) if ier == 1 else None


# The remaining three of the six keep-a-pair reductions, for completeness
# (the pure-ring argument says every 2D pair has constant negative trace; the
# runner checks all six). In each, the two dropped variables are slaved to
# quasi-steady state in terms of the kept pair.

def rhs_2d_E_ge(t, y, I_ext, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV):
    """Keep (E, g_e^I); slave I = Phi_I(g_e^I) and g_i^E = tau_GABA W^IE I."""
    E, g_eI = y
    I = gI(g_eI * DV_EXC_MV, sigma)
    g_iE = tau_gaba * WT_IE * I
    return [(-E + gE(I_ext - g_iE * DV_INH_MV, sigma)) / TAU_E_MS,
            -g_eI / TAU_AMPA_MS + WT_EI * E]


def fixed_point_2d_E_ge(I_ext, tau_gaba=TAU_GABA_MS, x0=None, sigma=SIGMA_V_MV):
    if x0 is None:
        x0 = (0.005, 0.01)
    sol, _, ier, _ = fsolve(
        lambda y: rhs_2d_E_ge(0.0, y, I_ext, tau_gaba, sigma), x0, full_output=True)
    return np.asarray(sol) if ier == 1 else None


def rhs_2d_I_gi(t, y, I_ext, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV):
    """Keep (I, g_i^E); slave E = Phi_E(I_ext - g_i^E) and g_e^I = tau_AMPA W^EI E."""
    I, g_iE = y
    E = gE(I_ext - g_iE * DV_INH_MV, sigma)
    g_eI = TAU_AMPA_MS * WT_EI * E
    return [(-I + gI(g_eI * DV_EXC_MV, sigma)) / TAU_I_MS,
            -g_iE / tau_gaba + WT_IE * I]


def fixed_point_2d_I_gi(I_ext, tau_gaba=TAU_GABA_MS, x0=None, sigma=SIGMA_V_MV):
    if x0 is None:
        x0 = (0.002, 0.02)
    sol, _, ier, _ = fsolve(
        lambda y: rhs_2d_I_gi(0.0, y, I_ext, tau_gaba, sigma), x0, full_output=True)
    return np.asarray(sol) if ier == 1 else None


def rhs_2d_I_ge(t, y, I_ext, tau_gaba=TAU_GABA_MS, sigma=SIGMA_V_MV):
    """Keep (I, g_e^I); slave g_i^E = tau_GABA W^IE I and E = Phi_E(I_ext - g_i^E)."""
    I, g_eI = y
    g_iE = tau_gaba * WT_IE * I
    E = gE(I_ext - g_iE * DV_INH_MV, sigma)
    return [(-I + gI(g_eI * DV_EXC_MV, sigma)) / TAU_I_MS,
            -g_eI / TAU_AMPA_MS + WT_EI * E]


def fixed_point_2d_I_ge(I_ext, tau_gaba=TAU_GABA_MS, x0=None, sigma=SIGMA_V_MV):
    if x0 is None:
        x0 = (0.002, 0.01)
    sol, _, ier, _ = fsolve(
        lambda y: rhs_2d_I_ge(0.0, y, I_ext, tau_gaba, sigma), x0, full_output=True)
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
        results.append({
            "I_ext": float(I_ext), "fp": [float(v) for v in y0],
            "eigs": [(float(e.real), float(e.imag)) for e in eigs],
        })
    return results


def plot_phase_planes(hopf, out_path, run_id, offset=0.4):
    """Project the 4D limit cycle onto every variable pair (issue #38's
    experiment 1): the trajectory is a closed loop living on a 2D ribbon —
    the centre manifold — even though no two physical variables collapse."""
    theme.apply()
    I_ext = hopf["I_ext_star"] + offset
    fp = fixed_point(I_ext)
    sol = solve_ivp(rhs_4d, (0, 700), fp + np.array([1e-3, 0, 0, 0]),
                    args=(I_ext,), method="LSODA",
                    rtol=1e-9, atol=1e-12, max_step=0.25, dense_output=True)
    period = 1000.0 / hopf["freq_star_Hz"]
    tt = np.linspace(700 - 4 * period, 700, 2000)
    Y = sol.sol(tt)
    labels = ["$E$", "$I$", "$g_e^I$", "$g_i^E$"]
    pairs = [(0, 1), (2, 3), (0, 3), (1, 2), (0, 2), (1, 3)]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.5), dpi=150)
    for ax, (a, b) in zip(axes.flat, pairs):
        ax.plot(Y[a], Y[b], color=theme.INK_BLACK, lw=1.0)
        ax.set_xlabel(labels[a], fontsize=theme.SIZE_LABEL)
        ax.set_ylabel(labels[b], fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_timeseries(hopf, out_path, run_id, offset=0.4):
    """The four state variables over the limit cycle (issue #38's experiment 1,
    timeseries half): E, g_e^I, I, g_i^E in loop order, sharing a time axis, so
    the round-trip phase lags E -> g_e^I -> I -> g_i^E -> E are visible."""
    theme.apply()
    I_ext = hopf["I_ext_star"] + offset
    fp = fixed_point(I_ext)
    sol = solve_ivp(rhs_4d, (0, 700), fp + np.array([1e-3, 0, 0, 0]),
                    args=(I_ext,), method="LSODA",
                    rtol=1e-9, atol=1e-12, max_step=0.25, dense_output=True)
    period = 1000.0 / hopf["freq_star_Hz"]
    tt = np.linspace(700 - 3 * period, 700, 1500)
    Y = sol.sol(tt)
    t = tt - tt[0]
    # loop order E -> g_e^I -> I -> g_i^E; one trace per panel, so near-black ink
    # throughout (colour would separate nothing — H13).
    rows = [(0, "$E$ rate", theme.INK_BLACK),
            (2, "$g_e^I$", theme.INK_BLACK),
            (1, "$I$ rate", theme.INK_BLACK),
            (3, "$g_i^E$", theme.INK_BLACK)]
    fig, axes = plt.subplots(4, 1, figsize=(8.0, 6.5), dpi=150, sharex=True)
    for ax, (idx, lab, col) in zip(axes, rows):
        ax.plot(t, Y[idx], color=col, lw=1.4)
        ax.set_ylabel(lab, fontsize=theme.SIZE_LABEL)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reduction_ladder(hopf4, hopf3, hopf2, out_path, run_id, I_ext=1.0):
    """g_i^E divergence after a kick at a common supra-threshold drive for
    the full 4D model, the 3D (AMPA-slaved) and 2D (rate-slaved) reductions.
    4D and 3D sustain; 2D rings down — the Hopf survives to 3D, not 2D."""
    theme.apply()
    fp4 = fixed_point(I_ext)
    fp3 = fixed_point_3d_qss(I_ext)
    fp2 = fixed_point_2d_qss(I_ext)
    s4 = solve_ivp(rhs_4d, (0, 400), fp4 + np.array([2e-3, 0, 0, 0]),
                   args=(I_ext,), method="LSODA", rtol=1e-8, atol=1e-11, max_step=0.5)
    s3 = solve_ivp(rhs_3d_qss, (0, 400), fp3 + np.array([2e-3, 0, 0]),
                   args=(I_ext,), method="LSODA", rtol=1e-8, atol=1e-11, max_step=0.5)
    s2 = solve_ivp(rhs_2d_qss, (0, 400), fp2 + np.array([0, 2e-3]),
                   args=(I_ext,), method="LSODA", rtol=1e-8, atol=1e-11, max_step=0.5)
    d4, d3, d2 = s4.y[3] - fp4[3], s3.y[2] - fp3[2], s2.y[1] - fp2[1]
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax.plot(s4.t, d4, color=theme.INK_BLACK, lw=1.4,
            label=f"4D full — Hopf ($f^\\star$ = {hopf4['freq_star_Hz']:.0f} Hz)")
    ax.plot(s3.t, d3, color=theme.INK_BLACK, lw=1.4, ls="--",
            label=f"3D, AMPA slaved — Hopf ($f^\\star$ = {hopf3['freq_star_Hz']:.0f} Hz)"
            if hopf3 else "3D, AMPA slaved")
    ax.plot(s2.t, d2, color=theme.DEEP_RED, lw=1.6,
            label="2D, rates slaved — rings down (no Hopf)")
    ax.axhline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    ax.set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("$g_i^E$ deviation from fixed point", fontsize=theme.SIZE_LABEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    fig.tight_layout()
    stamp_figure(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_bifurcation_compound(results, hopf, sweep, mf, meas, out_path, run_id):
    """Claim-3 anchor: the recruitment cliff as a predictable Hopf bifurcation.
    A — the 4D eigenvalue pair crossing into the right half-plane at I*.
    B — the hysteresis sweep (supercritical, reversible onset).
    C — gamma frequency vs τ_GABA, reference mean-field vs exp041 spiking."""
    theme.apply()
    from matplotlib.gridspec import GridSpec

    # Wide 3:1 strip — a 1×3 panel row reads better near-square per panel than
    # squeezed portrait into 16:9 (deliberate exception to the house ratio).
    fig = plt.figure(figsize=(13.5, 4.6), dpi=150)
    gs = GridSpec(1, 3, figure=fig, wspace=0.42,
                  top=0.86, bottom=0.16, left=0.06, right=0.97)

    # A — eigenvalues in the complex plane, coloured by drive
    axA = fig.add_subplot(gs[0, 0])
    xs = np.array([r["I_ext"] for r in results])
    eig_re = np.array([[e[0] for e in r["eigs"]] for r in results])
    eig_im = np.array([[e[1] for e in r["eigs"]] for r in results])
    sc = None
    for k in range(eig_re.shape[1]):
        sc = axA.scatter(eig_re[:, k], eig_im[:, k], c=xs, cmap="magma", s=4, linewidths=0)
    assert sc is not None  # eig_re has 4 columns, so the loop always assigns sc
    axA.axvline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    if hopf:
        w = hopf["omega_star"]
        axA.scatter([0, 0], [w, -w], facecolors="none",
                    edgecolors=theme.ELECTRIC_CYAN, s=60, lw=1.4, zorder=5)
    cbar = fig.colorbar(sc, ax=axA, fraction=0.046, pad=0.02)
    cbar.set_label("$I_\\text{ext}$ (nA)", fontsize=theme.SIZE_TICK - 1)
    cbar.ax.tick_params(labelsize=theme.SIZE_TICK - 1)
    axA.set_xlabel("Re$(\\lambda)$", fontsize=theme.SIZE_LABEL)
    axA.set_ylabel("Im$(\\lambda)$", fontsize=theme.SIZE_LABEL)
    axA.set_title(f"A  Hopf crossing at $I^\\star$ = {hopf['I_ext_star']:.2f} nA",
                  loc="left", fontsize=theme.SIZE_LABEL, fontweight="semibold")
    _despine(axA)

    # B — hysteresis sweep
    axB = fig.add_subplot(gs[0, 1])
    xu = [d["I_ext"] for d in sweep["up"]]
    yu = [d["amp"] for d in sweep["up"]]
    xd = [d["I_ext"] for d in sweep["down"]]
    yd = [d["amp"] for d in sweep["down"]]
    axB.plot(xu, yu, "o-", color=theme.INK_BLACK, lw=1.2, ms=4, label="drive ↑")
    axB.plot(xd, yd, "s--", color=theme.DEEP_RED, lw=1.0, ms=4,
             markerfacecolor="none", label="drive ↓")
    axB.axvline(hopf["I_ext_star"], color=theme.AMBER, lw=0.6, ls=":")
    axB.set_xlabel("$I_\\text{ext}$ (nA)", fontsize=theme.SIZE_LABEL)
    axB.set_ylabel("E amplitude (pk-pk)", fontsize=theme.SIZE_LABEL)
    axB.set_title("B  Supercritical, reversible onset",
                  loc="left", fontsize=theme.SIZE_LABEL, fontweight="semibold")
    axB.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    _despine(axB)

    # C — frequency vs τ_GABA: mean-field prediction vs spiking
    axC = fig.add_subplot(gs[0, 2])
    tg = [d["tau_gaba_ms"] for d in mf if d["f_star_Hz"] is not None]
    fs = [d["f_star_Hz"] for d in mf if d["f_star_Hz"] is not None]
    axC.plot(tg, fs, "o-", color=theme.INK_BLACK, lw=1.4, label="mean-field $f^\\star$")
    if meas:
        mt = sorted(meas)
        axC.plot(mt, [meas[t] for t in mt], "s--", color=theme.DEEP_RED, lw=1.3,
                 label="spiking $f_\\gamma$ (exp041)")
    axC.set_xlabel("$\\tau_\\text{GABA}$ (ms)", fontsize=theme.SIZE_LABEL)
    axC.set_ylabel("gamma frequency (Hz)", fontsize=theme.SIZE_LABEL)
    axC.set_title("C  Frequency from biophysics",
                  loc="left", fontsize=theme.SIZE_LABEL, fontweight="semibold")
    axC.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    _despine(axC)

    stamp_figure(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def sigma_sensitivity(sigma_grid, coarse_grid, convergence_grid):
    """Evaluate every headline bifurcation quantity across effective noise."""
    rows = []
    for sigma in sigma_grid:
        coarse = sweep(coarse_grid, sigma=sigma)
        hopf = find_hopf(coarse, sigma=sigma)
        fine_hopf = find_hopf(sweep(convergence_grid, sigma=sigma), sigma=sigma)
        if hopf is None:
            rows.append({"sigma_V_mV": float(sigma), "hopf_exists": False})
            continue
        criticality = hysteresis_sweep(
            hopf["I_ext_star"], sigma=sigma,
            span=HYSTERESIS_SPAN_NA, n=HYSTERESIS_POINTS,
        )
        cycle = limit_cycle_metrics(
            hopf, offset=LIMIT_CYCLE_OFFSET_NA, sigma=sigma
        )
        rows.append({
            "sigma_V_mV": float(sigma),
            "hopf_exists": True,
            "hopf": hopf,
            "fixed_point_at_hopf": {
                "E_per_ms": hopf["fp_at_star"][0],
                "I_per_ms": hopf["fp_at_star"][1],
                "g_eI": hopf["fp_at_star"][2],
                "g_iE": hopf["fp_at_star"][3],
            },
            "criticality": {k: v for k, v in criticality.items()
                            if k not in {"up", "down"}},
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
                    if fine_hopf else None
                ),
            },
        })
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
            row.get("criticality", {}).get("verdict") == "supercritical"
            for row in rows
        ),
    }


def plot_sigma_sensitivity(sensitivity, out_path, run_id):
    """Compact audit of threshold, frequency, onset state, and amplitude."""
    rows = [row for row in sensitivity["rows"] if row["hopf_exists"]]
    sigma = [row["sigma_V_mV"] for row in rows]
    theme.apply()
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 4.5), dpi=150)
    ax_threshold, ax_frequency, ax_fixed, ax_amplitude = axes.flat
    ax_threshold.plot(sigma, [r["hopf"]["I_ext_star"] for r in rows], "o-",
                      color=theme.INK_BLACK)
    ax_threshold.set_ylabel("$I^\\star_\\text{ext}$ (nA)")
    ax_frequency.plot(sigma, [r["hopf"]["freq_star_Hz"] for r in rows], "o-",
                      color=theme.INK_BLACK)
    ax_frequency.set_ylabel("$f^\\star$ (Hz)")
    ax_fixed.plot(sigma, [1000 * r["fixed_point_at_hopf"]["E_per_ms"] for r in rows],
                  "o-", color=theme.INK_BLACK, label="E")
    ax_fixed.plot(sigma, [1000 * r["fixed_point_at_hopf"]["I_per_ms"] for r in rows],
                  "s--", color=theme.DEEP_RED, label="I")
    ax_fixed.set_ylabel("fixed-point rate (Hz)")
    ax_fixed.legend(frameon=False, fontsize=theme.SIZE_LEGEND)
    ax_amplitude.plot(
        sigma, [1000 * r["limit_cycle"]["e_peak_to_peak"] for r in rows], "o-",
        color=theme.INK_BLACK,
    )
    ax_amplitude.set_ylabel("E amplitude (Hz, pk-pk)")
    verdict = "supercritical retained" if sensitivity["supercritical_retained"] else "verdict changes"
    ax_amplitude.text(0.02, 0.95, verdict, transform=ax_amplitude.transAxes,
                      ha="left", va="top", fontsize=theme.SIZE_ANNOTATION)
    for ax in axes.flat:
        ax.set_xlabel("$\\sigma_V$ (mV)")
        ax.axvline(SIGMA_V_MV, color=theme.GREY_MID, lw=0.7, ls=":")
        _despine(ax)
    fig.tight_layout()
    stamp_figure(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    run_id = "exp033-numerics"
    log_runner_event(SLUG, "started", run_id=run_id)

    print(f"[{SLUG}] sweeping I_ext (nA) for the reference 4D reduction "
          f"(σ_V = {SIGMA_V_MV} mV)")
    I_grid = np.linspace(0.0, 4.0, 401)
    results = sweep(I_grid)
    hopf = find_hopf(results, sigma=SIGMA_V_MV)
    if hopf:
        print(f"  4D Hopf: I_ext* = {hopf['I_ext_star']:.3f} nA, "
              f"omega* = {hopf['omega_star']:.4f} rad/ms, "
              f"f* = {hopf['freq_star_Hz']:.2f} Hz")
    else:
        print("  4D: no Hopf detected")

    criticality = None
    twod = None
    limitcyc = None
    if hopf:
        criticality = hysteresis_sweep(
            hopf["I_ext_star"], sigma=SIGMA_V_MV,
            span=HYSTERESIS_SPAN_NA, n=HYSTERESIS_POINTS,
        )
        print(f"  criticality: {criticality['verdict']} "
              f"(hysteresis gap {criticality['hyst_gap']:.2e}, "
              f"width {criticality['hyst_width_nA']} nA; "
              f"A² slope {criticality['A2_slope']:.3e}, "
              f"R²={criticality['A2_r2']:.3f})")
        plot_hysteresis(criticality, hopf, FIGURES / "hysteresis.svg", run_id)
        print(f"  wrote {FIGURES / 'hysteresis.svg'}")

        plot_eigenvalues_complex(
            results, hopf, FIGURES / "eigenvalues_complex.svg", run_id)
        print(f"  wrote {FIGURES / 'eigenvalues_complex.svg'}")
        twod = compute_2d_vs_4d(hopf, sigma=SIGMA_V_MV)
        limitcyc = plot_limit_cycle(
            hopf, FIGURES / "limit_cycle.svg", run_id,
            offset=LIMIT_CYCLE_OFFSET_NA, sigma=SIGMA_V_MV,
        )
        print(f"  wrote {FIGURES / 'limit_cycle.svg'}")

        plot_timeseries(hopf, FIGURES / "timeseries.svg", run_id)
        print(f"  wrote {FIGURES / 'timeseries.svg'}")
        plot_phase_planes(hopf, FIGURES / "phase_planes.svg", run_id)
        print(f"  wrote {FIGURES / 'phase_planes.svg'}")

    # Dimensional reductions: how low can the model go? Test all six choices
    # of which variable pair to keep (eliminate the other two by QSS) plus the
    # 3D AMPA-slaved ring.
    two_d_specs = [
        ("keep_E_I (Wilson-Cowan)", rhs_2d, fixed_point_2d_wc),
        ("keep_ge_gi (QSS rates)", rhs_2d_qss, fixed_point_2d_qss),
        ("keep_E_gi (fast/slow)", rhs_2d_fastslow, fixed_point_2d_fastslow),
        ("keep_E_ge", rhs_2d_E_ge, fixed_point_2d_E_ge),
        ("keep_I_gi", rhs_2d_I_gi, fixed_point_2d_I_gi),
        ("keep_I_ge", rhs_2d_I_ge, fixed_point_2d_I_ge),
    ]
    hopf3 = find_hopf(
        reduction_sweep(rhs_3d_qss, fixed_point_3d_qss, I_grid), refine=False
    )
    two_d = {}
    for label, rhs_fn, fp_fn in two_d_specs:
        h = find_hopf(reduction_sweep(rhs_fn, fp_fn, I_grid), refine=False)
        two_d[label] = h
        print(f"  2D {label}: "
              + (f"Hopf I*={h['I_ext_star']:.3f} nA" if h else "no Hopf (rings down)"))
    print("  3D (AMPA slaved): "
          + (f"Hopf I*={hopf3['I_ext_star']:.3f} nA, f*={hopf3['freq_star_Hz']:.2f} Hz"
             if hopf3 else "no Hopf"))
    if hopf:
        plot_reduction_ladder(hopf, hopf3, two_d["keep_ge_gi (QSS rates)"],
                              FIGURES / "reduction_ladder.svg", run_id)
        print(f"  wrote {FIGURES / 'reduction_ladder.svg'}")

    print(f"[{SLUG}] frequency vs tau_GABA (reference vs exp041 spiking)")
    mf_freq = frequency_vs_tau_gaba(
        [4.5, 6.0, 9.0, 12.0, 18.0, 27.0], I_grid, sigma=SIGMA_V_MV
    )
    meas_fgamma = load_exp041_fgamma()
    for d in mf_freq:
        m = meas_fgamma.get(d["tau_gaba_ms"])
        f = d["f_star_Hz"]
        print(f"    tau_GABA={d['tau_gaba_ms']:5}  reference f*="
              + (f"{f:.2f} Hz" if f else "—")
              + (f"  spiking f_gamma={m:.2f} Hz" if m else ""))
    plot_frequency_vs_tau_gaba(
        mf_freq, meas_fgamma, FIGURES / "freq_vs_tau_gaba.svg", run_id)
    print(f"  wrote {FIGURES / 'freq_vs_tau_gaba.svg'}")

    # Claim-3 anchor compound: Hopf crossing + hysteresis + frequency.
    if hopf and criticality:
        fig_bifurcation_compound(
            results, hopf, criticality, mf_freq, meas_fgamma,
            FIGURES / "bifurcation_compound.svg", run_id)
        print(f"  wrote {FIGURES / 'bifurcation_compound.svg'}")

    print(f"[{SLUG}] sigma_V sensitivity")
    sensitivity_grid = np.linspace(0.0, 1.2, 121)
    sensitivity_convergence_grid = np.linspace(0.0, 1.2, 241)
    sensitivity = sigma_sensitivity(
        SIGMA_V_GRID_MV, sensitivity_grid, sensitivity_convergence_grid
    )
    plot_sigma_sensitivity(
        sensitivity, FIGURES / "sigma_sensitivity.svg", run_id
    )
    print(f"  wrote {FIGURES / 'sigma_sensitivity.svg'}")

    summary = {
        "slug": SLUG,
        "config": {
            "tau_E_ms": TAU_E_MS, "tau_I_ms": TAU_I_MS,
            "tau_AMPA_ms": TAU_AMPA_MS, "tau_GABA_ms": TAU_GABA_MS,
            "W_tilde_EI": WT_EI, "W_tilde_IE": WT_IE,
            "dV_inh_mV": DV_INH_MV, "dV_exc_mV": DV_EXC_MV,
            "sigma_V_mV": SIGMA_V_MV,
            "cell_E": CELL_E, "cell_I": CELL_I,
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
                    if hopf else "no Hopf found"
                ),
            },
            {
                "label": "Hopf is supercritical (reversible onset, no hysteresis)",
                "passed": bool(criticality and criticality["verdict"] == "supercritical"),
                "detail": (
                    f"up/down sweeps coincide (gap {criticality['hyst_gap']:.2e}, "
                    f"width {criticality['hyst_width_nA']} nA); "
                    f"A² ∝ (I−I*) slope {criticality['A2_slope']:.3e}, "
                    f"R² = {criticality['A2_r2']:.3f}"
                    if criticality else "not evaluated"
                ),
            },
            {
                "label": "2D Wilson-Cowan reduction cannot sustain (rings down)",
                "passed": bool(twod and twod["pp_2d"] < 1e-4 <= twod["pp_4d"]),
                "detail": (
                    f"at I*+{twod['I_ext'] - hopf['I_ext_star']:.2g} nA: "
                    f"4D peak-to-peak {twod['pp_4d']:.3e}, 2D {twod['pp_2d']:.3e}"
                    if twod else "not evaluated"
                ),
            },
            {
                "label": "Minimal dimension is 3: 3D-by-QSS keeps the Hopf, all six 2D pairs lose it",
                "passed": bool(hopf3 is not None
                               and all(v is None for v in two_d.values())),
                "detail": (
                    f"3D (AMPA slaved): Hopf at I*={hopf3['I_ext_star']:.3f} nA, "
                    f"f*={hopf3['freq_star_Hz']:.2f} Hz; "
                    f"2D pairs with a Hopf: "
                    f"{[k for k, v in two_d.items() if v] or 'none (all six ring down)'}"
                    if hopf3 else "3D Hopf not found"
                ),
            },
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  wrote {FIGURES / 'numbers.json'}")
    log_runner_event(SLUG, "completed", run_id=run_id)


if __name__ == "__main__":
    main()
