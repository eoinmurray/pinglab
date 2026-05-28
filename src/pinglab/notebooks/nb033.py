"""033 — Wilson-Cowan mean-field bifurcation analysis of PING.

Numerics for the theory developed in src/docs/src/pages/notebooks/nb033.mdx.

Implements two mean-field reductions of the PING spiking network:
  1. 4D system: (E, I, g_e^I, g_i^E) — rates plus synaptic conductances.
  2. 2D-cubic system: (E, I) with explicit cubic E^3 self-feedback,
     derived by Taylor-expanding the sigmoid gain.

For each reduction, sweeps the external drive I_ext, tracks fixed
points, computes the Jacobian's eigenvalues, identifies the Hopf
bifurcation (smallest I_ext* where max Re(λ) = 0), and tests
super- vs subcritical character by direct ODE simulation around
the bifurcation.

Outputs figures and numbers.json to
src/docs/public/figures/notebooks/nb033/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))
import theme  # noqa: E402

SLUG = "nb033"
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG

# ── Biophysical parameters (units: ms, mV-equivalents, current units) ─────
TAU_E_MS = 20.0    # E membrane
TAU_I_MS = 5.0     # I membrane
TAU_AMPA_MS = 2.0
TAU_GABA_MS = 9.0

# Sigmoid gain parameters. Inflection set within the I_ext sweep range
# so the operating point can move across it. Steepness chosen so that
# max gain Phi'(theta) = r_max/(4 k) is large enough to overcome the
# product of damping rates around the E→I→E loop.
PHI_E_RMAX = 0.20   # 1/ms ≈ 200 Hz saturation
PHI_E_THETA = 1.5   # inflection point
PHI_E_K = 0.2       # steeper sigmoid → larger peak gain

PHI_I_RMAX = 0.30   # 1/ms ≈ 300 Hz saturation
PHI_I_THETA = 1.0
PHI_I_K = 0.15

# Coupling magnitudes in the Wilson-Cowan 1972 regime (their c_i ~ 4-13;
# we use larger W because our rates are in 1/ms not proportions ∈ [0,1]).
# The product W^EI * W^IE * Phi'_E(theta) * Phi'_I(theta) needs to be
# big enough relative to the product of decay rates around the loop.
W_EI = 80.0
W_IE = 60.0


# ── Gain function and its derivatives ──────────────────────────────────

def phi(x: np.ndarray, rmax: float, theta: float, k: float) -> np.ndarray:
    """Sigmoid gain in 1/ms (rate units)."""
    return rmax / (1.0 + np.exp(-(x - theta) / k))


def phi_p(x: np.ndarray, rmax: float, theta: float, k: float) -> np.ndarray:
    """First derivative dPhi/dx."""
    s = phi(x, rmax, theta, k) / rmax
    return (rmax / k) * s * (1 - s)


def phi_pp(x, rmax, theta, k):
    """Second derivative — zero at the inflection point x = theta."""
    s = phi(x, rmax, theta, k) / rmax
    return (rmax / (k * k)) * s * (1 - s) * (1 - 2 * s)


def phi_ppp(x, rmax, theta, k):
    """Third derivative."""
    s = phi(x, rmax, theta, k) / rmax
    return (rmax / (k**3)) * s * (1 - s) * (1 - 6 * s * (1 - s))


# Convenience wrappers
def PhiE(x): return phi(x, PHI_E_RMAX, PHI_E_THETA, PHI_E_K)
def PhiE_p(x): return phi_p(x, PHI_E_RMAX, PHI_E_THETA, PHI_E_K)
def PhiE_pp(x): return phi_pp(x, PHI_E_RMAX, PHI_E_THETA, PHI_E_K)
def PhiE_ppp(x): return phi_ppp(x, PHI_E_RMAX, PHI_E_THETA, PHI_E_K)
def PhiI(x): return phi(x, PHI_I_RMAX, PHI_I_THETA, PHI_I_K)
def PhiI_p(x): return phi_p(x, PHI_I_RMAX, PHI_I_THETA, PHI_I_K)


# ── 4D system ─────────────────────────────────────────────────────────

def rhs_4d(t, y, I_ext: float, w_ei: float = W_EI, w_ie: float = W_IE):
    """4D rate ODEs in state (E, I, g_e^I, g_i^E)."""
    E, I, g_eI, g_iE = y
    return [
        (-E + PhiE(I_ext - g_iE)) / TAU_E_MS,
        (-I + PhiI(g_eI)) / TAU_I_MS,
        (-g_eI + w_ei * E) / TAU_AMPA_MS,
        (-g_iE + w_ie * I) / TAU_GABA_MS,
    ]


def fixed_point_4d(I_ext, x0=None, w_ei=W_EI, w_ie=W_IE):
    """Find a steady state of the 4D system via fsolve."""
    if x0 is None:
        x0 = [0.001, 0.001, 0.001, 0.001]

    def residual(x):
        E, I, g_eI, g_iE = x
        return [
            -E + PhiE(I_ext - g_iE),
            -I + PhiI(g_eI),
            -g_eI + w_ei * E,
            -g_iE + w_ie * I,
        ]

    sol, info, ier, _msg = fsolve(residual, x0, full_output=True)
    if ier != 1:
        return None
    return sol


def jacobian_4d(fp, I_ext, w_ei=W_EI, w_ie=W_IE):
    """Jacobian of the 4D system at the fixed point."""
    E, I, g_eI, g_iE = fp
    pE = PhiE_p(I_ext - g_iE)
    pI = PhiI_p(g_eI)
    return np.array([
        [-1.0 / TAU_E_MS, 0.0, 0.0, -pE / TAU_E_MS],
        [0.0, -1.0 / TAU_I_MS, pI / TAU_I_MS, 0.0],
        [w_ei / TAU_AMPA_MS, 0.0, -1.0 / TAU_AMPA_MS, 0.0],
        [0.0, w_ie / TAU_GABA_MS, 0.0, -1.0 / TAU_GABA_MS],
    ])


def sweep_4d(I_ext_grid, w_ei=W_EI, w_ie=W_IE):
    """Sweep I_ext; return list of (I_ext, fp, eigenvalues)."""
    results = []
    x = None
    for I_ext in I_ext_grid:
        fp = fixed_point_4d(I_ext, x0=x, w_ei=w_ei, w_ie=w_ie)
        if fp is None:
            continue
        x = fp  # warm-start next solve
        J = jacobian_4d(fp, I_ext, w_ei=w_ei, w_ie=w_ie)
        eigs = linalg.eigvals(J)
        results.append({
            "I_ext": float(I_ext),
            "fp": fp.tolist(),
            "eigs": [(float(e.real), float(e.imag)) for e in eigs],
        })
    return results


def find_hopf(results):
    """Locate the smallest I_ext at which max Re(eig) becomes >= 0.
    Returns dict with I_ext_star, omega_star, eigenvalues at the bif."""
    prev_max = None
    for r in results:
        re_max = max(e[0] for e in r["eigs"])
        if prev_max is not None and prev_max < 0 <= re_max:
            # Bifurcation between this and previous point — interpolate.
            # For simplicity, take this point.
            # Identify the bifurcating complex pair.
            cand = [
                (e[0], e[1]) for e in r["eigs"]
                if abs(e[1]) > 1e-6 and e[0] >= -1e-2
            ]
            if not cand:
                continue
            cand.sort(key=lambda x: -x[0])
            re, im = cand[0]
            return {
                "I_ext_star": r["I_ext"],
                "omega_star": abs(im),
                "freq_star_Hz": 1000.0 * abs(im) / (2 * np.pi),
                "fp_at_star": r["fp"],
            }
        prev_max = re_max
    return None


# ── 2D-cubic system ──────────────────────────────────────────────────

# Coefficients of the cubic-in-E reduction, fit to PhiE via Taylor
# expansion at the inflection point. We expand:
#   PhiE(I_ext - W^{IE} I) ≈ PhiE(theta) + a(...) + b(...)^3
# where a = PhiE_p(theta), b = PhiE_ppp(theta)/6 (cubic Taylor coeff).
# Then use self-consistency at the fixed point to write the cubic in
# terms of E itself. The constants below are the linearised form
# coefficients (treating I_ext as the bifurcation parameter).

def cubic_coefficients():
    """Return (alpha, beta, gamma, delta) for the 2D-cubic reduction.
    Derived from the sigmoid expansion at the inflection point."""
    a = PhiE_p(PHI_E_THETA)              # linear gain at inflection
    b = PhiE_ppp(PHI_E_THETA) / 6.0      # cubic Taylor coefficient
    # In the reduced form  τ_E dE/dt = -E + α E - β E^3 - γ I + I_ext
    # α plays the role of effective gain (positive feedback when α > 1).
    # β > 0 stabilises at large E.
    # γ couples I → E suppression.
    # We absorb a into α and -b into β so that the reduced form is
    # universal in shape; constants come from the gain expansion.
    alpha = a              # effective linear gain
    beta = -b              # cubic coefficient; b < 0 for sigmoid so beta > 0
    gamma = a * W_IE       # γ = a * W^{IE} from the chain rule
    delta = PhiI_p(0.0) * W_EI  # δ from linearising PhiI at zero
    return alpha, beta, gamma, delta


def rhs_2d(t, y, I_ext, alpha, beta, gamma, delta):
    """2D-cubic ODE in state (E, I)."""
    E, I = y
    return [
        (-E + alpha * E - beta * E**3 - gamma * I + alpha * I_ext) / TAU_E_MS,
        (-I + delta * E) / TAU_I_MS,
    ]


def fixed_point_2d(I_ext, x0=None, *, alpha, beta, gamma, delta):
    if x0 is None:
        x0 = [0.001, 0.001]

    def residual(x):
        E, I = x
        return [
            -E + alpha * E - beta * E**3 - gamma * I + alpha * I_ext,
            -I + delta * E,
        ]

    sol, info, ier, _msg = fsolve(residual, x0, full_output=True)
    if ier != 1:
        return None
    return sol


def jacobian_2d(fp, alpha, beta, gamma, delta):
    E, I = fp
    return np.array([
        [(-1.0 + alpha - 3 * beta * E * E) / TAU_E_MS, -gamma / TAU_E_MS],
        [delta / TAU_I_MS, -1.0 / TAU_I_MS],
    ])


def sweep_2d(I_ext_grid, alpha, beta, gamma, delta):
    results = []
    x = None
    for I_ext in I_ext_grid:
        fp = fixed_point_2d(I_ext, x0=x, alpha=alpha, beta=beta,
                            gamma=gamma, delta=delta)
        if fp is None:
            continue
        x = fp
        J = jacobian_2d(fp, alpha, beta, gamma, delta)
        eigs = linalg.eigvals(J)
        results.append({
            "I_ext": float(I_ext),
            "fp": fp.tolist(),
            "eigs": [(float(e.real), float(e.imag)) for e in eigs],
        })
    return results


# ── Criticality test by direct simulation ─────────────────────────────

def amplitude_at(I_ext, system="4d", w_ei=W_EI, w_ie=W_IE,
                  alpha=None, beta=None, gamma=None, delta=None,
                  t_max=2000.0, t_settle=1500.0):
    """Integrate ODE from a small perturbation of the silent FP for
    `t_max` ms, then measure peak-to-peak E amplitude over the last
    (t_max - t_settle) ms. Detects limit cycle vs decay."""
    if system == "4d":
        fp = fixed_point_4d(I_ext, w_ei=w_ei, w_ie=w_ie)
        if fp is None:
            return 0.0
        y0 = [v + 0.005 for v in fp]   # small perturbation
        sol = solve_ivp(
            rhs_4d, (0, t_max), y0, args=(I_ext, w_ei, w_ie),
            method="LSODA", rtol=1e-6, atol=1e-9,
            dense_output=False, max_step=1.0,
        )
        if not sol.success:
            return 0.0
        mask = sol.t >= t_settle
        E = sol.y[0][mask]
    else:  # 2d
        fp = fixed_point_2d(I_ext, alpha=alpha, beta=beta,
                            gamma=gamma, delta=delta)
        if fp is None:
            return 0.0
        y0 = [v + 0.005 for v in fp]
        sol = solve_ivp(
            rhs_2d, (0, t_max), y0,
            args=(I_ext, alpha, beta, gamma, delta),
            method="LSODA", rtol=1e-6, atol=1e-9,
            max_step=1.0,
        )
        if not sol.success:
            return 0.0
        mask = sol.t >= t_settle
        E = sol.y[0][mask]
    if E.size < 10:
        return 0.0
    return float(E.max() - E.min())


# ── Plotting ──────────────────────────────────────────────────────────

def _stamp(fig, run_id):
    fig.text(
        0.995, 0.005, run_id, ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


def plot_eigenvalue_trajectories(res_4d, res_2d, hopf_4d, hopf_2d,
                                  out_path, run_id):
    """Two-panel: Re(eigenvalues) vs I_ext for each reduction. Vertical
    line at I_ext* (Hopf) for each."""
    theme.apply()
    fig, (ax_4d, ax_2d) = plt.subplots(1, 2, figsize=(12.0, 4.5), dpi=150)

    # 4D: 4 eigenvalues per I_ext
    xs = [r["I_ext"] for r in res_4d]
    eigs_re = np.array([[e[0] for e in r["eigs"]] for r in res_4d])
    eigs_im = np.array([[e[1] for e in r["eigs"]] for r in res_4d])
    for k in range(eigs_re.shape[1]):
        color = theme.INK_BLACK if abs(eigs_im[0, k]) < 1e-6 else theme.DEEP_RED
        ax_4d.plot(xs, eigs_re[:, k], color=color, lw=1.2, alpha=0.7)
    ax_4d.axhline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    if hopf_4d:
        ax_4d.axvline(hopf_4d["I_ext_star"], color=theme.AMBER, lw=1.0, ls="--",
                       label=f"$I^\\star = {hopf_4d['I_ext_star']:.2f}$, "
                             f"$f^\\star = {hopf_4d['freq_star_Hz']:.1f}$ Hz")
        ax_4d.legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="upper left")
    ax_4d.set_xlabel("$I_\\text{ext}$", fontsize=theme.SIZE_LABEL)
    ax_4d.set_ylabel("Re$(\\lambda)$", fontsize=theme.SIZE_LABEL)
    ax_4d.set_title("4D system eigenvalues", fontsize=theme.SIZE_TITLE)

    # 2D: 2 eigenvalues per I_ext
    xs2 = [r["I_ext"] for r in res_2d]
    eigs2_re = np.array([[e[0] for e in r["eigs"]] for r in res_2d])
    eigs2_im = np.array([[e[1] for e in r["eigs"]] for r in res_2d])
    for k in range(eigs2_re.shape[1]):
        color = theme.INK_BLACK if abs(eigs2_im[0, k]) < 1e-6 else theme.DEEP_RED
        ax_2d.plot(xs2, eigs2_re[:, k], color=color, lw=1.2, alpha=0.7)
    ax_2d.axhline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    if hopf_2d:
        ax_2d.axvline(hopf_2d["I_ext_star"], color=theme.AMBER, lw=1.0, ls="--",
                       label=f"$I^\\star = {hopf_2d['I_ext_star']:.2f}$, "
                             f"$f^\\star = {hopf_2d['freq_star_Hz']:.1f}$ Hz")
        ax_2d.legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="upper left")
    ax_2d.set_xlabel("$I_\\text{ext}$", fontsize=theme.SIZE_LABEL)
    ax_2d.set_ylabel("Re$(\\lambda)$", fontsize=theme.SIZE_LABEL)
    ax_2d.set_title("2D-cubic system eigenvalues", fontsize=theme.SIZE_TITLE)

    fig.suptitle(
        "Eigenvalue trajectories of the mean-field reductions",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_criticality(amp_data, out_path, run_id):
    """Two-panel: oscillation amplitude vs I_ext for both reductions.
    Supercritical Hopf gives sqrt onset (A^2 linear in I_ext - I_ext*)."""
    theme.apply()
    fig, (ax_a, ax_a2) = plt.subplots(1, 2, figsize=(12.0, 4.5), dpi=150)
    for label, color, marker, data in amp_data:
        xs = [d["I_ext"] for d in data]
        ys = [d["amp"] for d in data]
        i_star = data[0]["I_ext_star"]
        ax_a.plot(xs, ys, marker=marker, color=color, lw=1.0, label=label)
        ax_a.axvline(i_star, color=color, lw=0.6, ls=":", alpha=0.6)
        # A^2 vs (I - I*) panel
        xs_rel = [d["I_ext"] - i_star for d in data]
        ys_sq = [d["amp"]**2 for d in data]
        ax_a2.plot(xs_rel, ys_sq, marker=marker, color=color, lw=1.0,
                   label=label)
    ax_a.set_xlabel("$I_\\text{ext}$", fontsize=theme.SIZE_LABEL)
    ax_a.set_ylabel("E amplitude (peak-to-peak)", fontsize=theme.SIZE_LABEL)
    ax_a.set_title("Oscillation amplitude vs drive",
                   fontsize=theme.SIZE_TITLE)
    ax_a.legend(fontsize=theme.SIZE_LABEL, frameon=False)
    ax_a2.set_xlabel("$I_\\text{ext} - I^\\star$",
                     fontsize=theme.SIZE_LABEL)
    ax_a2.set_ylabel("E amplitude$^2$", fontsize=theme.SIZE_LABEL)
    ax_a2.set_title("Supercritical signature: $A^2 \\propto (I-I^\\star)$",
                    fontsize=theme.SIZE_TITLE)
    ax_a2.legend(fontsize=theme.SIZE_LABEL, frameon=False)
    ax_a2.axhline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    ax_a2.axvline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    fig.suptitle("Hopf criticality test by direct simulation",
                 fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    _stamp(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    run_id = "nb033-numerics"

    print(f"[{SLUG}] Sweeping I_ext for 4D and 2D-cubic reductions")
    I_grid = np.linspace(0, 12.0, 241)

    # 4D sweep
    res_4d = sweep_4d(I_grid)
    hopf_4d = find_hopf(res_4d)
    if hopf_4d:
        print(f"  4D: I_ext* = {hopf_4d['I_ext_star']:.3f}, "
              f"omega* = {hopf_4d['omega_star']:.3f} rad/ms, "
              f"f* = {hopf_4d['freq_star_Hz']:.2f} Hz")
    else:
        print("  4D: no Hopf detected in sweep")

    # 2D-cubic
    alpha, beta, gamma, delta = cubic_coefficients()
    print(f"  Cubic coefficients: α={alpha:.4f}, β={beta:.4f}, "
          f"γ={gamma:.4f}, δ={delta:.4f}")
    res_2d = sweep_2d(I_grid, alpha, beta, gamma, delta)
    hopf_2d = find_hopf(res_2d)
    if hopf_2d:
        print(f"  2D: I_ext* = {hopf_2d['I_ext_star']:.3f}, "
              f"omega* = {hopf_2d['omega_star']:.3f} rad/ms, "
              f"f* = {hopf_2d['freq_star_Hz']:.2f} Hz")
    else:
        print("  2D-cubic: no Hopf detected in sweep")

    # Eigenvalue figure
    plot_eigenvalue_trajectories(
        res_4d, res_2d, hopf_4d, hopf_2d,
        FIGURES / "eigenvalue_trajectories.png", run_id,
    )
    print(f"  wrote {FIGURES / 'eigenvalue_trajectories.png'}")

    # Criticality test: integrate ODE at I_ext just past each bifurcation
    amp_data = []
    if hopf_4d:
        i_star = hopf_4d["I_ext_star"]
        deltas = np.linspace(-0.1, 0.5, 13)
        amps = []
        for dx in deltas:
            amp = amplitude_at(i_star + dx, system="4d")
            amps.append({"I_ext": i_star + dx, "amp": amp,
                          "I_ext_star": i_star})
            print(f"    4D amp at I={i_star + dx:.3f}: {amp:.4f}")
        amp_data.append(("4D", theme.INK_BLACK, "o", amps))
    if hopf_2d:
        i_star = hopf_2d["I_ext_star"]
        deltas = np.linspace(-0.1, 0.5, 13)
        amps = []
        for dx in deltas:
            amp = amplitude_at(i_star + dx, system="2d",
                                alpha=alpha, beta=beta,
                                gamma=gamma, delta=delta)
            amps.append({"I_ext": i_star + dx, "amp": amp,
                          "I_ext_star": i_star})
            print(f"    2D amp at I={i_star + dx:.3f}: {amp:.4f}")
        amp_data.append(("2D-cubic", theme.DEEP_RED, "s", amps))

    if amp_data:
        plot_criticality(amp_data, FIGURES / "criticality.png", run_id)
        print(f"  wrote {FIGURES / 'criticality.png'}")

    # Write summary
    summary = {
        "slug": SLUG,
        "config": {
            "tau_E_ms": TAU_E_MS, "tau_I_ms": TAU_I_MS,
            "tau_AMPA_ms": TAU_AMPA_MS, "tau_GABA_ms": TAU_GABA_MS,
            "W_EI": W_EI, "W_IE": W_IE,
            "phi_E": dict(rmax=PHI_E_RMAX, theta=PHI_E_THETA, k=PHI_E_K),
            "phi_I": dict(rmax=PHI_I_RMAX, theta=PHI_I_THETA, k=PHI_I_K),
            "I_ext_grid": [float(I_grid.min()), float(I_grid.max()),
                            int(I_grid.size)],
        },
        "results": {
            "hopf_4d": hopf_4d,
            "hopf_2d": hopf_2d,
            "cubic_coefficients": dict(alpha=alpha, beta=beta,
                                        gamma=gamma, delta=delta),
        },
        "success_criteria": [
            {
                "label": "4D Hopf located",
                "passed": hopf_4d is not None,
                "detail": (f"I_ext* = {hopf_4d['I_ext_star']:.3f}"
                            if hopf_4d else "no Hopf"),
            },
            {
                "label": "2D-cubic Hopf located",
                "passed": hopf_2d is not None,
                "detail": (f"I_ext* = {hopf_2d['I_ext_star']:.3f}"
                            if hopf_2d else "no Hopf"),
            },
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  wrote {FIGURES / 'numbers.json'}")


if __name__ == "__main__":
    main()
