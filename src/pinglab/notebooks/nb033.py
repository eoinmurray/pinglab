"""033 — Wilson-Cowan mean-field bifurcation analysis of PING (4D).

Numerics for the theory in src/docs/src/pages/notebooks/nb033.mdx.

Implements the 4D mean-field reduction of the PING spiking network in
state (E, I, g_e^I, g_i^E) — rates plus synaptic conductances. Sweeps
the external drive I_ext, tracks fixed points, computes the Jacobian's
eigenvalues, identifies the Hopf bifurcation (smallest I_ext* where
max Re(λ) = 0), and tests super- vs subcritical character by direct
ODE simulation around the bifurcation.

The 2D-cubic alternative reduction is explored in nb034.

Outputs figures and numbers.json to
src/docs/public/figures/notebooks/nb033/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

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

# ── Biophysical parameters (units: ms, current units) ─────────────────
TAU_E_MS = 20.0    # E membrane
TAU_I_MS = 5.0     # I membrane
TAU_AMPA_MS = 2.0
TAU_GABA_MS = 9.0

# Sigmoid gain parameters chosen so loop gain crosses unity within
# the sweep range.
PHI_E_RMAX = 0.20   # 1/ms ≈ 200 Hz saturation
PHI_E_THETA = 1.5
PHI_E_K = 0.2

PHI_I_RMAX = 0.30
PHI_I_THETA = 1.0
PHI_I_K = 0.15

# Coupling magnitudes in the Wilson-Cowan 1972 regime.
W_EI = 80.0
W_IE = 60.0


def phi(x, rmax, theta, k):
    return rmax / (1.0 + np.exp(-(x - theta) / k))


def phi_p(x, rmax, theta, k):
    s = phi(x, rmax, theta, k) / rmax
    return (rmax / k) * s * (1 - s)


def PhiE(x): return phi(x, PHI_E_RMAX, PHI_E_THETA, PHI_E_K)
def PhiE_p(x): return phi_p(x, PHI_E_RMAX, PHI_E_THETA, PHI_E_K)
def PhiI(x): return phi(x, PHI_I_RMAX, PHI_I_THETA, PHI_I_K)
def PhiI_p(x): return phi_p(x, PHI_I_RMAX, PHI_I_THETA, PHI_I_K)


def rhs_4d(t, y, I_ext, w_ei=W_EI, w_ie=W_IE):
    """4D rate ODEs in state (E, I, g_e^I, g_i^E)."""
    E, I, g_eI, g_iE = y
    return [
        (-E + PhiE(I_ext - g_iE)) / TAU_E_MS,
        (-I + PhiI(g_eI)) / TAU_I_MS,
        (-g_eI + w_ei * E) / TAU_AMPA_MS,
        (-g_iE + w_ie * I) / TAU_GABA_MS,
    ]


def fixed_point(I_ext, x0=None, w_ei=W_EI, w_ie=W_IE):
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

    sol, _, ier, _ = fsolve(residual, x0, full_output=True)
    return sol if ier == 1 else None


def jacobian(fp, I_ext, w_ei=W_EI, w_ie=W_IE):
    E, I, g_eI, g_iE = fp
    pE = PhiE_p(I_ext - g_iE)
    pI = PhiI_p(g_eI)
    return np.array([
        [-1.0 / TAU_E_MS, 0.0, 0.0, -pE / TAU_E_MS],
        [0.0, -1.0 / TAU_I_MS, pI / TAU_I_MS, 0.0],
        [w_ei / TAU_AMPA_MS, 0.0, -1.0 / TAU_AMPA_MS, 0.0],
        [0.0, w_ie / TAU_GABA_MS, 0.0, -1.0 / TAU_GABA_MS],
    ])


def sweep(I_ext_grid, w_ei=W_EI, w_ie=W_IE):
    results = []
    x = None
    for I_ext in I_ext_grid:
        fp = fixed_point(I_ext, x0=x, w_ei=w_ei, w_ie=w_ie)
        if fp is None:
            continue
        x = fp
        J = jacobian(fp, I_ext, w_ei=w_ei, w_ie=w_ie)
        eigs = linalg.eigvals(J)
        results.append({
            "I_ext": float(I_ext),
            "fp": fp.tolist(),
            "eigs": [(float(e.real), float(e.imag)) for e in eigs],
        })
    return results


def find_hopf(results):
    """Smallest I_ext at which max Re(eig) becomes ≥ 0."""
    prev_max = None
    for r in results:
        re_max = max(e[0] for e in r["eigs"])
        if prev_max is not None and prev_max < 0 <= re_max:
            cand = [(e[0], e[1]) for e in r["eigs"]
                    if abs(e[1]) > 1e-6 and e[0] >= -1e-2]
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


def amplitude_at(I_ext, t_max=2000.0, t_settle=1500.0,
                  w_ei=W_EI, w_ie=W_IE):
    """Integrate ODE from a small perturbation of the FP; measure
    asymptotic peak-to-peak E amplitude."""
    fp = fixed_point(I_ext, w_ei=w_ei, w_ie=w_ie)
    if fp is None:
        return 0.0
    y0 = [v + 0.005 for v in fp]
    sol = solve_ivp(
        rhs_4d, (0, t_max), y0, args=(I_ext, w_ei, w_ie),
        method="LSODA", rtol=1e-6, atol=1e-9, max_step=1.0,
    )
    if not sol.success:
        return 0.0
    mask = sol.t >= t_settle
    E = sol.y[0][mask]
    if E.size < 10:
        return 0.0
    return float(E.max() - E.min())


def _stamp(fig, run_id):
    fig.text(
        0.995, 0.005, run_id, ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


def plot_eigenvalue_trajectories(results, hopf, out_path, run_id):
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    xs = [r["I_ext"] for r in results]
    eigs_re = np.array([[e[0] for e in r["eigs"]] for r in results])
    eigs_im = np.array([[e[1] for e in r["eigs"]] for r in results])
    for k in range(eigs_re.shape[1]):
        color = theme.INK_BLACK if abs(eigs_im[0, k]) < 1e-6 else theme.DEEP_RED
        ax.plot(xs, eigs_re[:, k], color=color, lw=1.2, alpha=0.7)
    ax.axhline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    if hopf:
        ax.axvline(
            hopf["I_ext_star"], color=theme.AMBER, lw=1.0, ls="--",
            label=(f"$I^\\star = {hopf['I_ext_star']:.2f}$, "
                   f"$f^\\star = {hopf['freq_star_Hz']:.1f}$ Hz"),
        )
        ax.legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="upper left")
    ax.set_xlabel("$I_\\text{ext}$", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Re$(\\lambda)$", fontsize=theme.SIZE_LABEL)
    ax.set_title("4D system eigenvalues", fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    _stamp(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_criticality(amps, hopf, out_path, run_id):
    theme.apply()
    fig, (ax_a, ax_a2) = plt.subplots(1, 2, figsize=(12.0, 4.5), dpi=150)
    xs = [d["I_ext"] for d in amps]
    ys = [d["amp"] for d in amps]
    i_star = hopf["I_ext_star"]
    ax_a.plot(xs, ys, marker="o", color=theme.INK_BLACK, lw=1.0)
    ax_a.axvline(i_star, color=theme.AMBER, lw=0.6, ls=":")
    ax_a.set_xlabel("$I_\\text{ext}$", fontsize=theme.SIZE_LABEL)
    ax_a.set_ylabel("E amplitude (peak-to-peak)", fontsize=theme.SIZE_LABEL)
    ax_a.set_title("Oscillation amplitude vs drive", fontsize=theme.SIZE_TITLE)
    xs_rel = [d["I_ext"] - i_star for d in amps]
    ys_sq = [d["amp"]**2 for d in amps]
    ax_a2.plot(xs_rel, ys_sq, marker="o", color=theme.INK_BLACK, lw=1.0)
    ax_a2.set_xlabel("$I_\\text{ext} - I^\\star$", fontsize=theme.SIZE_LABEL)
    ax_a2.set_ylabel("E amplitude$^2$", fontsize=theme.SIZE_LABEL)
    ax_a2.set_title("Supercritical signature: $A^2 \\propto (I-I^\\star)$",
                    fontsize=theme.SIZE_TITLE)
    ax_a2.axhline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    ax_a2.axvline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    fig.suptitle("Hopf criticality test by direct simulation",
                 fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    _stamp(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    run_id = "nb033-numerics"

    print(f"[{SLUG}] Sweeping I_ext for 4D reduction")
    I_grid = np.linspace(0, 12.0, 241)
    results = sweep(I_grid)
    hopf = find_hopf(results)
    if hopf:
        print(f"  4D Hopf: I_ext* = {hopf['I_ext_star']:.3f}, "
              f"omega* = {hopf['omega_star']:.3f} rad/ms, "
              f"f* = {hopf['freq_star_Hz']:.2f} Hz")
    else:
        print("  4D: no Hopf detected")

    plot_eigenvalue_trajectories(
        results, hopf,
        FIGURES / "eigenvalue_trajectories.png", run_id,
    )
    print(f"  wrote {FIGURES / 'eigenvalue_trajectories.png'}")

    amp_data = []
    if hopf:
        i_star = hopf["I_ext_star"]
        deltas = np.linspace(-0.1, 0.5, 13)
        for dx in deltas:
            amp = amplitude_at(i_star + dx)
            amp_data.append({"I_ext": i_star + dx, "amp": amp})
            print(f"    4D amp at I={i_star + dx:.3f}: {amp:.4f}")
        plot_criticality(amp_data, hopf, FIGURES / "criticality.png", run_id)
        print(f"  wrote {FIGURES / 'criticality.png'}")

    summary = {
        "slug": SLUG,
        "config": {
            "tau_E_ms": TAU_E_MS, "tau_I_ms": TAU_I_MS,
            "tau_AMPA_ms": TAU_AMPA_MS, "tau_GABA_ms": TAU_GABA_MS,
            "W_EI": W_EI, "W_IE": W_IE,
            "phi_E": dict(rmax=PHI_E_RMAX, theta=PHI_E_THETA, k=PHI_E_K),
            "phi_I": dict(rmax=PHI_I_RMAX, theta=PHI_I_THETA, k=PHI_I_K),
        },
        "results": {"hopf": hopf},
        "success_criteria": [
            {
                "label": "4D Hopf located",
                "passed": hopf is not None,
                "detail": (f"I_ext* = {hopf['I_ext_star']:.3f}, "
                           f"f* = {hopf['freq_star_Hz']:.2f} Hz"
                           if hopf else "no Hopf"),
            },
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  wrote {FIGURES / 'numbers.json'}")


if __name__ == "__main__":
    main()
