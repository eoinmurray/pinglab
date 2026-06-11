"""033 — Mean-field PING: 4D bifurcations and the 2D-cubic dead end.

Numerics for the theory in src/docs/src/pages/notebooks/nb033.mdx.

Two analyses on the same parameters:

1. **4D reduction** (state $(E, I, g_e^I, g_i^E)$). Sweeps the external
   drive $I_\text{ext}$, tracks fixed points, computes the Jacobian's
   eigenvalues, identifies the Hopf (smallest $I_\text{ext}^\star$ where
   max Re(λ) = 0), and tests super- vs subcritical character by direct
   ODE simulation around the bifurcation. This is the load-bearing
   reduction matched against the empirical PING gamma frequency.

2. **2D-cubic reduction** (FitzHugh-Nagumo-style polynomial system,
   adiabatically eliminating the synapses and Taylor-expanding the E
   gain at its inflection). Demonstrates that the textbook 2D collapse
   cannot Hopf for this architecture — the linear self-feedback
   coefficient α stays below the critical 1 + τ_E/τ_I because
   W^EE = 0 leaves no positive linear term on the diagonal. The
   cubic stabilises amplitude but cannot destabilise the fixed point.

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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from cli import theme  # noqa: E402

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


def phi_ppp(x, rmax, theta, k):
    s = phi(x, rmax, theta, k) / rmax
    return (rmax / (k**3)) * s * (1 - s) * (1 - 6 * s * (1 - s))


def PhiE(x): return phi(x, PHI_E_RMAX, PHI_E_THETA, PHI_E_K)
def PhiE_p(x): return phi_p(x, PHI_E_RMAX, PHI_E_THETA, PHI_E_K)
def PhiE_ppp(x): return phi_ppp(x, PHI_E_RMAX, PHI_E_THETA, PHI_E_K)
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


def classify_criticality(amp_data, hopf):
    """Super- vs subcritical from the amplitude-vs-drive curve.

    Supercritical Hopf: the limit cycle is born with zero amplitude and
    grows continuously, with A^2 ∝ (I - I*) just above threshold and no
    oscillation below it. Subcritical: amplitude appears discontinuously
    (a finite jump / hysteresis), so the network is silent below I* but
    a perturbation does not relax to a small cycle.
    """
    i_star = hopf["I_ext_star"]
    below = [d["amp"] for d in amp_data if d["I_ext"] < i_star - 1e-9]
    above = [(d["I_ext"] - i_star, d["amp"]) for d in amp_data
             if d["I_ext"] > i_star + 1e-9]
    amp_below = max(below) if below else 0.0
    slope, r2 = 0.0, 0.0
    if len(above) >= 2:
        x = np.array([a[0] for a in above])
        y = np.array([a[1] ** 2 for a in above])
        m, c = np.polyfit(x, y, 1)
        ss_res = float(np.sum((y - (m * x + c)) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        slope = float(m)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    supercritical = amp_below < 1e-3 and slope > 0 and r2 > 0.9
    return {
        "verdict": "supercritical" if supercritical else "subcritical/inconclusive",
        "A2_slope": slope,
        "A2_r2": float(r2),
        "amp_below_star": float(amp_below),
        "amplitude_curve": amp_data,
    }


# ── 2D-cubic reduction (Wilson-Cowan with FitzHugh-Nagumo-style cubic) ──
# Demonstrates the failure mode the 4D reduction was designed to escape.


def cubic_coefficients():
    """Taylor-expand Phi_E at its inflection. Even-order terms vanish
    by symmetry of the sigmoid. Returns (alpha, beta, gamma, delta)
    for the reduced system:

        tau_E dE/dt = -E + alpha E - beta E^3 - gamma I + alpha I_ext
        tau_I dI/dt = -I + delta E
    """
    a = PhiE_p(PHI_E_THETA)              # linear gain at inflection
    b = PhiE_ppp(PHI_E_THETA) / 6.0      # cubic Taylor coefficient
    alpha = a
    beta = -b                            # > 0 since b < 0 for sigmoid
    gamma = a * W_IE
    delta = PhiI_p(0.0) * W_EI
    return alpha, beta, gamma, delta


def cubic_fixed_point(I_ext, x0=None, *, alpha, beta, gamma, delta):
    if x0 is None:
        x0 = [0.001, 0.001]

    def residual(x):
        E, I = x
        return [
            -E + alpha * E - beta * E**3 - gamma * I + alpha * I_ext,
            -I + delta * E,
        ]

    sol, _, ier, _ = fsolve(residual, x0, full_output=True)
    return sol if ier == 1 else None


def cubic_jacobian(fp, alpha, beta, gamma, delta):
    E, I = fp
    return np.array([
        [(-1.0 + alpha - 3 * beta * E * E) / TAU_E_MS, -gamma / TAU_E_MS],
        [delta / TAU_I_MS, -1.0 / TAU_I_MS],
    ])


def cubic_sweep(I_grid, alpha, beta, gamma, delta):
    results = []
    x = None
    for I_ext in I_grid:
        fp = cubic_fixed_point(I_ext, x0=x, alpha=alpha, beta=beta,
                               gamma=gamma, delta=delta)
        if fp is None:
            continue
        x = fp
        J = cubic_jacobian(fp, alpha, beta, gamma, delta)
        eigs = linalg.eigvals(J)
        results.append({
            "I_ext": float(I_ext),
            "fp": fp.tolist(),
            "eigs": [(float(e.real), float(e.imag)) for e in eigs],
            "trace": float(np.trace(J)),
        })
    return results


def cubic_find_hopf(results):
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
            }
        prev_max = re_max
    return None


# ── 2D Wilson-Cowan field (synapses adiabatically eliminated) ─────────


def rhs_2d(t, y, I_ext, w_ei=W_EI, w_ie=W_IE):
    """The 2D rate field of nb033 eq (1) — no synaptic state variables."""
    E, I = y
    return [
        (-E + PhiE(I_ext - w_ie * I)) / TAU_E_MS,
        (-I + PhiI(w_ei * E)) / TAU_I_MS,
    ]


def fixed_point_2d(I_ext, w_ei=W_EI, w_ie=W_IE):
    def residual(x):
        E, I = x
        return [-E + PhiE(I_ext - w_ie * I), -I + PhiI(w_ei * E)]
    sol, _, ier, _ = fsolve(residual, [0.001, 0.001], full_output=True)
    return sol


def plot_2d_vs_4d_timeseries(hopf, out_path, run_id):
    """Same network, same drive above the 4D Hopf: 2D rings down, 4D sustains."""
    theme.apply()
    I_ext = hopf["I_ext_star"] + 3.0
    fp4, fp2 = fixed_point(I_ext), fixed_point_2d(I_ext)
    sol4 = solve_ivp(rhs_4d, (0, 300), [v + 0.02 for v in fp4],
                     args=(I_ext, W_EI, W_IE), method="LSODA",
                     rtol=1e-8, atol=1e-11, max_step=0.5)
    sol2 = solve_ivp(rhs_2d, (0, 300), [fp2[0] + 0.02, fp2[1] + 0.02],
                     args=(I_ext, W_EI, W_IE), method="LSODA",
                     rtol=1e-8, atol=1e-11, max_step=0.5)
    d4, d2 = sol4.y[0] - fp4[0], sol2.y[0] - fp2[0]
    pp4 = float(d4[sol4.t > 150].max() - d4[sol4.t > 150].min())
    pp2 = float(d2[sol2.t > 150].max() - d2[sol2.t > 150].min())
    print(f"  2D-vs-4D at I={I_ext:.2f}: 4D peak-to-peak={pp4:.4f}, 2D={pp2:.6f}")
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax.plot(sol2.t, d2, color=theme.DEEP_RED, lw=1.3,
            label="2D Wilson-Cowan — rings down to equilibrium")
    ax.plot(sol4.t, d4, color=theme.INK_BLACK, lw=1.3,
            label="4D conductance — sustains a limit cycle")
    ax.axhline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    ax.set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("$E - E^\\star$", fontsize=theme.SIZE_LABEL)
    ax.set_title(f"Same drive ($I_\\text{{ext}} = I^\\star + 3$) — only 4D oscillates",
                 fontsize=theme.SIZE_TITLE)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    fig.tight_layout()
    _stamp(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    ax.axvline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    if hopf:
        ax.scatter([0, 0], [hopf["omega_star"], -hopf["omega_star"]],
                   facecolors="none", edgecolors=theme.ELECTRIC_CYAN, s=70,
                   lw=1.4, label=f"crossing at $\\pm i\\omega^\\star$, "
                                 f"$f^\\star={hopf['freq_star_Hz']:.1f}$ Hz")
        ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper left")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("$I_\\text{ext}$", fontsize=theme.SIZE_LABEL)
    ax.set_xlabel("Re$(\\lambda)$", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Im$(\\lambda)$", fontsize=theme.SIZE_LABEL)
    ax.set_title("4D eigenvalues in the complex plane",
                 fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    _stamp(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_nb041_fgamma():
    p = FIGURES.parent / "nb041" / "numbers.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    meas = {}
    for r in d.get("results", []):
        meas.setdefault(r["tau_gaba_ms"], []).append(r["f_gamma_hz"])
    return {k: float(np.median(v)) for k, v in meas.items()}


def frequency_vs_tau_gaba(tau_list, I_grid):
    global TAU_GABA_MS
    saved = TAU_GABA_MS
    out = []
    for tg in tau_list:
        TAU_GABA_MS = tg
        h = find_hopf(sweep(I_grid))
        out.append({"tau_gaba_ms": tg,
                    "f_star_Hz": h["freq_star_Hz"] if h else None})
    TAU_GABA_MS = saved
    return out


def plot_frequency_vs_tau_gaba(mf, meas, out_path, run_id):
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    tg = [d["tau_gaba_ms"] for d in mf if d["f_star_Hz"] is not None]
    fs = [d["f_star_Hz"] for d in mf if d["f_star_Hz"] is not None]
    ax.plot(tg, fs, "o-", color=theme.INK_BLACK, lw=1.3,
            label="4D mean-field $f^\\star$")
    if meas:
        mt = sorted(meas)
        ax.plot(mt, [meas[t] for t in mt], "s--", color=theme.DEEP_RED, lw=1.3,
                label="spiking $f_\\gamma$ (nb041)")
    ax.set_xlabel("$\\tau_\\text{GABA}$ (ms)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("gamma frequency (Hz)", fontsize=theme.SIZE_LABEL)
    ax.set_title("Frequency falls with inhibitory decay in both",
                 fontsize=theme.SIZE_TITLE)
    ax.legend(fontsize=theme.SIZE_LEGEND, frameon=False, loc="upper right")
    fig.tight_layout()
    _stamp(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_limit_cycle(hopf, out_path, run_id):
    """4D limit cycle just above onset: E and I waveforms and the E→I lag."""
    theme.apply()
    I_ext = hopf["I_ext_star"] + 1.0
    fp = fixed_point(I_ext)
    sol = solve_ivp(rhs_4d, (0, 700), [v + 0.01 for v in fp],
                    args=(I_ext, W_EI, W_IE), method="LSODA",
                    rtol=1e-9, atol=1e-12, max_step=0.25, dense_output=True)
    period = 1000.0 / hopf["freq_star_Hz"]
    tt = np.linspace(700 - 3 * period, 700, 1500)
    Y = sol.sol(tt)
    E, I = Y[0], Y[1]
    # E→I phase lag from cross-correlation
    Ez, Iz = E - E.mean(), I - I.mean()
    lags = (np.arange(len(tt)) - len(tt) // 2) * (tt[1] - tt[0])
    xc = np.correlate(Iz, Ez, mode="same")
    lag_ms = float(lags[np.argmax(xc)])
    print(f"  limit cycle at I={I_ext:.2f}: I lags E by {lag_ms:.2f} ms "
          f"(tau_AMPA={TAU_AMPA_MS})")
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax.plot(tt - tt[0], E, color=theme.INK_BLACK, lw=1.3, label="$E$")
    ax.set_xlabel("time (ms)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("$E$ rate", fontsize=theme.SIZE_LABEL, color=theme.INK_BLACK)
    ax2 = ax.twinx()
    ax2.plot(tt - tt[0], I, color=theme.DEEP_RED, lw=1.3, label="$I$")
    ax2.set_ylabel("$I$ rate", fontsize=theme.SIZE_LABEL, color=theme.DEEP_RED)
    ax.set_title(f"4D limit cycle near onset — E leads I by ≈ {abs(lag_ms):.1f} ms",
                 fontsize=theme.SIZE_TITLE)
    fig.tight_layout()
    _stamp(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def hopf_locus(wei_grid, wie_grid, I_grid):
    Istar = np.full((len(wie_grid), len(wei_grid)), np.nan)
    for a, wie in enumerate(wie_grid):
        for b, wei in enumerate(wei_grid):
            h = find_hopf(sweep(I_grid, w_ei=wei, w_ie=wie))
            if h:
                Istar[a, b] = h["I_ext_star"]
    return Istar


def plot_hopf_locus(wei_grid, wie_grid, Istar, out_path, run_id):
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    im = ax.pcolormesh(wei_grid, wie_grid, Istar, cmap="magma", shading="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("$I_\\text{ext}^\\star$ (recruitment threshold)",
                   fontsize=theme.SIZE_LABEL)
    ax.set_xlabel("$W^{EI}$ (E→I drive)", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("$W^{IE}$ (I→E feedback)", fontsize=theme.SIZE_LABEL)
    ax.set_title("Where the recruitment cliff sits in the coupling plane",
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
    criticality = None
    if hopf:
        i_star = hopf["I_ext_star"]
        deltas = np.linspace(-0.1, 0.5, 13)
        for dx in deltas:
            amp = amplitude_at(i_star + dx)
            amp_data.append({"I_ext": i_star + dx, "amp": amp})
            print(f"    4D amp at I={i_star + dx:.3f}: {amp:.4f}")
        plot_criticality(amp_data, hopf, FIGURES / "criticality.png", run_id)
        print(f"  wrote {FIGURES / 'criticality.png'}")
        criticality = classify_criticality(amp_data, hopf)
        print(f"  criticality: {criticality['verdict']} "
              f"(A² slope {criticality['A2_slope']:.2e}, "
              f"R²={criticality['A2_r2']:.3f}, "
              f"amp below I* = {criticality['amp_below_star']:.4f})")

        plot_eigenvalues_complex(
            results, hopf, FIGURES / "eigenvalues_complex.png", run_id)
        print(f"  wrote {FIGURES / 'eigenvalues_complex.png'}")
        plot_2d_vs_4d_timeseries(hopf, FIGURES / "ts_2d_vs_4d.png", run_id)
        print(f"  wrote {FIGURES / 'ts_2d_vs_4d.png'}")
        plot_limit_cycle(hopf, FIGURES / "limit_cycle.png", run_id)
        print(f"  wrote {FIGURES / 'limit_cycle.png'}")

    print(f"[{SLUG}] frequency vs tau_GABA (mean-field vs nb041 spiking)")
    mf_freq = frequency_vs_tau_gaba([4.5, 6.0, 9.0, 12.0, 18.0, 27.0], I_grid)
    meas_fgamma = load_nb041_fgamma()
    for d in mf_freq:
        m = meas_fgamma.get(d["tau_gaba_ms"])
        print(f"    tau_GABA={d['tau_gaba_ms']:5}  mean-field f*={d['f_star_Hz']:.2f} Hz"
              + (f"  spiking f_gamma={m:.2f} Hz" if m else ""))
    plot_frequency_vs_tau_gaba(
        mf_freq, meas_fgamma, FIGURES / "freq_vs_tau_gaba.png", run_id)
    print(f"  wrote {FIGURES / 'freq_vs_tau_gaba.png'}")

    print(f"[{SLUG}] Hopf locus over the (W_EI, W_IE) coupling plane")
    wei_grid = np.linspace(20.0, 140.0, 9)
    wie_grid = np.linspace(20.0, 140.0, 9)
    Istar_grid = hopf_locus(wei_grid, wie_grid, np.linspace(0, 12.0, 161))
    plot_hopf_locus(wei_grid, wie_grid, Istar_grid,
                    FIGURES / "hopf_locus.png", run_id)
    print(f"  I* range across plane: "
          f"{np.nanmin(Istar_grid):.2f}–{np.nanmax(Istar_grid):.2f}; "
          f"spread along W_EI vs W_IE: "
          f"{np.nanstd(np.nanmean(Istar_grid, axis=0)):.3f} vs "
          f"{np.nanstd(np.nanmean(Istar_grid, axis=1)):.3f}")
    print(f"  wrote {FIGURES / 'hopf_locus.png'}")

    print(f"[{SLUG}] 2D-cubic reduction (FitzHugh-Nagumo-style)")
    alpha, beta, gamma, delta = cubic_coefficients()
    threshold = 1.0 + TAU_E_MS / TAU_I_MS
    print(f"  Taylor coefficients: α={alpha:.4f}, β={beta:.4f}, "
          f"γ={gamma:.4f}, δ={delta:.4f}")
    print(f"  Hopf threshold: α > 1 + τ_E/τ_I = {threshold:.3f}")
    print(f"  Have α = {alpha:.4f} — {'Hopf possible' if alpha > threshold else 'no Hopf'}")
    cubic_results = cubic_sweep(I_grid, alpha, beta, gamma, delta)
    cubic_hopf = cubic_find_hopf(cubic_results)
    if cubic_hopf:
        print(f"  2D-cubic Hopf at I_ext* = {cubic_hopf['I_ext_star']:.3f}, "
              f"f* = {cubic_hopf['freq_star_Hz']:.2f} Hz")
    else:
        print("  2D-cubic: no Hopf detected (as expected)")

    summary = {
        "slug": SLUG,
        "config": {
            "tau_E_ms": TAU_E_MS, "tau_I_ms": TAU_I_MS,
            "tau_AMPA_ms": TAU_AMPA_MS, "tau_GABA_ms": TAU_GABA_MS,
            "W_EI": W_EI, "W_IE": W_IE,
            "phi_E": dict(rmax=PHI_E_RMAX, theta=PHI_E_THETA, k=PHI_E_K),
            "phi_I": dict(rmax=PHI_I_RMAX, theta=PHI_I_THETA, k=PHI_I_K),
        },
        "results": {
            "hopf": hopf,
            "criticality": criticality,
            "frequency_vs_tau_gaba": {
                "mean_field": mf_freq,
                "spiking_nb041": meas_fgamma,
            },
            "two_d_cubic": {
                "cubic_coefficients": dict(
                    alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                ),
                "hopf_threshold_alpha": threshold,
                "hopf": cubic_hopf,
                "verdict": (
                    "no Hopf — α stays below 1 + τ_E/τ_I because W^EE = 0"
                    if cubic_hopf is None
                    else f"Hopf at I_ext* = {cubic_hopf['I_ext_star']:.3f}"
                ),
            },
        },
        "success_criteria": [
            {
                "label": "4D Hopf located",
                "passed": hopf is not None,
                "detail": (
                    f"I_ext* = {hopf['I_ext_star']:.3f}, "
                    f"f* = {hopf['freq_star_Hz']:.2f} Hz"
                    if hopf else "no Hopf found"
                ),
            },
            {
                "label": "4D Hopf is supercritical (continuous amplitude onset)",
                "passed": bool(criticality and criticality["verdict"] == "supercritical"),
                "detail": (
                    f"A² ∝ (I−I*): slope {criticality['A2_slope']:.2e}, "
                    f"R² = {criticality['A2_r2']:.3f}; silent below I* "
                    f"(amp = {criticality['amp_below_star']:.4f})"
                    if criticality else "not evaluated"
                ),
            },
            {
                "label": "2D-cubic verdict: no Hopf (architecture lacks W^EE)",
                "passed": cubic_hopf is None,
                "detail": f"α = {alpha:.4f} vs threshold {threshold:.3f}",
            },
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  wrote {FIGURES / 'numbers.json'}")


if __name__ == "__main__":
    main()
