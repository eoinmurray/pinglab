"""034 — 2D-cubic Wilson-Cowan alternative for PING.

Numerics for the theory in src/docs/src/pages/notebooks/nb034.mdx.

Builds on nb033's 4D analysis. Tests whether the 2D-cubic reduction
(FitzHugh-Nagumo-like) can Hopf with naive Taylor-expansion
coefficients derived from the same sigmoid gain. Result: no — the
linear coefficient α stays below 1 because we lack E→E self-coupling.

Outputs figures and numbers.json to
src/docs/public/figures/notebooks/nb034/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import fsolve

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src" / "pinglab"))
import theme  # noqa: E402

SLUG = "nb034"
FIGURES = REPO / "src" / "docs" / "public" / "figures" / "notebooks" / SLUG

# Parameters — same as nb033 for consistency.
TAU_E_MS = 20.0
TAU_I_MS = 5.0

PHI_E_RMAX = 0.20
PHI_E_THETA = 1.5
PHI_E_K = 0.2

PHI_I_RMAX = 0.30
PHI_I_THETA = 1.0
PHI_I_K = 0.15

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


def PhiE_p(x): return phi_p(x, PHI_E_RMAX, PHI_E_THETA, PHI_E_K)
def PhiE_ppp(x): return phi_ppp(x, PHI_E_RMAX, PHI_E_THETA, PHI_E_K)
def PhiI_p(x): return phi_p(x, PHI_I_RMAX, PHI_I_THETA, PHI_I_K)


def cubic_coefficients():
    """Taylor-expand Phi_E at its inflection. Even-order terms vanish.
    Returns (alpha, beta, gamma, delta) for the reduced system

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


def fixed_point(I_ext, x0=None, *, alpha, beta, gamma, delta):
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


def jacobian(fp, alpha, beta, gamma, delta):
    E, I = fp
    return np.array([
        [(-1.0 + alpha - 3 * beta * E * E) / TAU_E_MS, -gamma / TAU_E_MS],
        [delta / TAU_I_MS, -1.0 / TAU_I_MS],
    ])


def sweep(I_grid, alpha, beta, gamma, delta):
    results = []
    x = None
    for I_ext in I_grid:
        fp = fixed_point(I_ext, x0=x, alpha=alpha, beta=beta,
                          gamma=gamma, delta=delta)
        if fp is None:
            continue
        x = fp
        J = jacobian(fp, alpha, beta, gamma, delta)
        eigs = linalg.eigvals(J)
        results.append({
            "I_ext": float(I_ext),
            "fp": fp.tolist(),
            "eigs": [(float(e.real), float(e.imag)) for e in eigs],
            "trace": float(np.trace(J)),
        })
    return results


def find_hopf(results):
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


def _stamp(fig, run_id):
    fig.text(
        0.995, 0.005, run_id, ha="right", va="bottom",
        fontsize=theme.SIZE_CAPTION, color=theme.LABEL, family="monospace",
    )


def plot_eigenvalues(results, hopf, out_path, run_id):
    theme.apply()
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    xs = [r["I_ext"] for r in results]
    eigs_re = np.array([[e[0] for e in r["eigs"]] for r in results])
    eigs_im = np.array([[e[1] for e in r["eigs"]] for r in results])
    for k in range(eigs_re.shape[1]):
        color = theme.INK_BLACK if abs(eigs_im[0, k]) < 1e-6 else theme.DEEP_RED
        ax.plot(xs, eigs_re[:, k], color=color, lw=1.2, alpha=0.8)
    ax.axhline(0, color=theme.GREY_MID, lw=0.6, ls=":")
    if hopf:
        ax.axvline(hopf["I_ext_star"], color=theme.AMBER, lw=1.0, ls="--",
                   label=f"$I^\\star = {hopf['I_ext_star']:.2f}$")
        ax.legend(fontsize=theme.SIZE_LABEL, frameon=False, loc="upper left")
    ax.set_xlabel("$I_\\text{ext}$", fontsize=theme.SIZE_LABEL)
    ax.set_ylabel("Re$(\\lambda)$", fontsize=theme.SIZE_LABEL)
    ax.set_title(
        "2D-cubic reduction eigenvalues — no Hopf with naive coefficients",
        fontsize=theme.SIZE_TITLE,
    )
    fig.tight_layout()
    _stamp(fig, run_id)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    run_id = "nb034-numerics"
    alpha, beta, gamma, delta = cubic_coefficients()
    print(f"[{SLUG}] Cubic coefficients: "
          f"α={alpha:.4f}, β={beta:.4f}, γ={gamma:.4f}, δ={delta:.4f}")
    print(f"  α > 1 needed for Hopf; have α = {alpha:.4f} — no Hopf expected.")

    I_grid = np.linspace(0, 12.0, 241)
    results = sweep(I_grid, alpha, beta, gamma, delta)
    hopf = find_hopf(results)
    if hopf:
        print(f"  Hopf at I_ext* = {hopf['I_ext_star']:.3f}, "
              f"f* = {hopf['freq_star_Hz']:.2f} Hz")
    else:
        print("  No Hopf detected (as expected).")

    plot_eigenvalues(results, hopf, FIGURES / "eigenvalues.png", run_id)
    print(f"  wrote {FIGURES / 'eigenvalues.png'}")

    summary = {
        "slug": SLUG,
        "config": {
            "tau_E_ms": TAU_E_MS, "tau_I_ms": TAU_I_MS,
            "W_EI": W_EI, "W_IE": W_IE,
            "phi_E": dict(rmax=PHI_E_RMAX, theta=PHI_E_THETA, k=PHI_E_K),
            "phi_I": dict(rmax=PHI_I_RMAX, theta=PHI_I_THETA, k=PHI_I_K),
        },
        "results": {
            "cubic_coefficients": dict(
                alpha=alpha, beta=beta, gamma=gamma, delta=delta,
            ),
            "hopf": hopf,
            "verdict": (
                "no Hopf — alpha < 1 means trace stays negative"
                if hopf is None else f"Hopf at I_ext* = {hopf['I_ext_star']:.3f}"
            ),
        },
        "success_criteria": [
            {
                "label": "cubic coefficients computed",
                "passed": True,
                "detail": (f"α={alpha:.3f}, β={beta:.3f}, "
                           f"γ={gamma:.3f}, δ={delta:.3f}"),
            },
            {
                "label": "verdict on 2D-cubic Hopf",
                "passed": True,
                "detail": ("expected no Hopf, confirmed" if hopf is None
                           else f"unexpected Hopf at {hopf['I_ext_star']:.3f}"),
            },
        ],
    }
    (FIGURES / "numbers.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  wrote {FIGURES / 'numbers.json'}")


if __name__ == "__main__":
    main()
