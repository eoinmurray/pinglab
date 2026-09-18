"""Gain evaluation, equilibrium continuation and time-domain Hopf diagnostics.

Equations follow writings/exp115.typ Appendices A-C. This module has no run IO.
"""

from functools import lru_cache
from itertools import permutations

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq, root
from scipy.special import erfc, erfcx


class NumericalFailure(RuntimeError):
    """A calculation has no numerically acceptable value."""


class Model:
    def __init__(self, cfg, condition, *, tight=False):
        self.cfg, self.condition = cfg, condition
        self.tau = condition["tau_GABA_ms"]
        self.sigma = condition["sigma_mV"]
        self.kappa = condition["kappa"]
        factor = cfg["tightening_factor"] if tight else 1.0
        self.epsabs, self.epsrel = cfg["quad_abs"] / factor, cfg["quad_rel"] / factor
        self.residual_tol = cfg["equilibrium_residual"] / factor
        self.xtol = cfg["brent_abs_nA"] / factor
        self.a = 1 / (self.kappa * cfg["cells"]["E"]["tau_m_ms"])
        self.b = 1 / (self.kappa * cfg["cells"]["I"]["tau_m_ms"])
        self.c, self.d = 1 / cfg["tau_AMPA_ms"], 1 / self.tau
        self.Jei = cfg["dV_exc_mV"] * cfg["tau_AMPA_ms"] * cfg["G_EI_uS"]
        self.Jie = cfg["dV_inh_mV"] * self.tau * cfg["G_IE_uS"]
        a, b, c, d = self.a, self.b, self.c, self.d
        self.A1 = a + b + c + d
        self.A2 = a * b + a * c + a * d + b * c + b * d + c * d
        self.A3 = a * b * c + a * b * d + a * c * d + b * c * d
        self.Kh = (
            self.A1 * self.A2 * self.A3 - self.A3**2
        ) / self.A1**2 - a * b * c * d
        self.omega = np.sqrt(self.A3 / self.A1)
        # Per-instance caches are released with the condition, not retained by a
        # decorator holding every Model instance for the lifetime of the sweep.
        self.gain = lru_cache(maxsize=16384)(self._gain)

    def _gain(self, current, population):
        cfg, cell = self.cfg, self.cfg["cells"][population]
        mu = cfg["rest_mV"] + current / cell["g_L_uS"]
        alpha = (cfg["reset_mV"] - mu) / self.sigma
        beta = (cfg["threshold_mV"] - mu) / self.sigma
        shift = max(beta, 0.0) ** 2
        unscale = np.exp(-shift)

        def kernel(z):
            return erfcx(-z) * unscale if z < 0 else np.exp(z * z - shift) * erfc(-z)

        scaled_epsabs = self.epsabs * unscale
        if beta > 8:
            # t=beta*(beta-z) resolves the positive-tail boundary layer without
            # subtracting two large, nearly equal squares in its exponent.
            upper = beta * (cfg["threshold_mV"] - cfg["reset_mV"]) / self.sigma

            def tail_kernel(t):
                z = beta - t / beta
                return (
                    np.exp(-t * (2 - t / beta**2)) * erfc(-z) / beta
                    if z >= 0
                    else erfcx(-z) * unscale / beta
                )

            points = [n for n in (1, 4, 16, 64) if n < upper]
            result = quad(
                tail_kernel,
                0,
                upper,
                epsabs=scaled_epsabs,
                epsrel=self.epsrel,
                limit=cfg["quad_limit"],
                points=points,
                full_output=1,
            )
        else:
            result = quad(
                kernel,
                alpha,
                beta,
                epsabs=scaled_epsabs,
                epsrel=self.epsrel,
                limit=cfg["quad_limit"],
                full_output=1,
            )
        integral, error = result[:2]
        if len(result) != 3 or not np.isfinite(integral) or integral <= 0:
            raise NumericalFailure(
                f"Siegert quadrature failed: {population}, I={current}"
            )
        if error > max(scaled_epsabs, self.epsrel * abs(integral)):
            raise NumericalFailure("Siegert quadrature error target not met")
        prefactor = cell["tau_m_ms"] * np.sqrt(np.pi)
        denominator = cell["tau_ref_ms"] * unscale + prefactor * integral
        rate = unscale / denominator
        eta = 1 / (cell["g_L_uS"] * self.sigma)
        ha, hb = kernel(alpha), kernel(beta)
        hpa = 2 * alpha * ha + 2 * unscale / np.sqrt(np.pi)
        hpb = 2 * beta * hb + 2 * unscale / np.sqrt(np.pi)
        d1 = prefactor * eta * (ha - hb) / denominator
        d2 = prefactor * eta**2 * (hpb - hpa) / denominator
        values = (
            rate,
            -rate * d1,
            rate * (2 * d1 * d1 - d2),
        )
        if not np.isfinite(values).all():
            raise NumericalFailure("nonfinite gain or derivative")
        return values

    def currents(self, x, drive):
        return drive - self.cfg["dV_inh_mV"] * x[3], self.cfg["dV_exc_mV"] * x[2]

    def state(self, rates):
        e, i = rates
        return np.array(
            [
                e,
                i,
                self.cfg["tau_AMPA_ms"] * self.cfg["G_EI_uS"] * e,
                self.tau * self.cfg["G_IE_uS"] * i,
            ]
        )

    def field(self, x, drive):
        ie, ii = self.currents(x, drive)
        return np.array(
            [
                self.a * (-x[0] + self.gain(ie, "E")[0]),
                self.b * (-x[1] + self.gain(ii, "I")[0]),
                -self.c * x[2] + self.cfg["G_EI_uS"] * x[0],
                -self.d * x[3] + self.cfg["G_IE_uS"] * x[1],
            ]
        )

    def equilibrium(self, drive, guess=None):
        def residual(r):
            return [
                r[0] - self.gain(drive - self.Jie * r[1], "E")[0],
                r[1] - self.gain(self.Jei * r[0], "I")[0],
            ]

        def jacobian(r):
            return [
                [1, self.Jie * self.gain(drive - self.Jie * r[1], "E")[1]],
                [-self.Jei * self.gain(self.Jei * r[0], "I")[1], 1],
            ]

        start = self.cfg["initial_rates_per_ms"] if guess is None else guess[:2]
        try:
            candidate = root(
                residual, start, jac=jacobian, method="hybr", options={"xtol": 1e-10}
            )
        except (NumericalFailure, ValueError, FloatingPointError):
            candidate = None
        upper = min(1 / self.cfg["cells"]["E"]["tau_ref_ms"], self.gain(drive, "E")[0])

        def scalar(r):
            return (
                r
                - self.gain(drive - self.Jie * self.gain(self.Jei * r, "I")[0], "E")[0]
            )

        independent_e = brentq(scalar, 0, upper, xtol=1e-15, rtol=1e-14)
        independent = np.array(
            [independent_e, self.gain(self.Jei * independent_e, "I")[0]]
        )
        ceiling = np.array([1 / self.cfg["cells"][p]["tau_ref_ms"] for p in ("E", "I")])
        good = (
            candidate is not None
            and np.isfinite(candidate.x).all()
            and (candidate.x >= 0).all()
            and (candidate.x < ceiling).all()
            and max(abs(np.asarray(residual(candidate.x)))) <= self.residual_tol
            and max(abs(candidate.x - independent)) <= self.residual_tol
        )
        rates = candidate.x if good else independent
        x = self.state(rates)
        discrepancy = float(max(abs(rates - independent)))
        rerr = float(max(abs(np.asarray(residual(rates)))))
        ferr = float(max(abs(self.field(x, drive))))
        if (
            (rates < 0).any()
            or (rates >= ceiling).any()
            or max(rerr, ferr) > self.residual_tol
        ):
            raise NumericalFailure(f"unacceptable equilibrium at {drive}")
        return x, {
            "rate_residual": rerr,
            "flow_residual": ferr,
            "scalar_discrepancy": discrepancy,
            "scalar_fallback": not bool(good),
        }

    def local(self, x, drive):
        ie, ii = self.currents(x, drive)
        ge, gi = self.gain(ie, "E"), self.gain(ii, "I")
        u, v = (
            self.a * self.cfg["dV_inh_mV"] * ge[1],
            self.b * self.cfg["dV_exc_mV"] * gi[1],
        )
        A = np.array(
            [
                [-self.a, 0, 0, -u],
                [0, -self.b, v, 0],
                [self.cfg["G_EI_uS"], 0, -self.c, 0],
                [0, self.cfg["G_IE_uS"], 0, -self.d],
            ]
        )
        K = u * v * self.cfg["G_EI_uS"] * self.cfg["G_IE_uS"]
        return A, ge, gi, K

    def at_drive(self, drive, guess=None):
        x, residuals = self.equilibrium(drive, guess)
        return self.at_state(x, drive, residuals)

    def at_state(self, x, drive, residuals):
        A, ge, gi, K = self.local(x, drive)
        eigs = np.linalg.eigvals(A)
        return {
            "drive_nA": float(drive),
            "state": x.tolist(),
            "eigenvalues": [[float(z.real), float(z.imag)] for z in eigs],
            "gain_derivatives": [list(ge), list(gi)],
            "K_minus_KH": float(K - self.Kh),
            **residuals,
        }

    def crossing(self, left, right):
        def fn(drive):
            return self.at_drive(drive)["K_minus_KH"]

        drive = brentq(fn, left, right, xtol=self.xtol, rtol=self.cfg["brent_rel"])
        row = self.at_drive(drive)
        A, ge, gi, K = self.local(np.array(row["state"]), drive)
        denominator = 1 + self.Jie * self.Jei * ge[1] * gi[1]
        die, dii = 1 / denominator, self.Jei * ge[1] / denominator
        prefactor = (
            self.a
            * self.b
            * self.cfg["dV_inh_mV"]
            * self.cfg["dV_exc_mV"]
            * self.cfg["G_EI_uS"]
            * self.cfg["G_IE_uS"]
        )
        kp = prefactor * (ge[2] * gi[1] * die + ge[1] * gi[2] * dii)
        chi = (
            2
            * self.A3
            * kp
            / (
                (2 * self.A3) ** 2
                + 4 * self.omega**2 * (self.A2 - 2 * self.omega**2) ** 2
            )
        )
        eigs = np.array([complex(*z) for z in row["eigenvalues"]])
        critical_index = np.argmin(abs(eigs - 1j * self.omega))
        critical = eigs[critical_index]
        remaining = np.delete(
            eigs, [critical_index, np.argmin(abs(eigs + 1j * self.omega))]
        )
        a4 = self.a * self.b * self.c * self.d + K
        polynomial = [1, self.A1, self.A2, self.A3, a4]
        roots = np.roots(polynomial)
        root_error = min(max(abs(eigs - np.array(p))) for p in permutations(roots))
        fd = []
        for h in self.cfg["transversality_steps_nA"]:
            branches = [self.at_drive(drive + s * h) for s in (-1, 1)]
            matched = [
                min(
                    (complex(*z) for z in b["eigenvalues"]),
                    key=lambda z: abs(z - critical),
                )
                for b in branches
            ]
            fd.append(float((matched[1].real - matched[0].real) / (2 * h)))
        return {
            **row,
            "bracket_nA": [left, right],
            "frequency_Hz": float(1000 * self.omega / (2 * np.pi)),
            "omega_per_ms": float(self.omega),
            "chi_per_ms_nA": float(chi),
            "chi_finite_difference": fd,
            "direction": "loss" if chi > 0 else "gain",
            "norm_A": float(np.linalg.norm(A, 2)),
            "critical_real": float(abs(critical.real)),
            "remaining_max_real": float(max(remaining.real)),
            "eigen_separation": float(
                min(abs(np.delete(eigs, critical_index) - critical))
            ),
            "quartic_root_error": float(root_error),
            "hopf_identity_error": float(
                abs(self.A1 * self.A2 * self.A3 - self.A3**2 - self.A1**2 * a4)
                / (self.A1 * self.A2 * self.A3 + self.A3**2 + self.A1**2 * a4)
            ),
            "frequency_identity_error_Hz": float(
                abs(1000 * (critical.imag - self.omega) / (2 * np.pi))
            ),
        }


def amplitude_ramp(model, onset_drive):
    """Integrate matched upward/downward drive ramps around one Hopf onset."""
    cfg = model.cfg["criticality_ramp"]
    drives = onset_drive + np.linspace(*cfg["span_nA"], cfg["points"])
    sample_times = np.arange(
        cfg["observation_start_ms"],
        cfg["duration_ms"] + cfg["observation_step_ms"] / 2,
        cfg["observation_step_ms"],
    )
    state = model.equilibrium(float(drives[0]))[0]
    state = state.copy()
    state[0] += cfg["initial_rate_perturbation_per_ms"]
    branches = {}
    for direction, sequence in (("up", drives), ("down", drives[::-1])):
        observations, endpoints = [], []
        for drive in sequence:
            solution = solve_ivp(
                lambda _time, x: model.field(x, float(drive)),
                (0.0, cfg["duration_ms"]),
                state,
                method=cfg["solver"],
                rtol=cfg["rtol"],
                atol=cfg["atol"],
                max_step=cfg["max_step_ms"],
                t_eval=sample_times,
            )
            if (
                not solution.success
                or solution.t.size != sample_times.size
                or not np.isfinite(solution.y).all()
            ):
                raise NumericalFailure(
                    f"{direction} ramp failed at {drive:g} nA: {solution.message}"
                )
            state = solution.y[:, -1]
            observations.append(solution.y.T)
            endpoints.append(state.copy())
        if direction == "down":
            observations.reverse()
            endpoints.reverse()
        branches[direction] = np.asarray(observations)
        branches[direction + "_endpoint"] = np.asarray(endpoints)
    return {
        "drive_nA": drives,
        "observation_time_ms": sample_times,
        **branches,
    }


def scan(model, count, *, equilibrium_scan=None):
    """Every grid point is recorded; a failed solve cannot create a bracket."""
    rows, crossings, guess = [], [], None
    for index, drive in enumerate(np.linspace(*model.cfg["drive_interval_nA"], count)):
        try:
            shared = (
                None if equilibrium_scan is None else equilibrium_scan["rows"][index]
            )
            if shared is not None and "failure" not in shared:
                x = np.array(shared["state"])
                residuals = {
                    key: shared[key]
                    for key in (
                        "rate_residual",
                        "scalar_discrepancy",
                        "scalar_fallback",
                    )
                }
                residuals["flow_residual"] = float(max(abs(model.field(x, drive))))
                if residuals["flow_residual"] > model.residual_tol:
                    raise NumericalFailure(
                        "shared equilibrium fails scaled flow residual"
                    )
                row = model.at_state(x, drive, residuals)
            else:
                row = model.at_drive(drive, guess)
            guess = row["state"]
        except (NumericalFailure, ValueError, FloatingPointError) as exc:
            row, guess = {"drive_nA": float(drive), "failure": str(exc)}, None
        if rows and "failure" not in row and "failure" not in rows[-1]:
            left, right = rows[-1]["K_minus_KH"], row["K_minus_KH"]
            if left * right < 0 or left == 0 or (right == 0 and left != 0):
                try:
                    crossing = model.crossing(rows[-1]["drive_nA"], row["drive_nA"])
                    if (
                        not crossings
                        or abs(
                            crossing["drive_nA"] - crossings[-1].get("drive_nA", -10)
                        )
                        > 1e-8
                    ):
                        crossings.append(crossing)
                except (NumericalFailure, ValueError, np.linalg.LinAlgError) as exc:
                    crossings.append(
                        {
                            "bracket_nA": [rows[-1]["drive_nA"], row["drive_nA"]],
                            "failure": str(exc),
                        }
                    )
        rows.append(row)
    return {"count": count, "rows": rows, "crossings": crossings}
