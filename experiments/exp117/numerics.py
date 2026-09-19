"""Pure numerical operations for the independent exp117 mean-field model."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import eigvals
from scipy.optimize import brentq
from scipy.special import erf, erfcx


def _parameters(configuration, *, tau_gaba_ms=None):
    imported = configuration["imported_parameters"]
    choices = configuration["mean_field_choices"]
    return {
        "tau_r_E": choices["rate_relaxation_ms"]["E"],
        "tau_r_I": choices["rate_relaxation_ms"]["I"],
        "tau_m_E": imported["capacitance_nF"]["E"]
        / imported["leak_conductance_uS"]["E"],
        "tau_m_I": imported["capacitance_nF"]["I"]
        / imported["leak_conductance_uS"]["I"],
        "tau_ref_E": imported["refractory_ms"]["E"],
        "tau_ref_I": imported["refractory_ms"]["I"],
        "tau_ampa": imported["tau_AMPA_ms"],
        "tau_gaba": (
            imported["tau_GABA_reference_ms"]
            if tau_gaba_ms is None
            else float(tau_gaba_ms)
        ),
        "g_L_E": imported["leak_conductance_uS"]["E"],
        "g_L_I": imported["leak_conductance_uS"]["I"],
        "rest": imported["rest_mV"],
        "reset": imported["reset_mV"],
        "threshold": imported["threshold_mV"],
        "dv_exc": imported["reversal_mV"]["excitatory"] - imported["rest_mV"],
        "dv_inh": imported["rest_mV"] - imported["reversal_mV"]["inhibitory"],
        "G_EI": imported["summed_coupling_uS"]["E_to_I"],
        "G_IE": imported["summed_coupling_uS"]["I_to_E"],
        "sigma": choices["effective_voltage_noise_mV"],
    }


def _gain_integrand(value):
    if value < 0:
        return float(erfcx(-value))
    return float(np.exp(min(value * value, 700.0)) * (1.0 + erf(value)))


def gain(current_nA, *, tau_m, tau_ref, leak, parameters):
    mean_voltage = parameters["rest"] + current_nA / leak
    lower = (parameters["reset"] - mean_voltage) / parameters["sigma"]
    upper = (parameters["threshold"] - mean_voltage) / parameters["sigma"]
    integral, _ = quad(_gain_integrand, lower, upper, limit=200)
    rate = 1.0 / (tau_ref + tau_m * np.sqrt(np.pi) * integral)
    return float(rate), float(lower), float(upper)


def gain_with_derivative(current_nA, *, population, configuration):
    p = _parameters(configuration)
    suffix = "E" if population == "E" else "I"
    rate, lower, upper = gain(
        current_nA,
        tau_m=p[f"tau_m_{suffix}"],
        tau_ref=p[f"tau_ref_{suffix}"],
        leak=p[f"g_L_{suffix}"],
        parameters=p,
    )
    derivative = (
        p[f"tau_m_{suffix}"]
        * np.sqrt(np.pi)
        * rate**2
        * (_gain_integrand(upper) - _gain_integrand(lower))
        / (p[f"g_L_{suffix}"] * p["sigma"])
    )
    return rate, float(derivative)


def equilibrium(drive_nA, configuration, *, tau_gaba_ms=None):
    p = _parameters(configuration, tau_gaba_ms=tau_gaba_ms)
    root_cfg = configuration["numerical_protocol"]["equilibrium_root"]

    def mismatch(rate_E):
        g_e_I = p["tau_ampa"] * p["G_EI"] * rate_E
        rate_I, _ = gain_with_derivative(
            p["dv_exc"] * g_e_I, population="I", configuration=configuration
        )
        g_i_E = p["tau_gaba"] * p["G_IE"] * rate_I
        predicted_E, _ = gain_with_derivative(
            drive_nA - p["dv_inh"] * g_i_E,
            population="E",
            configuration=configuration,
        )
        return rate_E - predicted_E

    lower, upper = 0.0, 1.0 / p["tau_ref_E"]
    rate_E = float(
        brentq(
            mismatch,
            lower,
            upper,
            xtol=root_cfg["xtol"],
            rtol=root_cfg["rtol"],
        )
    )
    g_e_I = p["tau_ampa"] * p["G_EI"] * rate_E
    rate_I, _ = gain_with_derivative(
        p["dv_exc"] * g_e_I, population="I", configuration=configuration
    )
    g_i_E = p["tau_gaba"] * p["G_IE"] * rate_I
    state = np.array([rate_E, rate_I, g_e_I, g_i_E], dtype=float)
    if (
        not np.isfinite(state).all()
        or np.any(state < 0)
        or abs(mismatch(rate_E)) > root_cfg["residual_tolerance_per_ms"]
    ):
        raise RuntimeError(f"invalid equilibrium at I_ext={drive_nA}")
    return state


def analytical_jacobian(state, drive_nA, configuration, *, tau_gaba_ms=None):
    p = _parameters(configuration, tau_gaba_ms=tau_gaba_ms)
    _, _, g_e_I, g_i_E = state
    _, derivative_E = gain_with_derivative(
        drive_nA - p["dv_inh"] * g_i_E,
        population="E",
        configuration=configuration,
    )
    _, derivative_I = gain_with_derivative(
        p["dv_exc"] * g_e_I,
        population="I",
        configuration=configuration,
    )
    return np.array(
        [
            [
                -1 / p["tau_r_E"],
                0,
                0,
                -p["dv_inh"] * derivative_E / p["tau_r_E"],
            ],
            [
                0,
                -1 / p["tau_r_I"],
                p["dv_exc"] * derivative_I / p["tau_r_I"],
                0,
            ],
            [p["G_EI"], 0, -1 / p["tau_ampa"], 0],
            [0, p["G_IE"], 0, -1 / p["tau_gaba"]],
        ],
        dtype=float,
    )


def flow_rhs(_time_ms, state, drive_nA, configuration):
    """Evaluate the four-variable deterministic mean-field flow."""
    p = _parameters(configuration)
    rate_E, rate_I, g_e_I, g_i_E = state
    target_E, _, _ = gain(
        drive_nA - p["dv_inh"] * g_i_E,
        tau_m=p["tau_m_E"],
        tau_ref=p["tau_ref_E"],
        leak=p["g_L_E"],
        parameters=p,
    )
    target_I, _, _ = gain(
        p["dv_exc"] * g_e_I,
        tau_m=p["tau_m_I"],
        tau_ref=p["tau_ref_I"],
        leak=p["g_L_I"],
        parameters=p,
    )
    return np.array(
        [
            (-rate_E + target_E) / p["tau_r_E"],
            (-rate_I + target_I) / p["tau_r_I"],
            -g_e_I / p["tau_ampa"] + p["G_EI"] * rate_E,
            -g_i_E / p["tau_gaba"] + p["G_IE"] * rate_I,
        ],
        dtype=float,
    )


def criticality_ramps(hopf, configuration):
    """Integrate ascending and descending near-onset drive ramps."""
    cfg = configuration["numerical_protocol"]["criticality_ramp"]
    start, stop = cfg["span_relative_to_hopf_nA"]
    drives = np.linspace(
        hopf["I_ext_star_nA"] + start,
        hopf["I_ext_star_nA"] + stop,
        cfg["points"],
    )
    times = np.linspace(
        cfg["measurement_start_ms"],
        cfg["duration_ms"],
        cfg["measurement_samples"],
    )
    state = equilibrium(float(drives[0]), configuration).copy()
    state[0] += cfg["initial_rate_perturbation_per_ms"]
    branches = {}
    directions = (
        ("up", range(drives.size)),
        ("down", range(drives.size - 1, -1, -1)),
    )
    for name, indices in directions:
        states = np.empty((drives.size, 4, times.size), dtype=float)
        for index in indices:
            solution = solve_ivp(
                flow_rhs,
                (0.0, cfg["duration_ms"]),
                state,
                args=(float(drives[index]), configuration),
                method=cfg["solver"],
                rtol=cfg["rtol"],
                atol=cfg["atol"],
                max_step=cfg["max_step_ms"],
                t_eval=times,
            )
            if (
                not solution.success
                or solution.t.size != times.size
                or not np.isfinite(solution.y).all()
            ):
                raise RuntimeError(
                    f"criticality ramp failed at I_ext={drives[index]}: "
                    f"{solution.message}"
                )
            states[index] = solution.y
            state = solution.y[:, -1]
        branches[name] = states
    return {
        "drives_nA": drives,
        "measurement_times_ms": times,
        "states_up": branches["up"],
        "states_down": branches["down"],
    }


def equilibrium_eigenvalues(drive_nA, configuration, *, tau_gaba_ms=None):
    state = equilibrium(drive_nA, configuration, tau_gaba_ms=tau_gaba_ms)
    values = eigvals(
        analytical_jacobian(
            state,
            drive_nA,
            configuration,
            tau_gaba_ms=tau_gaba_ms,
        )
    )
    values = sorted(values, key=lambda value: (value.real, value.imag), reverse=True)
    return state, np.asarray(values)


def leading_positive_imaginary(values, threshold):
    eligible = [value for value in values if value.imag > threshold]
    if not eligible:
        return None
    return max(eligible, key=lambda value: value.real)


def continuation(configuration, *, tau_gaba_ms=None):
    grid = np.linspace(*configuration["numerical_protocol"]["drive_grid_nA"])
    threshold = configuration["numerical_protocol"][
        "complex_eigenvalue_threshold_per_ms"
    ]
    rows = []
    for drive in grid:
        state, values = equilibrium_eigenvalues(
            float(drive), configuration, tau_gaba_ms=tau_gaba_ms
        )
        leading = leading_positive_imaginary(values, threshold)
        rows.append(
            {
                "I_ext_nA": float(drive),
                "equilibrium": state.tolist(),
                "eigenvalues_per_ms": [
                    [float(value.real), float(value.imag)] for value in values
                ],
                "leading_complex_per_ms": None
                if leading is None
                else [float(leading.real), float(leading.imag)],
            }
        )
    return rows


def refine_hopf(rows, configuration, *, tau_gaba_ms=None):
    threshold = configuration["numerical_protocol"][
        "complex_eigenvalue_threshold_per_ms"
    ]
    bracket = None
    previous = None
    for row in rows:
        current = row["leading_complex_per_ms"]
        if previous is not None and previous["leading_complex_per_ms"] is not None and current is not None:
            if previous["leading_complex_per_ms"][0] < 0 <= current[0]:
                bracket = (previous["I_ext_nA"], row["I_ext_nA"])
                break
        previous = row
    if bracket is None:
        raise RuntimeError("no complex-pair stability crossing found")

    def leading_real(drive):
        _, values = equilibrium_eigenvalues(
            float(drive), configuration, tau_gaba_ms=tau_gaba_ms
        )
        leading = leading_positive_imaginary(values, threshold)
        if leading is None:
            raise RuntimeError(f"no complex eigenvalue at I_ext={drive}")
        return float(leading.real)

    refinement = configuration["numerical_protocol"]["hopf_refinement"]
    drive = float(
        brentq(
            leading_real,
            *bracket,
            xtol=refinement["xtol_nA"],
            rtol=refinement["rtol"],
        )
    )
    state, values = equilibrium_eigenvalues(
        drive, configuration, tau_gaba_ms=tau_gaba_ms
    )
    leading = leading_positive_imaginary(values, threshold)
    step = refinement["transversality_step_nA"]
    slope = (leading_real(drive + step) - leading_real(drive - step)) / (2 * step)
    pair_indices = np.argsort(
        [abs(value - leading) if leading is not None else np.inf for value in values]
    )[:1].tolist()
    conjugate = np.conjugate(leading)
    pair_indices += np.argsort([abs(value - conjugate) for value in values])[:1].tolist()
    remaining = [
        value for index, value in enumerate(values) if index not in set(pair_indices)
    ]
    return {
        "coarse_bracket_nA": list(bracket),
        "I_ext_star_nA": drive,
        "equilibrium": state.tolist(),
        "eigenvalues_per_ms": [
            [float(value.real), float(value.imag)] for value in values
        ],
        "critical_pair_per_ms": [float(leading.real), float(leading.imag)],
        "remaining_max_real_per_ms": float(max(value.real for value in remaining)),
        "crossing_slope_per_ms_per_nA": float(slope),
        "omega_Hopf_rad_per_ms": float(abs(leading.imag)),
        "f_Hopf_Hz": float(1000 * abs(leading.imag) / (2 * np.pi)),
    }
