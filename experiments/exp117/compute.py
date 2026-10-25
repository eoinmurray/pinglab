"""Compute native SNNLab mean-field continuation and near-onset dynamics.

SNNLab owns the closure, equilibrium search, Jacobian and ODE integration.
This stage retains the experiment's drive/decay sweeps, transversality check and
ascending/descending criticality protocol. Exports retain spikes/ms rate units.
"""

from __future__ import annotations

import argparse
import platform
import sys
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
import scipy
from experiments.exp117 import recipe
from experiments.exp117.custom import inputs
from pingstore.contracts import write_json_atomic
from snnlab import meanfield as mf

# The article/analysis payload uses spikes/ms; SNNLab's numerical states use Hz.
EXPORT_SCALE = np.array([0.001, 0.001, 1.0, 1.0])


def compute(*, run_id=None):
    with inputs.execution(REPO, "compute", sources={}, run_id=run_id) as run:
        cfg = recipe.validate(run.record["execution"]["configuration"])
        bundle = recipe.author_network(cfg)
        bundle.write(run.export / "network.bundle", visualise=False)
        bundle.visualise(run.export / "circuit.svg", view="circuit")
        model = recipe.mean_field_model(cfg, bundle=bundle)
        model.write(run.export / "mean-field-model.json")
        branch = continuation(model, cfg)
        hopf = refine_hopf(model, branch, cfg)
        ramps = criticality_ramps(model, hopf, cfg)
        tau_sweep = []
        reference_tau = cfg["imported_parameters"]["tau_GABA_reference_ms"]
        for tau in cfg["mean_field_choices"]["tau_GABA_sweep_ms"]:
            if tau == reference_tau:
                sweep_hopf = hopf
            else:
                swept = mf.substitute(model, {"I_to_E.tau": tau})
                swept.write(run.export / f"tau-{tau:g}-model.json")
                sweep_hopf = refine_hopf(swept, continuation(swept, cfg), cfg)
            tau_sweep.append({"tau_GABA_ms": tau, "hopf": sweep_hopf})
        write_json_atomic(
            run.export / "continuation.json",
            {
                "schema": "exp117.compute/v3",
                "configuration": cfg,
                "continuation": continuation_rows(branch, cfg),
                "hopf": hopf,
                "tau_GABA_sweep": tau_sweep,
            },
        )
        np.savez_compressed(run.export / "recording.npz", **ramps)
        run.record["execution"]["backend"] = "snnlab.meanfield rate-conductance ODE"
        run.record["execution"]["environment"] = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "snnlab": version("snnlab"),
        }
    return run.run_id


def continuation(model, configuration):
    """Follow the recipe's current-drive grid on one native equilibrium branch."""
    numerical = configuration["numerical_protocol"]
    return mf.continuation(
        model,
        "E.applied_current",
        np.linspace(*numerical["drive_grid_nA"]),
        guess=numerical["equilibrium_guess"],
    )


def leading_positive_imaginary(values, threshold):
    eligible = [value for value in values if value.imag > threshold]
    return max(eligible, key=lambda value: value.real) if eligible else None


def continuation_rows(branch, configuration):
    """Serialize native equilibria in the article's established rate convention."""
    threshold = configuration["numerical_protocol"][
        "complex_eigenvalue_threshold_per_ms"
    ]
    rows = []
    for drive, equilibrium, stability in branch:
        values = sorted(
            stability.eigenvalues,
            key=lambda value: (value.real, value.imag),
            reverse=True,
        )
        leading = leading_positive_imaginary(values, threshold)
        rows.append(
            {
                "I_ext_nA": drive,
                "equilibrium": (model_state(equilibrium.state) * EXPORT_SCALE).tolist(),
                "eigenvalues_per_ms": [[float(v.real), float(v.imag)] for v in values],
                "leading_complex_per_ms": None
                if leading is None
                else [float(leading.real), float(leading.imag)],
            }
        )
    return rows


def model_state(state):
    """Order named native states for the established four-column export."""
    return np.array(
        [
            state[name]
            for name in ("E.rate", "I.rate", "E_to_I.conductance", "I_to_E.conductance")
        ]
    )


def refine_hopf(model, branch, configuration):
    """Refine a native oscillatory stability crossing and check transversality."""
    numerical = configuration["numerical_protocol"]
    refinement = numerical["hopf_refinement"]
    crossings = mf.find_stability_crossings(
        model,
        "E.applied_current",
        branch,
        tolerance=refinement["xtol_nA"],
        eigenvalue_tolerance=numerical["complex_eigenvalue_threshold_per_ms"],
    )
    crossing = next(
        (
            row
            for row in crossings
            if row.oscillatory and row.stable_before and row.stable_after is False
        ),
        None,
    )
    if crossing is None:
        raise RuntimeError("no oscillatory loss of equilibrium stability found")
    drive = crossing.value
    step = refinement["transversality_step_nA"]
    neighbours = []
    for value in (drive - step, drive + step):
        equilibrium = mf.find_equilibrium(
            model,
            guess=crossing.equilibrium.state,
            parameters={"E.applied_current": value},
        )
        neighbours.append(mf.linearise(model, at=equilibrium).max_real)
    values = crossing.eigenvalues
    leading = leading_positive_imaginary(
        values, numerical["complex_eigenvalue_threshold_per_ms"]
    )
    pair = {
        int(np.argmin(abs(values - leading))),
        int(np.argmin(abs(values - leading.conjugate()))),
    }
    remaining = [value.real for index, value in enumerate(values) if index not in pair]
    return {
        "coarse_bracket_nA": list(crossing.bracket),
        "I_ext_star_nA": drive,
        "equilibrium": (
            model_state(crossing.equilibrium.state) * EXPORT_SCALE
        ).tolist(),
        "eigenvalues_per_ms": [[float(v.real), float(v.imag)] for v in values],
        "critical_pair_per_ms": [float(leading.real), float(leading.imag)],
        "remaining_max_real_per_ms": float(max(remaining)),
        "crossing_slope_per_ms_per_nA": float(
            (neighbours[1] - neighbours[0]) / (2 * step)
        ),
        "omega_Hopf_rad_per_ms": float(abs(leading.imag)),
        "f_Hopf_Hz": crossing.frequency_hz,
    }


def criticality_ramps(model, hopf, configuration):
    """Carry native states through ascending and descending drive conditions."""
    cfg = configuration["numerical_protocol"]["criticality_ramp"]
    start, stop = cfg["span_relative_to_hopf_nA"]
    drives = np.linspace(
        hopf["I_ext_star_nA"] + start, hopf["I_ext_star_nA"] + stop, cfg["points"]
    )
    times = np.linspace(
        cfg["measurement_start_ms"], cfg["duration_ms"], cfg["measurement_samples"]
    )
    interval = float(times[1] - times[0])
    initial = mf.find_equilibrium(
        model,
        guess=configuration["numerical_protocol"]["equilibrium_guess"],
        parameters={"E.applied_current": float(drives[0])},
    )
    state = dict(initial.state)
    state["E.rate"] += 1000 * cfg["initial_rate_perturbation_per_ms"]
    branches = {}
    for name, indices in (
        ("up", range(drives.size)),
        ("down", range(drives.size - 1, -1, -1)),
    ):
        states = np.empty((drives.size, 4, times.size), dtype=float)
        for index in indices:
            trajectory = mf.integrate(
                model,
                duration=cfg["duration_ms"],
                initial=state,
                parameters={"E.applied_current": float(drives[index])},
                sample_interval=interval,
                method=cfg["solver"],
                rtol=cfg["rtol"],
                atol=cfg["atol"] / EXPORT_SCALE,
                max_step=cfg["max_step_ms"],
            )
            # Native integration samples from zero; retain the measurement window
            # and carry the actual endpoint into the next drive condition.
            samples = np.searchsorted(trajectory.time_ms, times)
            if not np.allclose(trajectory.time_ms[samples], times, rtol=0, atol=1e-9):
                raise RuntimeError(
                    "native integration missed requested measurement times"
                )
            values = model_state(trajectory.state)
            states[index] = values[:, samples] * EXPORT_SCALE[:, None]
            state = {key: value[-1] for key, value in trajectory.state.items()}
        branches[name] = states
    return {
        "drives_nA": drives,
        "measurement_times_ms": times,
        "states_up": branches["up"],
        "states_down": branches["down"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    print(compute(**vars(parser.parse_args())))
