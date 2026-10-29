"""Bring the three autonomous rhythms closer in the nine-window E-only protocol.

Tune only inhibitory rate relaxation against uncoupled constant-drive controls.
An explicit validated reference supplies all weights, drives, E response times
and numerical settings. No firing-rate target is recalibrated.
"""

import argparse
import gzip
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]

import numpy as np
from experiments.exp157 import recipe
from pingstore.library import source_run, stage_run, write_json_atomic
from scipy.optimize import brentq
from snnlab import meanfield as mf


def integrate(model, cfg, *, uncoupled=False, refined=False):
    parameters = {
        name: 0.0
        for name in model.parameters
        if uncoupled
        and name.startswith(("A_E_to_C_", "B_E_to_C_"))
        and name.endswith(".gain")
    }
    solver = cfg["solver"]
    return mf.integrate(
        model,
        duration=cfg["duration_ms"],
        initial=recipe.initial_state(model, cfg),
        controls=recipe.drive_controls(cfg),
        parameters=parameters,
        sample_interval=cfg["sample_interval_ms"],
        method=solver["method"],
        rtol=solver["rtol"] / (10 if refined else 1),
        atol=solver["atol"] / (10 if refined else 1),
        max_step=solver["max_step_ms"] / (2 if refined else 1),
    )


def terminal_rates(solution, cfg):
    selected = (
        solution.time_ms
        >= cfg["target"]["ramp_end_ms"] - cfg["target"]["terminal_average_ms"]
    ) & (solution.time_ms < cfg["target"]["ramp_end_ms"])
    a, b = (float(np.mean(solution.state[c + "_E.rate"][selected])) for c in ("A", "B"))
    return dict(a_e_mean_hz=a, b_e_mean_hz=b, b_to_a_ratio=b / a)


def autonomous_parameters(model, cfg):
    parameters = {
        f"X_{c}.rate": cfg[
            {"A": "a_drive_hz", "B": "b_hold_drive_hz", "C": "c_drive_hz"}[c]
        ]
        for c in recipe.CIRCUITS
    }
    parameters.update(
        {
            name: 0.0
            for name in model.parameters
            if name.startswith(("A_E_to_C_", "B_E_to_C_")) and name.endswith(".gain")
        }
    )
    return parameters


def integrate_autonomous(model, cfg, *, parameters=None):
    values = {**autonomous_parameters(model, cfg), **(parameters or {})}
    solver = cfg["solver"]
    return mf.integrate(
        model,
        duration=cfg["frequency_calibration"]["duration_ms"],
        initial=recipe.initial_state(model, cfg),
        parameters=values,
        sample_interval=cfg["sample_interval_ms"],
        method=solver["method"],
        rtol=solver["rtol"],
        atol=solver["atol"],
        max_step=solver["max_step_ms"],
    )


def save_trajectory(solution, run, label):
    path = solution.write(run.scratch / "trajectory.json")
    with (
        path.open("rb") as source,
        gzip.open(run.export / f"{label}.json.gz", "wb") as target,
    ):
        shutil.copyfileobj(source, target)
    path.unlink()


def comparison_configuration(source_cfg):
    """Inherit the actual model and controls, changing only calibration settings."""
    if source_cfg.get("protocol") != "nine-500ms-windows-e-only":
        raise ValueError("requires an explicit nine-window E-only reference compute")
    cfg = recipe.configuration(
        b_hold_drive_hz=source_cfg["b_hold_drive_hz"],
        b_final_drive_hz=source_cfg["b_final_drive_hz"],
        tau_rate_ms=source_cfg["tau_rate_ms"],
        self_recurrence_scale=source_cfg["self_recurrence_scale"],
    )
    excluded = {
        "schema",
        "calibration_policy",
        "frequency_calibration",
    }
    if any(
        cfg[key] != value for key, value in source_cfg.items() if key not in excluded
    ):
        raise ValueError(
            "reference settings differ beyond the requested frequency calibration"
        )
    return cfg


def calibrate_frequencies(model, cfg):
    """Resolve three I relaxation times using measured E-rate peak intervals."""
    from experiments.exp157.analyse import peak_phase

    settings = cfg["frequency_calibration"]
    evaluations = []
    for circuit in recipe.CIRCUITS:
        target = settings["targets_hz"][circuit]

        def objective(tau, circuit=circuit, target=target):
            solution = integrate_autonomous(
                model, cfg, parameters={circuit + "_I.tau_rate": float(tau)}
            )
            selected = (solution.time_ms >= settings["measure_start_ms"]) & (
                solution.time_ms < settings["duration_ms"]
            )
            metrics, _, _ = peak_phase(
                solution.time_ms[selected],
                solution.state[circuit + "_E.rate"][selected],
                cfg["analysis"],
            )
            evaluations.append(dict(circuit=circuit, tau_i_ms=float(tau), **metrics))
            if not metrics["gamma"]:
                raise ValueError(f"{circuit}: no sustained rhythm at I tau={tau}")
            return metrics["frequency_hz"] - target

        tau = brentq(
            objective,
            *settings["i_tau_brackets_ms"][circuit],
            xtol=settings["tau_tolerance_ms"],
        )
        cfg["tau_rate_ms"][circuit]["I"] = float(tau)
        print(
            f"Calibrated {circuit}: I tau={tau:.6f} ms, target={target:g} Hz",
            flush=True,
        )
    return evaluations


def compute(reference, *, run_id=None):
    source = source_run(
        REPO / ".pingstore", reference, experiment=recipe.SLUG, stage="compute"
    )
    cfg = comparison_configuration(source.record["execution"]["configuration"])
    with stage_run(
        REPO,
        recipe.SLUG,
        "compute",
        configuration=cfg,
        inputs=dict(reference=source),
        run_id=run_id,
    ) as run:
        run.record["execution"]["snnlab_version"] = version("snnlab")
        run.record["execution"]["backend"] = (
            "snnlab.meanfield deterministic rate-conductance ODE on local CPU"
        )
        run.record["execution"]["comparison"] = (
            "Only the three I rate-relaxation times change; all weights, drives, E relaxation times and numerical settings inherit the explicit reference."
        )
        bundle = recipe.author_network(cfg)
        bundle.write(run.export / "network.bundle", visualise=False)
        shutil.copyfile(source.file("circuit.svg"), run.export / "circuit.svg")
        evaluations = calibrate_frequencies(recipe.build_model(cfg, bundle), cfg)
        run.record["execution"]["configuration"] = cfg
        write_json_atomic(run.export / "calibration.json", evaluations)
        model = recipe.build_model(cfg, bundle)
        model.write(run.export / "model.json")
        save_trajectory(integrate_autonomous(model, cfg), run, "autonomous")
        for label, uncoupled, refined in (
            ("protocol", False, False),
            ("uncoupled", True, False),
            ("refined", False, True),
        ):
            save_trajectory(
                integrate(model, cfg, uncoupled=uncoupled, refined=refined), run, label
            )
            print(f"Completed {label}", flush=True)
        source.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        required=True,
        help="Explicit compute supplying fixed weights, E response times and input schedules",
    )
    parser.add_argument("--run-id")
    compute(**vars(parser.parse_args()))
