"""Retain continuation and ODE solutions; never analyse, draw or publish."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "tools")]

import numpy as np
import scipy
from pingstore.contracts import PingstoreError
from scipy.integrate import solve_ivp

from experiments.exp033 import evidence, inputs, recipe
from experiments.exp033 import numerics as model


def integrate(
    rhs,
    initial,
    drive,
    duration,
    *,
    tau=recipe.TAU_GABA_MS,
    sigma=recipe.SIGMA_V_MV,
    rtol=1e-8,
    atol=1e-11,
    max_step=0.5,
    dense=False,
    method="LSODA",
):
    if initial is None or not np.isfinite(initial).all():
        raise PingstoreError("exp033 initial fixed point is unavailable")
    sol = solve_ivp(
        rhs,
        (0, duration),
        initial,
        args=(drive, tau, sigma),
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=dense,
    )
    if not sol.success or sol.t[-1] != duration or not np.isfinite(sol.y).all():
        raise PingstoreError(f"exp033 integration failed: {sol.message}")
    return sol


def fixed_point(drive, sigma=recipe.SIGMA_V_MV, tau=recipe.TAU_GABA_MS):
    fp = model.fixed_point(drive, tau_gaba=tau, sigma=sigma)
    if fp is None or not np.isfinite(fp).all():
        raise PingstoreError(f"exp033 fixed point failed at {drive}")
    return fp


def trajectory(sol, **metadata):
    return {**metadata, "t_ms": sol.t, "Y": sol.y}


def ramp(hopf, sigma, *, configuration=None):
    recipe_cfg = recipe.validate(configuration or recipe.configuration())
    cfg = recipe_cfg["hysteresis"]
    grid = np.linspace(
        hopf["I_ext_star"] + cfg["span_nA"][0],
        hopf["I_ext_star"] + cfg["span_nA"][1],
        cfg["points"],
    )
    state = fixed_point(grid[0], sigma=sigma, tau=recipe_cfg["tau_GABA_ms"]).copy()
    state[0] += 1e-3
    branches = {}
    for direction, drives in (("up", grid), ("down", grid[::-1])):
        rows = []
        for drive in drives:
            sol = integrate(
                model.rhs_4d,
                state,
                drive,
                cfg["t_max_ms"],
                sigma=sigma,
                rtol=cfg["rtol"],
                atol=cfg["atol"],
                max_step=cfg["max_step"],
                tau=recipe_cfg["tau_GABA_ms"],
                method=recipe_cfg["solver"],
            )
            state = sol.y[:, -1]
            rows.append(trajectory(sol, I_ext=float(drive)))
        branches[direction] = rows if direction == "up" else rows[::-1]
    return branches


def cycle(hopf, sigma, *, configuration=None):
    recipe_cfg = recipe.validate(configuration or recipe.configuration())
    cfg = recipe_cfg["cycle"]
    drive = hopf["I_ext_star"] + cfg["offset_nA"]
    sol = integrate(
        model.rhs_4d,
        fixed_point(drive, sigma=sigma, tau=recipe_cfg["tau_GABA_ms"])
        + [1e-3, 0, 0, 0],
        drive,
        cfg["t_max_ms"],
        sigma=sigma,
        rtol=cfg["rtol"],
        atol=cfg["atol"],
        max_step=cfg["max_step"],
        dense=True,
        tau=recipe_cfg["tau_GABA_ms"],
        method=recipe_cfg["solver"],
    )
    period = 1000.0 / hopf["freq_star_Hz"]
    result = trajectory(sol, I_ext=drive)
    for name, periods, count in (
        ("waveform", cfg["waveform_periods"], cfg["waveform_points"]),
        ("phase", cfg["phase_periods"], cfg["phase_points"]),
    ):
        tt = np.linspace(cfg["t_max_ms"] - periods * period, cfg["t_max_ms"], count)
        result[name] = {"t_ms": tt, "Y": sol.sol(tt)}
    return result


def comparison(hopf, configuration):
    cfg = configuration["comparison"]
    solver = configuration["comparison_and_ladder_solver"]
    drive = hopf["I_ext_star"] + cfg["offset_nA"]
    fp = fixed_point(
        drive,
        sigma=configuration["sigma_V_mV"],
        tau=configuration["tau_GABA_ms"],
    )
    return {
        "I_ext": drive,
        "fp": fp,
        "4d": trajectory(
            integrate(
                model.rhs_4d,
                fp + [2e-3, 0, 0, 0],
                drive,
                cfg["t_max_ms"],
                tau=configuration["tau_GABA_ms"],
                sigma=configuration["sigma_V_mV"],
                method=configuration["solver"],
                **solver,
            )
        ),
        "2d": trajectory(
            integrate(
                model.rhs_2d,
                [fp[0] + 2e-3, fp[1]],
                drive,
                cfg["t_max_ms"],
                tau=configuration["tau_GABA_ms"],
                sigma=configuration["sigma_V_mV"],
                method=configuration["solver"],
                **solver,
            )
        ),
    }


def ladder(configuration):
    cfg = configuration["ladder"]
    solver = configuration["comparison_and_ladder_solver"]
    result = {}
    for name, rhs, fn, kick in (
        ("4d", model.rhs_4d, model.fixed_point, [2e-3, 0, 0, 0]),
        ("3d", model.rhs_3d_qss, model.fixed_point_3d_qss, [2e-3, 0, 0]),
        ("2d", model.rhs_2d_qss, model.fixed_point_2d_qss, [0, 2e-3]),
    ):
        fp = fn(
            cfg["drive_nA"],
            configuration["tau_GABA_ms"],
            sigma=configuration["sigma_V_mV"],
        )
        if fp is None:
            raise PingstoreError("exp033 ladder fixed point failed")
        result[name] = trajectory(
            integrate(
                rhs,
                fp + kick,
                cfg["drive_nA"],
                cfg["t_max_ms"],
                tau=configuration["tau_GABA_ms"],
                sigma=configuration["sigma_V_mV"],
                method=configuration["solver"],
                **solver,
            ),
            fp=fp,
        )
    return result


def continuation(grid, *, sigma=None, tau=None, configuration=None):
    cfg = recipe.validate(configuration or recipe.configuration())
    sigma = cfg["sigma_V_mV"] if sigma is None else sigma
    tau = cfg["tau_GABA_ms"] if tau is None else tau
    rows = model.sweep(grid, tau_gaba=tau, sigma=sigma, eps=cfg["jacobian_eps"])
    refinement = cfg["hopf_refinement"]
    return {
        "sweep": rows,
        "hopf": model.find_hopf(
            rows,
            tau_gaba=tau,
            sigma=sigma,
            eps=cfg["jacobian_eps"],
            xtol=refinement["xtol"],
            rtol=refinement["rtol"],
        ),
    }


def simulate(configuration=None):
    cfg = recipe.validate(configuration or recipe.configuration())
    print("[theory] reference continuation", flush=True)
    grid = np.linspace(*cfg["drive_grid"])
    result = {
        "schema": "exp033.compute/v1",
        "recipe": cfg,
        "reference": continuation(grid, configuration=cfg),
        "reductions": {},
        "frequency": [],
        "sensitivity": [],
    }
    hopf = result["reference"]["hopf"]
    if hopf:
        print(
            f"[theory] reference ramps and cycle; onset {hopf['I_ext_star']:.6f} nA",
            flush=True,
        )
        result["reference"].update(
            ramp=ramp(hopf, cfg["sigma_V_mV"], configuration=cfg),
            cycle=cycle(hopf, cfg["sigma_V_mV"], configuration=cfg),
        )
        result["comparison"] = comparison(hopf, cfg)
        result["ladder"] = ladder(cfg)
    specs = (
        ("three_d_qss", model.rhs_3d_qss, model.fixed_point_3d_qss),
        ("keep_E_I (Wilson-Cowan)", model.rhs_2d, model.fixed_point_2d_wc),
        ("keep_ge_gi (QSS rates)", model.rhs_2d_qss, model.fixed_point_2d_qss),
        ("keep_E_gi (fast/slow)", model.rhs_2d_fastslow, model.fixed_point_2d_fastslow),
        ("keep_E_ge", model.rhs_2d_E_ge, model.fixed_point_2d_E_ge),
        ("keep_I_gi", model.rhs_2d_I_gi, model.fixed_point_2d_I_gi),
        ("keep_I_ge", model.rhs_2d_I_ge, model.fixed_point_2d_I_ge),
    )
    for name, rhs, fp in specs:
        print(f"[theory] reduction {name}", flush=True)
        result["reductions"][name] = model.reduction_sweep(
            rhs,
            fp,
            grid,
            tau_gaba=cfg["tau_GABA_ms"],
            sigma=cfg["sigma_V_mV"],
            eps=cfg["jacobian_eps"],
        )
    for tau in cfg["tau_grid_ms"]:
        print(f"[theory] inhibitory decay {tau:g} ms", flush=True)
        result["frequency"].append(
            {"tau_gaba_ms": tau, **continuation(grid, tau=tau, configuration=cfg)}
        )
    for sigma in cfg["sigma_grid_mV"]:
        print(f"[theory] noise sensitivity {sigma:g} mV", flush=True)
        coarse = continuation(
            np.linspace(*cfg["sensitivity_grid"]), sigma=sigma, configuration=cfg
        )
        fine = continuation(
            np.linspace(*cfg["convergence_grid"]), sigma=sigma, configuration=cfg
        )
        row = {"sigma_V_mV": sigma, **coarse, "convergence": fine}
        if coarse["hopf"]:
            row.update(
                ramp=ramp(coarse["hopf"], sigma, configuration=cfg),
                cycle=cycle(coarse["hopf"], sigma, configuration=cfg),
            )
        result["sensitivity"].append(row)
    return result


def compute(*, run_id=None):
    with inputs.execution(REPO, "compute", sources={}, run_id=run_id) as run:
        cfg = run.record["execution"]["configuration"]
        with model.gain_parameters(cfg):
            evidence.write(run.export, simulate(cfg))
        run.record["execution"]["environment"] = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        }
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="unused v4 identity reserved before dispatch")
    args = parser.parse_args()
    compute(run_id=args.run_id)


if __name__ == "__main__":
    main()
