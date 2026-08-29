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
from experiments.exp033 import evidence, inputs, recipe
from experiments.exp033 import numerics as model
from pingstore.contracts import PingstoreError, write_json_atomic
from scipy.integrate import solve_ivp


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
):
    if initial is None or not np.isfinite(initial).all():
        raise PingstoreError("exp033 initial fixed point is unavailable")
    sol = solve_ivp(
        rhs,
        (0, duration),
        initial,
        args=(drive, tau, sigma),
        method="LSODA",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=dense,
    )
    if not sol.success or sol.t[-1] != duration or not np.isfinite(sol.y).all():
        raise PingstoreError(f"exp033 integration failed: {sol.message}")
    return sol


def fixed_point(drive, sigma=recipe.SIGMA_V_MV):
    fp = model.fixed_point(drive, sigma=sigma)
    if fp is None or not np.isfinite(fp).all():
        raise PingstoreError(f"exp033 fixed point failed at {drive}")
    return fp


def trajectory(sol, **metadata):
    return {**metadata, "t_ms": sol.t, "Y": sol.y}


def ramp(hopf, sigma):
    cfg = recipe.configuration()["hysteresis"]
    grid = np.linspace(
        hopf["I_ext_star"] + cfg["span_nA"][0],
        hopf["I_ext_star"] + cfg["span_nA"][1],
        cfg["points"],
    )
    state = fixed_point(grid[0], sigma).copy()
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
            )
            state = sol.y[:, -1]
            rows.append(trajectory(sol, I_ext=float(drive)))
        branches[direction] = rows if direction == "up" else rows[::-1]
    return branches


def cycle(hopf, sigma):
    cfg = recipe.configuration()["cycle"]
    drive = hopf["I_ext_star"] + cfg["offset_nA"]
    sol = integrate(
        model.rhs_4d,
        fixed_point(drive, sigma) + [1e-3, 0, 0, 0],
        drive,
        cfg["t_max_ms"],
        sigma=sigma,
        rtol=cfg["rtol"],
        atol=cfg["atol"],
        max_step=cfg["max_step"],
        dense=True,
    )
    period = 1000.0 / hopf["freq_star_Hz"]
    result = trajectory(sol, I_ext=drive)
    for name, periods, count in (("waveform", 3, 1500), ("phase", 4, 2000)):
        tt = np.linspace(700 - periods * period, 700, count)
        result[name] = {"t_ms": tt, "Y": sol.sol(tt)}
    return result


def comparison(hopf):
    drive = hopf["I_ext_star"] + 1.0
    fp = fixed_point(drive)
    return {
        "I_ext": drive,
        "fp": fp,
        "4d": trajectory(integrate(model.rhs_4d, fp + [2e-3, 0, 0, 0], drive, 300)),
        "2d": trajectory(integrate(model.rhs_2d, [fp[0] + 2e-3, fp[1]], drive, 300)),
    }


def ladder():
    result = {}
    for name, rhs, fn, kick in (
        ("4d", model.rhs_4d, model.fixed_point, [2e-3, 0, 0, 0]),
        ("3d", model.rhs_3d_qss, model.fixed_point_3d_qss, [2e-3, 0, 0]),
        ("2d", model.rhs_2d_qss, model.fixed_point_2d_qss, [0, 2e-3]),
    ):
        fp = fn(1.0)
        if fp is None:
            raise PingstoreError("exp033 ladder fixed point failed")
        result[name] = trajectory(integrate(rhs, fp + kick, 1.0, 400), fp=fp)
    return result


def continuation(grid, *, sigma=recipe.SIGMA_V_MV, tau=recipe.TAU_GABA_MS):
    rows = model.sweep(grid, tau_gaba=tau, sigma=sigma)
    return {"sweep": rows, "hopf": model.find_hopf(rows, tau_gaba=tau, sigma=sigma)}


def simulate():
    cfg = recipe.configuration()
    grid = np.linspace(*cfg["drive_grid"])
    result = {
        "schema": "exp033.compute/v1",
        "recipe": cfg,
        "reference": continuation(grid),
        "reductions": {},
        "frequency": [],
        "sensitivity": [],
    }
    hopf = result["reference"]["hopf"]
    if hopf:
        result["reference"].update(
            ramp=ramp(hopf, recipe.SIGMA_V_MV), cycle=cycle(hopf, recipe.SIGMA_V_MV)
        )
        result["comparison"] = comparison(hopf)
        result["ladder"] = ladder()
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
        result["reductions"][name] = model.reduction_sweep(rhs, fp, grid)
    for tau in cfg["tau_grid_ms"]:
        result["frequency"].append({"tau_gaba_ms": tau, **continuation(grid, tau=tau)})
    for sigma in cfg["sigma_grid_mV"]:
        coarse = continuation(np.linspace(*cfg["sensitivity_grid"]), sigma=sigma)
        fine = continuation(np.linspace(*cfg["convergence_grid"]), sigma=sigma)
        row = {"sigma_V_mV": sigma, **coarse, "convergence": fine}
        if coarse["hopf"]:
            row.update(
                ramp=ramp(coarse["hopf"], sigma), cycle=cycle(coarse["hopf"], sigma)
            )
        result["sensitivity"].append(row)
    return result


def compute(*, run_id=None):
    with inputs.execution(REPO, "compute", sources={}, run_id=run_id) as run:
        evidence.write(run.export, simulate())
        write_json_atomic(
            run.evidence / "environment.json",
            {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
            },
        )
    return run.run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="unused v3 identity reserved before dispatch")
    args = parser.parse_args()
    compute(run_id=args.run_id)


if __name__ == "__main__":
    main()
