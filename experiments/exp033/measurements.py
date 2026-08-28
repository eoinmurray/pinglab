"""Measurements of retained numerical solutions; no integration or root solving."""

import numpy as np
from pingstore.contracts import PingstoreError

from . import recipe
from .summary import summary


def series(row, dimension, *, end=None):
    t, y = np.asarray(row["t_ms"]), np.asarray(row["Y"])
    if (
        t.ndim != 1
        or t.size < 2
        or y.shape != (dimension, t.size)
        or not np.isfinite(t).all()
        or not np.isfinite(y).all()
        or not (np.diff(t) > 0).all()
        or (end is not None and t[-1] != end)
    ):
        raise PingstoreError("invalid or incomplete exp033 trajectory")
    return t, y


def validate_sweep(rows, grid, dimension):
    if not rows:
        raise PingstoreError("empty exp033 continuation")
    drives = [r["I_ext"] for r in rows]
    if drives != sorted(set(drives)) or not set(drives) <= set(grid):
        raise PingstoreError("exp033 continuation disagrees with its recipe grid")
    for row in rows:
        fp, eigs = np.asarray(row["fp"]), np.asarray(row["eigs"])
        if (
            fp.shape != (dimension,)
            or eigs.shape != (dimension, 2)
            or not np.isfinite(fp).all()
            or not np.isfinite(eigs).all()
        ):
            raise PingstoreError("invalid exp033 fixed point or eigenvalues")


def coarse_hopf(rows):
    previous = None
    for row in rows:
        eigs = [complex(*e) for e in row["eigs"] if abs(e[1]) > 1e-6]
        leading = max(eigs, key=lambda e: e.real) if eigs else None
        if (
            previous is not None
            and previous[1] is not None
            and leading is not None
            and previous[1].real < 0 <= leading.real
        ):
            return {
                "I_ext_star": float(row["I_ext"]),
                "omega_star": float(abs(leading.imag)),
                "freq_star_Hz": float(1000 * abs(leading.imag) / (2 * np.pi)),
                "fp_at_star": row["fp"],
                "leading_eigenvalue": [float(leading.real), float(leading.imag)],
                "coarse_bracket_nA": [float(previous[0]["I_ext"]), float(row["I_ext"])],
            }
        previous = (row, leading)
    return None


def validate_continuation(record, grid):
    validate_sweep(record["sweep"], grid, 4)
    coarse = coarse_hopf(record["sweep"])
    hopf = record["hopf"]
    if (coarse is None) != (hopf is None):
        raise PingstoreError("exp033 Hopf detection disagrees with retained sweep")
    if hopf is None:
        return
    lo, hi = coarse["coarse_bracket_nA"]
    if (
        hopf["coarse_bracket_nA"] != [lo, hi]
        or not lo <= hopf["I_ext_star"] <= hi
        or abs(hopf["leading_eigenvalue"][0]) > 1e-7
        or not np.isclose(hopf["omega_star"], abs(hopf["leading_eigenvalue"][1]))
        or not np.isclose(hopf["freq_star_Hz"], 1000 * hopf["omega_star"] / (2 * np.pi))
    ):
        raise PingstoreError("invalid exp033 refined Hopf evidence")


def hysteresis(branches, hopf):
    cfg = recipe.configuration()["hysteresis"]
    grid = np.linspace(
        hopf["I_ext_star"] + cfg["span_nA"][0],
        hopf["I_ext_star"] + cfg["span_nA"][1],
        cfg["points"],
    )
    measured = {}
    for direction in ("up", "down"):
        rows = branches[direction]
        if [r["I_ext"] for r in rows] != grid.tolist():
            raise PingstoreError("incomplete exp033 hysteresis grid")
        values = []
        for row in rows:
            t, y = series(row, 4, end=cfg["t_max_ms"])
            e = y[0, t >= cfg["settle_start_ms"]]
            amplitude = float(e.max() - e.min()) if e.size >= 10 else 0.0
            values.append({"I_ext": row["I_ext"], "amp": amplitude})
        measured[direction] = values
    up, down = measured["up"], measured["down"]
    thr = cfg["threshold"]
    gap = float(max(abs(d["amp"] - u["amp"]) for u, d in zip(up, down)))
    on = next((u["I_ext"] for u in up if u["amp"] > thr), None)
    off = next((d["I_ext"] for d in down if d["amp"] > thr), None)
    width = float(on - off) if on is not None and off is not None else None
    above = [
        (u["I_ext"] - hopf["I_ext_star"], u["amp"])
        for u in up
        if u["I_ext"] > hopf["I_ext_star"] + 1e-9
    ]
    slope, r2 = 0.0, 0.0
    if len(above) >= 2:
        x = np.array([a[0] for a in above])
        ysq = np.array([a[1] ** 2 for a in above])
        m, c = np.polyfit(x, ysq, 1)
        ss_res = float(np.sum((ysq - (m * x + c)) ** 2))
        ss_tot = float(np.sum((ysq - ysq.mean()) ** 2))
        slope = float(m)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "verdict": "supercritical"
        if gap < thr and slope > 0 and r2 > 0.9
        else "subcritical/inconclusive",
        "hyst_gap": gap,
        "hyst_width_nA": width,
        "A2_slope": slope,
        "A2_r2": r2,
        **measured,
    }


def cycle(record):
    tt, y = series(record["waveform"], 4, end=700.0)
    if tt.size != 1500:
        raise PingstoreError("incomplete exp033 cycle sampling")
    e, i = y[0], y[1]
    ez, iz = e - e.mean(), i - i.mean()
    lags = (np.arange(len(tt)) - len(tt) // 2) * (tt[1] - tt[0])
    xc = np.correlate(iz, ez, mode="same")
    return {
        "I_ext": record["I_ext"],
        "e_leads_i_ms": float(abs(lags[np.argmax(xc)])),
        "e_peak_to_peak": float(np.ptp(e)),
    }


def spiking_medians(document):
    rows = document["results"]
    expected = {(tau, seed) for tau in recipe.TAU_GRID_MS for seed in (42, 43, 44)}
    keys = [(r["tau_gaba_ms"], r["seed"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise PingstoreError("exp033 requires all 18 exp041 frequency rows")
    if any(not np.isfinite(r["f_gamma_hz"]) or r["f_gamma_hz"] <= 0 for r in rows):
        raise PingstoreError("invalid exp041 frequencies")
    return {
        tau: float(
            np.median([r["f_gamma_hz"] for r in rows if r["tau_gaba_ms"] == tau])
        )
        for tau in recipe.TAU_GRID_MS
    }


def analyse(raw, frequencies):
    cfg = recipe.configuration()
    if raw.get("schema") != "exp033.compute/v1" or raw.get("recipe") != cfg:
        raise PingstoreError("exp033 compute evidence has an inconsistent recipe")
    grid = np.linspace(*cfg["drive_grid"])
    ref = raw["reference"]
    validate_continuation(ref, grid)
    hopf = ref["hopf"]
    crit, twod, lc = None, None, None
    coordinates = {"sweep": ref["sweep"]}
    if hopf:
        crit = hysteresis(ref["ramp"], hopf)
        lc = cycle(ref["cycle"])
        comp = raw["comparison"]
        twod = {"I_ext": comp["I_ext"]}
        for key, dimension in (("4d", 4), ("2d", 2)):
            t, y = series(comp[key], dimension, end=300)
            d = y[0] - comp["fp"][0]
            twod["pp_" + key] = float(d[t > 150].max() - d[t > 150].min())
        wave = ref["cycle"]["waveform"]
        coordinates.update(
            cycle={**lc, "t_ms": wave["t_ms"], "E": wave["Y"][0], "I": wave["Y"][1]},
            waveform=wave,
            phase=ref["cycle"]["phase"],
            ladder={},
        )
        series(coordinates["phase"], 4, end=700)
        for key, dimension, idx in (("4d", 4, 3), ("3d", 3, 2), ("2d", 2, 1)):
            row = raw["ladder"][key]
            t, y = series(row, dimension, end=400)
            coordinates["ladder"][key] = {
                "t_ms": t,
                "deviation": y[idx] - row["fp"][idx],
            }
    reductions = raw["reductions"]
    expected = {
        "three_d_qss",
        "keep_E_I (Wilson-Cowan)",
        "keep_ge_gi (QSS rates)",
        "keep_E_gi (fast/slow)",
        "keep_E_ge",
        "keep_I_gi",
        "keep_I_ge",
    }
    if set(reductions) != expected:
        raise PingstoreError("incomplete exp033 dimensional reductions")
    for key, rows in reductions.items():
        validate_sweep(rows, grid, 3 if key == "three_d_qss" else 2)
    h3 = coarse_hopf(reductions["three_d_qss"])
    two_d = {k: coarse_hopf(v) for k, v in reductions.items() if k != "three_d_qss"}
    if [r["tau_gaba_ms"] for r in raw["frequency"]] != cfg["tau_grid_ms"]:
        raise PingstoreError("incomplete exp033 inhibitory-decay sweep")
    freq = []
    for row in raw["frequency"]:
        validate_continuation(row, grid)
        h = row["hopf"]
        freq.append(
            {
                "tau_gaba_ms": row["tau_gaba_ms"],
                "f_star_Hz": h["freq_star_Hz"] if h else None,
                "I_ext_star": h["I_ext_star"] if h else None,
            }
        )
    if [r["sigma_V_mV"] for r in raw["sensitivity"]] != cfg["sigma_grid_mV"]:
        raise PingstoreError("incomplete exp033 noise sensitivity")
    sens_rows: list[dict] = []
    for row in raw["sensitivity"]:
        validate_continuation(row, np.linspace(*cfg["sensitivity_grid"]))
        validate_continuation(row["convergence"], np.linspace(*cfg["convergence_grid"]))
        h, fine = row["hopf"], row["convergence"]["hopf"]
        entry = {"sigma_V_mV": row["sigma_V_mV"], "hopf_exists": h is not None}
        if h:
            criticality = hysteresis(row["ramp"], h)
            cy = cycle(row["cycle"])
            entry.update(
                hopf=h,
                fixed_point_at_hopf=dict(
                    zip(("E_per_ms", "I_per_ms", "g_eI", "g_iE"), h["fp_at_star"])
                ),
                criticality={
                    k: v for k, v in criticality.items() if k not in ("up", "down")
                },
                limit_cycle={
                    "relative_drive_nA": recipe.LIMIT_CYCLE_OFFSET_NA,
                    "e_peak_to_peak": cy["e_peak_to_peak"],
                    "e_leads_i_ms": cy["e_leads_i_ms"],
                },
                convergence_check={
                    "comparison_grid_step_nA": 0.005,
                    "I_ext_star": fine["I_ext_star"] if fine else None,
                    "absolute_difference_nA": abs(h["I_ext_star"] - fine["I_ext_star"])
                    if fine
                    else None,
                },
            )
        sens_rows.append(entry)
    sensitivity = {
        "sigma_grid_mV": cfg["sigma_grid_mV"],
        "reference_sigma_mV": recipe.SIGMA_V_MV,
        "settings": {
            "drive_interval_nA": [0.0, 1.2],
            "coarse_grid_step_nA": 0.01,
            "hopf_refinement": "Brent root of leading complex eigenvalue real part",
            "hysteresis_span_relative_nA": list(recipe.HYSTERESIS_SPAN_NA),
            "hysteresis_points": recipe.HYSTERESIS_POINTS,
            "amplitude_threshold": 1e-4,
            "integration_t_max_ms": 2000.0,
            "integration_settle_start_ms": 1500.0,
            "limit_cycle_relative_drive_nA": recipe.LIMIT_CYCLE_OFFSET_NA,
        },
        "rows": sens_rows,
        "topology_retained": all(r["hopf_exists"] for r in sens_rows),
        "supercritical_retained": all(
            r.get("criticality", {}).get("verdict") == "supercritical"
            for r in sens_rows
        ),
    }
    return summary(
        hopf, crit, twod, lc, freq, spiking_medians(frequencies), h3, two_d, sensitivity
    ), coordinates
