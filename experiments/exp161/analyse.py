"""Validate the native trajectory and measure cue-dependent phase locking."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from experiments.exp161 import recipe
from pingstore.library import load_json, source_run, stage_run, write_json_atomic
from scipy.signal import find_peaks
from snnlab.meanfield import Trajectory


def read_trajectory(computed, cfg):
    d = load_json(computed.file("trajectory.json"))
    t = np.asarray(d["time_ms"])
    s = {k: np.asarray(v) for k, v in d["state"].items()}
    expected = np.append(
        np.arange(0, cfg["duration_ms"], cfg["dt_ms"]), cfg["duration_ms"]
    )
    np.testing.assert_array_equal(t, expected)
    model = load_json(computed.file("model.json"))
    if set(s) != set(model["state_order"]):
        raise ValueError("native model/trajectory states differ")
    if any(
        v.shape != t.shape or not np.isfinite(v).all() or v.min() < -1e-7
        for v in s.values()
    ):
        raise ValueError("invalid trajectory")
    parameters = {k: v["value"] for k, v in model["parameters"].items()}
    if (
        d["integration"]["parameters"] != parameters
        or d["integration"]["delay_mode"] != "explicit"
    ):
        raise ValueError("native model/trajectory parameters differ")
    for name, rate in cfg["input_rates_hz"].items():
        if parameters[name + ".rate"] != rate:
            raise ValueError("input rate differs from recipe")
    for c in "AB":
        for p in "EI":
            if parameters[c + "_E_to_C_" + p + ".coupling"] != cfg["coupling_us"]:
                raise ValueError("cross coupling differs from recipe")
    cue = s["cue_A.conductance"] + s["cue_B.conductance"]
    np.testing.assert_allclose(cue[t > 100], cue[t > 100][0], rtol=1e-7, atol=1e-10)
    if (
        s["cue_B.conductance"][np.argmin(abs(t - (cfg["switch_ms"] - 100)))] > 1e-8
        or s["cue_A.conductance"][np.argmin(abs(t - (cfg["switch_ms"] + 100)))] > 1e-8
    ):
        raise ValueError("cue did not switch targets")
    return Trajectory(t, s, d["integration"])


def measure(sol, cfg):
    t = sol.time_ms
    s = sol.state
    peaks = {}
    phases = {}
    for c in "ABC":
        y = np.array(s[c + "_E.rate"])
        p, _ = find_peaks(
            y,
            distance=round(cfg["phase"]["min_peak_distance_ms"] / cfg["dt_ms"]),
            prominence=max(
                cfg["phase"]["min_prominence_hz"],
                cfg["phase"]["prominence_fraction"]
                * np.ptp(y[t > cfg["timing"]["reference_start_ms"]]),
            ),
        )
        peaks[c] = p
        phases[c] = np.interp(t, t[p], np.arange(len(p)), left=np.nan, right=np.nan)
    result = []
    for (start, end), winner in zip(cfg["windows_ms"], "AB", strict=True):
        w = (t >= start) & (t <= end)
        row = dict(window_ms=[start, end], cued=winner, senders={})
        for c in "ABC":
            p = peaks[c]
            pt = t[p[(t[p] >= start) & (t[p] <= end)]]
            row[c] = dict(
                mean_rate_hz=float(np.mean(s[c + "_E.rate"][w])),
                frequency_hz=float(1000 / np.mean(np.diff(pt)))
                if len(pt) > 1
                else None,
                amplitude_hz=float(np.ptp(s[c + "_E.rate"][w])),
            )
        for c in "AB":
            valid = w & np.isfinite(phases[c]) & np.isfinite(phases["C"])
            delta = (phases[c] - phases["C"])[valid]
            row["senders"][c] = dict(
                plv=float(abs(np.mean(np.exp(2j * np.pi * delta)))),
                drift_cycles_s=float(np.polyfit(t[valid] / 1000, delta, 1)[0]),
                phase_span_cycles=float(np.ptp(delta)),
            )
        other = "B" if winner == "A" else "A"
        win = row["senders"][winner]
        lose = row["senders"][other]
        row["passed"] = bool(
            win["plv"] > cfg["phase"]["min_plv"]
            and abs(win["drift_cycles_s"]) < cfg["phase"]["max_abs_drift_cycles_s"]
            and win["phase_span_cycles"] < cfg["phase"]["max_cued_phase_span_cycles"]
            and lose["phase_span_cycles"] > cfg["phase"]["min_uncued_phase_span_cycles"]
            and row["C"]["amplitude_hz"] > cfg["phase"]["min_receiver_amplitude_hz"]
        )
        result.append(row)
    return dict(epochs=result, switched=all(r["passed"] for r in result)), dict(
        time=t, phases=phases, peaks=peaks
    )


def analyse(*, source, run_id=None):
    started = perf_counter()
    computed = source_run(
        REPO / ".pingstore", source, experiment=recipe.SLUG, stage="compute"
    )
    cfg = computed.record["execution"]["configuration"]
    if cfg["schema"] != recipe.SCHEMA:
        raise ValueError("unsupported recipe")
    sol = read_trajectory(computed, cfg)
    metrics, diag = measure(sol, cfg)
    with stage_run(
        REPO,
        recipe.SLUG,
        "analyse",
        inputs=dict(compute=computed),
        configuration=cfg,
        run_id=run_id,
    ) as run:
        write_json_atomic(
            run.export / "results.json",
            dict(schema="exp161.analysis/v1", metrics=metrics),
        )
        np.savez_compressed(
            run.export / "phase.npz",
            time_ms=sol.time_ms,
            **{c + "_phase": v for c, v in diag["phases"].items()},
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
        computed.check_unchanged()
    print(run.run_id, flush=True)
    return run.run_id


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    analyse(**vars(p.parse_args()))
