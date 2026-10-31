"""Validate the explicit compute pair and measure locking and volley timing."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from experiments.exp159 import recipe
from experiments.exp159.compute import verify_pair
from pingstore.library import load_json, source_run, stage_run, write_json_atomic
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from snnlab.meanfield import Trajectory


def rhythm(time, rate, window, settings):
    """Use absolute amplitude before peak phase, excluding numerical ripples."""
    selected = (time >= window[0]) & (time < window[1])
    t, y = time[selected], rate[selected]
    if len(t) < 3:
        raise ValueError("rhythm window needs at least three observations")
    amplitude = float(np.ptp(y))
    mean = float(np.mean(y))
    minimum = max(
        settings["min_amplitude_hz"], settings.get("min_relative_amplitude", 0) * mean
    )
    peaks, _ = find_peaks(
        y,
        distance=max(
            1, round(settings.get("min_peak_distance_ms", 15) / (time[1] - time[0]))
        ),
        prominence=max(
            0.1 * settings["min_amplitude_hz"],
            settings.get("prominence_fraction", 0.1) * amplitude,
        ),
    )
    intervals = np.diff(t[peaks])
    frequency = float(1000 / intervals.mean()) if len(intervals) else None
    cv = float(intervals.std() / intervals.mean()) if len(intervals) else None
    band = settings["frequency_band_hz"]
    failures = []
    if amplitude < minimum:
        failures.append("insufficient amplitude")
    if len(peaks) < settings["min_peaks"]:
        failures.append("too few peaks")
    if frequency is None or (band is not None and not band[0] <= frequency <= band[1]):
        failures.append("frequency unavailable or outside requested band")
    if cv is None or cv > settings["max_interval_cv"]:
        failures.append("irregular or unavailable peak intervals")
    summary = dict(
        valid=not failures,
        reasons=failures,
        mean_hz=mean,
        amplitude_hz=amplitude,
        required_amplitude_hz=minimum,
        peaks=len(peaks),
        frequency_hz=frequency,
        interval_cv=cv,
        window_ms=list(window),
    )
    phase = np.full(time.shape, np.nan)
    # Phase can be displayed through a coupling-induced frequency transition.
    # Regularity and sustained-amplitude checks still gate locking decisions.
    if amplitude >= minimum and len(peaks) >= 2:
        phase = np.interp(
            time, t[peaks], 2 * np.pi * np.arange(len(peaks)), left=np.nan, right=np.nan
        )
    return summary, phase


def measure(solution, cfg):
    t = np.asarray(solution.time_ms)
    state = solution.state
    settings = cfg["phase"]
    summaries, phases = {}, {}
    for c in ("A", "B"):
        summaries[c], phases[c] = rhythm(
            t,
            np.asarray(state[f"PING_{c}_E.rate"]),
            cfg["phase"]["window_ms"],
            settings,
        )
    valid = np.isfinite(phases["A"]) & np.isfinite(phases["B"])
    if valid.sum() < 3:
        raise ValueError("insufficient common phase-valid samples")
    delta = (phases["A"][valid] - phases["B"][valid]) / (2 * np.pi)
    phase_t = t[valid]
    drift = float(np.polyfit(phase_t / 1000, delta, 1)[0])
    plv = float(abs(np.exp(2j * np.pi * delta).mean()))
    span = float(np.ptp(delta))
    frequencies = {}
    for c in ("A", "B"):
        frequencies[c] = {
            label: rhythm(t, np.asarray(state[f"PING_{c}_E.rate"]), window, settings)[
                0
            ]["frequency_hz"]
            for label, window in [("before", [100, 500]), ("after", [1000, 1500])]
        }
    all_phase = {
        c: rhythm(t, np.asarray(state[f"PING_{c}_E.rate"]), [0, 1500], settings)[1]
        for c in ("A", "B")
    }
    full_delta = (all_phase["A"] - all_phase["B"]) / (2 * np.pi)
    peaks = {}
    for c in ("A", "B"):
        rate = np.asarray(state[f"PING_{c}_E.rate"])
        indices, _ = find_peaks(
            rate,
            distance=round(15 / (t[1] - t[0])),
            prominence=0.1 * np.ptp(rate[t >= 800]),
        )
        peaks[c] = t[indices]
    arrivals = peaks["A"] + cfg["network"]["cross"]["delay_ms"]
    arrivals = arrivals[(arrivals >= 1000) & (arrivals <= 1450)]
    gi = np.asarray(state["PING_B_I_to_E.conductance"])
    late = (t >= 1000) & (t <= 1500)
    normalized_gi = (np.interp(arrivals, t, gi) - gi[late].min()) / np.ptp(gi[late])
    slopes = np.interp(arrivals, t, np.gradient(gi, t))
    lags = []
    for a in arrivals:
        next_peaks = peaks["B"][(peaks["B"] > a) & (peaks["B"] < a + 25)]
        if len(next_peaks):
            lags.append(float(next_peaks[0] - a))
    metrics = dict(
        plv=plv,
        drift_hz=drift,
        phase_span_cycles=span,
        frequencies_hz=frequencies,
        rhythms=summaries,
        locked=bool(
            all(v["valid"] for v in summaries.values())
            and plv > cfg["phase"]["min_concentration"]
            and abs(drift) < cfg["phase"]["max_abs_drift_hz"]
            and span < cfg["phase"]["max_phase_span_cycles"]
        ),
        mean_relative_inhibition_at_arrival=float(normalized_gi.mean()),
        fraction_arrivals_during_inhibitory_decay=float(np.mean(slopes < 0)),
        mean_arrival_to_next_B_E_peak_ms=float(np.mean(lags)) if lags else None,
    )
    tail = phase_t >= 1000
    metrics["last_500_ms"] = dict(
        plv=float(abs(np.exp(2j * np.pi * delta[tail]).mean())),
        drift_hz=float(np.polyfit(phase_t[tail] / 1000, delta[tail], 1)[0]),
        phase_span_cycles=float(np.ptp(delta[tail])),
    )
    return metrics, dict(time=t, phase=full_delta, peaks=peaks, arrivals=arrivals)


def timing(data, cfg):
    settings = cfg["timing"]
    t = np.asarray(data["time_ms"])
    state = {k: np.asarray(v) for k, v in data["state"].items()}
    gi = state["PING_B_I_to_E.conductance"]
    gx = state["PING_A_E_to_PING_B_E_K_EE.conductance"]
    be = state["PING_B_E.rate"]
    late = (t >= settings["window_ms"][0]) & (t <= settings["window_ms"][1])
    relative_i = (gi - gi[late].min()) / np.ptp(gi[late])
    low = relative_i <= settings["low_inhibition_fraction"]
    epeaks, _ = find_peaks(
        be,
        distance=round(cfg["phase"]["min_peak_distance_ms"] / (t[1] - t[0])),
        prominence=0.1 * np.ptp(be[late]),
    )
    ipeaks, _ = find_peaks(
        gi,
        distance=round(cfg["phase"]["min_peak_distance_ms"] / (t[1] - t[0])),
        prominence=0.1 * np.ptp(gi[late]),
    )
    xpeaks, _ = find_peaks(
        gx,
        distance=round(cfg["phase"]["min_peak_distance_ms"] / (t[1] - t[0])),
        prominence=0.1 * np.ptp(gx[late]),
    )
    rows = []
    for xp in xpeaks:
        if not settings["window_ms"][0] <= t[xp] <= settings["window_ms"][1]:
            continue
        threshold = settings["input_rise_fraction"] * gx[xp]
        previous = np.flatnonzero(
            (t < t[xp]) & (t >= t[xp] - settings["preceding_ms"]) & (gx < threshold)
        )
        if not len(previous):
            continue
        onset = int(previous[-1] + 1)
        next_e = epeaks[
            (epeaks > onset) & (t[epeaks] < t[onset] + settings["output_search_ms"])
        ]
        if not len(next_e):
            continue
        ep = next_e[0]
        next_i = ipeaks[
            (ipeaks > ep) & (t[ipeaks] < t[ep] + settings["inhibition_search_ms"])
        ]
        if not len(next_i):
            continue
        rows.append(
            dict(
                input_20pct_rise_ms=float(t[onset]),
                inhibition_relative_at_rise=float(relative_i[onset]),
                inhibition_decaying_at_rise=bool(gi[onset] < gi[onset - 1]),
                input_rise_to_B_E_peak_ms=float(t[ep] - t[onset]),
                B_E_peak_to_inhibition_peak_ms=float(t[next_i[0]] - t[ep]),
            )
        )
    fraction = float(
        trapezoid(gx[late] * low[late], t[late]) / trapezoid(gx[late], t[late])
    )
    return dict(
        window_ms=settings["window_ms"],
        low_inhibition_definition="bottom quarter of observed inhibitory conductance range",
        fraction_of_incoming_conductance_area_during_low_inhibition=fraction,
        fraction_of_time_with_low_inhibition=float(low[late].mean()),
        fraction_input_rises_during_inhibitory_decay=float(
            np.mean([r["inhibition_decaying_at_rise"] for r in rows])
        ),
        mean_input_rise_to_B_E_peak_ms=float(
            np.mean([r["input_rise_to_B_E_peak_ms"] for r in rows])
        ),
        mean_B_E_peak_to_inhibition_peak_ms=float(
            np.mean([r["B_E_peak_to_inhibition_peak_ms"] for r in rows])
        ),
        per_volley=rows,
    )


def read_trajectory(computed, label, cfg):
    saved = load_json(computed.file(f"{label}-trajectory.json"))
    model = load_json(computed.file(f"{label}-model.json"))
    time = np.asarray(saved["time_ms"])
    state = {k: np.asarray(v) for k, v in saved["state"].items()}
    expected = np.append(
        np.arange(0, cfg["duration_ms"], cfg["sample_interval_ms"]), cfg["duration_ms"]
    )
    if not np.array_equal(time, expected) or set(state) != set(model["state_order"]):
        raise ValueError("invalid trajectory grid or states")
    if any(
        v.shape != time.shape or not np.isfinite(v).all() or np.min(v) < -1e-7
        for v in state.values()
    ):
        raise ValueError("invalid trajectory state values")
    parameters = {name: row["value"] for name, row in model["parameters"].items()}
    if (
        saved["integration"]["parameters"] != parameters
        or saved["integration"]["delay_mode"] != "explicit"
    ):
        raise ValueError("trajectory/model physics differ")
    k = 0 if label == "control" else cfg["coupling_us"]
    for name, weight in cfg["incoming_weights_us"].items():
        expected_weight = (
            weight * k / cfg["network"]["base_coupling_us"]
            if "_K_E" in name
            else weight * cfg["reduction_projection_scales"].get(name, 1)
        )
        if not np.isclose(
            parameters[name + ".coupling"], expected_weight, rtol=1e-12, atol=1e-14
        ):
            raise ValueError("trajectory differs from recorded coupling")
    for name, rate in cfg["network"]["input_rates_hz"].items():
        if parameters[name + ".rate"] != rate:
            raise ValueError("trajectory input differs from recorded drive")
    return Trajectory(time, state, saved["integration"]), saved


def analyse(*, source, run_id=None):
    started = perf_counter()
    computed = source_run(
        REPO / ".pingstore", source, experiment=recipe.SLUG, stage="compute"
    )
    cfg = computed.record["execution"]["configuration"]
    if cfg["schema"] != recipe.SCHEMA:
        raise ValueError("unsupported recipe")
    solutions = {}
    metrics = {}
    diagnostics = {}
    for name in ("control", "coupled"):
        solution, saved = read_trajectory(computed, name, cfg)
        solutions[name] = solution
        metrics[name], diagnostics[name] = measure(solution, cfg)
        if name == "coupled":
            details = timing(saved, cfg)
    verify_pair(
        solutions["control"], solutions["coupled"], onset_ms=cfg["network"]["onset_ms"]
    )
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
            dict(schema="exp159.analysis/v1", metrics=metrics, timing=details),
        )
        arrays = {}
        for name, d in diagnostics.items():
            arrays[name + "_phase"] = d["phase"]
            arrays[name + "_arrivals"] = d["arrivals"]
            arrays[name + "_B_peaks"] = d["peaks"]["B"]
        np.savez_compressed(
            run.export / "phase.npz", time_ms=solutions["control"].time_ms, **arrays
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
        computed.check_unchanged()
    return run.run_id


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    analyse(**vars(p.parse_args()))
