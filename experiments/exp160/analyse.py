"""Measure native spikes and conductances from the explicit matched pair."""

import argparse
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from experiments.exp160 import recipe
from pingstore.library import source_run, stage_run, write_json_atomic
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from snnlab.meanfield import Trajectory
from snnlab.sim import NetworkSnapshot, RecordingData


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
    plot_arrivals = arrivals[(arrivals >= 0) & (arrivals <= cfg["duration_ms"])]
    arrivals = arrivals[(arrivals >= 1000) & (arrivals <= 1450)]
    gi = np.asarray(state["PING_B_I_to_E.conductance"])
    late = (t >= 1000) & (t <= 1500)
    normalized_gi = (np.interp(arrivals, t, gi) - gi[late].min()) / np.ptp(gi[late])
    slopes = np.interp(arrivals, t, np.gradient(gi, t))
    lags = [float(peaks["B"][np.argmin(np.abs(peaks["B"] - a))] - a) for a in arrivals]
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
        mean_nearest_B_E_peak_offset_ms=float(np.mean(lags)) if lags else None,
    )
    tail = phase_t >= 1000
    metrics["last_500_ms"] = dict(
        plv=float(abs(np.exp(2j * np.pi * delta[tail]).mean())),
        drift_hz=float(np.polyfit(phase_t[tail] / 1000, delta[tail], 1)[0]),
        phase_span_cycles=float(np.ptp(delta[tail])),
    )
    return metrics, dict(time=t, phase=full_delta, peaks=peaks, arrivals=plot_arrivals)


def alignment(solution, spikes, cfg):
    """Descriptive timing criterion using native input and unsmoothed output peaks."""
    from scipy.integrate import trapezoid

    criteria = recipe.alignment_configuration()
    t = solution.time_ms
    gi = solution.state["PING_B_I_to_E.conductance"]
    gx = solution.state["PING_A_E_to_PING_B_E_K_EE.conductance"]
    start, end = cfg["timing"]["window_ms"]
    window = (t >= start) & (t <= end)
    threshold = float(
        gi[window].min() + cfg["timing"]["low_inhibition_fraction"] * np.ptp(gi[window])
    )
    distance = round(cfg["phase"]["min_peak_distance_ms"] / cfg["network"]["dt_ms"])
    input_peaks, _ = find_peaks(
        gx, distance=distance, prominence=0.1 * np.ptp(gx[window])
    )
    output = spikes["B_E"][:, 0, :].mean(axis=1)
    output_peaks, _ = find_peaks(
        output, distance=distance, prominence=0.1 * np.ptp(output[window])
    )
    rows = []
    for p in input_peaks:
        if not start <= t[p] <= end or not len(output_peaks):
            continue
        nearest = output_peaks[np.argmin(np.abs(t[output_peaks] - t[p]))]
        rows.append(
            dict(
                input_peak_ms=float(t[p]),
                receiver_peak_ms=float(t[nearest]),
                output_lag_ms=float(t[nearest] - t[p]),
                low_inhibition=bool(gi[p] <= threshold),
            )
        )
    total_area = float(trapezoid(gx[window], t[window]))
    low_fraction = float(np.mean([r["low_inhibition"] for r in rows])) if rows else 0.0
    later_fraction = (
        float(
            np.mean(
                [
                    criteria["min_output_lag_ms"] - 1e-8
                    <= r["output_lag_ms"]
                    <= criteria["max_output_lag_ms"]
                    for r in rows
                ]
            )
        )
        if rows
        else 0.0
    )
    return dict(
        window_ms=[start, end],
        inhibition_threshold_us=threshold,
        criteria=criteria,
        fraction_input_peaks_in_low_inhibition=low_fraction,
        fraction_output_peaks_following_input=later_fraction,
        input_area_fraction_in_low_inhibition=float(
            trapezoid(gx[window] * (gi[window] <= threshold), t[window]) / total_area
        )
        if total_area
        else 0.0,
        timing_passed=bool(
            rows
            and low_fraction >= criteria["min_fraction"]
            and later_fraction >= criteria["min_fraction"]
        ),
        per_volley=rows,
    )


def read_trial(computed, name, cfg):
    dt = cfg["network"]["dt_ms"]
    steps = round(cfg["duration_ms"] / dt)
    with np.load(computed.file(f"{name}-spikes.npz"), allow_pickle=False) as data:
        spikes = dict(data)
    if set(spikes) != {f"{c}_{p}" for c in ("A", "B") for p in ("E", "I")}:
        raise ValueError("missing population spikes")
    state = {}
    for c in ("A", "B"):
        for p in ("E", "I"):
            values = spikes[f"{c}_{p}"]
            size = cfg["network"]["populations"][p]["size"]
            if (
                values.shape != (steps, 1, size)
                or values.dtype != np.uint8
                or not ((values == 0) | (values == 1)).all()
            ):
                raise ValueError("invalid spike array")
            state[f"PING_{c}_{p}.rate"] = gaussian_filter1d(
                values[:, 0, :].mean(axis=1) * 1000 / dt, cfg["rate_smoothing_ms"] / dt
            )
    recording = RecordingData.read(computed.unit(f"{name}-conductances"))
    if set(recording.signals) != set(cfg["recording_signals"]):
        raise ValueError("missing native conductance")
    for signal, values in recording.signals.items():
        info = recording.info[signal]
        if (
            info.unit != "uS"
            or info.dt_ms != dt
            or values.shape != (steps, 1, cfg["network"]["populations"]["E"]["size"])
        ):
            raise ValueError("conductance coordinates/units differ")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("invalid conductance")
        state[signal] = values[:, 0, :].mean(axis=1)
    return spikes, Trajectory(np.arange(steps) * dt, state)


def analyse(*, source, run_id=None, selection_sources=()):
    started = perf_counter()
    computed = source_run(
        REPO / ".pingstore", source, experiment=recipe.SLUG, stage="compute"
    )
    cfg = computed.record["execution"]["configuration"]
    if cfg["schema"] != recipe.SCHEMA:
        raise ValueError("unsupported recipe")
    solutions = {}
    diagnostics = {}
    metrics = {}
    spikes = {}
    for name in ("control", "coupled"):
        spikes[name], solutions[name] = read_trial(computed, name, cfg)
        metrics[name], diagnostics[name] = measure(solutions[name], cfg)
    prefix = round(cfg["network"]["onset_ms"] / cfg["network"]["dt_ms"])
    for key in spikes["control"]:
        np.testing.assert_array_equal(
            spikes["control"][key][:prefix], spikes["coupled"][key][:prefix]
        )
    for p in ("E", "I"):
        np.testing.assert_array_equal(
            spikes["control"][f"A_{p}"], spikes["coupled"][f"A_{p}"]
        )
    # Native snapshots include execution scaling: authenticate it against the base reference.
    pin = computed.record["inputs"]["parameter_reference"]
    reference = source_run(
        REPO / ".pingstore",
        pin["run_id"],
        experiment="exp159",
        stage="compute",
        reference=pin,
    )
    base = NetworkSnapshot.read(
        reference.file(recipe.REFERENCE_SNAPSHOT)
    ).load_parameters()
    for name, k in [("control", 0), ("coupled", cfg["coupling_us"])]:
        initial = NetworkSnapshot.read(
            computed.file(f"{name}-execution/start/snapshot.json")
        ).load_parameters()
        for key, value in base.items():
            scale = recipe.scales(cfg, k).get(key.removesuffix(".weight"), 1)
            np.testing.assert_allclose(
                initial[key].numpy(), value.numpy() * scale, rtol=1e-6, atol=1e-9
            )
    reference.check_unchanged()
    arrival_alignment = alignment(solutions["coupled"], spikes["coupled"], cfg)
    selection = {
        f"selection_{index:02d}": source_run(
            REPO / ".pingstore", identity, experiment=recipe.SLUG, stage="analyse"
        )
        for index, identity in enumerate(selection_sources)
    }
    with stage_run(
        REPO,
        recipe.SLUG,
        "analyse",
        inputs=dict(compute=computed, **selection),
        configuration=cfg,
        run_id=run_id,
    ) as run:
        write_json_atomic(
            run.export / "results.json",
            dict(
                schema="exp160.analysis/v2",
                metrics=metrics,
                alignment=arrival_alignment,
            ),
        )
        arrays = dict(time_ms=solutions["control"].time_ms)
        for name, solution in solutions.items():
            for key, values in solution.state.items():
                arrays[name + "__" + key] = values
            arrays[name + "__phase"] = diagnostics[name]["phase"]
            arrays[name + "__arrivals"] = diagnostics[name]["arrivals"]
            arrays[name + "__B_peaks"] = diagnostics[name]["peaks"]["B"]
        np.savez_compressed(run.export / "traces.npz", **arrays)
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
        computed.check_unchanged()
        for item in selection.values():
            item.check_unchanged()
    print(metrics)
    return run.run_id


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True)
    p.add_argument("--run-id")
    p.add_argument(
        "--selection-source", dest="selection_sources", action="append", default=[]
    )
    analyse(**vars(p.parse_args()))
