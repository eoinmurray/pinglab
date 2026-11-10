"""Validate imported or newly computed spatial relay evidence without simulation."""

import argparse
import json
import shutil
import sys
from pathlib import Path
from time import perf_counter

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "libraries")]
import numpy as np
from experiments.exp172.recipe import (
    cycle_analysis_configuration,
    gaussian,
    selected_parameters,
    trial_plan,
    used_window_configuration,
)
from pingstore.library import canonical_export_file, source_run, stage_run
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares, nnls
from scipy.signal import find_peaks
from snnlab.sim import NetworkSnapshot, RecordingData


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fit_profile(y, centres, sigma, minimum):
    x = np.arange(len(y)) / len(y)
    templates = np.column_stack([gaussian(x, c, sigma) for c in centres])
    design = np.column_stack([np.ones(len(y)), templates])
    coefficients, _ = nnls(design, y)
    prediction = np.sum(design * coefficients[None, :], axis=1)
    denom = np.sum((y - y.mean()) ** 2)
    amplitudes = coefficients[1:]
    total = float(amplitudes.sum())
    # Shape recovery is a diagnostic, not a second choice of ER estimator.
    p0 = [*coefficients, *centres, sigma, sigma]
    lower = [0, 0, 0, centres[0] - 0.08, centres[1] - 0.08, 0.035, 0.035]
    upper = [np.inf, np.inf, np.inf, centres[0] + 0.08, centres[1] + 0.08, 0.18, 0.18]

    def residual(p):
        return (
            p[0] + p[1] * gaussian(x, p[3], p[5]) + p[2] * gaussian(x, p[4], p[6]) - y
        )

    shape = least_squares(residual, p0, bounds=(lower, upper))
    return (
        dict(
            background_hz=float(coefficients[0]),
            amplitudes_hz=amplitudes.tolist(),
            total_amplitude_hz=total,
            fraction_A=float(amplitudes[0] / total) if total >= minimum else None,
            r_squared=float(1 - np.sum((y - prediction) ** 2) / denom)
            if denom > 0
            else None,
            rmse_hz=float(np.sqrt(np.mean((y - prediction) ** 2))),
            recovered_centres=shape.x[3:5].tolist(),
            recovered_sigmas=shape.x[5:7].tolist(),
            shape_fit_success=bool(shape.success),
        ),
        prediction,
        templates,
    )


def copy_fidelity(target, output, n=8):
    """Direct spatial-rate reconstruction, with no fitted gain or offset."""
    target = np.asarray(target).reshape(-1, n).mean(1)
    output = np.asarray(output).reshape(-1, n).mean(1)
    mse = float(np.mean((output - target) ** 2))
    variance = float(np.var(target))
    return dict(
        bin_neurons=n,
        copy_r_squared=1 - mse / variance if variance > 0 else None,
        rmse_hz=float(np.sqrt(mse)),
        correlation=float(np.corrcoef(target, output)[0, 1])
        if np.std(target) > 0 and np.std(output) > 0
        else None,
        mean_rate_ratio=float(output.mean() / target.mean())
        if target.mean() > 0
        else None,
    )


def phase_metrics(spikes, cfg, start, end):
    dt = cfg["dt_ms"]
    t = np.arange(len(spikes["A_E"])) * dt
    phases, frequencies, rhythms = {}, {}, {}
    for c in "ABC":
        rate = gaussian_filter1d(spikes[c + "_E"].mean(1) * 1000 / dt, 1 / dt)
        peaks, _ = find_peaks(
            rate,
            distance=round(15 / dt),
            prominence=max(0.05, 0.1 * np.ptp(rate[t > 500])),
        )
        times = t[peaks]
        if len(times) < 2:
            continue
        phases[c] = np.interp(
            t, times, np.arange(len(times)), left=np.nan, right=np.nan
        )
        local = times[(times >= start) & (times < end)]
        frequencies[c] = (
            float(1000 / np.mean(np.diff(local))) if len(local) > 2 else None
        )
        intervals = np.diff(local)
        cv = float(intervals.std() / intervals.mean()) if len(intervals) > 1 else None
        rhythms[c] = dict(
            volleys=len(local),
            interval_cv=cv,
            valid=bool(
                len(local) >= 12
                and cv is not None
                and cv < 0.2
                and 30 <= frequencies[c] <= 80
            ),
        )
    plv = {}
    for c in "AB":
        if c in phases and "C" in phases:
            valid = (
                (t >= start)
                & (t < end)
                & np.isfinite(phases[c])
                & np.isfinite(phases["C"])
            )
            plv[c] = (
                float(
                    abs(
                        np.mean(
                            np.exp(2j * np.pi * (phases[c][valid] - phases["C"][valid]))
                        )
                    )
                )
                if valid.any()
                else None
            )
    return dict(
        frequencies_hz=frequencies,
        rhythms=rhythms,
        plv=plv,
        caveat="diagnostic peak-based phase; no CTC mechanism claim",
    )


def analyse(spikes, cfg, swap):
    centres = cfg["centres"][::-1] if swap else cfg["centres"]
    profiles, epochs = {}, []
    for epoch, (start, end) in enumerate(cfg["windows_ms"]):
        sl = slice(round(start / cfg["dt_ms"]), round(end / cfg["dt_ms"]))
        rates = {
            key: value[sl].mean(0) * 1000 / cfg["dt_ms"]
            for key, value in spikes.items()
        }
        fits = {}
        for c in "ABC":
            sigma = np.sqrt(
                cfg["spatial_sigma"] ** 2
                + cfg["sensory_kernel_sigma"] ** 2
                + (cfg["cross_kernel_sigma"] ** 2 if c == "C" else 0)
            )
            fits[c], _, _ = fit_profile(
                rates[c + "_E"], centres, sigma, cfg["minimum_total_amplitude_hz"]
            )
        fraction = fits["C"]["fraction_A"]
        attended = "AB"[epoch]
        er = (
            (fraction if epoch == 0 else 1 - fraction) if fraction is not None else None
        )
        fidelity = copy_fidelity(
            rates[attended + "_E"], rates["C_E"], cfg.get("fidelity_bin_neurons", 8)
        )
        epochs.append(
            dict(
                attended=attended,
                window_ms=[start, end],
                er=er,
                fits=fits,
                fidelity=fidelity,
                mean_rates_hz={k: float(v.mean()) for k, v in rates.items()},
                phase=phase_metrics(spikes, cfg, start, end),
            )
        )
        profiles.update({f"epoch{epoch}_{k}": v for k, v in rates.items()})
    return epochs, profiles


def spatial_timecourse(spikes, cfg, swap, window_ms=100.0, hop_ms=5.0):
    """Fit nonnegative background + fixed A/B templates in overlapping windows."""
    dt = cfg["dt_ms"]
    s = spikes["C_E"]
    width = round(window_ms / dt)
    hop = round(hop_ms / dt)
    x = np.arange(s.shape[1]) / s.shape[1]
    centres = cfg["centres"][::-1] if swap else cfg["centres"]
    sigma = np.sqrt(
        cfg["spatial_sigma"] ** 2
        + cfg["sensory_kernel_sigma"] ** 2
        + cfg["cross_kernel_sigma"] ** 2
    )
    design = np.column_stack(
        [np.ones(len(x)), *[gaussian(x, c, sigma) for c in centres]]
    )
    cumulative = np.vstack(
        [np.zeros((1, s.shape[1])), np.cumsum(s, axis=0, dtype=np.float64)]
    )
    rows = []
    for start in range(0, len(s) - width + 1, hop):
        end = start + width
        y = (cumulative[end] - cumulative[start]) * 1000 / (width * dt)
        coef, _ = nnls(design, y)
        prediction = np.sum(design * coef[None, :], axis=1)
        variance = np.sum((y - y.mean()) ** 2)
        rows.append(
            dict(
                time_ms=(start + end) * dt / 2,
                background_hz=float(coef[0]),
                A_amplitude_hz=float(coef[1]),
                B_amplitude_hz=float(coef[2]),
                fit_R2=float(1 - np.sum((y - prediction) ** 2) / variance)
                if variance > 0
                else None,
            )
        )
    return dict(
        window_ms=width * dt,
        hop_ms=hop * dt,
        method="NNLS: nonnegative uniform background + fixed unit-peak A/B Gaussian templates; overlapping windows, no extra temporal smoothing; coefficients are pattern amplitudes, not causal source attribution",
        template_sigma=float(sigma),
        centres=centres,
        points=rows,
    )


def cycle_selection_scores(epochs):
    """Balanced cue accuracy and dominant-pattern transitions per eligible pair."""
    successes, transitions, pairs = [], 0, 0
    for epoch in epochs:
        states = [cycle["dominant"] for cycle in epoch["cycles"]]
        successes.append(
            sum(state == epoch["attended"] for state in states) / len(states)
            if states
            else None
        )
        for before, after in zip(states, states[1:], strict=False):
            if before is not None and after is not None:
                pairs += 1
                transitions += before != after
    return dict(
        switching_success=float(np.mean(successes))
        if all(s is not None for s in successes) and successes
        else None,
        flicker_fraction=transitions / pairs if pairs else None,
        transitions=int(transitions),
        eligible_pairs=pairs,
        cycle_counts=[len(epoch["cycles"]) for epoch in epochs],
        resolved_counts=[
            sum(c["dominant"] is not None for c in epoch["cycles"]) for epoch in epochs
        ],
    )


def cycle_switch_latency(
    cycles, switch_ms, sustained_cycles, attended="B", end_ms=float("inf")
):
    """First post-cue cycle in a sustained attended run, numbered from one."""
    if sustained_cycles < 2:
        raise ValueError("Sustained switching requires at least two cycles")
    post = [c for c in cycles if c["start_ms"] >= switch_ms and c["end_ms"] <= end_ms]
    for i in range(len(post) - sustained_cycles + 1):
        run = post[i : i + sustained_cycles]
        if all(c["dominant"] == attended for c in run):
            return dict(
                cycles=i + 1,
                sustained_cycles=sustained_cycles,
                onset_ms=post[i]["start_ms"] - switch_ms,
                confirmation_ms=run[-1]["end_ms"] - switch_ms,
            )
    return dict(
        cycles=None,
        sustained_cycles=sustained_cycles,
        onset_ms=None,
        confirmation_ms=None,
    )


def cycle_spatial_fits(spikes, cfg, swap=False):
    """Decode all complete C E trough-to-trough cycles, including transients."""
    dt = cfg["dt_ms"]
    s = spikes["C_E"]
    t = np.arange(len(s)) * dt
    rate = gaussian_filter1d(s.mean(1) * 1000 / dt, 1 / dt)
    peaks, _ = find_peaks(
        rate,
        distance=round(15 / dt),
        prominence=max(0.05, 0.1 * np.ptp(rate[t > 500])),
    )
    centres = cfg["centres"][::-1] if swap else cfg["centres"]
    sigma = np.sqrt(
        cfg["spatial_sigma"] ** 2
        + cfg["sensory_kernel_sigma"] ** 2
        + cfg["cross_kernel_sigma"] ** 2
    )
    x = np.arange(s.shape[1]) / s.shape[1]
    design = np.column_stack(
        [np.ones(len(x)), *[gaussian(x, c, sigma) for c in centres]]
    )
    # Cut between volleys, rather than through each volley's spatially
    # unequal rising/falling halves. Centre ties in flat minima deterministically.
    boundaries = []
    for left, right in zip(peaks, peaks[1:], strict=False):
        valley = rate[left + 1 : right]
        minima = np.flatnonzero(valley == valley.min()) + left + 1
        boundary = minima[np.argmin(abs(minima - (left + right) / 2))]
        boundaries.append(int(boundary))
    all_cycles = []
    for start, end, peak in zip(boundaries, boundaries[1:], peaks[1:-1], strict=False):
        y = s[start:end].mean(0) * 1000 / dt
        coefficients, _ = nnls(design, y)
        a, b = map(float, coefficients[1:])
        dominant = (
            None
            if a + b < cfg["minimum_total_amplitude_hz"] or a == b
            else ("A" if a > b else "B")
        )
        all_cycles.append(
            dict(
                start_ms=float(start * dt),
                end_ms=float(end * dt),
                peak_ms=float(peak * dt),
                A_amplitude_hz=a,
                B_amplitude_hz=b,
                dominant=dominant,
            )
        )
    return all_cycles


def cycle_selection_metrics(spikes, cfg, swap=False, *, sustained_cycles):
    """Score whole C volleys within the declared measurement windows."""
    all_cycles = cycle_spatial_fits(spikes, cfg, swap)
    epochs = []
    for attended, window in zip(
        "AB", cfg["stability_search"]["windows_ms"], strict=True
    ):
        cycles = [
            c
            for c in all_cycles
            if c["start_ms"] >= window[0] and c["end_ms"] <= window[1]
        ]
        epoch = dict(attended=attended, window_ms=window, cycles=cycles)
        scores = cycle_selection_scores([epoch])
        epoch.update(
            switching_success=scores["switching_success"],
            flicker_fraction=scores["flicker_fraction"],
            switching_latency=cycle_switch_latency(
                all_cycles,
                0.0 if attended == "A" else cfg["switch_ms"],
                sustained_cycles,
                attended=attended,
                end_ms=cfg["switch_ms"] if attended == "A" else cfg["duration_ms"],
            ),
        )
        epochs.append(epoch)
    return dict(
        **cycle_selection_scores(epochs),
        epochs=epochs,
        switching_latency=cycle_switch_latency(
            all_cycles, cfg["switch_ms"], sustained_cycles
        ),
        minimum_total_amplitude_hz=cfg["minimum_total_amplitude_hz"],
        method="One NNLS spatial fit per complete C E trough-to-trough cycle containing one detected peak; troughs are population-rate minima between successive peaks, with flat-minimum ties nearest the peak midpoint; existing 1 ms peak smoothing and 15 ms minimum peak spacing; no multi-cycle averaging",
        decoder_version="trough-to-trough/v1",
    )


def stability_metrics(spikes, cfg, swap=False):
    """Quantify spatial competition without hiding excursions behind long smoothing."""
    settings = cfg["stability_search"]
    course = spatial_timecourse(
        spikes, cfg, swap, settings["fit_window_ms"], settings["fit_hop_ms"]
    )
    rows = course["points"]
    times = np.array([r["time_ms"] for r in rows])
    amplitudes = np.array([[r["A_amplitude_hz"], r["B_amplitude_hz"]] for r in rows])
    total = amplitudes.sum(1)
    resolved = total >= cfg["minimum_total_amplitude_hz"]
    dominance = amplitudes[:, 0] - amplitudes[:, 1]
    windows = []
    for i, (start, end) in enumerate(settings["windows_ms"]):
        mask = (times >= start) & (times < end)
        attended = amplitudes[mask, i]
        other = amplitudes[mask, 1 - i]
        usable = resolved[mask]
        wrong = (other >= attended) | ~usable
        runs = []
        length = 0
        for failed in wrong:
            if failed:
                length += 1
            elif length:
                runs.append(length)
                length = 0
        if length:
            runs.append(length)
        windows.append(
            dict(
                attended="AB"[i],
                window_ms=[start, end],
                wrong_or_unresolved_fraction=float(wrong.mean()),
                leakage_fraction=float(
                    other.sum() / max((attended + other).sum(), 1e-12)
                ),
                maximum_excursion_centres_ms=float(
                    max(runs, default=0) * course["hop_ms"]
                ),
                unresolved_fraction=float((~usable).mean()),
                mean_attended_amplitude_hz=float(attended.mean()),
            )
        )
    # Use only post-switch observations; report window ends to avoid look-ahead.
    half = course["window_ms"] / 2
    eligible = (times - half >= cfg["switch_ms"]) & resolved
    need = round(settings["sustained_dominance_ms"] / course["hop_ms"])
    latency = None
    for i in np.flatnonzero(eligible):
        if i + need > len(times):
            break
        if np.all(eligible[i : i + need] & (dominance[i : i + need] < 0)):
            latency = float(times[i] + half - cfg["switch_ms"])
            break
    return dict(
        window_ms=course["window_ms"],
        hop_ms=course["hop_ms"],
        epochs=windows,
        worst_window_wrong_fraction=max(
            w["wrong_or_unresolved_fraction"] for w in windows
        ),
        mean_wrong_fraction=float(
            np.mean([w["wrong_or_unresolved_fraction"] for w in windows])
        ),
        switch_latency_ms=latency,
        caveat="Overlapping fits are correlated; fitted spatial identity does not establish input origin. Latency is the end of the first exclusively post-switch fit beginning a 100 ms run of B dominance.",
    )


def materialise_compute(source, destination):
    """Restore canonical native paths into disposable stage work, never into a run."""
    destination.mkdir(parents=True, exist_ok=False)
    reverse = {
        stored: original
        for original, stored in source.record.get("scientific_artifacts", {})
        .get("native_paths", {})
        .items()
    }
    for path in source.outputs.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source.outputs).as_posix()
        target = destination / reverse.get(relative, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    return destination


def compare_metrics(actual, expected):
    """Require recomputation to agree with retained original measurements."""
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            raise ValueError("Metric keys differ")
        for key, value in actual.items():
            compare_metrics(value, expected[key])
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise ValueError("Metric lengths differ")
        for a, b in zip(actual, expected, strict=False):
            compare_metrics(a, b)
    elif isinstance(actual, (float, int)) and not isinstance(actual, bool):
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
    elif actual != expected:
        raise ValueError(f"Metric mismatch: {actual!r} != {expected!r}")


def validate_parameters(directory, cfg, expected):
    params = NetworkSnapshot.read(directory / "start/snapshot.json").load_parameters()
    if params.keys() != expected.keys():
        raise ValueError("Executed parameter keys differ from the pinned source")
    for key in expected:
        np.testing.assert_array_equal(params[key].numpy(), expected[key].numpy())
    for value in params.values():
        if not np.isfinite(value.numpy()).all() or (value < 0).any():
            raise ValueError("Invalid weights")
    np.testing.assert_array_equal(
        params["A_E_to_C_E.weight"], params["B_E_to_C_E.weight"]
    )
    if cfg.get("cross_total_us") is not None:
        np.testing.assert_allclose(
            params["A_E_to_C_E.weight"].sum(0), cfg["cross_total_us"], rtol=2e-6
        )
    if cfg.get("c_ee_scale") == 0 and params["C_E_to_E.weight"].count_nonzero():
        raise ValueError("C recurrence enabled")


def report_trial_scope(trials, cfg):
    """Select the declared report trials from possibly larger historical runs."""
    recorded = {trial["id"]: trial for trial in trials}
    if len(recorded) != len(trials):
        raise ValueError("Duplicate trial")
    planned = [item["id"] for item in trial_plan(cfg)]
    if not set(planned).issubset(recorded):
        raise ValueError("Incomplete trial grid")
    return [recorded[key] for key in planned], sorted(set(recorded) - set(planned))


def validate(data, original_parameters, source_configuration):
    cfg = json.loads((data / "configuration.json").read_text())
    summary = json.loads((data / "summary.json").read_text())
    trials, excluded = report_trial_scope(summary["trials"], cfg)
    summary = dict(summary, trials=trials, excluded_trial_ids=excluded)
    expected = {item["id"]: item for item in trial_plan(cfg)}
    for trial in trials:
        directory = data / trial["id"]
        local = json.loads((directory / "configuration.json").read_text())
        compare_metrics(local["trial"], expected[trial["id"]])
        resolved = cfg
        for key in resolved:
            compare_metrics(local[key], resolved[key])
        with np.load(directory / "spikes.npz") as z:
            spikes = dict(z)
        if set(spikes) != {c + "_" + p for c in "ABC" for p in "EI"}:
            raise ValueError("Missing spikes")
        for key, value in spikes.items():
            shape = (
                round(cfg["duration_ms"] / cfg["dt_ms"]),
                cfg["network"]["populations"][key[-1]]["size"],
            )
            if value.shape != shape or not np.isin(value, [0, 1]).all():
                raise ValueError("Invalid spikes")
        epochs, profiles = analyse(spikes, resolved, trial["swap"])
        compare_metrics(epochs, trial["epochs"])
        if "stability" in trial:
            compare_metrics(
                stability_metrics(spikes, resolved, trial["swap"]), trial["stability"]
            )
        compare_metrics(json.loads((directory / "metrics.json").read_text()), trial)
        with np.load(directory / "profiles.npz") as z:
            for key, value in profiles.items():
                np.testing.assert_array_equal(value, z[key])
        validate_parameters(
            directory,
            resolved,
            selected_parameters(original_parameters, resolved, source_configuration),
        )
    return cfg, summary


def receptive_window_evidence(arrays):
    """Sample native inhibition at volley centres; no receptivity threshold."""
    cases = {}
    for case in ["original", "delayed"]:
        points = []
        for sender in "AB":
            index = int(np.argmax(arrays[case + "_arrival_" + sender]))
            points.append(
                dict(
                    sender=sender,
                    time_ms=float(arrays["time_ms"][index]),
                    inhibition_us=float(arrays[case + "_inhibition_us"][index]),
                )
            )
        points.sort(key=lambda point: point["time_ms"])
        assert points[0]["inhibition_us"] < points[1]["inhibition_us"]
        cases[case] = points
    return dict(
        cases=cases,
        hard_receptivity_threshold=None,
        arrival_reference="Peak of the displayed 1-ms-smoothed sender arrival rate, not the first afferent spike",
        interpretation="Low inhibition is a measured proxy for receptivity; membrane state and excitatory input also matter",
    )


def analyse_mechanism(data, output):
    """Validate the packet-only intervention and export its measured time series."""
    from experiments.exp172.recipe import shifted_packet

    cfg = json.loads((data / "configuration.json").read_text())
    descriptor = json.loads((data / "mechanism.json").read_text())
    settings = descriptor["settings"]
    dt = cfg["dt_ms"]
    cases = {
        key: dict(np.load(data / descriptor[key] / "spikes.npz"))
        for key in ["original", "control", "delayed"]
    }
    original = cases["original"]
    control, delayed = cases["control"], cases["delayed"]
    for key in original:
        np.testing.assert_array_equal(control[key], original[key])
        if key[0] in "AB":
            np.testing.assert_array_equal(delayed[key], original[key])
        stop = round(settings["packet_window_ms"][0] / dt)
        np.testing.assert_array_equal(delayed[key][:stop], original[key][:stop])
    packet = original[settings["sender"] + "_E"]
    for label, expected in [
        ("control", packet),
        ("delayed", shifted_packet(packet, dt, settings)),
    ]:
        with np.load(data / descriptor[label] / "replayed_spikes.npz") as z:
            np.testing.assert_array_equal(z["spikes"], expected)
    params = NetworkSnapshot.read(
        data / descriptor["original"] / "start/snapshot.json"
    ).load_parameters()
    graph = json.loads(
        (data / descriptor["original"] / "network.bundle/graph.json").read_text()
    )
    for label in ["control", "delayed"]:
        path = data / descriptor[label]
        other = NetworkSnapshot.read(path / "start/snapshot.json").load_parameters()
        assert set(other) == set(params)
        for key in params:
            np.testing.assert_array_equal(other[key], params[key])
        other_graph = json.loads((path / "network.bundle/graph.json").read_text())
        port = "replay_" + settings["sender"]
        other_graph["inputs"] = [i for i in other_graph["inputs"] if i["id"] != port]
        projection = next(
            p
            for p in other_graph["projections"]
            if p["id"] == settings["sender"] + "_E_to_C_E"
        )
        assert projection["source"] == port + ".value"
        projection["source"] = settings["sender"] + "_E.spikes"
        compare_metrics(other_graph, graph)
    t = np.arange(len(packet)) * dt
    start, end = settings["display_window_ms"]
    keep = (t >= start) & (t < end)
    arrays = {"time_ms": t[keep]}
    outcomes = {}
    sigma = np.sqrt(
        sum(
            cfg[k] ** 2
            for k in ["spatial_sigma", "sensory_kernel_sigma", "cross_kernel_sigma"]
        )
    )
    n = original["C_E"].shape[1]
    design = np.column_stack(
        [np.ones(n), *[gaussian(np.arange(n) / n, c, sigma) for c in cfg["centres"]]]
    )
    lo, hi = [round(v / dt) for v in settings["outcome_window_ms"]]
    baseline_inhibition = None
    for label in ["original", "control", "delayed"]:
        recording = RecordingData.read(
            canonical_export_file(data, descriptor[label], "recording")
        )
        info = recording.info["C_I_to_E.conductance"]
        conductance = recording.signals["C_I_to_E.conductance"]
        assert info.unit == "uS" and info.dt_ms == dt
        assert tuple(info.axes) == ("time", "batch", "cell")
        assert conductance.shape == (len(t), 1, n)
        assert np.isfinite(conductance).all() and (conductance >= 0).all()
        if label == "control":
            assert baseline_inhibition is not None
            np.testing.assert_array_equal(conductance, baseline_inhibition)
        if label == "original":
            baseline_inhibition = conductance.copy()
        aa, bb = nnls(design, cases[label]["C_E"][lo:hi].mean(0) * 1000 / dt)[0][1:]
        outcomes[label] = dict(
            A_amplitude_hz=float(aa),
            B_amplitude_hz=float(bb),
            attended_dominance=float((aa - bb) / (aa + bb)),
        )
        if label == "control":
            continue
        arrays[label + "_inhibition_us"] = conductance[:, 0, :].mean(
            1, dtype=np.float64
        )[keep]
        for p in "EI":
            arrays[label + "_C_" + p] = cases[label]["C_" + p][keep]
        for sender in "AB":
            source = original[sender + "_E"]
            if label == "delayed" and sender == settings["sender"]:
                source = shifted_packet(source, dt, settings)
            projection = next(
                p for p in graph["projections"] if p["id"] == sender + "_E_to_C_E"
            )
            assert projection["delay"]["unit"] == "ms"
            steps = round(projection["delay"]["value"] / dt)
            rate = source.mean(1) * 1000 / dt
            arrival = np.r_[np.zeros(steps), rate[:-steps]] if steps else rate
            arrays[label + "_arrival_" + sender] = gaussian_filter1d(arrival, 1 / dt)[
                keep
            ]
    assert (
        outcomes["original"]["attended_dominance"]
        < 0
        < outcomes["delayed"]["attended_dominance"]
    )
    reference = float(
        arrays["time_ms"][np.argmax(arrays["original_arrival_" + settings["attended"]])]
    )
    np.savez_compressed(output / "mechanism.npz", **arrays)
    return dict(
        configuration=cfg,
        settings=settings,
        outcomes=outcomes,
        attended_arrival_peak_ms=reference,
        receptive_window=receptive_window_evidence(arrays),
        identical_control=True,
        identical_weights=True,
        conserved_packet_counts=True,
        identical_senders=True,
        identical_preintervention_prefix=True,
        limitation="Post hoc single-trial illustration of sender timing at C's inhibitory gate; independent background-driven premature inhibition is a separate route.",
    )


def mechanism_analysis_run(computed, run_id=None):
    started = perf_counter()
    with stage_run(
        REPO,
        "exp172",
        "analyse",
        inputs={"compute": computed},
        run_id=run_id,
        configuration=dict(
            method="validate timing-only intervention and native inhibitory conductances"
        ),
    ) as run:
        work = materialise_compute(computed, run.scratch / "evidence")
        measured = analyse_mechanism(work, run.export)
        dump(
            run.export / "results.json",
            dict(mechanism=measured, elapsed_seconds=perf_counter() - started),
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    print(run.run_id, flush=True)
    return run.run_id


def analyse_candidate(data, baseline=None):
    descriptor = json.loads((data / "candidate.json").read_text())
    directory = data / descriptor["trial_id"]
    cfg = json.loads((directory / "configuration.json").read_text())
    spikes = dict(np.load(directory / "spikes.npz"))
    params = NetworkSnapshot.read(directory / "start/snapshot.json").load_parameters()
    if baseline is None:
        original = data / descriptor["original"]
        baseline = (
            json.loads((original / "configuration.json").read_text()),
            dict(np.load(original / "spikes.npz")),
            NetworkSnapshot.read(original / "start/snapshot.json").load_parameters(),
        )
    original_cfg, original_spikes, original_params = baseline
    for key, value in original_cfg.items():
        if key not in {
            "background_hz",
            "trial",
            "input_seed_sequences",
            "stage_mode",
            "candidate",
        }:
            compare_metrics(cfg[key], value)
    assert cfg["background_hz"] == descriptor["settings"]["background_hz"]
    assert set(params) == set(original_params)
    for key in params:
        np.testing.assert_array_equal(params[key], original_params[key])
    for key in ["A_E", "A_I", "B_E", "B_I"]:
        np.testing.assert_array_equal(spikes[key], original_spikes[key])
    measured = cycle_selection_metrics(
        spikes, cfg, False, **cycle_analysis_configuration()
    )
    return dict(
        configuration=cfg,
        settings=descriptor["settings"],
        trial_id=descriptor["trial_id"],
        original_background_hz=original_cfg["background_hz"],
        cycle_selection=measured,
        unchanged_weights=True,
        unchanged_senders=True,
        limitation="Selected on one seed from separate sweeps; only C independent background drive changes",
    )


def candidate_analysis_run(computed, run_id=None):
    started = perf_counter()
    reference = computed.record["inputs"]["reference_compute"]
    original = source_run(
        REPO / ".pingstore",
        reference["run_id"],
        experiment="exp172",
        stage="compute",
        reference=reference,
    )
    descriptor = json.loads(computed.file("candidate.json").read_text())
    trial = descriptor["original"]
    baseline = (
        json.loads(original.file(trial, "configuration.json").read_text()),
        dict(np.load(original.file(trial, "spikes.npz"))),
        NetworkSnapshot.read(
            original.file(trial, "start/snapshot.json")
        ).load_parameters(),
    )
    with stage_run(
        REPO,
        "exp172",
        "analyse",
        inputs={"compute": computed, "baseline_compute": original},
        run_id=run_id,
        configuration=dict(
            method="validate fixed-weight higher-background candidate and decode complete C volleys"
        ),
    ) as run:
        work = materialise_compute(computed, run.scratch / "evidence")
        candidate = analyse_candidate(work, baseline)
        dump(
            run.export / "results.json",
            dict(candidate=candidate, elapsed_seconds=perf_counter() - started),
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    return run.run_id


def seed_summary(values):
    """Equal-weight observed-seed mean and sample SD; missing is never zero."""
    observed = np.array([v for v in values if v is not None], dtype=float)
    if not np.isfinite(observed).all():
        raise ValueError("Non-finite seed metric")
    return dict(
        n=int(len(observed)),
        total=len(values),
        mean=float(observed.mean()) if len(observed) else None,
        sd=float(observed.std(ddof=1)) if len(observed) > 1 else None,
    )


def aggregate_background(rows, settings, control_key="background_hz"):
    expected = {
        (seed, rate, att)
        for seed in settings["seeds"]
        for rate in settings["rates_hz"]
        for att in "AB"
    }
    identities = [(r["seed"], r[control_key], r["attended"]) for r in rows]
    if len(set(identities)) != len(rows) or set(identities) != expected:
        raise ValueError("Missing or duplicate sweep seed/condition")
    result = []
    for att in "AB":
        for rate in settings["rates_hz"]:
            selected = [
                next(
                    r
                    for r in rows
                    if r["seed"] == seed
                    and r[control_key] == rate
                    and r["attended"] == att
                )
                for seed in settings["seeds"]
            ]
            result.append(
                dict(
                    attended=att,
                    **{control_key: rate},
                    **{
                        key: seed_summary([r[key] for r in selected])
                        for key in ["S", "F", "latency_cycles", "C_E_hz", "C_I_hz"]
                    },
                )
            )
    return result


def background_analysis_run(computed, run_id=None):
    from experiments.exp172.recipe import background_sweep_plan

    started = perf_counter()
    descriptor = json.loads(computed.file("background-plan.json").read_text())
    settings = descriptor["settings"]
    plan = background_sweep_plan(settings)
    compare_metrics(plan, descriptor["trials"])
    ref = computed.record["inputs"]["reference_compute"]
    reference = source_run(
        REPO / ".pingstore",
        ref["run_id"],
        experiment="exp172",
        stage="compute",
        reference=ref,
    )
    baseline_cfg = reference.record["execution"]["configuration"]
    parameters = NetworkSnapshot.read(
        reference.file(descriptor["baseline_trial"], "start/snapshot.json")
    ).load_parameters()
    with np.load(reference.file(descriptor["baseline_trial"], "spikes.npz")) as z:
        pinned_spikes = dict(z)
    sender_reference = {}
    for seed in settings["seeds"]:
        item = next(
            i
            for i in plan
            if i["seed"] == seed and i["background_hz"] == baseline_cfg["background_hz"]
        )
        with np.load(computed.file(item["id"], "spikes.npz")) as z:
            sender_reference[seed] = {k: z[k] for k in ["A_E", "A_I", "B_E", "B_I"]}
    with stage_run(
        REPO,
        "exp172",
        "analyse",
        inputs={"compute": computed, "baseline_compute": reference},
        run_id=run_id,
        configuration=dict(
            method="equal-weight five-input-seed means and sample SD",
            latency_missing="conditional mean with observed-seed counts",
        ),
    ) as run:
        rows = []
        for item in plan:
            cfg = json.loads(
                computed.file(item["id"], "configuration.json").read_text()
            )
            compare_metrics(cfg["trial"], item)
            for key, value in baseline_cfg.items():
                if key not in {"background_hz", "stage_mode", "background_sweep"}:
                    compare_metrics(cfg[key], value)
            assert cfg["background_hz"] == item["background_hz"]
            actual = NetworkSnapshot.read(
                computed.file(item["id"], "start/snapshot.json")
            ).load_parameters()
            assert set(actual) == set(parameters)
            for key in actual:
                np.testing.assert_array_equal(actual[key], parameters[key])
            with np.load(computed.file(item["id"], "spikes.npz")) as z:
                spikes = dict(z)
            assert set(spikes) == {c + "_" + p for c in "ABC" for p in "EI"}
            for key, array in spikes.items():
                assert array.shape == (
                    round(cfg["duration_ms"] / cfg["dt_ms"]),
                    cfg["network"]["populations"][key[-1]]["size"],
                )
            for key, value in sender_reference[item["seed"]].items():
                np.testing.assert_array_equal(spikes[key], value)
            if (
                item["seed"] == 185
                and item["background_hz"] == baseline_cfg["background_hz"]
            ):
                for key, value in pinned_spikes.items():
                    np.testing.assert_array_equal(spikes[key], value)
            measured = cycle_selection_metrics(
                spikes, cfg, False, **cycle_analysis_configuration()
            )
            for epoch in measured["epochs"]:
                lo, hi = [round(v / cfg["dt_ms"]) for v in epoch["window_ms"]]
                states = [c["dominant"] for c in epoch["cycles"]]
                pairs = [
                    (a, b)
                    for a, b in zip(states, states[1:], strict=False)
                    if a is not None and b is not None
                ]
                rows.append(
                    dict(
                        trial_id=item["id"],
                        seed=item["seed"],
                        background_hz=item["background_hz"],
                        attended=epoch["attended"],
                        S=epoch["switching_success"],
                        F=epoch["flicker_fraction"],
                        latency_cycles=epoch["switching_latency"]["cycles"],
                        cycles=len(states),
                        resolved_cycles=sum(s is not None for s in states),
                        changes=sum(a != b for a, b in pairs),
                        eligible_pairs=len(pairs),
                        C_E_hz=float(spikes["C_E"][lo:hi].mean() * 1000 / cfg["dt_ms"]),
                        C_I_hz=float(spikes["C_I"][lo:hi].mean() * 1000 / cfg["dt_ms"]),
                    )
                )
        summary = aggregate_background(rows, settings)
        export = dict(
            settings=settings,
            configuration=baseline_cfg,
            seed_rows=rows,
            summary=summary,
            unchanged_weights=True,
            unchanged_senders_within_seed=True,
            baseline_185_reproduced=True,
            decoder_version="trough-to-trough/v1",
            limitation="Five input seeds with one fixed circuit, including the original tuning seed 185; no random-circuit or independent held-out validation claim",
        )
        dump(
            run.export / "results.json",
            dict(background_sweep=export, elapsed_seconds=perf_counter() - started),
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    return run.run_id


def cue_analysis_run(computed, run_id=None):
    """Validate fixed inputs/weights and exact 3-Hz controls before averaging."""
    from experiments.exp172.recipe import best_background, cue_sweep_plan

    started = perf_counter()
    descriptor = json.loads(computed.file("cue-plan.json").read_text())
    settings = descriptor["settings"]
    plan = cue_sweep_plan(settings)
    compare_metrics(plan, descriptor["trials"])
    ref = computed.record["inputs"]["reference_compute"]
    reference = source_run(
        REPO / ".pingstore",
        ref["run_id"],
        experiment="exp172",
        stage="compute",
        reference=ref,
    )
    ref_analysis = computed.record["inputs"]["background_analyse"]
    background_analysis = source_run(
        REPO / ".pingstore",
        ref_analysis["run_id"],
        experiment="exp172",
        stage="analyse",
        reference=ref_analysis,
    )
    background = json.loads(background_analysis.file("results.json").read_text())[
        "background_sweep"
    ]
    assert background_analysis.record["inputs"]["compute"] == ref
    assert settings["background_hz"] == best_background(background["summary"])
    baseline_cfg = json.loads(
        reference.file(descriptor["baseline_trial"], "configuration.json").read_text()
    )
    parameters = NetworkSnapshot.read(
        reference.file(descriptor["baseline_trial"], "start/snapshot.json")
    ).load_parameters()
    with np.load(reference.file(descriptor["baseline_trial"], "input_rates.npz")) as z:
        fixed_rates = {k: z[k] for k in ("X_A", "X_B", "X_C")}
    with stage_run(
        REPO,
        "exp172",
        "analyse",
        inputs={"compute": computed},
        run_id=run_id,
        configuration=dict(
            method="equal-weight five-input-seed cue means and sample SD",
            latency_missing="conditional mean with observed-seed counts",
        ),
    ) as run:
        rows = []
        for item in plan:
            cfg = json.loads(
                computed.file(item["id"], "configuration.json").read_text()
            )
            compare_metrics(cfg["trial"], item)
            for key, value in baseline_cfg.items():
                if key not in {"trial", "input_seed_sequences", "cue_hz", "stage_mode"}:
                    compare_metrics(cfg[key], value)
            assert cfg["cue_hz"] == item["cue_hz"]
            actual = NetworkSnapshot.read(
                computed.file(item["id"], "start/snapshot.json")
            ).load_parameters()
            assert set(actual) == set(parameters)
            for key in actual:
                np.testing.assert_array_equal(actual[key], parameters[key])
            with np.load(computed.file(item["id"], "input_rates.npz")) as z:
                for key, rates in fixed_rates.items():
                    np.testing.assert_array_equal(z[key], rates)
                np.testing.assert_array_equal(
                    z["U"], np.full_like(z["U"], item["cue_hz"])
                )
            with np.load(computed.file(item["id"], "spikes.npz")) as z:
                spikes = dict(z)
            assert set(spikes) == {c + "_" + p for c in "ABC" for p in "EI"}
            for key, array in spikes.items():
                assert array.shape == (
                    round(cfg["duration_ms"] / cfg["dt_ms"]),
                    cfg["network"]["populations"][key[-1]]["size"],
                )
            if item["cue_hz"] == settings["reference_cue_hz"]:
                original = next(
                    r["trial_id"]
                    for r in background["seed_rows"]
                    if r["seed"] == item["seed"]
                    and r["background_hz"] == settings["background_hz"]
                )
                with np.load(reference.file(original, "spikes.npz")) as z:
                    for key in spikes:
                        np.testing.assert_array_equal(spikes[key], z[key])
            measured = cycle_selection_metrics(
                spikes, cfg, False, **cycle_analysis_configuration()
            )
            for epoch in measured["epochs"]:
                lo, hi = [round(v / cfg["dt_ms"]) for v in epoch["window_ms"]]
                states = [c["dominant"] for c in epoch["cycles"]]
                pairs = [
                    (a, b)
                    for a, b in zip(states, states[1:], strict=False)
                    if a is not None and b is not None
                ]
                rows.append(
                    dict(
                        trial_id=item["id"],
                        seed=item["seed"],
                        cue_hz=item["cue_hz"],
                        attended=epoch["attended"],
                        S=epoch["switching_success"],
                        F=epoch["flicker_fraction"],
                        latency_cycles=epoch["switching_latency"]["cycles"],
                        cycles=len(states),
                        resolved_cycles=sum(s is not None for s in states),
                        changes=sum(a != b for a, b in pairs),
                        eligible_pairs=len(pairs),
                        C_E_hz=float(spikes["C_E"][lo:hi].mean() * 1000 / cfg["dt_ms"]),
                        C_I_hz=float(spikes["C_I"][lo:hi].mean() * 1000 / cfg["dt_ms"]),
                    )
                )
        export = dict(
            settings=settings,
            configuration=baseline_cfg,
            seed_rows=rows,
            summary=aggregate_background(rows, settings, control_key="cue_hz"),
            unchanged_weights=True,
            unchanged_sensory_background_rates=True,
            reference_cue_reproduced_all_seeds=True,
            decoder_version="trough-to-trough/v1",
            limitation="Background selected and cue assessed on the same five input seeds in one fixed circuit; not held-out validation",
        )
        dump(
            run.export / "results.json",
            dict(cue_sweep=export, elapsed_seconds=perf_counter() - started),
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    return run.run_id


def contrast_analysis_run(computed, run_id=None):
    """Validate contrast profiles, fixed weights and exact equal-contrast controls."""
    from experiments.exp172.recipe import contrast_rate_profiles, contrast_sweep_plan

    started = perf_counter()
    descriptor = json.loads(computed.file("contrast-plan.json").read_text())
    settings = descriptor["settings"]
    plan = contrast_sweep_plan(settings)
    previous_rows = []
    if "previous_analyse" in computed.record["inputs"]:
        previous_ref = computed.record["inputs"]["previous_analyse"]
        previous = source_run(
            REPO / ".pingstore",
            previous_ref["run_id"],
            experiment="exp172",
            stage="analyse",
            reference=previous_ref,
        )
        old = json.loads(previous.file("results.json").read_text())["contrast_sweep"]
        assert old["equal_contrast_reproduced_all_seeds"] and old["unchanged_weights"]
        assert {k: v for k, v in old["settings"].items() if k != "rates_hz"} == {
            k: v for k, v in settings.items() if k != "rates_hz"
        }
        previous_compute_ref = previous.record["inputs"]["compute"]
        previous_compute = source_run(
            REPO / ".pingstore",
            previous_compute_ref["run_id"],
            experiment="exp172",
            stage="compute",
            reference=previous_compute_ref,
        )
        assert (
            previous_compute.record["inputs"]["cue_analyse"]
            == computed.record["inputs"]["cue_analyse"]
        )
        previous_rows = old["seed_rows"]
        plan = [
            item
            for item in plan
            if item["contrast_ratio"] not in old["settings"]["rates_hz"]
        ]
    compare_metrics(plan, descriptor["trials"])
    ref = computed.record["inputs"]["reference_compute"]
    reference = source_run(
        REPO / ".pingstore",
        ref["run_id"],
        experiment="exp172",
        stage="compute",
        reference=ref,
    )
    ref_analysis = computed.record["inputs"]["cue_analyse"]
    cue_analysis = source_run(
        REPO / ".pingstore",
        ref_analysis["run_id"],
        experiment="exp172",
        stage="analyse",
        reference=ref_analysis,
    )
    cue = json.loads(cue_analysis.file("results.json").read_text())["cue_sweep"]
    assert cue_analysis.record["inputs"]["compute"] == ref
    assert settings["background_hz"] == cue["settings"]["background_hz"]
    baseline_cfg = json.loads(
        reference.file(descriptor["baseline_trial"], "configuration.json").read_text()
    )
    parameters = NetworkSnapshot.read(
        reference.file(descriptor["baseline_trial"], "start/snapshot.json")
    ).load_parameters()
    with np.load(reference.file(descriptor["baseline_trial"], "input_rates.npz")) as z:
        fixed_rates = {k: z[k] for k in ("X_C",)}
    with stage_run(
        REPO,
        "exp172",
        "analyse",
        inputs={
            "compute": computed,
            **({"previous_analyse": previous} if previous_rows else {}),
        },
        run_id=run_id,
        configuration=dict(
            method="equal-weight five-input-seed contrast-cue means and sample SD",
            latency_missing="conditional mean with observed-seed counts",
        ),
    ) as run:
        rows = list(previous_rows)
        for item in plan:
            cfg = json.loads(
                computed.file(item["id"], "configuration.json").read_text()
            )
            compare_metrics(cfg["trial"], item)
            for key, value in baseline_cfg.items():
                if key not in {"trial", "input_seed_sequences", "cue_hz", "stage_mode"}:
                    compare_metrics(cfg[key], value)
            assert cfg["cue_hz"] == item["cue_hz"]
            actual = NetworkSnapshot.read(
                computed.file(item["id"], "start/snapshot.json")
            ).load_parameters()
            assert set(actual) == set(parameters)
            for key in actual:
                np.testing.assert_array_equal(actual[key], parameters[key])
            with np.load(computed.file(item["id"], "input_rates.npz")) as z:
                for key, rates in fixed_rates.items():
                    np.testing.assert_array_equal(z[key], rates)
                assert item["cue_hz"] == cfg["cue_hz"] == 0
                assert cfg["contrast_ratio"] == item["contrast_ratio"]
                np.testing.assert_array_equal(z["U"], np.zeros_like(z["U"]))
                for key, expected in contrast_rate_profiles(
                    cfg, item["contrast_ratio"]
                ).items():
                    np.testing.assert_array_equal(z[key], expected)
            with np.load(computed.file(item["id"], "spikes.npz")) as z:
                spikes = dict(z)
            assert set(spikes) == {c + "_" + p for c in "ABC" for p in "EI"}
            for key, array in spikes.items():
                assert array.shape == (
                    round(cfg["duration_ms"] / cfg["dt_ms"]),
                    cfg["network"]["populations"][key[-1]]["size"],
                )
            if item["contrast_ratio"] == settings["reference_ratio"]:
                original = next(
                    r["trial_id"]
                    for r in cue["seed_rows"]
                    if r["seed"] == item["seed"] and r["cue_hz"] == 0
                )
                with np.load(reference.file(original, "spikes.npz")) as z:
                    for key in spikes:
                        np.testing.assert_array_equal(spikes[key], z[key])
            measured = cycle_selection_metrics(
                spikes, cfg, False, **cycle_analysis_configuration()
            )
            for epoch in measured["epochs"]:
                lo, hi = [round(v / cfg["dt_ms"]) for v in epoch["window_ms"]]
                states = [c["dominant"] for c in epoch["cycles"]]
                pairs = [
                    (a, b)
                    for a, b in zip(states, states[1:], strict=False)
                    if a is not None and b is not None
                ]
                rows.append(
                    dict(
                        trial_id=item["id"],
                        seed=item["seed"],
                        contrast_ratio=item["contrast_ratio"],
                        attended=epoch["attended"],
                        S=epoch["switching_success"],
                        F=epoch["flicker_fraction"],
                        latency_cycles=epoch["switching_latency"]["cycles"],
                        cycles=len(states),
                        resolved_cycles=sum(s is not None for s in states),
                        changes=sum(a != b for a, b in pairs),
                        eligible_pairs=len(pairs),
                        C_E_hz=float(spikes["C_E"][lo:hi].mean() * 1000 / cfg["dt_ms"]),
                        C_I_hz=float(spikes["C_I"][lo:hi].mean() * 1000 / cfg["dt_ms"]),
                    )
                )
        export = dict(
            settings=settings,
            configuration=baseline_cfg,
            seed_rows=rows,
            summary=aggregate_background(rows, settings, control_key="contrast_ratio"),
            unchanged_weights=True,
            verified_contrast_profiles=True,
            unchanged_background_rates=True,
            uniform_cue_disabled=True,
            equal_contrast_reproduced_all_seeds=True,
            decoder_version="trough-to-trough/v1",
            examples=[
                dict(
                    trial_id=next(
                        r["id"]
                        for r in contrast_sweep_plan(settings)
                        if r["seed"] == settings["seeds"][0]
                        and r["contrast_ratio"] == ratio
                    ),
                    seed=settings["seeds"][0],
                    contrast_ratio=ratio,
                )
                for ratio in settings["example_ratios"]
            ],
            limitation="Contrast cues change both spatial contrast and mean sensory drive; five tuning input seeds in one fixed circuit, not held-out validation",
        )
        dump(
            run.export / "results.json",
            dict(contrast_sweep=export, elapsed_seconds=perf_counter() - started),
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    return run.run_id


def circular_arrival_density(phases, bins, smoothing_bins):
    """Normalised circular histogram; never condition on successful selection."""
    values = (np.asarray(phases, dtype=float) + 0.5) % 1 - 0.5
    if not np.isfinite(values).all():
        raise ValueError("Non-finite arrival phase")
    edges = np.linspace(-0.5, 0.5, bins + 1)
    if not len(values):
        return dict(n=0, edges=edges.tolist(), density=None)
    counts, _ = np.histogram(values, bins=edges)
    density = gaussian_filter1d(counts.astype(float), smoothing_bins, mode="wrap")
    density /= density.sum() * (edges[1] - edges[0])
    vector = np.exp(2j * np.pi * values).mean()
    return dict(
        n=len(values),
        edges=edges.tolist(),
        density=density.tolist(),
        circular_mean_cycles=float(np.angle(vector) / (2 * np.pi)),
        concentration=float(abs(vector)),
    )


def arrival_case_distribution(spikes, inhibition, cfg, graph, settings):
    dt = cfg["dt_ms"]
    time = np.arange(len(inhibition)) * dt
    smooth = gaussian_filter1d(inhibition, settings["rate_smoothing_ms"] / dt)
    prominence = max(
        1e-9, settings["peak_prominence_fraction"] * np.ptp(smooth[time > 500])
    )
    peaks, _ = find_peaks(
        smooth,
        distance=round(settings["minimum_peak_spacing_ms"] / dt),
        prominence=prominence,
    )
    troughs = []
    for left, right in zip(peaks[:-1], peaks[1:], strict=True):
        valley = inhibition[left + 1 : right]
        minima = np.flatnonzero(valley == valley.min()) + left + 1
        troughs.append(int(minima[np.argmin(abs(minima - (left + right) / 2))]))
    arrivals = {}
    delays = {}
    for sender in "AB":
        rate = gaussian_filter1d(
            spikes[sender + "_E"].mean(1) * 1000 / dt,
            settings["rate_smoothing_ms"] / dt,
        )
        detected, _ = find_peaks(
            rate,
            distance=round(settings["minimum_peak_spacing_ms"] / dt),
            prominence=max(
                0.05, settings["peak_prominence_fraction"] * np.ptp(rate[time > 500])
            ),
        )
        projection = next(
            p for p in graph["projections"] if p["id"] == sender + "_E_to_C_E"
        )
        assert projection["delay"]["unit"] == "ms"
        delays[sender] = projection["delay"]["value"]
        arrivals[sender] = time[detected] + delays[sender]
    profiles = []
    rows = []
    cycles = []
    phase_grid = np.linspace(-0.5, 0.5, settings["profile_samples"], endpoint=False)
    for attended, window in zip(
        "AB", cfg["stability_search"]["windows_ms"], strict=True
    ):
        for lo, hi in zip(troughs[:-1], troughs[1:], strict=True):
            start, end = lo * dt, hi * dt
            if start < window[0] or end > window[1]:
                continue
            cycles.append(dict(start_ms=start, end_ms=end, attended=attended))
            profiles.append(
                np.interp(
                    start + np.mod(phase_grid, 1) * (end - start), time, inhibition
                )
            )
            for sender in "AB":
                selected = arrivals[sender][
                    (arrivals[sender] >= start - 1e-8) & (arrivals[sender] < end - 1e-8)
                ]
                for arrival in selected:
                    phase = ((arrival - start) / (end - start) + 0.5) % 1 - 0.5
                    rows.append(
                        dict(
                            sender=sender,
                            role="attended" if sender == attended else "unattended",
                            cue_attended=attended,
                            arrival_ms=float(arrival),
                            phase_cycles=float(phase),
                            cycle_start_ms=start,
                            cycle_end_ms=end,
                            inhibition_us=float(np.interp(arrival, time, inhibition)),
                        )
                    )
    assert len({(r["sender"], r["arrival_ms"]) for r in rows}) == len(rows)
    if not profiles:
        raise ValueError("No complete C inhibitory cycles in the settled windows")
    distributions = []
    for sender in "AB":
        for role in ["attended", "unattended"]:
            subset = [r for r in rows if r["sender"] == sender and r["role"] == role]
            distributions.append(
                dict(
                    sender=sender,
                    role=role,
                    **circular_arrival_density(
                        [r["phase_cycles"] for r in subset],
                        settings["histogram_bins"],
                        settings["smoothing_bins"],
                    ),
                )
            )
    return dict(
        background_hz=cfg["background_hz"],
        seed=settings["seed"],
        distributions=distributions,
        arrivals=rows,
        cycles=cycles,
        phase_grid=phase_grid.tolist(),
        mean_inhibition_us=np.mean(profiles, axis=0).tolist(),
        delays_ms=delays,
    )


def arrivals_analysis_run(computed, run_id=None):
    started = perf_counter()
    descriptor = json.loads(computed.file("arrival-plan.json").read_text())
    settings = descriptor["settings"]
    inputs = {"compute": computed}
    sources = {}
    for case in descriptor["cases"]:
        ref = computed.record["inputs"][case["reference_role"]]
        sources[case["label"]] = source_run(
            REPO / ".pingstore",
            ref["run_id"],
            experiment="exp172",
            stage="compute",
            reference=ref,
        )
    with stage_run(
        REPO,
        "exp172",
        "analyse",
        inputs=inputs,
        run_id=run_id,
        configuration=dict(
            method="circular sender arrival density relative to native C inhibitory troughs",
            settings=settings,
        ),
    ) as run:
        panels = []
        reference_senders = None
        reference_parameters = None
        for case in descriptor["cases"]:
            source = sources[case["label"]]
            trial = case["trial_id"]
            reference_trial = case["reference_trial"]
            cfg = json.loads(computed.file(trial, "configuration.json").read_text())
            parameters = NetworkSnapshot.read(
                computed.file(trial, "start/snapshot.json")
            ).load_parameters()
            pinned = NetworkSnapshot.read(
                source.file(reference_trial, "start/snapshot.json")
            ).load_parameters()
            for key in parameters:
                np.testing.assert_array_equal(parameters[key], pinned[key])
            with np.load(computed.file(trial, "spikes.npz")) as z:
                spikes = dict(z)
            with np.load(source.file(reference_trial, "spikes.npz")) as z:
                for key in z.files:
                    np.testing.assert_array_equal(spikes[key], z[key])
            if reference_senders is None:
                reference_senders = {k: spikes[k] for k in ["A_E", "A_I", "B_E", "B_I"]}
                reference_parameters = parameters
            else:
                for key in reference_senders:
                    np.testing.assert_array_equal(spikes[key], reference_senders[key])
                for key in parameters:
                    np.testing.assert_array_equal(
                        parameters[key], reference_parameters[key]
                    )
            recording = RecordingData.read(computed.file(trial, "recording"))
            key = "C_I_to_E.conductance"
            info = recording.info[key]
            array = recording.signals[key]
            assert (
                info.unit == "uS"
                and info.dt_ms == cfg["dt_ms"]
                and tuple(info.axes) == ("time", "batch", "cell")
            )
            assert (
                array.shape == (len(spikes["C_E"]), 1, spikes["C_E"].shape[1])
                and np.isfinite(array).all()
                and (array >= 0).all()
            )
            inhibition = array[:, 0, :].mean(1, dtype=np.float64)
            graph = json.loads(
                computed.file(trial, "network.bundle/graph.json").read_text()
            )
            panel = arrival_case_distribution(spikes, inhibition, cfg, graph, settings)
            panels.append(dict(panel, figure=1 if case["label"] == "figure1" else 6))
        measured = dict(
            settings=settings,
            panels=panels,
            reference_pins=computed.record["inputs"],
            exact_replay=True,
            limitation="Exact single seed-185 trials of Figures 1 and 6; all arrivals included, not just successful transmissions. Cycles and arrivals are correlated. Inhibitory trough is a phase reference and proxy, not a measured hard receptive-window boundary.",
        )
        dump(
            run.export / "results.json",
            dict(
                arrival_distribution=measured, elapsed_seconds=perf_counter() - started
            ),
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    return run.run_id


def low_inhibition_intervals(phase, inhibition, fraction):
    """Threshold a measured mean profile; interpolate crossings in phase."""
    phase, inhibition = np.asarray(phase), np.asarray(inhibition)
    if not (0 < fraction < 1) or not np.isfinite(inhibition).all():
        raise ValueError("Invalid inhibition profile or threshold fraction")
    if (
        len(phase) != len(inhibition)
        or len(phase) < 2
        or not (np.diff(phase) > 0).all()
    ):
        raise ValueError("Phase grid must increase and match the profile")
    lo, hi = float(inhibition.min()), float(inhibition.max())
    threshold = lo + fraction * (hi - lo)
    intervals = []
    if hi > lo:
        inside = inhibition <= threshold
        start = float(phase[0]) if inside[0] else None
        for i in np.flatnonzero(inside[1:] != inside[:-1]):
            cross = float(
                phase[i]
                + (phase[i + 1] - phase[i])
                * (threshold - inhibition[i])
                / (inhibition[i + 1] - inhibition[i])
            )
            if inside[i]:
                intervals.append([start, cross])
                start = None
            else:
                start = cross
        if start is not None:
            intervals.append([start, float(phase[-1])])
    return dict(
        bounds_cycles=intervals, threshold_us=threshold, minimum_us=lo, maximum_us=hi
    )


def measured_inhibition_profile(time, inhibition, cycles, windows, phase, delay):
    """Average conductance in reversed C peak phase, shifted by sender delay."""
    peaks = np.array([c["peak_ms"] for c in cycles])
    counts = np.arange(len(peaks))
    profiles, cycle_times = [], []
    for i, cycle in enumerate(cycles):
        if not any(
            cycle["start_ms"] >= lo and cycle["end_ms"] <= hi for lo, hi in windows
        ):
            continue
        # A positive sender-minus-C phase means a sender volley precedes C.
        emission = np.interp(i - phase, counts, peaks, left=np.nan, right=np.nan)
        arrival = emission + delay
        if (
            not np.isfinite(arrival).all()
            or arrival.min() < 0
            or arrival.max() > time[-1]
        ):
            continue
        profiles.append(np.interp(arrival, time, inhibition))
        cycle_times.append(cycle["peak_ms"])
    if not profiles:
        raise ValueError("No complete measured inhibition profiles")
    return np.mean(profiles, axis=0), cycle_times


def phase_inhibition_analysis_run(computed, run_id=None):
    """Measure native C inhibition in the same peak reference as Figure 3."""
    started = perf_counter()
    plan = json.loads(computed.file("phase-inhibition-plan.json").read_text())
    trial, settings = plan["trial_id"], plan["settings"]
    cfg = json.loads(computed.file(trial, "configuration.json").read_text())
    ref = computed.record["inputs"]["reference_compute"]
    source = source_run(
        REPO / ".pingstore",
        ref["run_id"],
        experiment="exp172",
        stage="compute",
        reference=ref,
    )
    with np.load(computed.file(trial, "spikes.npz")) as z:
        spikes = dict(z)
    with np.load(source.file(plan["reference_trial"], "spikes.npz")) as original:
        assert set(original.files) == set(spikes)
        for key in original.files:
            np.testing.assert_array_equal(spikes[key], original[key])
    recording = RecordingData.read(computed.file(trial, "recording"))
    info = recording.info["C_I_to_E.conductance"]
    values = recording.signals["C_I_to_E.conductance"]
    assert info.unit == "uS" and info.dt_ms == cfg["dt_ms"]
    assert tuple(info.axes) == ("time", "batch", "cell")
    assert values.shape == (len(spikes["C_E"]), 1, spikes["C_E"].shape[1])
    assert np.isfinite(values).all() and (values >= 0).all()
    inhibition = values[:, 0, :].mean(1, dtype=np.float64)
    graph = json.loads(computed.file(trial, "network.bundle/graph.json").read_text())
    delays = [
        next(p for p in graph["projections"] if p["id"] == c + "_E_to_C_E")["delay"]
        for c in "AB"
    ]
    assert delays[0] == delays[1] and delays[0]["unit"] == "ms"
    delay = delays[0]["value"]
    cycles = cycle_spatial_fits(spikes, cfg)
    phase = np.linspace(-0.5, 0.5, settings["profile_samples"])
    time = np.arange(len(inhibition)) * cfg["dt_ms"]
    profile, cycle_times = measured_inhibition_profile(
        time,
        inhibition,
        cycles,
        cfg["stability_search"]["windows_ms"],
        phase,
        delay,
    )
    measured = dict(
        **low_inhibition_intervals(phase, profile, settings["low_inhibition_fraction"]),
        phase_cycles=phase.tolist(),
        mean_inhibition_us=profile.tolist(),
        cycles=len(cycle_times),
        cycle_peaks_ms=cycle_times,
        delay_ms=delay,
        reference_trial=plan["reference_trial"],
        seed=cfg["trial"]["seed"],
        source_reference=ref,
        settings=settings,
        exact_replay=True,
        method="Mean native C_I_to_E conductance over complete settled C E cycles; reverse peak-interpolated C phase plus native sender delay; half dynamic-range threshold",
        limitation="Low inhibition is a receptivity proxy, not a measured transmission threshold. Mapping to sender-minus-C phase assumes locally matched sender/C periods; dominance is not used.",
    )
    with stage_run(
        REPO,
        "exp172",
        "analyse",
        run_id=run_id,
        inputs={"compute": computed},
        configuration=settings,
    ) as run:
        dump(run.export / "results.json", dict(phase_inhibition=measured))
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    return run.run_id


def probe_response_window(phase, response, fraction):
    """Half-maximum of positive net extra spikes; interpolate phase crossings."""
    phase, response = np.asarray(phase, dtype=float), np.asarray(response, dtype=float)
    if len(phase) != len(response) or len(phase) < 2 or not (np.diff(phase) > 0).all():
        raise ValueError("Probe phase grid must increase and match response")
    if not np.isfinite(response).all() or not 0 < fraction < 1:
        raise ValueError("Invalid response profile")
    peak = float(response.max())
    if peak <= 0:
        return dict(
            bounds_cycles=[], threshold_extra_spikes=None, peak_extra_spikes=peak
        )
    threshold = fraction * peak
    inside = response >= threshold
    intervals = []
    start = float(phase[0]) if inside[0] else None
    for i in np.flatnonzero(inside[1:] != inside[:-1]):
        cross = float(
            phase[i]
            + (phase[i + 1] - phase[i])
            * (threshold - response[i])
            / (response[i + 1] - response[i])
        )
        if inside[i]:
            intervals.append([start, cross])
            start = None
        else:
            start = cross
    if start is not None:
        intervals.append([start, float(phase[-1])])
    return dict(
        bounds_cycles=intervals,
        threshold_extra_spikes=threshold,
        peak_extra_spikes=peak,
    )


def probe_analysis_run(computed, run_id=None):
    started = perf_counter()
    plan = json.loads(computed.file("probe-plan.json").read_text())
    cfg = computed.record["execution"]["configuration"]
    settings = plan["settings"]
    dt = cfg["dt_ms"]
    assert plan["exact_control"]
    count = len(plan["anchors"])
    phases = np.array(settings["phases_cycles"])
    excess = np.full((count, len(phases)), np.nan)
    records = []
    with np.load(computed.file("probe-spikes.npz")) as z:
        for row in plan["trials"]:
            control = z[f"control{row['anchor']:02d}_C_E"]
            probed = z[row["id"] + "_C_E"]
            start = (
                row["probe_step"] + round(settings["delay_ms"] / dt) - row["start_step"]
            )
            end = start + round(settings["response_ms"] / dt)
            assert 0 <= start < end <= len(probed) and probed.shape == control.shape
            # C cannot respond to the new path before its native delay elapses.
            np.testing.assert_array_equal(probed[:start], control[:start])
            actual, baseline = (
                int(probed[start:end].sum()),
                int(control[start:end].sum()),
            )
            value = actual - baseline
            assert np.isnan(excess[row["anchor"], row["phase_index"]])
            excess[row["anchor"], row["phase_index"]] = value
            records.append(
                dict(
                    row,
                    response_spikes=actual,
                    control_spikes=baseline,
                    extra_spikes=value,
                )
            )
    assert np.isfinite(excess).all()
    profile = excess.mean(0)
    # The last endpoint repeats the first circular phase bin for interpolation.
    grid = np.r_[phases, 0.5]
    response = np.r_[profile, profile[0]]
    maximum_phase_error = max(
        abs(row["phase_cycles"] - phases[row["phase_index"]]) for row in records
    )
    measured = dict(
        **probe_response_window(grid, response, settings["window_fraction"]),
        phase_cycles=grid.tolist(),
        mean_extra_spikes=response.tolist(),
        sd_extra_spikes=np.r_[excess.std(0, ddof=1), excess.std(0, ddof=1)[0]].tolist(),
        per_cycle_extra_spikes=excess.tolist(),
        trials=records,
        anchors=plan["anchors"],
        cue_profiles={
            c: excess[[a["attended"] == c for a in plan["anchors"]]].mean(0).tolist()
            for c in "AB"
        },
        cycles=count,
        probes=len(records),
        maximum_phase_error_cycles=maximum_phase_error,
        reference_trial=plan["reference_trial"],
        seed=plan["reference_seed"],
        reference_background=plan["reference_background"],
        settings=settings,
        exact_control=True,
        method="Fixed uniform AMPA volley at 16 C peak phases in 8 native states; paired net extra C E spikes in first 5 ms after native delivery; shade mean response >= half its positive peak",
        limitation="Single input seed, eight correlated cycle states. Probe amplitude and response horizon define the window; it is not a universal transmission boundary. Phase uses unperturbed C peak timing with locally matched sender/C periods.",
    )
    with stage_run(
        REPO,
        "exp172",
        "analyse",
        run_id=run_id,
        inputs={"compute": computed},
        configuration=settings,
    ) as run:
        dump(run.export / "results.json", dict(probe_window=measured))
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    return run.run_id


def population_volley_cycles(values, dt):
    """Whole observed volleys, with the same peak/trough rules as C decoding."""
    time = np.arange(len(values)) * dt
    rate = gaussian_filter1d(values.mean(1) * 1000 / dt, 1 / dt)
    peaks, _ = find_peaks(
        rate,
        distance=round(15 / dt),
        prominence=max(0.05, 0.1 * np.ptp(rate[time > 500])),
    )
    boundaries = []
    for left, right in zip(peaks[:-1], peaks[1:], strict=True):
        valley = rate[left + 1 : right]
        minima = np.flatnonzero(valley == valley.min()) + left + 1
        boundaries.append(int(minima[np.argmin(abs(minima - (left + right) / 2))]))
    return [
        dict(start_step=lo, end_step=hi, peak_step=int(peak))
        for lo, hi, peak in zip(
            boundaries[:-1], boundaries[1:], peaks[1:-1], strict=True
        )
    ]


def natural_volley_plan(spikes, cfg):
    """All native packets with complete one/two-cycle outcomes in settled windows."""
    dt = cfg["dt_ms"]
    receiver = population_volley_cycles(spikes["C_E"], dt)
    cpeaks = np.array([row["peak_step"] for row in receiver])
    delay_steps = round(cfg["network"]["cross"]["delay_ms"] / dt)
    events = []
    for sender in "AB":
        volleys = population_volley_cycles(spikes[sender + "_E"], dt)
        peaks = np.array([row["peak_step"] for row in volleys])
        for packet in volleys:
            arrival = packet["peak_step"] + delay_steps
            candidates = [
                i
                for i, cycle in enumerate(receiver[:-1])
                if cycle["start_step"] <= arrival < cycle["end_step"]
            ]
            if not candidates:
                continue
            index = candidates[0]
            cycle, following = receiver[index : index + 2]
            first = min(packet["start_step"], cycle["start_step"])
            last = max(packet["end_step"] + delay_steps, following["end_step"])
            if not any(
                first * dt >= lo and last * dt <= hi
                for lo, hi in cfg["stability_search"]["windows_ms"]
            ):
                continue
            if last != following["end_step"]:
                continue
            phase = (
                np.interp(cpeaks[index], peaks, np.arange(len(peaks))) - index + 0.5
            ) % 1 - 0.5
            count = int(
                spikes[sender + "_E"][packet["start_step"] : packet["end_step"]].sum()
            )
            if count == 0:
                continue
            events.append(
                dict(
                    id=f"delete{sender}-{packet['peak_step']:06d}",
                    sender=sender,
                    packet_start=packet["start_step"],
                    packet_end=packet["end_step"],
                    packet_peak=packet["peak_step"],
                    packet_spikes=count,
                    phase_cycles=float(phase),
                    reference_peak_ms=float(cpeaks[index] * dt),
                    start_step=first - 1,
                    end_step=following["end_step"],
                    response_start=cycle["start_step"],
                    response_end=cycle["end_step"],
                    response_end_two=following["end_step"],
                    attended="A" if cpeaks[index] * dt < cfg["switch_ms"] else "B",
                )
            )
    return sorted(events, key=lambda row: (row["start_step"], row["id"]))


def spatial_count_amplitudes(spikes, cfg):
    """Existing spatial templates fitted to integrated spikes, not a short rate bin."""
    sigma = np.sqrt(
        sum(
            cfg[key] ** 2
            for key in ["spatial_sigma", "sensory_kernel_sigma", "cross_kernel_sigma"]
        )
    )
    x = np.arange(spikes.shape[1]) / spikes.shape[1]
    design = np.column_stack(
        [np.ones(len(x)), *[gaussian(x, centre, sigma) for centre in cfg["centres"]]]
    )
    return nnls(design, spikes.sum(0, dtype=np.float64))[0][1:]


def used_phase_summary(events, settings):
    """Shade only sampled phase bins with a strong median persistent spatial effect."""
    edges = np.linspace(-0.5, 0.5, settings["phase_bins"] + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        subset = [event for event in events if lo <= event["phase_cycles"] < hi]
        scores = [event["persistent_effect"] for event in subset]
        bins.append(
            dict(
                lo=float(lo),
                hi=float(hi),
                n=len(subset),
                median_effect=float(np.median(scores)) if scores else None,
                positive_events=sum(
                    score > settings["effect_tolerance"] for score in scores
                ),
                sender_counts={
                    c: sum(event["sender"] == c for event in subset) for c in "AB"
                },
                cue_counts={
                    c: sum(event["attended"] == c for event in subset) for c in "AB"
                },
            )
        )
    supported = [
        row["median_effect"] for row in bins if row["n"] >= settings["minimum_events"]
    ]
    peak = max(supported, default=0)
    threshold = (
        settings["window_fraction"] * peak
        if peak > settings["effect_tolerance"]
        else None
    )
    bounds = []
    for row in bins:
        row["selected"] = bool(
            threshold is not None
            and row["n"] >= settings["minimum_events"]
            and row["median_effect"] >= threshold
            and row["median_effect"] > settings["effect_tolerance"]
        )
        if row["selected"]:
            if bounds and np.isclose(bounds[-1][1], row["lo"]):
                bounds[-1][1] = row["hi"]
            else:
                bounds.append([row["lo"], row["hi"]])
    return dict(
        bounds_cycles=bounds,
        phase_bins=bins,
        threshold_effect=threshold,
        peak_effect=peak,
    )


def used_window_analysis_run(computed, run_id=None):
    started = perf_counter()
    plan = json.loads(computed.file("used-window-plan.json").read_text())
    cfg = computed.record["execution"]["configuration"]
    settings = plan["settings"]
    analysis_protocol = used_window_configuration()
    for key, value in settings.items():
        if key not in ("sensitivity_phase_bins", "sensitivity_window_fractions"):
            assert analysis_protocol[key] == value
    settings = dict(
        settings,
        **{
            key: analysis_protocol[key]
            for key in ["sensitivity_phase_bins", "sensitivity_window_fractions"]
        },
    )
    events = []
    with np.load(computed.file("used-window-spikes.npz")) as z:
        for event in plan["events"]:
            control, deleted = z[event["control_id"]], z[event["id"]]
            lo = event["start_step"]
            unaffected = (
                event["packet_start"]
                + round(cfg["network"]["cross"]["delay_ms"] / cfg["dt_ms"])
                - lo
            )
            np.testing.assert_array_equal(control[:unaffected], deleted[:unaffected])
            effects = []
            outcomes = []
            own = "AB".index(event["sender"])
            for horizon, endpoint in [
                (1, event["response_end"]),
                (2, event["response_end_two"]),
            ]:
                start, end = event["response_start"] - lo, endpoint - lo
                assert 0 <= start < end <= len(control)
                baseline = spatial_count_amplitudes(control[start:end], cfg)
                removal = spatial_count_amplitudes(deleted[start:end], cfg)
                delta = (baseline - removal) / horizon
                own_effect, contrast_effect = (
                    float(delta[own]),
                    float(delta[own] - delta[1 - own]),
                )
                effects.extend([own_effect, contrast_effect])
                outcomes.append(
                    dict(
                        cycles=horizon,
                        control_amplitudes=baseline.tolist(),
                        deleted_amplitudes=removal.tolist(),
                        own_effect=own_effect,
                        contrast_effect=contrast_effect,
                        net_spikes=int(control[start:end].sum())
                        - int(deleted[start:end].sum()),
                    )
                )
            events.append(
                dict(event, persistent_effect=min(effects), outcomes=outcomes)
            )
    measured = dict(
        **used_phase_summary(events, settings),
        events=events,
        packets=len(events),
        states=plan["states"],
        reference_trial=plan["reference_trial"],
        seed=plan["reference_seed"],
        reference_background=plan["reference_background"],
        settings=settings,
        exact_control=plan["exact_control"],
        phase_bins_count=settings["phase_bins"],
        method="All observed sender packets deleted on their native C E pathway. Persistent effect is the minimum of own-pattern and own-minus-competing-pattern count-amplitude losses over one and two fixed baseline C cycles, normalized per cycle. Shade phase bins with >=3 packets and median effect >=half the positive supported peak.",
        limitation="Used transmission phases in one saved trial, not a universal receptive gate. Sparse bins are unmeasured, not proven inaccessible. Packets and cycle states are correlated; leave-one-packet-out effects include recurrent competition. No dominance labels select packets or shaded phases.",
    )
    measured["sensitivity"] = [
        dict(
            phase_bin_count=bins,
            window_fraction=fraction,
            **used_phase_summary(
                events, dict(settings, phase_bins=bins, window_fraction=fraction)
            ),
        )
        for bins in settings["sensitivity_phase_bins"]
        for fraction in settings["sensitivity_window_fractions"]
    ]
    measured["leave_one_state_out"] = [
        dict(
            omitted_peak_ms=peak,
            **used_phase_summary(
                [row for row in events if row["reference_peak_ms"] != peak], settings
            ),
        )
        for peak in sorted({row["reference_peak_ms"] for row in events})
    ]
    measured["condition_summaries"] = {
        field: {
            value: used_phase_summary(
                [row for row in events if row[field] == value], settings
            )
            for value in "AB"
        }
        for field in ["sender", "attended"]
    }
    measured["strength_normalised_sensitivity"] = used_phase_summary(
        [
            dict(row, persistent_effect=row["persistent_effect"] / row["packet_spikes"])
            for row in events
        ],
        settings,
    )
    with stage_run(
        REPO,
        "exp172",
        "analyse",
        run_id=run_id,
        inputs={"compute": computed},
        configuration=settings,
    ) as run:
        dump(run.export / "results.json", dict(used_window=measured))
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    return run.run_id


def main(source, run_id=None):
    started = perf_counter()
    settings = cycle_analysis_configuration()
    computed = source_run(
        REPO / ".pingstore",
        source,
        experiment="exp172",
        stage="compute",
    )
    cfg = computed.record["execution"]["configuration"]
    if cfg.get("stage_mode") == "used_window":
        return used_window_analysis_run(computed, run_id)
    if cfg.get("stage_mode") == "probe_window":
        return probe_analysis_run(computed, run_id)
    if cfg.get("stage_mode") == "phase_inhibition":
        return phase_inhibition_analysis_run(computed, run_id)
    if cfg.get("stage_mode") == "arrival_distribution":
        return arrivals_analysis_run(computed, run_id)
    if cfg.get("stage_mode") == "contrast_sweep":
        return contrast_analysis_run(computed, run_id)
    if cfg.get("stage_mode") == "cue_sweep":
        return cue_analysis_run(computed, run_id)
    if cfg.get("stage_mode") == "background_sweep":
        return background_analysis_run(computed, run_id)
    if cfg.get("stage_mode") == "candidate":
        return candidate_analysis_run(computed, run_id)
    if cfg.get("stage_mode") == "mechanism":
        return mechanism_analysis_run(computed, run_id)
    reference = computed.record["inputs"]["parameter_reference"]
    parameter_source = source_run(
        REPO / ".pingstore",
        reference["run_id"],
        experiment="exp172",
        stage="compute",
        reference=reference,
    )
    original_parameters = NetworkSnapshot.read(
        parameter_source.file(cfg["source_snapshot"])
    ).load_parameters()
    source_configuration = json.loads(
        parameter_source.file("configuration.json").read_text()
    )
    with stage_run(
        REPO,
        "exp172",
        "analyse",
        inputs={"compute": computed},
        run_id=run_id,
        configuration=dict(
            method="recompute all saved-spike spatial/phase metrics", **settings
        ),
    ) as run:
        work = materialise_compute(computed, run.scratch / "evidence")
        cfg, summary = validate(work, original_parameters, source_configuration)
        cycle_selection = {}
        for trial in summary["trials"]:
            with np.load(work / trial["id"] / "spikes.npz") as z:
                cycle_selection[trial["id"]] = cycle_selection_metrics(
                    dict(z), cfg, trial["swap"], **settings
                )
        dump(
            run.export / "results.json",
            dict(
                mechanism=analyse_mechanism(work, run.export)
                if (work / "mechanism.json").exists()
                else None,
                candidate=analyse_candidate(work)
                if (work / "candidate.json").exists()
                else None,
                validated_trials=len(summary["trials"]),
                excluded_historical_trial_ids=summary["excluded_trial_ids"],
                trials=summary["trials"],
                cycle_selection=cycle_selection,
                single_seed=len(cfg["seeds"]) == 1,
                limitation="Tuned single-seed selected-cue demonstration; no held-out test or isolated CTC mechanism",
                elapsed_seconds=perf_counter() - started,
            ),
        )
        run.record["execution"]["elapsed_seconds"] = perf_counter() - started
    print(f"Analysis {perf_counter() - started:.2f} s; no simulation")
    return run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-id")
    main(**vars(parser.parse_args()))
